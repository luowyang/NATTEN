#!/usr/bin/env bash
# Build locally or validate an existing CI wheel. Never publishes or removes envs.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEV_WORKTREE="$(dirname "$(dirname "$SCRIPT_DIR")")/wt_dev"
SOURCE_ENV="${CONDA_PREFIX:-}"
ENV_PATH=""
DIST_DIR=""
WHEEL=""
MANIFEST=""
OUTPUT=""
SMOKE_TIMEOUT=1200
SKIP_ENV_CLONE=0
EXTRA_SMOKE_TESTS=""
CONDA_BIN="${CONDA_EXE:-conda}"
fail() { echo "ERROR: $*" >&2; exit 1; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev-worktree|--source-env|--env-path|--dist-dir|--wheel|--manifest|--output|--smoke-timeout|--extra-smoke-tests)
      [[ $# -ge 2 ]] || fail "$1 requires a value"
      case "$1" in
        --dev-worktree) DEV_WORKTREE="$2";; --source-env) SOURCE_ENV="$2";;
        --env-path) ENV_PATH="$2";; --dist-dir) DIST_DIR="$2";; --wheel) WHEEL="$2";;
        --manifest) MANIFEST="$2";; --output) OUTPUT="$2";; --smoke-timeout) SMOKE_TIMEOUT="$2";;
        --extra-smoke-tests) EXTRA_SMOKE_TESTS="$2";;
      esac
      shift 2 ;;
    --skip-env-clone) SKIP_ENV_CLONE=1; shift ;;
    -h|--help)
      echo 'Usage: build_wheel.sh --dev-worktree PATH --env-path PATH [--source-env PATH | --skip-env-clone]'
      echo '  [--dist-dir NEW_DIR] [--wheel PATH --manifest PATH] [--output NEW_JSON]'
      echo '  [--smoke-timeout SECONDS] [--extra-smoke-tests "PATH PATH ..."]'
      echo 'Existing CI wheels are installed and validated unchanged; source HEAD must match their manifest.'
      exit 0 ;;
    *) fail "Unknown option: $1" ;;
  esac
done
[[ -n "$ENV_PATH" ]] || fail "Specify an isolated --env-path"
[[ "$SMOKE_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || fail "--smoke-timeout must be positive"
DEV_WORKTREE="$(realpath "$DEV_WORKTREE")"
ENV_PATH="$(realpath -m "$ENV_PATH")"
[[ -z "$SOURCE_ENV" || "$ENV_PATH" != "$(realpath -m "$SOURCE_ENV")" ]] || fail "--env-path must differ from source environment"
SOURCE_SHA="$(git -C "$DEV_WORKTREE" rev-parse HEAD)"
[[ -z "$(git -C "$DEV_WORKTREE" status --porcelain --untracked-files=all --ignore-submodules=none)" ]] || fail "Source checkout must be clean"
if [[ -n "$WHEEL" ]]; then
  [[ -n "$MANIFEST" ]] || fail "--wheel requires its original --manifest"
  WHEEL="$(realpath "$WHEEL")"; MANIFEST="$(realpath "$MANIFEST")"
else
  [[ -z "$MANIFEST" ]] || fail "--manifest requires --wheel"
  [[ -n "$DIST_DIR" && ! -e "$DIST_DIR" ]] || fail "Local builds require a new --dist-dir (no stale wheels)"
  DIST_DIR="$(realpath -m "$DIST_DIR")"
fi
if [[ -e "$ENV_PATH" ]]; then
  [[ "$SKIP_ENV_CLONE" -eq 1 ]] || fail "Environment exists; explicitly --skip-env-clone to reuse it"
else
  [[ "$SKIP_ENV_CLONE" -eq 0 && -n "$SOURCE_ENV" ]] || fail "Provide --source-env to create the isolated environment"
  "$CONDA_BIN" create -y --clone "$SOURCE_ENV" -p "$ENV_PATH"
fi
if [[ -z "$WHEEL" ]]; then
  "$CONDA_BIN" run -p "$ENV_PATH" --no-capture-output pip install 'build>=1.3.0' 'setuptools<82'
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
  export NATTEN_CUDA_ARCH="${NATTEN_CUDA_ARCH:-9.0}"
  export NATTEN_N_WORKERS="${NATTEN_N_WORKERS:-1}"
  export NATTEN_AUTOGEN_POLICY="${NATTEN_AUTOGEN_POLICY:-fine}"
  export NATTEN_VERBOSE=1 NATTEN_IS_BUILDING_DIST=0
  git -C "$DEV_WORKTREE" submodule update --init --recursive
  mkdir -p "$DIST_DIR"
  (cd "$DEV_WORKTREE" && "$CONDA_BIN" run -p "$ENV_PATH" --no-capture-output python -m build --no-isolation --wheel -o "$DIST_DIR")
  shopt -s nullglob
  WHEELS=("$DIST_DIR"/*.whl)
  [[ ${#WHEELS[@]} -eq 1 ]] || fail "Expected exactly one newly built wheel"
  WHEEL="${WHEELS[0]}"
  MANIFEST="$DIST_DIR/MANIFEST.json"
  RECIPE_SHA="$(git -C "$SCRIPT_DIR" rev-parse HEAD)"
  "$CONDA_BIN" run -p "$ENV_PATH" --no-capture-output python "$SCRIPT_DIR/local_manifest.py" "$WHEEL" "$MANIFEST" "$SOURCE_SHA" "$RECIPE_SHA"
fi
[[ -n "$OUTPUT" ]] || OUTPUT="$(dirname "$WHEEL")/GPU-VALIDATION-$(basename "$WHEEL").json"
# Validate provenance before pip installation, and again after the GPU suite.
"$CONDA_BIN" run -p "$ENV_PATH" --no-capture-output python -I "$SCRIPT_DIR/validate_wheel.py" --preflight --wheel "$WHEEL" --manifest "$MANIFEST" --source "$DEV_WORKTREE" --output "$OUTPUT"
"$CONDA_BIN" run -p "$ENV_PATH" --no-capture-output pip install --force-reinstall --no-deps "$WHEEL"
TEST_ARGS=()
if [[ -n "$EXTRA_SMOKE_TESTS" ]]; then
  shopt -s nullglob
  TEST_ARGS=(--tests "$DEV_WORKTREE"/tests/test_varlen*.py "$DEV_WORKTREE"/tests/test_*_varlen.py)
  read -r -a EXTRA <<< "$EXTRA_SMOKE_TESTS"
  TEST_ARGS+=("${EXTRA[@]}")
fi
"$CONDA_BIN" run -p "$ENV_PATH" --no-capture-output python -I "$SCRIPT_DIR/validate_wheel.py" --wheel "$WHEEL" --manifest "$MANIFEST" --source "$DEV_WORKTREE" --output "$OUTPUT" --timeout "$SMOKE_TIMEOUT" "${TEST_ARGS[@]}"

#!/usr/bin/env bash
# nvcc-wrapper.sh -- sccache-through-nvcc wrapper. Installed by
# .github/actions/prepare-build-env/action.yml as
# /tmp/sccache-wrappers/nvcc and pointed at via CUDACXX, so CMake's
# compiler detection and every `nvcc` invocation in the build go through
# here instead of the real compiler directly.
#
# Without NATTEN_CI_SHARD_COUNT/NATTEN_CI_SHARD_INDEX set (the `build`
# job never sets them), this is functionally identical to the original,
# unconditional `exec sccache "$REAL_NVCC" "$@"`.
#
# With those set (the `warm` job's sharded precompile matrix, one runner
# per shard), a csrc/ translation unit that hashes to a DIFFERENT shard
# than this one is compiled from a fixed empty .cu stand-in instead,
# using the real compiler directly -- bypassing sccache entirely so no
# empty object ever lands in the shared cache under that file's real
# key. A file that hashes to THIS shard (or any non-csrc/ source, e.g.
# CMake's own compiler-identification probe, or a no-source invocation
# such as device link) takes the normal `sccache "$REAL_NVCC" "$@"`
# path. Across all NATTEN_CI_SHARD_COUNT shards running concurrently,
# every real translation unit therefore gets compiled for real by
# exactly one shard and primes the shared sccache cache -- and the
# `build` job's later, unsharded rebuild of the same tree issues
# byte-identical compile commands for every file, so it hits cache on
# all of them instead of compiling anything cold.
set -u

REAL_NVCC="${REAL_NVCC:?REAL_NVCC must be set}"

# Locate the .cu source argument, if any: nvcc/CMake invocations carry
# many other arguments (include paths, -D defines, --generate-code,
# object/output files, ...), so require BOTH a .cu suffix AND that the
# path exists as a file, to avoid matching an unrelated token that
# happens to end in .cu.
SRC=""
for arg in "$@"; do
  if [[ "$arg" == *.cu && -f "$arg" ]]; then
    SRC="$arg"
    break
  fi
done

# REL (the path after the first /csrc/) is only meaningful for a real
# source-tree translation unit. A non-csrc/ source (e.g. CMake's
# CMakeCUDACompilerId.cu probe, which lives under a build directory, not
# csrc/) or a no-source invocation (e.g. device link) leaves REL empty
# and always falls through to the unconditional passthrough below --
# there is nothing meaningful to shard-hash or to log to the
# failure/compiled marker files for those.
REL=""
if [[ -n "$SRC" && "$SRC" == *"/csrc/"* ]]; then
  REL="${SRC#*/csrc/}"
fi

if [[ -n "${NATTEN_CI_SHARD_COUNT:-}" && -n "$REL" ]]; then
  SHARD_INDEX="${NATTEN_CI_SHARD_INDEX:?NATTEN_CI_SHARD_INDEX must be set alongside NATTEN_CI_SHARD_COUNT}"
  HASH="$(printf '%s' "$REL" | cksum | cut -d' ' -f1)"
  if [[ $((HASH % NATTEN_CI_SHARD_COUNT)) -ne "$SHARD_INDEX" ]]; then
    # Foreign shard: compile a fixed empty stand-in with the real
    # compiler, not through sccache, so the cache never sees an empty
    # object under this file's real key.
    EMPTY_CU=/tmp/natten-empty.cu
    [[ -f "$EMPTY_CU" ]] || : > "$EMPTY_CU"
    ARGS=()
    REPLACED=0
    for arg in "$@"; do
      if [[ "$REPLACED" -eq 0 && "$arg" == "$SRC" ]]; then
        ARGS+=("$EMPTY_CU")
        REPLACED=1
      else
        ARGS+=("$arg")
      fi
    done
    exec "$REAL_NVCC" "${ARGS[@]}"
  fi

  # Own shard: real compile through sccache, then record the outcome for
  # the shard's report and for the "compiled nothing" sanity check.
  sccache "$REAL_NVCC" "$@"
  RC=$?
  if [[ "$RC" -ne 0 ]]; then
    echo "$REL" >> /tmp/natten-shard-failures.txt
  else
    echo "$REL" >> /tmp/natten-shard-compiled.txt
  fi
  exit "$RC"
fi

# No shard variables, or a non-csrc/no-source invocation: unconditional
# passthrough, identical to the pre-sharding wrapper.
exec sccache "$REAL_NVCC" "$@"

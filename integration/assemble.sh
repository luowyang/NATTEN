#!/usr/bin/env bash
# Assemble a pinned set of topics into dev. Never pushes.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_PATH=""
BRANCHES_FILE="$SCRIPT_DIR/branches.txt"
DO_TAG=0
RESOLVE_MANUALLY=0
CONTINUE_MODE=0
POSITIONAL=()
fail() { echo "ERROR: $*" >&2; exit 1; }
usage() {
  cat <<'USAGE'
Usage: assemble.sh <N> <repo_path> [--worktree PATH] [--branches-file PATH]
                   [--tag] [--resolve-manually] [--continue]
Defaults: worktree is ../wt_dev beside repo_path; topics are integration/branches.txt.
Every listed topic is required. Use refs/heads/NAME or refs/remotes/origin/NAME
when local and remote branches differ. --continue uses the saved SHA lock,
not current branch tips. Resolve and commit a paused merge before continuing.
USAGE
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --worktree|--branches-file)
      [[ $# -ge 2 ]] || fail "$1 requires a value"
      if [[ "$1" == --worktree ]]; then WORKTREE_PATH="$2"; else BRANCHES_FILE="$2"; fi
      shift 2 ;;
    --tag) DO_TAG=1; shift ;;
    --resolve-manually) RESOLVE_MANUALLY=1; shift ;;
    --continue) CONTINUE_MODE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) fail "Unknown option: $1" ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
[[ ${#POSITIONAL[@]} -eq 2 ]] || { usage >&2; exit 1; }
N="${POSITIONAL[0]}"
[[ "$N" =~ ^[0-9]+$ ]] || fail "Fork build number must be a non-negative integer"
REPO_PATH="$(cd -- "${POSITIONAL[1]}" && pwd -P)"
git -C "$REPO_PATH" rev-parse --git-dir >/dev/null
WORKTREE_PATH="$(realpath -m "${WORKTREE_PATH:-$(dirname "$REPO_PATH")/wt_dev}")"
BRANCHES_FILE="$(realpath -m "$BRANCHES_FILE")"
[[ "$WORKTREE_PATH" != "$REPO_PATH" && "$REPO_PATH" != "$WORKTREE_PATH/"* ]] || fail "Worktree must not contain repo_path"

registered() {
  git -C "$REPO_PATH" worktree list --porcelain | grep -qxF "worktree $WORKTREE_PATH"
}
check_clean() {
  local status
  # Ignored build products and nested submodule files are still user data.
  status="$(git -C "$WORKTREE_PATH" status --porcelain=v1 --untracked-files=all --ignored=matching --ignore-submodules=none)"
  [[ -z "$status" ]] || fail "Worktree is not empty of changes/untracked/ignored files; preserve them first: $WORKTREE_PATH"
  git -C "$WORKTREE_PATH" submodule foreach --quiet --recursive '
    test -z "$(git status --porcelain=v1 --untracked-files=all --ignored=matching)" || {
      echo "Dirty or untracked/ignored submodule files: $displaypath" >&2; exit 1;
    }
  '
}
TOPICS=()
SHAS=()
BASE_SHA=""
if [[ "$CONTINUE_MODE" -eq 1 ]]; then
  registered || fail "No registered worktree to continue: $WORKTREE_PATH"
  [[ "$(git -C "$WORKTREE_PATH" symbolic-ref --short HEAD)" == dev ]] || fail "Expected dev worktree"
  ! git -C "$WORKTREE_PATH" rev-parse --verify --quiet MERGE_HEAD >/dev/null || fail "Resolve and commit the pending merge first"
  check_clean
  LOCK="$(git -C "$WORKTREE_PATH" rev-parse --absolute-git-dir)/natten-assembly.tsv"
  [[ -f "$LOCK" ]] || fail "No saved assembly lock; cannot continue by guessing branch tips"
  IFS=$'\t' read -r lock_kind lock_number < "$LOCK"
  [[ "$lock_kind" == fork-build && "$lock_number" == "$N" ]] || fail "Saved assembly build number differs from $N"
  while IFS=$'\t' read -r kind ref sha; do
    case "$kind" in
      base) [[ -z "$BASE_SHA" && "$ref" == refs/remotes/origin/main ]] || fail "Invalid base lock"; BASE_SHA="$sha" ;;
      topic) TOPICS+=("$ref"); SHAS+=("$sha") ;;
      fork-build) : ;;
      *) fail "Invalid assembly lock record: $kind" ;;
    esac
  done < "$LOCK"
  [[ "$BASE_SHA" =~ ^[0-9a-f]{40}$ && ${#TOPICS[@]} -gt 0 ]] || fail "Incomplete assembly lock"
  git -C "$WORKTREE_PATH" merge-base --is-ancestor "$BASE_SHA" HEAD || fail "Worktree is not based on saved base SHA"
else
  [[ -f "$BRANCHES_FILE" ]] || fail "Missing branches file: $BRANCHES_FILE"
  git -C "$REPO_PATH" fetch origin
  BASE_SHA="$(git -C "$REPO_PATH" rev-parse --verify 'refs/remotes/origin/main^{commit}')"
  # Resolve the complete input set before changing a worktree or branch.
  while IFS= read -r line || [[ -n "$line" ]]; do
    topic="$(printf '%s' "$line" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
    [[ -n "$topic" && "$topic" != \#* ]] || continue
    [[ "$topic" != *$'\t'* && "$topic" != *' '* ]] || fail "Invalid topic: $topic"
    if [[ "$topic" == refs/* || "$topic" == origin/* || "$topic" =~ ^[0-9a-f]{40}$ ]]; then
      ref="$topic"
      sha="$(git -C "$REPO_PATH" rev-parse --verify "${ref}^{commit}")" || fail "Missing required topic: $topic"
    else
      local_sha="$(git -C "$REPO_PATH" rev-parse --verify --quiet "refs/heads/${topic}^{commit}" || true)"
      remote_sha="$(git -C "$REPO_PATH" rev-parse --verify --quiet "refs/remotes/origin/${topic}^{commit}" || true)"
      [[ -n "$local_sha$remote_sha" ]] || fail "Missing required topic: $topic"
      [[ -z "$local_sha" || -z "$remote_sha" || "$local_sha" == "$remote_sha" ]] || fail "Ambiguous topic $topic: local and origin differ; choose an explicit ref"
      if [[ -n "$remote_sha" ]]; then ref="refs/remotes/origin/$topic"; sha="$remote_sha"; else ref="refs/heads/$topic"; sha="$local_sha"; fi
    fi
    [[ "$ref" != refs/heads/dev && "$ref" != refs/remotes/origin/dev && "$ref" != origin/dev ]] || fail "dev cannot be an assembly topic"
    TOPICS+=("$ref"); SHAS+=("$sha")
  done < "$BRANCHES_FILE"
  [[ ${#TOPICS[@]} -gt 0 ]] || fail "No required topics were listed"
  if registered; then
    [[ "$(git -C "$WORKTREE_PATH" symbolic-ref --short HEAD)" == dev ]] || fail "Refusing to replace a worktree on a branch other than dev"
    ! git -C "$WORKTREE_PATH" rev-parse --verify --quiet MERGE_HEAD >/dev/null || fail "Pending merge; resolve and use --continue"
    check_clean
    # Git requires --force for populated submodules; all data was checked above.
    git -C "$REPO_PATH" worktree remove --force "$WORKTREE_PATH"
  fi
  [[ ! -e "$WORKTREE_PATH" ]] || fail "Unregistered path exists: $WORKTREE_PATH"
  git -C "$REPO_PATH" worktree add -B dev "$WORKTREE_PATH" "$BASE_SHA"
  LOCK="$(git -C "$WORKTREE_PATH" rev-parse --absolute-git-dir)/natten-assembly.tsv"
  {
    printf 'fork-build\t%s\n' "$N"
    printf 'base\trefs/remotes/origin/main\t%s\n' "$BASE_SHA"
    for i in "${!TOPICS[@]}"; do printf 'topic\t%s\t%s\n' "${TOPICS[$i]}" "${SHAS[$i]}"; done
  } > "$LOCK"
  # Preserve any existing attributes; only the changelog needs union merging.
  printf '\nCHANGELOG.md merge=union\n' >> "$WORKTREE_PATH/.gitattributes"
  git -C "$WORKTREE_PATH" add .gitattributes
  git -C "$WORKTREE_PATH" commit -m 'Integration: union-merge topic changelog entries'
fi

for i in "${!TOPICS[@]}"; do
  sha="${SHAS[$i]}"
  [[ "$sha" =~ ^[0-9a-f]{40}$ ]] || fail "Invalid topic SHA in lock"
  git -C "$WORKTREE_PATH" cat-file -e "${sha}^{commit}"
  echo "Merging ${TOPICS[$i]} at $sha"
  if ! git -C "$WORKTREE_PATH" -c rerere.enabled=true merge --no-ff --no-edit --rerere-autoupdate "$sha"; then
    if git -C "$WORKTREE_PATH" rev-parse --verify --quiet MERGE_HEAD >/dev/null && [[ -z "$(git -C "$WORKTREE_PATH" diff --name-only --diff-filter=U)" ]]; then
      git -C "$WORKTREE_PATH" -c rerere.enabled=true commit --no-edit
    elif [[ "$RESOLVE_MANUALLY" -eq 1 ]]; then
      echo "Resolve and commit this merge, then run the same command with --continue. Inputs remain locked in $LOCK." >&2
      exit 2
    else
      if git -C "$WORKTREE_PATH" rev-parse --verify --quiet MERGE_HEAD >/dev/null; then git -C "$WORKTREE_PATH" merge --abort; fi
      exit 2
    fi
  fi
done
for sha in "$BASE_SHA" "${SHAS[@]}"; do
  git -C "$WORKTREE_PATH" merge-base --is-ancestor "$sha" HEAD || fail "Missing locked ancestor: $sha"
done

# Initialize the final merged submodule revisions, not only the base revisions.
git -C "$WORKTREE_PATH" submodule sync --recursive
git -C "$WORKTREE_PATH" submodule update --init --recursive

VERSION_FILE="$WORKTREE_PATH/src/natten/version.py"
CURRENT_VERSION="$(sed -nE 's/^__version__[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' "$VERSION_FILE")"
[[ -n "$CURRENT_VERSION" && "$CURRENT_VERSION" != *$'\n'* ]] || fail "Cannot read unique version from $VERSION_FILE"
FULL_VERSION="${CURRENT_VERSION%%+*}+fork.${N}"
TAG="fork/$FULL_VERSION"
if [[ "$DO_TAG" -eq 1 ]] && git -C "$REPO_PATH" show-ref --verify --quiet "refs/tags/$TAG"; then fail "Tag already exists: $TAG"; fi
sed -i "s/^__version__[[:space:]]*=.*/__version__ = \"${FULL_VERSION}\"/" "$VERSION_FILE"
mkdir -p "$WORKTREE_PATH/integration"
cp "$LOCK" "$WORKTREE_PATH/integration/assembly.lock"
git -C "$WORKTREE_PATH" add src/natten/version.py integration/assembly.lock
if ! git -C "$WORKTREE_PATH" diff --cached --quiet; then git -C "$WORKTREE_PATH" commit -m "Fork build $FULL_VERSION"; fi
if [[ "$DO_TAG" -eq 1 ]]; then git -C "$WORKTREE_PATH" tag "$TAG" HEAD; fi
echo "Assembled $FULL_VERSION at $(git -C "$WORKTREE_PATH" rev-parse HEAD)"
echo "Locked inputs: $WORKTREE_PATH/integration/assembly.lock"

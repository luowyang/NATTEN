# Fork CI

This `.github/` directory is fork-only tooling. It does not exist upstream
and is not meant to ever go through an upstream PR -- see `dev`'s own
purpose below.

## The dev-branch model (context for the workflows below)

`dev` is a throwaway integration branch, rebuilt from scratch each time:
`origin/main` (upstream) with every topic branch in
`integration/branches.txt` merged on top (`integration/assemble.sh`),
then one commit that stamps a PEP 440 local-version label
(`<base>+fork.N`) onto `src/natten/version.py`. Each fork release is a tag
`fork/<version>` on a specific `dev` commit. `dev` itself is never pushed
and never opened as a PR; only the tag is pushed. `fork-ci` (this branch)
is a separate, permanent topic branch carrying just the `.github/`
workflow files -- `assemble.sh` merges it into `dev` like any other topic
branch (see `branches.txt`), so every fork build gets these workflows too.

## Tag -> release flow

```
integration/assemble.sh N /cpfs/luowenyang/natten_dev/NATTEN --tag
git -C /cpfs/luowenyang/natten_dev/wt_dev push origin refs/tags/fork/<version>
```

Pushing that tag triggers `wheel.yml`, which builds the wheel, runs a
CPU-only import smoke test, uploads the wheel + manifest as a workflow
artifact, and creates (or updates, if one already exists) a GitHub
pre-release for the tag with the wheel and manifest attached.

Consumers fetch a release build with:

```
gh release download fork/<version> --repo luowyang/NATTEN --pattern '*.whl' --dir <dir>
```

(or `gh api repos/luowyang/NATTEN/releases/tags/fork/<version>` + `curl`
following the redirect to `objects.githubusercontent.com`).

You can also rebuild and publish a wheel for an **existing** tag with
current CI logic (e.g. after a workflow bugfix) without re-running
`assemble.sh`:

```
gh workflow run wheel.yml --ref fork-ci -f ref=fork/<version>
```

`--ref fork-ci` selects which copy of the workflow *file* runs;
`-f ref=fork/<version>` tells it what to actually check out and build.
`wheel.yml` never deletes or recreates a release that already exists --
it only creates one if missing, or uploads (`--clobber`) the wheel +
manifest onto an existing one.

## Why GitHub-hosted runners, not self-hosted

The original design used one self-hosted GPU runner on dev machine 2, so
wheel builds and GPU tests would happen on the same hardware the fork is
actually developed on. That runner never came online: the org's shared
egress proxy (`10.33.197.9:3128`) allows outbound
`github.com`/`api.github.com`/`objects.githubusercontent.com`/
`codeload.github.com` traffic but returns 403 on CONNECT to
`pipelines.actions.githubusercontent.com` (confirmed directly in a runner
registration attempt's own diag log: `GET .../connectionData ...
HttpRequestException: The proxy tunnel request to proxy
'http://10.33.197.9:3128/' failed with status code '403'`) and to
`uploads.github.com` -- both required for a self-hosted runner to
register, poll for jobs, or upload anything. This is a network policy
choice (downloads only, no uploads, no proxy exception requested), not a
CI misconfiguration, so it isn't something to retry or work around here.

Consequence: **`wheel.yml` cannot run GPU kernel tests.** It still builds
a real CUDA extension (`NATTEN_CUDA_ARCH=9.0`, targeting sm_90a) using a
CUDA 12.8 toolkit installed on the runner itself, and verifies the wheel
installs and `import natten` works -- but that's an import smoke test,
not kernel correctness. **GPU validation stays local and manual**, on the
dev machines, via the existing `integration/` scripts
(`build_wheel.sh`'s own smoke-test phase, `run_extended.sh`, etc.) --
those are unrelated to and unaffected by anything in `.github/`.

`ci/runner/` (sibling to this repo, not part of it) is left as a one-line
marker explaining the above; the downloaded runner tarball and the
registration/lifecycle scripts written for it were deleted rather than
left dormant, since this was a considered decision, not a temporary
outage.

## The three workflows

- **`ci.yml`** -- `ubuntu-latest`, on push to `fork-ci`/`dev` and on
  `workflow_dispatch`. Lint (`ufmt check`, `flake8`, `mypy` on
  `src/natten`) plus `tests/test_varlen_layout.py`'s host-only classes,
  when that file exists on the ref being tested (it's added by this
  fork's varlen topic branches, not upstream -- see that file's own
  comment in the workflow for which refs have it). No CUDA build.

- **`wheel.yml`** -- `ubuntu-latest`, on push of tags matching `fork/*`
  and on `workflow_dispatch` (input: `ref`). Builds the wheel (see
  above), uploads it + a manifest as an artifact, and on a `fork/*` tag
  ref creates/updates that tag's GitHub Release.

- **`extended.yml`** -- **currently non-functional**, see the status note
  at the top of the file itself. It targets the self-hosted runner this
  fork no longer has; a dispatch will queue forever. Left in place,
  unaddressed, pending a decision (delete it, or redesign what "extended
  coverage" means without a GPU runner in CI).

## Fork-only

Everything under `.github/` exists only on `fork-ci` (and, once merged,
`dev`) -- it is never meant to go upstream via a PR.

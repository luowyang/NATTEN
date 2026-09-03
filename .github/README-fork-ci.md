# Fork CI

This `.github/` directory is fork-only tooling for [luowyang/NATTEN](https://github.com/luowyang/NATTEN),
a fork of [NATTEN](https://github.com/SHI-Labs/NATTEN). It does not exist upstream and is not meant to
go through an upstream PR.

## What this CI is for

Pushing a tag `fork/<version>` (e.g. `fork/0.21.7+fork.2`) builds a NATTEN wheel and publishes it as a
GitHub pre-release, so downstream projects can install a specific fork build without building from
source themselves.

Each build targets exactly one combination:

- **CUDA 12.8**
- **PyTorch 2.11**
- **Python 3.11** (cp311, `manylinux_x86_64`)
- **sm90 only** (`NATTEN_CUDA_ARCH=9.0`, i.e. Hopper; the wheel has no other architecture's kernels)

The build runs on a GitHub-hosted runner, which has no GPU. It compiles a real CUDA extension and
verifies `import natten` works, but cannot run GPU kernel tests. GPU correctness is validated
separately, outside this CI, on machines that have the right hardware.

The wheel declares no `torch` dependency (checked: its metadata has no `Requires-Dist` at all, only
`Requires-Python: >=3.9`), so `pip` will not pull one in for you — install it into an environment that
already has a matching torch (2.11, cu128).

## Consumer flow: installing a released wheel

Download the wheel and its manifest for a given tag:

```bash
gh release download fork/0.21.7+fork.N --repo luowyang/NATTEN --pattern '*.whl'
gh release download fork/0.21.7+fork.N --repo luowyang/NATTEN --pattern 'MANIFEST-*.txt'
```

Verify the wheel against the SHA256 recorded in the manifest before installing it:

```bash
sha256sum natten-0.21.7+fork.N-cp311-cp311-linux_x86_64.whl
grep SHA256 MANIFEST-0.21.7+fork.N.txt
# the two hashes must match
```

Install into an environment that already has torch 2.11 (cu128) and Python 3.11:

```bash
pip install --no-deps natten-0.21.7+fork.N-cp311-cp311-linux_x86_64.whl
```

`--no-deps` is required (see above: the wheel intentionally declares no dependencies, so a plain
`pip install` would not fail on a mismatched torch, it would just not know to check).

## Maintainer flow: cutting a release

`dev` is a throwaway integration branch, rebuilt from scratch each time: `origin/main` (upstream) with
every topic branch in `integration/branches.txt` merged on top (`integration/assemble.sh`), then one
commit that stamps a PEP 440 local-version label (`<base>+fork.N`) onto `src/natten/version.py`. `dev`
itself is never pushed and never opened as a PR; only the tag is pushed. `fork-ci` (this branch) is a
separate, permanent topic branch carrying just the `.github/` workflow files — `assemble.sh` merges it
into `dev` like any other topic branch, so every fork build gets these workflows too.

```bash
integration/assemble.sh N <path-to-fork-checkout> --tag
git -C <assemble.sh's --worktree, default wt_dev> push origin refs/tags/fork/<version>
```

Pushing that tag triggers `wheel.yml`, which builds the wheel, runs a CPU-only import smoke test,
uploads the wheel + manifest as a workflow artifact, and creates (or updates, if one already exists)
a GitHub pre-release for the tag with the wheel and manifest attached. `wheel.yml` never deletes or
recreates a release that already exists — it only creates one if missing, or replaces
(`--clobber`) the wheel + manifest on an existing one.

### Backfilling assets for an existing tag

To rebuild and publish a wheel for a tag that already exists (e.g. after a workflow bugfix), without
re-running `assemble.sh`:

```bash
gh workflow run wheel.yml --ref fork-ci -f ref=fork/<version>
```

`--ref fork-ci` selects which copy of the workflow *file* runs; `-f ref=fork/<version>` tells it what
to actually check out and build. This also doubles as the recovery path when a build run is killed by
the job timeout (see below) — re-dispatch the same command; the build resumes from cache rather than
starting cold.

## Reading CI logs

The network this CI is dispatched and monitored from cannot reach hosted-run logs or artifacts the
normal way — `gh run view --log`, the jobs/logs API, and artifact downloads are all unreachable from
there. Every `wheel.yml` run works around this by pushing its own logs to an orphan branch, `ci-logs`,
via a plain `git push` over `github.com` (only the *reading* side is blocked; a runner pushing to
`github.com` is unaffected). Files land under `runs/<run_id>-<attempt>/`:

- `started.txt` — pushed right after dependencies install; confirms the run started and records initial
  disk/memory/CPU state.
- `sampler.txt` — resource samples (`free -m`, top-10 RSS processes, `df -h /`) taken every minute during
  the build and pushed every 5 samples, so a run that's still going can be checked without waiting for it
  to finish.
- `summary.txt` — pushed at the end (`if: always()`, so this lands even on failure): every step's outcome,
  final disk/memory state, and `sccache --show-stats`.
- `build.log` (or `build.log.gz` if over 1 MB) — the full `python -m build` output.

Read any of these with:

```bash
gh api "repos/luowyang/NATTEN/contents/runs/<run_id>-<attempt>/summary.txt?ref=ci-logs" --jq .content | base64 -d
```

(swap the filename for `started.txt`, `sampler.txt`, or `build.log`; pipe through `gunzip` as well for
`build.log.gz`). `<run_id>` and `<attempt>` come from the workflow run URL, e.g.
`.../actions/runs/33680769628` is run id `33680769628`, attempt `1` on the first try.

A clean build's `build.log` also gets scanned for lines matching `error|Error|Killed|No space|fatal`;
the last 20 matches are emitted as `::error::` workflow annotations, visible in the Actions UI or via
`gh run view -v` without needing any of the blocked endpoints.

## Measured operating facts

These were established by measurement (see commit `a5be24b` on this branch for the full methodology)
and shape how the workflow is configured:

- **`NATTEN_N_WORKERS=1`.** The build compiles with a single worker, not the runner's 4 vCPUs, because
  memory — not CPU — is the binding constraint. The three largest translation units (all in
  `hopper_fna_bwd`) each peak at roughly **10.5 GB** of process-tree RSS (`nvcc` plus its `cicc`/`ptxas`
  children, sampled every 2s) when compiled with this build's actual flags (`NATTEN_CUDA_ARCH=9.0`,
  `NATTEN_AUTOGEN_POLICY=fine`), measured locally (dev machine 2, CPU-only compile — no GPU needed to
  compile). The runner has 16 GB of RAM. Two concurrent workers risk two such units landing together for
  roughly 21 GB combined, which does not fit; one worker keeps peak usage under ~11 GB, with headroom.
- **Cold build: ~5h00m** of the workflow's 6-hour (360 min) job timeout, with an 8.6% sccache hit rate
  (nothing in cache yet).
- **Warm rebuild of an unchanged tree: ~26 min**, 100% sccache hits.
- **If a run hits the 6-hour limit:** re-dispatch the identical command
  (`gh workflow run wheel.yml --ref fork-ci -f ref=<tag>`). sccache
  (`SCCACHE_GHA_ENABLED=true`, scoped to this repo's Actions cache) writes each compiled object to
  cache as soon as it compiles, not just at the end of a successful build, so the new run picks up
  every object the killed run already finished and only compiles the remainder.
- **Cache budget:** this repo's total Actions cache is around 10 GB, of which the CUDA 12.8 toolkit
  installer alone accounts for ~5.4 GB. That leaves proportionally less room for sccache's own object
  cache; if cache pressure evicts sccache entries, expect a build closer to the ~5h cold case than the
  ~26 min warm one.

## Why GitHub-hosted runners, not self-hosted

The build runs on GitHub-hosted runners because this fork's network does not allow self-hosted runners
to reach GitHub Actions; there is no GPU on hosted runners, so kernel tests run elsewhere.

Consequence: **`wheel.yml` cannot run GPU kernel tests.** It still builds a real CUDA extension
(`NATTEN_CUDA_ARCH=9.0`, targeting sm_90a) using a CUDA 12.8 toolkit installed on the runner itself, and
verifies the wheel installs and `import natten` works — but that's an import smoke test, not kernel
correctness. **GPU validation happens outside this CI**, via this fork's own build/test tooling
(`integration/build_wheel.sh`'s own smoke-test phase, `run_extended.sh`, etc.) — those are unrelated to
and unaffected by anything in `.github/`.

## The three workflows

- **`ci.yml`** — `ubuntu-latest`, on push to `fork-ci`/`dev` and on `workflow_dispatch`. Lint
  (`ufmt check`, `flake8`, `mypy` on `src/natten`) plus `tests/test_varlen_layout.py`'s host-only
  classes, when that file exists on the ref being tested (it's added by this fork's varlen topic
  branches, not upstream — see that file's own comment in the workflow for which refs have it). No CUDA
  build.

- **`wheel.yml`** — `ubuntu-latest`, on push of tags matching `fork/*` and on `workflow_dispatch`
  (input: `ref`). Builds the wheel (see above), uploads it + a manifest as an artifact, and on a
  `fork/*` tag ref creates/updates that tag's GitHub Release.

- **`extended.yml`** — **currently non-functional**, see the status note at the top of the file itself.
  It targets a self-hosted runner this fork no longer has; a dispatch will queue forever. Left in
  place, unaddressed, pending a decision (delete it, or redesign what "extended coverage" means without
  a GPU runner in CI).

## Fork-only

Everything under `.github/` exists only on `fork-ci` (and, once merged, `dev`) — it is never meant to
go upstream via a PR.

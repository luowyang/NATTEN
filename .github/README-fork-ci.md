# Fork CI

This `.github/` directory is fork-only tooling for [luowyang/NATTEN](https://github.com/luowyang/NATTEN),
a fork of [NATTEN](https://github.com/SHI-Labs/NATTEN). It does not exist upstream and is not meant to
go through an upstream PR.

## What this CI is for

Building a NATTEN wheel and publishing it where downstream projects can install it, without anyone
having to build from source themselves. `wheel.yml` does that on three **channels**, in the PyTorch
tradition: a permanent release you pin against, a nightly you track, and a warm build that publishes
nothing and only exists to keep the compiler cache current.

| Channel | Triggered by | Version | Published as | Kept |
| --- | --- | --- | --- | --- |
| `release` | push of a `fork/*` tag (`release.yml` dispatches the build onto `fork-ci`, see below), or a dispatch whose `ref` input is a `fork/*` tag | `src/natten/version.py` exactly as `assemble.sh` stamped it, e.g. `0.21.7+fork.3` | a normal GitHub **Release** on that tag, wheel + manifest attached | permanently |
| `nightly` | a dispatch with `channel=nightly` — normally the 19:00 UTC schedule in `nightly.yml` on `main` | stamped by CI as `<newest fork/* tag's version>.dev<YYYYMMDD>`, e.g. `0.21.7+fork.3.dev20260922` | a GitHub **pre-release** tagged `nightly/<YYYYMMDD>`, created on the commit that was built, wheel + manifest attached | 14 days, then deleted tag and all |
| `warm` | push to `dev`, or any dispatch that does not name a `fork/*` tag | whatever the branch already carries | nothing is published — the wheel is a workflow artifact only | 30 days (artifact retention) |

A dispatch picks its channel with the `channel` input; the default, `auto`, means "a `fork/*` tag is a
release, anything else is warm". All three channels run the identical `gate` → `warm` → `build`
pipeline and compile the identical tree with identical flags — the channel only decides what version
the wheel carries and where it goes afterwards.

Why `.dev<date>` sorts where it should: PEP 440 compares local-version labels part by part, so
`0.21.7+fork.3` < `0.21.7+fork.3.dev20260922` < `0.21.7+fork.4`. A nightly therefore reads, to `pip`
and to any version-range check, as "after the last release, before the next one", which is exactly
what it is.

A nightly whose `dev` has not moved since the previous nightly is **built but not published** — the
build still runs, because that is what keeps sccache warm, but there is no new pre-release, and the
manifest records `Published: skipped-unchanged`.

Each build produces one wheel per Python version, all sharing the rest of the combination:

- **CUDA 12.8**
- **PyTorch 2.11**
- **Python 3.10 and 3.11** (cp310 and cp311, `linux_x86_64`; one wheel and one manifest each)
- **sm90 only** (`NATTEN_CUDA_ARCH=9.0`, i.e. Hopper; the wheel has no other architecture's kernels)

Releases up to `fork/0.21.7+fork.4` carry the cp311 wheel only, with its manifest named
`MANIFEST-<version>.txt`.

The build runs on a GitHub-hosted runner, which has no GPU. It compiles a real CUDA extension and
verifies `import natten` works, but cannot run GPU kernel tests. GPU correctness is validated
separately, outside this CI, on machines that have the right hardware.

The wheel declares no `torch` dependency (checked: its metadata has no `Requires-Dist` at all, only
`Requires-Python: >=3.9`), so `pip` will not pull one in for you — install it into an environment that
already has a matching torch (2.11, cu128).

## Consumer flow: installing a wheel

**Pin a release** (`fork/<version>`) for anything you expect to reproduce later — those are permanent,
and they are the only thing that is. **Track a nightly** (`nightly/<date>`) only when you need a fix
that has not been cut into a release yet, and expect it to be gone in 14 days; nothing that has to be
rebuildable months from now should depend on one. **Never depend on a warm build** — it publishes
nothing at all, and the workflow artifact it does leave behind is unreachable from this network
anyway.

Download the wheel and its manifest, by tag:

```bash
# a release
gh release download fork/0.21.7+fork.N --repo luowyang/NATTEN --pattern '*.whl'
gh release download fork/0.21.7+fork.N --repo luowyang/NATTEN --pattern 'MANIFEST-*.txt'

# a nightly (see `gh release list --repo luowyang/NATTEN` for what still exists)
gh release download nightly/20260922 --repo luowyang/NATTEN --pattern '*.whl'
gh release download nightly/20260922 --repo luowyang/NATTEN --pattern 'MANIFEST-*.txt'
```

Release attachments are the one GitHub surface this network can reach; Actions artifacts and run logs
are not (see "Reading CI logs" below). That is why every channel that publishes anything publishes it
as a release asset.

Verify the wheel against the SHA256 recorded in the manifest before installing it:

```bash
sha256sum natten-0.21.7+fork.N-cp311-cp311-linux_x86_64.whl
grep SHA256 MANIFEST-0.21.7+fork.N-cp311.txt
# the two hashes must match; cp310 likewise
```

Install the wheel for your Python into an environment that already has torch 2.11 (cu128):

```bash
pip install --no-deps natten-0.21.7+fork.N-cp311-cp311-linux_x86_64.whl   # Python 3.11
pip install --no-deps natten-0.21.7+fork.N-cp310-cp310-linux_x86_64.whl   # Python 3.10
```

`--no-deps` is required (see above: the wheel intentionally declares no dependencies, so a plain
`pip install` would not fail on a mismatched torch, it would just not know to check).

The manifest inside each release says which channel produced it (`Channel:`), which commit was built
(`Built from commit:`), and, for a nightly, which release its version is expressed against
(`Base tag:`).

## Maintainer flow

`dev` is a throwaway integration branch, rebuilt from scratch each time: `origin/main` (upstream) with
every topic branch in `integration/branches.txt` merged on top (`integration/assemble.sh`), then one
commit that stamps a PEP 440 local-version label (`<base>+fork.N`) onto `src/natten/version.py`.
`fork-ci` (this branch) is a separate, permanent topic branch carrying just the `.github/` workflow
files — `assemble.sh` merges it into `dev` like any other topic branch, so every fork build gets these
workflows too.

### Cutting a release

```bash
integration/assemble.sh N <path-to-fork-checkout> --tag
git -C <assemble.sh's --worktree, default wt_dev> push origin refs/tags/fork/<version>
```

Pushing that tag fires `release.yml`, a one-step workflow that dispatches `wheel.yml` on `fork-ci`
with `ref=<the tag>` and `channel=release`. The indirection is for the cache: GitHub's Actions cache is
readable only from the ref that wrote it or from the default branch, so a build triggered directly by
the tag push could never reuse what the nightlies (dispatched on `fork-ci`) left in sccache and every
release was a cold build (fork.3: 1h56m). Dispatched onto `fork-ci`, a release of a tree the nightly
already built skips the warm shards and finishes in about 40 minutes. The build then runs on the
`release` channel exactly as before: it builds the wheel, runs a CPU-only
import smoke test, uploads the wheel + manifest as a workflow artifact, and creates (or updates, if
one already exists) a GitHub Release for the tag with the wheel and manifest attached. A release is
**not** marked pre-release — `fork/*` is this fork's real release channel, and a pre-release is
excluded from "latest" and from tooling that only looks at full releases. `wheel.yml` never deletes or
recreates a release that already exists; it only creates one if missing, or replaces (`--clobber`) the
wheel + manifest on an existing one.

If you also push `dev` in the same breath, push it **before** the tag, or expect two runs: a `dev`
push is a `warm` build in its own right, and the two serialize behind the `wheel-build` concurrency
group. The warm one is redundant when a tag build of the same tree is already queued.

The `fork/*` releases cut before this changed were created with `--prerelease`, and CI leaves an
existing release's flags alone. Flip the old ones by hand once, if you want "latest" to mean anything:

```bash
gh release edit fork/<version> --repo luowyang/NATTEN --prerelease=false
```

### Keeping the cache warm

Pushing `dev` runs `wheel.yml` on the `warm` channel: same build, nothing published. Its only job is
to leave sccache warm so the next release or nightly of that tree skips straight to the real build.
You can also dispatch one by hand against any ref:

```bash
gh workflow run wheel.yml --repo luowyang/NATTEN --ref fork-ci -f ref=<branch-or-sha> -f channel=warm
```

### Nightlies

`nightly.yml` on `main` dispatches one every day at 19:00 UTC (03:00 Beijing). To run one by hand:

```bash
gh workflow run wheel.yml --repo luowyang/NATTEN --ref fork-ci -f ref=dev -f channel=nightly
```

CI stamps the version itself (`<newest fork/* tag's version>.dev<YYYYMMDD>`) by rewriting
`src/natten/version.py` in the runner's workspace, without committing it — `setup.py`'s
`get_version()` reads that file and nothing else, so it is the only hook, and it is the same one
`assemble.sh` uses for a tag. The stamp cannot make a nightly compile cold: the version string never
reaches a compiler, so every nvcc command line, and therefore every sccache key, is identical to an
unstamped build of the same tree.

Nightlies older than 14 days are deleted, tag and all, at the end of each nightly run. Only
`nightly/<YYYYMMDD>` tags are ever considered; `fork/*` releases are permanent.

### Backfilling assets for an existing tag

To rebuild and publish a wheel for a tag that already exists (e.g. after a workflow bugfix), without
re-running `assemble.sh`:

```bash
gh workflow run wheel.yml --repo luowyang/NATTEN --ref fork-ci -f ref=fork/<version>
```

`--ref fork-ci` selects which copy of the workflow *file* runs; `-f ref=fork/<version>` tells it what
to actually check out and build, and `channel` can be left at `auto` because a `fork/*` ref resolves
to `release` on its own. This also doubles as the recovery path when a build run is killed by the job
timeout (see below) — re-dispatch the same command; the build resumes from cache rather than starting
cold.

## The nightly dispatcher on `main`

`main` is this fork's default branch and an otherwise untouched mirror of upstream. It carries exactly
one fork-specific file, `.github/workflows/nightly.yml`, for one reason: **GitHub fires `schedule`
only from the default branch.** A cron on `fork-ci` would never run.

So `nightly.yml` holds the schedule and nothing else. Its single job dispatches the real build:

```bash
gh workflow run wheel.yml --repo luowyang/NATTEN --ref fork-ci -f ref=dev -f channel=nightly
```

`--ref fork-ci` and `-f ref=dev` are answering two different questions — which copy of the workflow
*file* runs, and what gets *built*. Splitting them means a nightly always uses the current `fork-ci`
workflow, even when `dev` was last assembled from an older one.

The dispatch is also why this works at all: events created with the automatic `GITHUB_TOKEN`
deliberately do not start new workflow runs, so that workflows cannot trigger each other in a loop.
`workflow_dispatch` is one of the two documented exceptions. The same rule, plus the fact that
`wheel.yml`'s push trigger only matches `fork/*` and `dev`, is why the `nightly/<date>` tag a run
creates for itself cannot start another run.

The job needs `actions: write` and nothing else — it creates no tag, release or commit of its own.
`wheel.yml` does all of that under its own `contents: write`.

## Reading CI logs

The network this CI is dispatched and monitored from cannot reach hosted-run logs or artifacts the
normal way — `gh run view --log`, the jobs/logs API, and artifact downloads are all unreachable from
there. Every `wheel.yml` run works around this by pushing its own logs to an orphan branch, `ci-logs`,
via a plain `git push` over `github.com` (only the *reading* side is blocked; a runner pushing to
`github.com` is unaffected). Files land under `runs/<run_id>-<attempt>/`:

- `shard-<index>.txt` — one per `warm` shard (`<index>` is `0`..`SHARD_COUNT-1`), pushed when that
  shard's job ends (`if: always()`): elapsed time, own-shard compiled/failed file counts and names, and
  `sccache --show-stats` as seen by that shard.

Each `build` matrix leg writes the other four files into its own subdirectory, `py3.10/` or `py3.11/`:

- `started.txt` — pushed by `build`, right after dependencies install; confirms the run started and
  records initial disk/memory/CPU state and `warm`'s aggregate result.
- `sampler.txt` — resource samples (`free -m`, top-10 RSS processes, `df -h /`) taken every minute during
  `build`'s compile and pushed every 5 samples, so a run that's still going can be checked without
  waiting for it to finish.
- `summary.txt` — pushed by `build` at the end (`if: always()`, so this lands even on failure): every
  step's outcome, `warm`'s aggregate result, final disk/memory state, and `sccache --show-stats`.
- `build.log` (or `build.log.gz` if over 1 MB) — the full `python -m build` output from `build` (each
  shard's own build output is not pushed in full, only the tail included in its `shard-<index>.txt`).

Read any of these with:

```bash
gh api "repos/luowyang/NATTEN/contents/runs/<run_id>-<attempt>/py3.11/summary.txt?ref=ci-logs" --jq .content | base64 -d
```

(swap the filename for `started.txt`, `sampler.txt`, or `build.log`; pipe through `gunzip` as well for
`build.log.gz`). `<run_id>` and `<attempt>` come from the workflow run URL, e.g.
`.../actions/runs/33680769628` is run id `33680769628`, attempt `1` on the first try.

A clean build's `build.log` also gets scanned for lines matching `error|Error|Killed|No space|fatal`;
the last 20 matches are emitted as `::error::` workflow annotations, visible in the Actions UI or via
`gh run view -v` without needing any of the blocked endpoints.

## How a cold build is parallelized

A cold build — nothing in sccache yet — compiling every CUDA translation unit serially, one worker at a
time (`NATTEN_N_WORKERS=1`, see below), takes about 5 hours on a single runner: slow to iterate on, and
close to unworkable against even the 6-hour job timeout. `wheel.yml` runs a `warm` job before `build` to
avoid paying that cost serially:

- **Topology.** `warm` is a `strategy.matrix` of `SHARD_COUNT` (currently 8) parallel runners,
  `shard: 0..7`. Each shard runs the *same* `python -m build --no-isolation --wheel -o dist/` command as
  `build`, with the same `NATTEN_*`/`SCCACHE_*` environment, plus two extra variables that only the nvcc
  wrapper reads: `NATTEN_CI_SHARD_INDEX` and `NATTEN_CI_SHARD_COUNT`. Both jobs check out the same ref
  identically as their own first step (a local composite action can't do its own checkout — GitHub
  Actions resolves `uses: ./path` from whatever checkout already exists in the job's workspace, so
  checkout can't be a step inside the very composite action being referenced; confirmed the hard way,
  every job failing in seconds with "Can't find 'action.yml' ... Did you forget to run actions/checkout
  before running your local action?" when an earlier revision tried it). Shared setup after that
  (Python, CUDA 12.8, sccache, the compiler wrapper scripts) lives in one composite action,
  `.github/actions/prepare-build-env`, used by both jobs — GitHub Actions workflow YAML has no anchors
  or merge keys, so this is the supported way to keep two jobs' steps from drifting apart; if they did,
  `warm` and `build` would issue different compile commands for the same file and sccache would miss.

- **Hash sharding.** The nvcc wrapper (`.github/scripts/nvcc-wrapper.sh`) hashes each `csrc/`-tree
  translation unit's path with `cksum` and reduces it mod `NATTEN_CI_SHARD_COUNT`. A file that reduces to
  *this* shard's own index compiles for real, through sccache, exactly as `build` would. A file that
  reduces to a *different* shard's index is compiled instead from a fixed empty `.cu` stand-in, using the
  real compiler directly — bypassing sccache entirely, so no empty object is ever written into the shared
  cache under that file's real key. Compiling an empty file costs roughly 1–2 seconds versus the real
  file's minutes-scale cost, so a shard's wall time is dominated by its own ~1/`SHARD_COUNT` of the real
  translation units, not by touching everyone else's.

- **Why `build` then hits cache.** Across all shards, every real translation unit gets compiled for real
  by exactly one shard, landing in sccache (`SCCACHE_GHA_ENABLED=true`, this repo's Actions cache) under
  the same key `build` will look up later — because `warm`'s own-shard compile commands are byte-for-byte
  identical to `build`'s (same composite action, same checkout, same environment). When `build` runs
  afterwards (`needs: warm`), it recompiles every file with that identical command line and hits cache on
  (almost) all of them, instead of compiling anything cold.

- **Python 3.11 only.** `warm` runs prepare-build-env with its default Python, 3.11. sccache keys each
  nvcc stage on that stage's own input, so the `build` leg for Python 3.10 hits what the shards primed
  for every translation unit that includes no torch header. The ones that do include one (the Hopper
  kernel families and the `src/` entry points) carry `Python.h` and a per-version include path, miss,
  and get compiled by that leg itself, serially, the first time a tree is built: on fork.5, 62 of 331
  nvcc compile requests, which took `Build wheel` 2h55m against 29 min on the 3.11 leg.

- **Failure handling.** A shard's own `python -m build` very likely fails or produces a useless wheel
  (most of its objects are empty stand-ins) — that's expected and doesn't matter; that artifact is never
  used. What decides the shard's own pass/fail is whether it compiled its *own* shard's real files without
  error: the step fails only if it recorded an own-shard compile failure, or recorded no own-shard compile
  at all (a sign the wrapper itself is broken). `build` runs even if a shard failed
  (`if: ${{ !cancelled() }}`) — it then simply compiles whatever that shard would have cached, itself,
  cold, so a `warm` failure costs time, not correctness. `build` keeps its full 360-minute timeout as a
  backstop for exactly that case, rather than assuming `warm` succeeded.

- **Forcing a cold build.** Dispatch with a `cache_namespace` that has never been used before — it
  becomes `SCCACHE_GHA_VERSION`, which sccache folds into every cache key, so a new value means an empty
  effective cache regardless of what the default namespace already holds:

  ```bash
  gh workflow run wheel.yml --repo luowyang/NATTEN --ref <branch> -f cache_namespace=<unique-label>
  ```

  Dispatching again with the *same* `cache_namespace` reuses that namespace's now-warm cache. Leave
  `cache_namespace` blank for normal use, including every tag-triggered release build — sccache then
  uses its own default namespace, same as before this input existed.

## Skipping the warm phase on an unchanged fingerprint

`wheel.yml` starts with a `gate` job. It hashes the `csrc/` tree, the CUTLASS submodule commit,
`setup.py`, `csrc/CMakeLists.txt`, every `scripts/autogen_*.py` file, the pinned torch and CUDA
versions, the three `NATTEN_*` build variables, `SHARD_COUNT`, and `cache_namespace`. The resulting
fingerprint names a tiny marker cache entry.

- A marker miss runs all warm shards. After the real build installs and imports successfully, the build
  job saves `gate-marker.txt` under that fingerprint's cache key.
- A marker hit skips all warm shards and starts the real build directly.
- The real build always compiles every missing object from source. sccache keys objects from the actual
  preprocessed source, independently of the gate fingerprint, so a stale marker can waste time but
  cannot make the build reuse a stale object.

The channel is deliberately **not** part of the fingerprint, and neither is the nightly version stamp.
All three channels compile the identical tree with identical flags, so they must share one marker and
one set of sccache entries — that is what lets a nightly of an unchanged `dev` hit the marker that
yesterday's build saved and skip the warm shards entirely.

The gate prints its fingerprint and decision in the Actions log; each build leg repeats them in
`runs/<run_id>-<attempt>/py<version>/summary.txt` on `ci-logs`.

## Measured operating facts

These were established by measurement (see commit `a5be24b` on this branch for the full methodology)
and shape how the workflow is configured:

- **Warm-fingerprint hit: 32m05s.** Workflow
  [33778616697](https://github.com/luowyang/NATTEN/actions/runs/33778616697) skipped `warm`, then built,
  installed, imported, and uploaded the wheel successfully in 32m05s. The comparable warm-cache
  baseline [33729368562](https://github.com/luowyang/NATTEN/actions/runs/33729368562) took 55m52s.
  A marker hit is only a scheduling hint: if the underlying sccache objects have expired, `build`
  recompiles them safely and the run may be slower.

- **`NATTEN_N_WORKERS=1`.** The build compiles with a single worker, not the runner's 4 vCPUs, because
  memory — not CPU — is the binding constraint. The three largest translation units (all in
  `hopper_fna_bwd`) each peak at roughly **10.5 GB** of process-tree RSS (`nvcc` plus its `cicc`/`ptxas`
  children, sampled every 2s) when compiled with this build's actual flags (`NATTEN_CUDA_ARCH=9.0`,
  `NATTEN_AUTOGEN_POLICY=fine`), measured locally (dev machine 2, CPU-only compile — no GPU needed to
  compile). The runner has 16 GB of RAM. Two concurrent workers risk two such units landing together for
  roughly 21 GB combined, which does not fit; one worker keeps peak usage under ~11 GB, with headroom.
- **Cold build (sharded, `SHARD_COUNT=8`): 1h35m** total wall time (workflow `createdAt` 03:42:28Z ->
  `updatedAt` 05:17:30Z). `warm` phase (8 shards in parallel): 56 min (earliest shard start to latest
  shard finish); per-shard elapsed (the `Precompile shard` step's own timer): min 24m57s (shard 0), max
  48m33s (shard 6), median 29m03s; own-shard compiled-file counts ranged 22-33 (sum 202 across all 8
  shards), 0 failures on any shard. `build` then took 39 min (`Prepare build environment` 6m24s,
  `Build wheel` 32m10s), with sccache hitting 800/815 requests overall (98.16%) and, of the 203 CUDA
  translation units specifically, 198 hits / 5 misses (97.54%) -- `build` simply compiles a miss itself,
  same as it always compiles everything; a small number of misses do not fail the build, they just cost
  it a few extra minutes. (`sccache --show-stats` gives aggregate counts only; to see WHICH units a
  run compiled, read that run's `sampler.txt` on `ci-logs`, which snapshots the running compiler process
  tree every minute.) Measured on run
  [`33712330046`](https://github.com/luowyang/NATTEN/actions/runs/33712330046).
- **Re-dispatch of an unchanged tree: about 56 min** in the steady state -- run
  [`33729368562`](https://github.com/luowyang/NATTEN/actions/runs/33729368562) took 55m52s with sccache
  at 815/815 (100.00%, 203 of 203 CUDA translation units) and every shard also at 100%. This is slower
  than the pre-sharding warm rebuild below, because a re-dispatch still runs the whole `warm` phase even
  when nothing needs compiling.
- **One caveat, cause not yet identified.** The FIRST re-dispatch after the sharded cold build, run
  [`33719144384`](https://github.com/luowyang/NATTEN/actions/runs/33719144384), took 2h09m: `build` hit
  only 164 of 203 CUDA translation units and recompiled 39, all of them in the four `hopper_*` families
  (the largest objects); all 128 `fna` units hit, and sccache reported zero read errors, write errors and
  timeouts. Every shard in that same run hit 100%, including files `build` missed. The next re-dispatch,
  same commit and same `cache_namespace` with nothing changed, hit 100%. So the recompiles were a
  one-time event rather than a per-run tax, but why they happened is still open. It costs time only: a
  miss makes `build` compile that unit from source, which is what it would do without any cache.
- **Pre-sharding baseline** (single unsharded job, everything compiled serially by one job): ~5h00m
  cold (8.6% sccache hit rate, nothing in cache yet), ~26 min warm (100% sccache hits) -- see commit
  `a5be24b` on this branch for that methodology. Kept for context on how much sharding helped; the
  numbers above are the ones that describe how this workflow actually runs today.
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

## The workflows

- **`ci.yml`** (`fork-ci`) — `ubuntu-latest`, on push to `fork-ci`/`dev` and on `workflow_dispatch`.
  Lint (`ufmt check`, `flake8`, `mypy` on `src/natten`) plus `tests/test_varlen_layout.py`'s host-only
  classes, when that file exists on the ref being tested (it's added by this fork's varlen topic
  branches, not upstream — see that file's own comment in the workflow for which refs have it). No CUDA
  build.

- **`wheel.yml`** (`fork-ci`) — `ubuntu-latest`, on push of tags matching `fork/*`, on push to `dev`,
  and on `workflow_dispatch` (inputs: `ref`, `channel`, `cache_namespace`). Three jobs: `gate` resolves
  the channel and checks whether this exact build fingerprint is already warm; `warm` conditionally
  runs the sharded precompile described above; `build`, one leg per Python version, always builds the
  wheel, uploads it + a manifest as an artifact, and then publishes according to the channel — a
  Release for `release`, a `nightly/<date>` pre-release plus a 14-day prune for `nightly`, nothing for
  `warm`.

- **`nightly.yml`** (`main`) — the 19:00 UTC schedule, and nothing else. One job, `actions: write`,
  which dispatches `wheel.yml` on the `nightly` channel. It lives on `main` because that is the only
  branch GitHub fires `schedule` from; see "The nightly dispatcher on `main`" above.

## Fork-only

Everything under `.github/` on `fork-ci` (and, once merged, `dev`) exists only in this fork — it is
never meant to go upstream via a PR. The same goes for the single `nightly.yml` on `main`, which is
otherwise an untouched upstream mirror.

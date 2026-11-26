# Fork integration and GPU acceptance

These versioned tools replace the workstation copies of `assemble.sh`,
`branches.txt`, and `build_wheel.sh`. Run the tools from the chosen recipe checkout;
pass the source checkout separately. They never push or publish.

## Assemble

```sh
bash integration/assemble.sh 6 /path/to/NATTEN \
  --worktree /path/to/wt_dev --branches-file integration/branches.txt
```

Every listed topic is required. Bare branch names are accepted only when the
local and origin revisions agree, or only one exists. Use `refs/heads/topic` or
`refs/remotes/origin/topic` to select explicitly. All SHAs are resolved before
changing `dev`. Existing tracked changes, untracked files, ignored products,
and submodule files prevent worktree replacement.

Use `--resolve-manually` to leave a conflicting merge in place. Resolve and
commit the merge, then rerun with `--continue` and the same build number.
Continuation first checks tracked, untracked, ignored, and submodule files,
then reuses the saved base/topic SHAs even if branches or the input list move. The completed candidate includes `integration/assembly.lock`.
`--tag` creates a local fork tag only; it does not prove the wheel passed tests.

## Validate an existing CI wheel

Use a clean source checkout at the manifest's `source_sha`, and an isolated
conda environment with the matching Python and Torch/CUDA versions. Download
the original wheel and its JSON manifest together; do not rebuild it.

```sh
bash integration/build_wheel.sh \
  --dev-worktree /path/to/pinned-source \
  --env-path /path/to/validation-env --skip-env-clone \
  --wheel /path/to/artifacts/natten-VERSION-cp311-cp311-linux_x86_64.whl \
  --manifest /path/to/artifacts/MANIFEST-VERSION-cp311.json \
  --output /path/to/evidence/GPU-VALIDATION-cp311.json
```

To create a new isolated environment, replace `--skip-env-clone` with
`--source-env /path/to/compatible-env`. Existing environments are never removed.
The `CONDA_EXE` environment variable selects the conda executable.

Validation verifies the wheel hash, Python tag, source revision, installed
package and native extension bytes against that same wheel, and an actual CUDA
forward/backward operation. Pytest runs with isolated Python (`-I`); an in-process
guard verifies the installed bytes and import paths at session start, after
collection, and at session end. Parent checks also run before and after the suite.
It then runs both varlen test filename patterns from the pinned source checkout.
`--extra-smoke-tests 'tests/test_extra.py ...'` adds required files; missing
files fail. The default timeout is 1200 seconds. Failure, timeout, unavailable
CUDA, empty collection, or an entirely skipped suite cannot produce a passed
attestation. The shell returns the validator's nonzero exit code.

Keep evidence outside the source checkout. Each successful invocation emits a
single-wheel JSON attestation bound to `source_sha`, `recipe_sha`, `build_id`,
and the original wheel's SHA256. Run separately for Python 3.10 and 3.11; the
publisher requires both original assets and matching acceptance proofs.

## Local development build

Omit `--wheel` and `--manifest` and supply a new `--dist-dir` to build locally,
then run the same validation. Build defaults are Hopper (`9.0`), one compiler
worker, and `fine` autogen; override using the corresponding `NATTEN_*`
environment variables. Local manifests use `channel=local` and do not replace
CI provenance for promotion. Only CI-produced artifacts qualify for that path.

# Fork build and release pipeline

The fork assembles upstream and topic commits into a source candidate. A separate
recipe checkout supplies CI tools. Every build records both revisions before
starting the Python 3.10 / 3.11 matrix.

## Entry points

| Entry | Source | Result |
| --- | --- | --- |
| `dev.yml`, push to dev | Pushed commit | Host-validated artifacts; no release |
| `nightly.yml`, scheduled on main | dev resolved once | Complete host-validated nightly |
| `release.yml`, fork tag or manual tag | Tagged commit | Candidate artifacts awaiting GPU validation |
| `promote.yml`, manual | Successful candidate run/attempt | Formal release after both wheel hashes pass GPU validation |
| `wheel.yml`, manual | Explicit branch/tag/commit | Selected channel through the same controller |

The three automatic entry points call `dispatch.yml@fork-ci`. It dispatches
`wheel.yml` on fork-ci, identifies the child using a unique request ID, and
propagates that child's final result. Two bounded waiter jobs accommodate a
cold compile longer than the hosted runner's six-hour per-job limit. The final
waiter fails on failure, cancellation or timeout; an accepted dispatch alone is
never considered delivery success.

## Immutable build identity

`gate` resolves `source_sha`, `recipe_sha`, channel, version, Python tags and
`build_id=run_id-attempt`, writing `build-plan.json`. All subsequent source
checkouts use `source_sha`; actions, wrappers and verification tools come from
`recipe_sha` under `.ci-tools/`. A caller-supplied expected recipe is checked
before building. Source and recipe never inherit each other's moving branch.

Nightlies use `nightly/YYYYMMDD-run_id-attempt` and append
`.devYYYYMMDD.run_id.attempt` to the assembled fork's local version. Each run has
its own identity; same-day builds cannot overwrite another build. Formal tags
must equal `fork/<wheel version>` and point at the recorded source commit.
Published assets are immutable. Repeating promotion of identical assets is
idempotent; different bytes under an existing published tag are rejected.

Use **Re-run all jobs** for a failed wheel run. Partial reruns are rejected at the
start of the build/verification stages, since successful legs and the gate may
otherwise retain the previous attempt's identity. Artifacts include the attempt
in their names and never overwrite an older attempt.

## Build and host validation

The pinned source's lint/type checks and optional varlen host tests run before
CUDA compilation. Recipe mechanism tests also run against the pinned recipe.
The standalone `ci.yml` runs lint and CPU tests on pushes and pull requests;
all logged pipelines use explicit Bash with pipefail.

CUDA builds run on Ubuntu 24.04, with Hopper (`9.0`) code generation, fine autogen
and one compiler worker per hosted runner. `prepare-build-env` installs the
Python, Torch and CUDA environment used by warm/build jobs. Both Python wheels
must pass static metadata/native-content checks and strict installed-package
verification. Importing a stub-only package is a failure: the checker directly
loads `natten.libnatten` and verifies installed file bytes against the wheel.

Each matrix job uploads only its wheel and JSON manifest. A separate verifier
requires exactly one cp310 and one cp311 wheel, matching build identity and
SHA256 values. Logs are uploaded as Actions artifacts and remain available in
Actions job logs; builds no longer push logs to the `ci-logs` source branch.

The warm shards only populate sccache. Their placeholder objects bypass the
cache, and their wheels are never used. A cache miss or failed warm shard causes
the real build to compile normally. The marker is saved only after the complete
matrix passes. Cache state never substitutes for artifact validation.

## Publishing

Only `publish-nightly` or `promote` has write permission for releases. The
publisher verifies the entire asset set before making remote changes, creates a
draft, uploads missing assets, downloads them to verify hashes, and only then
makes the release public. A failed upload leaves a draft for an identical retry;
it cannot expose a partial public release.

Nightlies explicitly state that they passed host checks and have no GPU kernel
validation. After a complete nightly becomes public, the publisher downloads
its assets again and checks their hashes. Only then does retention remove old
nightlies while protecting that release and the newest complete nightly. Formal
fork releases are never part of retention. Source-only deduplication was removed:
updated build recipes need their own artifacts even when source is unchanged.

Formal release candidates require GPU validation of the **original CI wheels**.
Follow [the integration instructions](../integration/README.md) with a clean source
checkout at the manifest's SHA, a matching Python/Torch environment, and the recipe
checkout at `recipe_sha`. Run once per Python wheel. Both proofs must identify
the same source, recipe and build, and the exact wheel hashes. Timeout, failed or
entirely skipped tests cannot produce a passing proof.

Merge the two single-wheel proof objects by keeping their common
`schema_version`, `source_sha`, `recipe_sha`, `build_id` and combining their
nonoverlapping `wheels` mappings. Pass that JSON to `promote.yml` together with
`build_run_id` and `build_attempt`. Promotion downloads the original artifacts
from that successful candidate attempt and uses its original recipe. The
published release includes `GPU-VALIDATION.json`.

## Assembly and rollout

Versioned tools live in [integration/](../integration/README.md). They require
every listed topic, reject ambiguous local/remote branch choices, preserve dirty
worktrees, and record exact base/topic SHAs. `--continue` uses the saved lock,
not newly moved branch tips. These tools never push or publish.

Roll out in this order:

1. Complete the separate Ubuntu 24.04 wheel fix validation.
2. Land the controller, tools and tests on fork-ci; run CPU checks and a full
   two-Python **warm** build before enabling publication.
3. Put the thin `nightly.yml` entry and `promote.yml` entry on main. For the current
   dev branch, remove the old embedded wheel controller and install the new
   `dev.yml` / `release.yml` entries, or rebuild dev from the updated topic set.
   Future assembly includes these entries through the versioned recipe branch.
4. Switch workstation invocations to the versioned integration tools. Do not
   overwrite local working changes while migrating old copies.
5. Exercise nightly publication and formal GPU promotion only after their
   prerequisites pass. Existing published fork assets are preserved.

For a new controller branch under review, dispatch a warm build directly on that
branch with an explicit source SHA. Automatic dispatchers continue to target
fork-ci until rollout, so staged validation cannot alter the active nightly.

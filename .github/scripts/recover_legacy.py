#!/usr/bin/env python3
"""One-run recovery of original #92 nightly artifacts; never deletes assets."""

import argparse
import re
from pathlib import Path

import publish_wheels as publisher
import validate_wheel as validator

RUN = 45926076910
SOURCE = "c570a7501a7d0d757a28277dbd0db578b1bd137f"
WORKFLOW = "526bb811146d326139ac0b6ebf8ec51c3cb40433"
VERSION = "0.21.7+fork.5.dev20261127"
TAG = "nightly/20261127"


def verify_run(github, run_id):
    if run_id != RUN:
        raise ValueError("Recovery is restricted to run 45926076910")
    run = github.api(f"actions/runs/{RUN}")
    if (
        run["id"],
        run["run_attempt"],
        run["head_sha"],
        run["status"],
        run["conclusion"],
    ) != (RUN, 1, WORKFLOW, "completed", "success"):
        raise ValueError("Run identity or success mismatch")
    jobs = []
    page = 1
    while True:
        batch = github.api(
            f"actions/runs/{RUN}/attempts/1/jobs?per_page=100&page={page}"
        )["jobs"]
        jobs.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    builds = [j for j in jobs if re.fullmatch(r"build \(3\.(10|11)\)", j["name"])]
    if {j["name"] for j in builds} != {"build (3.10)", "build (3.11)"} or len(
        builds
    ) != 2:
        raise ValueError("Expected exactly two build jobs")
    for job in builds:
        if (
            job["conclusion"] != "success"
            or job["status"] != "completed"
            or job["head_sha"] != WORKFLOW
            or job["run_attempt"] != 1
        ):
            raise ValueError("Build job identity or success mismatch")
        steps = [
            s for s in job["steps"] if s["name"] == "Install wheel + import smoke test"
        ]
        if (
            len(steps) != 1
            or steps[0]["conclusion"] != "success"
            or steps[0]["status"] != "completed"
        ):
            raise ValueError("Missing passing install/import evidence")


def verify_bundle(directory, repository):
    manifests = list(directory.rglob("MANIFEST-*.txt"))
    if len(manifests) != 2:
        raise ValueError("Expected exactly two original text manifests")
    assets, tags = {}, set()
    for path in manifests:
        if path.is_symlink():
            raise ValueError("Symlinked manifest")
        text = path.read_text()
        if not text.startswith(
            "NATTEN fork CI build manifest\n===============================\n"
        ):
            raise ValueError("Unknown legacy manifest header")
        fields = {}
        for line in text.splitlines():
            match = re.fullmatch(r"([^:]+?)\s+: (.*)", line)
            if match:
                key, value = match.group(1).strip(), match.group(2)
                if key in fields:
                    raise ValueError("Duplicate manifest field: " + key)
                fields[key] = value
        expected = {
            "Built from commit": SOURCE,
            "Workflow commit": WORKFLOW,
            "Workflow run": f"https://github.com/{repository}/actions/runs/{RUN}",
            "natten.__version__": VERSION,
            "Channel": "nightly",
        }
        if any(fields.get(k) != v for k, v in expected.items()):
            raise ValueError("Legacy manifest identity mismatch")
        filename = fields.get("Wheel file", "")
        match = re.fullmatch(
            r"natten-" + re.escape(VERSION) + r"-(cp310|cp311)-\1-[A-Za-z0-9_.]+\.whl",
            filename,
        )
        if (
            not match
            or match[1] in tags
            or path.name != f"MANIFEST-{VERSION}-{match[1]}.txt"
        ):
            raise ValueError("Manifest filename or wheel Python tag mismatch")
        tags.add(match[1])
        wheel = path.parent / filename
        if not wheel.is_file() or wheel.is_symlink():
            raise ValueError("Missing or symlinked wheel")
        if (
            not re.fullmatch("[0-9a-f]{64}", fields.get("SHA256", ""))
            or publisher.digest(wheel) != fields["SHA256"]
        ):
            raise ValueError("Legacy wheel SHA256 mismatch")
        validator.validate_static(wheel, VERSION, match[1])
        assets[path.name], assets[filename] = path, wheel
    if {p.resolve() for p in directory.rglob("*.whl")} != {
        p.resolve() for n, p in assets.items() if n.endswith(".whl")
    }:
        raise ValueError("Unexpected wheel")
    return assets


def publish(github, assets):
    tag_sha = github.tag_sha(TAG)
    if tag_sha is not None and tag_sha != SOURCE:
        raise ValueError("Existing tag targets another commit")
    release = github.release(TAG)

    def identity(value):
        if (
            value["tag_name"] != TAG
            or not value["prerelease"]
            or (tag_sha != SOURCE and value["target_commitish"] != SOURCE)
        ):
            raise ValueError("Existing release identity mismatch")

    if release is not None:
        identity(release)
        if not release["draft"]:
            if tag_sha != SOURCE:
                raise ValueError("Published release requires matching source tag")
            publisher.verify_remote_assets(github, release, assets)
            return release
    else:
        if tag_sha is None:
            github.api(
                "git/refs",
                method="POST",
                data={"ref": f"refs/tags/{TAG}", "sha": SOURCE},
            )
        release = github.api(
            "releases",
            method="POST",
            data={
                "tag_name": TAG,
                "target_commitish": SOURCE,
                "name": TAG,
                "draft": True,
                "prerelease": True,
                "body": f"Original artifacts recovered from workflow run {RUN}, attempt 1.\nSource: `{SOURCE}`\nWorkflow: `{WORKFLOW}`\nVersion: `{VERSION}`\nHost install/import passed; GPU kernel tests were not run.",
            },
        )
    missing = publisher.verify_remote_assets(
        github, release, assets, allow_missing=True
    )
    if missing:
        github.upload(TAG, [assets[name] for name in sorted(missing)])
    release = github.release(TAG)
    identity(release)
    if not release["draft"]:
        raise ValueError("Release changed state during upload")
    publisher.verify_remote_assets(github, release, assets)
    if github.tag_sha(TAG) != SOURCE:
        raise ValueError("Tag changed during upload")
    github.api(f"releases/{release['id']}", method="PATCH", data={"draft": False})
    result = github.release(TAG)
    identity(result)
    if result["draft"] or github.tag_sha(TAG) != SOURCE:
        raise ValueError("Public release identity/state mismatch")
    publisher.verify_remote_assets(github, result, assets)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", type=int, required=True, choices=[RUN])
    args = parser.parse_args()
    github = publisher.GitHub(args.repository)
    assets = verify_bundle(args.artifacts, args.repository)
    verify_run(github, args.run_id)
    result = publish(github, assets)
    print(f"Verified public recovery: {result['html_url']}")


if __name__ == "__main__":
    main()

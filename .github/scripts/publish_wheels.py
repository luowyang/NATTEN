#!/usr/bin/env python3
"""Verify a complete build, then publish its immutable assets through a draft.

No build or GPU test is run here. Formal releases require evidence for these
exact wheels; nightlies explicitly advertise host validation only.
"""

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

PYTHON_TAGS = {"cp310", "cp311"}
IDENTITY_FIELDS = ("source_sha", "recipe_sha", "version", "build_id", "channel")


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def verify_bundle(directory, identity):
    """Validate every leg before any GitHub request or release mutation."""
    for field in ("source_sha", "recipe_sha"):
        if not re.fullmatch(r"[0-9a-f]{40}", identity[field]):
            raise ValueError(f"Invalid {field}")
    if not re.fullmatch(r"[0-9]+-[0-9]+", identity["build_id"]):
        raise ValueError("build_id must be run_id-attempt")
    if identity["channel"] not in ("warm", "nightly", "release"):
        raise ValueError("Invalid channel")
    assets = {}
    wheels = {}
    manifests = list(directory.rglob("MANIFEST-*.json"))
    if len(manifests) != len(PYTHON_TAGS):
        raise ValueError("Expected exactly two wheel manifests")
    for path in manifests:
        manifest = read_json(path)
        if manifest.get("schema_version") != 1:
            raise ValueError(f"Unknown manifest schema: {path}")
        if any(manifest.get(key) != identity[key] for key in IDENTITY_FIELDS):
            raise ValueError(f"Build identity mismatch: {path}")
        tag = manifest.get("python_tag")
        if tag not in PYTHON_TAGS or tag in wheels:
            raise ValueError(f"Duplicate or unexpected Python tag: {tag}")
        if manifest.get("host_validation") != "passed":
            raise ValueError(f"Host validation has not passed: {path}")
        filename = manifest.get("wheel", "")
        pattern = (
            rf"natten-{re.escape(identity['version'])}-{tag}-{tag}-[A-Za-z0-9_.]+\.whl"
        )
        if not re.fullmatch(pattern, filename):
            raise ValueError(f"Wheel filename does not match build: {filename}")
        if path.name != f"MANIFEST-{identity['version']}-{tag}.json":
            raise ValueError(f"Unexpected manifest filename: {path.name}")
        wheel = path.parent / filename
        if not wheel.is_file() or wheel.is_symlink() or path.is_symlink():
            raise ValueError(f"Missing or symlinked wheel/manifest: {path}")
        if digest(wheel) != manifest.get("sha256"):
            raise ValueError(f"Wheel checksum mismatch: {filename}")
        assets[path.name] = path
        assets[filename] = wheel
        wheels[tag] = manifest
    if {p.resolve() for p in directory.rglob("*.whl")} != {
        path.resolve() for name, path in assets.items() if name.endswith(".whl")
    }:
        raise ValueError("Unexpected wheel in artifact directory")
    return assets, wheels


def verify_gpu_attestation(path, identity, wheels):
    verify_gpu_records(read_json(path), identity, wheels)


def verify_gpu_records(evidence, identity, wheels):
    if evidence.get("schema_version") != 1 or any(
        evidence.get(key) != identity[key]
        for key in ("source_sha", "recipe_sha", "build_id")
    ):
        raise ValueError("GPU attestation build identity mismatch")
    records = evidence.get("wheels")
    expected = {item["wheel"]: item["sha256"] for item in wheels.values()}
    if not isinstance(records, dict) or set(records) != set(expected):
        raise ValueError("GPU attestation must cover exactly both wheels")
    for filename, checksum in expected.items():
        record = records[filename]
        if (
            not isinstance(record, dict)
            or record.get("sha256") != checksum
            or record.get("status") != "passed"
            or not isinstance(record.get("evidence"), str)
            or not record["evidence"].strip()
        ):
            raise ValueError(f"Missing passing GPU evidence for {filename}")


def merge_gpu_attestations(paths, identity, wheels):
    merged = {
        "schema_version": 1,
        **{key: identity[key] for key in ("source_sha", "recipe_sha", "build_id")},
        "wheels": {},
    }
    for path in paths:
        evidence = read_json(path)
        if any(evidence.get(key) != merged[key] for key in merged if key != "wheels"):
            raise ValueError("GPU attestation build identity mismatch")
        records = evidence.get("wheels")
        if not isinstance(records, dict) or not records:
            raise ValueError("GPU attestation contains no wheel records")
        if merged["wheels"].keys() & records.keys():
            raise ValueError("Duplicate wheel in GPU attestations")
        merged["wheels"].update(records)
    verify_gpu_records(merged, identity, wheels)
    return merged


def verify_tag(tag, identity):
    if identity["channel"] == "release":
        if tag != f"fork/{identity['version']}":
            raise ValueError("Release tag must match the wheel version")
    elif identity["channel"] == "nightly":
        match = re.fullmatch(r"nightly/(\d{8})-(\d+)-(\d+)", tag)
        if not match:
            raise ValueError("Nightly tag must include UTC date, run ID and attempt")
        day, run, attempt = match.groups()
        datetime.strptime(day, "%Y%m%d")
        if f"{run}-{attempt}" != identity["build_id"] or not identity[
            "version"
        ].endswith(f".dev{day}.{run}.{attempt}"):
            raise ValueError("Nightly tag/version/build_id disagree")
    else:
        raise ValueError("Warm builds cannot be published")


class GitHub:
    def __init__(self, repository):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("Expected owner/repository")
        self.repository = repository

    def command(self, args, *, input_text=None):
        return subprocess.run(
            ["gh", *args],
            input=input_text,
            check=True,
            text=True,
            capture_output=True,
        ).stdout

    def api(self, endpoint, *, method="GET", data=None, missing_ok=False):
        args = ["api", f"repos/{self.repository}/{endpoint}", "--method", method]
        if data is not None:
            args += ["--input", "-"]
        try:
            output = self.command(
                args, input_text=None if data is None else json.dumps(data)
            )
        except subprocess.CalledProcessError as error:
            if missing_ok and "(HTTP 404)" in (error.stderr or ""):
                return None
            raise
        return json.loads(output) if output.strip() else None

    def release(self, tag):
        return self.api(f"releases/tags/{quote(tag, safe='')}", missing_ok=True)

    def tag_sha(self, tag):
        value = self.api(f"git/ref/tags/{quote(tag, safe='')}", missing_ok=True)
        if value is None:
            return None
        obj = value["object"]
        seen = set()
        while obj["type"] == "tag":
            if obj["sha"] in seen:
                raise ValueError("Cyclic annotated tag")
            seen.add(obj["sha"])
            obj = self.api(f"git/tags/{obj['sha']}")["object"]
        if obj["type"] != "commit":
            raise ValueError("Release tag does not point at a commit")
        return obj["sha"]

    def releases(self):
        pages = json.loads(
            self.command(
                [
                    "api",
                    f"repos/{self.repository}/releases?per_page=100",
                    "--paginate",
                    "--slurp",
                ]
            )
        )
        return [release for page in pages for release in page]

    def download(self, tag, name, directory):
        self.command(
            [
                "release",
                "download",
                tag,
                "--repo",
                self.repository,
                "--pattern",
                name,
                "--dir",
                str(directory),
            ]
        )

    def upload(self, tag, paths):
        self.command(
            ["release", "upload", tag, *map(str, paths), "--repo", self.repository]
        )


def verify_remote_assets(github, release, assets, *, allow_missing=False):
    remote = {item["name"] for item in release["assets"]}
    if len(remote) != len(release["assets"]) or remote - assets.keys():
        raise ValueError("Unexpected or duplicate remote release assets")
    missing = assets.keys() - remote
    if missing and not allow_missing:
        raise ValueError(f"Release is incomplete: {sorted(missing)}")
    with tempfile.TemporaryDirectory(prefix="natten-release-verify-") as temp:
        directory = Path(temp)
        for name in sorted(remote):
            github.download(release["tag_name"], name, directory)
            if digest(directory / name) != digest(assets[name]):
                raise ValueError(f"Existing release asset differs: {name}")
    return missing


def publish(github, assets, identity, tag):
    """Only a verified complete draft can transition to a public release."""
    verify_tag(tag, identity)
    source_sha = identity["source_sha"]
    tag_sha = github.tag_sha(tag)
    if tag_sha is not None and tag_sha != source_sha:
        raise ValueError("Remote tag points at a different source commit")
    if identity["channel"] == "release" and tag_sha is None:
        raise ValueError("Formal release tag must already exist")
    release = github.release(tag)
    if release is not None and release["prerelease"] != (
        identity["channel"] == "nightly"
    ):
        raise ValueError("Existing release channel does not match the build")
    if release is not None and not release["draft"]:
        if tag_sha != source_sha:
            raise ValueError("Published release has no matching source tag")
        verify_remote_assets(github, release, assets)
        return release
    if release is None:
        # Reserve the immutable source tag before publishing the draft.
        if tag_sha is None:
            github.api(
                "git/refs",
                method="POST",
                data={"ref": f"refs/tags/{tag}", "sha": source_sha},
            )
        validation = (
            "Host and GPU validation passed for the attached wheel checksums."
            if identity["channel"] == "release"
            else "Host validation passed. GPU kernel tests were not run for this nightly."
        )
        release = github.api(
            "releases",
            method="POST",
            data={
                "tag_name": tag,
                "target_commitish": source_sha,
                "name": tag,
                "body": (
                    f"{validation}\n\nSource: `{source_sha}`\n"
                    f"Build recipe: `{identity['recipe_sha']}`\n"
                    f"Build: `{identity['build_id']}`"
                ),
                "draft": True,
                "prerelease": identity["channel"] == "nightly",
            },
        )
    elif release["target_commitish"] != source_sha and tag_sha != source_sha:
        raise ValueError("Draft does not target the expected source commit")
    missing = verify_remote_assets(github, release, assets, allow_missing=True)
    if missing:
        github.upload(tag, [assets[name] for name in sorted(missing)])
    release = github.release(tag)
    if not release["draft"]:
        raise ValueError("Release changed state while assets were uploaded")
    verify_remote_assets(github, release, assets)
    if github.tag_sha(tag) != source_sha:
        raise ValueError("Release tag changed during upload")
    github.api(f"releases/{release['id']}", method="PATCH", data={"draft": False})
    result = github.release(tag)
    if result["draft"]:
        raise ValueError("Release did not become public")
    return result


def prune_nightlies(github, published, retention_days, *, today=None):
    """Run only after publication succeeds; never remove the verified release."""
    if published["draft"] or not published["prerelease"]:
        raise ValueError("Retention requires a published nightly")
    if retention_days < 1:
        raise ValueError("Retention must be positive")
    today = today or datetime.now(timezone.utc).date()
    cutoff = today - timedelta(days=retention_days)
    releases = github.releases()
    # A late rerun of an old build must also preserve the newest complete
    # nightly. Legacy releases use text manifests; completeness here is an
    # inventory check, while the just-published release was verified bytewise.
    complete = []
    for release in releases:
        if release["draft"] or not release["prerelease"]:
            continue
        if not re.fullmatch(r"nightly/\d{8}(?:-\d+-\d+)?", release["tag_name"]):
            continue
        names = {asset["name"] for asset in release["assets"]}
        versions = {}
        for name in names:
            match = re.fullmatch(
                r"natten-(.+)-(cp310|cp311)-\2-[A-Za-z0-9_.]+\.whl", name
            )
            if match:
                version, python_tag = match.groups()
                if any(
                    f"MANIFEST-{version}-{python_tag}.{ext}" in names
                    for ext in ("json", "txt")
                ):
                    versions.setdefault(version, set()).add(python_tag)
        if PYTHON_TAGS in versions.values():
            complete.append(release)
    protected = {published["id"]}
    if complete:
        newest = max(
            complete,
            key=lambda item: tuple(
                int(part) for part in item["tag_name"].split("/")[1].split("-")
            ),
        )
        protected.add(newest["id"])
    for release in releases:
        tag = release["tag_name"]
        if release["id"] in protected or release["draft"] or not release["prerelease"]:
            continue
        match = re.fullmatch(r"nightly/(\d{8})(?:-\d+-\d+)?", tag)
        if match is None:
            continue
        day = datetime.strptime(match.group(1), "%Y%m%d").date()
        if day < cutoff:
            github.api(f"releases/{release['id']}", method="DELETE")
            # Delete only the ref belonging to this expired nightly.
            github.api(f"git/refs/tags/{quote(tag, safe='')}", method="DELETE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--repository", default="luowyang/NATTEN")
    parser.add_argument(
        "--channel", choices=("warm", "nightly", "release"), required=True
    )
    parser.add_argument("--tag")
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--recipe-sha", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--gpu-attestation", type=Path, action="append")
    parser.add_argument("--retention-days", type=int, default=14)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.retention_days < 1:
        parser.error("--retention-days must be positive")
    identity = {key: getattr(args, key) for key in IDENTITY_FIELDS}
    assets, wheels = verify_bundle(args.artifacts, identity)
    if args.verify_only:
        print("Both Python wheels and manifests passed host asset validation.")
        return
    if not args.tag:
        parser.error("--tag is required for publication")
    if args.channel == "release":
        if args.gpu_attestation is None:
            parser.error("Formal releases require --gpu-attestation")
        gpu_evidence = merge_gpu_attestations(args.gpu_attestation, identity, wheels)
    elif args.gpu_attestation is not None:
        parser.error("GPU promotion evidence is only accepted for formal releases")
    with tempfile.TemporaryDirectory(prefix="natten-release-evidence-") as temp:
        if args.gpu_attestation is not None:
            evidence = Path(temp) / "GPU-VALIDATION.json"
            evidence.write_text(
                json.dumps(gpu_evidence, indent=2, sort_keys=True) + "\n"
            )
            assets[evidence.name] = evidence
        github = GitHub(args.repository)
        release = publish(github, assets, identity, args.tag)
        if args.channel == "nightly":
            prune_nightlies(github, release, args.retention_days)
        print(release["html_url"])


if __name__ == "__main__":
    main()

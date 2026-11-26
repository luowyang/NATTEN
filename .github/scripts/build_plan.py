#!/usr/bin/env python3
"""Freeze source, recipe and package identity before any matrix jobs start."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

VERSION_FILE = Path("src/natten/version.py")


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def source_version(path=VERSION_FILE):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            version = ast.literal_eval(node.value)
            if not isinstance(version, str) or not re.fullmatch(
                r"\d+\.\d+\.\d+(?:\+fork\.\d+)?", version
            ):
                raise ValueError(f"Unsupported source version: {version!r}")
            return version
    raise ValueError("Source does not declare __version__")


def resolve(channel, source_ref, source_sha, recipe_sha, version, build_id, date):
    for value in (source_sha, recipe_sha):
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("Source and recipe must resolve to commit SHAs")
    if not re.fullmatch(r"[0-9]+-[0-9]+", build_id):
        raise ValueError("build_id must be run_id-attempt")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/+-]*", source_ref):
        raise ValueError("Invalid source ref")
    datetime.strptime(date, "%Y%m%d")
    if channel == "auto":
        channel = "release" if source_ref.startswith("fork/") else "warm"
    tag = ""
    if channel == "release":
        if source_ref != f"fork/{version}":
            raise ValueError("Release ref must equal fork/<source version>")
        tag = source_ref
    elif channel == "nightly":
        if "+fork." not in version:
            raise ValueError("Nightly source must be an assembled fork version")
        version += f".dev{date}.{build_id.replace('-', '.')}"
        tag = f"nightly/{date}-{build_id}"
    elif channel != "warm":
        raise ValueError(f"Unknown channel: {channel}")
    return {
        "schema_version": 1,
        "source_sha": source_sha,
        "recipe_sha": recipe_sha,
        "source_ref": source_ref,
        "version": version,
        "build_id": build_id,
        "channel": channel,
        "tag": tag,
        "python_tags": ["cp310", "cp311"],
    }


def assert_checkouts(plan, recipe):
    if git("rev-parse", "HEAD") != plan["source_sha"]:
        raise ValueError("Source checkout differs from the frozen plan")
    if git("-C", str(recipe), "rev-parse", "HEAD") != plan["recipe_sha"]:
        raise ValueError("Recipe checkout differs from the frozen plan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--source-ref", required=True)
    create.add_argument("--channel", required=True)
    create.add_argument("--recipe-sha", required=True)
    create.add_argument("--build-id", required=True)
    create.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    create.add_argument("--output", type=Path, required=True)
    stamp = sub.add_parser("stamp")
    stamp.add_argument("--plan", type=Path, required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--plan", type=Path, required=True)
    manifest.add_argument("--wheel", type=Path, required=True)
    manifest.add_argument("--python-tag", required=True)
    for cmd in (create, stamp, manifest):
        cmd.add_argument("--recipe", type=Path, default=Path(".ci-tools"))
    args = parser.parse_args()
    if args.command == "create":
        expected = os.environ.get("EXPECTED_RECIPE_SHA", "")
        if expected and expected != args.recipe_sha:
            raise ValueError(
                "Dispatched recipe differs from the caller's frozen recipe"
            )
        namespace = os.environ.get("CACHE_NAMESPACE", "")
        if namespace and not re.fullmatch(r"[A-Za-z0-9._-]+", namespace):
            raise ValueError("Invalid cache namespace")
        plan = resolve(
            args.channel,
            args.source_ref,
            git("rev-parse", "HEAD"),
            args.recipe_sha,
            source_version(),
            args.build_id,
            args.date,
        )
        assert_checkouts(plan, args.recipe)
        if plan["channel"] == "release":
            # A similarly named branch must not impersonate the release tag.
            git("fetch", "--no-tags", "origin", f"refs/tags/{plan['tag']}")
            if git("rev-parse", "FETCH_HEAD^{commit}") != plan["source_sha"]:
                raise ValueError("Release tag does not point at the source checkout")
        inputs = {
            "recipe": plan["recipe_sha"],
            "csrc": git("rev-parse", "HEAD:csrc"),
            "cutlass": git("rev-parse", "HEAD:third_party/cutlass"),
            "setup": git("rev-parse", "HEAD:setup.py"),
            "scripts": git("rev-parse", "HEAD:scripts"),
            "arch": os.environ["NATTEN_CUDA_ARCH"],
            "autogen": os.environ["NATTEN_AUTOGEN_POLICY"],
            "namespace": os.environ.get("CACHE_NAMESPACE", ""),
        }
        plan["cache_key"] = (
            "natten-warm-"
            + hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        )
        args.output.write_text(json.dumps(plan, indent=2) + "\n")
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            for key in (
                "source_sha",
                "recipe_sha",
                "version",
                "channel",
                "tag",
                "build_id",
                "cache_key",
            ):
                print(f"{key}={plan[key]}", file=output)
        print(json.dumps(plan, indent=2))
        return
    plan = json.loads(args.plan.read_text())
    assert_checkouts(plan, args.recipe)
    if args.command == "stamp":
        if plan["channel"] == "nightly":
            text, count = re.subn(
                r"^__version__\s*=.*$",
                f'__version__ = "{plan["version"]}"',
                VERSION_FILE.read_text(),
                flags=re.MULTILINE,
            )
            if count != 1:
                raise ValueError("Expected exactly one version assignment")
            VERSION_FILE.write_text(text)
        elif source_version() != plan["version"]:
            raise ValueError("Source version differs from the frozen plan")
        return
    if args.python_tag not in plan["python_tags"]:
        raise ValueError("Unexpected Python leg")
    record = {
        key: plan[key]
        for key in (
            "schema_version",
            "source_sha",
            "recipe_sha",
            "version",
            "build_id",
            "channel",
        )
    }
    record.update(
        python_tag=args.python_tag,
        wheel=args.wheel.name,
        sha256=hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        host_validation="passed",
        toolchain={
            "torch": importlib.metadata.version("torch"),
            "nvcc": subprocess.check_output(["nvcc", "--version"], text=True).strip(),
            "python": subprocess.check_output(
                ["python", "--version"], text=True
            ).strip(),
            "runner": "ubuntu-24.04",
            "arch": os.environ["NATTEN_CUDA_ARCH"],
            "autogen": os.environ["NATTEN_AUTOGEN_POLICY"],
        },
    )
    path = args.wheel.parent / f'MANIFEST-{plan["version"]}-{args.python_tag}.json'
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(path.read_text())


if __name__ == "__main__":
    main()

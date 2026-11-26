#!/usr/bin/env python3
"""Fetch a successful candidate and its immutable plan for GPU promotion."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def gh(*args):
    return subprocess.check_output(["gh", *args], text=True, timeout=300)


def check_run(run, plan, run_id, attempt):
    if run["id"] != run_id or run["run_attempt"] != attempt:
        raise ValueError("Run identity differs from the requested build attempt")
    if run["status"] != "completed" or run["conclusion"] != "success":
        raise ValueError("Only successful complete candidate builds can be promoted")
    if (
        run["path"] != ".github/workflows/wheel.yml"
        or run["event"] != "workflow_dispatch"
    ):
        raise ValueError("Artifacts must come from the wheel controller")
    if plan["build_id"] != f"{run_id}-{attempt}" or plan["channel"] != "release":
        raise ValueError("Plan does not identify this release candidate")
    if plan["recipe_sha"] != run["head_sha"]:
        raise ValueError("Recipe SHA does not match the producing workflow")
    for key in ("source_sha", "recipe_sha"):
        if not re.fullmatch(r"[0-9a-f]{40}", plan[key]):
            raise ValueError("Invalid commit identity in candidate plan")
    if plan["tag"] != f'fork/{plan["version"]}':
        raise ValueError("Candidate tag and version disagree")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    if args.run_id <= 0 or args.attempt <= 0:
        raise ValueError("Run ID and attempt must be positive")
    identity = f"{args.run_id}-{args.attempt}"
    run = json.loads(
        gh(
            "api",
            f"repos/{args.repository}/actions/runs/{args.run_id}/attempts/{args.attempt}",
        )
    )
    gh(
        "run",
        "download",
        str(args.run_id),
        "--repo",
        args.repository,
        "--name",
        f"natten-plan-{identity}",
        "--dir",
        "plan",
    )
    plan = json.loads(Path("plan/build-plan.json").read_text())
    check_run(run, plan, args.run_id, args.attempt)
    # Parse without executing interpolation from workflow inputs.
    attestation = json.loads(os.environ["GPU_ATTESTATION"])
    Path("gpu-attestation.json").write_text(json.dumps(attestation, indent=2) + "\n")
    for tag in ("cp310", "cp311"):
        gh(
            "run",
            "download",
            str(args.run_id),
            "--repo",
            args.repository,
            "--name",
            f"natten-wheel-{identity}-{tag}",
            "--dir",
            "artifacts",
        )
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        print(f'recipe_sha={plan["recipe_sha"]}', file=output)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        raise SystemExit(error.returncode) from error

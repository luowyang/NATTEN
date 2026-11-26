#!/usr/bin/env python3
"""Dispatch a uniquely identified wheel run and propagate its final result."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlencode


def api(endpoint, payload=None):
    args = ["gh", "api", endpoint]
    data = None
    if payload is not None:
        args += ["--method", "POST", "--input", "-"]
        data = json.dumps(payload)
    result = subprocess.run(
        args, input=data, text=True, capture_output=True, check=True, timeout=60
    )
    return json.loads(result.stdout) if result.stdout.strip() else None


def matching_run(runs, request_id, recipe_sha):
    matches = [run for run in runs if run["display_title"] == f"wheel {request_id}"]
    if len(matches) > 1:
        raise ValueError("Multiple workflow runs claim the same dispatch request")
    if not matches:
        return None
    run = matches[0]
    if run["head_sha"] != recipe_sha or run["event"] != "workflow_dispatch":
        raise ValueError("Dispatched workflow recipe/event differs from the request")
    return run


def dispatch(
    repository,
    source_ref,
    channel,
    request_id,
    recipe_sha,
    timeout=480,
    poll_seconds=10,
):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", request_id):
        raise ValueError("Invalid request ID")
    if not re.fullmatch(r"[0-9a-f]{40}", recipe_sha):
        raise ValueError("Expected frozen recipe SHA")
    branch = api(f"repos/{repository}/git/ref/heads/fork-ci")
    if branch["object"]["sha"] != recipe_sha:
        raise ValueError("fork-ci moved before dispatch; retry with the new recipe")
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    api(
        f"repos/{repository}/actions/workflows/wheel.yml/dispatches",
        {
            "ref": "fork-ci",
            "inputs": {
                "ref": source_ref,
                "channel": channel,
                "request_id": request_id,
                "expected_recipe_sha": recipe_sha,
            },
        },
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        runs = []
        page = 1
        while True:
            query = urlencode(
                {
                    "event": "workflow_dispatch",
                    "branch": "fork-ci",
                    "created": ">=" + created,
                    "per_page": 100,
                    "page": page,
                }
            )
            batch = api(f"repos/{repository}/actions/workflows/wheel.yml/runs?{query}")[
                "workflow_runs"
            ]
            runs.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        run = matching_run(runs, request_id, recipe_sha)
        if run is not None:
            print(f'Dispatched {run["html_url"]}', flush=True)
            return run["id"]
        time.sleep(poll_seconds)
    raise TimeoutError(f"No wheel run appeared for request {request_id}")


def await_run(repository, run_id, recipe_sha, *, final, timeout=20700, poll_seconds=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = api(f"repos/{repository}/actions/runs/{run_id}")
        if run["head_sha"] != recipe_sha:
            raise ValueError("Waiting on a different recipe")
        if run["status"] == "completed":
            if run["conclusion"] != "success":
                raise RuntimeError(f'Wheel run {run["html_url"]}: {run["conclusion"]}')
            print(f'Wheel run succeeded: {run["html_url"]}', flush=True)
            return True
        time.sleep(poll_seconds)
    if final:
        raise TimeoutError(
            f"Wheel run {run_id} did not complete within the waiting budget"
        )
    print(f"Run {run_id} still pending; handing off to the final waiter", flush=True)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("dispatch", "wait"))
    parser.add_argument("--repository", required=True)
    parser.add_argument("--recipe-sha", required=True)
    parser.add_argument("--source-ref")
    parser.add_argument("--channel", choices=("warm", "nightly", "release"))
    parser.add_argument("--request-id")
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    if args.command == "dispatch":
        if not all((args.source_ref, args.channel, args.request_id)):
            parser.error("dispatch needs --source-ref, --channel and --request-id")
        run_id = dispatch(
            args.repository,
            args.source_ref,
            args.channel,
            args.request_id,
            args.recipe_sha,
        )
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            print(f"run_id={run_id}", file=output)
            print(f"recipe_sha={args.recipe_sha}", file=output)
    else:
        if not args.run_id or args.run_id <= 0:
            parser.error("wait needs a positive --run-id")
        completed = await_run(
            args.repository, args.run_id, args.recipe_sha, final=args.final
        )
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            print(f"completed={str(completed).lower()}", file=output)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        raise SystemExit(error.returncode) from error

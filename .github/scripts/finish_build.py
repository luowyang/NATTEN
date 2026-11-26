#!/usr/bin/env python3
"""Pass the frozen build identity to the matrix verifier or publisher."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "publish"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--repository", default="luowyang/NATTEN")
    parser.add_argument("--gpu-attestation", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    command = [
        sys.executable,
        str(Path(__file__).with_name("publish_wheels.py")),
        "--artifacts",
        str(args.artifacts),
        "--repository",
        args.repository,
    ]
    for key in ("channel", "tag", "source_sha", "recipe_sha", "version", "build_id"):
        command += ["--" + key.replace("_", "-"), plan[key]]
    if args.command == "verify":
        command.append("--verify-only")
    if args.gpu_attestation:
        command += ["--gpu-attestation", str(args.gpu_attestation)]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        raise SystemExit(error.returncode) from error

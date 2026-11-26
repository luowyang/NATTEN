"""Validate an installed wheel against its build manifest and pinned source tests."""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from subprocess import run


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def check_source(source, expected_sha):
    if git(source, "rev-parse", "HEAD") != expected_sha:
        raise ValueError("Source checkout HEAD differs from manifest source_sha")
    if git(
        source,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--ignore-submodules=none",
    ):
        raise ValueError("Source checkout has uncommitted changes")


def check_artifact(wheel, manifest):
    required = (
        "source_sha",
        "recipe_sha",
        "build_id",
        "version",
        "python_tag",
        "wheel",
        "sha256",
    )
    if manifest.get("schema_version") != 1 or any(
        not manifest.get(k) for k in required
    ):
        raise ValueError("Incomplete build manifest")
    for key in ("source_sha", "recipe_sha"):
        if not re.fullmatch(r"[0-9a-f]{40}", manifest[key]):
            raise ValueError(f"Invalid {key}")
    if manifest["wheel"] != wheel.name:
        raise ValueError("Wheel filename differs from manifest")
    if wheel.name.split("-")[-3] != manifest["python_tag"]:
        raise ValueError("Wheel Python tag differs from manifest")
    hasher = hashlib.sha256()
    with wheel.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if manifest["sha256"] != digest:
        raise ValueError("Wheel SHA256 differs from manifest")
    if manifest["python_tag"] != f"cp{sys.version_info.major}{sys.version_info.minor}":
        raise ValueError("Use the Python version that matches this wheel")
    return digest


def check_installed(wheel, manifest):
    script = Path(__file__).resolve().parents[1] / ".github/scripts/validate_wheel.py"
    spec = importlib.util.spec_from_file_location("fork_installed_wheel_check", script)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    return checker.validate_installed(
        wheel, manifest["version"], manifest["python_tag"]
    )


def check_native_gpu(wheel, manifest):
    check_installed(wheel, manifest)
    import natten
    import torch

    if not torch.cuda.is_available():
        raise ValueError("CUDA device is required for GPU validation")
    # Require an actual native kernel, even when the selected pytest classes skip.
    q, k, v = [
        torch.randn(
            1, 16, 2, 32, device="cuda", dtype=torch.float16, requires_grad=True
        )
        for _ in range(3)
    ]
    out = natten.na1d(q, k, v, kernel_size=3, backend="cutlass-fna")
    out.float().square().mean().backward()
    torch.cuda.synchronize()
    if out.shape != q.shape or not torch.isfinite(out).all().item():
        raise ValueError("Native GPU forward smoke failed")
    if any(
        t.grad is None or not torch.isfinite(t.grad).all().item() for t in (q, k, v)
    ):
        raise ValueError("Native GPU backward smoke failed")


def check_junit(path):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    if not cases or not any(case.find("skipped") is None for case in cases):
        raise ValueError("GPU test selection ran no tests (empty or all skipped)")
    if any(
        case.find("failure") is not None or case.find("error") is not None
        for case in cases
    ):
        raise ValueError("JUnit report contains failed tests")


def validate(args):
    wheel = args.wheel.resolve()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Refusing to overwrite an existing validation result")
    # Reports must not make the pinned source checkout dirty while it is tested.
    if output == source or source in output.parents:
        raise ValueError("Write validation evidence outside the source checkout")
    manifest = json.loads(args.manifest.read_text())
    digest = check_artifact(wheel, manifest)
    check_source(source, manifest["source_sha"])
    if args.preflight:
        return 0
    check_native_gpu(wheel, manifest)
    tests = args.tests or sorted(
        {*source.glob("tests/test_varlen*.py"), *source.glob("tests/test_*_varlen.py")}
    )
    if not tests:
        raise ValueError("No varlen tests selected")
    tests = [p.resolve() if p.is_absolute() else (source / p).resolve() for p in tests]
    if any(not p.is_file() or source not in p.parents for p in tests):
        raise ValueError(
            "Every selected test must exist inside the pinned source checkout"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    junit = output.with_suffix(".xml")
    log = output.with_suffix(".log")
    if junit.exists() or log.exists():
        raise ValueError("Use a new output path; validation evidence already exists")
    env = dict(
        os.environ,
        NATTEN_LOG_LEVEL="CRITICAL",
        PYTORCH_NO_CUDA_MEMORY_CACHING="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    try:
        with log.open("x") as stream:
            result = run(
                [
                    sys.executable,
                    "-I",
                    str(Path(__file__).resolve()),
                    "--pytest-worker",
                    "--wheel",
                    str(wheel),
                    "--manifest",
                    str(args.manifest.resolve()),
                    "--source",
                    str(source),
                    "--output",
                    str(output),
                    f"--junitxml={junit}",
                    "--tests",
                    *map(str, tests),
                ],
                cwd=source,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
    except subprocess.TimeoutExpired:
        print(f"GPU validation timed out; see {log}", file=sys.stderr)
        return 124
    if result.returncode:
        return result.returncode
    check_junit(junit)
    check_source(source, manifest["source_sha"])
    if check_artifact(wheel, manifest) != digest:
        raise ValueError("Wheel changed during validation")
    check_installed(wheel, manifest)
    proof = {
        "schema_version": 1,
        **{k: manifest[k] for k in ("source_sha", "recipe_sha", "build_id")},
        "wheels": {
            wheel.name: {"sha256": digest, "status": "passed", "evidence": str(log)}
        },
    }
    with output.open("x") as stream:
        json.dump(proof, stream, indent=2)
        stream.write("\n")
    print(f"GPU validation passed for {wheel.name}: {output}")
    return 0


class InstalledWheelGuard:
    """Check inside pytest, after initial and nested conftest imports, and at exit."""

    def __init__(self, wheel, manifest):
        self.wheel = wheel
        self.manifest = manifest

    def check(self):
        check_artifact(self.wheel, self.manifest)
        check_installed(self.wheel, self.manifest)

    def pytest_sessionstart(self, session):
        self.check()

    def pytest_collection_finish(self, session):
        self.check()

    def pytest_sessionfinish(self, session, exitstatus):
        self.check()


def run_pytest_worker(args):
    import pytest

    manifest = json.loads(args.manifest.read_text())
    check_source(args.source, manifest["source_sha"])
    return pytest.main(
        [
            "-x",
            "-v",
            "--import-mode=importlib",
            f"--junitxml={args.junitxml}",
            *map(str, args.tests),
        ],
        plugins=[InstalledWheelGuard(args.wheel, manifest)],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--pytest-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--junitxml", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--tests", type=Path, nargs="*")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.pytest_worker:
        if args.junitxml is None or not args.tests:
            parser.error("pytest worker requires JUnit path and selected tests")
        return run_pytest_worker(args)
    return validate(args)


if __name__ == "__main__":
    sys.exit(main())

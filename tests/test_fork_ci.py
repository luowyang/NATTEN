"""Failure propagation tests for the fork's CPU CI steps."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class ForkCiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
        cls.job = cls.workflow["jobs"]["lint-and-host-tests"]
        cls.steps = {step["id"]: step for step in cls.job["steps"] if "id" in step}

    def shell_for(self, step):
        shell = step.get(
            "shell",
            self.job.get("defaults", {})
            .get("run", {})
            .get(
                "shell", self.workflow.get("defaults", {}).get("run", {}).get("shell")
            ),
        )
        # GitHub's unspecified Linux shell omits pipefail. Exercise the actual
        # selected shell, so removing defaults.run.shell breaks these probes.
        if shell == "bash":
            return ["bash", "--noprofile", "--norc", "-eo", "pipefail"]
        if shell is None:
            return ["bash", "-e"]
        self.fail(f"Unsupported CI shell in the failure probe: {shell}")

    def test_tool_failures_survive_logging_pipelines(self):
        tools = {
            "ufmt": "ufmt",
            "flake8": "flake8",
            "mypy": "mypy",
            "host_tests": "pytest",
            "fork_tests": "python",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step_id, command in tools.items():
                with self.subTest(step=step_id):
                    tool = root / command
                    tool.write_text('#!/bin/sh\necho "injected failure" >&2\nexit 7\n')
                    tool.chmod(0o755)
                    script = root / "step.sh"
                    step = self.steps[step_id]
                    script.write_text(step["run"])
                    result = subprocess.run(
                        [*self.shell_for(step), str(script)],
                        cwd=root,
                        env={
                            **os.environ,
                            "PATH": str(root) + ":" + os.environ["PATH"],
                        },
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(
                        result.returncode, 7, result.stdout + result.stderr
                    )
                    self.assertIn("injected failure", result.stdout)

    def test_absent_optional_suite_is_reported_and_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, summary = root / "output", root / "summary"
            script = root / "step.sh"
            step = self.steps["host_suite"]
            script.write_text(step["run"])
            result = subprocess.run(
                [*self.shell_for(step), str(script)],
                cwd=root,
                env={
                    **os.environ,
                    "GITHUB_OUTPUT": str(output),
                    "GITHUB_STEP_SUMMARY": str(summary),
                },
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("present=false", output.read_text())
            self.assertIn("skipped", summary.read_text())
            self.assertEqual(
                self.steps["host_tests"]["if"],
                "steps.host_suite.outputs.present == 'true'",
            )


if __name__ == "__main__":
    unittest.main()

"""Exercise build-plan freezing and workflow shell boundaries without GitHub."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
PLAN_SCRIPT = ROOT / ".github/scripts/build_plan.py"
SPEC = importlib.util.spec_from_file_location("fork_build_plan", PLAN_SCRIPT)
plan_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan_module)


class BuildPlanFreezingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.recipe = self.source / ".ci-tools"
        self.plan = self.root / "plan.json"
        self.env = {
            **os.environ,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_AUTHOR_NAME": "Fixture",
            "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
            "GIT_COMMITTER_NAME": "Fixture",
            "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
            "GITHUB_OUTPUT": str(self.root / "outputs"),
            "NATTEN_CUDA_ARCH": "9.0",
            "NATTEN_AUTOGEN_POLICY": "fine",
        }
        self.source.mkdir()
        self.git(self.source, "init", "-b", "dev")
        for member, text in {
            "src/natten/version.py": '__version__ = "0.21.7+fork.8"\n',
            "csrc/fixture.cu": "// first source\n",
            "scripts/autogen_fixture.py": "# fixture\n",
            "setup.py": "# setup fixture\n",
            "third_party/cutlass/fixture.h": "// fixture\n",
        }.items():
            path = self.source / member
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.git(self.source, "add", ".")
        self.git(self.source, "commit", "-m", "source fixture")
        self.source_sha = self.git(self.source, "rev-parse", "HEAD")
        self.recipe.mkdir()
        self.git(self.recipe, "init", "-b", "fork-ci")
        (self.recipe / "recipe.txt").write_text("first recipe\n")
        self.git(self.recipe, "add", ".")
        self.git(self.recipe, "commit", "-m", "recipe fixture")
        self.recipe_sha = self.git(self.recipe, "rev-parse", "HEAD")

    def git(self, directory, *args):
        return subprocess.check_output(
            ["git", "-C", str(directory), *args],
            env=self.env,
            stderr=subprocess.PIPE,
            text=True,
        ).strip()

    def command(self, *args):
        return subprocess.run(
            [sys.executable, str(PLAN_SCRIPT), *args],
            cwd=self.source,
            env=self.env,
            capture_output=True,
            text=True,
        )

    def create(self):
        result = self.command(
            "create",
            "--source-ref",
            "dev",
            "--channel",
            "nightly",
            "--recipe-sha",
            self.recipe_sha,
            "--build-id",
            "100-1",
            "--date",
            "20261126",
            "--output",
            str(self.plan),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(self.plan.read_text())

    def test_moving_source_ref_does_not_change_frozen_build(self):
        plan = self.create()
        (self.source / "csrc/fixture.cu").write_text("// later source\n")
        self.git(self.source, "add", "csrc/fixture.cu")
        self.git(self.source, "commit", "-m", "move dev")
        self.assertNotEqual(self.git(self.source, "rev-parse", "dev"), self.source_sha)
        self.git(self.source, "checkout", "--detach", plan["source_sha"])
        result = self.command("stamp", "--plan", str(self.plan))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            (self.source / "csrc/fixture.cu").read_text(), "// first source\n"
        )
        self.assertIn(
            plan["version"], (self.source / "src/natten/version.py").read_text()
        )

    def test_wrong_source_checkout_stops_before_stamping(self):
        self.create()
        version = self.source / "src/natten/version.py"
        original = version.read_text()
        (self.source / "csrc/fixture.cu").write_text("// later source\n")
        self.git(self.source, "add", "csrc/fixture.cu")
        self.git(self.source, "commit", "-m", "move dev")
        result = self.command("stamp", "--plan", str(self.plan))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Source checkout differs", result.stderr)
        self.assertEqual(version.read_text(), original)

    def test_wrong_recipe_checkout_stops_before_stamping(self):
        self.create()
        version = self.source / "src/natten/version.py"
        original = version.read_text()
        (self.recipe / "recipe.txt").write_text("changed recipe\n")
        self.git(self.recipe, "add", "recipe.txt")
        self.git(self.recipe, "commit", "-m", "move recipe")
        result = self.command("stamp", "--plan", str(self.plan))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Recipe checkout differs", result.stderr)
        self.assertEqual(version.read_text(), original)

    def test_separate_attempts_get_distinct_nightly_identities(self):
        first = plan_module.resolve(
            "nightly",
            "dev",
            self.source_sha,
            self.recipe_sha,
            "0.21.7+fork.8",
            "100-1",
            "20261126",
        )
        retry = plan_module.resolve(
            "nightly",
            "dev",
            self.source_sha,
            self.recipe_sha,
            "0.21.7+fork.8",
            "100-2",
            "20261126",
        )
        self.assertNotEqual(first["version"], retry["version"])
        self.assertNotEqual(first["tag"], retry["tag"])

    def test_source_ref_shell_payload_is_rejected_without_execution(self):
        sentinel = self.root / "executed"
        result = self.command(
            "create",
            "--source-ref",
            f"dev$(touch {sentinel})",
            "--channel",
            "nightly",
            "--recipe-sha",
            self.recipe_sha,
            "--build-id",
            "100-1",
            "--output",
            str(self.plan),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(sentinel.exists())
        self.assertFalse(self.plan.exists())


class CompositeInputTests(unittest.TestCase):
    def test_cache_namespace_cannot_execute_shell(self):
        action = yaml.safe_load(
            (ROOT / ".github/actions/prepare-build-env/action.yml").read_text()
        )
        step = next(
            step
            for step in action["runs"]["steps"]
            if step.get("name") == "Set sccache cache namespace"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root / "executed"
            payload = f"$(touch {sentinel})"
            script = root / "step.sh"
            script.write_text(
                step["run"].replace("${{ inputs.cache_namespace }}", payload)
            )
            env = {**os.environ, "GITHUB_ENV": str(root / "env")}
            env.update(
                {
                    key: value.replace("${{ inputs.cache_namespace }}", payload)
                    for key, value in step.get("env", {}).items()
                }
            )
            subprocess.run(
                ["bash", "--noprofile", "--norc", "-eo", "pipefail", str(script)],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertFalse(sentinel.exists(), "namespace was expanded as shell code")


class PromotionIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "fork_prepare_promotion", ROOT / ".github/scripts/prepare_promotion.py"
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def setUp(self):
        self.run = {
            "id": 100,
            "run_attempt": 2,
            "status": "completed",
            "conclusion": "success",
            "path": ".github/workflows/wheel.yml",
            "event": "workflow_dispatch",
            "head_sha": "a" * 40,
        }
        self.plan = {
            "build_id": "100-2",
            "channel": "release",
            "recipe_sha": "a" * 40,
            "source_sha": "b" * 40,
            "version": "0.21.7+fork.8",
            "tag": "fork/0.21.7+fork.8",
        }

    def test_successful_matching_attempt_is_accepted(self):
        self.module.check_run(self.run, self.plan, 100, 2)

    def test_failed_cancelled_or_incomplete_run_is_rejected(self):
        for status, conclusion in (
            ("completed", "failure"),
            ("completed", "cancelled"),
            ("in_progress", None),
        ):
            with self.subTest(status=status, conclusion=conclusion):
                run = {**self.run, "status": status, "conclusion": conclusion}
                with self.assertRaisesRegex(ValueError, "successful complete"):
                    self.module.check_run(run, self.plan, 100, 2)

    def test_artifacts_from_other_run_or_attempt_are_rejected(self):
        for field, wrong in (("id", 99), ("run_attempt", 1)):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "Run identity"):
                    self.module.check_run({**self.run, field: wrong}, self.plan, 100, 2)

    def test_recipe_identity_cannot_be_replaced(self):
        plan = {**self.plan, "recipe_sha": "c" * 40}
        with self.assertRaisesRegex(ValueError, "Recipe SHA"):
            self.module.check_run(self.run, plan, 100, 2)

    def test_non_candidate_and_mixed_identity_plans_are_rejected(self):
        for field, wrong in (
            ("channel", "nightly"),
            ("build_id", "100-1"),
            ("tag", "fork/0.21.7+fork.7"),
            ("source_sha", "dev"),
        ):
            with self.subTest(field=field):
                plan = copy.deepcopy(self.plan)
                plan[field] = wrong
                with self.assertRaises(ValueError):
                    self.module.check_run(self.run, plan, 100, 2)


class DispatchResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "fork_dispatch_wheel", ROOT / ".github/scripts/dispatch_wheel.py"
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def run_record(self, **updates):
        return {
            "id": 100,
            "display_title": "wheel request-123",
            "head_sha": "a" * 40,
            "event": "workflow_dispatch",
            "status": "completed",
            "conclusion": "success",
            "html_url": "https://example.invalid/runs/100",
            **updates,
        }

    def test_matching_uses_request_identity_not_latest_run(self):
        unrelated = self.run_record(id=101, display_title="wheel someone-else")
        expected = self.run_record()
        self.assertEqual(
            self.module.matching_run([unrelated, expected], "request-123", "a" * 40),
            expected,
        )
        self.assertIsNone(
            self.module.matching_run([unrelated], "request-123", "a" * 40)
        )

    def test_duplicate_request_and_wrong_recipe_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Multiple workflow runs"):
            self.module.matching_run(
                [self.run_record(), self.run_record(id=101)], "request-123", "a" * 40
            )
        for updates in ({"head_sha": "b" * 40}, {"event": "push"}):
            with self.subTest(updates=updates):
                with self.assertRaisesRegex(ValueError, "recipe/event"):
                    self.module.matching_run(
                        [self.run_record(**updates)], "request-123", "a" * 40
                    )

    def test_dispatch_waits_for_matching_run_to_appear(self):
        responses = iter(
            [
                {"object": {"sha": "a" * 40}},
                None,
                {
                    "workflow_runs": [
                        self.run_record(id=200, display_title="wheel unrelated")
                    ]
                },
                {"workflow_runs": [self.run_record()]},
            ]
        )
        with mock.patch.object(
            self.module, "api", side_effect=lambda *args: next(responses)
        ):
            with mock.patch.object(self.module.time, "sleep"):
                with mock.patch.object(
                    self.module.time, "monotonic", side_effect=[0, 1, 2]
                ):
                    self.assertEqual(
                        self.module.dispatch(
                            "owner/repo",
                            "dev",
                            "nightly",
                            "request-123",
                            "a" * 40,
                            timeout=10,
                        ),
                        100,
                    )

    def test_moved_recipe_prevents_dispatch(self):
        with mock.patch.object(
            self.module, "api", return_value={"object": {"sha": "b" * 40}}
        ) as api:
            with self.assertRaisesRegex(ValueError, "moved before dispatch"):
                self.module.dispatch(
                    "owner/repo", "dev", "nightly", "request-123", "a" * 40
                )
            self.assertFalse(any(len(call.args) > 1 for call in api.call_args_list))

    def test_child_failure_cancel_and_timeout_propagate(self):
        for conclusion in ("failure", "cancelled", "timed_out"):
            with self.subTest(conclusion=conclusion):
                with mock.patch.object(
                    self.module,
                    "api",
                    return_value=self.run_record(conclusion=conclusion),
                ):
                    with mock.patch.object(
                        self.module.time, "monotonic", side_effect=[0, 1]
                    ):
                        with self.assertRaisesRegex(RuntimeError, conclusion):
                            self.module.await_run(
                                "owner/repo", 100, "a" * 40, final=False
                            )

    def test_completed_success_is_accepted(self):
        with mock.patch.object(self.module, "api", return_value=self.run_record()):
            with mock.patch.object(self.module.time, "monotonic", side_effect=[0, 1]):
                self.assertTrue(
                    self.module.await_run("owner/repo", 100, "a" * 40, final=True)
                )

    def test_waiter_rejects_different_recipe(self):
        with mock.patch.object(
            self.module, "api", return_value=self.run_record(head_sha="b" * 40)
        ):
            with mock.patch.object(self.module.time, "monotonic", side_effect=[0, 1]):
                with self.assertRaisesRegex(ValueError, "different recipe"):
                    self.module.await_run("owner/repo", 100, "a" * 40, final=False)

    def test_only_first_waiter_can_hand_off_pending_child(self):
        for final in (False, True):
            with self.subTest(final=final):
                pending = self.run_record(status="in_progress", conclusion=None)
                with mock.patch.object(self.module, "api", return_value=pending):
                    with mock.patch.object(self.module.time, "sleep"):
                        with mock.patch.object(
                            self.module.time, "monotonic", side_effect=[0, 1, 2]
                        ):
                            if final:
                                with self.assertRaises(TimeoutError):
                                    self.module.await_run(
                                        "owner/repo",
                                        100,
                                        "a" * 40,
                                        final=True,
                                        timeout=2,
                                    )
                            else:
                                self.assertFalse(
                                    self.module.await_run(
                                        "owner/repo",
                                        100,
                                        "a" * 40,
                                        final=False,
                                        timeout=2,
                                    )
                                )

    def test_network_failure_exits_nonzero_without_success_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gh = root / "gh"
            gh.write_text('#!/bin/sh\necho "injected API failure" >&2\nexit 9\n')
            gh.chmod(0o755)
            output = root / "output"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / ".github/scripts/dispatch_wheel.py"),
                    "wait",
                    "--repository",
                    "owner/repo",
                    "--run-id",
                    "100",
                    "--recipe-sha",
                    "a" * 40,
                ],
                env={
                    **os.environ,
                    "PATH": str(root) + ":" + os.environ["PATH"],
                    "GITHUB_OUTPUT": str(output),
                },
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("succeeded", result.stdout)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

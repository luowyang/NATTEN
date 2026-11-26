"""Host-only regression tests for fork assembly and wheel validation boundaries."""

import argparse
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
ASSEMBLE = ROOT / "integration/assemble.sh"
spec = importlib.util.spec_from_file_location(
    "fork_validate_wheel", ROOT / "integration/validate_wheel.py"
)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class GitFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.origin = self.root / "origin.git"
        self.work = self.root / "dev"
        self.branches = self.root / "branches.txt"
        self.env = dict(
            os.environ,
            GIT_AUTHOR_NAME="Fixture",
            GIT_AUTHOR_EMAIL="fixture@example.invalid",
            GIT_COMMITTER_NAME="Fixture",
            GIT_COMMITTER_EMAIL="fixture@example.invalid",
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL=os.devnull,
        )
        self.run_command(["git", "init", "--bare", str(self.origin)])
        self.run_command(["git", "init", "-b", "main", str(self.repo)])
        (self.repo / "src/natten").mkdir(parents=True)
        (self.repo / "src/natten/version.py").write_text('__version__ = "0.21.7"\n')
        (self.repo / "shared.txt").write_text("base\n")
        (self.repo / ".gitignore").write_text("ignored.bin\n")
        self.git("add", ".")
        self.git("commit", "-m", "base")
        self.git("remote", "add", "origin", str(self.origin))
        self.git("push", "origin", "main")
        self.base = self.git("rev-parse", "HEAD").stdout.strip()

    def run_command(self, args, check=True):
        return subprocess.run(
            args, env=self.env, text=True, capture_output=True, check=check
        )

    def git(self, *args, cwd=None, check=True):
        return self.run_command(
            ["git", "-C", str(cwd or self.repo), *args], check=check
        )

    def topic(self, name="topic", path="topic.txt", content="topic\n"):
        self.git("checkout", "-b", name, "main")
        (self.repo / path).write_text(content)
        self.git("add", path)
        self.git("commit", "-m", name)
        self.git("push", "origin", name)
        return self.git("rev-parse", "HEAD").stdout.strip()

    def assemble(self, *args, number="1"):
        return self.run_command(
            [
                "bash",
                str(ASSEMBLE),
                number,
                str(self.repo),
                "--worktree",
                str(self.work),
                "--branches-file",
                str(self.branches),
                *args,
            ],
            check=False,
        )

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class AssemblyTests(GitFixture):
    def test_missing_required_topic_fails_before_creating_worktree_or_tag(self):
        self.branches.write_text("missing-topic\n")
        result = self.assemble("--tag")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing required topic", result.stderr)
        self.assertFalse(self.work.exists())
        self.assertEqual(self.git("tag", "--list").stdout, "")

    def test_empty_topic_list_fails(self):
        self.branches.write_text("# No inputs\n")
        self.assertNotEqual(self.assemble().returncode, 0)
        self.assertFalse(self.work.exists())

    def test_local_remote_disagreement_requires_explicit_ref(self):
        remote_sha = self.topic()
        (self.repo / "local-only.txt").write_text("not published\n")
        self.git("add", ".")
        self.git("commit", "-m", "local changes")
        local_sha = self.git("rev-parse", "HEAD").stdout.strip()
        self.branches.write_text("topic\n")
        result = self.assemble()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Ambiguous topic", result.stderr)
        self.assertFalse(self.work.exists())
        self.branches.write_text("refs/remotes/origin/topic\n")
        self.assert_success(self.assemble("--tag"))
        self.assertEqual(
            self.git("merge-base", "--is-ancestor", remote_sha, "dev").returncode, 0
        )
        self.assertNotEqual(
            self.git(
                "merge-base", "--is-ancestor", local_sha, "dev", check=False
            ).returncode,
            0,
        )
        lock = (self.work / "integration/assembly.lock").read_text()
        self.assertIn(remote_sha, lock)
        self.assertNotIn(local_sha, lock)

    def test_rebuild_preserves_tracked_untracked_and_ignored_work(self):
        self.topic()
        self.branches.write_text("topic\n")
        self.assert_success(self.assemble())
        for name, content in (
            ("shared.txt", "unstaged\n"),
            ("untracked.txt", "untracked\n"),
            ("ignored.bin", "ignored\n"),
        ):
            with self.subTest(name=name):
                file = self.work / name
                original = file.read_text() if file.exists() else None
                file.write_text(content)
                before = self.git("rev-parse", "dev").stdout
                result = self.assemble(number="2")
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(file.read_text(), content)
                self.assertEqual(self.git("rev-parse", "dev").stdout, before)
                if original is None:
                    file.unlink()
                else:
                    file.write_text(original)

    def test_rebuild_preserves_staged_work(self):
        self.topic()
        self.branches.write_text("topic\n")
        self.assert_success(self.assemble())
        (self.work / "shared.txt").write_text("staged\n")
        self.git("add", "shared.txt", cwd=self.work)
        self.assertNotEqual(self.assemble(number="2").returncode, 0)
        self.assertIn("staged", self.git("show", ":shared.txt", cwd=self.work).stdout)

    def test_continue_uses_locked_topics_and_base_after_refs_move(self):
        self.topic("first", "shared.txt", "first\n")
        self.topic("conflict", "shared.txt", "conflict\n")
        old_later = self.topic("later", "later.txt", "old later\n")
        self.branches.write_text("first\nconflict\nlater\n")
        result = self.assemble("--resolve-manually")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        (self.work / "shared.txt").write_text("resolved\n")
        self.git("add", "shared.txt", cwd=self.work)
        self.git("commit", "--no-edit", cwd=self.work)
        (self.repo / "later.txt").write_text("new later\n")
        self.git("add", "later.txt")
        self.git("commit", "-m", "advance later")
        new_later = self.git("rev-parse", "HEAD").stdout.strip()
        self.git("push", "origin", "later")
        self.git("checkout", "main")
        (self.repo / "new-base.txt").write_text("new base\n")
        self.git("add", ".")
        self.git("commit", "-m", "advance base")
        new_base = self.git("rev-parse", "HEAD").stdout.strip()
        self.git("push", "origin", "main")
        self.branches.write_text("missing-topic-now\n")
        self.assert_success(self.assemble("--continue", "--tag"))
        self.assertEqual((self.work / "later.txt").read_text(), "old later\n")
        for sha in (new_base, new_later):
            self.assertNotEqual(
                self.git(
                    "merge-base", "--is-ancestor", sha, "dev", check=False
                ).returncode,
                0,
            )
        lock = (self.work / "integration/assembly.lock").read_text()
        self.assertIn(old_later, lock)
        self.assertIn(self.base, lock)
        self.assertNotEqual(self.assemble("--continue", number="2").returncode, 0)

    def test_submodule_ignored_assets_survive_rebuild(self):
        self.env["GIT_ALLOW_PROTOCOL"] = "file"
        child = self.root / "child"
        self.run_command(["git", "init", "-b", "main", str(child)])
        (child / "file.txt").write_text("child\n")
        (child / ".gitignore").write_text("ignored.bin\n")
        self.git("add", ".", cwd=child)
        self.git("commit", "-m", "child", cwd=child)
        self.git("submodule", "add", str(child), "third_party/child")
        self.git("commit", "-am", "add child")
        self.git("push", "origin", "main")
        self.topic()
        self.branches.write_text("topic\n")
        self.assert_success(self.assemble())
        artifact = self.work / "third_party/child/ignored.bin"
        artifact.write_text("must survive\n")
        self.assertNotEqual(self.assemble(number="2").returncode, 0)
        self.assertEqual(artifact.read_text(), "must survive\n")
        artifact.unlink()
        self.assert_success(self.assemble(number="2"))
        self.assertEqual(
            (self.work / "third_party/child/file.txt").read_text(), "child\n"
        )

    def test_rebuild_does_not_remove_another_branch_worktree(self):
        self.topic()
        self.branches.write_text("topic\n")
        self.assert_success(self.assemble())
        self.git("checkout", "-b", "valuable-branch", cwd=self.work)
        self.assertNotEqual(self.assemble(number="2").returncode, 0)
        self.assertEqual(
            self.git("symbolic-ref", "--short", "HEAD", cwd=self.work).stdout.strip(),
            "valuable-branch",
        )

    def test_continue_preserves_ignored_file_that_later_topic_tracks(self):
        self.topic("first", "shared.txt", "first\n")
        self.topic("conflict", "shared.txt", "conflict\n")
        self.topic("later")
        (self.repo / "ignored.bin").write_text("from topic\n")
        self.git("add", "-f", "ignored.bin")
        self.git("commit", "-m", "track ignored file")
        self.git("push", "origin", "later")
        self.branches.write_text("first\nconflict\nlater\n")
        self.assertEqual(self.assemble("--resolve-manually").returncode, 2)
        (self.work / "shared.txt").write_text("resolved\n")
        self.git("add", "shared.txt", cwd=self.work)
        self.git("commit", "--no-edit", cwd=self.work)
        artifact = self.work / "ignored.bin"
        artifact.write_text("uncommitted valuable artifact\n")
        before = self.git("rev-parse", "HEAD", cwd=self.work).stdout
        self.assertNotEqual(self.assemble("--continue").returncode, 0)
        self.assertEqual(artifact.read_text(), "uncommitted valuable artifact\n")
        self.assertEqual(self.git("rev-parse", "HEAD", cwd=self.work).stdout, before)

    def test_continue_preserves_ignored_submodule_file(self):
        self.env["GIT_ALLOW_PROTOCOL"] = "file"
        child = self.root / "child"
        self.run_command(["git", "init", "-b", "main", str(child)])
        (child / "file.txt").write_text("child\n")
        (child / ".gitignore").write_text("ignored.bin\n")
        self.git("add", ".", cwd=child)
        self.git("commit", "-m", "child", cwd=child)
        self.git("submodule", "add", str(child), "third_party/child")
        self.git("commit", "-am", "add child")
        self.git("push", "origin", "main")
        self.topic("first", "shared.txt", "first\n")
        self.topic("conflict", "shared.txt", "conflict\n")
        self.branches.write_text("first\nconflict\n")
        self.assertEqual(self.assemble("--resolve-manually").returncode, 2)
        (self.work / "shared.txt").write_text("resolved\n")
        self.git("add", "shared.txt", cwd=self.work)
        self.git("commit", "--no-edit", cwd=self.work)
        self.git("submodule", "update", "--init", cwd=self.work)
        artifact = self.work / "third_party/child/ignored.bin"
        artifact.write_text("valuable submodule artifact\n")
        self.assertNotEqual(self.assemble("--continue").returncode, 0)
        self.assertEqual(artifact.read_text(), "valuable submodule artifact\n")

    def test_continue_rejects_unresolved_merge(self):
        self.topic("first", "shared.txt", "first\n")
        self.topic("conflict", "shared.txt", "conflict\n")
        self.branches.write_text("first\nconflict\n")
        self.assertEqual(self.assemble("--resolve-manually").returncode, 2)
        self.assertNotEqual(self.assemble("--continue").returncode, 0)
        self.assertEqual(
            self.git("rev-parse", "--verify", "MERGE_HEAD", cwd=self.work).returncode, 0
        )


class WheelValidationTests(GitFixture):
    def setUp(self):
        super().setUp()
        (self.repo / "tests").mkdir()
        (self.repo / "tests/test_varlen_fixture.py").write_text(
            "def test_fixture(): pass\n"
        )
        self.git("add", ".")
        self.git("commit", "-m", "fixture tests")
        tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        self.wheel = self.root / f"natten-0.21.7+fork.1-{tag}-{tag}-linux_x86_64.whl"
        self.wheel.write_bytes(b"wheel fixture")
        self.manifest = {
            "schema_version": 1,
            "source_sha": self.git("rev-parse", "HEAD").stdout.strip(),
            "recipe_sha": "b" * 40,
            "build_id": "123-1",
            "version": "0.21.7+fork.1",
            "python_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
            "wheel": self.wheel.name,
            "sha256": hashlib.sha256(self.wheel.read_bytes()).hexdigest(),
        }
        self.manifest_path = self.root / "MANIFEST.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.output = self.root / "GPU-VALIDATION.json"
        self.args = argparse.Namespace(
            wheel=self.wheel,
            manifest=self.manifest_path,
            source=self.repo,
            output=self.output,
            timeout=1,
            tests=None,
            preflight=False,
        )

        self.installed_patch = mock.patch.object(
            validator, "check_installed", return_value={}
        )
        self.installed_patch.start()
        self.addCleanup(self.installed_patch.stop)

    def installed_fixture(self):
        self.installed_patch.stop()
        tag = self.manifest["python_tag"]
        version = self.manifest["version"]
        native = f"natten/libnatten.cpython-{tag[2:]}-x86_64-linux-gnu.so"
        elf = b"\x7fELF\x02\x01" + b"\x00" * 10 + b"\x03\x00\x3e\x00" + b"\x00" * 44
        entries = {
            f"natten-{version}.dist-info/METADATA": f"Metadata-Version: 2.1\nName: natten\nVersion: {version}\n".encode(),
            f"natten-{version}.dist-info/WHEEL": f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {tag}-{tag}-linux_x86_64\n".encode(),
            "natten/__init__.py": b"# fixture\n",
            "natten/version.py": f'__version__ = "{version}"\n'.encode(),
            native: elf,
        }
        site = self.root / "site-packages"
        with zipfile.ZipFile(self.wheel, "w") as archive:
            for member, data in entries.items():
                archive.writestr(member, data)
                file = site / member
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(data)
        self.manifest["sha256"] = hashlib.sha256(self.wheel.read_bytes()).hexdigest()
        self.manifest_path.write_text(json.dumps(self.manifest))
        dist = SimpleNamespace(
            version=version, locate_file=lambda member: site / member
        )
        package = SimpleNamespace(
            __file__=str(site / "natten/__init__.py"),
            __version__=version,
            HAS_LIBNATTEN=True,
        )
        modules = {
            "torch": SimpleNamespace(),
            "natten": package,
            "natten.libnatten": SimpleNamespace(__file__=str(site / native)),
        }
        import importlib
        import importlib.metadata
        import sysconfig

        original_import = importlib.import_module
        original_distribution = importlib.metadata.distribution
        original_path = sysconfig.get_path
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            mock.patch.object(
                importlib,
                "import_module",
                side_effect=lambda name, *a, **k: (
                    sys.modules.get(name, modules[name])
                    if name in modules
                    else original_import(name, *a, **k)
                ),
            )
        )
        stack.enter_context(
            mock.patch.object(
                importlib.metadata,
                "distribution",
                side_effect=lambda name: (
                    dist if name == "natten" else original_distribution(name)
                ),
            )
        )
        stack.enter_context(
            mock.patch.object(
                sysconfig,
                "get_path",
                side_effect=lambda name, *a, **k: (
                    str(site)
                    if name in ("purelib", "platlib")
                    else original_path(name, *a, **k)
                ),
            )
        )
        stack.enter_context(mock.patch.dict(sys.modules, modules))
        return site / native

    def test_same_version_wrong_native_is_rejected_before_gpu(self):
        native = self.installed_fixture()
        native.write_bytes(native.read_bytes() + b"different build")
        with self.assertRaisesRegex(ValueError, "installed native extension differs"):
            validator.check_native_gpu(self.wheel, self.manifest)
        self.assertFalse(self.output.exists())

    def test_installed_bytes_are_rechecked_after_pytest(self):
        native = self.installed_fixture()
        invoke = self.pytest_result("<testsuite><testcase/></testsuite>")

        def modify_native(command, **kwargs):
            result = invoke(command, **kwargs)
            native.write_bytes(native.read_bytes() + b"changed while testing")
            return result

        with (
            mock.patch.object(validator, "check_native_gpu"),
            mock.patch.object(validator, "run", side_effect=modify_native),
        ):
            with self.assertRaisesRegex(
                ValueError, "installed native extension differs"
            ):
                validator.validate(self.args)
        self.assertFalse(self.output.exists())

    def test_pytest_worker_ignores_pythonpath_shadow(self):
        shadow = self.root / "shadow"
        shadow.mkdir()
        (shadow / "pytest.py").write_text(
            'raise RuntimeError("SHADOW_PYTEST_SELECTED")\n'
        )
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                str(ROOT / "integration/validate_wheel.py"),
                "--pytest-worker",
                "--wheel",
                str(self.wheel),
                "--manifest",
                str(self.manifest_path),
                "--source",
                str(self.repo),
                "--output",
                str(self.output),
                f"--junitxml={self.output.with_suffix('.xml')}",
                "--tests",
                str(self.repo / "tests/test_varlen_fixture.py"),
            ],
            env=dict(
                self.env, PYTHONPATH=str(shadow), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
            ),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("SHADOW_PYTEST_SELECTED", result.stdout + result.stderr)
        self.assertIn("BadZipFile", result.stdout + result.stderr)

    def test_pytest_guard_rejects_conftest_package_shadow(self):
        self.installed_fixture()
        (self.repo / "tests/conftest.py").write_text(
            "import sys\nfrom types import SimpleNamespace\nsys.modules['natten'] = SimpleNamespace(__file__='/shadow/natten/__init__.py')\n"
        )
        self.git("add", ".")
        self.git("commit", "-m", "conftest modifies imported package")
        self.manifest["source_sha"] = self.git("rev-parse", "HEAD").stdout.strip()
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.args.junitxml = self.output.with_suffix(".xml")
        self.args.tests = [self.repo / "tests/test_varlen_fixture.py"]
        diagnostics = io.StringIO()
        with (
            mock.patch.dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),
            redirect_stdout(diagnostics),
            redirect_stderr(diagnostics),
        ):
            self.assertNotEqual(validator.run_pytest_worker(self.args), 0)
        self.assertIn("natten import is shadowed", diagnostics.getvalue())
        self.assertFalse(self.output.exists())

    def pytest_result(self, xml, returncode=0):
        def run(command, **kwargs):
            path = next(
                arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")
            )
            Path(path).write_text(xml)
            return subprocess.CompletedProcess(command, returncode)

        return run

    def test_shell_preserves_validator_timeout_exit_code(self):
        conda = self.root / "fake-conda"
        conda.write_text(
            '#!/usr/bin/env bash\ncase " $* " in\n  *--preflight*) exit 0 ;;\n  *validate_wheel.py*) exit 124 ;;\n  *) exit 0 ;;\nesac\n'
        )
        conda.chmod(0o755)
        env_path = self.root / "validation-env"
        env_path.mkdir()
        result = subprocess.run(
            [
                "bash",
                str(ROOT / "integration/build_wheel.sh"),
                "--dev-worktree",
                str(self.repo),
                "--env-path",
                str(env_path),
                "--skip-env-clone",
                "--wheel",
                str(self.wheel),
                "--manifest",
                str(self.manifest_path),
                "--output",
                str(self.output),
            ],
            env=dict(self.env, CONDA_EXE=str(conda)),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 124, result.stdout + result.stderr)
        self.assertFalse(self.output.exists())

    def test_hash_mismatch_stops_before_native_or_tests(self):
        self.wheel.write_bytes(b"different wheel")
        with mock.patch.object(validator, "check_native_gpu") as native:
            with self.assertRaisesRegex(ValueError, "SHA256"):
                validator.validate(self.args)
            native.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_wrong_source_or_dirty_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "HEAD"):
            validator.check_source(self.repo, "a" * 40)
        (self.repo / "changed.txt").write_text("uncommitted\n")
        with self.assertRaisesRegex(ValueError, "uncommitted"):
            validator.check_source(self.repo, self.manifest["source_sha"])

    def test_gpu_unavailable_does_not_produce_proof(self):
        with mock.patch.object(
            validator, "check_native_gpu", side_effect=ValueError("CUDA unavailable")
        ):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                validator.validate(self.args)
        self.assertFalse(self.output.exists())

    def test_timeout_is_nonzero_and_has_no_passed_attestation(self):
        with (
            mock.patch.object(validator, "check_native_gpu"),
            mock.patch.object(
                validator, "run", side_effect=subprocess.TimeoutExpired("pytest", 1)
            ),
        ):
            self.assertEqual(validator.validate(self.args), 124)
        self.assertFalse(self.output.exists())

    def test_all_skipped_does_not_produce_proof(self):
        with (
            mock.patch.object(validator, "check_native_gpu"),
            mock.patch.object(
                validator,
                "run",
                side_effect=self.pytest_result(
                    "<testsuite><testcase><skipped/></testcase></testsuite>"
                ),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "all skipped"):
                validator.validate(self.args)
        self.assertFalse(self.output.exists())

    def test_failed_tests_propagate_exit_code(self):
        with (
            mock.patch.object(validator, "check_native_gpu"),
            mock.patch.object(
                validator,
                "run",
                side_effect=self.pytest_result(
                    "<testsuite><testcase><failure/></testcase></testsuite>", 1
                ),
            ),
        ):
            self.assertEqual(validator.validate(self.args), 1)
        self.assertFalse(self.output.exists())

    def test_missing_requested_test_does_not_run_pytest(self):
        self.args.tests = [Path("tests/missing.py")]
        with (
            mock.patch.object(validator, "check_native_gpu"),
            mock.patch.object(validator, "run") as run,
        ):
            with self.assertRaisesRegex(ValueError, "Every selected test"):
                validator.validate(self.args)
            run.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_source_change_during_tests_prevents_attestation(self):
        invoke = self.pytest_result("<testsuite><testcase/></testsuite>")

        def change_source(command, **kwargs):
            result = invoke(command, **kwargs)
            (self.repo / "shared.txt").write_text("changed during test\n")
            return result

        with (
            mock.patch.object(validator, "check_native_gpu"),
            mock.patch.object(validator, "run", side_effect=change_source),
        ):
            with self.assertRaisesRegex(ValueError, "uncommitted"):
                validator.validate(self.args)
        self.assertFalse(self.output.exists())

    def test_success_proof_binds_original_wheel_and_build(self):
        with (
            mock.patch.object(validator, "check_native_gpu"),
            mock.patch.object(
                validator,
                "run",
                side_effect=self.pytest_result("<testsuite><testcase/></testsuite>"),
            ),
        ):
            self.assertEqual(validator.validate(self.args), 0)
        proof = json.loads(self.output.read_text())
        for key in ("source_sha", "recipe_sha", "build_id"):
            self.assertEqual(proof[key], self.manifest[key])
        self.assertEqual(
            proof["wheels"][self.wheel.name]["sha256"], self.manifest["sha256"]
        )
        self.assertEqual(proof["wheels"][self.wheel.name]["status"], "passed")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            validator.validate(self.args)


if __name__ == "__main__":
    unittest.main()

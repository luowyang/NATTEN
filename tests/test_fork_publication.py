"""Release state-machine regressions; no network, GPU, or real releases."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/publish_wheels.py"
SPEC = importlib.util.spec_from_file_location("publish_wheels", SCRIPT)
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


class FakeGitHub:
    def __init__(self):
        self.tags = {}
        self.items = {}
        self.content = {}
        self.events = []
        self.upload_error = False
        self.corrupt_upload = False
        self.change_tag_during_upload = False
        self.next_id = 1

    def tag_sha(self, tag):
        return self.tags.get(tag)

    def release(self, tag):
        return copy.deepcopy(self.items.get(tag))

    def releases(self):
        return copy.deepcopy(list(self.items.values()))

    def api(self, endpoint, *, method="GET", data=None):
        self.events.append((method, endpoint, data))
        if endpoint == "git/refs" and method == "POST":
            tag = data["ref"].removeprefix("refs/tags/")
            if tag in self.tags:
                raise RuntimeError("Tag already exists")
            self.tags[tag] = data["sha"]
            return {}
        if endpoint == "releases" and method == "POST":
            item = dict(data, id=self.next_id, assets=[], html_url="fixture://release")
            self.next_id += 1
            self.items[item["tag_name"]] = item
            return copy.deepcopy(item)
        if endpoint.startswith("releases/"):
            item = next(x for x in self.items.values() if str(x["id"]) == endpoint[9:])
            if method == "PATCH":
                item.update(data)
                return copy.deepcopy(item)
            if method == "DELETE":
                del self.items[item["tag_name"]]
                return None
        if endpoint.startswith("git/refs/tags/") and method == "DELETE":
            tag = endpoint[len("git/refs/tags/") :]
            from urllib.parse import unquote

            self.tags.pop(unquote(tag), None)
            return None
        raise AssertionError((method, endpoint, data))

    def upload(self, tag, paths):
        self.events.append(("UPLOAD", tag, None))
        for index, path in enumerate(paths):
            if self.upload_error and index == 1:
                raise RuntimeError("Injected upload failure")
            data = path.read_bytes()
            self.content[(tag, path.name)] = data + (
                b"corrupt" if self.corrupt_upload else b""
            )
            self.items[tag]["assets"].append({"name": path.name})
        if self.change_tag_during_upload:
            self.tags[tag] = "f" * 40

    def download(self, tag, name, directory):
        (directory / name).write_bytes(self.content[(tag, name)])


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = {
            "source_sha": "a" * 40,
            "recipe_sha": "b" * 40,
            "version": "0.21.7+fork.5.dev20261126.123.1",
            "build_id": "123-1",
            "channel": "nightly",
        }
        self.tag = "nightly/20261126-123-1"
        self.gh = FakeGitHub()
        self.make_bundle()

    def make_bundle(self):
        for tag in ("cp310", "cp311"):
            directory = self.root / tag
            directory.mkdir(exist_ok=True)
            filename = f"natten-{self.identity['version']}-{tag}-{tag}-linux_x86_64.whl"
            wheel = directory / filename
            wheel.write_bytes(f"fixture wheel {tag}".encode())
            manifest = dict(
                self.identity,
                schema_version=1,
                python_tag=tag,
                wheel=filename,
                sha256=publisher.digest(wheel),
                host_validation="passed",
            )
            (directory / f"MANIFEST-{self.identity['version']}-{tag}.json").write_text(
                json.dumps(manifest)
            )

    def bundle(self):
        return publisher.verify_bundle(self.root, self.identity)

    def manifest_path(self, tag="cp310"):
        return next((self.root / tag).glob("MANIFEST-*.json"))

    def change_manifest(self, **fields):
        path = self.manifest_path()
        value = json.loads(path.read_text())
        value.update(fields)
        path.write_text(json.dumps(value))

    def publish(self):
        assets, _ = self.bundle()
        return publisher.publish(self.gh, assets, self.identity, self.tag)

    def test_complete_bundle_publishes_only_after_upload_verification(self):
        result = self.publish()
        self.assertFalse(result["draft"])
        create = next(x for x in self.gh.events if x[1] == "releases")
        self.assertTrue(create[2]["draft"])
        self.assertTrue(create[2]["prerelease"])
        self.assertIn("GPU kernel tests were not run", create[2]["body"])
        self.assertEqual(len(result["assets"]), 4)
        self.assertEqual(self.gh.events[-1][0], "PATCH")

    def test_missing_leg_cannot_create_release(self):
        self.manifest_path().unlink()
        with self.assertRaisesRegex(ValueError, "exactly two"):
            self.publish()
        self.assertEqual(self.gh.events, [])

    def test_missing_wheel_cannot_create_release(self):
        next((self.root / "cp310").glob("*.whl")).unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.publish()
        self.assertEqual(self.gh.events, [])

    def test_invalid_checksum_cannot_create_release(self):
        self.change_manifest(sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.publish()
        self.assertEqual(self.gh.events, [])

    def test_mixed_source_recipe_version_and_build_are_rejected(self):
        path = self.manifest_path()
        original = path.read_text()
        for field in publisher.IDENTITY_FIELDS:
            with self.subTest(field=field):
                path.write_text(original)
                self.change_manifest(**{field: "different"})
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    self.bundle()

    def test_failed_host_validation_is_rejected(self):
        self.change_manifest(host_validation="failed")
        with self.assertRaisesRegex(ValueError, "Host validation"):
            self.bundle()

    def test_unexpected_extra_wheel_is_rejected(self):
        (self.root / "stale.whl").write_bytes(b"stale")
        with self.assertRaisesRegex(ValueError, "Unexpected wheel"):
            self.bundle()

    def test_upload_failure_keeps_partial_draft_private_and_can_resume(self):
        self.gh.upload_error = True
        with self.assertRaisesRegex(RuntimeError, "upload failure"):
            self.publish()
        self.assertTrue(self.gh.items[self.tag]["draft"])
        self.assertFalse(any(x[0] == "PATCH" for x in self.gh.events))
        self.gh.upload_error = False
        self.assertFalse(self.publish()["draft"])
        self.assertEqual(len(self.gh.items[self.tag]["assets"]), 4)

    def test_uploaded_corruption_prevents_publication(self):
        self.gh.corrupt_upload = True
        with self.assertRaisesRegex(ValueError, "asset differs"):
            self.publish()
        self.assertTrue(self.gh.items[self.tag]["draft"])

    def test_tag_mismatch_rejected_before_mutation(self):
        self.gh.tags[self.tag] = "f" * 40
        with self.assertRaisesRegex(ValueError, "different source"):
            self.publish()
        self.assertEqual(self.gh.events, [])

    def test_tag_change_during_upload_prevents_publication(self):
        self.gh.change_tag_during_upload = True
        with self.assertRaisesRegex(ValueError, "changed during upload"):
            self.publish()
        self.assertTrue(self.gh.items[self.tag]["draft"])

    def test_published_assets_are_immutable_and_idempotent(self):
        self.publish()
        events = list(self.gh.events)
        self.publish()
        self.assertEqual(self.gh.events, events)
        name = next(name for tag, name in self.gh.content if name.endswith(".whl"))
        self.gh.content[(self.tag, name)] = b"unexpected binary"
        with self.assertRaisesRegex(ValueError, "asset differs"):
            self.publish()
        self.assertEqual(self.gh.events, events)

    def test_incomplete_public_release_is_not_silently_repaired(self):
        self.publish()
        self.gh.items[self.tag]["assets"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.publish()

    def test_nightly_tag_cannot_alias_another_attempt_or_version(self):
        for tag in (
            "nightly/20261126",
            "nightly/20261126-123-2",
            "nightly/20261125-123-1",
        ):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                publisher.verify_tag(tag, self.identity)

    def test_retention_preserves_verified_release_even_on_late_rerun(self):
        published = self.publish()
        old = dict(
            published,
            id=99,
            tag_name="nightly/20261008",
            assets=[],
        )
        self.gh.items[old["tag_name"]] = old
        permanent = dict(old, id=100, tag_name="fork/0.21.7+fork.4", prerelease=False)
        self.gh.items[permanent["tag_name"]] = permanent
        draft = dict(old, id=101, tag_name="nightly/20260901-122-1", draft=True)
        self.gh.items[draft["tag_name"]] = draft
        publisher.prune_nightlies(self.gh, published, 14, today=date(2027, 1, 1))
        self.assertIn(self.tag, self.gh.items)
        self.assertNotIn(old["tag_name"], self.gh.items)
        self.assertIn(permanent["tag_name"], self.gh.items)
        self.assertIn(draft["tag_name"], self.gh.items)

    def test_retention_cannot_run_on_unpublished_draft(self):
        with self.assertRaisesRegex(ValueError, "published nightly"):
            publisher.prune_nightlies(self.gh, {"draft": True}, 14)
        self.assertEqual(self.gh.events, [])

    def test_gpu_attestation_requires_exact_checksums_for_both_legs(self):
        _, wheels = self.bundle()
        evidence = {
            "schema_version": 1,
            **{
                key: self.identity[key]
                for key in ("source_sha", "recipe_sha", "build_id")
            },
            "wheels": {
                item["wheel"]: {
                    "sha256": item["sha256"],
                    "status": "passed",
                    "evidence": f"logs/{tag}.log",
                }
                for tag, item in wheels.items()
            },
        }
        path = self.root / "gpu.json"
        path.write_text(json.dumps(evidence))
        publisher.verify_gpu_attestation(path, self.identity, wheels)
        name = next(iter(evidence["wheels"]))
        for bad in (
            {**evidence["wheels"][name], "sha256": "0" * 64},
            {**evidence["wheels"][name], "status": "timeout"},
            {**evidence["wheels"][name], "evidence": ""},
        ):
            with self.subTest(bad=bad):
                copy_evidence = copy.deepcopy(evidence)
                copy_evidence["wheels"][name] = bad
                path.write_text(json.dumps(copy_evidence))
                with self.assertRaisesRegex(ValueError, "GPU evidence"):
                    publisher.verify_gpu_attestation(path, self.identity, wheels)
        evidence["wheels"].pop(name)
        path.write_text(json.dumps(evidence))
        with self.assertRaisesRegex(ValueError, "exactly both"):
            publisher.verify_gpu_attestation(path, self.identity, wheels)

    def test_verify_only_never_needs_github_or_gpu_evidence(self):
        argv = [str(SCRIPT), "--artifacts", str(self.root), "--verify-only"]
        for key, value in self.identity.items():
            argv.extend(["--" + key.replace("_", "-"), value])
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(
                publisher,
                "GitHub",
                side_effect=AssertionError("Unexpected network client"),
            ),
        ):
            publisher.main()

    def test_split_gpu_attestations_merge_and_reject_duplicate_or_foreign_build(self):
        _, wheels = self.bundle()
        paths = []
        for tag, item in wheels.items():
            evidence = {
                "schema_version": 1,
                **{
                    key: self.identity[key]
                    for key in ("source_sha", "recipe_sha", "build_id")
                },
                "wheels": {
                    item["wheel"]: {
                        "sha256": item["sha256"],
                        "status": "passed",
                        "evidence": f"logs/{tag}.log",
                    }
                },
            }
            path = self.root / f"gpu-{tag}.json"
            path.write_text(json.dumps(evidence))
            paths.append(path)
        merged = publisher.merge_gpu_attestations(paths, self.identity, wheels)
        self.assertEqual(len(merged["wheels"]), 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            publisher.merge_gpu_attestations(paths + paths[:1], self.identity, wheels)
        with self.assertRaisesRegex(ValueError, "exactly both"):
            publisher.merge_gpu_attestations(paths[:1], self.identity, wheels)
        evidence = json.loads(paths[0].read_text())
        evidence["recipe_sha"] = "c" * 40
        paths[0].write_text(json.dumps(evidence))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            publisher.merge_gpu_attestations(paths, self.identity, wheels)

    def formal_bundle(self):
        self.root = self.root / "formal"
        self.root.mkdir()
        self.identity["channel"] = "release"
        self.identity["version"] = "0.21.7+fork.5"
        self.tag = "fork/0.21.7+fork.5"
        self.make_bundle()

    def cli_args(self):
        argv = [str(SCRIPT), "--artifacts", str(self.root), "--tag", self.tag]
        for key, value in self.identity.items():
            argv.extend(["--" + key.replace("_", "-"), value])
        return argv

    def test_formal_release_cli_requires_gpu_evidence_before_github(self):
        self.formal_bundle()
        with (
            mock.patch.object(sys, "argv", self.cli_args()),
            mock.patch.object(
                publisher, "GitHub", side_effect=AssertionError("Unexpected network")
            ),
            mock.patch("sys.stderr"),
            self.assertRaises(SystemExit) as error,
        ):
            publisher.main()
        self.assertEqual(error.exception.code, 2)

    def test_formal_release_publishes_verified_gpu_evidence_asset(self):
        self.formal_bundle()
        _, wheels = self.bundle()
        evidence = {
            "schema_version": 1,
            **{
                key: self.identity[key]
                for key in ("source_sha", "recipe_sha", "build_id")
            },
            "wheels": {
                item["wheel"]: {
                    "sha256": item["sha256"],
                    "status": "passed",
                    "evidence": "gpu.log",
                }
                for item in wheels.values()
            },
        }
        path = self.root / "gpu.json"
        path.write_text(json.dumps(evidence))
        self.gh.tags[self.tag] = self.identity["source_sha"]
        argv = self.cli_args() + ["--gpu-attestation", str(path)]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(publisher, "GitHub", return_value=self.gh),
        ):
            publisher.main()
        release = self.gh.items[self.tag]
        self.assertFalse(release["draft"])
        self.assertFalse(release["prerelease"])
        self.assertEqual(len(release["assets"]), 5)
        uploaded = json.loads(self.gh.content[(self.tag, "GPU-VALIDATION.json")])
        self.assertEqual(uploaded, evidence)

    def test_formal_release_requires_existing_correct_tag(self):
        self.formal_bundle()
        with self.assertRaisesRegex(ValueError, "must already exist"):
            self.publish()
        self.assertEqual(self.gh.events, [])

    def test_warm_bundle_verification_needs_no_release_tag(self):
        self.identity["channel"] = "warm"
        self.make_bundle()
        argv = self.cli_args() + ["--verify-only"]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(
                publisher, "GitHub", side_effect=AssertionError("Unexpected network")
            ),
        ):
            publisher.main()

    def test_retention_preserves_newer_complete_nightly_on_late_rerun(self):
        published = self.publish()
        newer = dict(published, id=99, tag_name="nightly/20261201-124-1")
        self.gh.items[newer["tag_name"]] = newer
        incomplete = dict(
            published, id=100, tag_name="nightly/20261202-125-1", assets=[]
        )
        self.gh.items[incomplete["tag_name"]] = incomplete
        publisher.prune_nightlies(self.gh, published, 14, today=date(2027, 1, 1))
        self.assertIn(self.tag, self.gh.items)
        self.assertIn(newer["tag_name"], self.gh.items)
        self.assertNotIn(incomplete["tag_name"], self.gh.items)

    def test_finish_build_verify_end_to_end_without_external_programs(self):
        plan = self.root / "plan.json"
        for channel in ("warm", "nightly", "release"):
            with self.subTest(channel=channel):
                self.identity["channel"] = channel
                self.make_bundle()
                plan.write_text(
                    json.dumps(
                        dict(self.identity, tag="" if channel == "warm" else self.tag)
                    )
                )
                result = subprocess.run(
                    [
                        sys.executable,
                        "-S",
                        str(SCRIPT.with_name("finish_build.py")),
                        "verify",
                        "--plan",
                        str(plan),
                        "--artifacts",
                        str(self.root),
                    ],
                    env=dict(os.environ, PATH=""),
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Both Python wheels", result.stdout)
        self.manifest_path().unlink()
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(SCRIPT.with_name("finish_build.py")),
                "verify",
                "--plan",
                str(plan),
                "--artifacts",
                str(self.root),
            ],
            env=dict(os.environ, PATH=""),
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly two", result.stderr)
        self.make_bundle()
        self.change_manifest(build_id="123-2")
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(SCRIPT.with_name("finish_build.py")),
                "verify",
                "--plan",
                str(plan),
                "--artifacts",
                str(self.root),
            ],
            env=dict(os.environ, PATH=""),
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("identity mismatch", result.stderr)

    def test_publisher_verify_runs_without_site_packages(self):
        argv = [sys.executable, "-S", *self.cli_args(), "--verify-only"]
        result = subprocess.run(
            argv, env=dict(os.environ, PATH=""), text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_existing_release_channel_cannot_change_silently(self):
        self.publish()
        self.gh.items[self.tag]["prerelease"] = False
        events = list(self.gh.events)
        with self.assertRaisesRegex(ValueError, "release channel"):
            self.publish()
        self.assertEqual(self.gh.events, events)

    def test_gh_only_treats_http_404_as_missing(self):
        github = publisher.GitHub("owner/repo")
        for stderr in ("gh: API rate limit exceeded (HTTP 403)", "network unavailable"):
            error = subprocess.CalledProcessError(1, "gh", stderr=stderr)
            with mock.patch.object(github, "command", side_effect=error):
                with self.assertRaises(subprocess.CalledProcessError):
                    github.release("tag")
        error = subprocess.CalledProcessError(
            1, "gh", stderr="gh: Not Found (HTTP 404)"
        )
        with mock.patch.object(github, "command", side_effect=error):
            self.assertIsNone(github.release("tag"))

    def test_annotated_tag_is_resolved_to_commit(self):
        github = publisher.GitHub("owner/repo")
        with mock.patch.object(
            github,
            "api",
            side_effect=[
                {"object": {"type": "tag", "sha": "c" * 40}},
                {"object": {"type": "commit", "sha": "a" * 40}},
            ],
        ):
            self.assertEqual(github.tag_sha("fork/version"), "a" * 40)


if __name__ == "__main__":
    unittest.main()

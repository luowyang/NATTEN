"""Wheel integrity and native-load checks, using CPU-only fixtures."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/scripts/validate_wheel.py"
SPEC = importlib.util.spec_from_file_location("fork_validate_wheel", SCRIPT)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)
VERSION = "0.21.7+fork.8"
PYTHON_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"
NATIVE = f"natten/libnatten.cpython-{PYTHON_TAG[2:]}-x86_64-linux-gnu.so"
ELF = b"\x7fELF\x02\x01" + b"\x00" * 10 + b"\x03\x00\x3e\x00" + b"\x00" * 44


class ForkWheelValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.wheel = (
            self.root / f"natten-{VERSION}-{PYTHON_TAG}-{PYTHON_TAG}-linux_x86_64.whl"
        )
        self.metadata_dir = f"natten-{VERSION}.dist-info"
        self.entries = {
            f"{self.metadata_dir}/METADATA": (
                f"Metadata-Version: 2.1\nName: natten\nVersion: {VERSION}\n"
            ).encode(),
            f"{self.metadata_dir}/WHEEL": (
                "Wheel-Version: 1.0\nRoot-Is-Purelib: false\n"
                f"Tag: {PYTHON_TAG}-{PYTHON_TAG}-linux_x86_64\n"
            ).encode(),
            "natten/__init__.py": b"# fixture package\n",
            "natten/version.py": f'__version__ = "{VERSION}"\n'.encode(),
            "natten/functional.py": b"# functional wrapper fixture\n",
            "natten/ops/forward.py": b"# nested wrapper fixture\n",
            NATIVE: ELF,
        }
        self.write_wheel()

    def write_wheel(self):
        with zipfile.ZipFile(self.wheel, "w") as archive:
            for member, contents in self.entries.items():
                archive.writestr(member, contents)

    def validate(self):
        return validator.validate_static(self.wheel, VERSION, PYTHON_TAG)

    def installed_context(self):
        site = self.root / "site-packages"
        for member, contents in self.entries.items():
            path = site / member
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        dist = SimpleNamespace(
            version=VERSION, locate_file=lambda member: site / member
        )
        self.package = SimpleNamespace(
            __file__=str(site / "natten/__init__.py"),
            __version__=VERSION,
            HAS_LIBNATTEN=True,
        )
        self.native = SimpleNamespace(__file__=str(site / NATIVE))
        self.modules = {
            "torch": SimpleNamespace(),
            "natten": self.package,
            "natten.libnatten": self.native,
        }
        for patch in (
            mock.patch.object(
                validator.importlib.metadata, "distribution", return_value=dist
            ),
            mock.patch.object(validator.sysconfig, "get_path", return_value=str(site)),
            mock.patch.object(
                validator.importlib,
                "import_module",
                side_effect=self.modules.__getitem__,
            ),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        return site

    def test_static_check_identifies_native_artifact(self):
        result = self.validate()
        self.assertEqual(result["native_member"], NATIVE)
        self.assertEqual(result["sha256"], validator.sha256(self.wheel))
        self.assertEqual(result["validation"], "static")

    def test_missing_native_extension_fails(self):
        del self.entries[NATIVE]
        self.write_wheel()
        with self.assertRaisesRegex(ValueError, "one native"):
            self.validate()

    def test_invalid_native_binary_fails(self):
        self.entries[NATIVE] = b"not a compiled shared object"
        self.write_wheel()
        with self.assertRaisesRegex(ValueError, "ELF shared object"):
            self.validate()

    def test_metadata_version_mismatch_fails(self):
        key = f"{self.metadata_dir}/METADATA"
        self.entries[key] = self.entries[key].replace(VERSION.encode(), b"0.1")
        self.write_wheel()
        with self.assertRaisesRegex(ValueError, "METADATA"):
            self.validate()

    def test_expected_version_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "version"):
            validator.validate_static(self.wheel, "0.1", PYTHON_TAG)

    def test_wheel_metadata_tag_mismatch_fails(self):
        key = f"{self.metadata_dir}/WHEEL"
        self.entries[key] = self.entries[key].replace(b"linux_x86_64", b"linux_aarch64")
        self.write_wheel()
        with self.assertRaisesRegex(ValueError, "WHEEL tags"):
            self.validate()

    def test_native_abi_mismatch_fails(self):
        self.entries[NATIVE.replace(PYTHON_TAG[2:], "999")] = self.entries.pop(NATIVE)
        self.write_wheel()
        with self.assertRaisesRegex(ValueError, "native extension ABI"):
            self.validate()

    def test_installed_native_matches_wheel(self):
        self.installed_context()
        result = validator.validate_installed(self.wheel, VERSION, PYTHON_TAG)
        self.assertEqual(result["validation"], "native-import")

    def test_stub_package_is_rejected(self):
        self.installed_context()
        self.package.HAS_LIBNATTEN = False
        with self.assertRaisesRegex(ValueError, "stub kernels"):
            validator.validate_installed(self.wheel, VERSION, PYTHON_TAG)

    def test_native_import_error_propagates(self):
        self.installed_context()

        def import_module(name):
            if name == "natten.libnatten":
                raise ImportError("undefined native symbol")
            return self.modules[name]

        with mock.patch.object(
            validator.importlib, "import_module", side_effect=import_module
        ):
            with self.assertRaisesRegex(ImportError, "undefined native symbol"):
                validator.validate_installed(self.wheel, VERSION, PYTHON_TAG)

    def test_source_tree_shadowing_is_rejected(self):
        self.installed_context()
        self.package.__file__ = str(self.root / "src/natten/__init__.py")
        with self.assertRaisesRegex(ValueError, "shadowed"):
            validator.validate_installed(self.wheel, VERSION, PYTHON_TAG)

    def test_native_from_another_installation_is_rejected(self):
        self.installed_context()
        self.native.__file__ = str(self.root / "other/libnatten.so")
        with self.assertRaisesRegex(ValueError, "not from the installed"):
            validator.validate_installed(self.wheel, VERSION, PYTHON_TAG)

    def test_changed_installed_native_fails_hash_check(self):
        site = self.installed_context()
        (site / NATIVE).write_bytes(ELF + b"changed build")
        with self.assertRaisesRegex(ValueError, "installed native extension differs"):
            validator.validate_installed(self.wheel, VERSION, PYTHON_TAG)

    def test_changed_installed_python_file_fails(self):
        site = self.installed_context()
        (site / "natten/__init__.py").write_text("# different package\n")
        with self.assertRaisesRegex(ValueError, "installed natten/__init__.py differs"):
            validator.validate_installed(self.wheel, VERSION, PYTHON_TAG)

    def test_changed_installed_wrapper_fails(self):
        site = self.installed_context()
        (site / "natten/functional.py").write_text("# changed wrapper\n")
        with self.assertRaisesRegex(
            ValueError, "installed natten/functional.py differs"
        ):
            validator.validate_installed(self.wheel, VERSION, PYTHON_TAG)

    def test_changed_nested_wrapper_fails(self):
        site = self.installed_context()
        (site / "natten/ops/forward.py").write_text("# changed nested wrapper\n")
        with self.assertRaisesRegex(
            ValueError, "installed natten/ops/forward.py differs"
        ):
            validator.validate_installed(self.wheel, VERSION, PYTHON_TAG)

    def test_extra_installed_python_module_is_rejected(self):
        site = self.installed_context()
        (site / "natten/ops/stale.py").write_text("# stale module from another build\n")
        with self.assertRaisesRegex(ValueError, "extra=.*natten/ops/stale.py"):
            validator.validate_installed(self.wheel, VERSION, PYTHON_TAG)

    def test_missing_installed_wrapper_is_rejected(self):
        site = self.installed_context()
        (site / "natten/functional.py").unlink()
        with self.assertRaisesRegex(ValueError, "missing=.*natten/functional.py"):
            validator.validate_installed(self.wheel, VERSION, PYTHON_TAG)

    def test_python_cache_directory_does_not_change_module_inventory(self):
        site = self.installed_context()
        cache = site / "natten/__pycache__"
        cache.mkdir()
        (cache / "debug.py").write_text("# cache-side diagnostic\n")
        (cache / "functional.cpython-311.pyc").write_bytes(b"cached bytecode")
        result = validator.validate_installed(self.wheel, VERSION, PYTHON_TAG)
        self.assertEqual(result["validation"], "native-import")

    def test_wrapper_symlink_outside_package_is_rejected(self):
        site = self.installed_context()
        wrapper = site / "natten/functional.py"
        outside = self.root / "external-wrapper.py"
        outside.write_bytes(wrapper.read_bytes())
        wrapper.unlink()
        wrapper.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "outside the natten package"):
            validator.validate_installed(self.wheel, VERSION, PYTHON_TAG)

    def test_interpreter_python_tag_mismatch_fails(self):
        result = self.validate()
        with mock.patch.object(validator, "validate_static", return_value=result):
            with self.assertRaisesRegex(ValueError, "running interpreter"):
                validator.validate_installed(self.wheel, VERSION, "cp999")

    def test_invalid_wheel_cli_exits_nonzero(self):
        del self.entries[NATIVE]
        self.write_wheel()
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                str(SCRIPT),
                "static",
                str(self.wheel),
                "--version",
                VERSION,
                "--python-tag",
                PYTHON_TAG,
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("one native", result.stderr)


if __name__ == "__main__":
    unittest.main()

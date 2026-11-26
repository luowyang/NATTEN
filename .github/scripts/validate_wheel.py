#!/usr/bin/env python3
"""Validate a built wheel and, separately, its installed native extension."""

import argparse
import email.parser
import hashlib
import importlib
import importlib.metadata
import json
import re
import sys
import sysconfig
import zipfile
from pathlib import Path

from packaging.tags import parse_tag
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_static(wheel, version, python_tag):
    wheel = Path(wheel)
    name, wheel_version, _, tags = parse_wheel_filename(wheel.name)
    if canonicalize_name(name) != "natten" or wheel_version != Version(version):
        raise ValueError("wheel name or version does not match the build")
    if not tags or any(
        tag.interpreter != python_tag
        or tag.abi != python_tag
        or not tag.platform.endswith("x86_64")
        or not tag.platform.startswith(("linux_", "manylinux"))
        for tag in tags
    ):
        raise ValueError("wheel tags do not match the requested Python/Linux ABI")

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("wheel has duplicate archive members")
        metadata_paths = [p for p in names if p.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError("wheel must contain exactly one METADATA file")
        metadata_path = metadata_paths[0]
        metadata = email.parser.BytesParser().parsebytes(archive.read(metadata_path))
        if (
            canonicalize_name(metadata["Name"] or "") != "natten"
            or Version(metadata["Version"] or "0") != wheel_version
        ):
            raise ValueError("METADATA name/version differs from wheel filename")
        wheel_metadata = email.parser.BytesParser().parsebytes(
            archive.read(metadata_path.rsplit("/", 1)[0] + "/WHEEL")
        )
        metadata_tags = set()
        for tag in wheel_metadata.get_all("Tag", []):
            metadata_tags.update(parse_tag(tag))
        if metadata_tags != tags:
            raise ValueError("WHEEL tags differ from wheel filename")
        if wheel_metadata["Root-Is-Purelib"] != "false":
            raise ValueError("native wheel cannot be marked pure Python")

        native_members = [
            p
            for p in names
            if re.fullmatch(r"natten/libnatten\.cpython-\d+-x86_64-linux-gnu\.so", p)
        ]
        if len(native_members) != 1:
            raise ValueError("wheel must contain one native libnatten extension")
        native_member = native_members[0]
        if f".cpython-{python_tag[2:]}-" not in native_member:
            raise ValueError("native extension ABI differs from wheel Python tag")
        with archive.open(native_member) as stream:
            header = stream.read(64)
            # ELF64, little endian, x86-64 shared object.
            if (
                len(header) != 64
                or header[:6] != b"\x7fELF\x02\x01"
                or header[16:20] != b"\x03\x00\x3e\x00"
            ):
                raise ValueError("libnatten is not an x86-64 ELF shared object")
            digest = hashlib.sha256(header)
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        for required in ("natten/__init__.py", "natten/version.py"):
            if required not in names:
                raise ValueError(f"wheel is missing {required}")

    return {
        "wheel": wheel.name,
        "sha256": sha256(wheel),
        "version": str(wheel_version),
        "python_tag": python_tag,
        "native_member": native_member,
        "native_sha256": digest.hexdigest(),
        "validation": "static",
    }


def validate_installed(wheel, version, python_tag):
    result = validate_static(wheel, version, python_tag)
    runtime_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if python_tag != runtime_tag:
        raise ValueError("wheel Python tag differs from the running interpreter")
    dist = importlib.metadata.distribution("natten")
    if Version(dist.version) != Version(version):
        raise ValueError("installed distribution version differs from the wheel")
    package_path = Path(dist.locate_file("natten/__init__.py")).resolve()
    install_roots = {
        Path(sysconfig.get_path(key)).resolve() for key in ("purelib", "platlib")
    }
    if package_path.parent.parent not in install_roots:
        raise ValueError("natten is not installed in this interpreter's site-packages")

    # Torch loads its shared libraries before the extension is imported. No CUDA
    # device operation occurs here. ImportError must propagate, never become a
    # successful check of natten's pure-Python stubs.
    importlib.import_module("torch")
    natten = importlib.import_module("natten")
    if Path(natten.__file__).resolve() != package_path:
        raise ValueError("natten import is shadowed by a different source tree")
    native = importlib.import_module("natten.libnatten")
    if not natten.HAS_LIBNATTEN:
        raise ValueError("natten loaded stub kernels instead of the native extension")
    if Version(natten.__version__) != Version(version):
        raise ValueError("imported package version differs from the wheel")
    native_path = Path(dist.locate_file(result["native_member"])).resolve()
    if Path(native.__file__).resolve() != native_path:
        raise ValueError("native import is not from the installed distribution")
    if native_path.parent != package_path.parent:
        raise ValueError("native extension is outside the installed natten package")
    if sha256(native_path) != result["native_sha256"]:
        raise ValueError("installed native extension differs from the wheel")
    with zipfile.ZipFile(wheel) as archive:
        python_members = {
            member
            for member in archive.namelist()
            if member.startswith("natten/") and member.endswith(".py")
        }
        package_root = package_path.parent
        installed_members = {
            "natten/" + path.relative_to(package_root).as_posix()
            for path in package_root.rglob("*.py")
            if "__pycache__" not in path.relative_to(package_root).parts
        }
        if installed_members != python_members:
            missing = sorted(python_members - installed_members)
            extra = sorted(installed_members - python_members)
            raise ValueError(
                f"installed Python module inventory differs: missing={missing}, extra={extra}"
            )
        for member in sorted(python_members):
            installed = Path(dist.locate_file(member)).resolve()
            if not installed.is_relative_to(package_root):
                raise ValueError(f"installed {member} is outside the natten package")
            if installed.read_bytes() != archive.read(member):
                raise ValueError(f"installed {member} differs from the wheel")
    result.update(
        validation="native-import",
        package_path=str(package_path),
        native_path=str(native_path),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("static", "installed"))
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--python-tag", required=True)
    args = parser.parse_args()
    validate = validate_static if args.mode == "static" else validate_installed
    print(
        json.dumps(validate(args.wheel, args.version, args.python_tag), sort_keys=True)
    )


if __name__ == "__main__":
    main()

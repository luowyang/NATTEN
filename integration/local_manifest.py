"""Record provenance for an unpublished local wheel build."""

import hashlib
import json
import sys
import time
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile


def main():
    wheel, output = map(Path, sys.argv[1:3])
    source_sha, recipe_sha = sys.argv[3:5]
    with ZipFile(wheel) as archive:
        metadata = [p for p in archive.namelist() if p.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("Expected one wheel METADATA file")
        version = BytesParser().parsebytes(archive.read(metadata[0]))["Version"]
    hasher = hashlib.sha256()
    with wheel.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    manifest = {
        "schema_version": 1,
        "source_sha": source_sha,
        "recipe_sha": recipe_sha,
        "build_id": f"local-{time.time_ns()}",
        "channel": "local",
        "version": version,
        "python_tag": wheel.name.split("-")[-3],
        "wheel": wheel.name,
        "sha256": hasher.hexdigest(),
        "host_validation": "not-run",
    }
    with output.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()

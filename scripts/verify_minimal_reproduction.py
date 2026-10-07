#!/usr/bin/env python3
"""Verify the bundled minimal data manifest and parse each archive safely."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "figures" / "minimal_verification_manifest.json"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0.0":
        raise ValueError("Unsupported minimal verification manifest version")

    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Size mismatch: {entry['path']}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {entry['path']}")
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if not archive.files:
                    raise ValueError(f"Empty NPZ archive: {entry['path']}")

    print(f"Verified {len(manifest['files'])} minimal verification files.")


if __name__ == "__main__":
    main()

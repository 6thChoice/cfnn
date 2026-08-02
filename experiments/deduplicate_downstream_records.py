"""Remove scientifically identical retry duplicates while preserving raw evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


VOLATILE_KEYS = {
    "device",
    "peak_memory_bytes",
    "total_wall_seconds",
    "training_wall_seconds",
    "wall_seconds",
}
KEY_FIELDS = ("task", "family", "target_budget", "data_seed", "init_seed")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scientific_payload(value):
    if isinstance(value, dict):
        return {
            key: _scientific_payload(item)
            for key, item in value.items()
            if key not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [_scientific_payload(item) for item in value]
    return value


def _fingerprint(record: dict) -> str:
    payload = json.dumps(
        _scientific_payload(record), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _record_key(record: dict) -> tuple:
    return tuple(record[field] for field in KEY_FIELDS)


def deduplicate_evaluation_root(
    root: Path,
    *,
    audit_name: str = "deduplication_audit.json",
) -> dict:
    """Back up and rewrite JSONL files only when retry payloads agree."""
    root = Path(root)
    invalidated = root / "invalidated" / "retry_duplicates"
    planned = []
    removed = []
    for path in sorted((root / "evaluation").rglob("*.jsonl")):
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        records = [json.loads(line) for line in lines]
        first_by_key = {}
        retained = []
        file_removed = []
        for line_number, record in enumerate(records, 1):
            key = _record_key(record)
            fingerprint = _fingerprint(record)
            if key not in first_by_key:
                first_by_key[key] = (fingerprint, line_number)
                retained.append(record)
                continue
            first_fingerprint, first_line = first_by_key[key]
            if fingerprint != first_fingerprint:
                raise ValueError(
                    "scientifically different duplicate retry for "
                    f"{key} in {path} (lines {first_line} and {line_number})"
                )
            item = {
                "key": dict(zip(KEY_FIELDS, key)),
                "path": str(path),
                "retained_line": first_line,
                "removed_line": line_number,
                "scientific_sha256": fingerprint,
            }
            file_removed.append(item)
            removed.append(item)
        if file_removed:
            planned.append((path, retained, file_removed))

    backups = []
    for path, retained, file_removed in planned:
        relative = path.relative_to(root)
        backup = invalidated / relative
        if backup.exists():
            raise ValueError(f"refusing to overwrite retry backup: {backup}")
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        source_sha256 = _sha256(backup)
        temporary = path.with_suffix(path.suffix + ".deduplicating")
        temporary.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in retained)
        )
        temporary.replace(path)
        backups.append({
            "source_path": str(path),
            "backup_path": str(backup),
            "source_sha256": source_sha256,
            "cleaned_sha256": _sha256(path),
            "retained_record_count": len(retained),
            "removed_record_count": len(file_removed),
        })

    audit = {
        "policy": "retain_first_scientifically_identical_retry",
        "ignored_volatile_keys": sorted(VOLATILE_KEYS),
        "removed_record_count": len(removed),
        "removed": removed,
        "backups": backups,
    }
    invalidated.mkdir(parents=True, exist_ok=True)
    (invalidated / audit_name).write_text(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--audit-name", default="deduplication_audit.json")
    args = parser.parse_args()
    print(json.dumps(
        deduplicate_evaluation_root(args.root, audit_name=args.audit_name),
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()

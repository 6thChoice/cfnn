"""Create and verify the Task9 production preflight freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unittest
from datetime import datetime, timezone
from pathlib import Path

from peak_sensitive_benchmark import (
    PROVENANCE_ROOT,
    ROOT,
    _canonical_bytes,
    _config_hash,
    _file_hash,
    _protocol_manifest,
    _role_manifest_paths,
    _sha256_bytes,
)


FREEZE_FILENAME = "preflight_freeze.json"
FREEZE_SCHEMA_VERSION = "task9-preflight-freeze-v1"
CANONICAL_PREFLIGHT_TEST_MODULES = (
    "test_peak_sensitive_metrics",
    "test_peak_sensitive_protocol",
    "test_peak_sensitive_benchmark",
    "test_peak_sensitive_artifacts",
    "test_downstream_protocol",
    "test_downstream_benchmark",
    "test_downstream_domain_baselines",
)
CANONICAL_PREFLIGHT_TEST_COMMAND = "python -m unittest " + " ".join(
    CANONICAL_PREFLIGHT_TEST_MODULES
)
CANONICAL_PREFLIGHT_TEST_LOG_HEADER = (
    "TASK9_PREFLIGHT_CANONICAL_COMMAND: " + CANONICAL_PREFLIGHT_TEST_COMMAND
)
_IDENTITY_FIELDS = (
    "protocol_key", "protocol_sha256", "config_sha256", "protocol_mode",
    "smoke_root_marker", "code_files", "input_files", "raw_root",
)
_REQUIRED_FILE_LABELS = {
    "config", "schema", "raw_source_manifest", "raw_source_audit",
    "candidate_inventory", "protocol_manifest", "environment_log", "preflight_test_log",
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _scope_from_values(tasks: list[str], families: list[str], budgets: list[int],
                       frf_densities: list[int]) -> dict:
    return {
        "tasks": list(tasks),
        "families": list(families),
        "budgets": [int(value) for value in budgets],
        "frf_densities": [int(value) for value in frf_densities],
    }


def _frozen_config_scope(config: dict) -> dict:
    return _scope_from_values(
        config["fixed_scientific_tasks"], config["families"], config["budgets"],
        config["frf_densities"],
    )


def _require_full_frozen_scope(config: dict, scope: dict) -> dict:
    expected_scope = _frozen_config_scope(config)
    if scope != expected_scope:
        raise ValueError("preflight freeze scope must exactly match frozen config axes")
    return expected_scope


def canonical_preflight_test_log_header() -> str:
    return CANONICAL_PREFLIGHT_TEST_LOG_HEADER


def _expected_preflight_test_inventory() -> dict:
    suite = unittest.defaultTestLoader.loadTestsFromNames(
        CANONICAL_PREFLIGHT_TEST_MODULES
    )
    return {
        "test_modules": list(CANONICAL_PREFLIGHT_TEST_MODULES),
        "canonical_command": CANONICAL_PREFLIGHT_TEST_COMMAND,
        "tests_run": suite.countTestCases(),
    }


def _freeze_path(output_root: Path) -> Path:
    return Path(output_root) / FREEZE_FILENAME


def _file_record(label: str, path: Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"missing preflight prerequisite: {label}: {path}")
    return {"label": label, "path": str(path), "sha256": _file_hash(path)}


def _protocol_identity(manifest: dict) -> dict:
    return {field: manifest.get(field) for field in _IDENTITY_FIELDS}


def _expected_task_shards(scope: dict) -> list[dict]:
    return [
        {"task": task, "frf_density": density}
        for task in scope["tasks"]
        for density in (scope["frf_densities"] if task == "aluminium_frf" else [None])
    ]


def _candidate_counts(inventory: dict, scope: dict) -> dict:
    if inventory.get("scope") != scope:
        raise ValueError("incomplete candidate inventory scope: scope differs from preflight scope")
    if inventory.get("task_shards") != _expected_task_shards(scope):
        raise ValueError("incomplete candidate inventory scope: task shards differ from preflight scope")
    tasks = inventory.get("tasks")
    eligibility = inventory.get("family_eligibility")
    if not isinstance(tasks, dict) or set(tasks) != set(scope["tasks"]):
        raise ValueError("incomplete candidate inventory scope: task set is incomplete")
    if not isinstance(eligibility, dict) or set(eligibility) != set(scope["families"]):
        raise ValueError("incomplete candidate inventory scope: family eligibility is incomplete")

    cells = []
    total = 0
    for task in scope["tasks"]:
        families = tasks.get(task)
        if not isinstance(families, dict) or set(families) != set(scope["families"]):
            raise ValueError(f"incomplete candidate inventory scope: families missing for {task}")
        for family in scope["families"]:
            budgets = families.get(family)
            expected_budgets = {str(int(budget)) for budget in scope["budgets"]}
            if not isinstance(budgets, dict) or set(budgets) != expected_budgets:
                raise ValueError(f"incomplete candidate inventory scope: budgets missing for {task}/{family}")
            for budget in scope["budgets"]:
                candidates = budgets[str(int(budget))]
                if not isinstance(candidates, list):
                    raise ValueError(f"incomplete candidate inventory scope: invalid candidates for {task}/{family}/{budget}")
                if eligibility[family].get("status") == "eligible" and not candidates:
                    raise ValueError(f"incomplete candidate inventory scope: no candidates for {task}/{family}/{budget}")
                count = len(candidates)
                total += count
                cells.append({
                    "task": task, "family": family, "budget": int(budget), "count": count,
                })
    return {
        "scope_sha256": _sha256_bytes(_canonical_bytes(scope)),
        "task_shard_count": len(_expected_task_shards(scope)),
        "candidate_cell_count": len(cells),
        "candidate_total": total,
        "by_task_family_budget": cells,
    }


def _parse_preflight_test_log(path: Path) -> dict:
    record = _file_record("preflight_test_log", path)
    text = Path(path).read_text()
    nonempty = [line for line in text.splitlines() if line.strip()]
    expected = _expected_preflight_test_inventory()
    expected_run_line = rf"Ran {expected['tests_run']} tests? in .+"
    if (
        len(nonempty) < 3
        or nonempty[0] != CANONICAL_PREFLIGHT_TEST_LOG_HEADER
        or nonempty.count(CANONICAL_PREFLIGHT_TEST_LOG_HEADER) != 1
    ):
        raise ValueError("preflight test log lacks the canonical command marker")
    if not re.fullmatch(expected_run_line, nonempty[-2]) or nonempty[-1] != "OK":
        raise ValueError("preflight test log lacks final zero-failure evidence")
    if re.search(r"^FAILED", text, flags=re.MULTILINE):
        raise ValueError("preflight test log reports failures")
    return {
        "log_path": record["path"], "log_sha256": record["sha256"],
        "final_status": "OK", "tests_run": expected["tests_run"], "failures": 0, "errors": 0,
        "test_modules": expected["test_modules"],
        "canonical_command": expected["canonical_command"],
    }


def _scope_is_covered(requested_scope: dict, frozen_scope: dict) -> bool:
    required = {"tasks", "families", "budgets", "frf_densities"}
    if not isinstance(requested_scope, dict) or not isinstance(frozen_scope, dict):
        return False
    if set(requested_scope) != required or set(frozen_scope) != required:
        return False
    if any(not isinstance(requested_scope[key], list) or not isinstance(frozen_scope[key], list)
           for key in required):
        return False
    return all(
        set(requested_scope[key]).issubset(set(frozen_scope[key]))
        for key in required
    )


def _load_expected_protocol(config_path: Path, schema_path: Path, raw_root: Path, scope: dict) -> tuple[dict, dict]:
    config = json.loads(Path(config_path).read_text())
    if config.get("protocol_mode", "production") != "production":
        raise ValueError("preflight freezes are production-only")
    manifest = _protocol_manifest(
        config, Path(config_path), Path(schema_path), ["preflight-freeze"],
        raw_root=Path(raw_root), tasks=scope["tasks"],
    )
    return config, manifest


def _required_file_records(config_path: Path, schema_path: Path, raw_root: Path, output_root: Path,
                           tests_log: Path, environment_log: Path) -> list[dict]:
    records = [
        _file_record("config", config_path),
        _file_record("schema", schema_path),
        _file_record("raw_source_manifest", Path(raw_root) / "source_manifest.json"),
        _file_record("raw_source_audit", PROVENANCE_ROOT / "source_audit.json"),
        _file_record("candidate_inventory", Path(output_root) / "candidate_manifest.json"),
        _file_record("protocol_manifest", Path(output_root) / "protocol_manifest.json"),
        _file_record("environment_log", environment_log),
        _file_record("preflight_test_log", tests_log),
    ]
    records.extend(
        _file_record(f"role_manifest:{path.name}", path) for path in _role_manifest_paths()
    )
    return records


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(_canonical_json(value) + "\n")
    temporary.replace(path)


def create_preflight_freeze(*, config_path: Path, schema_path: Path, raw_root: Path, output_root: Path,
                            scope: dict, tests_log: Path, environment_log: Path) -> dict:
    """Record a hash-complete production preflight after tests and inventory succeed."""
    config = json.loads(Path(config_path).read_text())
    scope = _require_full_frozen_scope(config, scope)
    config, expected_manifest = _load_expected_protocol(config_path, schema_path, raw_root, scope)
    protocol_path = Path(output_root) / "protocol_manifest.json"
    inventory_path = Path(output_root) / "candidate_manifest.json"
    if not protocol_path.is_file() or not inventory_path.is_file():
        raise ValueError("missing inventory or protocol manifest required for preflight freeze")
    actual_manifest = json.loads(protocol_path.read_text())
    if _protocol_identity(actual_manifest) != _protocol_identity(expected_manifest):
        raise ValueError("protocol manifest does not match current transitive code and input hashes")
    inventory = json.loads(inventory_path.read_text())
    if inventory.get("protocol_key") != config["protocol_key"]:
        raise ValueError("candidate inventory protocol key mismatch")
    if inventory.get("config_sha256") != _config_hash(config):
        raise ValueError("candidate inventory config SHA-256 mismatch")
    if inventory.get("protocol_sha256") != expected_manifest["protocol_sha256"]:
        raise ValueError("candidate inventory protocol SHA-256 mismatch")
    counts = _candidate_counts(inventory, scope)
    test_evidence = _parse_preflight_test_log(tests_log)
    files = _required_file_records(
        config_path, schema_path, raw_root, output_root, tests_log, environment_log,
    )
    labels = {record["label"] for record in files}
    if not _REQUIRED_FILE_LABELS.issubset(labels):
        raise ValueError("preflight freeze is missing required hash records")
    freeze = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "protocol": {
            "identity": _protocol_identity(expected_manifest),
            "protocol_manifest_sha256": _file_hash(protocol_path),
        },
        "config": {
            "path": str(Path(config_path).resolve()),
            "sha256": _file_hash(config_path),
        },
        "schema": {
            "path": str(Path(schema_path).resolve()),
            "sha256": _file_hash(schema_path),
        },
        "candidate_inventory": {
            "path": str(inventory_path.resolve()), "sha256": _file_hash(inventory_path),
            "scope": scope, "counts": counts,
        },
        "preflight_tests": test_evidence,
        "environment": {
            "log_path": str(Path(environment_log).resolve()),
            "log_sha256": _file_hash(environment_log),
            "hardware": expected_manifest.get("hardware"),
            "platform": expected_manifest.get("platform"),
            "python_version": expected_manifest.get("python_version"),
            "torch_version": expected_manifest.get("torch_version"),
        },
        "files": files,
    }
    _atomic_write_json(_freeze_path(output_root), freeze)
    return freeze


def verify_preflight_freeze(*, config_path: Path, schema_path: Path, raw_root: Path, output_root: Path,
                            requested_scope: dict) -> dict:
    """Recompute every frozen prerequisite and reject stale or incomplete production state."""
    path = _freeze_path(output_root)
    if not path.is_file():
        raise ValueError(f"missing preflight freeze: {path}")
    freeze = json.loads(path.read_text())
    if freeze.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise ValueError("unsupported or incomplete preflight freeze")
    frozen_scope = freeze.get("candidate_inventory", {}).get("scope")
    config = json.loads(Path(config_path).read_text())
    expected_scope = _frozen_config_scope(config)
    if not isinstance(frozen_scope, dict) or frozen_scope != expected_scope:
        raise ValueError("preflight freeze does not cover requested production scope")
    if not _scope_is_covered(requested_scope, frozen_scope):
        raise ValueError("preflight freeze does not cover requested production scope")
    config, expected_manifest = _load_expected_protocol(config_path, schema_path, raw_root, frozen_scope)
    protocol_path = Path(output_root) / "protocol_manifest.json"
    inventory_path = Path(output_root) / "candidate_manifest.json"
    if not protocol_path.is_file() or not inventory_path.is_file():
        raise ValueError("missing protocol manifest or candidate inventory required by preflight freeze")
    actual_manifest = json.loads(protocol_path.read_text())
    frozen_identity = freeze.get("protocol", {}).get("identity")
    if _protocol_identity(expected_manifest) != frozen_identity:
        raise ValueError("preflight freeze protocol or transitive code SHA-256 mismatch")
    if _protocol_identity(actual_manifest) != frozen_identity:
        raise ValueError("protocol manifest no longer matches preflight freeze")
    if freeze.get("protocol", {}).get("protocol_manifest_sha256") != _file_hash(protocol_path):
        raise ValueError("protocol manifest SHA-256 mismatch")
    if freeze.get("config", {}).get("sha256") != _file_hash(config_path):
        raise ValueError("config SHA-256 mismatch")
    if freeze.get("schema", {}).get("sha256") != _file_hash(schema_path):
        raise ValueError("schema SHA-256 mismatch")
    inventory = json.loads(inventory_path.read_text())
    if inventory.get("config_sha256") != _config_hash(config):
        raise ValueError("candidate inventory config SHA-256 mismatch")
    if inventory.get("protocol_sha256") != expected_manifest["protocol_sha256"]:
        raise ValueError("candidate inventory protocol SHA-256 mismatch")
    if freeze["candidate_inventory"].get("sha256") != _file_hash(inventory_path):
        raise ValueError("candidate inventory SHA-256 mismatch")
    counts = _candidate_counts(inventory, frozen_scope)
    if counts != freeze["candidate_inventory"].get("counts"):
        raise ValueError("candidate inventory counts or scope mismatch")
    records = freeze.get("files")
    if not isinstance(records, list) or not _REQUIRED_FILE_LABELS.issubset(
        {record.get("label") for record in records if isinstance(record, dict)}
    ):
        raise ValueError("incomplete preflight freeze hash records")
    for record in records:
        if not isinstance(record, dict) or not {"label", "path", "sha256"}.issubset(record):
            raise ValueError("incomplete preflight freeze hash record")
        current = _file_record(record["label"], Path(record["path"]))
        if current["sha256"] != record["sha256"]:
            if record["label"] == "candidate_inventory":
                raise ValueError("candidate inventory SHA-256 mismatch")
            raise ValueError(f"preflight {record['label']} SHA-256 mismatch")
    evidence = freeze.get("preflight_tests", {})
    parsed = _parse_preflight_test_log(Path(evidence.get("log_path", "")))
    if parsed != evidence:
        raise ValueError("preflight test log evidence mismatch")
    return freeze


def _filter(values: list, requested: str | None, cast=str) -> list:
    if not requested:
        return list(values)
    selected = [cast(value.strip()) for value in requested.split(",") if value.strip()]
    unknown = set(selected) - set(values)
    if unknown:
        raise ValueError(f"unknown requested values: {sorted(unknown)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("create", "verify", "test-log-header"))
    parser.add_argument("--config", type=Path, default=ROOT / "peak_sensitive_config.json")
    parser.add_argument("--schema", type=Path, default=ROOT / "peak_sensitive_results_schema.json")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "downstream_data" / "raw")
    parser.add_argument("--output-root", type=Path, default=ROOT / "peak_sensitive_results")
    parser.add_argument("--tests-log", type=Path)
    parser.add_argument("--environment-log", type=Path)
    parser.add_argument("--tasks")
    parser.add_argument("--families")
    parser.add_argument("--budgets")
    parser.add_argument("--densities")
    args = parser.parse_args()
    if args.command == "test-log-header":
        print(canonical_preflight_test_log_header())
        return
    config = json.loads(args.config.read_text())
    scope = _scope_from_values(
        _filter(config["fixed_scientific_tasks"], args.tasks),
        _filter(config["families"], args.families),
        _filter(config["budgets"], args.budgets, int),
        _filter(config["frf_densities"], args.densities, int),
    )
    if args.command == "create":
        provenance = args.output_root / "provenance"
        freeze = create_preflight_freeze(
            config_path=args.config, schema_path=args.schema, raw_root=args.raw_root,
            output_root=args.output_root, scope=scope,
            tests_log=args.tests_log or provenance / "preflight_tests.log",
            environment_log=args.environment_log or provenance / "preflight_environment.txt",
        )
    else:
        freeze = verify_preflight_freeze(
            config_path=args.config, schema_path=args.schema, raw_root=args.raw_root,
            output_root=args.output_root, requested_scope=scope,
        )
    print(_canonical_json({"status": "passed", "freeze": str(_freeze_path(args.output_root)),
                           "scope": freeze["candidate_inventory"]["scope"]}))


if __name__ == "__main__":
    main()

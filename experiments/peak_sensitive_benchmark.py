"""Frozen v2 runner for the peak-sensitive complex-response benchmark."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from scipy.signal import hilbert

from confirmatory_artifacts import build_run_manifest, freeze_budget_grid
from confirmatory_models import architecture_grid, build_model, enumerate_candidates
from confirmatory_protocol import DatasetBundle, seeded_build
from downstream_benchmark import (
    DEVICE,
    _compact_metadata,
    build_v2_validation_scorers,
    rank_v2_candidates,
    v2_candidate_rank_key,
)
from downstream_protocol import (
    load_microwave_fano,
    load_microstrip_resonator,
    make_controlled_fano_bundle,
)
from frontier_science.eis import load_figshare_eis_curves
from downstream_training import fit_task_model, prepare_bundle
from peak_sensitive_protocol import (
    make_frf_density_bundle,
    load_hybrid_supercapacitor_eis,
)
from peak_sensitive_result_validation import validate_peak_sensitive_record
from resonance_windows import (
    _complex_response,
    _match_features,
    _metric_axis,
    _metric_features,
    complex_window_metrics,
    derive_window_spec,
)
from budget_matching import Candidate, within_budget


ROOT = Path(__file__).resolve().parent
PROVENANCE_ROOT = ROOT / "peak_sensitive_results" / "provenance"
SMOKE_ROOT_MARKER = "task7-explicit-smoke-root-v1"
UNDERFIT_CRITERION = "validation_global_nrmse_ge_1.0"
STAGES = ("inventory", "tune", "freeze-budgets", "audit-tuning", "evaluate", "all")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _config_hash(config: dict) -> str:
    return _sha256_bytes(_canonical_bytes(config))


def _protocol_hash(config_path: Path, schema_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (Path(config_path), Path(schema_path)):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _json_normalized(value):
    return json.loads(json.dumps(_json_safe(value), sort_keys=True, allow_nan=False))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(_json_safe(value), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"corrupted JSONL {path} line {line_number}: {error.msg}") from error
        if not isinstance(value, dict):
            raise ValueError(f"corrupted JSONL {path} line {line_number}: record must be an object")
        records.append(value)
    return records


@contextmanager
def _jsonl_lock(path: Path):
    """Hold a per-shard advisory lock for read/check/train/append transactions."""
    lock_path = Path(f"{path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def locked_append_once(path: Path, record: dict, *, delay_seconds: float = 0.0) -> bool:
    """Append one unique record under the same transaction used by shard resume."""
    with _jsonl_lock(path):
        records = _load_jsonl(path)
        assert_unique_records(records)
        key = _record_key(record)
        if key in {_record_key(item) for item in records}:
            return False
        if delay_seconds:
            time.sleep(float(delay_seconds))
        _append_jsonl(path, record)
        return True


def _file_hash(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _inventory_path(output_root: Path) -> Path:
    return Path(output_root) / "candidate_manifest.json"


def _task_directory(task: str, frf_density: int | None = None) -> Path:
    directory = Path(_slug(task))
    if task == "aluminium_frf":
        if frf_density is None:
            raise ValueError("aluminium_frf shards require an explicit density")
        directory /= f"density_{int(frf_density)}"
    elif frf_density is not None:
        raise ValueError(f"density is not an axis for {task}")
    return directory


def _selection_path(output_root: Path, task: str, family: str, budget: int,
                    frf_density: int | None = None) -> Path:
    return Path(output_root) / "selections" / _task_directory(task, frf_density) / f"{_slug(family)}__{int(budget)}.json"


def _screen_path(output_root: Path, task: str, family: str, budget: int,
                 frf_density: int | None = None) -> Path:
    return Path(output_root) / "tuning_screen" / _task_directory(task, frf_density) / f"{_slug(family)}__{int(budget)}.jsonl"


def _final_path(output_root: Path, task: str, family: str, budget: int,
                frf_density: int | None = None) -> Path:
    return Path(output_root) / "tuning_final" / _task_directory(task, frf_density) / f"{_slug(family)}__{int(budget)}.jsonl"


def _evaluation_path(output_root: Path, task: str, family: str, budget: int,
                     frf_density: int | None = None) -> Path:
    return Path(output_root) / "evaluation" / _task_directory(task, frf_density) / f"{_slug(family)}__{int(budget)}.jsonl"


def _role_manifest_paths() -> list[Path]:
    return [
        PROVENANCE_ROOT / "microwave_fano_role_manifest.json",
        PROVENANCE_ROOT / "microstrip_resonator_role_manifest.json",
        PROVENANCE_ROOT / "frf_density_manifest.json",
        PROVENANCE_ROOT / "figshare_battery_eis_role_manifest.json",
        PROVENANCE_ROOT / "curve_manifest.json",
    ]


def _verified_consumed_inputs(raw_root: Path, tasks: list[str]) -> list[Path]:
    raw_root = Path(raw_root).resolve()
    source_manifest_path = raw_root / "source_manifest.json"
    inputs = [source_manifest_path]
    if not source_manifest_path.exists():
        raise FileNotFoundError(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text())
    source_rows = {row["path"]: row for row in source_manifest.get("files", [])}

    for task in tasks:
        if task in {"microwave_fano", "microstrip_resonator"}:
            role, _ = _role_manifest(task)
            path = raw_root / role["raw_relative_source_file"]
            if not path.exists() or _file_hash(path) != role["source_sha256"]:
                raise ValueError(f"{task} role-manifest source SHA-256 mismatch")
            inputs.append(path)
        elif task == "aluminium_frf":
            role, _ = _role_manifest(task)
            for row in role["inventory"]:
                path = raw_root / row["source_files"][0]
                source_row = source_rows.get(row["source_files"][0])
                if source_row is None or not path.exists() or _file_hash(path) != row["source_sha256"]:
                    raise ValueError("Aluminium FRF consumed source SHA-256 mismatch")
                inputs.append(path)
        elif task == "hybrid_supercapacitor_eis":
            role, _ = _role_manifest(task)
            rows = role["tuning"] + role["evaluation"]
            for relative in sorted({row["source_file"] for row in rows}):
                row = next(item for item in rows if item["source_file"] == relative)
                path = raw_root / relative
                if not path.exists() or _file_hash(path) != row["source_sha256"]:
                    raise ValueError("hybrid_supercapacitor_eis consumed source SHA-256 mismatch")
                inputs.append(path)
        elif task == "figshare_battery_eis":
            role, _ = _role_manifest(task)
            source_root = ROOT / role["source"]["raw_root"]
            for row in role["curves"].values():
                path = source_root / row["path"]
                if not path.exists() or _file_hash(path) != row["sha256"]:
                    raise ValueError("Figshare battery consumed source SHA-256 mismatch")
                inputs.append(path)
    return sorted(set(inputs), key=lambda path: str(path.resolve()))


def _protocol_manifest(config: dict, config_path: Path, schema_path: Path, command: list[str],
                       *, raw_root: Path, tasks: list[str]) -> dict:
    code_paths = [
        Path(__file__), ROOT / "peak_sensitive_protocol.py", ROOT / "resonance_windows.py",
        ROOT / "downstream_training.py", ROOT / "downstream_benchmark.py",
        ROOT / "downstream_protocol.py", ROOT / "downstream_metrics.py",
        ROOT / "confirmatory_models.py", ROOT / "confirmatory_protocol.py",
        ROOT / "confirmatory_artifacts.py", ROOT / "confirmatory_training.py",
        ROOT / "budget_matching.py", ROOT / "fair_baselines.py",
        ROOT / "cofrnet_variants.py", ROOT / "frontier_science" / "eis.py",
        ROOT / "frontier_science" / "protocol.py",
        ROOT / "peak_sensitive_result_validation.py",
        ROOT / "peak_sensitive_preflight.py",
        config_path, schema_path,
    ]
    manifest = build_run_manifest(config, command, code_paths)
    manifest["protocol_key"] = config["protocol_key"]
    manifest["protocol_sha256"] = _protocol_hash(config_path, schema_path)
    manifest["protocol_mode"] = config.get("protocol_mode", "production")
    manifest["smoke_root_marker"] = config.get("smoke_root_marker")
    consumed_inputs = _verified_consumed_inputs(raw_root, tasks)
    input_paths = _role_manifest_paths() + consumed_inputs
    manifest["input_files"] = [
        {"path": str(path.resolve()), "sha256": _file_hash(path)} for path in input_paths
    ]
    manifest["raw_root"] = {
        "canonical_path": str(Path(raw_root).resolve()),
        "source_manifest_sha256": _file_hash(Path(raw_root).resolve() / "source_manifest.json"),
        "consumed_file_sha256": [
            {"path": str(path.resolve()), "sha256": _file_hash(path)}
            for path in consumed_inputs
        ],
    }
    return manifest


def _manifest_tasks(config: dict, requested_tasks: list[str]) -> list[str]:
    if config.get("protocol_mode", "production") == "production":
        return list(config["fixed_scientific_tasks"])
    return list(requested_tasks)


def ensure_protocol_root(output_root: Path, manifest: dict) -> None:
    """Refuse mixed protocol artifacts before any stage appends a record."""
    path = Path(output_root) / "protocol_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("raw_root") != manifest.get("raw_root"):
            raise ValueError("output root contains different raw root identity or hashes")
        compared = (
            "protocol_key", "protocol_sha256", "config_sha256", "protocol_mode",
            "smoke_root_marker", "code_files", "input_files",
        )
        if any(existing.get(key) != manifest.get(key) for key in compared):
            raise ValueError("output root contains different config, code, or input hashes")
        return
    _write_json(path, manifest)


def _candidate_grid(family: str, input_dim: int, output_dim: int):
    return enumerate_candidates(
        family, input_dim, output_dim,
        protocol_grid="v2" if family == "CFNN" else None,
    )


def _smoke_candidate_grid(family: str, input_dim: int, output_dim: int, *,
                          budgets: list[int], tolerance: float,
                          maximum_per_budget: int = 3) -> list[Candidate]:
    """Stream a small eligible prefix without materializing the production grid."""
    matched = {int(budget): 0 for budget in budgets}
    candidates: list[Candidate] = []
    protocol_grid = "v2" if family == "CFNN" else None
    for model_config in architecture_grid(family, protocol_grid=protocol_grid):
        model, _ = build_model(family, input_dim, output_dim, model_config, seed=0)
        candidate = Candidate(
            json.dumps(model_config, sort_keys=True, separators=(",", ":")),
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            model_config,
        )
        eligible = [
            int(budget) for budget in budgets
            if matched[int(budget)] < maximum_per_budget
            and within_budget(candidate.actual_parameters, int(budget), tolerance)
        ]
        if not eligible:
            continue
        candidates.append(candidate)
        for budget in eligible:
            matched[budget] += 1
        if all(count >= maximum_per_budget for count in matched.values()):
            break
    return candidates


def build_inventory(config: dict, output_root: Path, *, tasks: list[str],
                    families: list[str], budgets: list[int],
                    frf_densities: list[int] | None = None, smoke: bool = False,
                    protocol_sha256: str | None = None) -> dict:
    """Freeze candidate grids, recording unavailable optional dependencies explicitly."""
    inventory = {
        "protocol_key": config["protocol_key"],
        "protocol_mode": config.get("protocol_mode", "production"),
        "protocol_sha256": protocol_sha256,
        "config_sha256": _config_hash(config),
        "scope": {
            "tasks": list(tasks), "families": list(families), "budgets": [int(item) for item in budgets],
            "frf_densities": [int(item) for item in (frf_densities or config["frf_densities"])],
        },
        "task_shards": [
            {"task": task, "frf_density": density}
            for task in tasks
            for density in (
                (frf_densities or config["frf_densities"])
                if task == "aluminium_frf" else [None]
            )
        ],
        "tasks": {},
        "family_eligibility": {},
    }
    cache: dict[tuple[str, int, int], list] = {}
    for task in tasks:
        spec = config["tasks"][task]
        inventory["tasks"][task] = {}
        for family in families:
            key = (family, int(spec["input_dim"]), int(spec["output_dim"]))
            try:
                if key not in cache:
                    cache[key] = (
                        _smoke_candidate_grid(
                            family, key[1], key[2], budgets=budgets,
                            tolerance=float(config["budget_tolerance"]),
                        )
                        if smoke else _candidate_grid(family, key[1], key[2])
                    )
                frozen = freeze_budget_grid(
                    cache[key], budgets, tolerance=float(config["budget_tolerance"]),
                    minimum_per_budget=1 if smoke else int(config["minimum_candidates_per_budget"]),
                    maximum_per_budget=3 if smoke else int(config["maximum_candidates_per_budget"]),
                )
                inventory["tasks"][task][family] = frozen
                inventory["family_eligibility"].setdefault(family, {
                    "status": "eligible", "reason": None,
                })
            except (ImportError, ModuleNotFoundError) as error:
                # PyKAN is optional. Inventory must be factual rather than inventing candidates.
                inventory["tasks"][task][family] = {str(budget): [] for budget in budgets}
                inventory["family_eligibility"][family] = {
                    "status": "missing_dependency", "reason": str(error),
                }
    _write_json(_inventory_path(output_root), inventory)
    return inventory


def _seal_bundle(bundle: DatasetBundle) -> DatasetBundle:
    return DatasetBundle(
        x_train=bundle.x_train, y_train=bundle.y_train,
        x_validation=bundle.x_validation, y_validation=bundle.y_validation,
        x_test=np.empty((0, bundle.x_train.shape[1]), dtype=np.float32),
        y_test=np.empty((0, bundle.y_train.shape[1]), dtype=np.float32),
        metadata={**bundle.metadata, "test_access": "sealed_for_validation_only_stage"},
    )


def _split_curve(curve, task: str, data_seed: int, spec: dict, *,
                 scientific_role: str, visibility: str) -> DatasetBundle:
    n_train, n_validation = int(spec["n_train"]), int(spec["n_validation"])
    if n_train + n_validation >= len(curve.frequency):
        raise ValueError(f"{task} split leaves no held-out test frequencies")
    order = np.random.default_rng(int(data_seed) + 32452843).permutation(len(curve.frequency))
    train, validation, test = np.sort(order[:n_train]), np.sort(order[n_train:n_train + n_validation]), np.sort(order[n_train + n_validation:])
    point_id = lambda index: f"{curve.curve_id}:frequency:{int(index)}"
    opened = visibility == "opened"
    return DatasetBundle(
        x_train=np.asarray(curve.frequency[train], dtype=np.float64).reshape(-1, 1),
        y_train=np.asarray(curve.response[train], dtype=np.float32),
        x_validation=np.asarray(curve.frequency[validation], dtype=np.float64).reshape(-1, 1),
        y_validation=np.asarray(curve.response[validation], dtype=np.float32),
        x_test=(
            np.asarray(curve.frequency[test], dtype=np.float64).reshape(-1, 1)
            if opened else np.empty((0, 1), dtype=np.float64)
        ),
        y_test=(
            np.asarray(curve.response[test], dtype=np.float32)
            if opened else np.empty((0, 2), dtype=np.float32)
        ),
        metadata={
            "task": task, "data_seed": int(data_seed), "curve_id": curve.curve_id,
            "scientific_role": scientific_role,
            "test_access": (
                "opened_confirmatory_test" if opened
                else "sealed_for_validation_only_stage"
            ),
            "conditions": curve.conditions, "source_files": list(curve.source_files),
            "curve_metadata": curve.metadata,
            "split_strategy": "within_curve_peak_agnostic_random_frequency_mask",
            "test_target": "held_out_measured_frequencies",
            "train_point_ids": [point_id(index) for index in train],
            "validation_point_ids": [point_id(index) for index in validation],
            "test_point_ids": [point_id(index) for index in test],
        },
    )


_ROLE_MANIFEST_NAMES = {
    "microwave_fano": "microwave_fano_role_manifest.json",
    "microstrip_resonator": "microstrip_resonator_role_manifest.json",
    "aluminium_frf": "frf_density_manifest.json",
    "figshare_battery_eis": "figshare_battery_eis_role_manifest.json",
    "hybrid_supercapacitor_eis": "curve_manifest.json",
}


def _role_manifest(task: str) -> tuple[dict, Path]:
    if task not in _ROLE_MANIFEST_NAMES:
        raise ValueError(f"task {task!r} has no measured role manifest")
    path = PROVENANCE_ROOT / _ROLE_MANIFEST_NAMES[task]
    manifest = json.loads(path.read_text())
    if manifest.get("protocol_key") != "cfnn-peak-sensitive-extension-2.0.0":
        raise ValueError(f"role manifest protocol mismatch for {task}")
    if manifest.get("task") != task:
        raise ValueError(f"role manifest task mismatch for {task}")
    return manifest, path


def _curve_id_for_seed(task: str, scientific_role: str, data_seed: int,
                       config: dict) -> tuple[str, dict, Path]:
    if scientific_role not in {"tuning", "evaluation"}:
        raise ValueError("scientific_role must be tuning or evaluation")
    seed_key = f"{scientific_role}_data_seeds"
    seeds = [int(value) for value in config[seed_key]]
    if int(data_seed) not in seeds:
        raise ValueError(f"data seed {data_seed} is not frozen for {scientific_role}")
    manifest, path = _role_manifest(task)
    mapping = manifest.get("seed_curve_id_map", {}).get(scientific_role, {})
    expected_keys = {str(seed) for seed in seeds}
    if set(mapping) != expected_keys:
        raise ValueError(f"role manifest seed scope mismatch for {task}/{scientific_role}")
    curve_ids = [mapping[str(seed)] for seed in seeds]
    if len(curve_ids) != len(set(curve_ids)):
        raise ValueError(f"role manifest repeats curve IDs for {task}/{scientific_role}")
    return str(mapping[str(int(data_seed))]), manifest, path


@lru_cache(maxsize=2)
def _figshare_battery_curves(raw_root: Path):
    return load_figshare_eis_curves(Path(raw_root))


def _battery_curve(curve_id: str, manifest: dict):
    source_root = ROOT / manifest["source"]["raw_root"]
    curves = _figshare_battery_curves(source_root)
    by_id = {}
    for curve in curves:
        metadata = curve.metadata
        identifier = (
            f"cell:{metadata['cell_id']}:test:{int(str(metadata['test_id']).split('_')[-1])}:"
            f"soc:{int(metadata['soc_percent']):03d}"
        )
        by_id[identifier] = curve
    if curve_id not in by_id:
        raise ValueError(f"missing frozen Figshare battery curve {curve_id}")
    curve = by_id[curve_id]
    row = manifest["curves"][curve_id]
    source_path = source_root / row["path"]
    if _file_hash(source_path) != row["sha256"]:
        raise ValueError(f"Figshare battery source hash mismatch for {curve_id}")
    return SimpleNamespace(
        curve_id=curve_id,
        frequency=np.asarray(curve.axis, dtype=np.float64),
        response=np.asarray(curve.response, dtype=np.float32),
        conditions={
            "cell_id": curve.metadata["cell_id"],
            "test_id": curve.metadata["test_id"],
            "soc_percent": int(curve.metadata["soc_percent"]),
        },
        source_files=(str(Path(manifest["source"]["raw_root"]) / row["path"]),),
        metadata={**curve.metadata, "source_sha256": row["sha256"], "source_article_id": 23736582},
    )


def _attach_bundle_contract(bundle: DatasetBundle, *, scientific_role: str,
                            visibility: str, role_manifest_path: Path | None,
                            source_sha256: str | None) -> DatasetBundle:
    metadata = {
        **bundle.metadata,
        "scientific_role": scientific_role,
        "test_access": (
            "opened_confirmatory_test" if visibility == "opened"
            else "sealed_for_validation_only_stage"
        ),
        "role_manifest_sha256": _file_hash(role_manifest_path) if role_manifest_path else None,
        "source_sha256": source_sha256,
    }
    return DatasetBundle(
        x_train=bundle.x_train, y_train=bundle.y_train,
        x_validation=bundle.x_validation, y_validation=bundle.y_validation,
        x_test=bundle.x_test, y_test=bundle.y_test, metadata=metadata,
    )


def make_peak_sensitive_bundle(task: str, data_seed: int, config: dict, raw_root: Path,
                               *, scientific_role: str, visibility: str,
                               frf_density: int | None = None) -> DatasetBundle:
    """Build one frozen scientific instance under an explicit visibility contract."""
    if scientific_role not in {"tuning", "evaluation"}:
        raise ValueError("scientific_role must be tuning or evaluation")
    if visibility not in {"sealed", "opened"}:
        raise ValueError("visibility must be sealed or opened")
    if scientific_role == "tuning" and visibility != "sealed":
        raise ValueError("tuning scientific instances may only be sealed")
    spec = config["tasks"][task]
    manifest_path = None
    source_sha256 = None
    if task == "controlled_fano":
        bundle = make_controlled_fano_bundle(data_seed, spec, visibility=visibility)
        bundle = DatasetBundle(
            bundle.x_train, bundle.y_train, bundle.x_validation, bundle.y_validation,
            bundle.x_test, bundle.y_test,
            {**bundle.metadata, "curve_id": f"controlled_fano:seed:{int(data_seed)}"},
        )
    elif task in {"microwave_fano", "microstrip_resonator"}:
        curve_id, manifest, manifest_path = _curve_id_for_seed(
            task, scientific_role, data_seed, config
        )
        source_path = Path(raw_root) / manifest["raw_relative_source_file"]
        if not source_path.exists() or _file_hash(source_path) != manifest["source_sha256"]:
            raise ValueError(f"{task} role-manifest source SHA-256 mismatch")
        source_sha256 = str(manifest["source_sha256"])
        loader = load_microwave_fano if task == "microwave_fano" else load_microstrip_resonator
        curve = next((item for item in loader(Path(raw_root)) if item.curve_id == curve_id), None)
        if curve is None:
            raise ValueError(f"missing frozen measured curve {curve_id}")
        bundle = _split_curve(
            curve, task, data_seed, spec,
            scientific_role=scientific_role, visibility=visibility,
        )
    elif task == "aluminium_frf":
        curve_id, manifest, manifest_path = _curve_id_for_seed(
            task, scientific_role, data_seed, config
        )
        if frf_density not in [int(value) for value in config["frf_densities"]]:
            raise ValueError("aluminium_frf requires a frozen explicit density")
        bundle = make_frf_density_bundle(
            curve_id, data_seed, int(frf_density), raw_root, visibility=visibility,
        )
        source_sha256 = str(next(
            row["source_sha256"] for row in manifest["inventory"]
            if row["curve_id"] == curve_id
        ))
    elif task == "figshare_battery_eis":
        curve_id, manifest, manifest_path = _curve_id_for_seed(
            task, scientific_role, data_seed, config
        )
        bundle = _split_curve(
            _battery_curve(curve_id, manifest), task, data_seed, spec,
            scientific_role=scientific_role, visibility=visibility,
        )
        source_sha256 = str(manifest["curves"][curve_id]["sha256"])
    elif task == "hybrid_supercapacitor_eis":
        curve_id, manifest, manifest_path = _curve_id_for_seed(
            task, scientific_role, data_seed, config
        )
        role_row = next(row for row in manifest[scientific_role] if row["curve_id"] == curve_id)
        source_path = Path(raw_root) / role_row["source_file"]
        if not source_path.exists() or _file_hash(source_path) != role_row["source_sha256"]:
            raise ValueError("hybrid_supercapacitor_eis role-manifest source SHA-256 mismatch")
        source_sha256 = str(role_row["source_sha256"])
        curve = next(item for item in load_hybrid_supercapacitor_eis(Path(raw_root)) if item.curve_id == curve_id)
        bundle = _split_curve(
            curve, task, data_seed, spec,
            scientific_role=scientific_role, visibility=visibility,
        )
    else:
        raise ValueError(f"no frozen peak-sensitive data adapter for task {task!r}")
    if visibility == "sealed" and len(bundle.y_test):
        bundle = _seal_bundle(bundle)
    return _attach_bundle_contract(
        bundle, scientific_role=scientific_role, visibility=visibility,
        role_manifest_path=manifest_path, source_sha256=source_sha256,
    )


def assert_same_bundle_instance(sealed: DatasetBundle, opened: DatasetBundle) -> None:
    """Assert the sealed window source and opened test bundle are one instance."""
    keys = ("task", "curve_id", "data_seed", "scientific_role", "density")
    for key in keys:
        if sealed.metadata.get(key) != opened.metadata.get(key):
            raise ValueError(f"sealed/opened bundle identity mismatch: {key}")
    for role in ("train", "validation", "test"):
        key = f"{role}_point_ids"
        if sealed.metadata.get(key) != opened.metadata.get(key):
            raise ValueError(f"sealed/opened bundle identity mismatch: {key}")
    for name in ("x_train", "y_train", "x_validation", "y_validation"):
        if not np.array_equal(getattr(sealed, name), getattr(opened, name)):
            raise ValueError(f"sealed/opened bundle identity mismatch: {name}")


def _init_seed(config: dict, data_seed: int) -> int:
    index = [int(seed) for seed in config["tuning_data_seeds"]].index(int(data_seed))
    return int(config["evaluation_init_seeds"][index])


def _window_for_bundle(task: str, bundle: DatasetBundle, config: dict):
    # This is deliberately called before seeded model construction and receives no test role.
    return derive_window_spec(
        task, bundle.x_train, bundle.y_train, bundle.x_validation, bundle.y_validation,
        bundle.metadata, config["peak_detector"],
    )


def _quality_factor(feature: dict, axis_space: str) -> float | None:
    center = float(feature["center"])
    half_width = float(feature["half_width"])
    if half_width <= 0.0:
        return None
    if axis_space == "log10_frequency":
        center_frequency = 10.0 ** center
        bandwidth = 10.0 ** (center + half_width) - 10.0 ** (center - half_width)
    else:
        center_frequency = center
        bandwidth = 2.0 * half_width
    if center_frequency <= 0.0 or bandwidth <= 0.0:
        return None
    return float(center_frequency / bandwidth)


def _kk_proxy(frequency: np.ndarray, response: np.ndarray) -> float | None:
    if len(frequency) < 8 or np.any(frequency <= 0.0):
        return None
    order = np.argsort(frequency, kind="mergesort")
    log_frequency = np.log10(np.asarray(frequency, dtype=np.float64)[order])
    impedance = np.asarray(response, dtype=np.complex128)[order]
    if np.any(np.diff(log_frequency) <= 0.0):
        return None
    uniform = np.linspace(log_frequency[0], log_frequency[-1], len(log_frequency))
    real = np.interp(uniform, log_frequency, impedance.real)
    imaginary = np.interp(uniform, log_frequency, impedance.imag)
    transformed = np.imag(hilbert(real - np.mean(real)))
    scale = float(np.std(imaginary))
    if not math.isfinite(scale) or scale <= np.finfo(float).eps:
        return None
    residual = min(
        np.sqrt(np.mean((imaginary - transformed) ** 2)),
        np.sqrt(np.mean((imaginary + transformed) ** 2)),
    )
    return float(residual / scale)


def scientific_metrics(task: str, frequency, truth, prediction, window_spec, *,
                       training_status: str, validation_global_nrmse: float | None) -> dict:
    """Compute scientifically named diagnostics with explicit applicability."""
    values, axis = _metric_axis(window_spec, frequency)
    truth_complex = _complex_response(truth, "truth", require_finite=False)
    prediction_complex = _complex_response(prediction, "prediction", require_finite=False)
    applicability = {}

    q_error = None
    if window_spec.status != "ok":
        applicability["quality_factor_or_bandwidth_error"] = {
            "status": "unidentifiable", "reason": "window_has_no_identifiable_feature",
        }
    elif np.all(np.isfinite(truth_complex)) and np.all(np.isfinite(prediction_complex)):
        true_features = _metric_features(task, axis, truth_complex, window_spec.detector)
        predicted_features = _metric_features(task, axis, prediction_complex, window_spec.detector)
        spacing = float(np.median(np.diff(axis))) if len(axis) > 1 else np.finfo(float).eps
        matches = _match_features(
            true_features, predicted_features,
            float(window_spec.detector.get("matching_tolerance", 2.0 * spacing)),
        )
        if matches:
            true_feature, predicted_feature = max(
                matches, key=lambda pair: (pair[0]["prominence"], -pair[0]["center"])
            )
            true_q = _quality_factor(true_feature, window_spec.axis_space)
            predicted_q = _quality_factor(predicted_feature, window_spec.axis_space)
            if true_q is not None and predicted_q is not None:
                q_error = float(abs(predicted_q - true_q) / true_q)
        applicability["quality_factor_or_bandwidth_error"] = {
            "status": "applicable" if q_error is not None else "unidentifiable",
            "reason": (
                "relative_quality_factor_error_from_matched_fwhm_features"
                if q_error is not None else "no_matched_feature_with_positive_fwhm"
            ),
        }
    else:
        applicability["quality_factor_or_bandwidth_error"] = {
            "status": "unidentifiable", "reason": "nonfinite_response",
        }

    eis = task in {"figshare_battery_eis", "hybrid_supercapacitor_eis"}
    dominant_error = None
    kk_proxy = None
    if eis and np.all(np.isfinite(truth_complex)) and np.all(np.isfinite(prediction_complex)):
        true_index = int(np.argmax(-truth_complex.imag))
        predicted_index = int(np.argmax(-prediction_complex.imag))
        dominant_error = float(abs(np.log10(values[predicted_index]) - np.log10(values[true_index])))
        kk_proxy = _kk_proxy(values, prediction_complex)
        applicability["dominant_relaxation_frequency_error"] = {
            "status": "applicable", "reason": "negative_imaginary_impedance_maximum",
        }
        applicability["kramers_kronig_consistency_proxy"] = {
            "status": "applicable" if kk_proxy is not None else "unidentifiable",
            "reason": (
                "descriptive_hilbert_proxy_not_formal_kk_validation"
                if kk_proxy is not None else "insufficient_or_degenerate_frequency_grid"
            ),
        }
    else:
        reason = "task_is_not_eis" if not eis else "nonfinite_response"
        for name in ("dominant_relaxation_frequency_error", "kramers_kronig_consistency_proxy"):
            applicability[name] = {"status": "not_applicable" if not eis else "unidentifiable", "reason": reason}

    underfit = None
    if training_status == "ok" and validation_global_nrmse is not None and math.isfinite(float(validation_global_nrmse)):
        underfit = float(float(validation_global_nrmse) >= 1.0)
        applicability["underfit_rate"] = {
            "status": "applicable", "reason": "frozen_validation_global_nrmse_threshold_1.0",
        }
    else:
        applicability["underfit_rate"] = {
            "status": "unidentifiable", "reason": "training_failure_is_reported_separately",
        }
    return {
        "quality_factor_or_bandwidth_error": q_error,
        "dominant_relaxation_frequency_error": dominant_error,
        "kramers_kronig_consistency_proxy": kk_proxy,
        "underfit_rate": underfit,
        "underfit_criterion": UNDERFIT_CRITERION,
        "metric_applicability": applicability,
    }


def summarize_selection_rows(rows: list[dict]) -> dict:
    """Preserve failures/no-feature rows while aggregating identifiable peak losses."""
    peak = []
    global_losses = []
    no_feature_count = 0
    failure_count = 0
    for row in rows:
        if row.get("status") != "ok":
            failure_count += 1
        global_loss = row.get("validation_global_nrmse")
        if global_loss is not None and math.isfinite(float(global_loss)):
            global_losses.append(float(global_loss))
        if row.get("window_spec", {}).get("status") != "ok":
            no_feature_count += 1
            continue
        loss = row.get("validation_resonance_nrmse")
        if row.get("status") == "ok" and loss is not None and math.isfinite(float(loss)) and float(loss) > 0.0:
            peak.append(float(loss))
    return {
        "median_validation_resonance_nrmse": float(np.median(peak)) if peak else None,
        "median_validation_global_nrmse": float(np.median(global_losses)) if global_losses else None,
        "peak_available_count": len(peak),
        "global_available_count": len(global_losses),
        "no_feature_count": no_feature_count,
        "failure_count": failure_count,
        "record_count": len(rows),
    }


def train_one(task: str, family: str, candidate: dict, optimizer: dict, *, data_seed: int,
              init_seed: int, config: dict, raw_root: Path, phase: str,
              protocol_sha256: str, candidate_manifest_sha256: str,
              selected_common_budget: int | None = None, window_spec=None,
              frf_density: int | None = None, bundle: DatasetBundle | None = None,
              protocol_config: dict | None = None) -> dict:
    """Build only after the sealed WindowSpec and train one frozen instance."""
    started = time.perf_counter()
    protocol_config = protocol_config or config
    evaluation = phase == "evaluation"
    scientific_role = "evaluation" if evaluation else "tuning"
    visibility = "opened" if evaluation else "sealed"
    if bundle is None:
        bundle = make_peak_sensitive_bundle(
            task, data_seed, protocol_config, raw_root,
            scientific_role=scientific_role, visibility=visibility,
            frf_density=frf_density,
        )
    if window_spec is None:
        if evaluation:
            sealed = make_peak_sensitive_bundle(
                task, data_seed, protocol_config, raw_root,
                scientific_role="evaluation", visibility="sealed",
                frf_density=frf_density,
            )
            assert_same_bundle_instance(sealed, bundle)
            window_spec = _window_for_bundle(task, sealed, protocol_config)
        else:
            window_spec = _window_for_bundle(task, bundle, protocol_config)
    task_spec = protocol_config["tasks"][task]
    model, model_metadata = seeded_build(
        init_seed,
        lambda: build_model(
            family, int(task_spec["input_dim"]), int(task_spec["output_dim"]),
            candidate["config"], seed=init_seed,
        ),
    )
    model = model.to(DEVICE)
    prepared = prepare_bundle(bundle, "complex_regression", DEVICE)
    resonance_scorer, global_scorer = build_v2_validation_scorers(
        window_spec, bundle.x_validation[:, 0],
        log_weighted=bool(task_spec.get("log_frequency_weighted_metric", False)),
        output_mean=prepared.y_mean, output_std=prepared.y_std,
    )
    training = config["training"]
    trained = fit_task_model(
        model, prepared, "complex_regression", learning_rate=optimizer["lr"],
        weight_decay=optimizer["weight_decay"], max_steps=int(training["max_steps"]),
        min_steps=int(training["min_steps"]), patience=int(training["patience"]),
        checkpoints=training["checkpoints"], gradient_clip=float(training["gradient_clip"]),
        batch_size=task_spec.get("batch_size"), batch_seed=int(data_seed) + 49979687,
        validation_interval=int(training["validation_interval"]), validation_scorer=resonance_scorer,
        global_validation_scorer=global_scorer,
    )
    record = {
        "protocol_key": protocol_config["protocol_key"],
        "protocol_mode": protocol_config.get("protocol_mode", "production"),
        "smoke_root_marker": protocol_config.get("smoke_root_marker"),
        "config_sha256": _config_hash(protocol_config),
        "protocol_sha256": protocol_sha256,
        "stage": "peak_sensitive_evaluation" if evaluation else "peak_sensitive_tuning",
        "stage_role": "confirmatory_test" if evaluation else "validation_only",
        "phase": phase,
        "scientific_role": scientific_role,
        "test_visibility": visibility,
        "task": task, "task_role": task_spec["role"], "problem_type": "complex_regression",
        "family": family, "target_budget": int(candidate["target_budget"]),
        "data_seed": int(data_seed), "init_seed": int(init_seed),
        "candidate_key": candidate["candidate_key"], "config": candidate["config"],
        "actual_parameters": int(candidate["actual_parameters"]), "optimizer": optimizer,
        "best_step": trained.best_step, "optimizer_steps": trained.optimizer_steps,
        "validation_score": trained.best_validation_score,
        "validation_resonance_nrmse": trained.best_validation_score,
        "validation_global_nrmse": trained.best_global_validation_score,
        "status": trained.status, "curve": trained.curve,
        "training_wall_seconds": trained.wall_seconds, "total_wall_seconds": time.perf_counter() - started,
        "peak_memory_bytes": trained.peak_memory_bytes, "device": str(DEVICE),
        "dataset": _compact_metadata(bundle.metadata), "model_metadata": model_metadata,
        "source_manifest_sha256": bundle.metadata.get("source_manifest_sha256"),
        "window_spec": window_spec.to_dict(),
        "initialization_mode": model_metadata.get("initialization_mode", "fixed"),
        "gradient_clip_events": trained.gradient_clip_events,
        "gradient_clip_rate": trained.gradient_clip_rate,
        "gradient_clip_threshold": float(training["gradient_clip"]),
        "training_min_steps": int(training["min_steps"]),
        "training_max_steps": int(training["max_steps"]),
        "training_patience": int(training["patience"]),
        "training_checkpoints": [int(value) for value in training["checkpoints"]],
        "training_batch_size": int(len(prepared.x_train)),
        "batch_seed": int(data_seed) + 49979687,
        "validation_interval": int(training["validation_interval"]),
        "validation_evaluations": trained.validation_evaluations,
        "tuning_seed": None if evaluation else int(data_seed),
        "frf_density": int(frf_density) if frf_density is not None else None,
        "candidate_manifest_sha256": candidate_manifest_sha256,
    }
    if evaluation:
        if selected_common_budget is None:
            raise ValueError("evaluation requires a frozen common budget before validation")
        metrics = complex_window_metrics(
            bundle.x_test[:, 0], bundle.y_test, trained.prediction, window_spec,
            log_weighted=bool(task_spec.get("log_frequency_weighted_metric", False)),
        )
        named_metrics = scientific_metrics(
            task, bundle.x_test[:, 0], bundle.y_test, trained.prediction, window_spec,
            training_status=trained.status,
            validation_global_nrmse=trained.best_global_validation_score,
        )
        record.update({
            "test_metrics": {
                **metrics,
                **named_metrics,
                "numerical_failure_rate": float(metrics["status"] == "numerical_failure"),
            },
            "primary_metric_name": "resonance_window_complex_nrmse",
            "primary_metric_value": metrics.get("resonance_window_complex_nrmse"),
            "selected_common_budget": int(selected_common_budget),
        })
    return record


def _record_key(record: dict) -> tuple:
    return (
        str(record.get("stage")), str(record.get("phase")), str(record.get("task")),
        str(record.get("family")), int(record.get("target_budget", -1)),
        record.get("frf_density"), str(record.get("candidate_key")),
        str(record.get("optimizer", {}).get("key")), int(record.get("data_seed", -1)),
        int(record.get("init_seed", -1)),
    )


def assert_unique_records(records: list[dict]) -> None:
    keys = [_record_key(record) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate result keys")


def validate_tuning_record(record: dict, *, inventory: dict | None = None,
                           candidate_manifest_sha256: str | None = None) -> None:
    if record.get("stage") != "peak_sensitive_tuning" or record.get("stage_role") != "validation_only":
        raise ValueError("tuning records must be validation-only")
    if "test_metrics" in record or "primary_metric_value" in record:
        raise ValueError("tuning records must not contain test metrics")
    if record.get("dataset", {}).get("test_access") != "sealed_for_validation_only_stage":
        raise ValueError("tuning records must retain sealed test access")
    if inventory is not None:
        validate_peak_sensitive_record(
            record, inventory=inventory,
            candidate_manifest_sha256=candidate_manifest_sha256,
        )


def _validate_record_scope(record: dict, *, task: str, family: str, budget: int,
                           phase: str, frf_density: int | None) -> None:
    expected = {
        "task": task, "family": family, "target_budget": int(budget),
        "phase": phase, "frf_density": frf_density,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(f"record {field} is outside shard scope")


def _screening_config(config: dict) -> dict:
    screening = deepcopy(config)
    steps = int(config["screening"]["steps"])
    screening["training"].update({
        "max_steps": steps, "min_steps": steps, "patience": steps,
        "checkpoints": sorted(set(step for step in config["training"]["checkpoints"] if step <= steps) | {steps}),
    })
    return screening


def tune_shard(config: dict, inventory: dict, output_root: Path, task: str, family: str,
               budget: int, raw_root: Path, protocol_sha256: str,
               frf_density: int | None = None) -> dict:
    candidates = inventory["tasks"][task][family][str(int(budget))]
    selection_path = _selection_path(output_root, task, family, budget, frf_density)
    inventory_hash = _file_hash(_inventory_path(output_root))
    if not candidates:
        selection = {"task": task, "family": family, "target_budget": int(budget),
                     "frf_density": frf_density, "status": "missing_budget_match",
                     "protocol_sha256": protocol_sha256,
                     "config_sha256": _config_hash(config),
                     "candidate_manifest_sha256": inventory_hash}
        _write_json(selection_path, selection)
        return selection
    candidates = [{**candidate, "target_budget": int(budget)} for candidate in candidates]
    windows = {
        int(data_seed): _window_for_bundle(
            task,
            make_peak_sensitive_bundle(
                task, int(data_seed), config, raw_root,
                scientific_role="tuning", visibility="sealed",
                frf_density=frf_density,
            ),
            config,
        )
        for data_seed in config["tuning_data_seeds"]
    }
    screen_path = _screen_path(output_root, task, family, budget, frf_density)
    screening_optimizer = config["screening"]["optimizer_choice_by_family"][family]
    screening_config = _screening_config(config)
    with _jsonl_lock(screen_path):
        screen_records = _load_jsonl(screen_path)
        assert_unique_records(screen_records)
        for record in screen_records:
            _validate_record_scope(
                record, task=task, family=family, budget=budget,
                phase="screening", frf_density=frf_density,
            )
            validate_tuning_record(
                record, inventory=inventory,
                candidate_manifest_sha256=inventory_hash,
            )
        completed = {_record_key(record) for record in screen_records}
        for candidate in candidates:
            for data_seed in config["tuning_data_seeds"]:
                init_seed = _init_seed(config, data_seed)
                planned = {
                    "stage": "peak_sensitive_tuning", "phase": "screening", "task": task,
                    "family": family, "target_budget": int(budget), "frf_density": frf_density,
                    "candidate_key": candidate["candidate_key"], "optimizer": screening_optimizer,
                    "data_seed": int(data_seed), "init_seed": int(init_seed),
                }
                if _record_key(planned) in completed:
                    continue
                record = train_one(
                    task, family, candidate, screening_optimizer, data_seed=int(data_seed),
                    init_seed=init_seed, config=screening_config, protocol_config=config,
                    raw_root=raw_root, phase="screening", protocol_sha256=protocol_sha256,
                    candidate_manifest_sha256=inventory_hash,
                    window_spec=windows[int(data_seed)], frf_density=frf_density,
                )
                validate_tuning_record(
                    record, inventory=inventory,
                    candidate_manifest_sha256=inventory_hash,
                )
                _append_jsonl(screen_path, record)
                screen_records.append(record)
                completed.add(_record_key(record))
    expected = len(candidates) * len(config["tuning_data_seeds"])
    if len(screen_records) != expected:
        raise ValueError(f"incomplete screening for {task}/{family}/{budget}: {len(screen_records)}/{expected}")
    finalist_keys = set(rank_v2_candidates(screen_records, min(int(config["finalists_per_budget"]), len(candidates))))
    finalists = [candidate for candidate in candidates if candidate["candidate_key"] in finalist_keys]
    final_path = _final_path(output_root, task, family, budget, frf_density)
    with _jsonl_lock(final_path):
        final_records = _load_jsonl(final_path)
        assert_unique_records(final_records)
        for record in final_records:
            _validate_record_scope(
                record, task=task, family=family, budget=budget,
                phase="convergence", frf_density=frf_density,
            )
            validate_tuning_record(
                record, inventory=inventory,
                candidate_manifest_sha256=inventory_hash,
            )
        completed = {_record_key(record) for record in final_records}
        for candidate in finalists:
            for optimizer in config["optimizer_grids"][family]:
                for data_seed in config["tuning_data_seeds"]:
                    init_seed = _init_seed(config, data_seed)
                    planned = {
                        "stage": "peak_sensitive_tuning", "phase": "convergence", "task": task,
                        "family": family, "target_budget": int(budget), "frf_density": frf_density,
                        "candidate_key": candidate["candidate_key"], "optimizer": optimizer,
                        "data_seed": int(data_seed), "init_seed": int(init_seed),
                    }
                    if _record_key(planned) in completed:
                        continue
                    record = train_one(
                        task, family, candidate, optimizer, data_seed=int(data_seed),
                        init_seed=init_seed, config=config, raw_root=raw_root,
                        phase="convergence", protocol_sha256=protocol_sha256,
                        candidate_manifest_sha256=inventory_hash,
                        window_spec=windows[int(data_seed)], frf_density=frf_density,
                    )
                    validate_tuning_record(
                        record, inventory=inventory,
                        candidate_manifest_sha256=inventory_hash,
                    )
                    _append_jsonl(final_path, record)
                    final_records.append(record)
                    completed.add(_record_key(record))
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in final_records:
        groups.setdefault((record["candidate_key"], record["optimizer"]["key"]), []).append(record)
    complete = {key: rows for key, rows in groups.items() if len(rows) == len(config["tuning_data_seeds"])}
    expected_groups = len(finalists) * len(config["optimizer_grids"][family])
    if len(complete) != expected_groups:
        raise ValueError(f"incomplete convergence tuning for {task}/{family}/{budget}")
    winner_key = min(complete, key=lambda key: (v2_candidate_rank_key(complete[key]), key))
    winner_rows = complete[winner_key]
    winner_candidate = next(candidate for candidate in candidates if candidate["candidate_key"] == winner_key[0])
    winner_optimizer = next(item for item in config["optimizer_grids"][family] if item["key"] == winner_key[1])
    summary = summarize_selection_rows(winner_rows)
    selection = {
        "task": task, "family": family, "target_budget": int(budget), "status": "selected",
        "frf_density": frf_density,
        "candidate": winner_candidate, "optimizer": winner_optimizer,
        "selected_step": int(round(np.median([row["best_step"] for row in winner_rows]))),
        **summary,
        "screened_candidate_count": len(candidates), "finalist_count": len(finalists),
        "optimizer_count": len(config["optimizer_grids"][family]),
        "tuning_seed_count": len(config["tuning_data_seeds"]),
        "protocol_sha256": protocol_sha256,
        "config_sha256": _config_hash(config),
        "candidate_manifest_sha256": inventory_hash,
        "screen_jsonl_sha256": _file_hash(screen_path),
        "final_jsonl_sha256": _file_hash(final_path),
    }
    _write_json(selection_path, selection)
    return selection


def _canonical_selected_cell(config: dict, candidates: list[dict], *, task: str,
                             family: str, budget: int, frf_density: int | None,
                             protocol_sha256: str, inventory_hash: str,
                             screen_path: Path, final_path: Path,
                             screen_rows: list[dict], final_rows: list[dict]) -> dict:
    """Reconstruct one selection from frozen candidates and complete tuning rows."""
    finalist_count = min(int(config["finalists_per_budget"]), len(candidates))
    finalist_keys = set(rank_v2_candidates(screen_rows, finalist_count))
    finalists = [
        {**candidate, "target_budget": int(budget)}
        for candidate in candidates
        if candidate["candidate_key"] in finalist_keys
    ]
    if len(finalists) != finalist_count:
        raise ValueError("screening did not produce the frozen finalist count")

    expected_groups = {
        (candidate["candidate_key"], optimizer["key"])
        for candidate in finalists
        for optimizer in config["optimizer_grids"][family]
    }
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in final_rows:
        groups.setdefault((row["candidate_key"], row["optimizer"]["key"]), []).append(row)
    if set(groups) != expected_groups:
        raise ValueError("convergence rows do not exactly match recomputed screening finalists")
    tuning_seed_count = len(config["tuning_data_seeds"])
    if any(len(rows) != tuning_seed_count for rows in groups.values()):
        raise ValueError("convergence finalist/optimizer groups are incomplete")

    winner_key = min(groups, key=lambda key: (v2_candidate_rank_key(groups[key]), key))
    winner_rows = groups[winner_key]
    winner_candidate = next(
        candidate for candidate in finalists if candidate["candidate_key"] == winner_key[0]
    )
    winner_optimizer = next(
        optimizer for optimizer in config["optimizer_grids"][family]
        if optimizer["key"] == winner_key[1]
    )
    return {
        "task": task,
        "family": family,
        "target_budget": int(budget),
        "status": "selected",
        "frf_density": frf_density,
        "candidate": winner_candidate,
        "optimizer": winner_optimizer,
        "selected_step": int(round(np.median([row["best_step"] for row in winner_rows]))),
        **summarize_selection_rows(winner_rows),
        "screened_candidate_count": len(candidates),
        "finalist_count": len(finalists),
        "optimizer_count": len(config["optimizer_grids"][family]),
        "tuning_seed_count": tuning_seed_count,
        "protocol_sha256": protocol_sha256,
        "config_sha256": _config_hash(config),
        "candidate_manifest_sha256": inventory_hash,
        "screen_jsonl_sha256": _file_hash(screen_path),
        "final_jsonl_sha256": _file_hash(final_path),
    }


def audit_tuning(config: dict, inventory: dict, output_root: Path, *, tasks: list[str],
                 families: list[str], budgets: list[int], protocol_sha256: str,
                 frf_densities: list[int] | None = None) -> dict:
    """Validate every loaded tuning record and all exact-scope stage artifacts."""
    errors = []
    densities = [int(value) for value in (frf_densities or config["frf_densities"])]
    scope = {
        "tasks": list(tasks), "families": list(families),
        "budgets": [int(value) for value in budgets], "frf_densities": densities,
    }
    inventory_path = _inventory_path(output_root)
    inventory_hash = _file_hash(inventory_path)
    if inventory.get("scope") != scope:
        errors.append("candidate inventory scope differs from requested audit scope")
    if inventory.get("config_sha256") != _config_hash(config):
        errors.append("candidate inventory config hash mismatch")
    if inventory.get("protocol_sha256") != protocol_sha256:
        errors.append("candidate inventory protocol hash mismatch")
    input_hashes = {str(inventory_path): inventory_hash}
    cell_count = 0
    record_count = 0
    for task in tasks:
        task_densities = densities if task == "aluminium_frf" else [None]
        for density in task_densities:
            for family in families:
                for budget in budgets:
                    cell_count += 1
                    candidates = inventory["tasks"][task][family][str(int(budget))]
                    selection_path = _selection_path(output_root, task, family, budget, density)
                    if not selection_path.exists():
                        errors.append(f"missing selection {task}/{density}/{family}/{budget}")
                        continue
                    selection = json.loads(selection_path.read_text())
                    input_hashes[str(selection_path)] = _file_hash(selection_path)
                    expected_status = "selected" if candidates else "missing_budget_match"
                    if selection.get("status") != expected_status:
                        errors.append(f"wrong selection status {task}/{density}/{family}/{budget}")
                        continue
                    for field, expected_value in (
                        ("task", task), ("family", family), ("target_budget", int(budget)),
                        ("frf_density", density), ("protocol_sha256", protocol_sha256),
                        ("config_sha256", _config_hash(config)),
                        ("candidate_manifest_sha256", inventory_hash),
                    ):
                        if selection.get(field) != expected_value:
                            errors.append(f"selection {field} mismatch {task}/{density}/{family}/{budget}")
                    if not candidates:
                        continue
                    paths = (
                        (_screen_path(output_root, task, family, budget, density),
                         len(candidates) * len(config["tuning_data_seeds"]), "screening"),
                        (_final_path(output_root, task, family, budget, density),
                         min(len(candidates), int(config["finalists_per_budget"]))
                         * len(config["optimizer_grids"][family])
                         * len(config["tuning_data_seeds"]), "convergence"),
                    )
                    phase_rows = {}
                    for path, expected_count, phase in paths:
                        try:
                            with _jsonl_lock(path):
                                rows = _load_jsonl(path)
                                assert_unique_records(rows)
                                for row in rows:
                                    _validate_record_scope(
                                        row, task=task, family=family, budget=budget,
                                        phase=phase, frf_density=density,
                                    )
                                    validate_tuning_record(
                                        row, inventory=inventory,
                                        candidate_manifest_sha256=inventory_hash,
                                    )
                        except Exception as error:
                            errors.append(f"{phase} {task}/{density}/{family}/{budget}: {error}")
                            rows = []
                        if len(rows) != expected_count:
                            errors.append(
                                f"incomplete {phase} {task}/{density}/{family}/{budget}: "
                                f"{len(rows)}/{expected_count}"
                            )
                        phase_rows[phase] = rows
                        if path.exists():
                            digest = _file_hash(path)
                            input_hashes[str(path)] = digest
                            selection_field = "screen_jsonl_sha256" if phase == "screening" else "final_jsonl_sha256"
                            if selection.get(selection_field) != digest:
                                errors.append(f"selection {selection_field} mismatch {task}/{density}/{family}/{budget}")
                        record_count += len(rows)
                    try:
                        canonical_selection = _canonical_selected_cell(
                            config, candidates, task=task, family=family, budget=int(budget),
                            frf_density=density, protocol_sha256=protocol_sha256,
                            inventory_hash=inventory_hash,
                            screen_path=paths[0][0], final_path=paths[1][0],
                            screen_rows=phase_rows.get("screening", []),
                            final_rows=phase_rows.get("convergence", []),
                        )
                    except Exception as error:
                        errors.append(
                            f"selection reconstruction {task}/{density}/{family}/{budget}: {error}"
                        )
                    else:
                        for field, expected_value in canonical_selection.items():
                            if selection.get(field) != expected_value:
                                errors.append(
                                    f"selection {field} mismatch {task}/{density}/{family}/{budget}"
                                )
                        unexpected = set(selection) - set(canonical_selection)
                        if unexpected:
                            errors.append(
                                f"selection has unexpected fields {task}/{density}/{family}/{budget}: "
                                f"{sorted(unexpected)}"
                            )
    audit = {
        "protocol_key": config["protocol_key"],
        "protocol_mode": config.get("protocol_mode", "production"),
        "protocol_sha256": protocol_sha256,
        "config_sha256": _config_hash(config),
        "candidate_manifest_sha256": inventory_hash,
        "scope": scope,
        "scope_sha256": _sha256_bytes(_canonical_bytes(scope)),
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "tasks": tasks, "families": families, "budgets": budgets,
        "frf_densities": densities,
        "cell_count": cell_count, "record_count": record_count,
        "input_file_sha256": input_hashes,
    }
    _write_json(Path(output_root) / "tuning_audit.json", audit)
    if errors:
        raise ValueError("tuning audit failed:\n" + "\n".join(errors[:30]))
    return audit


def analyze_common_budget(losses: dict[str, dict[int, float]], *, budgets: list[int] | None = None,
                          tolerance: float = 0.05, minimum_families: int = 6) -> dict:
    """Apply the frozen available-family relative-loss rule with an audit trail."""
    valid = {}
    for family, values in losses.items():
        available = {}
        for budget, loss in values.items():
            if loss is None:
                continue
            value = float(loss)
            if math.isfinite(value) and value > 0.0:
                available[int(budget)] = value
        if available:
            valid[family] = available
    budget_axis = sorted(
        {int(item) for item in budgets}
        if budgets is not None else {budget for values in valid.values() for budget in values}
    )
    relative = {
        family: {budget: value / min(values.values()) for budget, value in values.items()}
        for family, values in valid.items()
    }
    available_count = {
        budget: sum(budget in values for values in relative.values()) for budget in budget_axis
    }
    pooled = {
        budget: float(np.median([
            values[budget] for values in relative.values() if budget in values
        ]))
        for budget in budget_axis
        if available_count[budget] >= int(minimum_families)
    }
    if not pooled:
        raise ValueError(
            f"at least {minimum_families} available families are required at a candidate budget"
        )
    best = min(pooled.values())
    selected = min(
        budget for budget, value in pooled.items()
        if value <= best * (1.0 + float(tolerance))
    )
    return {
        "selected_common_budget": selected,
        "losses": valid,
        "relative_losses": relative,
        "available_family_count": available_count,
        "pooled_median_relative_losses": pooled,
        "ineligible_budgets": [budget for budget in budget_axis if budget not in pooled],
    }


def select_common_budget(losses: dict[str, dict[int, float]], *, tolerance: float = 0.05,
                         minimum_families: int = 6) -> int:
    return int(analyze_common_budget(
        losses, tolerance=tolerance, minimum_families=minimum_families,
    )["selected_common_budget"])


def _exact_passed_audit(config: dict, inventory: dict, output_root: Path, *, scope: dict,
                        protocol_sha256: str) -> tuple[dict, Path]:
    path = Path(output_root) / "tuning_audit.json"
    if not path.exists():
        raise ValueError("missing successful tuning audit")
    audit = json.loads(path.read_text())
    expected = {
        "status": "passed",
        "scope": scope,
        "scope_sha256": _sha256_bytes(_canonical_bytes(scope)),
        "config_sha256": _config_hash(config),
        "protocol_sha256": protocol_sha256,
        "candidate_manifest_sha256": _file_hash(_inventory_path(output_root)),
    }
    for field, value in expected.items():
        if audit.get(field) != value:
            raise ValueError(f"tuning audit {field} does not match exact freeze scope")
    if audit.get("protocol_mode") != config.get("protocol_mode", "production"):
        raise ValueError("tuning audit protocol mode mismatch")
    for source, digest in audit.get("input_file_sha256", {}).items():
        source_path = Path(source)
        if not source_path.exists() or _file_hash(source_path) != digest:
            raise ValueError(f"tuning audit input hash changed: {source}")
    return audit, path


def _canonical_frozen_budgets(config: dict, output_root: Path, *, scope: dict,
                              protocol_sha256: str, audit: dict,
                              audit_path: Path) -> dict:
    tasks = list(scope["tasks"])
    families = list(scope["families"])
    budgets = [int(value) for value in scope["budgets"]]
    densities = [int(value) for value in scope["frf_densities"]]
    inventory_path = _inventory_path(output_root)
    frozen = {
        "protocol_key": config["protocol_key"],
        "protocol_mode": config.get("protocol_mode", "production"),
        "config_sha256": _config_hash(config),
        "protocol_sha256": protocol_sha256,
        "scope": scope,
        "scope_sha256": _sha256_bytes(_canonical_bytes(scope)),
        "selection_rule": config["common_budget_selection"],
        "tuning_audit_sha256": _file_hash(audit_path),
        "candidate_manifest_sha256": _file_hash(inventory_path),
        "input_jsonl_sha256": {
            path: digest for path, digest in audit["input_file_sha256"].items()
            if path.endswith(".jsonl")
        },
        "selection_file_sha256": {},
        "tasks": {},
    }
    for task in tasks:
        losses = {}
        exclusions = []
        task_selection_hashes = {}
        task_densities = densities if task == "aluminium_frf" else [None]
        for family in families:
            family_losses = {}
            for budget in budgets:
                density_losses = []
                for density in task_densities:
                    path = _selection_path(output_root, task, family, budget, density)
                    cell = {
                        "family": family, "budget": int(budget), "frf_density": density,
                        "selection_path": str(path),
                    }
                    if not path.exists():
                        exclusions.append({**cell, "state": "missing_selection"})
                        continue
                    digest = _file_hash(path)
                    task_selection_hashes[str(path)] = digest
                    selection = json.loads(path.read_text())
                    if selection.get("status") != "selected":
                        exclusions.append({**cell, "state": selection.get("status", "invalid_selection")})
                        continue
                    loss = selection.get("median_validation_resonance_nrmse")
                    if loss is None or not math.isfinite(float(loss)) or float(loss) <= 0.0:
                        state = "no_identifiable_feature" if selection.get("no_feature_count", 0) else "missing_peak_loss"
                        exclusions.append({
                            **cell, "state": state,
                            "failure_count": int(selection.get("failure_count", 0)),
                            "global_available_count": int(selection.get("global_available_count", 0)),
                        })
                        continue
                    density_losses.append(float(loss))
                    if selection.get("failure_count", 0) or selection.get("no_feature_count", 0):
                        exclusions.append({
                            **cell, "state": "partially_available",
                            "failure_count": int(selection.get("failure_count", 0)),
                            "no_feature_count": int(selection.get("no_feature_count", 0)),
                        })
                if density_losses:
                    family_losses[int(budget)] = float(np.median(density_losses))
            if family_losses:
                losses[family] = family_losses
        analysis = analyze_common_budget(
            losses, budgets=[int(value) for value in budgets],
            tolerance=float(config["common_budget_selection"]["relative_loss_threshold"]),
            minimum_families=int(config["common_budget_selection"]["minimum_family_eligibility"]),
        )
        selected = int(analysis["selected_common_budget"])
        frozen["tasks"][task] = {
            **analysis,
            "selected_common_budget": selected,
            "density_scope": task_densities,
            "exclusions": exclusions,
            "selection_file_sha256": task_selection_hashes,
            "eligible_families_at_selected_budget": sorted(
                family for family, values in analysis["losses"].items() if selected in values
            ),
        }
        frozen["selection_file_sha256"].update(task_selection_hashes)
    return frozen


def freeze_budgets(config: dict, inventory: dict, output_root: Path, *, tasks: list[str],
                   families: list[str], budgets: list[int], protocol_sha256: str,
                   frf_densities: list[int] | None = None) -> dict:
    densities = [int(value) for value in (frf_densities or config["frf_densities"])]
    scope = {
        "tasks": list(tasks), "families": list(families),
        "budgets": [int(value) for value in budgets], "frf_densities": densities,
    }
    audit, audit_path = _exact_passed_audit(
        config, inventory, output_root, scope=scope, protocol_sha256=protocol_sha256,
    )
    frozen = _canonical_frozen_budgets(
        config, output_root, scope=scope, protocol_sha256=protocol_sha256,
        audit=audit, audit_path=audit_path,
    )
    _write_json(Path(output_root) / "frozen_task_budgets.json", frozen)
    return frozen


def require_evaluation_prerequisites(output_root: Path, task: str, *, config: dict | None = None,
                                     inventory: dict | None = None,
                                     protocol_sha256: str | None = None) -> int:
    root = Path(output_root)
    budget_path, audit_path = root / "frozen_task_budgets.json", root / "tuning_audit.json"
    if not budget_path.exists():
        raise ValueError("missing frozen task budgets")
    if not audit_path.exists() or json.loads(audit_path.read_text()).get("status") != "passed":
        raise ValueError("missing successful tuning audit")
    if config is None or inventory is None:
        raise ValueError("exact evaluation prerequisites require config and candidate inventory")
    frozen = json.loads(budget_path.read_text())
    audit = json.loads(audit_path.read_text())
    scope = audit.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("tuning audit has no exact scope")
    audit, audit_path = _exact_passed_audit(
        config, inventory, output_root, scope=scope, protocol_sha256=protocol_sha256,
    )
    canonical = _canonical_frozen_budgets(
        config, output_root, scope=scope, protocol_sha256=protocol_sha256,
        audit=audit, audit_path=audit_path,
    )
    if frozen != _json_normalized(canonical):
        raise ValueError("frozen budget artifact differs from canonical audit reconstruction")
    task_budgets = frozen.get("tasks", {})
    if task not in task_budgets:
        raise ValueError(f"no frozen common budget for {task}")
    return int(task_budgets[task]["selected_common_budget"])


def evaluate_shard(config: dict, inventory: dict, output_root: Path, task: str, family: str,
                   raw_root: Path, protocol_sha256: str,
                   frf_density: int | None = None) -> None:
    budget = require_evaluation_prerequisites(
        output_root, task, config=config, inventory=inventory,
        protocol_sha256=protocol_sha256,
    )
    selection_path = _selection_path(output_root, task, family, budget, frf_density)
    if not selection_path.exists():
        raise ValueError(f"missing selected configuration: {selection_path}")
    selection = json.loads(selection_path.read_text())
    if selection.get("status") != "selected":
        return
    selection_sha256 = _file_hash(selection_path)
    frozen = json.loads((Path(output_root) / "frozen_task_budgets.json").read_text())
    if frozen["tasks"][task]["selection_file_sha256"].get(str(selection_path)) != selection_sha256:
        raise ValueError("evaluation selection hash is absent from the frozen budget artifact")
    path = _evaluation_path(output_root, task, family, budget, frf_density)
    candidate = {**selection["candidate"], "target_budget": budget}
    inventory_hash = _file_hash(_inventory_path(output_root))
    windows = {}
    evaluation_bundles = {}
    for data_seed in config["evaluation_data_seeds"]:
        data_seed = int(data_seed)
        sealed = make_peak_sensitive_bundle(
            task, data_seed, config, raw_root,
            scientific_role="evaluation", visibility="sealed",
            frf_density=frf_density,
        )
        windows[data_seed] = _window_for_bundle(task, sealed, config)
        opened = make_peak_sensitive_bundle(
            task, data_seed, config, raw_root,
            scientific_role="evaluation", visibility="opened",
            frf_density=frf_density,
        )
        assert_same_bundle_instance(sealed, opened)
        evaluation_bundles[data_seed] = opened
    with _jsonl_lock(path):
        existing = _load_jsonl(path)
        assert_unique_records(existing)
        for row in existing:
            _validate_record_scope(
                row, task=task, family=family, budget=budget,
                phase="evaluation", frf_density=frf_density,
            )
            if row.get("selection_sha256") != selection_sha256:
                raise ValueError("existing evaluation record selection hash mismatch")
            validate_peak_sensitive_record(
                row, inventory=inventory,
                candidate_manifest_sha256=inventory_hash,
            )
        completed = {_record_key(record) for record in existing}
        for data_seed in config["evaluation_data_seeds"]:
            for init_seed in config["evaluation_init_seeds"]:
                planned = {
                    "stage": "peak_sensitive_evaluation", "phase": "evaluation", "task": task,
                    "family": family, "target_budget": budget, "frf_density": frf_density,
                    "candidate_key": candidate["candidate_key"], "optimizer": selection["optimizer"],
                    "data_seed": int(data_seed), "init_seed": int(init_seed),
                }
                if _record_key(planned) in completed:
                    continue
                record = train_one(
                    task, family, candidate, selection["optimizer"], data_seed=int(data_seed),
                    init_seed=int(init_seed), config=config, raw_root=raw_root,
                    phase="evaluation", protocol_sha256=protocol_sha256,
                    candidate_manifest_sha256=inventory_hash,
                    selected_common_budget=budget, window_spec=windows[int(data_seed)],
                    frf_density=frf_density, bundle=evaluation_bundles[int(data_seed)],
                )
                record["selection_path"] = str(selection_path)
                record["selection_sha256"] = selection_sha256
                validate_peak_sensitive_record(
                    _json_safe(record), inventory=inventory,
                    candidate_manifest_sha256=inventory_hash,
                )
                _append_jsonl(path, record)
                completed.add(_record_key(record))


def make_smoke_record() -> dict:
    """A schema-valid validation-only record used to keep smoke validation deterministic."""
    config = _smoke_config(json.loads((ROOT / "peak_sensitive_config.json").read_text()))
    bundle = make_peak_sensitive_bundle(
        "controlled_fano", 547, config, ROOT / "downstream_data" / "raw",
        scientific_role="tuning", visibility="sealed",
    )
    return {
        "protocol_key": "cfnn-peak-sensitive-extension-2.0.0",
        "protocol_mode": "smoke", "smoke_root_marker": SMOKE_ROOT_MARKER,
        "config_sha256": _config_hash(config),
        "protocol_sha256": _protocol_hash(ROOT / "peak_sensitive_config.json", ROOT / "peak_sensitive_results_schema.json"),
        "stage": "peak_sensitive_tuning", "stage_role": "validation_only",
        "phase": "screening", "scientific_role": "tuning", "test_visibility": "sealed",
        "task": "controlled_fano",
        "task_role": "controlled_mechanism", "problem_type": "complex_regression", "family": "CFNN",
        "target_budget": 128, "data_seed": 547, "init_seed": 11003, "candidate_key": "{}",
        "config": {}, "actual_parameters": 128, "optimizer": {"key": "lr1e-3_wd0", "lr": 0.001, "weight_decay": 0.0},
        "best_step": 5, "optimizer_steps": 5, "validation_score": 0.2, "status": "ok", "curve": [],
        "training_wall_seconds": 0.0, "total_wall_seconds": 0.0, "device": "cpu",
        "dataset": _compact_metadata(bundle.metadata),
        "source_manifest_sha256": bundle.metadata.get("source_manifest_sha256"),
        "window_spec": {"task": "controlled_fano", "axis_space": "linear", "intervals": [[0.4, 0.6]], "source_role": "generator_metadata", "status": "ok", "detector": {}},
        "validation_resonance_nrmse": 0.2, "validation_global_nrmse": 0.2,
        "initialization_mode": "fixed", "gradient_clip_events": 0, "gradient_clip_rate": 0.0,
        "gradient_clip_threshold": 5.0, "training_min_steps": 2, "training_max_steps": 5,
        "training_patience": 5, "training_checkpoints": [2, 5], "training_batch_size": 64,
        "batch_seed": 49980234, "validation_interval": 1, "validation_evaluations": 5,
        "tuning_seed": 547, "frf_density": None, "candidate_manifest_sha256": "a" * 64,
    }


def _filter(available: list, requested: str | None, cast=str) -> list:
    if not requested:
        return list(available)
    values = [cast(value.strip()) for value in requested.split(",") if value.strip()]
    unknown = set(values) - set(available)
    if unknown:
        raise ValueError(f"unknown requested values: {sorted(unknown)}")
    return values


def _smoke_config(config: dict) -> dict:
    result = deepcopy(config)
    result["protocol_mode"] = "smoke"
    result["smoke_root_marker"] = SMOKE_ROOT_MARKER
    result["tuning_data_seeds"] = result["tuning_data_seeds"][:1]
    result["evaluation_data_seeds"] = result["evaluation_data_seeds"][:1]
    result["evaluation_init_seeds"] = result["evaluation_init_seeds"][:1]
    result["finalists_per_budget"] = 1
    result["training"].update({"screening_steps": 5, "min_steps": 2, "max_steps": 5,
                               "patience": 5, "validation_interval": 1, "checkpoints": [2, 5]})
    result["screening"]["steps"] = 5
    for family in result["families"]:
        result["optimizer_grids"][family] = result["optimizer_grids"][family][:1]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "peak_sensitive_config.json")
    parser.add_argument("--schema", type=Path, default=ROOT / "peak_sensitive_results_schema.json")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "downstream_data" / "raw")
    parser.add_argument("--output-root", type=Path, default=ROOT / "peak_sensitive_results")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--tasks")
    parser.add_argument("--families")
    parser.add_argument("--budgets")
    parser.add_argument("--densities")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-root-marker")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.smoke:
        if args.smoke_root_marker != SMOKE_ROOT_MARKER or args.output_root.name != "smoke":
            raise ValueError("smoke requires the explicit Task7 smoke marker and a smoke-named root")
        if args.stage not in {"inventory", "tune"}:
            raise ValueError("the explicit Task7 smoke protocol is inventory/tune only")
        config = _smoke_config(config)
    elif args.smoke_root_marker is not None:
        raise ValueError("production runs cannot carry a smoke root marker")
    tasks = _filter(config["fixed_scientific_tasks"], args.tasks)
    families = _filter(config["families"], args.families)
    budgets = _filter(config["budgets"], args.budgets, int)
    frf_densities = _filter(config["frf_densities"], args.densities, int)
    scope = {
        "tasks": tasks, "families": families, "budgets": budgets,
        "frf_densities": frf_densities,
    }
    if not args.smoke and args.stage != "inventory":
        # Production stages may not create a result-root artifact before this gate passes.
        from peak_sensitive_preflight import verify_preflight_freeze

        verify_preflight_freeze(
            config_path=args.config, schema_path=args.schema, raw_root=args.raw_root,
            output_root=args.output_root, requested_scope=scope,
        )
    manifest = _protocol_manifest(
        config, args.config, args.schema, sys.argv,
        raw_root=args.raw_root, tasks=_manifest_tasks(config, tasks),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    ensure_protocol_root(args.output_root, manifest)
    inventory_path = _inventory_path(args.output_root)
    if args.stage == "inventory":
        inventory = build_inventory(config, args.output_root, tasks=tasks, families=families,
                                    budgets=budgets, frf_densities=frf_densities,
                                    smoke=args.smoke, protocol_sha256=manifest["protocol_sha256"])
    else:
        if not inventory_path.exists():
            raise ValueError("missing candidate inventory required by preflight freeze")
        inventory = json.loads(inventory_path.read_text())
    if args.stage == "inventory":
        return
    if args.stage in {"tune", "all"}:
        for task in tasks:
            densities = frf_densities if task == "aluminium_frf" else [None]
            for density in densities:
                for family in families:
                    for budget in budgets:
                        tune_shard(
                            config, inventory, args.output_root, task, family, budget,
                            args.raw_root, manifest["protocol_sha256"], density,
                        )
    if args.stage in {"audit-tuning", "all"}:
        audit_tuning(
            config, inventory, args.output_root, tasks=tasks, families=families,
            budgets=budgets, protocol_sha256=manifest["protocol_sha256"],
            frf_densities=frf_densities,
        )
    if args.stage in {"freeze-budgets", "all"}:
        freeze_budgets(
            config, inventory, args.output_root, tasks=tasks, families=families,
            budgets=budgets, protocol_sha256=manifest["protocol_sha256"],
            frf_densities=frf_densities,
        )
    if args.stage in {"evaluate", "all"}:
        for task in tasks:
            densities = frf_densities if task == "aluminium_frf" else [None]
            for density in densities:
                for family in families:
                    evaluate_shard(
                        config, inventory, args.output_root, task, family, args.raw_root,
                        manifest["protocol_sha256"], density,
                    )


if __name__ == "__main__":
    main()

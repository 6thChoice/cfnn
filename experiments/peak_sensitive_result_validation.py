"""Schema and frozen-protocol validation for peak-sensitive result records."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "peak_sensitive_config.json"
SCHEMA_PATH = ROOT / "peak_sensitive_results_schema.json"
SMOKE_ROOT_MARKER = "task7-explicit-smoke-root-v1"
PROVENANCE_ROOT = ROOT / "peak_sensitive_results" / "provenance"

ROLE_MANIFEST_NAMES = {
    "microwave_fano": "microwave_fano_role_manifest.json",
    "microstrip_resonator": "microstrip_resonator_role_manifest.json",
    "aluminium_frf": "frf_density_manifest.json",
    "figshare_battery_eis": "figshare_battery_eis_role_manifest.json",
    "hybrid_supercapacitor_eis": "curve_manifest.json",
}

with CONFIG_PATH.open() as handle:
    FROZEN_CONFIG = json.load(handle)
with SCHEMA_PATH.open() as handle:
    RESULTS_SCHEMA = json.load(handle)

_SCHEMA_VALIDATOR = Draft202012Validator(RESULTS_SCHEMA)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _protocol_hash() -> str:
    digest = hashlib.sha256()
    for path in (CONFIG_PATH, SCHEMA_PATH):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


FROZEN_CONFIG_SHA256 = _canonical_hash(FROZEN_CONFIG)
FROZEN_PROTOCOL_SHA256 = _protocol_hash()


def _protocol_error(message: str) -> None:
    raise ValueError(message)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _point_hash(point_ids: list[str]) -> str:
    encoded = json.dumps(point_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _point_contract(point_ids: dict[str, list[str]]) -> dict:
    contract = {}
    for role, values in point_ids.items():
        contract[f"{role}_point_ids_count"] = len(values)
        contract[f"{role}_point_ids_sha256"] = _point_hash(values)
    return contract


def _role_manifest(task: str) -> tuple[dict, Path]:
    path = PROVENANCE_ROOT / ROLE_MANIFEST_NAMES[task]
    manifest = json.loads(path.read_text())
    if manifest.get("task") != task:
        _protocol_error(f"role manifest task mismatch for {task}")
    if manifest.get("protocol_key") != FROZEN_CONFIG["protocol_key"]:
        _protocol_error(f"role manifest protocol mismatch for {task}")
    return manifest, path


def _manifest_curve_identity(task: str, role: str, data_seed: int) -> tuple[str, int, str, str]:
    manifest, path = _role_manifest(task)
    try:
        curve_id = str(manifest["seed_curve_id_map"][role][str(int(data_seed))])
    except KeyError:
        _protocol_error(f"role manifest has no curve for {task}/{role}/{data_seed}")
    if task in {"microwave_fano", "microstrip_resonator"}:
        point_count = int(manifest["selected_curve_point_counts"][curve_id])
        source_sha256 = str(manifest["source_sha256"])
    elif task == "aluminium_frf":
        row = next(item for item in manifest["inventory"] if item["curve_id"] == curve_id)
        point_count = int(row["point_count"])
        source_sha256 = str(row["source_sha256"])
    elif task == "figshare_battery_eis":
        row = manifest["curves"][curve_id]
        point_count = int(row["point_count"])
        source_sha256 = str(row["sha256"])
    else:
        row = next(item for item in manifest[role] if item["curve_id"] == curve_id)
        point_count = int(row["point_count"])
        source_sha256 = str(row["source_sha256"])
    return curve_id, point_count, source_sha256, _file_hash(path)


def _frf_parent_seed(curve_id: str, data_seed: int, role: str) -> int:
    digest = hashlib.sha256(
        f"frf-density-{role}-parent-v2:{curve_id}:{int(data_seed)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _expected_dataset_identity(record: dict) -> tuple[dict, str | None]:
    task = record["task"]
    data_seed = int(record["data_seed"])
    role = record["scientific_role"]
    access = (
        "sealed_for_validation_only_stage"
        if role == "tuning" else "opened_confirmatory_test"
    )
    expected = {
        "task": task,
        "data_seed": data_seed,
        "scientific_role": role,
        "test_access": access,
    }
    source_manifest_sha256 = None
    if task == "controlled_fano":
        spec = FROZEN_CONFIG["tasks"][task]
        n_train = int(spec["n_train"])
        n_validation = int(spec["n_validation"])
        n_test = int(spec["n_test"])
        rng = np.random.default_rng(data_seed)
        rng.uniform(float(spec.get("domain", [0.0, 1.0])[0]),
                    float(spec.get("domain", [0.0, 1.0])[1]),
                    size=n_train + n_validation)
        order = rng.permutation(n_train + n_validation)
        train = np.sort(order[:n_train])
        validation = np.sort(order[n_train:])
        expected.update(
            curve_id=f"controlled_fano:seed:{data_seed}",
            role_manifest_sha256=None,
            source_sha256=None,
            **_point_contract({
                "train": [f"observed:{int(index)}" for index in train],
                "validation": [f"observed:{int(index)}" for index in validation],
                "test": [f"dense:{index}" for index in range(n_test)],
            }),
        )
        return expected, source_manifest_sha256

    curve_id, point_count, source_sha256, role_manifest_sha256 = (
        _manifest_curve_identity(task, role, data_seed)
    )
    expected.update(
        curve_id=curve_id,
        role_manifest_sha256=role_manifest_sha256,
        source_sha256=source_sha256,
    )
    if task == "aluminium_frf":
        manifest, _ = _role_manifest(task)
        density = int(record["frf_density"])
        parent_train_count = int(manifest["parent_train_points"])
        parent_validation_count = int(manifest["validation_points"])
        validation_count = int(manifest["validation_by_density"][str(density)])
        train_rng = np.random.default_rng(_frf_parent_seed(curve_id, data_seed, "train"))
        parent_train = train_rng.permutation(point_count)[:parent_train_count]
        train_set = {int(index) for index in parent_train}
        validation_pool = np.asarray(
            [index for index in range(point_count) if index not in train_set], dtype=np.int64
        )
        validation_rng = np.random.default_rng(
            _frf_parent_seed(curve_id, data_seed, "validation")
        )
        parent_validation = validation_pool[
            validation_rng.permutation(len(validation_pool))[:parent_validation_count]
        ]
        train = parent_train[:density]
        validation = parent_validation[:validation_count]
        selected = {int(index) for index in train} | {int(index) for index in validation}
        test = [index for index in range(point_count) if index not in selected]
        point_ids = lambda values: [f"{curve_id}:frequency:{int(index)}" for index in values]
        parent_train_ids = point_ids(parent_train)
        parent_validation_ids = point_ids(parent_validation)
        parent_mask = {
            "curve_id": curve_id,
            "data_seed": data_seed,
            "parent_train_point_ids": parent_train_ids,
            "parent_validation_point_ids": parent_validation_ids,
        }
        expected.update(
            density=density,
            parent_mask_hash=_canonical_hash(parent_mask),
            parent_train_order_sha256=_point_hash(parent_train_ids),
            parent_validation_order_sha256=_point_hash(parent_validation_ids),
            **_point_contract({
                "parent_train": parent_train_ids,
                "parent_validation": parent_validation_ids,
                "train": point_ids(train),
                "validation": point_ids(validation),
                "test": point_ids(test),
            }),
        )
        source_manifest_sha256 = str(manifest["source_manifest_sha256"])
    else:
        spec = FROZEN_CONFIG["tasks"][task]
        n_train = int(spec["n_train"])
        n_validation = int(spec["n_validation"])
        order = np.random.default_rng(data_seed + 32452843).permutation(point_count)
        train = np.sort(order[:n_train])
        validation = np.sort(order[n_train:n_train + n_validation])
        test = np.sort(order[n_train + n_validation:])
        point_ids = lambda values: [f"{curve_id}:frequency:{int(index)}" for index in values]
        expected.update(**_point_contract({
            "train": point_ids(train),
            "validation": point_ids(validation),
            "test": point_ids(test),
        }))
    return expected, source_manifest_sha256


def _validate_dataset_identity(record: dict) -> None:
    expected, source_manifest_sha256 = _expected_dataset_identity(record)
    dataset = record["dataset"]
    for field, value in expected.items():
        if dataset.get(field) != value:
            _protocol_error(f"dataset {field} does not match frozen scientific identity")
    if record.get("source_manifest_sha256") != source_manifest_sha256:
        _protocol_error("source_manifest_sha256 does not match frozen source identity")


def _validate_metric_semantics(record: dict) -> None:
    if record["stage"] != "peak_sensitive_evaluation":
        return
    metrics = record["test_metrics"]
    for name in (
        "missed_feature_rate", "false_feature_rate",
        "numerical_failure_rate", "underfit_rate",
    ):
        value = metrics[name]
        if value is not None and not (0.0 <= float(value) <= 1.0):
            _protocol_error(f"{name} must be null or within [0,1]")

    expected_failure = float(metrics["status"] == "numerical_failure")
    if metrics["numerical_failure_rate"] != expected_failure:
        _protocol_error("numerical_failure_rate is inconsistent with test metric status")
    global_loss = record.get("validation_global_nrmse")
    expected_underfit = (
        float(float(global_loss) >= 1.0)
        if record["status"] == "ok"
        and global_loss is not None
        and math.isfinite(float(global_loss))
        else None
    )
    if metrics["underfit_rate"] != expected_underfit:
        _protocol_error("underfit_rate is inconsistent with frozen validation criterion")

    applicability = metrics["metric_applicability"]
    eis = record["task"] in {"figshare_battery_eis", "hybrid_supercapacitor_eis"}
    expectations = {
        "quality_factor_or_bandwidth_error": None,
        "dominant_relaxation_frequency_error": eis,
        "kramers_kronig_consistency_proxy": eis,
        "underfit_rate": expected_underfit is not None,
    }
    for name, task_applicable in expectations.items():
        value = metrics[name]
        status = applicability[name]["status"]
        if name in {"dominant_relaxation_frequency_error", "kramers_kronig_consistency_proxy"} and not eis:
            if value is not None or status != "not_applicable":
                _protocol_error(f"{name} must be null and not_applicable outside EIS tasks")
        elif value is None:
            if status != "unidentifiable":
                _protocol_error(f"null {name} must be marked unidentifiable")
        elif status != "applicable":
            _protocol_error(f"numeric {name} must be marked applicable")


def _expected_source_role(task: str) -> str:
    source = FROZEN_CONFIG["tasks"][task]["window_source"]
    if source == "generator_metadata":
        return "generator_metadata"
    if source == "train_validation_observed_response":
        return "train_validation"
    _protocol_error(f"no window source mapping is frozen for task {task!r}")


def _same_optimizer(left: dict, right: dict) -> bool:
    return (
        left.get("key") == right.get("key")
        and float(left.get("lr")) == float(right.get("lr"))
        and float(left.get("weight_decay")) == float(right.get("weight_decay"))
    )


def _validate_window(record: dict, task_config: dict) -> None:
    task = record["task"]
    window = record["window_spec"]
    if window["task"] != task:
        _protocol_error("window_spec.task must equal task")
    expected_axis = task_config["frequency_transform"]
    if window["axis_space"] != expected_axis:
        _protocol_error(f"window_spec.axis_space must be {expected_axis!r} for task {task!r}")
    status = window["status"]
    intervals = window["intervals"]
    if status == "no_identifiable_feature":
        if intervals or window["source_role"] != "no_identifiable_feature":
            _protocol_error(
                "no_identifiable_feature windows require source_role=no_identifiable_feature "
                "and empty intervals"
            )
    elif status == "ok":
        expected_source = _expected_source_role(task)
        if window["source_role"] != expected_source:
            _protocol_error(f"window_spec.source_role must be {expected_source!r} for task {task!r}")
        if not intervals:
            _protocol_error("ok windows require at least one interval")
    else:
        _protocol_error(f"unsupported window status {status!r}")
    previous_start = None
    for start, end in intervals:
        if start >= end:
            _protocol_error("window intervals must have strictly increasing endpoints")
        if previous_start is not None and start <= previous_start:
            _protocol_error("window intervals must be ordered by increasing start")
        previous_start = start


def _validate_controls(record: dict) -> None:
    mode = record["protocol_mode"]
    if mode == "production":
        if record["smoke_root_marker"] is not None:
            _protocol_error("production records cannot carry a smoke root marker")
        if record["config_sha256"] != FROZEN_CONFIG_SHA256:
            _protocol_error("record config_sha256 does not match the frozen configuration")
        if record["protocol_sha256"] != FROZEN_PROTOCOL_SHA256:
            _protocol_error("record protocol_sha256 does not match current config/schema bytes")
        training = FROZEN_CONFIG["training"]
        expected = {
            "training_min_steps": int(training["min_steps"]),
            "training_max_steps": int(training["max_steps"]),
            "training_patience": int(training["patience"]),
            "training_checkpoints": training["checkpoints"],
            "validation_interval": int(training["validation_interval"]),
            "gradient_clip_threshold": float(training["gradient_clip"]),
        }
        if record["phase"] == "screening":
            steps = int(FROZEN_CONFIG["screening"]["steps"])
            expected.update(
                training_min_steps=steps,
                training_max_steps=steps,
                training_patience=steps,
                training_checkpoints=[step for step in training["checkpoints"] if step <= steps],
            )
        for field, value in expected.items():
            if record[field] != value:
                _protocol_error(f"{field} does not match frozen {record['phase']} controls")
        if record["status"] == "ok":
            steps = int(record["optimizer_steps"])
            if record["phase"] == "screening" and steps != int(FROZEN_CONFIG["screening"]["steps"]):
                _protocol_error("successful production screening must run exactly 500 steps")
            if record["phase"] in {"convergence", "evaluation"} and not (
                int(training["min_steps"]) <= steps <= int(training["max_steps"])
            ):
                _protocol_error("successful convergence/evaluation steps must be within 500..4000")
    elif mode == "smoke":
        if record["smoke_root_marker"] != SMOKE_ROOT_MARKER:
            _protocol_error("smoke records require the explicit Task7 smoke root marker")
        if record["stage"] != "peak_sensitive_tuning":
            _protocol_error("the Task7 smoke protocol is tuning-only")
        if not (1 <= int(record["training_min_steps"]) <= int(record["training_max_steps"]) <= 5):
            _protocol_error("smoke training controls must remain inside the explicit five-step protocol")
        if int(record["optimizer_steps"]) > int(record["training_max_steps"]):
            _protocol_error("smoke optimizer_steps exceed the smoke maximum")
    else:
        _protocol_error(f"unsupported protocol_mode {mode!r}")


def _validate_inventory_membership(record: dict, inventory: dict,
                                   candidate_manifest_sha256: str | None) -> None:
    if candidate_manifest_sha256 is None:
        _protocol_error("actual candidate manifest SHA-256 is required")
    if record["candidate_manifest_sha256"] != candidate_manifest_sha256:
        _protocol_error("record candidate_manifest_sha256 differs from the actual artifact")
    try:
        candidates = inventory["tasks"][record["task"]][record["family"]][str(record["target_budget"])]
    except (KeyError, TypeError) as error:
        _protocol_error("record scope is absent from the candidate inventory")
    matches = [
        candidate for candidate in candidates
        if candidate["candidate_key"] == record["candidate_key"]
    ]
    if len(matches) != 1:
        _protocol_error("record candidate is not a unique inventory member")
    candidate = matches[0]
    if candidate["config"] != record["config"] or int(candidate["actual_parameters"]) != int(record["actual_parameters"]):
        _protocol_error("record candidate metadata differs from the candidate inventory")


def _validate_semantics(record: dict, *, inventory: dict | None = None,
                        candidate_manifest_sha256: str | None = None) -> None:
    task = record["task"]
    task_config = FROZEN_CONFIG["tasks"].get(task)
    if task_config is None:
        _protocol_error(f"task {task!r} is not present in the frozen configuration")
    if record["task_role"] != task_config["role"]:
        _protocol_error("task_role does not match the frozen task role")
    if record["problem_type"] != "complex_regression":
        _protocol_error("problem_type must be complex_regression")
    if record["family"] not in FROZEN_CONFIG["families"]:
        _protocol_error("family is not frozen")
    if record["init_seed"] not in FROZEN_CONFIG["evaluation_init_seeds"]:
        _protocol_error(f"init_seed {record['init_seed']!r} is not frozen for this protocol")

    stage = record["stage"]
    tuning = stage == "peak_sensitive_tuning"
    expected_role = "tuning" if tuning else "evaluation"
    expected_visibility = "sealed" if tuning else "opened"
    expected_stage_role = "validation_only" if tuning else "confirmatory_test"
    expected_phase = {"screening", "convergence"} if tuning else {"evaluation"}
    seed_key = "tuning_data_seeds" if tuning else "evaluation_data_seeds"
    if record["stage_role"] != expected_stage_role:
        _protocol_error("stage_role does not match stage")
    if record["scientific_role"] != expected_role:
        _protocol_error("scientific_role does not match stage")
    if record["test_visibility"] != expected_visibility:
        _protocol_error("test_visibility does not match stage")
    if record["phase"] not in expected_phase:
        _protocol_error("phase does not match stage")
    if record["data_seed"] not in FROZEN_CONFIG[seed_key]:
        _protocol_error("data_seed is not frozen for this stage")
    if record["tuning_seed"] != (record["data_seed"] if tuning else None):
        _protocol_error("tuning_seed must equal tuning data_seed and be null in evaluation")
    dataset = record["dataset"]
    if dataset.get("scientific_role") != expected_role:
        _protocol_error("dataset scientific_role does not match record")
    expected_access = "sealed_for_validation_only_stage" if tuning else "opened_confirmatory_test"
    if dataset.get("test_access") != expected_access:
        _protocol_error("dataset test_access does not match stage visibility")

    density = record["frf_density"]
    if task == "aluminium_frf":
        if density not in FROZEN_CONFIG["frf_densities"]:
            _protocol_error("Aluminium FRF requires a frozen density")
    elif density is not None:
        _protocol_error("frf_density must be null outside Aluminium FRF")

    budget = int(record["target_budget"])
    relative_error = abs(int(record["actual_parameters"]) - budget) / budget
    if relative_error > float(FROZEN_CONFIG["budget_tolerance"]) + 1e-12:
        _protocol_error("actual_parameters are outside the frozen +/-5% budget")
    canonical_key = json.dumps(record["config"], sort_keys=True, separators=(",", ":"))
    if record["candidate_key"] != canonical_key:
        _protocol_error("candidate_key must be the canonical model config")

    if record["phase"] == "screening":
        optimizer_candidates = [FROZEN_CONFIG["screening"]["optimizer_choice_by_family"][record["family"]]]
    else:
        optimizer_candidates = FROZEN_CONFIG["optimizer_grids"][record["family"]]
    if not any(_same_optimizer(record["optimizer"], item) for item in optimizer_candidates):
        _protocol_error("optimizer is not a frozen member for this family/phase")

    if tuning:
        expected_index = FROZEN_CONFIG["tuning_data_seeds"].index(record["data_seed"])
        if record["init_seed"] != FROZEN_CONFIG["evaluation_init_seeds"][expected_index]:
            _protocol_error("tuning init_seed must map by frozen data-seed index")
        if record.get("selected_common_budget") is not None:
            _protocol_error("tuning cannot select a common budget")
    else:
        if record["selected_common_budget"] != budget:
            _protocol_error("evaluation selected_common_budget must equal target_budget")
        if record["primary_metric_value"] != record["test_metrics"]["resonance_window_complex_nrmse"]:
            _protocol_error("primary_metric_value must equal the resonance-window test metric")

    _validate_controls(record)
    _validate_window(record, task_config)
    _validate_dataset_identity(record)
    _validate_metric_semantics(record)
    if inventory is not None:
        _validate_inventory_membership(record, inventory, candidate_manifest_sha256)


def validate_peak_sensitive_record(record: dict, *, inventory: dict | None = None,
                                   candidate_manifest_sha256: str | None = None) -> None:
    """Validate one loaded or new tuning/evaluation record in production."""
    _SCHEMA_VALIDATOR.validate(record)
    _validate_semantics(
        record, inventory=inventory,
        candidate_manifest_sha256=candidate_manifest_sha256,
    )

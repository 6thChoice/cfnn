"""Pinned measured-data adapters for the peak-sensitive protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from confirmatory_protocol import DatasetBundle
from downstream_protocol import MeasuredCurve


HSC_SOURCE_KEY = "hybrid_supercapacitor_eis"
HSC_ARTICLE_ID = 24321496
HSC_VERSION = 4
HSC_FILENAME = "Supercap_EIS.mat"
FRF_TASK = "aluminium_frf"
FRF_DENSITIES = (24, 48, 96, 192)
FRF_PARENT_TRAIN_POINTS = max(FRF_DENSITIES)
FRF_VALIDATION_POINTS = 96
FRF_MINIMUM_POINTS = FRF_PARENT_TRAIN_POINTS + FRF_VALIDATION_POINTS + 1
FRF_VALIDATION_BY_DENSITY = {24: 12, 48: 24, 96: 48, 192: 96}
FRF_TUNING_CURVE_COUNT = 3
FRF_EVALUATION_CURVE_COUNT = 5
FRF_ROLE_SELECTION_METHOD = "frf-source-manifest-sha256-order-v1"
FRF_REPLACEMENT_RULE = "source_manifest_sha256_order_skip_ineligible_v1"
FRF_ESTIMATOR_VERSION = "aluminium_frf_dense_h1_v1"
FRF_DOMAIN_RULE = "all_finite_positive_frequency_bins_v1"
FRF_ESTIMATOR_PARAMETERS = {
    "estimator": "H1",
    "estimator_version": FRF_ESTIMATOR_VERSION,
    "force_channel_sensor_type": "Force",
    "response_channel_sensor_type": "Accelerometer",
    "response_scale_m_per_s2": 9.81,
    "window": "hann",
    "nperseg": 4096,
    "noverlap": 2048,
    "detrend": "constant",
    "scaling": "spectrum",
    "domain_rule": FRF_DOMAIN_RULE,
}
ROOT = Path(__file__).resolve().parent


def _mat_field(structure: object, name: str, context: str) -> object:
    if not hasattr(structure, name):
        raise ValueError(f"missing MATLAB field {name!r} in {context}")
    return getattr(structure, name)


def _strict_hsc_arrays(entry: object, context: str) -> tuple[np.ndarray, np.ndarray]:
    frequency = np.asarray(_mat_field(entry, "Freq", context), dtype=np.float64).reshape(-1)
    real = np.asarray(_mat_field(entry, "Real", context), dtype=np.float64).reshape(-1)
    imaginary = np.asarray(_mat_field(entry, "Imm", context), dtype=np.float64).reshape(-1)
    if len(frequency) < 2 or real.shape != frequency.shape or imaginary.shape != frequency.shape:
        raise ValueError(f"invalid impedance-array shape in {context}")
    if not (
        np.all(np.isfinite(frequency))
        and np.all(np.isfinite(real))
        and np.all(np.isfinite(imaginary))
    ):
        raise ValueError(f"nonfinite impedance data in {context}")
    if np.any(frequency <= 0.0):
        raise ValueError(f"nonpositive frequency in {context}")
    if np.any(np.diff(frequency) <= 0.0):
        raise ValueError(f"nonmonotonic or duplicate frequency axis in {context}")
    return frequency, np.column_stack((real, imaginary)).astype(np.float32)


@lru_cache(maxsize=2)
def load_hybrid_supercapacitor_eis(raw_root: Path) -> list[MeasuredCurve]:
    """Read the released Figshare v4 EIS table without interpolation or sorting."""
    path = Path(raw_root) / HSC_SOURCE_KEY / HSC_FILENAME
    if not path.exists():
        raise FileNotFoundError(path)
    document = loadmat(path, squeeze_me=True, struct_as_record=False)
    root = document.get("Supercap_EIS")
    if root is None:
        raise ValueError(f"missing Supercap_EIS root structure in {path}")
    data = _mat_field(root, "data", "Supercap_EIS")
    info = _mat_field(root, "info", "Supercap_EIS")
    declared_soc = {
        int(value) for value in np.asarray(
            _mat_field(info, "State_Charge", "Supercap_EIS.info")
        ).reshape(-1)
    }
    declared_temperatures = {
        int(value) for value in np.asarray(
            _mat_field(info, "Temperature", "Supercap_EIS.info")
        ).reshape(-1)
    }
    curves: list[MeasuredCurve] = []
    for soc_field in sorted(getattr(data, "_fieldnames", ())):
        match = re.fullmatch(r"SOC(\d+)", soc_field)
        if not match:
            raise ValueError(f"unexpected SOC field {soc_field!r}")
        soc_percent = int(match.group(1))
        if soc_percent not in declared_soc:
            raise ValueError(f"undeclared SOC value {soc_percent} in {soc_field}")
        entries = np.asarray(_mat_field(data, soc_field, "Supercap_EIS.data"), dtype=object).reshape(-1)
        for entry in entries:
            temperature_c = int(np.asarray(
                _mat_field(entry, "Temp", soc_field)
            ).reshape(()))
            if temperature_c not in declared_temperatures:
                raise ValueError(
                    f"undeclared temperature {temperature_c} in {soc_field}"
                )
            context = f"{soc_field}/temperature:{temperature_c}"
            frequency, response = _strict_hsc_arrays(entry, context)
            curves.append(MeasuredCurve(
                curve_id=f"soc:{soc_percent:03d}:temperature:{temperature_c:+03d}",
                frequency=frequency,
                response=response,
                conditions={
                    "temperature_c": temperature_c,
                    "soc_percent": soc_percent,
                },
                source_files=(HSC_FILENAME,),
                metadata={
                    "source_key": HSC_SOURCE_KEY,
                    "source_article_id": HSC_ARTICLE_ID,
                    "source_version": HSC_VERSION,
                    "matlab_path": f"Supercap_EIS.data.{soc_field}",
                    "frequency_unit": "Hz",
                    "response_encoding": "released_real_and_Imm_columns",
                    "source_imaginary_field": "Imm",
                    "imaginary_sign_normalization": (
                        "identity_as_released_negative_imaginary_impedance"
                    ),
                    "device_id": None,
                    "repetition_id": None,
                },
            ))
    if len(curves) != len(declared_soc) * len(declared_temperatures):
        raise ValueError(
            "incomplete HSC condition grid: "
            f"found {len(curves)} curves for {len(declared_soc)} SOC values and "
            f"{len(declared_temperatures)} temperatures"
        )
    if len({curve.curve_id for curve in curves}) != len(curves):
        raise ValueError("duplicate hybrid-supercapacitor curve IDs")
    return curves


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_curve_order(curve: MeasuredCurve) -> str:
    return hashlib.sha256(
        f"hsc-condition-stratification-v1:{curve.curve_id}".encode("ascii")
    ).hexdigest()


def _stratified_hsc_roles(
    curves: list[MeasuredCurve], tuning_count: int, evaluation_count: int
) -> tuple[list[MeasuredCurve], list[MeasuredCurve]]:
    """Freeze a diverse condition set before any model-related input exists."""
    requested = tuning_count + evaluation_count
    if requested > len(curves):
        raise ValueError(f"requested {requested} HSC curves but only {len(curves)} exist")
    remaining = sorted(curves, key=_stable_curve_order)
    selected: list[MeasuredCurve] = []
    used_temperatures: set[int] = set()
    used_soc: set[int] = set()
    while len(selected) < requested:
        candidate = min(
            remaining,
            key=lambda curve: (
                -(int(curve.conditions["temperature_c"] not in used_temperatures)),
                -(int(curve.conditions["soc_percent"] not in used_soc)),
                _stable_curve_order(curve),
            ),
        )
        remaining.remove(candidate)
        selected.append(candidate)
        used_temperatures.add(int(candidate.conditions["temperature_c"]))
        used_soc.add(int(candidate.conditions["soc_percent"]))
    return selected[:tuning_count], selected[tuning_count:]


def _manifest_curve_row(curve: MeasuredCurve, source_sha256: str) -> dict:
    return {
        "curve_id": curve.curve_id,
        "temperature_c": int(curve.conditions["temperature_c"]),
        "soc_percent": int(curve.conditions["soc_percent"]),
        "source_file": f"{HSC_SOURCE_KEY}/{HSC_FILENAME}",
        "source_sha256": source_sha256,
    }


def audit_peak_sensitive_sources(raw_root: Path, config: dict) -> dict:
    """Audit the frozen HSC release and select curve roles without model access."""
    raw_root = Path(raw_root)
    source_spec = json.loads((ROOT / "downstream_sources.json").read_text())[HSC_SOURCE_KEY]
    task_spec = config["tasks"][HSC_SOURCE_KEY]
    if int(source_spec["article_id"]) != HSC_ARTICLE_ID:
        raise ValueError("hybrid-supercapacitor source article is not pinned")
    if int(source_spec["version"]) != HSC_VERSION:
        raise ValueError("hybrid-supercapacitor source version is not pinned")
    if task_spec["source_key"] != HSC_SOURCE_KEY:
        raise ValueError("hybrid-supercapacitor task source key mismatch")

    source_path = raw_root / HSC_SOURCE_KEY / HSC_FILENAME
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source_sha256 = _sha256(source_path)
    expected_data_file = source_spec["expected_data_file"]
    if source_path.stat().st_size != int(expected_data_file["size"]):
        raise ValueError("hybrid-supercapacitor raw file size differs from pinned metadata")
    source_md5 = hashlib.md5(source_path.read_bytes()).hexdigest()
    if source_md5 != expected_data_file["figshare_md5"]:
        raise ValueError("hybrid-supercapacitor raw file MD5 differs from pinned metadata")
    curves = load_hybrid_supercapacitor_eis(raw_root)
    temperatures = {int(curve.conditions["temperature_c"]) for curve in curves}
    soc_values = {int(curve.conditions["soc_percent"]) for curve in curves}
    pairs = {
        (int(curve.conditions["temperature_c"]), int(curve.conditions["soc_percent"]))
        for curve in curves
    }
    point_counts = Counter(len(curve.frequency) for curve in curves)
    tuning, evaluation = _stratified_hsc_roles(
        curves,
        int(task_spec["tuning_condition_count"]),
        int(task_spec["evaluation_condition_count"]),
    )
    if {curve.curve_id for curve in tuning} & {curve.curve_id for curve in evaluation}:
        raise ValueError("HSC tuning and evaluation curve IDs overlap")

    hierarchy_availability = {
        "curve_id": True,
        "device_id": any(curve.metadata.get("device_id") is not None for curve in curves),
        "repetition_id": any(
            curve.metadata.get("repetition_id") is not None for curve in curves
        ),
    }
    for hierarchy_key, available in hierarchy_availability.items():
        if not available:
            continue
        def identities(role_curves: list[MeasuredCurve]) -> set[str]:
            if hierarchy_key == "curve_id":
                return {curve.curve_id for curve in role_curves}
            return {
                str(curve.metadata[hierarchy_key]) for curve in role_curves
                if curve.metadata.get(hierarchy_key) is not None
            }
        if identities(tuning) & identities(evaluation):
            raise ValueError(f"HSC {hierarchy_key} crosses tuning/evaluation roles")

    tuning_rows = [_manifest_curve_row(curve, source_sha256) for curve in tuning]
    evaluation_rows = [_manifest_curve_row(curve, source_sha256) for curve in evaluation]
    curve_manifest = {
        "schema_version": "1.0.0",
        "protocol_key": config["protocol_key"],
        "task": HSC_SOURCE_KEY,
        "selection_method": "hsc-condition-stratification-v1",
        "hierarchy_availability": hierarchy_availability,
        "tuning_curve_ids": [row["curve_id"] for row in tuning_rows],
        "evaluation_curve_ids": [row["curve_id"] for row in evaluation_rows],
        "tuning": tuning_rows,
        "evaluation": evaluation_rows,
    }
    return {
        "schema_version": "1.0.0",
        "protocol_key": config["protocol_key"],
        "source": {
            "key": HSC_SOURCE_KEY,
            "provider": source_spec["provider"],
            "article_id": HSC_ARTICLE_ID,
            "version": HSC_VERSION,
            "api_url": source_spec["api_url"],
            "license": source_spec["license"],
            "expected_file_role": source_spec["expected_file_role"],
            "source_file": f"{HSC_SOURCE_KEY}/{HSC_FILENAME}",
            "source_file_sha256": source_sha256,
            "source_file_size": source_path.stat().st_size,
            "figshare_file_id": int(expected_data_file["file_id"]),
            "figshare_md5": source_md5,
        },
        "curve_count": len(curves),
        "condition_counts": {
            "temperature_c": len(temperatures),
            "soc_percent": len(soc_values),
            "temperature_soc_pairs": len(pairs),
        },
        "point_count_distribution": {
            str(count): int(number) for count, number in sorted(point_counts.items())
        },
        "curve_manifest": curve_manifest,
    }


def _root_relative(path: Path) -> str:
    """Return a stable workspace-relative path for provenance records."""
    try:
        return path.resolve().relative_to(ROOT.parent.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _frf_task_config() -> dict:
    config = json.loads((ROOT / "peak_sensitive_config.json").read_text())
    task_config = config["tasks"][FRF_TASK]
    validation_by_density = {
        int(density): int(count)
        for density, count in task_config["validation_by_density"].items()
    }
    if validation_by_density != FRF_VALIDATION_BY_DENSITY:
        raise ValueError(
            "Aluminium FRF validation_by_density must map 24/48/96/192 to 12/24/48/96"
        )
    if task_config.get("density_axis") != list(FRF_DENSITIES):
        raise ValueError("Aluminium FRF density axis differs from the frozen Task6 design")
    return task_config


def _frf_source_manifest(raw_root: Path) -> tuple[dict, Path, str]:
    path = Path(raw_root) / "source_manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload.get("files"), list):
        raise ValueError("FRF source manifest has no file inventory")
    return payload, path, _sha256(path)


def _manifest_path_for_curve(source_file: str) -> str:
    source_file = str(source_file).replace("\\", "/")
    return source_file if source_file.startswith(f"{FRF_TASK}/") else f"{FRF_TASK}/{source_file}"


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verified_frf_tdms_paths(raw_root: Path) -> tuple[dict, Path, str, list[tuple[Path, dict]]]:
    """Validate all released TDMS inputs before dense FRF preprocessing."""
    source_manifest, source_manifest_path, source_manifest_sha256 = _frf_source_manifest(raw_root)
    expected_rows = {
        str(row["path"]): row
        for row in source_manifest["files"]
        if isinstance(row, dict)
        and re.fullmatch(r"aluminium_frf/Waveforms Point\d+\.tdms", str(row.get("path", "")))
    }
    directory = Path(raw_root) / FRF_TASK
    actual_paths = sorted(directory.glob("Waveforms Point*.tdms"))
    actual_manifest_paths = {_manifest_path_for_curve(path.name) for path in actual_paths}
    if len(expected_rows) != 25 or actual_manifest_paths != set(expected_rows):
        raise ValueError("Aluminium TDMS inventory does not match the pinned 25-file manifest")

    verified_paths: list[tuple[Path, dict]] = []
    for path in actual_paths:
        manifest_path = _manifest_path_for_curve(path.name)
        row = expected_rows[manifest_path]
        if row.get("status") != "verified":
            raise ValueError(f"Aluminium TDMS source is not verified: {manifest_path}")
        expected_sha256 = str(row.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError(f"Aluminium TDMS source has no pinned SHA-256: {manifest_path}")
        if _sha256(path) != expected_sha256:
            raise ValueError(f"Aluminium TDMS source SHA-256 mismatch: {manifest_path}")
        verified_paths.append((path, row))
    return source_manifest, source_manifest_path, source_manifest_sha256, verified_paths


def dense_frf_domain_indices(frequency: np.ndarray) -> np.ndarray:
    """Select the frozen, response-independent dense FRF domain."""
    frequency = np.asarray(frequency, dtype=np.float64).reshape(-1)
    return np.flatnonzero(np.isfinite(frequency) & (frequency > 0.0))


@lru_cache(maxsize=2)
def _load_dense_aluminium_frf_cached(
    raw_root_text: str, source_manifest_sha256: str
) -> tuple[MeasuredCurve, ...]:
    """Compute the frozen H1 representation after source integrity checks."""
    del source_manifest_sha256  # The cache key invalidates when the manifest changes.
    from nptdms import TdmsFile
    from scipy.signal import csd, welch

    raw_root = Path(raw_root_text)
    _, _, _, verified_paths = _verified_frf_tdms_paths(raw_root)
    estimator_sha256 = _canonical_sha256(FRF_ESTIMATOR_PARAMETERS)
    curves: list[MeasuredCurve] = []
    for path, source_row in verified_paths:
        point_match = re.search(r"Point(\d+)", path.name)
        if point_match is None:
            raise ValueError(f"unable to derive point ID from {path.name}")
        point = int(point_match.group(1))
        tdms = TdmsFile.read(path)
        channels = [channel for group in tdms.groups() for channel in group.channels()]
        force_channel = next(
            channel for channel in channels
            if "Force" in str(channel.properties.get("NI_SV_SensorType", ""))
        )
        acceleration_channel = next(
            channel for channel in channels
            if "Accelerometer" in str(channel.properties.get("NI_SV_SensorType", ""))
        )
        force = np.asarray(force_channel[:], dtype=np.float64)
        acceleration = np.asarray(acceleration_channel[:], dtype=np.float64) * 9.81
        if len(force) != len(acceleration) or len(force) < FRF_ESTIMATOR_PARAMETERS["nperseg"]:
            raise ValueError(f"insufficient aligned TDMS samples in {path.name}")
        sampling_rate = 1.0 / float(force_channel.properties["wf_increment"])
        frequency, cross_spectrum = csd(
            force,
            acceleration,
            fs=sampling_rate,
            window=FRF_ESTIMATOR_PARAMETERS["window"],
            nperseg=FRF_ESTIMATOR_PARAMETERS["nperseg"],
            noverlap=FRF_ESTIMATOR_PARAMETERS["noverlap"],
            detrend=FRF_ESTIMATOR_PARAMETERS["detrend"],
            scaling=FRF_ESTIMATOR_PARAMETERS["scaling"],
        )
        _, force_spectrum = welch(
            force,
            fs=sampling_rate,
            window=FRF_ESTIMATOR_PARAMETERS["window"],
            nperseg=FRF_ESTIMATOR_PARAMETERS["nperseg"],
            noverlap=FRF_ESTIMATOR_PARAMETERS["noverlap"],
            detrend=FRF_ESTIMATOR_PARAMETERS["detrend"],
            scaling=FRF_ESTIMATOR_PARAMETERS["scaling"],
        )
        floor = np.finfo(float).eps * max(float(np.max(force_spectrum)), 1.0)
        response = cross_spectrum / np.maximum(force_spectrum, floor)
        domain_indices = dense_frf_domain_indices(frequency)
        frequency = np.asarray(frequency[domain_indices], dtype=np.float64)
        response = np.column_stack((response.real, response.imag))[domain_indices]
        if not np.all(np.isfinite(response)):
            raise ValueError(f"nonfinite H1 response in {path.name}")
        curves.append(MeasuredCurve(
            curve_id=f"point:{point:02d}",
            frequency=frequency,
            response=np.asarray(response, dtype=np.float32),
            conditions={"point": point},
            source_files=(path.name,),
            metadata={
                "source_record": 7758683,
                "source_status": source_row["status"],
                "source_file_sha256": source_row["sha256"],
                "source_sha256": source_row["sha256"],
                "frequency_unit": "Hz",
                "response_unit": "m_per_s2_per_N",
                "frf_estimator": "H1",
                "frf_estimator_version": FRF_ESTIMATOR_VERSION,
                "frf_estimator_parameters": dict(FRF_ESTIMATOR_PARAMETERS),
                "frf_estimator_parameters_sha256": estimator_sha256,
                "sampling_rate_hz": sampling_rate,
                "nperseg": FRF_ESTIMATOR_PARAMETERS["nperseg"],
                "noverlap": FRF_ESTIMATOR_PARAMETERS["noverlap"],
                "window": FRF_ESTIMATOR_PARAMETERS["window"],
                "domain_rule": FRF_DOMAIN_RULE,
                "domain_point_count": int(len(frequency)),
            },
        ))
    return tuple(curves)


def load_dense_aluminium_frf(raw_root: Path) -> list[MeasuredCurve]:
    """Load all finite positive H1 frequency bins from verified raw TDMS files."""
    raw_root = Path(raw_root)
    _, _, source_manifest_sha256, _ = _verified_frf_tdms_paths(raw_root)
    return list(_load_dense_aluminium_frf_cached(str(raw_root.resolve()), source_manifest_sha256))


def _frf_source_selection_key(record: dict) -> str:
    return hashlib.sha256(
        (
            "frf-density-source-selection-v1:"
            f"{record['curve_id']}:{record['source_sha256']}"
        ).encode("ascii")
    ).hexdigest()


def select_frf_curve_roles(
    inventory: list[dict],
    *,
    tuning_count: int = FRF_TUNING_CURVE_COUNT,
    evaluation_count: int = FRF_EVALUATION_CURVE_COUNT,
) -> dict:
    """Assign frozen FRF roles from source metadata only, with deterministic fallback."""
    if tuning_count < 1 or evaluation_count < 1:
        raise ValueError("FRF role counts must be positive")
    ordered = sorted(inventory, key=_frf_source_selection_key)
    selected: dict[str, list[str]] = {"tuning": [], "evaluation": []}
    replacements: list[dict] = []
    pending_exclusions: list[dict] = []
    cursor = 0
    for role, required_count in (("tuning", tuning_count), ("evaluation", evaluation_count)):
        while len(selected[role]) < required_count:
            if cursor == len(ordered):
                raise ValueError(
                    "FRF source inventory has fewer eligible curves than the frozen role count"
                )
            candidate = ordered[cursor]
            cursor += 1
            if not candidate["eligible"]:
                pending_exclusions.append(candidate)
                continue
            selected[role].append(candidate["curve_id"])
            for excluded in pending_exclusions:
                replacements.append({
                    "role": role,
                    "excluded_curve_id": excluded["curve_id"],
                    "replacement_curve_id": candidate["curve_id"],
                    "reason": excluded["exclusion_reason"],
                })
            pending_exclusions.clear()
    return {
        "selection_method": FRF_ROLE_SELECTION_METHOD,
        "tuning_curve_ids": selected["tuning"],
        "evaluation_curve_ids": selected["evaluation"],
        "selected_replacements": replacements,
    }


def audit_frf_density_sources(
    raw_root: Path,
    *,
    output_path: Path | None = None,
) -> dict:
    """Audit the frozen FRF inventory before any model-dependent selection."""
    raw_root = Path(raw_root)
    task_config = _frf_task_config()
    source_manifest, source_manifest_path, source_manifest_sha256 = _frf_source_manifest(raw_root)
    _, _, _, verified_paths = _verified_frf_tdms_paths(raw_root)
    verified_files = {
        _manifest_path_for_curve(path.name): row for path, row in verified_paths
    }
    curves = load_dense_aluminium_frf(raw_root)
    curve_ids = [curve.curve_id for curve in curves]
    if len(set(curve_ids)) != len(curve_ids):
        raise ValueError("duplicate aluminium FRF curve IDs")

    inventory: list[dict] = []
    eligible_curve_ids: list[str] = []
    excluded_curves: list[dict] = []
    for curve in curves:
        source_paths = [_manifest_path_for_curve(source) for source in curve.source_files]
        if len(source_paths) != 1 or source_paths[0] not in verified_files:
            raise ValueError(f"FRF curve {curve.curve_id} has no verified TDMS source")
        source_row = verified_files[source_paths[0]]
        eligible = len(curve.frequency) >= FRF_MINIMUM_POINTS
        reason = None if eligible else "insufficient_points"
        row = {
            "curve_id": curve.curve_id,
            "point_count": int(len(curve.frequency)),
            "source_files": source_paths,
            "source_status": source_row["status"],
            "source_sha256": source_row["sha256"],
            "eligible": eligible,
            "exclusion_reason": reason,
        }
        inventory.append(row)
        if eligible:
            eligible_curve_ids.append(curve.curve_id)
        else:
            excluded_curves.append({
                "curve_id": curve.curve_id,
                "point_count": int(len(curve.frequency)),
                "reason": reason,
                "source_files": source_paths,
            })

    selection = select_frf_curve_roles(inventory)

    audit = {
        "schema_version": "2.0.0",
        "task": FRF_TASK,
        "source_manifest_path": _root_relative(source_manifest_path),
        "source_manifest_sha256": source_manifest_sha256,
        "curve_count": len(curves),
        "curve_ids": curve_ids,
        "minimum_points": FRF_MINIMUM_POINTS,
        "parent_train_points": FRF_PARENT_TRAIN_POINTS,
        "validation_points": FRF_VALIDATION_POINTS,
        "validation_by_density": task_config["validation_by_density"],
        "minimum_test_points": 1,
        "role_selection": {
            "tuning_curve_count": FRF_TUNING_CURVE_COUNT,
            "evaluation_curve_count": FRF_EVALUATION_CURVE_COUNT,
            "selection_method": FRF_ROLE_SELECTION_METHOD,
            "replacement_rule": FRF_REPLACEMENT_RULE,
        },
        "dense_loader": {
            "estimator_parameters": dict(FRF_ESTIMATOR_PARAMETERS),
            "estimator_parameters_sha256": _canonical_sha256(FRF_ESTIMATOR_PARAMETERS),
            "domain_rule": FRF_DOMAIN_RULE,
        },
        "inventory": inventory,
        "eligible_curve_ids": eligible_curve_ids,
        "excluded_curves": excluded_curves,
        "selection_method": selection["selection_method"],
        "tuning_curve_ids": selection["tuning_curve_ids"],
        "evaluation_curve_ids": selection["evaluation_curve_ids"],
        "replacement_policy": {
            "rule": FRF_REPLACEMENT_RULE,
            "performance_replacement_allowed": False,
            "selected_replacements": selection["selected_replacements"],
        },
    }
    if output_path is not None:
        _write_json(Path(output_path), audit)
    return audit


def _frf_parent_seed(curve_id: str, data_seed: int, role: str) -> int:
    digest = hashlib.sha256(
        f"frf-density-{role}-parent-v2:{curve_id}:{int(data_seed)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _frf_point_id(curve_id: str, index: int) -> str:
    return f"{curve_id}:frequency:{int(index)}"


def _point_id_sha256(point_ids: list[str]) -> str:
    return _canonical_sha256(point_ids)


def make_frf_density_bundle(
    curve_id: str,
    data_seed: int,
    n_train: int,
    raw_root: Path,
    *,
    visibility: str = "opened",
) -> DatasetBundle:
    """Build one nested, within-curve Aluminium FRF density bundle."""
    if visibility not in {"sealed", "opened"}:
        raise ValueError("visibility must be sealed or opened")
    if int(n_train) not in FRF_DENSITIES:
        raise ValueError(f"FRF density must be one of {FRF_DENSITIES}")
    n_train = int(n_train)
    validation_count = int(_frf_task_config()["validation_by_density"][str(n_train)])
    audit = audit_frf_density_sources(raw_root)
    curves = load_dense_aluminium_frf(Path(raw_root))
    by_id = {curve.curve_id: curve for curve in curves}
    if curve_id not in by_id:
        raise ValueError(f"unknown frozen Aluminium FRF curve ID: {curve_id}")
    if curve_id not in audit["eligible_curve_ids"]:
        excluded = next(row for row in audit["excluded_curves"] if row["curve_id"] == curve_id)
        raise ValueError(
            f"FRF curve {curve_id} requires at least {FRF_MINIMUM_POINTS} points; "
            f"source-manifest audit excluded it for {excluded['reason']} "
            f"({excluded['point_count']} points)"
        )

    curve = by_id[curve_id]
    point_count = len(curve.frequency)
    train_rng = np.random.default_rng(_frf_parent_seed(curve_id, data_seed, "train"))
    parent_train_indices = train_rng.permutation(point_count)[:FRF_PARENT_TRAIN_POINTS]
    train_set = {int(index) for index in parent_train_indices}
    validation_pool = np.asarray(
        [index for index in range(point_count) if index not in train_set], dtype=np.int64
    )
    validation_rng = np.random.default_rng(_frf_parent_seed(curve_id, data_seed, "validation"))
    parent_validation_indices = validation_pool[
        validation_rng.permutation(len(validation_pool))[:FRF_VALIDATION_POINTS]
    ]
    train_indices = parent_train_indices[:n_train]
    validation_indices = parent_validation_indices[:validation_count]
    selected = set(int(index) for index in train_indices) | {
        int(index) for index in validation_indices
    }
    test_indices = np.asarray(
        [index for index in range(point_count) if index not in selected],
        dtype=np.int64,
    )
    point_ids = lambda indices: [_frf_point_id(curve_id, index) for index in indices]
    parent_train_point_ids = point_ids(parent_train_indices)
    parent_validation_point_ids = point_ids(parent_validation_indices)
    validation_point_ids = point_ids(validation_indices)
    parent_mask_payload = {
        "curve_id": curve_id,
        "data_seed": int(data_seed),
        "parent_train_point_ids": parent_train_point_ids,
        "parent_validation_point_ids": parent_validation_point_ids,
    }
    parent_mask_hash = hashlib.sha256(
        json.dumps(parent_mask_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "task": FRF_TASK,
        "curve_id": curve_id,
        "data_seed": int(data_seed),
        "density": n_train,
        "split_strategy": "disjoint_parent_orders_nested_train_validation_prefix",
        "parent_mask_hash": parent_mask_hash,
        "parent_train_point_ids": parent_train_point_ids,
        "parent_train_order_sha256": _point_id_sha256(parent_train_point_ids),
        "parent_validation_point_ids": parent_validation_point_ids,
        "parent_validation_order_sha256": _point_id_sha256(parent_validation_point_ids),
        "train_point_ids": point_ids(train_indices),
        "validation_point_ids": validation_point_ids,
        "test_point_ids": point_ids(test_indices),
        "source_manifest_sha256": audit["source_manifest_sha256"],
        "source_files": list(curve.source_files),
        "conditions": curve.conditions,
        "curve_metadata": curve.metadata,
        "frequency_dtype_before_standardization": "float64",
        "minimum_points": FRF_MINIMUM_POINTS,
        "validation_count": validation_count,
        "test_target": "held_out_measured_frequencies",
    }
    return DatasetBundle(
        x_train=np.asarray(curve.frequency[train_indices], dtype=np.float64).reshape(-1, 1),
        y_train=np.asarray(curve.response[train_indices], dtype=np.float32),
        x_validation=np.asarray(curve.frequency[validation_indices], dtype=np.float64).reshape(-1, 1),
        y_validation=np.asarray(curve.response[validation_indices], dtype=np.float32),
        x_test=(
            np.asarray(curve.frequency[test_indices], dtype=np.float64).reshape(-1, 1)
            if visibility == "opened" else np.empty((0, 1), dtype=np.float64)
        ),
        y_test=(
            np.asarray(curve.response[test_indices], dtype=np.float32)
            if visibility == "opened" else np.empty((0, 2), dtype=np.float32)
        ),
        metadata=metadata,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--frf-density-audit", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "peak_sensitive_config.json")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "downstream_data" / "raw")
    parser.add_argument(
        "--output", type=Path,
        default=None,
    )
    parser.add_argument(
        "--curve-manifest-output", type=Path,
        default=ROOT / "peak_sensitive_results" / "provenance" / "curve_manifest.json",
    )
    args = parser.parse_args()
    if args.audit == args.frf_density_audit:
        parser.error("select exactly one of --audit or --frf-density-audit")
    if args.frf_density_audit:
        output = args.output or ROOT / "peak_sensitive_results" / "provenance" / "frf_density_manifest.json"
        audit = audit_frf_density_sources(args.raw_root, output_path=output)
        print(json.dumps({
            "curve_count": audit["curve_count"],
            "eligible_curve_count": len(audit["eligible_curve_ids"]),
            "tuning_curve_ids": audit["tuning_curve_ids"],
            "evaluation_curve_ids": audit["evaluation_curve_ids"],
            "output": output.as_posix(),
        }, sort_keys=True))
        return
    output = args.output or ROOT / "peak_sensitive_results" / "provenance" / "source_audit.json"
    audit = audit_peak_sensitive_sources(args.raw_root, json.loads(args.config.read_text()))
    _write_json(output, audit)
    _write_json(args.curve_manifest_output, audit["curve_manifest"])
    print(json.dumps({
        "curve_count": audit["curve_count"],
        "output": output.as_posix(),
        "curve_manifest_output": args.curve_manifest_output.as_posix(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

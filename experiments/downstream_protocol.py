"""Dataset construction for the CFNN downstream validation study."""
from __future__ import annotations

import hashlib
import json
import re
import argparse
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from itertools import product
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from confirmatory_protocol import DatasetBundle


@dataclass(frozen=True)
class MeasuredCurve:
    curve_id: str
    frequency: np.ndarray
    response: np.ndarray
    conditions: dict
    source_files: tuple[str, ...]
    metadata: dict


def _fano_response(frequency: np.ndarray, parameters: dict) -> np.ndarray:
    response = (
        parameters["background_real"]
        + 1j * parameters["background_imag"]
        + parameters["background_slope"] * (frequency - 0.5)
    ).astype(np.complex128)
    centers = [0.5] if parameters["separation"] == 0.0 else [
        0.5 - parameters["separation"] / 2.0,
        0.5 + parameters["separation"] / 2.0,
    ]
    for index, center in enumerate(centers):
        width = parameters["width"] * (1.0 + 0.25 * index)
        q = parameters["q"] if index == 0 else -0.5 * parameters["q"]
        amplitude = 1.0 if index == 0 else 0.65
        z = (frequency - center) / width
        response = response + amplitude * ((q + z) / (z + 1j) - 1.0)
    return response


def _fano_parameters(data_seed: int) -> tuple[dict, str, int]:
    schedule = list(product(
        (0.018, 0.035, 0.070),
        (-2.0, -0.75, 0.75, 2.0),
        (0.0, 0.08, 0.16),
        (0.0, 0.12),
    ))
    schedule_index = (int(data_seed) * 2654435761) % len(schedule)
    width, q, separation, background_slope = schedule[schedule_index]
    rng = np.random.default_rng(int(data_seed) + 15485863)
    parameters = {
        "width": float(width),
        "q": float(q),
        "separation": float(separation),
        "background_slope": float(background_slope),
        "background_real": float(rng.uniform(-0.08, 0.08)),
        "background_imag": float(rng.uniform(-0.08, 0.08)),
    }
    encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
    return parameters, hashlib.sha256(encoded).hexdigest(), int(schedule_index)


def _as_two_columns(response: np.ndarray) -> np.ndarray:
    return np.column_stack((response.real, response.imag)).astype(np.float32)


def make_controlled_fano_bundle(data_seed: int, config: dict, *,
                                visibility: str = "opened") -> DatasetBundle:
    """Build one noisy sparse Fano reconstruction task and a clean dense test."""
    if visibility not in {"sealed", "opened"}:
        raise ValueError("visibility must be sealed or opened")
    n_train = int(config["n_train"])
    n_validation = int(config["n_validation"])
    n_test = int(config["n_test"])
    noise = float(config["noise"])
    lower, upper = map(float, config.get("domain", (0.0, 1.0)))
    if not lower < upper:
        raise ValueError("Fano domain must have positive width")
    if min(n_train, n_validation, n_test) < 2:
        raise ValueError("Fano splits require at least two points")

    parameters, parameter_key, schedule_index = _fano_parameters(data_seed)
    rng = np.random.default_rng(int(data_seed))
    observed = np.sort(rng.uniform(lower, upper, size=n_train + n_validation))
    order = rng.permutation(len(observed))
    train_indices = np.sort(order[:n_train])
    validation_indices = np.sort(order[n_train:])
    x_train = observed[train_indices]
    x_validation = observed[validation_indices]
    opened = visibility == "opened"
    x_test = (
        np.linspace(lower, upper, n_test, dtype=np.float64)
        if opened else np.empty(0, dtype=np.float64)
    )

    train_clean = _fano_response(x_train, parameters)
    validation_clean = _fano_response(x_validation, parameters)
    test_clean = _fano_response(x_test, parameters) if opened else None
    scale = max(float(np.sqrt(np.mean(np.abs(train_clean) ** 2))), 1e-8)
    train_noise = noise * scale * (
        rng.standard_normal(n_train) + 1j * rng.standard_normal(n_train)
    ) / np.sqrt(2.0)
    validation_noise = noise * scale * (
        rng.standard_normal(n_validation) + 1j * rng.standard_normal(n_validation)
    ) / np.sqrt(2.0)

    return DatasetBundle(
        x_train=x_train.astype(np.float32).reshape(-1, 1),
        y_train=_as_two_columns(train_clean + train_noise),
        x_validation=x_validation.astype(np.float32).reshape(-1, 1),
        y_validation=_as_two_columns(validation_clean + validation_noise),
        x_test=x_test.astype(np.float32).reshape(-1, 1),
        y_test=(
            _as_two_columns(test_clean)
            if opened else np.empty((0, 2), dtype=np.float32)
        ),
        metadata={
            "task": "controlled_fano",
            "data_seed": int(data_seed),
            "parameter_key": parameter_key,
            "schedule_index": schedule_index,
            "parameters": parameters,
            "noise": noise,
            "split_strategy": "sparse_noisy_points_to_clean_dense_grid",
            "test_target": "clean_dense_same_response",
            "train_point_ids": [f"observed:{int(index)}" for index in train_indices],
            "validation_point_ids": [f"observed:{int(index)}" for index in validation_indices],
            "test_point_ids": [f"dense:{index}" for index in range(n_test)],
        },
    )


@lru_cache(maxsize=2)
def load_microwave_fano(raw_root: Path) -> list[MeasuredCurve]:
    """Read released power-sweep magnitude/phase measurements from Zenodo 7767046."""
    archive = Path(raw_root) / "microwave_fano" / "Fano Interference Data and Code.zip"
    if not archive.exists():
        raise FileNotFoundError(archive)
    curves = []
    with ZipFile(archive) as container:
        members = sorted(
            name for name in container.namelist()
            if name.endswith(".npz") and "powersweep" in name
        )
        for member in members:
            with np.load(BytesIO(container.read(member)), allow_pickle=False) as data:
                frequency = np.asarray(data["frequency"], dtype=np.float64).reshape(-1)
                amplitude = np.asarray(data["amplitude"], dtype=np.float64)
                phase = np.asarray(data["phase"], dtype=np.float64)
                powers = np.asarray(data["power"], dtype=np.float64).reshape(-1)
            if amplitude.shape != phase.shape or amplitude.shape != (len(powers), len(frequency)):
                raise ValueError(f"unexpected Fano power-sweep shape in {member}")
            coupling = "overcoupled" if "/overcoupled/" in member else "undercoupled"
            match = re.search(r"resonator_(\d+)", member)
            if not match:
                raise ValueError(f"cannot identify resonator in {member}")
            resonator = int(match.group(1))
            for power_index, power in enumerate(powers):
                complex_response = amplitude[power_index] * np.exp(1j * phase[power_index])
                curves.append(MeasuredCurve(
                    curve_id=f"{coupling}:r{resonator}:p{power_index:02d}",
                    frequency=frequency.copy(),
                    response=_as_two_columns(complex_response),
                    conditions={
                        "coupling": coupling,
                        "resonator": resonator,
                        "power_dbm": float(power),
                    },
                    source_files=(archive.name,),
                    metadata={
                        "source_record": 7767046,
                        "archive_member": member,
                        "response_encoding": "amplitude_exp_i_phase",
                        "frequency_unit": "Hz",
                    },
                ))
    if not curves:
        raise ValueError("no microwave Fano power-sweep curves found")
    return curves


@lru_cache(maxsize=2)
def load_microstrip_resonator(raw_root: Path) -> list[MeasuredCurve]:
    """Read raw measured S12 curves in the frozen 2 GHz resonance window."""
    archive = Path(raw_root) / "microstrip_resonator" / "Dataset.zip"
    if not archive.exists():
        raise FileNotFoundError(archive)
    curves = []
    with ZipFile(archive) as container:
        members = sorted(
            name for name in container.namelist()
            if name.startswith("04_data_microstrip_line/Raw data/")
            and name.lower().endswith(".s2p")
        )
        for member in members:
            values = np.loadtxt(
                BytesIO(container.read(member)), delimiter=",", comments="#"
            )
            if values.ndim != 2 or values.shape[1] != 9:
                raise ValueError(f"unexpected Touchstone table shape in {member}")
            frequency = values[:, 0]
            keep = (frequency >= 1.6e9) & (frequency <= 2.4e9)
            if int(np.count_nonzero(keep)) < 1800:
                raise ValueError(f"unexpected 2 GHz resonance grid in {member}")
            filename = Path(member).name
            condition_match = re.search(
                r"_T(-?\d+(?:\.\d+)?)_H(\d+(?:\.\d+)?)_", filename
            )
            if not condition_match:
                raise ValueError(f"cannot parse environmental conditions from {filename}")
            sample_id = Path(member).parent.name
            cycle = int(filename.split("_", 1)[0])
            curves.append(MeasuredCurve(
                curve_id=f"{sample_id}:cycle:{cycle:03d}",
                frequency=frequency[keep].astype(np.float64),
                response=values[keep, 5:7].astype(np.float32),
                conditions={
                    "sample_id": sample_id,
                    "cycle": cycle,
                    "temperature_c": float(condition_match.group(1)),
                    "relative_humidity_percent": float(condition_match.group(2)),
                },
                source_files=(archive.name,),
                metadata={
                    "source_record": 14175959,
                    "archive_member": member,
                    "frequency_unit": "Hz",
                    "s_parameter": "S12",
                    "response_encoding": "real_imaginary",
                    "frozen_frequency_interval_hz": [1.6e9, 2.4e9],
                    "native_frequency_step_hz": float(np.median(np.diff(frequency))),
                    "window_point_count": int(np.count_nonzero(keep)),
                    "interval_rationale": "contains_the_reported_2GHz_humidity_resonator",
                },
            ))
    if len(curves) != 270:
        raise ValueError(f"expected 270 raw microstrip curves, found {len(curves)}")
    return curves


def _frf_interval(curves: list[MeasuredCurve]) -> tuple[float, float, list[float]]:
    from scipy.signal import find_peaks

    frequency = curves[0].frequency
    magnitude_db = np.asarray([
        20.0 * np.log10(np.maximum(np.linalg.norm(curve.response, axis=1), 1e-20))
        for curve in curves
    ])
    median_db = np.median(magnitude_db, axis=0)
    eligible = frequency >= max(5.0, float(frequency[1]))
    eligible_indices = np.flatnonzero(eligible)
    local_peaks, _ = find_peaks(median_db[eligible], prominence=6.0, distance=10)
    peaks = eligible_indices[local_peaks]
    if len(peaks) < 3:
        raise ValueError(f"FRF audit found only {len(peaks)} peaks with the frozen criterion")
    end = (
        float((frequency[peaks[2]] + frequency[peaks[3]]) / 2.0)
        if len(peaks) > 3 else float(frequency[-1])
    )
    return float(frequency[eligible_indices[0]]), end, [float(frequency[index]) for index in peaks]


@lru_cache(maxsize=2)
def load_aluminium_frf(raw_root: Path) -> list[MeasuredCurve]:
    """Construct H1 accelerance FRFs from the 25 released impact measurements."""
    from nptdms import TdmsFile
    from scipy.signal import csd, welch

    directory = Path(raw_root) / "aluminium_frf"
    paths = sorted(directory.glob("Waveforms Point*.tdms"))
    if len(paths) != 25:
        raise ValueError(f"expected 25 aluminium TDMS files, found {len(paths)}")
    full_curves = []
    for path in paths:
        point_match = re.search(r"Point(\d+)", path.name)
        point = int(point_match.group(1)) if point_match else -1
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
        sample_interval = float(force_channel.properties["wf_increment"])
        sampling_rate = 1.0 / sample_interval
        nperseg = min(4096, len(force))
        noverlap = nperseg // 2
        frequency, cross_spectrum = csd(
            force, acceleration, fs=sampling_rate, window="hann",
            nperseg=nperseg, noverlap=noverlap, detrend="constant",
            scaling="spectrum",
        )
        _, force_spectrum = welch(
            force, fs=sampling_rate, window="hann", nperseg=nperseg,
            noverlap=noverlap, detrend="constant", scaling="spectrum",
        )
        floor = np.finfo(float).eps * max(float(np.max(force_spectrum)), 1.0)
        response = cross_spectrum / np.maximum(force_spectrum, floor)
        full_curves.append(MeasuredCurve(
            curve_id=f"point:{point:02d}",
            frequency=frequency.astype(np.float64),
            response=_as_two_columns(response),
            conditions={"point": point},
            source_files=(path.name,),
            metadata={
                "source_record": 7758683,
                "frequency_unit": "Hz",
                "response_unit": "m_per_s2_per_N",
                "frf_estimator": "H1",
                "sampling_rate_hz": sampling_rate,
                "nperseg": nperseg,
                "noverlap": noverlap,
                "window": "hann",
            },
        ))
    lower, upper, audited_peaks = _frf_interval(full_curves)
    curves = []
    for curve in full_curves:
        keep = (curve.frequency >= lower) & (curve.frequency <= upper)
        metadata = {
            **curve.metadata,
            "frozen_frequency_interval_hz": [lower, upper],
            "audit_peak_frequencies_hz": audited_peaks,
            "interval_rule": "first_three_median_log_frf_peaks_prominence_6db_distance_10bins",
        }
        curves.append(MeasuredCurve(
            curve_id=curve.curve_id,
            frequency=curve.frequency[keep],
            response=curve.response[keep],
            conditions=curve.conditions,
            source_files=curve.source_files,
            metadata=metadata,
        ))
    return curves


def audit_measured_sources(raw_root: Path) -> dict:
    """Summarize frozen measured inputs without executing any model."""
    raw_root = Path(raw_root)
    loaders = {
        "microwave_fano": load_microwave_fano,
        "microstrip_resonator": load_microstrip_resonator,
        "aluminium_frf": load_aluminium_frf,
    }
    tasks = {}
    for task, loader in loaders.items():
        curves = loader(raw_root)
        if len({curve.curve_id for curve in curves}) != len(curves):
            raise ValueError(f"duplicate measured curve IDs for {task}")
        if not all(
            np.all(np.isfinite(curve.frequency))
            and np.all(np.diff(curve.frequency) > 0)
            and np.all(np.isfinite(curve.response))
            and curve.response.shape == (len(curve.frequency), 2)
            for curve in curves
        ):
            raise ValueError(f"invalid measured curve in {task}")
        point_counts = Counter(len(curve.frequency) for curve in curves)
        tasks[task] = {
            "curve_count": len(curves),
            "point_count_distribution": {
                str(count): int(frequency) for count, frequency in sorted(point_counts.items())
            },
            "frequency_min_hz": float(min(curve.frequency[0] for curve in curves)),
            "frequency_max_hz": float(max(curve.frequency[-1] for curve in curves)),
            "condition_fields": sorted({
                key for curve in curves for key in curve.conditions
            }),
            "source_records": sorted({
                int(curve.metadata["source_record"]) for curve in curves
            }),
        }
    source_manifest = raw_root / "source_manifest.json"
    if not source_manifest.exists():
        raise FileNotFoundError(source_manifest)
    return {
        "schema_version": "1.0.0",
        "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        "tasks": tasks,
    }


def aluminium_point_coordinates() -> dict[int, tuple[float, float]]:
    """Return the released 5-by-5 impact grid in metres."""
    x_values = (0.0, 0.065, 0.13, 0.195, 0.26)
    y_values = (0.0, 0.0615, 0.123, 0.1845, 0.246)
    return {
        row * 5 + column + 1: (x, y)
        for row, y in enumerate(y_values)
        for column, x in enumerate(x_values)
    }


def _group_transfer_curves(task: str, raw_root: Path) -> list[MeasuredCurve]:
    if task == "microwave_fano":
        return load_microwave_fano(raw_root)
    if task == "microstrip_resonator":
        return load_microstrip_resonator(raw_root)
    if task == "aluminium_frf":
        return load_aluminium_frf(raw_root)
    raise ValueError(f"unknown measured group-transfer task: {task}")


def _condition_columns(task: str, curve: MeasuredCurve) -> tuple[float, ...]:
    if task == "microwave_fano":
        return (float(curve.conditions["power_dbm"]),)
    if task == "microstrip_resonator":
        return (
            float(curve.conditions["temperature_c"]),
            float(curve.conditions["relative_humidity_percent"]),
        )
    point = int(curve.conditions["point"])
    return aluminium_point_coordinates()[point]


def _group_transfer_stratum(task: str, curve: MeasuredCurve) -> tuple:
    if task == "microwave_fano":
        return (curve.conditions["coupling"], int(curve.conditions["resonator"]))
    if task == "microstrip_resonator":
        return (curve.conditions["sample_id"],)
    return ("plate",)


def _sample_curve_indices(
    curve: MeasuredCurve,
    count: int | None,
    seed: int,
    role: str,
) -> np.ndarray:
    if count is None or int(count) >= len(curve.frequency):
        return np.arange(len(curve.frequency), dtype=np.int64)
    encoded = f"{int(seed)}:{role}:{curve.curve_id}".encode()
    local_seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    rng = np.random.default_rng(local_seed)
    return np.sort(rng.choice(len(curve.frequency), size=int(count), replace=False))


def _stack_group_curves(
    task: str,
    curves: list[MeasuredCurve],
    *,
    frequency_count: int | None,
    data_seed: int,
    role: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    inputs = []
    targets = []
    point_ids = []
    for curve in curves:
        indices = _sample_curve_indices(curve, frequency_count, data_seed, role)
        conditions = np.asarray(_condition_columns(task, curve), dtype=np.float64)
        repeated = np.repeat(conditions.reshape(1, -1), len(indices), axis=0)
        inputs.append(np.column_stack((curve.frequency[indices], repeated)))
        targets.append(curve.response[indices])
        point_ids.extend(
            f"{curve.curve_id}:frequency:{int(index)}" for index in indices
        )
    return (
        np.concatenate(inputs, axis=0).astype(np.float64),
        np.concatenate(targets, axis=0).astype(np.float32),
        point_ids,
    )


def make_group_transfer_bundle(
    task: str,
    data_seed: int,
    config: dict,
    raw_root: Path,
) -> DatasetBundle:
    """Hold out complete measured curves while exposing physical conditions."""
    suffix = "_group_transfer"
    base_task = str(config.get("base_task") or task.removesuffix(suffix))
    if task != f"{base_task}{suffix}":
        raise ValueError(f"group-transfer task/config mismatch: {task}/{base_task}")
    curves = _group_transfer_curves(base_task, Path(raw_root))
    target = curves[int(data_seed) % len(curves)]
    stratum = _group_transfer_stratum(base_task, target)
    pool = [
        curve for curve in curves
        if curve.curve_id != target.curve_id
        and _group_transfer_stratum(base_task, curve) == stratum
    ]
    n_validation = int(config["validation_curve_count"])
    n_train = int(config["train_curve_count"])
    if len(pool) < n_validation + n_train:
        raise ValueError(
            f"insufficient group-transfer curves for {task}: "
            f"need {n_validation + n_train}, found {len(pool)}"
        )
    rng = np.random.default_rng(int(data_seed) + 86028121)
    order = rng.permutation(len(pool))
    validation_curves = [pool[index] for index in order[:n_validation]]
    train_curves = [
        pool[index] for index in order[n_validation:n_validation + n_train]
    ]
    train_count = (
        None if base_task == "aluminium_frf"
        else int(config["train_frequency_count"])
    )
    validation_count = (
        None if base_task == "aluminium_frf"
        else int(config["validation_frequency_count"])
    )
    x_train, y_train, train_ids = _stack_group_curves(
        base_task, train_curves, frequency_count=train_count,
        data_seed=data_seed, role="train",
    )
    x_validation, y_validation, validation_ids = _stack_group_curves(
        base_task, validation_curves, frequency_count=validation_count,
        data_seed=data_seed, role="validation",
    )
    x_test, y_test, test_ids = _stack_group_curves(
        base_task, [target], frequency_count=None,
        data_seed=data_seed, role="test",
    )
    expected_dim = int(config["input_dim"])
    if x_train.shape[1] != expected_dim:
        raise ValueError(
            f"group-transfer input dimension mismatch for {task}: "
            f"expected {expected_dim}, found {x_train.shape[1]}"
        )
    condition_encoding = {
        "microwave_fano": ["power_dbm"],
        "microstrip_resonator": ["temperature_c", "relative_humidity_percent"],
        "aluminium_frf": ["impact_x_m", "impact_y_m"],
    }[base_task]
    metadata = {
        "task": task,
        "base_task": base_task,
        "data_seed": int(data_seed),
        "split_mode": "group_transfer",
        "split_strategy": "complete_curve_holdout_within_physical_stratum",
        "stratum": list(stratum),
        "condition_encoding": condition_encoding,
        "condition_source": (
            "zenodo_record_7758683_released_5_by_5_grid"
            if base_task == "aluminium_frf" else "released_curve_metadata"
        ),
        "train_curve_ids": [curve.curve_id for curve in train_curves],
        "validation_curve_ids": [curve.curve_id for curve in validation_curves],
        "test_curve_ids": [target.curve_id],
        "train_point_ids": train_ids,
        "validation_point_ids": validation_ids,
        "test_point_ids": test_ids,
        "test_target": "complete_held_out_measured_condition_curve",
        "conditions": target.conditions,
        "curve_metadata": target.metadata,
        "source_files": sorted({
            source for curve in train_curves + validation_curves + [target]
            for source in curve.source_files
        }),
    }
    return DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
        metadata=metadata,
    )


def make_measured_bundle(
    task: str,
    data_seed: int,
    config: dict,
    raw_root: Path,
    split_mode: str,
) -> DatasetBundle:
    """Create a deterministic measured-response split without point leakage."""
    if split_mode != "within_curve":
        raise ValueError("only within_curve is available until grouped condition encoders are frozen")
    if task == "microwave_fano":
        curves = load_microwave_fano(raw_root)
    elif task == "microstrip_resonator":
        curves = load_microstrip_resonator(raw_root)
    elif task == "aluminium_frf":
        curves = load_aluminium_frf(raw_root)
    else:
        raise ValueError(f"unknown measured task: {task}")
    curve = curves[int(data_seed) % len(curves)]
    n_train = int(config["n_train"])
    n_validation = int(config["n_validation"])
    if n_train + n_validation >= len(curve.frequency):
        raise ValueError("measured split leaves no held-out test frequencies")
    rng = np.random.default_rng(int(data_seed) + 32452843)
    order = rng.permutation(len(curve.frequency))
    train_indices = np.sort(order[:n_train])
    validation_indices = np.sort(order[n_train:n_train + n_validation])
    test_indices = np.sort(order[n_train + n_validation:])
    identifier = lambda index: f"{curve.curve_id}:frequency:{int(index)}"
    metadata = {
        "task": task,
        "data_seed": int(data_seed),
        "curve_id": curve.curve_id,
        "conditions": curve.conditions,
        "source_files": list(curve.source_files),
        "curve_metadata": curve.metadata,
        "split_strategy": "within_curve_peak_agnostic_random_frequency_mask",
        "test_target": "held_out_measured_frequencies",
        "train_point_ids": [identifier(index) for index in train_indices],
        "validation_point_ids": [identifier(index) for index in validation_indices],
        "test_point_ids": [identifier(index) for index in test_indices],
    }
    return DatasetBundle(
        x_train=curve.frequency[train_indices].astype(np.float64).reshape(-1, 1),
        y_train=curve.response[train_indices].astype(np.float32),
        x_validation=curve.frequency[validation_indices].astype(np.float64).reshape(-1, 1),
        y_validation=curve.response[validation_indices].astype(np.float32),
        x_test=curve.frequency[test_indices].astype(np.float64).reshape(-1, 1),
        y_test=curve.response[test_indices].astype(np.float32),
        metadata=metadata,
    )


def _read_conventional_frame(task: str, raw_root: Path):
    import pandas as pd

    directory = Path(raw_root) / task
    archives = list(directory.glob("*.zip"))
    if len(archives) != 1:
        raise ValueError(f"expected one source archive for {task}, found {len(archives)}")
    with ZipFile(archives[0]) as container:
        members = [name for name in container.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"expected one tabular file in {archives[0].name}")
        payload = BytesIO(container.read(members[0]))
        if task == "energy":
            frame = pd.read_excel(payload, engine="openpyxl")
            feature_names = [f"X{index}" for index in range(1, 9)]
            target_name = "Y1"
            problem_type = "regression"
        elif task == "credit":
            frame = pd.read_excel(payload, header=1, engine="xlrd")
            target_name = "default payment next month"
            feature_names = [column for column in frame.columns if column not in {"ID", target_name}]
            problem_type = "classification"
        elif task == "airfoil":
            feature_names = [
                "frequency_hz", "angle_of_attack_deg", "chord_length_m",
                "free_stream_velocity_m_per_s", "displacement_thickness_m",
            ]
            target_name = "sound_pressure_db"
            frame = pd.read_csv(payload, sep="\t", header=None, names=feature_names + [target_name])
            problem_type = "regression"
        elif task == "appliances":
            frame = pd.read_csv(payload)
            target_name = "Appliances"
            excluded = {"date", target_name, "rv1", "rv2"}
            feature_names = [column for column in frame.columns if column not in excluded]
            problem_type = "regression"
        else:
            raise ValueError(f"unknown conventional task: {task}")
    if frame[feature_names + [target_name]].isna().any().any():
        raise ValueError(f"missing values found in canonical {task} columns")
    return (
        frame[feature_names].to_numpy(dtype=np.float32),
        frame[[target_name]].to_numpy(dtype=np.float32),
        feature_names,
        target_name,
        problem_type,
        archives[0].name,
        members[0],
    )


def make_conventional_bundle(
    task: str,
    split_seed: int,
    raw_root: Path,
    include_test: bool = False,
) -> DatasetBundle:
    """Parse a canonical tabular source and seal test labels during exploration."""
    from sklearn.model_selection import train_test_split

    x, y, feature_names, target_name, problem_type, archive_name, member_name = (
        _read_conventional_frame(task, raw_root)
    )
    row_ids = np.arange(len(x), dtype=np.int64)
    stratify = y.reshape(-1) if problem_type == "classification" else None
    development_ids, test_ids = train_test_split(
        row_ids,
        test_size=0.20,
        random_state=int(split_seed),
        shuffle=True,
        stratify=stratify,
    )
    development_stratify = y[development_ids].reshape(-1) if stratify is not None else None
    train_ids, validation_ids = train_test_split(
        development_ids,
        test_size=0.17 / 0.80,
        random_state=int(split_seed) + 1,
        shuffle=True,
        stratify=development_stratify,
    )
    train_ids = np.asarray(sorted(train_ids), dtype=np.int64)
    validation_ids = np.asarray(sorted(validation_ids), dtype=np.int64)
    test_ids = np.asarray(sorted(test_ids), dtype=np.int64)
    sealed_hash = hashlib.sha256(
        json.dumps(test_ids.tolist(), separators=(",", ":")).encode()
    ).hexdigest()
    if include_test:
        x_test = x[test_ids]
        y_test = y[test_ids]
        test_access = "open_for_frozen_confirmatory_evaluation"
    else:
        x_test = np.empty((0, x.shape[1]), dtype=np.float32)
        y_test = np.empty((0, y.shape[1]), dtype=np.float32)
        test_access = "sealed"
    class_balance = None
    if problem_type == "classification":
        class_balance = {
            "train_positive_rate": float(np.mean(y[train_ids])),
            "validation_positive_rate": float(np.mean(y[validation_ids])),
        }
        if include_test:
            class_balance.update({
                "overall_positive_rate": float(np.mean(y)),
                "test_positive_rate": float(np.mean(y[test_ids])),
            })
    metadata = {
        "task": task,
        "split_seed": int(split_seed),
        "problem_type": problem_type,
        "feature_names": list(feature_names),
        "target_name": target_name,
        "source_archive": archive_name,
        "source_member": member_name,
        "source_row_count": int(len(x)),
        "split_strategy": (
            "stratified_random_63_17_20" if problem_type == "classification"
            else "random_63_17_20"
        ),
        "train_row_ids": train_ids.tolist(),
        "validation_row_ids": validation_ids.tolist(),
        "sealed_test_row_ids_sha256": sealed_hash,
        "test_access": test_access,
        "class_balance": class_balance,
    }
    if include_test:
        metadata["test_row_ids"] = test_ids.tolist()
    return DatasetBundle(
        x_train=x[train_ids],
        y_train=y[train_ids],
        x_validation=x[validation_ids],
        y_validation=y[validation_ids],
        x_test=x_test,
        y_test=y_test,
        metadata=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit",))
    parser.add_argument(
        "--raw-root", type=Path, default=Path(__file__).with_name("downstream_data") / "raw"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = audit_measured_sources(args.raw_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))
    print(json.dumps({
        task: summary["curve_count"] for task, summary in audit["tasks"].items()
    }, sort_keys=True))


if __name__ == "__main__":
    main()

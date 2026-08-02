"""Create the Microstrip low-calibration response-recovery figure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PACKAGE_ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENT_ROOT))

from ai4science_microstrip_b import (  # noqa: E402
    extract_microstrip_events,
    make_b2_calibration_split,
    microstrip_response_metrics,
)
from confirmatory_models import build_model  # noqa: E402
from confirmatory_protocol import seeded_build  # noqa: E402
from downstream_protocol import load_microstrip_resonator  # noqa: E402
from downstream_training import fit_task_model, prepare_bundle  # noqa: E402


RAW_ROOT = PACKAGE_ROOT / "data" / "raw"
SELECTION_ROOT = (
    PACKAGE_ROOT
    / "results"
    / "selections"
    / "microstrip_resonator_group_transfer"
)
OUT_DIR = PACKAGE_ROOT / "results" / "figures"
OUT_STEM = OUT_DIR / "microstrip_low_calibration_recovery"
CACHE_PATH = OUT_DIR / "microstrip_low_calibration_recovery_cache.npz"
RAW_RECORDS_PATH = (
    PACKAGE_ROOT
    / "results"
    / "ai4science"
    / "microstrip_b2_missed_band_negative_control_seed_expansion"
    / "raw_records.jsonl"
)
Q_ERROR_CACHE_PATH = (
    PACKAGE_ROOT
    / "results"
    / "ai4science"
    / "microstrip"
    / "q_linewidth_median_relative_error_by_calibration.json"
)

SAMPLE_ID = "s34"
TARGET_CURVE_ID = "s34:cycle:023"
TARGET_TEMPERATURE_C = 25.0
TARGET_RH_PERCENT = 50.0
CALIBRATION_COUNT = 16
SUPPORT_COUNT = 16
VALIDATION_COUNT = 24
SPLIT_SEED = 233
INIT_SEED = 1013
MODEL_BUDGET = 256

FAMILIES = [
    ("CFNN", "CFNN", "#009988", "-"),
    ("MLP", "MLP", "#EE7733", (0, (3, 1.7))),
    ("Local nested CF control", "Local CF control", "#0077BB", (0, (1.2, 1.4))),
]


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _as_complex(response: np.ndarray) -> np.ndarray:
    response = np.asarray(response, dtype=np.float64)
    return response[:, 0] + 1j * response[:, 1]


def _load_selection(family: str) -> dict:
    path = SELECTION_ROOT / f"{_slug(family)}__{MODEL_BUDGET}.json"
    selection = json.loads(path.read_text())
    if selection.get("status") != "selected":
        raise ValueError(f"selection is not usable: {path}")
    return selection


def _train_prediction(family: str, bundle, device: torch.device) -> np.ndarray:
    selection = _load_selection(family)
    candidate = selection["candidate"]
    optimizer = selection["optimizer"]
    model, _ = seeded_build(
        INIT_SEED,
        lambda: build_model(
            family,
            input_dim=3,
            output_dim=2,
            config=candidate["config"],
            seed=INIT_SEED,
        ),
    )
    model = model.to(device)
    prepared = prepare_bundle(bundle, "complex_regression", device)
    trained = fit_task_model(
        model,
        prepared,
        "complex_regression",
        learning_rate=float(optimizer["lr"]),
        weight_decay=float(optimizer["weight_decay"]),
        max_steps=140,
        min_steps=50,
        patience=50,
        checkpoints=[20, 80, 140],
        gradient_clip=5.0,
        batch_size=None,
        batch_seed=SPLIT_SEED + 49979687,
        validation_interval=10,
    )
    if trained.status != "ok":
        raise RuntimeError(f"{family} training failed: {trained.status}")
    return trained.prediction


def _target_indices(bundle) -> np.ndarray:
    indices = []
    for row_index, point_id in enumerate(bundle.metadata["test_point_ids"]):
        if point_id.split(":frequency:")[0] == TARGET_CURVE_ID:
            indices.append(row_index)
    if not indices:
        raise ValueError(f"target curve not present in test bundle: {TARGET_CURVE_ID}")
    return np.asarray(indices, dtype=np.int64)


def _calibration_indices(bundle) -> np.ndarray:
    indices = []
    prefix = f"{TARGET_CURVE_ID}:frequency:"
    for point_id in bundle.metadata["calibration_point_ids"]:
        if point_id.startswith(prefix):
            indices.append(int(point_id.rsplit(":", 1)[-1]))
    if not indices:
        raise ValueError(f"target curve not present in calibration IDs: {TARGET_CURVE_ID}")
    return np.asarray(sorted(indices), dtype=np.int64)


def _build_cache() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curves = load_microstrip_resonator(RAW_ROOT)
    curve_map = {curve.curve_id: curve for curve in curves}
    target_curve = curve_map[TARGET_CURVE_ID]
    bundle = make_b2_calibration_split(
        curves,
        sample_id=SAMPLE_ID,
        target_temperature_c=TARGET_TEMPERATURE_C,
        target_relative_humidity_percent=TARGET_RH_PERCENT,
        calibration_frequency_count=CALIBRATION_COUNT,
        support_frequency_count=SUPPORT_COUNT,
        validation_frequency_count=VALIDATION_COUNT,
        seed=SPLIT_SEED,
        calibration_sampling_mode="missed_band",
        calibration_excluded_band_linewidths=1.0,
    )
    target_test_indices = _target_indices(bundle)
    order = np.argsort(bundle.x_test[target_test_indices, 0], kind="mergesort")
    target_test_indices = target_test_indices[order]
    calibration_indices = _calibration_indices(bundle)
    events = extract_microstrip_events(target_curve.frequency, target_curve.response)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arrays: dict[str, np.ndarray | float | int | str] = {
        "frequency": np.asarray(bundle.x_test[target_test_indices, 0], dtype=np.float64),
        "truth": np.asarray(bundle.y_test[target_test_indices], dtype=np.float64),
        "calibration_frequency": target_curve.frequency[calibration_indices].astype(np.float64),
        "calibration_response": target_curve.response[calibration_indices].astype(np.float64),
        "f0_hz": float(events["resonance_frequency_hz"]),
        "linewidth_hz": float(events["linewidth_hz"]),
        "target_curve_id": TARGET_CURVE_ID,
        "calibration_count": CALIBRATION_COUNT,
        "support_count": SUPPORT_COUNT,
        "split_seed": SPLIT_SEED,
        "init_seed": INIT_SEED,
        "device": str(device),
        "truth_event_f0_hz": float(events["resonance_frequency_hz"]),
        "truth_event_linewidth_hz": float(events["linewidth_hz"]),
    }
    metric_rows = []
    for family, _, _, _ in FAMILIES:
        print(f"training {family}", flush=True)
        prediction = _train_prediction(family, bundle, device)
        target_prediction = prediction[target_test_indices]
        arrays[f"prediction_{_slug(family)}"] = np.asarray(target_prediction, dtype=np.float64)
        metrics = microstrip_response_metrics(arrays["frequency"], arrays["truth"], target_prediction)
        predicted_events = metrics["predicted_events"]
        arrays[f"event_f0_hz_{_slug(family)}"] = float(predicted_events["resonance_frequency_hz"])
        arrays[f"event_linewidth_hz_{_slug(family)}"] = float(predicted_events["linewidth_hz"])
        metric_rows.append(
            {
                "family": family,
                **{
                    key: metrics.get(key)
                    for key in (
                        "complex_nrmse",
                        "peak_window_complex_nrmse",
                        "resonance_frequency_abs_error_hz",
                        "quality_factor_relative_error",
                        "local_phase_transition_abs_error_hz",
                        "phase_mae_rad",
                    )
                },
            }
        )
    np.savez_compressed(CACHE_PATH, **arrays)
    (OUT_DIR / "microstrip_low_calibration_recovery_metrics.json").write_text(
        json.dumps(metric_rows, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _load_cache() -> np.lib.npyio.NpzFile:
    if not CACHE_PATH.exists():
        _build_cache()
    cache = np.load(CACHE_PATH, allow_pickle=False)
    expected = {
        "target_curve_id": TARGET_CURVE_ID,
        "calibration_count": CALIBRATION_COUNT,
        "support_count": SUPPORT_COUNT,
        "split_seed": SPLIT_SEED,
        "init_seed": INIT_SEED,
    }
    for key, value in expected.items():
        cached = cache[key]
        cached_value = cached.item() if getattr(cached, "shape", ()) == () else cached
        if cached_value != value:
            cache.close()
            _build_cache()
            return np.load(CACHE_PATH, allow_pickle=False)
    required = ["truth_event_f0_hz", "truth_event_linewidth_hz"]
    required.extend(f"event_f0_hz_{_slug(family)}" for family, _, _, _ in FAMILIES)
    required.extend(f"event_linewidth_hz_{_slug(family)}" for family, _, _, _ in FAMILIES)
    if any(key not in cache.files for key in required):
        cache.close()
        _build_cache()
        return np.load(CACHE_PATH, allow_pickle=False)
    return cache


def _magnitude_db(response: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(_as_complex(response)), 1e-12))


def _draw_context(ax: plt.Axes, cache: np.lib.npyio.NpzFile, x_ghz: np.ndarray, truth_db: np.ndarray) -> None:
    calibration_x = cache["calibration_frequency"] / 1e9
    calibration_db = _magnitude_db(cache["calibration_response"])
    ax.plot(x_ghz, truth_db, color="#68727A", lw=1.35, label="Measured truth", zorder=4)
    ax.scatter(
        calibration_x,
        calibration_db,
        s=16,
        color="#222D35",
        edgecolor="white",
        linewidth=0.35,
        label=f"{CALIBRATION_COUNT} calibration points",
        zorder=6,
    )
    ax.set_title("a  Low-calibration design", loc="left", pad=5)
    ax.set_ylabel(r"Magnitude $|S_{12}|$ (dB)")
    ax.legend(loc="lower left", frameon=False, handlelength=1.7, borderpad=0.2)


def _draw_events(ax: plt.Axes, cache: np.lib.npyio.NpzFile) -> None:
    rows = [
        ("Measured truth", float(cache["truth_event_f0_hz"]), float(cache["truth_event_linewidth_hz"]), "#9AA3AA", "-"),
    ]
    for family, label, color, linestyle in FAMILIES:
        rows.append(
            (
                label,
                float(cache[f"event_f0_hz_{_slug(family)}"]),
                float(cache[f"event_linewidth_hz_{_slug(family)}"]),
                color,
                linestyle,
            )
        )
    truth_f0_ghz = rows[0][1] / 1e9
    truth_half_width_ghz = 0.5 * rows[0][2] / 1e9
    ax.axvspan(
        truth_f0_ghz - truth_half_width_ghz,
        truth_f0_ghz + truth_half_width_ghz,
        color="#EEF1F3",
        zorder=0,
    )
    ax.axvline(truth_f0_ghz, color="#5C6670", lw=0.75, zorder=1)

    y_positions = np.arange(len(rows))[::-1]
    for y, (label, f0_hz, linewidth_hz, color, _linestyle) in zip(y_positions, rows):
        f0_ghz = f0_hz / 1e9
        half_width = 0.5 * linewidth_hz / 1e9
        ax.hlines(
            y,
            f0_ghz - half_width,
            f0_ghz + half_width,
            color=color,
            lw=3.1 if label != "Measured truth" else 2.7,
            linestyle="-",
            zorder=3,
        )
        ax.plot(
            [f0_ghz],
            [y],
            marker="o",
            ms=4.2,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.35,
            zorder=4,
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row[0] for row in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_title("b  Event recovery", loc="left", pad=5)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_xlim(1.72, 2.28)
    ax.set_ylim(-0.65, len(rows) - 0.35)


def _load_q_error_by_family() -> dict[int, dict[str, float]]:
    if Q_ERROR_CACHE_PATH.exists():
        cached = json.loads(Q_ERROR_CACHE_PATH.read_text())["grouped"]
        return {
            int(calibration_count): {
                family: np.nan if value is None else float(value)
                for family, value in family_values.items()
            }
            for calibration_count, family_values in cached.items()
        }

    values: dict[tuple[int, str], list[float]] = {}
    allowed_families = {family for family, _, _, _ in FAMILIES}
    with RAW_RECORDS_PATH.open() as handle:
        for line in handle:
            record = json.loads(line)
            family = record.get("family")
            if (
                record.get("status") != "ok"
                or record.get("calibration_sampling_mode") != "missed_band"
                or family not in allowed_families
            ):
                continue
            calibration_count = int(record["calibration_frequency_count"])
            for metric in record.get("curve_metrics", []):
                q_error = metric.get("quality_factor_relative_error")
                if (
                    metric.get("status") != "ok"
                    or not metric.get("truth_q_available")
                    or not metric.get("truth_event_identifiable")
                    or q_error is None
                ):
                    continue
                values.setdefault((calibration_count, family), []).append(float(q_error))

    calibration_counts = sorted({key[0] for key in values})
    grouped: dict[int, dict[str, float]] = {}
    for calibration_count in calibration_counts:
        grouped[calibration_count] = {}
        for family, _, _, _ in FAMILIES:
            family_values = values.get((calibration_count, family), [])
            if not family_values:
                grouped[calibration_count][family] = np.nan
                continue
            grouped[calibration_count][family] = float(np.median(family_values))
    return grouped


def _draw_transfer_effects(ax: plt.Axes) -> None:
    grouped = _load_q_error_by_family()
    calibration_counts = sorted(grouped)
    x = np.arange(len(calibration_counts), dtype=np.float64)
    bar_width = 0.22
    offsets = np.linspace(-bar_width, bar_width, len(FAMILIES))
    for offset, (family, label, color, _linestyle) in zip(offsets, FAMILIES):
        y = [grouped[calibration_count][family] for calibration_count in calibration_counts]
        ax.bar(
            x + offset,
            y,
            width=bar_width,
            color=color,
            label=label,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    ax.set_title("c  Q/linewidth comparison", loc="left", pad=5)
    ax.set_xlabel("Target calibration points")
    ax.set_ylabel("Median relative error")
    ax.set_xticks(x)
    ax.set_xticklabels([str(count) for count in calibration_counts])
    ax.set_ylim(0.0, 0.56)
    ax.legend(loc="upper center", frameon=False, ncol=3, handlelength=1.2, columnspacing=0.75, borderpad=0.2)


def draw(cache: np.lib.npyio.NpzFile) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 6.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    frequency = cache["frequency"]
    truth = cache["truth"]
    x_ghz = frequency / 1e9
    truth_db = _magnitude_db(truth)
    f0_ghz = float(cache["f0_hz"]) / 1e9
    linewidth_ghz = float(cache["linewidth_hz"]) / 1e9

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.74), constrained_layout=False)
    fig.subplots_adjust(left=0.078, right=0.995, bottom=0.18, top=0.88, wspace=0.38)

    _draw_context(axes[0], cache, x_ghz, truth_db)
    _draw_events(axes[1], cache)
    _draw_transfer_effects(axes[2])

    band_half_width = linewidth_ghz
    for ax in (axes[0],):
        ax.axvspan(f0_ghz - band_half_width, f0_ghz + band_half_width, color="#EEF1F3", zorder=0)
        ax.axvline(f0_ghz, color="#5C6670", lw=0.75, zorder=1)
        ax.set_xlabel("Frequency (GHz)")
        ax.grid(axis="both", color="#DDE2E6", lw=0.45)
    for ax in axes[1:]:
        ax.grid(axis="both", color="#DDE2E6", lw=0.45)
    axes[0].set_xlim(float(np.min(x_ghz)), float(np.max(x_ghz)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cache = _load_cache()
    draw(cache)
    print(
        {
            "pdf": str(OUT_STEM.with_suffix(".pdf")),
            "png": str(OUT_STEM.with_suffix(".png")),
            "cache": str(CACHE_PATH),
            "metrics": str(OUT_DIR / "microstrip_low_calibration_recovery_metrics.json"),
        }
    )


if __name__ == "__main__":
    main()

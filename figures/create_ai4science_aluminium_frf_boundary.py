"""Create the Aluminium FRF AI4Science boundary figure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "experiment_refine"
sys.path.insert(0, str(EXPERIMENT_ROOT))

from ai4science_aluminium_c import load_aluminium_frf_full  # noqa: E402


RAW_ROOT = EXPERIMENT_ROOT / "downstream_data" / "raw"
RESULT_ROOT = EXPERIMENT_ROOT / "ai4science_results"
OUT_DIR = PACKAGE_ROOT / "results" / "figures"
OUT_STEM = OUT_DIR / "aluminium_frf_boundary"

CATALOGUE_PATH = (
    RESULT_ROOT
    / "aluminium_c_truth_modal_catalogue_and_diagnostics"
    / "truth_modal_catalogue.json"
)
SAME_SCHEDULE_SUMMARY_PATH = (
    RESULT_ROOT
    / "aluminium_c_online_half_power_same_schedule_model_pilot"
    / "analysis_summary.json"
)
INTERPOLATION_RECORDS_PATH = (
    RESULT_ROOT
    / "aluminium_c_online_half_power_multipoint_interpolation_guardrail"
    / "raw_records.jsonl"
)

TARGET_CURVE_ID = "point:01"
TARGET_POINT = 1
TARGET_BUDGET = 64
MAX_DISPLAY_FREQUENCY_HZ = 340.0

MODEL_FAMILIES = [
    ("Signed-log linear interpolation", "Signed-log interp.", "#CC6677"),
    ("Complex linear interpolation", "Complex interp.", "#6F777D"),
    ("Complex modal fit interpolation", "Complex modal fit", "#AA4499"),
    ("Global complex modal prior", "Global modal prior", "#997700"),
    ("Low-order global modal prior", "Low-order prior", "#5F6C7B"),
    ("CFNN", "CFNN", "#009988"),
    ("MLP", "MLP", "#EE7733"),
    ("Local nested CF control", "Local CF", "#0077BB"),
]


def _as_complex(response: np.ndarray) -> np.ndarray:
    response = np.asarray(response, dtype=np.float64)
    return response[:, 0] + 1j * response[:, 1]


def _magnitude_db(response: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(_as_complex(response)), 1e-30))


def _load_target_curve() -> tuple[np.ndarray, np.ndarray]:
    curves = {curve.curve_id: curve for curve in load_aluminium_frf_full(RAW_ROOT)}
    curve = curves[TARGET_CURVE_ID]
    return np.asarray(curve.frequency, dtype=np.float64), np.asarray(curve.response, dtype=np.float64)


def _load_target_modes() -> list[dict]:
    catalogue = json.loads(CATALOGUE_PATH.read_text())
    rows = [
        row for row in catalogue
        if row["curve_id"] == TARGET_CURVE_ID and int(row["mode_rank"]) <= 5
    ]
    return sorted(rows, key=lambda row: int(row["mode_rank"]))


def _load_observed_indices() -> np.ndarray:
    with INTERPOLATION_RECORDS_PATH.open() as handle:
        for line in handle:
            record = json.loads(line)
            if (
                record.get("status") == "ok"
                and record.get("curve_id") == TARGET_CURVE_ID
                and record.get("point") == TARGET_POINT
                and record.get("budget") == TARGET_BUDGET
                and record.get("family") == "Complex linear interpolation"
                and record.get("sampling_strategy") == "catalogue_band_half_power_online"
            ):
                return np.asarray(record["observed_indices"], dtype=np.int64)
    raise ValueError("target observed-index schedule not found")


def _load_half_power_rows(budget: int) -> list[dict]:
    with INTERPOLATION_RECORDS_PATH.open() as handle:
        for line in handle:
            record = json.loads(line)
            if (
                record.get("status") == "ok"
                and record.get("curve_id") == TARGET_CURVE_ID
                and record.get("point") == TARGET_POINT
                and int(record.get("budget")) == int(budget)
                and record.get("family") == "Complex linear interpolation"
                and record.get("sampling_strategy") == "catalogue_band_half_power_online"
            ):
                return sorted(
                    record["metrics"]["half_power"]["mode_rows"],
                    key=lambda row: int(row["mode_rank"]),
                )
    raise ValueError(f"half-power rows not found for budget {budget}")


def _load_same_schedule_rows() -> list[dict]:
    summary = json.loads(SAME_SCHEDULE_SUMMARY_PATH.read_text())
    allowed = {family for family, _, _ in MODEL_FAMILIES}
    rows = [
        row for row in summary["summary_rows"]
        if row.get("family") in allowed and int(row.get("budget")) in {48, 64}
    ]
    return sorted(rows, key=lambda row: (int(row["budget"]), row["family"]))


def _draw_frf_target(ax: plt.Axes) -> None:
    frequency, response = _load_target_curve()
    magnitude = _magnitude_db(response)
    observed_indices = _load_observed_indices()
    modes = _load_target_modes()
    mask = frequency <= MAX_DISPLAY_FREQUENCY_HZ

    ax.plot(
        frequency[mask],
        magnitude[mask],
        color="#68727A",
        lw=1.25,
        label="Measured FRF",
        zorder=3,
    )
    visible_observed = observed_indices[frequency[observed_indices] <= MAX_DISPLAY_FREQUENCY_HZ]
    ax.scatter(
        frequency[visible_observed],
        magnitude[visible_observed],
        s=14,
        color="#222D35",
        edgecolor="white",
        linewidth=0.35,
        label=f"{TARGET_BUDGET} observed bins",
        zorder=5,
    )
    mode_frequency = np.asarray([row["frequency_hz"] for row in modes], dtype=np.float64)
    mode_magnitude = np.interp(mode_frequency, frequency, magnitude)
    ax.scatter(
        mode_frequency,
        mode_magnitude,
        s=22,
        marker="D",
        color="#009988",
        edgecolor="white",
        linewidth=0.35,
        label="Target modes",
        zorder=6,
    )
    for mode in modes:
        ax.axvline(float(mode["frequency_hz"]), color="#009988", lw=0.55, alpha=0.35, zorder=1)

    ax.set_title("a  Modal FRF target", loc="left", pad=5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(0, MAX_DISPLAY_FREQUENCY_HZ)
    ax.legend(loc="lower right", frameon=False, handlelength=1.6, borderpad=0.2)


def _draw_recovery_ladder(ax: plt.Axes) -> None:
    rows = [row for row in _load_same_schedule_rows() if int(row["budget"]) == 64]
    row_by_family = {row["family"]: row for row in rows}
    y = np.arange(len(MODEL_FAMILIES), dtype=np.float64)[::-1]
    stage_specs = [
        ("median_half_power_predicted_ok_mode_count", "Predicted ok", "o", "#8A9399", 0.16),
        ("median_matched_mode_count", "Matched", "s", "#0077BB", 0.00),
        ("median_half_power_passed_mode_count", "Passed", "D", "#009988", -0.16),
    ]
    for y_pos, (family, _label, _color) in zip(y, MODEL_FAMILIES):
        ax.hlines(y_pos, 0, 5, color="#DDE2E6", lw=1.0, zorder=1)
        row = row_by_family[family]
        for key, stage_label, marker, color, offset in stage_specs:
            value = float(row[key])
            ax.scatter(
                [value],
                [y_pos + offset],
                s=26,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                label=stage_label if y_pos == y[0] else None,
                zorder=4,
            )
    ax.axvline(5.0, color="#A7B0B8", lw=0.75, linestyle=(0, (2.0, 1.5)), zorder=1)
    ax.set_title("b  Modal recovery stages (64 bins)", loc="left", pad=5)
    ax.set_xlabel("Recovered modes")
    ax.set_xlim(-0.25, 5.35)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label, _ in MODEL_FAMILIES])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.65, len(MODEL_FAMILIES) - 0.35)
    ax.legend(loc="lower right", frameon=False, ncol=1, handlelength=1.0, borderpad=0.2)


def _draw_capability_boundary(ax: plt.Axes) -> None:
    rows = [row for row in _load_same_schedule_rows() if int(row["budget"]) == 64]
    row_by_family = {row["family"]: row for row in rows}
    label_offsets = {
        "Signed-log linear interpolation": (0.018, -0.18),
        "Complex linear interpolation": (0.018, 0.12),
        "Complex modal fit interpolation": (0.018, 0.10),
        "Global complex modal prior": (0.018, 0.02),
        "Low-order global modal prior": (0.018, -0.20),
    }
    display_y_offsets = {
        "CFNN": 0.16,
        "MLP": -0.16,
        "Local nested CF control": 0.0,
    }
    for family, label, color in MODEL_FAMILIES:
        row = row_by_family[family]
        x_value = float(row["median_peak_window_complex_nrmse"])
        y_value = float(row["median_matched_mode_count"])
        passed_value = float(row["median_half_power_passed_mode_count"])
        y_display = y_value + display_y_offsets.get(family, 0.0)
        ax.scatter(
            [x_value],
            [y_display],
            s=32 + 16 * passed_value,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
        if family in label_offsets:
            dx, dy = label_offsets[family]
            ax.text(
                x_value + dx,
                y_display + dy,
                label,
                ha="left",
                va="center",
                fontsize=5.6,
                color="#222D35",
            )
    ax.text(
        0.89,
        0.34,
        "CFNN / MLP /\nLocal CF",
        ha="left",
        va="center",
        fontsize=5.6,
        color="#222D35",
    )
    ax.axhline(5.0, color="#A7B0B8", lw=0.7, linestyle=(0, (2.0, 1.5)), zorder=1)
    ax.axvspan(0.2, 0.42, color="#EEF7F4", zorder=0)
    ax.axvspan(0.9, 1.08, color="#F7F1EE", zorder=0)
    ax.set_title("c  Recovery capability boundary", loc="left", pad=5)
    ax.set_xlabel("Peak-window complex NRMSE")
    ax.set_ylabel("Matched modes (of 5)")
    ax.set_xlim(0.2, 1.13)
    ax.set_ylim(-0.35, 5.55)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.text(0.22, 0.18, "marker size: passed modes", fontsize=5.5, color="#5C6670")


def draw() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 5.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.72), constrained_layout=False)
    fig.subplots_adjust(left=0.078, right=0.995, bottom=0.18, top=0.88, wspace=0.38)

    _draw_frf_target(axes[0])
    _draw_recovery_ladder(axes[1])
    _draw_capability_boundary(axes[2])

    for ax in axes:
        ax.grid(axis="both", color="#DDE2E6", lw=0.45, zorder=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    draw()
    print({
        "pdf": str(OUT_STEM.with_suffix(".pdf")),
        "png": str(OUT_STEM.with_suffix(".png")),
    })


if __name__ == "__main__":
    main()

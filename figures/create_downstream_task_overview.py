"""Create the downstream scientific task overview figure for the CFNN paper."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "experiment_refine"
RAW_ROOT = EXPERIMENT_ROOT / "downstream_data" / "raw"
OUT_DIR = PACKAGE_ROOT / "results" / "figures"
OUT_STEM = OUT_DIR / "downstream_task_overview"

sys.path.insert(0, str(EXPERIMENT_ROOT))
from downstream_protocol import load_microwave_fano, load_microstrip_resonator  # noqa: E402
from peak_sensitive_protocol import load_dense_aluminium_frf  # noqa: E402


@dataclass(frozen=True)
class PanelData:
    key: str
    label: str
    title: str
    curve_id: str
    frequency: np.ndarray
    magnitude: np.ndarray
    train_frequency: np.ndarray
    train_magnitude: np.ndarray
    x_unit: str
    y_scale: str = "linear"


def _load_role_manifest(name: str) -> dict:
    path = EXPERIMENT_ROOT / "peak_sensitive_results" / "provenance" / name
    return json.loads(path.read_text())


def _curve_by_id(curves: list, curve_id: str):
    for curve in curves:
        if curve.curve_id == curve_id:
            return curve
    raise ValueError(f"missing frozen curve {curve_id}")


def _within_curve_train_indices(point_count: int, data_seed: int, n_train: int, n_validation: int) -> np.ndarray:
    if n_train + n_validation >= point_count:
        raise ValueError("split leaves no held-out frequency points")
    order = np.random.default_rng(int(data_seed) + 32452843).permutation(point_count)
    return np.sort(order[:n_train])


def _frf_train_indices(curve_id: str, data_seed: int, n_train: int, point_count: int) -> np.ndarray:
    # Mirrors peak_sensitive_protocol._frf_parent_seed without importing private helpers.
    import hashlib

    encoded = f"frf-density-parent-mask-v1:{curve_id}:{int(data_seed)}:train".encode("ascii")
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    parent = np.random.default_rng(seed).permutation(point_count)[:192]
    return np.sort(parent[:n_train])


def _magnitude(response: np.ndarray) -> np.ndarray:
    response = np.asarray(response, dtype=np.float64)
    return np.hypot(response[:, 0], response[:, 1])


def _normalize_frequency(frequency: np.ndarray) -> np.ndarray:
    frequency = np.asarray(frequency, dtype=np.float64)
    span = float(frequency[-1] - frequency[0])
    if span <= 0.0:
        raise ValueError("frequency axis must be strictly increasing")
    return (frequency - float(frequency[0])) / span


def _panel_from_curve(task: str, curve, *, data_seed: int, n_train: int, n_validation: int,
                      label: str, title: str, unit: str, y_scale: str = "linear") -> PanelData:
    train_indices = _within_curve_train_indices(len(curve.frequency), data_seed, n_train, n_validation)
    magnitude = _magnitude(curve.response)
    return PanelData(
        key=task,
        label=label,
        title=title,
        curve_id=curve.curve_id,
        frequency=_normalize_frequency(curve.frequency),
        magnitude=magnitude,
        train_frequency=_normalize_frequency(curve.frequency)[train_indices],
        train_magnitude=magnitude[train_indices],
        x_unit=unit,
        y_scale=y_scale,
    )


def load_panels() -> list[PanelData]:
    data_seed = 719

    fano_manifest = _load_role_manifest("microwave_fano_role_manifest.json")
    fano_curve_id = fano_manifest["seed_curve_id_map"]["evaluation"][str(data_seed)]
    fano_curve = _curve_by_id(load_microwave_fano(RAW_ROOT), fano_curve_id)

    microstrip_manifest = _load_role_manifest("microstrip_resonator_role_manifest.json")
    microstrip_curve_id = microstrip_manifest["seed_curve_id_map"]["evaluation"][str(data_seed)]
    microstrip_curve = _curve_by_id(load_microstrip_resonator(RAW_ROOT), microstrip_curve_id)

    frf_manifest = _load_role_manifest("frf_density_manifest.json")
    frf_curve_id = frf_manifest["evaluation_curve_ids"][0]
    frf_curve = _curve_by_id(load_dense_aluminium_frf(RAW_ROOT), frf_curve_id)
    frf_train = _frf_train_indices(frf_curve.curve_id, data_seed, 24, len(frf_curve.frequency))
    frf_mag = _magnitude(frf_curve.response)

    return [
        _panel_from_curve(
            "microwave_fano",
            fano_curve,
            data_seed=data_seed,
            n_train=128,
            n_validation=128,
            label="a",
            title="Microwave Fano curve",
            unit="Released power sweep",
        ),
        _panel_from_curve(
            "microstrip_resonator",
            microstrip_curve,
            data_seed=data_seed,
            n_train=128,
            n_validation=128,
            label="b",
            title="Microstrip transmission curve",
            unit="Humidity resonator window",
        ),
        PanelData(
            key="aluminium_frf",
            label="c",
            title="Aluminium plate response",
            curve_id=frf_curve.curve_id,
            frequency=_normalize_frequency(frf_curve.frequency),
            magnitude=frf_mag,
            train_frequency=_normalize_frequency(frf_curve.frequency)[frf_train],
            train_magnitude=frf_mag[frf_train],
            x_unit="Dense H1 frequency bins",
            y_scale="log",
        ),
    ]


def _shade_feature_window(ax: plt.Axes, panel: PanelData) -> None:
    plotted = np.log10(np.maximum(panel.magnitude, 1e-18)) if panel.y_scale == "log" else panel.magnitude
    gradient = np.abs(np.gradient(plotted, panel.frequency))
    center = int(np.nanargmax(gradient))
    half_width = max(0.025, 8.0 / len(panel.frequency))
    left = max(0.0, float(panel.frequency[center] - half_width))
    right = min(1.0, float(panel.frequency[center] + half_width))
    ax.axvspan(left, right, color="#D9EAF7", alpha=0.45, lw=0)


def draw(panels: list[PanelData]) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.75), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.78, wspace=0.32)
    line_color = "#22577A"
    sample_face = "#FFFFFF"
    sample_edge = "#C44536"
    for ax, panel in zip(axes, panels):
        _shade_feature_window(ax, panel)
        ax.plot(panel.frequency, panel.magnitude, color=line_color, lw=1.35, solid_capstyle="round")
        ax.scatter(
            panel.train_frequency,
            panel.train_magnitude,
            s=13,
            facecolor=sample_face,
            edgecolor=sample_edge,
            linewidth=0.75,
            zorder=3,
        )
        ax.set_title(f"{panel.label}  {panel.title}", loc="left", pad=5)
        ax.set_xlabel("Normalized frequency")
        ax.set_xlim(-0.015, 1.015)
        ax.grid(axis="y", color="#D9DDE3", lw=0.45, alpha=0.9)
        ax.grid(axis="x", color="#EEF1F4", lw=0.35, alpha=0.6)
        if panel.y_scale == "log":
            ax.set_yscale("log")
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[0].set_ylabel("Complex-response magnitude")

    line_proxy = plt.Line2D([0], [0], color=line_color, lw=1.35)
    point_proxy = plt.Line2D(
        [0], [0], marker="o", linestyle="None", markerfacecolor=sample_face,
        markeredgecolor=sample_edge, markeredgewidth=0.75, markersize=4.5,
    )
    shade_proxy = plt.Rectangle((0, 0), 1, 1, facecolor="#D9EAF7", edgecolor="none", alpha=0.6)
    fig.legend(
        [line_proxy, point_proxy, shade_proxy],
        ["Full measured response", "Frozen training samples", "Sharp local region"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        ncol=3,
        columnspacing=1.7,
        handlelength=1.9,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    panels = load_panels()
    draw(panels)
    print(json.dumps({
        "pdf": str(OUT_STEM.with_suffix(".pdf")),
        "png": str(OUT_STEM.with_suffix(".png")),
        "panels": [
            {
                "key": panel.key,
                "curve_id": panel.curve_id,
                "points": int(len(panel.frequency)),
                "training_points": int(len(panel.train_frequency)),
            }
            for panel in panels
        ],
    }, indent=2))


if __name__ == "__main__":
    main()

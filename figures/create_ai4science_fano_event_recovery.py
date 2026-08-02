"""Create the AI4Science Microwave Fano event-recovery figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = (
    REPO_ROOT
    / "experiment_refine"
    / "ai4science_results"
    / "microwave_fano_event_focused_consolidation"
    / "summary.json"
)
OUT_DIR = REPO_ROOT / "CFNN_paper_nmi" / "img" / "ai4science"
OUT_STEM = OUT_DIR / "fano_event_recovery"

COMPARISON_ORDER = ["MLP", "Local nested CF control"]
COMPARISON_LABELS = {
    "MLP": "vs MLP",
    "Local nested CF control": "vs local CF control",
}
COMPARISON_COLORS = {
    "MLP": "#E76F51",
    "Local nested CF control": "#2A9D8F",
}

RESPONSE_METRICS = [
    ("global_complex_nrmse", "Global response"),
    ("peak_window_complex_nrmse", "Peak-window response"),
]

EVENT_METRICS = [
    ("f0_abs_error_hz", r"Resonance center $f_0$"),
    ("quality_factor_relative_error", r"Quality factor $Q$"),
    ("q_abs_error", r"Fano asymmetry $q$"),
]


def _load_rows() -> pd.DataFrame:
    with SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = pd.DataFrame(payload["event_focus_rows"])
    rows["_comparison_order"] = rows["comparison_family"].map(
        {name: index for index, name in enumerate(COMPARISON_ORDER)}
    )
    return rows.sort_values(["metric", "_comparison_order"])


def _metric_matrix(rows: pd.DataFrame, metrics: list[tuple[str, str]]) -> np.ndarray:
    matrix = np.zeros((len(metrics), len(COMPARISON_ORDER)), dtype=float)
    for metric_index, (metric, _) in enumerate(metrics):
        metric_rows = rows[rows["metric"].eq(metric)]
        for comparison_index, comparison in enumerate(COMPARISON_ORDER):
            value = metric_rows[metric_rows["comparison_family"].eq(comparison)]["effect"].iloc[0]
            matrix[metric_index, comparison_index] = float(value)
    return matrix


def _draw_panel(
    ax: plt.Axes,
    rows: pd.DataFrame,
    metrics: list[tuple[str, str]],
    *,
    title: str,
    xlim: tuple[float, float],
    xticks: list[float],
) -> None:
    values = _metric_matrix(rows, metrics)
    y = np.arange(len(metrics))
    bar_height = 0.30
    offsets = [-bar_height / 1.8, bar_height / 1.8]

    for index, comparison in enumerate(COMPARISON_ORDER):
        ax.barh(
            y + offsets[index],
            values[:, index],
            height=bar_height,
            color=COMPARISON_COLORS[comparison],
            edgecolor="#FFFFFF",
            linewidth=0.45,
            label=COMPARISON_LABELS[comparison],
            zorder=3,
        )

    ax.axvline(0.0, color="#5C6670", lw=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in metrics])
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xlabel("Signed effect on error metrics")
    ax.set_title(title, loc="left", y=1.16, pad=0)
    ax.grid(axis="x", color="#DDE2E6", lw=0.45)
    ax.tick_params(axis="y", length=0)


def draw(rows: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.55), constrained_layout=False)
    fig.subplots_adjust(left=0.19, right=0.995, bottom=0.22, top=0.76, wspace=0.44)

    _draw_panel(
        axes[0],
        rows,
        RESPONSE_METRICS,
        title="a  Complex-response recovery",
        xlim=(-0.004, 0.058),
        xticks=[0.00, 0.02, 0.04],
    )
    _draw_panel(
        axes[1],
        rows,
        EVENT_METRICS,
        title="b  Event-parameter recovery",
        xlim=(-0.30, 0.54),
        xticks=[-0.2, 0.0, 0.2, 0.4],
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.995),
        frameon=False,
        ncol=2,
        handlelength=1.8,
        columnspacing=1.4,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = _load_rows()
    draw(rows)
    print(
        {
            "pdf": str(OUT_STEM.with_suffix(".pdf")),
            "png": str(OUT_STEM.with_suffix(".png")),
        }
    )


if __name__ == "__main__":
    main()

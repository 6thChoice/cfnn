"""Create the downstream performance summary figure for the CFNN paper."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "experiment_refine" / "peak_sensitive_results" / "artifacts"
OUT_DIR = PACKAGE_ROOT / "results" / "figures"
OUT_STEM = OUT_DIR / "downstream_performance_summary"

FAMILIES = [
    ("CFNN", "CFNN"),
    ("MLP", "MLP"),
    ("KAN", "KAN"),
    ("SIREN", "SIREN"),
    ("Fourier-feature MLP", "RFF-MLP"),
    ("Gaussian RBF", "RBF"),
    ("Local nested CF control", "Nested CF"),
    ("Rational activation NN", "Rational NN"),
]

FAMILY_COLORS = {
    "CFNN": "#22577A",
    "MLP": "#68717A",
    "KAN": "#E76F51",
    "SIREN": "#2A9D8F",
    "Fourier-feature MLP": "#9467BD",
    "Gaussian RBF": "#8C6D31",
    "Local nested CF control": "#4C78A8",
    "Rational activation NN": "#C44E52",
}

FAMILY_MARKERS = {
    "CFNN": "o",
    "MLP": "s",
    "KAN": "^",
    "SIREN": "D",
    "Fourier-feature MLP": "P",
    "Gaussian RBF": "X",
    "Local nested CF control": "v",
    "Rational activation NN": "*",
}


def _load_summary() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT_ROOT / "summary.csv")


def _ordered_rows(summary: pd.DataFrame, task: str, density: float | None = None) -> pd.DataFrame:
    rows = summary[summary["task"].eq(task)].copy()
    if density is None:
        rows = rows[rows["frf_density"].isna()]
    else:
        rows = rows[rows["frf_density"].eq(float(density))]
    wanted = [family for family, _ in FAMILIES]
    rows = rows[rows["family"].isin(wanted)].copy()
    rows["_order"] = rows["family"].map({family: index for index, (family, _) in enumerate(FAMILIES)})
    return rows.sort_values("_order")


def _draw_error_bar_panel(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    title: str,
    x_limit: float,
    show_y_labels: bool,
) -> None:
    labels = [label for _, label in FAMILIES]
    y = np.arange(len(FAMILIES))
    peak = rows["median_peak_nrmse"].to_numpy(dtype=float)
    global_error = rows["median_global_nrmse"].to_numpy(dtype=float)

    peak_colors = ["#E76F51" if family == "CFNN" else "#F0B09C" for family in rows["family"]]
    global_colors = ["#2A9D8F" if family == "CFNN" else "#A7D8D0" for family in rows["family"]]
    ax.barh(
        y - 0.17,
        peak,
        height=0.30,
        color=peak_colors,
        edgecolor="#FFFFFF",
        linewidth=0.45,
        label="Peak-window",
        zorder=3,
    )
    ax.barh(
        y + 0.17,
        global_error,
        height=0.30,
        color=global_colors,
        edgecolor="#FFFFFF",
        linewidth=0.45,
        label="Global",
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels if show_y_labels else [])
    ax.invert_yaxis()
    ax.set_xlim(0.0, x_limit)
    ax.set_xlabel("Median NRMSE")
    ax.set_title(title, loc="left", pad=6)
    ax.grid(axis="x", color="#DDE2E6", lw=0.45)
    ax.tick_params(axis="y", length=0)
    if show_y_labels:
        ax.get_yticklabels()[0].set_fontweight("bold")


def _draw_frf_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    densities = [24, 96, 192]
    for family, label in FAMILIES:
        values = []
        for density in densities:
            rows = _ordered_rows(summary, "aluminium_frf", density=float(density))
            values.append(float(rows[rows["family"].eq(family)]["median_peak_nrmse"].iloc[0]))
        is_cfnn = family == "CFNN"
        ax.plot(
            densities,
            values,
            marker=FAMILY_MARKERS[family],
            color=FAMILY_COLORS[family],
            lw=1.65 if is_cfnn else 0.88,
            ms=5.0 if is_cfnn else 3.5,
            alpha=1.0 if is_cfnn else 0.70,
            label=label,
            zorder=4 if is_cfnn else 2,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(densities)
    ax.set_xticklabels([str(value) for value in densities])
    ax.set_xlabel("FRF training samples")
    ax.set_ylabel("Peak-window median NRMSE")
    ax.set_ylim(0.995, 1.18)
    ax.set_title("c  Aluminium plate response", loc="left", pad=6)
    ax.grid(axis="y", color="#DDE2E6", lw=0.45)
    ax.legend(
        frameon=False,
        loc="upper right",
        ncol=2,
        handlelength=1.4,
        columnspacing=0.8,
        borderaxespad=0.2,
        fontsize=5.7,
    )


def draw(summary: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.1,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.08), constrained_layout=False)
    fig.subplots_adjust(left=0.115, right=0.995, bottom=0.18, top=0.83, wspace=0.36)

    _draw_error_bar_panel(
        axes[0],
        _ordered_rows(summary, "microwave_fano"),
        title="a  Microwave Fano curve",
        x_limit=0.265,
        show_y_labels=True,
    )
    _draw_error_bar_panel(
        axes[1],
        _ordered_rows(summary, "microstrip_resonator"),
        title="b  Microstrip transmission curve",
        x_limit=0.095,
        show_y_labels=False,
    )
    _draw_frf_panel(axes[2], summary)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        ncol=2,
        columnspacing=1.5,
        handlelength=1.6,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = _load_summary()
    draw(summary)
    print({
        "pdf": str(OUT_STEM.with_suffix(".pdf")),
        "png": str(OUT_STEM.with_suffix(".png")),
    })


if __name__ == "__main__":
    main()

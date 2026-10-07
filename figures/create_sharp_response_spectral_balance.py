"""Create a sharp-response spectral-balance figure for the CFNN paper."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PACKAGE_ROOT / "results" / "figures"
OUT_STEM = OUT_DIR / "sharp_response_spectral_balance"

ROWS = [
    ("CFNN-Hybrid", 0.0094, 0.0106, "#2A9D8F"),
    ("SIREN", 0.0316, 0.0229, "#8E63B7"),
    ("RFF-MLP", 0.0229, 0.0236, "#33A6A0"),
    ("Chebyshev-KAN", 0.0178, 0.0250, "#4C78A8"),
    ("MLP", 0.0213, 0.0389, "#E76F51"),
]


def _draw_band_panel(ax: plt.Axes) -> None:
    labels = [row[0] for row in ROWS]
    low = np.array([row[1] for row in ROWS])
    high = np.array([row[2] for row in ROWS])
    y = np.arange(len(labels))

    ax.barh(
        y - 0.17,
        low,
        height=0.30,
        color="#6BAED6",
        edgecolor="white",
        linewidth=0.45,
        label="Low frequency",
        zorder=3,
    )
    ax.barh(
        y + 0.17,
        high,
        height=0.30,
        color="#F28E7C",
        edgecolor="white",
        linewidth=0.45,
        label="High frequency",
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 0.043)
    ax.set_xlabel("Relative residual error")
    ax.set_title("a  Low- and high-frequency residuals", loc="left", pad=6)
    ax.grid(axis="x", color="#DDE2E6", lw=0.45)
    ax.tick_params(axis="y", length=0)
    ax.get_yticklabels()[0].set_fontweight("bold")
    return ax.get_legend_handles_labels()


def _draw_balance_panel(ax: plt.Axes) -> None:
    for label, low, high, color in ROWS:
        ax.scatter(
            low,
            high,
            s=42 if label == "CFNN-Hybrid" else 30,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=4 if label == "CFNN-Hybrid" else 3,
        )
        dx = 0.0010
        dy = 0.0010
        if label == "CFNN-Hybrid":
            dx, dy = 0.0011, -0.0019
        elif label == "MLP":
            dx, dy = -0.0064, 0.0011
        elif label == "SIREN":
            dx, dy = 0.0011, -0.0017
        ax.text(low + dx, high + dy, label, fontsize=6.1, color="#222222")

    limit = 0.043
    ax.plot([0.0, limit], [0.0, limit], color="#7F878E", lw=0.75, ls=(0, (3, 2)))
    ax.fill_between([0.0, limit], [0.0, limit], [limit, limit], color="#F28E7C", alpha=0.09)
    ax.fill_between([0.0, limit], [0.0, 0.0], [0.0, limit], color="#6BAED6", alpha=0.08)
    ax.set_xlim(0.006, 0.035)
    ax.set_ylim(0.007, 0.042)
    ax.set_xlabel("Low-frequency residual")
    ax.set_ylabel("High-frequency residual")
    ax.set_title("b  Frequency-balance view", loc="left", pad=6)
    ax.grid(color="#DDE2E6", lw=0.45)


def _draw_worst_band_panel(ax: plt.Axes) -> None:
    labels = [row[0] for row in ROWS]
    values = np.array([max(row[1], row[2]) for row in ROWS])
    y = np.arange(len(labels))
    colors = [row[3] for row in ROWS]

    ax.barh(
        y,
        values,
        height=0.42,
        color=colors,
        alpha=0.86,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 0.043)
    ax.set_xlabel("Worst-band residual")
    ax.set_title("c  Error controlled in both bands", loc="left", pad=6)
    ax.grid(axis="x", color="#DDE2E6", lw=0.45)
    ax.tick_params(axis="y", length=0)
    ax.get_yticklabels()[0].set_fontweight("bold")


def draw() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.3,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.72), constrained_layout=False)
    fig.subplots_adjust(left=0.115, right=0.995, bottom=0.22, top=0.78, wspace=0.42)

    handles, labels = _draw_band_panel(axes[0])
    _draw_balance_panel(axes[1])
    _draw_worst_band_panel(axes[2])

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
    draw()
    print({
        "pdf": str(OUT_STEM.with_suffix(".pdf")),
        "png": str(OUT_STEM.with_suffix(".png")),
    })


if __name__ == "__main__":
    main()

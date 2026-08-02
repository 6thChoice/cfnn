"""Create a production diagnostic figure for CFNN and CFNN-Hybrid."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = (
    REPO_ROOT
    / "experiment_refine"
    / "stability_production_results"
    / "stability_production_results.json"
)
OUT_DIR = REPO_ROOT / "CFNN_paper_nmi" / "img" / "stability"
OUT_STEM = OUT_DIR / "cfnn_hybrid_production_diagnostic"

MODELS = ["CFNN", "CFNN-Hybrid"]
COLORS = {
    "CFNN": "#496A8E",
    "CFNN-Hybrid": "#2A9D8F",
}
SEED_OFFSETS = np.array([-0.045, -0.022, 0.0, 0.022, 0.045])


def _load_results() -> dict:
    return json.loads(RESULT_PATH.read_text())


def _runs_by_seed(data: dict, condition: str) -> dict[str, dict[int, dict]]:
    runs = data["designs"]["parameter_matched"][condition]["runs"]
    return {
        model: {int(run["seed"]): run for run in runs[model]}
        for model in MODELS
    }


def _draw_paired_loss_panel(
    ax: plt.Axes,
    data: dict,
    *,
    condition: str,
    title: str,
    ylim: tuple[float, float],
    yticks: list[float],
) -> None:
    by_seed = _runs_by_seed(data, condition)
    seeds = list(data["seeds"])
    x_positions = {"CFNN": 0.0, "CFNN-Hybrid": 1.0}

    for index, seed in enumerate(seeds):
        x0 = x_positions["CFNN"] + SEED_OFFSETS[index]
        x1 = x_positions["CFNN-Hybrid"] + SEED_OFFSETS[index]
        y0 = by_seed["CFNN"][seed]["test_loss"]
        y1 = by_seed["CFNN-Hybrid"][seed]["test_loss"]
        ax.plot([x0, x1], [y0, y1], color="#A9B4BE", lw=0.75, zorder=1)

    for model in MODELS:
        values = np.array([by_seed[model][seed]["test_loss"] for seed in seeds])
        x = x_positions[model] + SEED_OFFSETS
        ax.scatter(
            x,
            values,
            s=22,
            color=COLORS[model],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        ax.errorbar(
            x_positions[model],
            mean,
            yerr=sd,
            fmt="_",
            ms=24,
            color="#111111",
            ecolor="#111111",
            elinewidth=0.85,
            capsize=3,
            capthick=0.85,
            zorder=4,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(MODELS)
    ax.set_xlim(-0.28, 1.28)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel("Held-out loss")
    ax.set_title(title, loc="left", pad=6)
    ax.grid(axis="y", color="#DDE2E6", lw=0.45)


def _draw_ratio_panel(ax: plt.Axes, data: dict) -> None:
    seeds = list(data["seeds"])
    labels = ["Fixed", "Tuned"]
    y_positions = np.array([1.0, 0.0])
    color_map = {"Fixed": "#88A9C3", "Tuned": "#2A9D8F"}

    for row, condition in enumerate(["fixed", "tuned"]):
        by_seed = _runs_by_seed(data, condition)
        ratios = np.array([
            by_seed["CFNN-Hybrid"][seed]["test_loss"] / by_seed["CFNN"][seed]["test_loss"]
            for seed in seeds
        ])
        y = y_positions[row]
        y_jitter = y + SEED_OFFSETS
        for value, y_seed in zip(ratios, y_jitter):
            ax.plot([1.0, value], [y_seed, y_seed], color="#D9DEE3", lw=1.0, zorder=1)
        ax.scatter(
            ratios,
            y_jitter,
            s=24,
            color=color_map[labels[row]],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        mean = float(ratios.mean())
        sd = float(ratios.std(ddof=1))
        ax.errorbar(
            mean,
            y,
            xerr=sd,
            fmt="o",
            ms=3.4,
            color=color_map[labels[row]],
            ecolor="#222222",
            elinewidth=0.8,
            capsize=2.5,
            capthick=0.8,
            zorder=4,
        )

    ax.axvline(1.0, color="#555555", lw=0.8, ls=(0, (3, 2)))
    ax.set_xlim(0.0, 1.1)
    ax.set_ylim(-0.62, 1.32)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Hybrid / CFNN held-out loss")
    ax.set_title("c  Seed-wise relative loss", loc="left", pad=6)
    ax.grid(axis="x", color="#DDE2E6", lw=0.45)
    ax.tick_params(axis="y", length=0)


def draw(data: dict) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.3,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "legend.fontsize": 6.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.65), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.22, top=0.85, wspace=0.38)

    _draw_paired_loss_panel(
        axes[0],
        data,
        condition="fixed",
        title="a  Fixed training protocol",
        ylim=(0.0, 1.02),
        yticks=[0.0, 0.25, 0.50, 0.75, 1.00],
    )
    _draw_paired_loss_panel(
        axes[1],
        data,
        condition="tuned",
        title="b  Tuned training protocol",
        ylim=(0.10, 0.18),
        yticks=[0.10, 0.12, 0.14, 0.16, 0.18],
    )
    _draw_ratio_panel(axes[2], data)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = _load_results()
    draw(data)
    print({
        "pdf": str(OUT_STEM.with_suffix(".pdf")),
        "png": str(OUT_STEM.with_suffix(".png")),
    })


if __name__ == "__main__":
    main()

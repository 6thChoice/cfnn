"""Create a training-loss stability figure for CFNN and CFNN-Hybrid."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PACKAGE_ROOT / "results" / "figures" / "cfnn_hybrid_training_stability_cache.json"
OUT_DIR = PACKAGE_ROOT / "results" / "figures"
OUT_STEM = OUT_DIR / "cfnn_hybrid_training_stability"

MODELS = ["CFNN", "CFNN-Hybrid"]
COLORS = {
    "CFNN": "#6E7378",
    "CFNN-Hybrid": "#2A9D8F",
}
LINESTYLES = {
    "CFNN": (0, (4, 2)),
    "CFNN-Hybrid": "-",
}

REPRESENTATIVE_CASE = {
    "func_name": "f7_high_freq",
    "depth": 8,
    "seed": 123,
}


def _load_results() -> dict:
    return json.loads(RESULT_PATH.read_text())


def _selected_experiments(data: dict) -> list[dict]:
    return [exp for exp in data["representative_loss_histories"] if exp["model_type"] in MODELS]


def _get_case(data: dict, model: str) -> np.ndarray:
    for exp in _selected_experiments(data):
        if (
            exp["model_type"] == model
            and exp["func_name"] == REPRESENTATIVE_CASE["func_name"]
            and int(exp["depth"]) == REPRESENTATIVE_CASE["depth"]
            and int(exp["seed"]) == REPRESENTATIVE_CASE["seed"]
        ):
            return np.asarray(exp["loss_history"], dtype=float)
    raise KeyError((model, REPRESENTATIVE_CASE))


def _large_jump_count(loss_history: list[float], threshold: float = 1.6) -> int:
    values = np.asarray(loss_history, dtype=float)
    values = np.maximum(values, 1e-12)
    ratios = values[1:] / values[:-1]
    return int(np.sum(ratios > threshold))


def _max_rebound_factor(loss_history: list[float]) -> float:
    values = np.asarray(loss_history, dtype=float)
    values = np.maximum(values, 1e-12)
    return float(np.max(values[1:] / values[:-1]))


def _draw_full_curve(ax: plt.Axes, data: dict) -> None:
    for model in MODELS:
        loss = _get_case(data, model)
        epochs = np.arange(1, len(loss) + 1)
        ax.plot(
            epochs,
            loss,
            lw=1.05,
            color=COLORS[model],
            linestyle=LINESTYLES[model],
            label=model,
            alpha=0.96,
        )
    ax.set_yscale("log")
    ax.set_xlim(1, 2000)
    ax.set_ylim(0.30, 80)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("a  Full training loss trajectory", loc="left", pad=6)
    ax.grid(axis="both", color="#DDE2E6", lw=0.45)


def _draw_zoom_curve(ax: plt.Axes, data: dict) -> None:
    start, stop = 680, 765
    for model in MODELS:
        loss = _get_case(data, model)
        epochs = np.arange(1, len(loss) + 1)
        mask = (epochs >= start) & (epochs <= stop)
        ax.plot(
            epochs[mask],
            loss[mask],
            lw=1.15,
            color=COLORS[model],
            linestyle=LINESTYLES[model],
            label=model,
            alpha=0.96,
        )
    ax.set_yscale("log")
    ax.set_xlim(start, stop)
    ax.set_ylim(0.34, 80)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("b  Late-epoch instability window", loc="left", pad=6)
    ax.grid(axis="both", color="#DDE2E6", lw=0.45)


def _draw_rebound_distribution(ax: plt.Axes, data: dict) -> None:
    rows = {model: [] for model in MODELS}
    for exp in data["rebound_summary_rows"]:
        if exp["func_name"] not in {"f4_composite", "f7_high_freq"}:
            continue
        rows[exp["model_type"]].append(float(exp["max_rebound_factor"]))

    y_positions = {"CFNN": 1.0, "CFNN-Hybrid": 0.0}
    jitter = np.linspace(-0.105, 0.105, 18)
    rng = np.random.default_rng(7)
    rng.shuffle(jitter)

    for model in MODELS:
        values = np.asarray(rows[model], dtype=float)
        y = np.full(len(values), y_positions[model]) + jitter[: len(values)]
        ax.scatter(
            values,
            y,
            s=18,
            color=COLORS[model],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.88,
            zorder=3,
        )
        median = float(np.median(values))
        q1, q3 = np.percentile(values, [25, 75])
        ax.errorbar(
            median,
            y_positions[model],
            xerr=np.array([[median - q1], [q3 - median]]),
            fmt="|",
            ms=14,
            color="#151515",
            ecolor="#151515",
            elinewidth=0.9,
            capsize=3,
            zorder=4,
        )

    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(MODELS)
    ax.set_ylim(-0.42, 1.42)
    ax.set_xscale("log")
    ax.set_xlim(0.95, 180)
    ax.set_xticks([1, 2, 5, 10, 50, 100])
    ax.set_xticklabels(["1", "2", "5", "10", "50", "100"])
    ax.axvline(1.0, color="#555555", lw=0.75, ls=(0, (3, 2)))
    ax.set_xlabel("Maximum one-step loss increase")
    ax.set_title("c  Loss rebound across diagnostics", loc="left", pad=6)
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

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.72), constrained_layout=False)
    fig.subplots_adjust(left=0.078, right=0.995, bottom=0.215, top=0.78, wspace=0.42)

    _draw_full_curve(axes[0], data)
    _draw_zoom_curve(axes[1], data)
    _draw_rebound_distribution(axes[2], data)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.8,
    )

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

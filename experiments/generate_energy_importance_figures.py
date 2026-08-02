#!/usr/bin/env python3
"""Generate table-consistent Energy feature-importance figures.

This replaces the stale Energy interpretability figures whose visual emphasis was
Boost/Hybrid. The available fair rerun artifacts in experiment_refine contain the
5-seed summary table but not raw SHAP arrays for every model, so these figures
visualize the revised Table 9 metrics directly instead of fabricating per-sample
SHAP values.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("/home/zxc/CodeBase/cofrnet/CFNN_paper/img/interpretable")
EXP_DIR = Path("/home/zxc/CodeBase/cofrnet/experiment_refine")

MODELS = ["MLP", "CFNN", "CFNN-Boost", "CFNN-MoE", "CFNN-Hybrid"]
SPEARMAN = np.array([0.432, 0.494, 0.386, 0.664, 0.448])
SPEARMAN_STD = np.array([0.038, 0.180, 0.085, 0.151, 0.031])
CONSISTENCY = np.array([63.6, 62.1, 64.3, 75.0, 64.3])
CONSISTENCY_STD = np.array([1.4, 7.4, 8.8, 7.5, 3.2])
TOP3 = np.array([46.7, 46.7, 26.7, 46.7, 46.7])
TOP3_STD = np.array([16.3, 16.3, 13.3, 16.3, 16.3])
TOP5 = np.array([80.0, 80.0, 80.0, 80.0, 80.0])
TOP5_STD = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
R2 = np.array([0.9970, 0.9932, 0.9244, 0.9864, 0.9976])
R2_STD = np.array([0.0003, 0.0025, 0.0271, 0.0064, 0.0002])

COLORS = ["#777777", "#4C78A8", "#F58518", "#54A24B", "#B279A2"]
MOE_IDX = MODELS.index("CFNN-MoE")


def annotate_bars(ax, bars, fmt="{:.3f}", dy=0.01):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=9, fontweight="bold")


def plot_comparison():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0))
    x = np.arange(len(MODELS))

    panels = [
        (axes[0, 0], SPEARMAN, SPEARMAN_STD, "Spearman ρ with domain hierarchy", "ρ", "{:.3f}", 0.015),
        (axes[0, 1], CONSISTENCY, CONSISTENCY_STD, "Ranking consistency", "%", "{:.1f}", 1.2),
        (axes[1, 0], TOP3, TOP3_STD, "Top-3 recovery", "%", "{:.1f}", 1.2),
        (axes[1, 1], TOP5, TOP5_STD, "Top-5 recovery", "%", "{:.1f}", 1.2),
    ]

    for ax, vals, errs, title, ylabel, fmt, dy in panels:
        bars = ax.bar(x, vals, yerr=errs, capsize=4, color=COLORS, alpha=0.88,
                      edgecolor="white", linewidth=1.0)
        bars[MOE_IDX].set_edgecolor("black")
        bars[MOE_IDX].set_linewidth(2.2)
        annotate_bars(ax, bars, fmt=fmt, dy=dy)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0, 0].set_ylim(0, max(SPEARMAN + SPEARMAN_STD) + 0.18)
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.set_ylim(0, 105)

    fig.suptitle(
        "Energy Efficiency interpretability under the fair 5-seed protocol\n"
        "CFNN-MoE shows the strongest domain-ranking alignment; Top-5 recovery is tied",
        fontsize=15, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUT_DIR / "feature_importance_comparison.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(EXP_DIR / "energy_feature_importance_comparison_updated.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_correlation():
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    sizes = 170 + 7.5 * (CONSISTENCY - CONSISTENCY.min())
    for i, model in enumerate(MODELS):
        ax.errorbar(SPEARMAN[i], CONSISTENCY[i], xerr=SPEARMAN_STD[i], yerr=CONSISTENCY_STD[i],
                    fmt="none", ecolor=COLORS[i], capsize=4, alpha=0.8)
        ax.scatter(SPEARMAN[i], CONSISTENCY[i], s=sizes[i], color=COLORS[i],
                   edgecolor="black" if i == MOE_IDX else "white",
                   linewidth=2.0 if i == MOE_IDX else 1.0, zorder=3)
        ax.text(SPEARMAN[i] + 0.012, CONSISTENCY[i] + 0.55, model,
                fontsize=10, fontweight="bold" if i == MOE_IDX else "normal")

    ax.axvline(SPEARMAN[MOE_IDX], color=COLORS[MOE_IDX], linestyle=":", alpha=0.45)
    ax.axhline(CONSISTENCY[MOE_IDX], color=COLORS[MOE_IDX], linestyle=":", alpha=0.45)
    ax.set_xlabel("Spearman ρ with thermodynamic feature hierarchy", fontsize=12)
    ax.set_ylabel("Ranking consistency (%)", fontsize=12)
    ax.set_title("Domain-alignment summary for Energy feature attribution", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0.30, 0.85)
    ax.set_ylim(50, 88)

    note = (
        "Fair Table 9 metrics, mean ± std over 5 seeds.\n"
        "Predictive R² is high for all strong models; interpretability alignment separates them."
    )
    ax.text(0.02, 0.03, note, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92))

    fig.tight_layout()
    out = OUT_DIR / "feature_importance_correlation.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(EXP_DIR / "energy_feature_importance_correlation_updated.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    out1 = plot_comparison()
    out2 = plot_correlation()
    print(f"wrote {out1}")
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()

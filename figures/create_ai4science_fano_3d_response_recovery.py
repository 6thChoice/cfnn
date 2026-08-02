"""Create a 3D Microwave Fano response-shape recovery figure."""
from __future__ import annotations

import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = PACKAGE_ROOT / "results" / "figures"
SOURCE_CACHE = OUT_DIR / "fano_response_recovery_cache.npz"
OUT_STEM = OUT_DIR / "fano_3d_response_recovery"

FAMILIES = [
    ("prediction_cfnn", "CFNN", "#009988", "-"),
    ("prediction_mlp", "MLP", "#EE7733", (0, (3, 1.7))),
    ("prediction_local_nested_cf_control", "Local CF control", "#0077BB", (0, (1.2, 1.4))),
]


def _ensure_cache() -> None:
    if SOURCE_CACHE.exists():
        return
    runpy.run_path(str(SCRIPT_DIR / "create_ai4science_fano_response_recovery.py"), run_name="__main__")


def _as_complex(response: np.ndarray) -> np.ndarray:
    response = np.asarray(response)
    return response[:, 0].astype(np.float64) + 1j * response[:, 1].astype(np.float64)


def _set_3d_style(ax, *, title: str, show_z_label: bool = True) -> None:
    ax.set_title(title, loc="left", y=1.06, pad=0)
    ax.set_xlabel(r"$f - f_0$ (MHz)", labelpad=-1)
    ax.set_ylabel(r"Real$(S)$", labelpad=-1)
    ax.set_zlabel(r"Imag$(S)$" if show_z_label else "", labelpad=-1)
    if not show_z_label:
        ax.set_zticklabels([])
    ax.view_init(elev=23, azim=-57)
    ax.tick_params(axis="both", which="major", pad=-2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
        axis.pane.set_edgecolor((0.88, 0.90, 0.92, 1.0))
    ax.grid(True, color="#DDE2E6", linewidth=0.45)


def _draw_target(ax, x_mhz: np.ndarray, truth: np.ndarray, x_obs: np.ndarray, observed: np.ndarray) -> None:
    ax.plot(x_mhz, truth.real, truth.imag, color="#6F777D", lw=1.45, alpha=0.72, label="Measured truth", zorder=4)
    ax.scatter(
        x_obs,
        observed.real,
        observed.imag,
        s=16,
        color="#9AA3AA",
        edgecolor="white",
        linewidth=0.25,
        alpha=0.82,
        label="Sparse observations",
        zorder=6,
    )


def _draw_observations(ax, x_obs: np.ndarray, observed: np.ndarray, *, mask: np.ndarray | None = None) -> None:
    if mask is None:
        mask = np.ones_like(x_obs, dtype=bool)
    ax.scatter(
        x_obs[mask],
        observed.real[mask],
        observed.imag[mask],
        s=10,
        color="#A9B1B7",
        edgecolor="white",
        linewidth=0.2,
        alpha=0.46,
        label="Sparse observations",
        zorder=3,
    )


def _draw_recovery(
    ax,
    cache: np.lib.npyio.NpzFile,
    x_mhz: np.ndarray,
    truth: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> None:
    if mask is None:
        mask = np.ones_like(x_mhz, dtype=bool)
    ax.plot(
        x_mhz[mask],
        truth.real[mask],
        truth.imag[mask],
        color="#8A9399",
        lw=1.25,
        alpha=0.62,
        label="Measured truth",
        zorder=3,
    )
    for key, label, color, linestyle in FAMILIES:
        prediction = _as_complex(cache[key])
        ax.plot(
            x_mhz[mask],
            prediction.real[mask],
            prediction.imag[mask],
            color=color,
            lw=1.65,
            linestyle=linestyle,
            label=label,
            zorder=6,
        )


def _draw_residuals(
    ax: plt.Axes,
    cache: np.lib.npyio.NpzFile,
    x_mhz: np.ndarray,
    truth: np.ndarray,
    *,
    window_half_width_mhz: float,
) -> None:
    sample_count = 145
    sample_index = np.unique(np.linspace(0, len(x_mhz) - 1, sample_count, dtype=int))
    for key, label, color, linestyle in FAMILIES:
        prediction = _as_complex(cache[key])
        residual = np.abs(prediction - truth)
        ax.plot(
            x_mhz[sample_index],
            residual[sample_index],
            color=color,
            lw=1.65,
            linestyle=linestyle,
            label=label,
            zorder=4,
        )
    ax.axvspan(-window_half_width_mhz, window_half_width_mhz, color="#EEF1F3", zorder=0)
    ax.axvline(0.0, color="#5C6670", lw=0.75, zorder=1)
    ax.set_title("c  Recovery residuals", loc="left", y=1.06, pad=0)
    ax.set_xlabel(r"$f - f_0$ (MHz)")
    ax.set_ylabel("Complex error")
    ax.grid(axis="both", color="#DDE2E6", lw=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _set_limits(ax, x_mhz: np.ndarray, responses: list[np.ndarray], *, mask: np.ndarray | None = None) -> None:
    if mask is None:
        mask = np.ones_like(x_mhz, dtype=bool)
    real = np.concatenate([response.real[mask] for response in responses])
    imag = np.concatenate([response.imag[mask] for response in responses])
    real_pad = max((real.max() - real.min()) * 0.10, 1e-4)
    imag_pad = max((imag.max() - imag.min()) * 0.10, 1e-4)
    ax.set_xlim(float(x_mhz[mask].min()), float(x_mhz[mask].max()))
    ax.set_ylim(float(real.min() - real_pad), float(real.max() + real_pad))
    ax.set_zlim(float(imag.min() - imag_pad), float(imag.max() + imag_pad))


def draw(cache: np.lib.npyio.NpzFile) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 6.9,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 6.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    frequency = cache["frequency"]
    f0_hz = float(cache["f0_hz"])
    linewidth_hz = float(cache["linewidth_hz"])
    x_mhz = (frequency - f0_hz) / 1e6
    x_obs = (cache["x_observed"] - f0_hz) / 1e6
    truth = _as_complex(cache["truth"])
    observed = _as_complex(cache["y_observed"])
    predictions = [_as_complex(cache[key]) for key, _, _, _ in FAMILIES]
    window_half_width_mhz = 2.0 * linewidth_hz / 1e6
    zoom_mask = np.abs(x_mhz) <= 1.35 * linewidth_hz / 1e6

    fig = plt.figure(figsize=(7.15, 2.92), constrained_layout=False)
    axes = [
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2, projection="3d"),
        fig.add_subplot(1, 3, 3),
    ]
    fig.subplots_adjust(left=0.012, right=0.985, bottom=0.14, top=0.79, wspace=0.34)

    _draw_target(axes[0], x_mhz, truth, x_obs, observed)
    _set_3d_style(axes[0], title="a  Target response shape")
    _set_limits(axes[0], x_mhz, [truth])

    _draw_recovery(axes[1], cache, x_mhz, truth, mask=zoom_mask)
    obs_zoom_mask = np.abs(x_obs) <= 1.35 * linewidth_hz / 1e6
    _draw_observations(axes[1], x_obs, observed, mask=obs_zoom_mask)
    _set_3d_style(axes[1], title="b  Resonance-window recovery", show_z_label=False)
    _set_limits(axes[1], x_mhz, [truth, *predictions], mask=zoom_mask)

    _draw_residuals(axes[2], cache, x_mhz, truth, window_half_width_mhz=window_half_width_mhz)

    handles, labels = [], []
    for ax in axes[:2]:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    ax_handles, ax_labels = axes[2].get_legend_handles_labels()
    for handle, label in zip(ax_handles, ax_labels):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.01),
        frameon=False,
        ncol=5,
        handlelength=1.6,
        columnspacing=0.9,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _ensure_cache()
    cache = np.load(SOURCE_CACHE, allow_pickle=False)
    draw(cache)
    print(
        {
            "pdf": str(OUT_STEM.with_suffix(".pdf")),
            "png": str(OUT_STEM.with_suffix(".png")),
            "cache": str(SOURCE_CACHE),
        }
    )


if __name__ == "__main__":
    main()

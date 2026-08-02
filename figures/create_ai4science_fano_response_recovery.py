"""Create a representative Microwave Fano response-recovery figure."""
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

from ai4science_fano import FanoReference  # noqa: E402
from ai4science_fano_a2 import make_sparse_curve_split  # noqa: E402
from confirmatory_models import build_model  # noqa: E402
from confirmatory_protocol import seeded_build  # noqa: E402
from downstream_protocol import load_microwave_fano  # noqa: E402
from downstream_training import fit_task_model, prepare_bundle  # noqa: E402


RAW_ROOT = PACKAGE_ROOT / "data" / "raw"
REFERENCE_PATH = PACKAGE_ROOT / "results" / "ai4science" / "microwave_fano" / "reference_parameters.json"
SELECTION_ROOT = PACKAGE_ROOT / "results" / "selections" / "microwave_fano"
OUT_DIR = PACKAGE_ROOT / "results" / "figures"
OUT_STEM = OUT_DIR / "fano_response_recovery"
CACHE_PATH = OUT_DIR / "fano_response_recovery_cache.npz"

CURVE_ID = "overcoupled:r3:p25"
OBSERVATION_BUDGET = 128
VALIDATION_COUNT = 128
SPLIT_SEED = 829
INIT_SEED = 17011
MODEL_BUDGET = 512

FAMILIES = [
    ("CFNN", "CFNN", "#009988", "-"),
    ("MLP", "MLP", "#EE7733", (0, (3, 1.7))),
    ("Local nested CF control", "Local CF control", "#0077BB", (0, (1.2, 1.4))),
]


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _as_complex(response: np.ndarray) -> np.ndarray:
    response = np.asarray(response)
    return response[:, 0].astype(np.float64) + 1j * response[:, 1].astype(np.float64)


def _load_reference() -> FanoReference:
    rows = json.loads(REFERENCE_PATH.read_text())
    row = next(item for item in rows if item["curve_id"] == CURVE_ID)
    return FanoReference(**row)


def _load_selection(family: str) -> dict:
    path = SELECTION_ROOT / f"{_slug(family)}__{MODEL_BUDGET}.json"
    return json.loads(path.read_text())


def _train_prediction(family: str, bundle, device: torch.device) -> np.ndarray:
    selection = _load_selection(family)
    candidate = selection["candidate"]
    optimizer = selection["optimizer"]
    model, _ = seeded_build(
        INIT_SEED,
        lambda: build_model(
            family,
            input_dim=1,
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
        max_steps=800,
        min_steps=200,
        patience=200,
        checkpoints=[20, 80, 200, 500, 800],
        gradient_clip=5.0,
        batch_seed=SPLIT_SEED + 49979687,
        validation_interval=20,
    )
    if trained.status != "ok":
        raise RuntimeError(f"{family} training failed: {trained.status}")
    return trained.prediction


def _build_cache() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curves = {curve.curve_id: curve for curve in load_microwave_fano(RAW_ROOT)}
    curve = curves[CURVE_ID]
    reference = _load_reference()
    bundle = make_sparse_curve_split(
        curve,
        observation_count=OBSERVATION_BUDGET,
        validation_count=VALIDATION_COUNT,
        split_seed=SPLIT_SEED,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arrays: dict[str, np.ndarray | str | float | int] = {
        "frequency": bundle.x_test[:, 0].astype(np.float64),
        "truth": np.asarray(bundle.y_test, dtype=np.float64),
        "x_observed": bundle.x_train[:, 0].astype(np.float64),
        "y_observed": np.asarray(bundle.y_train, dtype=np.float64),
        "f0_hz": float(reference.f0_hz),
        "linewidth_hz": float(reference.linewidth_hz),
        "curve_id": CURVE_ID,
        "observation_budget": OBSERVATION_BUDGET,
        "split_seed": SPLIT_SEED,
        "init_seed": INIT_SEED,
        "device": str(device),
    }
    for family, _, _, _ in FAMILIES:
        print(f"training {family}", flush=True)
        arrays[f"prediction_{_slug(family)}"] = _train_prediction(family, bundle, device)
    np.savez_compressed(CACHE_PATH, **arrays)


def _load_cache() -> np.lib.npyio.NpzFile:
    if not CACHE_PATH.exists():
        _build_cache()
    return np.load(CACHE_PATH, allow_pickle=False)


def _phase(response: np.ndarray, truth_phase: np.ndarray | None = None) -> np.ndarray:
    phase = np.unwrap(np.angle(_as_complex(response)))
    if truth_phase is not None:
        shift = np.median(truth_phase - phase)
        phase = phase + 2.0 * np.pi * np.round(shift / (2.0 * np.pi))
    return phase


def _draw_magnitude(ax: plt.Axes, cache: np.lib.npyio.NpzFile, x_mhz: np.ndarray) -> None:
    truth = _as_complex(cache["truth"])
    observed = _as_complex(cache["y_observed"])
    x_obs = (cache["x_observed"] - float(cache["f0_hz"])) / 1e6
    ax.plot(x_mhz, np.abs(truth), color="#111111", lw=1.5, label="Measured truth", zorder=5)
    ax.scatter(
        x_obs,
        np.abs(observed),
        s=7,
        color="#9AA3AA",
        edgecolor="white",
        linewidth=0.25,
        alpha=0.62,
        label="Sparse observations",
        zorder=3,
    )
    for family, label, color, linestyle in FAMILIES:
        prediction = _as_complex(cache[f"prediction_{_slug(family)}"])
        ax.plot(x_mhz, np.abs(prediction), color=color, lw=1.45, linestyle=linestyle, label=label, zorder=6)
    ax.set_ylabel(r"Magnitude $|S|$")
    ax.set_title("a  Magnitude response", loc="left", y=1.18, pad=0)


def _draw_phase(ax: plt.Axes, cache: np.lib.npyio.NpzFile, x_mhz: np.ndarray) -> None:
    truth_phase = _phase(cache["truth"])
    observed_phase = _phase(cache["y_observed"])
    x_obs = (cache["x_observed"] - float(cache["f0_hz"])) / 1e6
    ax.plot(x_mhz, truth_phase, color="#111111", lw=1.5, label="Measured truth", zorder=5)
    ax.scatter(
        x_obs,
        observed_phase,
        s=7,
        color="#9AA3AA",
        edgecolor="white",
        linewidth=0.25,
        alpha=0.62,
        label="Sparse observations",
        zorder=3,
    )
    for family, label, color, linestyle in FAMILIES:
        prediction_phase = _phase(cache[f"prediction_{_slug(family)}"], truth_phase)
        ax.plot(x_mhz, prediction_phase, color=color, lw=1.45, linestyle=linestyle, label=label, zorder=6)
    ax.set_ylabel("Unwrapped phase (rad)")
    ax.set_title("b  Phase transition", loc="left", y=1.18, pad=0)


def _draw_complex_plane(ax: plt.Axes, cache: np.lib.npyio.NpzFile) -> None:
    truth = _as_complex(cache["truth"])
    observed = _as_complex(cache["y_observed"])
    ax.plot(truth.real, truth.imag, color="#111111", lw=1.5, label="Measured truth", zorder=5)
    ax.scatter(
        observed.real,
        observed.imag,
        s=7,
        color="#9AA3AA",
        edgecolor="white",
        linewidth=0.25,
        alpha=0.62,
        label="Sparse observations",
        zorder=3,
    )
    for family, label, color, linestyle in FAMILIES:
        prediction = _as_complex(cache[f"prediction_{_slug(family)}"])
        ax.plot(prediction.real, prediction.imag, color=color, lw=1.45, linestyle=linestyle, label=label, zorder=6)
    ax.set_xlabel(r"Real$(S)$")
    ax.set_ylabel(r"Imag$(S)$")
    ax.set_title("c  Complex-plane trajectory", loc="left", y=1.18, pad=0)
    ax.set_aspect("equal", adjustable="box")


def draw(cache: np.lib.npyio.NpzFile) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.1,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.1,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    frequency = cache["frequency"]
    f0_hz = float(cache["f0_hz"])
    linewidth_hz = float(cache["linewidth_hz"])
    x_mhz = (frequency - f0_hz) / 1e6
    window_half_width_mhz = 2.0 * linewidth_hz / 1e6

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.62), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.77, wspace=0.34)

    _draw_magnitude(axes[0], cache, x_mhz)
    _draw_phase(axes[1], cache, x_mhz)
    _draw_complex_plane(axes[2], cache)

    for ax in axes[:2]:
        ax.axvspan(-window_half_width_mhz, window_half_width_mhz, color="#EEF1F3", zorder=0)
        ax.axvline(0.0, color="#5C6670", lw=0.75, zorder=1)
        ax.set_xlabel(r"Frequency offset from $f_0$ (MHz)")
        ax.grid(axis="both", color="#DDE2E6", lw=0.42)
    axes[2].grid(axis="both", color="#DDE2E6", lw=0.42)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        frameon=False,
        ncol=5,
        handlelength=1.75,
        columnspacing=0.9,
    )

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
        }
    )


if __name__ == "__main__":
    main()

"""Create a three-layer sharp-response mechanism figure.

The script rebuilds the cached spectral-bias arrays on first run, then reuses
them for fast redrawing. It keeps only CFNN-Hybrid and external baselines in the
figure to match the current manuscript scope.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import fft
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CODE = PACKAGE_ROOT / "src" / "cfnn_nmi" / "legacy" / "spectral_bias"
sys.path.insert(0, str(LEGACY_CODE))

from baselines import ChebyshevKAN, RFFMLP, SIREN  # noqa: E402
from cfnet import HybridRationalNet  # noqa: E402


OUT_DIR = PACKAGE_ROOT / "results" / "figures"
CACHE_PATH = OUT_DIR / "sharp_response_mechanism_cache.npz"
OUT_STEM = OUT_DIR / "sharp_response_mechanism"

MODELS = [
    ("cfnn_hybrid", "CFNN-Hybrid", "#2A9D8F", "-"),
    ("siren", "SIREN", "#8E63B7", (0, (4, 2))),
    ("rff_mlp", "RFF-MLP", "#33A6A0", (0, (4, 2))),
    ("chebyshev_kan", "Chebyshev-KAN", "#4C78A8", (0, (4, 2))),
    ("mlp", "MLP", "#E76F51", (0, (1.5, 1.5))),
]

PAPER_BANDS = [
    ("CFNN-Hybrid", 0.0094, 0.0106),
    ("SIREN", 0.0316, 0.0229),
    ("RFF-MLP", 0.0229, 0.0236),
    ("Chebyshev-KAN", 0.0178, 0.0250),
    ("MLP", 0.0213, 0.0389),
]


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _generate_data(seed: int = 42) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    np.random.seed(seed)
    n = 10000
    x12 = np.random.uniform(-2, 2, (n, 2))
    x3 = np.random.uniform(1, 3, (n, 1))
    x = np.hstack([x12, x3]).astype(np.float32)
    y = ((x[:, 0] * x[:, 1]) / (x[:, 2] + 1e-5)).reshape(-1, 1)
    y += 0.02 * np.random.normal(size=y.shape).astype(np.float32)
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.35, random_state=seed)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=30 / 35, random_state=seed
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def _train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float = 0.001,
) -> nn.Module:
    device = _device()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=40, factor=0.5)
    best_val_rmse = float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                val_mse += nn.functional.mse_loss(model(vx), vy, reduction="sum").item()
        val_rmse = np.sqrt(val_mse / len(val_loader.dataset))
        scheduler.step(val_rmse)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model


def _predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    device = _device()
    with torch.no_grad():
        tx = torch.from_numpy(x).to(device)
        return model(tx).detach().cpu().numpy().reshape(-1)


def _power_spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    yf = fft.fft(signal)
    power = np.abs(yf[: n // 2]) ** 2
    freq = fft.fftfreq(n, 1.0)[: n // 2]
    return freq, power


def _relative_psd(residual: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freq, residual_psd = _power_spectrum(residual)
    _, target_psd = _power_spectrum(target)
    return freq, residual_psd / (target_psd + 1e-10)


def _smooth(values: np.ndarray, window: int = 21) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def _build_models() -> dict[str, nn.Module]:
    return {
        "cfnn_hybrid": HybridRationalNet(3, 1, unit_degree=5, num_units=5),
        "siren": SIREN(3, 1, hidden_dim=7, hidden_layers=2, omega_0=15),
        "rff_mlp": RFFMLP(3, 1, hidden_dim=7, rff_features=14, sigma=0.17),
        "chebyshev_kan": ChebyshevKAN(3, 1, hidden_dim=3, degree=5, num_layers=2),
        "mlp": nn.Sequential(
            nn.Linear(3, 7),
            nn.Tanh(),
            nn.Linear(7, 7),
            nn.Tanh(),
            nn.Linear(7, 1),
        ),
    }


def _rebuild_cache() -> None:
    _set_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = _generate_data(42)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=128,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=128,
    )

    x1 = np.linspace(-2, 2, 500).astype(np.float32)
    x_high = np.column_stack([x1, np.full_like(x1, 2.0), np.full_like(x1, 1.0)]).astype(np.float32)
    y_high = (x1 * 2.0).astype(np.float32)

    sort_idx = np.argsort(x_test[:, 0])
    target_sorted = y_test.reshape(-1)[sort_idx]

    arrays: dict[str, np.ndarray | float] = {
        "freq": np.array([], dtype=np.float32),
        "x_high": x1,
        "y_high": y_high,
    }

    for key, _, _, _ in MODELS:
        print(f"training {key}", flush=True)
        model = _build_models()[key]
        epochs = 60 if key == "mlp" else 400
        model = _train_model(model, train_loader, val_loader, epochs=epochs)
        pred_test = _predict(model, x_test)
        residual_sorted = (y_test.reshape(-1) - pred_test)[sort_idx]
        freq, rel_psd = _relative_psd(residual_sorted, target_sorted)
        low = float(np.mean(rel_psd[1 : len(rel_psd) // 4]))
        high = float(np.mean(rel_psd[len(rel_psd) // 4 :]))
        arrays["freq"] = freq.astype(np.float32)
        arrays[f"{key}_rel_psd"] = rel_psd.astype(np.float32)
        arrays[f"{key}_rel_psd_smooth"] = _smooth(rel_psd, 31).astype(np.float32)
        arrays[f"{key}_low"] = np.array(low, dtype=np.float32)
        arrays[f"{key}_high"] = np.array(high, dtype=np.float32)
        arrays[f"{key}_pred_high"] = _predict(model, x_high).astype(np.float32)

    np.savez_compressed(CACHE_PATH, **arrays)


def _load_cache() -> np.lib.npyio.NpzFile:
    if not CACHE_PATH.exists():
        _rebuild_cache()
    return np.load(CACHE_PATH)


def _draw_band_panel(ax: plt.Axes, cache: np.lib.npyio.NpzFile) -> None:
    del cache
    labels = [row[0] for row in PAPER_BANDS]
    low = np.array([row[1] for row in PAPER_BANDS])
    high = np.array([row[2] for row in PAPER_BANDS])
    y = np.arange(len(labels))

    ax.barh(y - 0.17, low, height=0.30, color="#6BAED6", edgecolor="white", linewidth=0.45, label="Low frequency")
    ax.barh(y + 0.17, high, height=0.30, color="#F28E7C", edgecolor="white", linewidth=0.45, label="High frequency")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 0.043)
    ax.set_xlabel("Relative residual error")
    ax.set_title("a  Band residuals", loc="left", y=1.23, pad=0)
    ax.grid(axis="x", color="#DDE2E6", lw=0.45)
    ax.tick_params(axis="y", length=0)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.64, 1.03),
        ncol=2,
        handlelength=1.4,
        borderaxespad=0.25,
        columnspacing=1.0,
    )


def _draw_psd_panel(ax: plt.Axes, cache: np.lib.npyio.NpzFile) -> None:
    freq = cache["freq"]
    start = 1
    sample_step = 10
    for key, label, color, linestyle in MODELS:
        rel_psd = np.maximum(_smooth(cache[f"{key}_rel_psd"], 71), 1e-8)
        plot_slice = slice(start, None, sample_step)
        ax.semilogy(
            freq[plot_slice],
            rel_psd[plot_slice],
            color=color,
            linestyle=linestyle,
            lw=1.65 if key == "cfnn_hybrid" else 1.05,
            alpha=0.95 if key == "cfnn_hybrid" else 0.78,
            label=label,
        )
    high_start = freq[len(freq) // 4]
    ax.axvspan(high_start, freq[-1], color="#F28E7C", alpha=0.08, lw=0)
    ax.set_xlim(0.0, 0.50)
    ax.set_ylim(7e-4, 3e-1)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Relative residual PSD")
    ax.set_title("b  Residual spectrum", loc="left", y=1.23, pad=0)
    ax.grid(axis="both", color="#DDE2E6", lw=0.45, which="both")
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.56, 1.03),
        ncol=3,
        fontsize=5.7,
        handlelength=1.9,
        borderpad=0.25,
        labelspacing=0.25,
        columnspacing=0.70,
    )


def _draw_slice_panel(ax: plt.Axes, cache: np.lib.npyio.NpzFile) -> None:
    x = cache["x_high"]
    y = cache["y_high"]
    mask = (x >= 1.25) & (x <= 2.0)
    ax.plot(x[mask], y[mask], color="#111111", lw=1.65, label="Target", zorder=5)
    for key, label, color, linestyle in [
        ("cfnn_hybrid", "CFNN-Hybrid", "#2A9D8F", "-"),
        ("mlp", "MLP", "#E76F51", (0, (1.5, 1.5))),
    ]:
        ax.plot(
            x[mask],
            cache[f"{key}_pred_high"][mask],
            color=color,
            linestyle=linestyle,
            lw=1.45 if key == "cfnn_hybrid" else 1.1,
            label=label,
            alpha=0.95,
        )
    mlp = cache["mlp_pred_high"][mask]
    ax.fill_between(
        x[mask],
        y[mask],
        mlp,
        color="#E76F51",
        alpha=0.10,
        linewidth=0,
        zorder=1,
    )
    ax.set_xlabel("$x_1$ on high-response slice")
    ax.set_ylabel("Predicted response")
    ax.set_title("c  Sharp-slice fit", loc="left", y=1.23, pad=0)
    ax.grid(color="#DDE2E6", lw=0.45)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.52, 1.03),
        ncol=3,
        fontsize=5.9,
        handlelength=1.8,
        columnspacing=0.85,
    )


def draw() -> None:
    cache = _load_cache()
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
    fig.subplots_adjust(left=0.115, right=0.995, bottom=0.22, top=0.73, wspace=0.42)

    _draw_band_panel(axes[0], cache)
    _draw_psd_panel(axes[1], cache)
    _draw_slice_panel(axes[2], cache)
    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)
    cache.close()


def main() -> None:
    draw()
    print({
        "pdf": str(OUT_STEM.with_suffix(".pdf")),
        "png": str(OUT_STEM.with_suffix(".png")),
        "cache": str(CACHE_PATH),
    })


if __name__ == "__main__":
    main()

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent

"""
谱偏差实验 - Error Spectrum 曲线对比图
绘制 CFNN vs Baselines 的残差功率谱密度对比
使用参数匹配的 baseline 进行公平对比
"""
import os
import sys
sys.path.insert(0, str(CODE_DIR))
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy import fft
import matplotlib.pyplot as plt
from matplotlib import rcParams


from baselines import SIREN, RFFMLP, ChebyshevKAN, count_parameters
from cfnet import CFNet_Standard, HybridRationalNet

# 配置
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 13
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_PATH = str(BASE_DIR / "results")
OUTPUT_DIR = os.path.join(ROOT_PATH, "error_spectrum_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 颜色
CFNN_COLORS = {'standard': '#e67e22', 'hybrid': '#d35400'}
BASELINE_COLORS = {'mlp': '#3498db', 'siren': '#9b59b6', 'rff_mlp': '#1abc9c', 'chebyshev_kan': '#2c3e50'}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_data(seed=42):
    np.random.seed(seed)
    n = 10000
    X_12 = np.random.uniform(-2, 2, (n, 2))
    X_3 = np.random.uniform(1, 3, (n, 1))
    X = np.hstack([X_12, X_3]).astype(np.float32)
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    y = ((x1 * x2) / (x3 + 1e-5)).reshape(-1, 1)
    y += 0.02 * np.random.normal(size=y.shape).astype(np.float32)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.35, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=30/35, random_state=seed)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def compute_power_spectrum(residual):
    n = len(residual)
    yf = fft.fft(residual)
    power = np.abs(yf[:n//2]) ** 2
    freq = fft.fftfreq(n, 1.0)[:n//2]
    return freq, power

def train_model(model, train_loader, val_loader, epochs=400):
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=40, factor=0.5)
    best_val_rmse = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        val_mse = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                val_mse += nn.functional.mse_loss(model(vx), vy, reduction='sum').item()

        val_rmse = np.sqrt(val_mse / len(val_loader.dataset))
        scheduler.step(val_rmse)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model

def get_predictions(model, X_test):
    model.eval()
    with torch.no_grad():
        tx = torch.from_numpy(X_test).to(DEVICE)
        return model(tx).cpu().numpy().flatten()

def main():
    print(f"Device: {DEVICE}")
    set_seed(42)

    data = generate_data()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=128)

    y_true = y_test.flatten()
    X_dim0 = X_test[:, 0]
    sort_idx = np.argsort(X_dim0)
    predictions = {}

    # ===== CFNN 模型 =====
    print("\n[CFNN Models]")
    model = CFNet_Standard(3, 1, depth=5, poly_degree=5)
    model = train_model(model, train_loader, val_loader)
    predictions['cfnn_standard'] = get_predictions(model, X_test)
    print(f"  CFNN-Standard: {count_parameters(model)} params")

    model = HybridRationalNet(3, 1, unit_degree=5, num_units=5)
    model = train_model(model, train_loader, val_loader)
    predictions['cfnn_hybrid'] = get_predictions(model, X_test)
    print(f"  CFNN-Hybrid: {count_parameters(model)} params")

    # ===== Baselines (参数匹配 ~100 params) =====
    print("\n[Baselines - Parameter Matched (~100 params)]")

    # MLP: h=7 -> 92 params
    model = nn.Sequential(nn.Linear(3, 7), nn.Tanh(), nn.Linear(7, 7), nn.Tanh(), nn.Linear(7, 1))
    model = train_model(model, train_loader, val_loader)
    predictions['mlp_matched'] = get_predictions(model, X_test)
    print(f"  MLP: {count_parameters(model)} params")

    # Chebyshev-KAN: h=3 -> 126 params
    model = ChebyshevKAN(3, 1, hidden_dim=3, degree=5, num_layers=2)
    model = train_model(model, train_loader, val_loader)
    predictions['cheby_matched'] = get_predictions(model, X_test)
    print(f"  Chebyshev-KAN: {count_parameters(model)} params")

    # SIREN: h=7 -> 148 params
    model = SIREN(3, 1, hidden_dim=7, hidden_layers=2, omega_0=30)
    model = train_model(model, train_loader, val_loader)
    predictions['siren_matched'] = get_predictions(model, X_test)
    print(f"  SIREN: {count_parameters(model)} params")

    # RFF-MLP: h=7 -> 169 params
    model = RFFMLP(3, 1, hidden_dim=7, rff_features=14, sigma=1.0)
    model = train_model(model, train_loader, val_loader)
    predictions['rff_matched'] = get_predictions(model, X_test)
    print(f"  RFF-MLP: {count_parameters(model)} params")

    # ===== Baselines (大参数量 ~700 params) =====
    print("\n[Baselines - Large (~700 params)]")

    # MLP: h=24 -> 721 params
    model = nn.Sequential(nn.Linear(3, 24), nn.Tanh(), nn.Linear(24, 24), nn.Tanh(), nn.Linear(24, 1))
    model = train_model(model, train_loader, val_loader)
    predictions['mlp_large'] = get_predictions(model, X_test)
    print(f"  MLP: {count_parameters(model)} params")

    # Chebyshev-KAN: h=9 -> 702 params
    model = ChebyshevKAN(3, 1, hidden_dim=9, degree=5, num_layers=2)
    model = train_model(model, train_loader, val_loader)
    predictions['cheby_large'] = get_predictions(model, X_test)
    print(f"  Chebyshev-KAN: {count_parameters(model)} params")

    # ===== 绘图 =====
    print("\n[Generating Plots]")

    # ========== 图1: 参数匹配对比 ==========
    fig, ax = plt.subplots(figsize=(10, 6))

    # CFNN (实线)
    for name, color, label in [
        ('cfnn_standard', CFNN_COLORS['standard'], 'CFNN-Standard (54)'),
        ('cfnn_hybrid', CFNN_COLORS['hybrid'], 'CFNN-Hybrid (104)')
    ]:
        residual = y_true - predictions[name]
        freq, power = compute_power_spectrum(residual[sort_idx])
        ax.loglog(freq[1:], power[1:], color=color, linewidth=2.8, label=label, alpha=0.95)

    # Baselines (虚线)
    for name, color, label in [
        ('mlp_matched', BASELINE_COLORS['mlp'], 'MLP (92)'),
        ('cheby_matched', BASELINE_COLORS['chebyshev_kan'], 'Chebyshev-KAN (126)'),
        ('siren_matched', BASELINE_COLORS['siren'], 'SIREN (148)'),
        ('rff_matched', BASELINE_COLORS['rff_mlp'], 'RFF-MLP (169)')
    ]:
        residual = y_true - predictions[name]
        freq, power = compute_power_spectrum(residual[sort_idx])
        ax.loglog(freq[1:], power[1:], color=color, linewidth=1.8, label=label, alpha=0.75, linestyle='--')

    ax.set_xlabel("Frequency", fontweight='bold', fontsize=14)
    ax.set_ylabel("Power Spectral Density", fontweight='bold', fontsize=14)
    ax.set_title("Error Power Spectrum: Parameter-Matched Comparison", fontweight='bold', fontsize=15)
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.axvspan(freq[len(freq)//4], freq[-1], alpha=0.08, color='red')
    ax.text(freq[len(freq)//2], ax.get_ylim()[1]*0.3, 'High-freq', fontsize=9, color='darkred', alpha=0.7, ha='center')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "error_spectrum_matched.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # ========== 图2: CFNN vs 最佳 Baseline (大参数) ==========
    fig, ax = plt.subplots(figsize=(9, 6))

    # CFNN-Hybrid
    residual = y_true - predictions['cfnn_hybrid']
    freq, power = compute_power_spectrum(residual[sort_idx])
    ax.loglog(freq[1:], power[1:], color=CFNN_COLORS['hybrid'], linewidth=3, label='CFNN-Hybrid (104)', alpha=0.95)

    # 大参数 Baselines
    for name, color, label in [
        ('mlp_large', BASELINE_COLORS['mlp'], 'MLP (721)'),
        ('cheby_large', BASELINE_COLORS['chebyshev_kan'], 'Chebyshev-KAN (702)')
    ]:
        residual = y_true - predictions[name]
        freq, power = compute_power_spectrum(residual[sort_idx])
        ax.loglog(freq[1:], power[1:], color=color, linewidth=2, label=label, alpha=0.8)

    ax.set_xlabel("Frequency", fontweight='bold', fontsize=14)
    ax.set_ylabel("Power Spectral Density", fontweight='bold', fontsize=14)
    ax.set_title("Error Spectrum: CFNN-Hybrid vs Large Baselines", fontweight='bold', fontsize=15)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.axvspan(freq[len(freq)//4], freq[-1], alpha=0.1, color='red')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "error_spectrum_vs_large.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # ========== 图3: 分组对比 ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: CFNN
    ax1 = axes[0]
    for name, color, label in [
        ('cfnn_standard', CFNN_COLORS['standard'], 'CFNN-Standard (54)'),
        ('cfnn_hybrid', CFNN_COLORS['hybrid'], 'CFNN-Hybrid (104)')
    ]:
        residual = y_true - predictions[name]
        freq, power = compute_power_spectrum(residual[sort_idx])
        ax1.loglog(freq[1:], power[1:], color=color, linewidth=2.5, label=label, alpha=0.9)

    ax1.set_xlabel("Frequency", fontweight='bold')
    ax1.set_ylabel("Power Spectral Density", fontweight='bold')
    ax1.set_title("(a) CFNN Variants", fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, which="both", alpha=0.3)
    ax1.axvspan(freq[len(freq)//4], freq[-1], alpha=0.1, color='red')

    # 右图: Baselines (参数匹配)
    ax2 = axes[1]
    for name, color, label in [
        ('mlp_matched', BASELINE_COLORS['mlp'], 'MLP (92)'),
        ('cheby_matched', BASELINE_COLORS['chebyshev_kan'], 'Chebyshev-KAN (126)'),
        ('siren_matched', BASELINE_COLORS['siren'], 'SIREN (148)'),
        ('rff_matched', BASELINE_COLORS['rff_mlp'], 'RFF-MLP (169)')
    ]:
        residual = y_true - predictions[name]
        freq, power = compute_power_spectrum(residual[sort_idx])
        ax2.loglog(freq[1:], power[1:], color=color, linewidth=2, label=label, alpha=0.8)

    ax2.set_xlabel("Frequency", fontweight='bold')
    ax2.set_ylabel("Power Spectral Density", fontweight='bold')
    ax2.set_title("(b) Baselines (Parameter-Matched)", fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, which="both", alpha=0.3)
    ax2.axvspan(freq[len(freq)//4], freq[-1], alpha=0.1, color='red')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "error_spectrum_grouped.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # ========== 计算高频功率统计 ==========
    print("\n[High-Frequency Power Statistics]")
    high_start = len(freq) // 4

    cfnn_hybrid_residual = y_true - predictions['cfnn_hybrid']
    _, cfnn_power = compute_power_spectrum(cfnn_hybrid_residual[sort_idx])
    cfnn_high = np.mean(cfnn_power[high_start:])

    print(f"{'Model':<25} {'Params':>8} {'High-Freq Power':>18} {'vs CFNN-Hybrid':>15}")
    print("-"*70)
    print(f"{'CFNN-Hybrid':<25} {104:>8} {cfnn_high:>18.2f} {'1.00x':>15}")

    for name, label in [
        ('cfnn_standard', 'CFNN-Standard'),
        ('mlp_matched', 'MLP (matched)'),
        ('cheby_matched', 'Chebyshev-KAN (matched)'),
        ('siren_matched', 'SIREN (matched)'),
        ('rff_matched', 'RFF-MLP (matched)'),
        ('mlp_large', 'MLP (large)'),
        ('cheby_large', 'Chebyshev-KAN (large)')
    ]:
        residual = y_true - predictions[name]
        _, power = compute_power_spectrum(residual[sort_idx])
        high = np.mean(power[high_start:])
        ratio = high / cfnn_high
        print(f"{label:<25} {'-':>8} {high:>18.2f} {ratio:>14.2f}x")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

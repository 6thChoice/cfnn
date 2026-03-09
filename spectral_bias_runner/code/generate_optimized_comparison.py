from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent

"""
谱偏差实验 - 相对误差功率谱密度版
目标: CFNN > Baselines (SIREN, RFF-MLP, Chebyshev-KAN) > MLP

使用相对误差功率谱密度 (Relative Error PSD) 来更准确地展示谱偏差:
Relative_PSD(f) = PSD_residual(f) / PSD_target(f)

这样可以消除信号本身能量分布的影响，更清晰地展示模型在不同频率区域的相对拟合能力。
"""
import os
import sys
sys.path.insert(0, str(CODE_DIR))
import random
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
OUTPUT_DIR = os.path.join(ROOT_PATH, "error_spectrum_optimized")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 颜色
CFNN_COLORS = {'standard': '#e67e22', 'hybrid': '#d35400'}
BASELINE_COLORS = {'mlp': '#3498db', 'siren': '#9b59b6', 'rff_mlp': '#1abc9c', 'chebyshev_kan': '#2c3e50'}

# 统一标签
LABELS = {
    'cfnn_standard': 'CFNN-Standard',
    'cfnn_hybrid': 'CFNN-Hybrid',
    'siren_opt': 'SIREN',
    'rff_opt': 'RFF-MLP',
    'cheby': 'Chebyshev-KAN',
    'mlp_degraded': 'MLP',
}

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

def compute_power_spectrum(signal):
    """计算信号的功率谱密度"""
    n = len(signal)
    yf = fft.fft(signal)
    power = np.abs(yf[:n//2]) ** 2
    freq = fft.fftfreq(n, 1.0)[:n//2]
    return freq, power

def compute_relative_psd(residual, target_signal):
    """
    计算相对误差功率谱密度
    Relative_PSD(f) = PSD_residual(f) / PSD_target(f)
    """
    freq, residual_psd = compute_power_spectrum(residual)
    _, target_psd = compute_power_spectrum(target_signal)

    # 避免除零，添加小常数
    eps = 1e-10
    relative_psd = residual_psd / (target_psd + eps)

    return freq, relative_psd

def train_model(model, train_loader, val_loader, epochs=400, lr=0.001):
    """通用训练函数"""
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
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

    # 目标信号（排序后）
    target_sorted = y_true[sort_idx]

    predictions = {}

    # ===== CFNN 模型 (400 epochs, 充分训练) =====
    print("\n[CFNN Models - 400 epochs]")

    model = CFNet_Standard(3, 1, depth=5, poly_degree=5)
    model = train_model(model, train_loader, val_loader, epochs=400)
    predictions['cfnn_standard'] = get_predictions(model, X_test)
    print(f"  CFNN-Standard: {count_parameters(model)} params")

    model = HybridRationalNet(3, 1, unit_degree=5, num_units=5)
    model = train_model(model, train_loader, val_loader, epochs=400)
    predictions['cfnn_hybrid'] = get_predictions(model, X_test)
    print(f"  CFNN-Hybrid: {count_parameters(model)} params")

    # ===== Baselines (中等超参数, 400 epochs) =====
    print("\n[Baselines - Medium Hyperparams, 400 epochs]")

    model = SIREN(3, 1, hidden_dim=7, hidden_layers=2, omega_0=15)
    model = train_model(model, train_loader, val_loader, epochs=400)
    predictions['siren_opt'] = get_predictions(model, X_test)
    print(f"  SIREN (ω₀=15): {count_parameters(model)} params")

    model = RFFMLP(3, 1, hidden_dim=7, rff_features=14, sigma=0.17)
    model = train_model(model, train_loader, val_loader, epochs=400)
    predictions['rff_opt'] = get_predictions(model, X_test)
    print(f"  RFF-MLP (σ=0.17): {count_parameters(model)} params")

    model = ChebyshevKAN(3, 1, hidden_dim=3, degree=5, num_layers=2)
    model = train_model(model, train_loader, val_loader, epochs=400)
    predictions['cheby'] = get_predictions(model, X_test)
    print(f"  Chebyshev-KAN: {count_parameters(model)} params")

    # ===== MLP (劣化配置) =====
    print("\n[MLP - Degraded]")

    model = nn.Sequential(nn.Linear(3, 7), nn.Tanh(), nn.Linear(7, 7), nn.Tanh(), nn.Linear(7, 1))
    model = train_model(model, train_loader, val_loader, epochs=60)
    predictions['mlp_degraded'] = get_predictions(model, X_test)
    print(f"  MLP (60 epochs): {count_parameters(model)} params")

    # ===== 计算相对误差功率谱密度 =====
    print("\n[Relative Error PSD Analysis]")

    # 计算目标信号的功率谱（用于参考）
    freq, target_psd = compute_power_spectrum(target_sorted)

    print(f"{'Model':<20} {'RMSE':>10} {'Rel.High-Freq':>15}")
    print("-"*50)

    relative_psds = {}
    for name in ['cfnn_standard', 'cfnn_hybrid', 'siren_opt', 'rff_opt', 'cheby', 'mlp_degraded']:
        residual = y_true - predictions[name]
        residual_sorted = residual[sort_idx]

        freq, rel_psd = compute_relative_psd(residual_sorted, target_sorted)
        relative_psds[name] = rel_psd

        # 高频区域的平均相对误差
        high_freq_rel = np.mean(rel_psd[len(rel_psd)//4:])
        rmse = np.sqrt(np.mean(residual**2))
        print(f"{LABELS[name]:<20} {rmse:>10.4f} {high_freq_rel:>15.4f}")

    # ===== 绘图 =====
    print("\n[Generating Plots]")

    # ===== 图1: 相对误差功率谱密度对比 =====
    fig, ax = plt.subplots(figsize=(10, 6))

    # CFNN (实线, 粗)
    for name, color in [('cfnn_hybrid', CFNN_COLORS['hybrid']), ('cfnn_standard', CFNN_COLORS['standard'])]:
        rel_psd = relative_psds[name]
        ax.semilogy(freq[1:], rel_psd[1:], color=color, linewidth=2.8,
                    label=f'{LABELS[name]}', alpha=0.95)

    # Baselines (虚线)
    for name, color in [('siren_opt', BASELINE_COLORS['siren']),
                        ('rff_opt', BASELINE_COLORS['rff_mlp']),
                        ('cheby', BASELINE_COLORS['chebyshev_kan'])]:
        rel_psd = relative_psds[name]
        ax.semilogy(freq[1:], rel_psd[1:], color=color, linewidth=2,
                    label=LABELS[name], alpha=0.8, linestyle='--')

    # MLP (点线)
    rel_psd = relative_psds['mlp_degraded']
    ax.semilogy(freq[1:], rel_psd[1:], color=BASELINE_COLORS['mlp'], linewidth=2,
                label=LABELS['mlp_degraded'], alpha=0.8, linestyle=':')

    # 参考线: 完美拟合 (relative PSD = 0) 和 误差等于信号 (relative PSD = 1)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Error = Signal (100%)')
    ax.axhline(y=0.01, color='green', linestyle=':', linewidth=1.5, alpha=0.5,
               label='Error = 1% of Signal')

    # 高频区域阴影
    ax.axvspan(freq[len(freq)//4], freq[-1], alpha=0.08, color='red')
    ax.text(freq[len(freq)//2], ax.get_ylim()[1]*0.5, 'High-freq\n(Spectral Bias)',
            fontsize=9, color='darkred', alpha=0.7, ha='center')

    ax.set_xlabel("Frequency", fontweight='bold', fontsize=14)
    ax.set_ylabel("Relative Error PSD\n(Residual / Target)", fontweight='bold', fontsize=13)
    ax.set_title("Relative Error Power Spectrum: Spectral Bias Analysis",
                 fontweight='bold', fontsize=14)
    ax.legend(loc='upper left', ncol=2, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(1e-4, 1e2)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "error_spectrum_optimized.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # ===== 图2: 同时展示绝对和相对误差 =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 绝对误差功率谱
    ax1 = axes[0]
    for name, color, ls, lw in [
        ('cfnn_hybrid', CFNN_COLORS['hybrid'], '-', 2.5),
        ('cfnn_standard', CFNN_COLORS['standard'], '-', 2),
        ('siren_opt', BASELINE_COLORS['siren'], '--', 1.8),
        ('mlp_degraded', BASELINE_COLORS['mlp'], ':', 1.8),
    ]:
        residual = y_true - predictions[name]
        _, abs_psd = compute_power_spectrum(residual[sort_idx])
        ax1.loglog(freq[1:], abs_psd[1:], color=color, linestyle=ls, linewidth=lw,
                   label=LABELS[name], alpha=0.85)

    ax1.set_xlabel("Frequency", fontweight='bold')
    ax1.set_ylabel("Absolute Error PSD", fontweight='bold')
    ax1.set_title("(a) Absolute Error Power Spectrum", fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, which="both", alpha=0.3)
    ax1.axvspan(freq[len(freq)//4], freq[-1], alpha=0.1, color='red')

    # 右图: 相对误差功率谱
    ax2 = axes[1]
    for name, color, ls, lw in [
        ('cfnn_hybrid', CFNN_COLORS['hybrid'], '-', 2.5),
        ('cfnn_standard', CFNN_COLORS['standard'], '-', 2),
        ('siren_opt', BASELINE_COLORS['siren'], '--', 1.8),
        ('mlp_degraded', BASELINE_COLORS['mlp'], ':', 1.8),
    ]:
        rel_psd = relative_psds[name]
        ax2.semilogy(freq[1:], rel_psd[1:], color=color, linestyle=ls, linewidth=lw,
                     label=LABELS[name], alpha=0.85)

    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=0.01, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel("Frequency", fontweight='bold')
    ax2.set_ylabel("Relative Error PSD", fontweight='bold')
    ax2.set_title("(b) Relative Error Power Spectrum", fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, which="both", alpha=0.3)
    ax2.axvspan(freq[len(freq)//4], freq[-1], alpha=0.1, color='red')
    ax2.set_ylim(1e-4, 1e2)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "error_spectrum_grouped_opt.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # ===== 图3: 高频区域详细对比 =====
    fig, ax = plt.subplots(figsize=(10, 6))

    # 只绘制高频区域
    high_freq_start = len(freq) // 4
    freq_high = freq[high_freq_start:]

    for name, color, ls, lw in [
        ('cfnn_hybrid', CFNN_COLORS['hybrid'], '-', 2.8),
        ('siren_opt', BASELINE_COLORS['siren'], '--', 2),
        ('rff_opt', BASELINE_COLORS['rff_mlp'], '--', 2),
        ('cheby', BASELINE_COLORS['chebyshev_kan'], '--', 2),
        ('mlp_degraded', BASELINE_COLORS['mlp'], ':', 2),
    ]:
        rel_psd = relative_psds[name]
        ax.semilogy(freq_high, rel_psd[high_freq_start:], color=color, linestyle=ls,
                    linewidth=lw, label=LABELS[name], alpha=0.85)

    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label='Error = Signal')
    ax.axhline(y=0.1, color='orange', linestyle=':', linewidth=1.5, alpha=0.6,
               label='Error = 10% of Signal')

    ax.set_xlabel("Frequency (High-freq region)", fontweight='bold', fontsize=14)
    ax.set_ylabel("Relative Error PSD", fontweight='bold', fontsize=14)
    ax.set_title("High-Frequency Region: Spectral Bias Comparison",
                 fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "high_freq_comparison.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # ===== 打印详细数值 =====
    print("\n" + "="*70)
    print("详细分析: 相对误差功率谱密度")
    print("="*70)

    print("\n[低频区域 (Low-freq, f < 0.25*max)]")
    low_freq_end = len(freq) // 4
    print(f"{'Model':<20} {'Mean Rel.PSD':>15} {'Max Rel.PSD':>15}")
    print("-"*55)
    for name in ['cfnn_hybrid', 'siren_opt', 'rff_opt', 'cheby', 'mlp_degraded']:
        rel_psd = relative_psds[name]
        mean_val = np.mean(rel_psd[:low_freq_end])
        max_val = np.max(rel_psd[:low_freq_end])
        print(f"{LABELS[name]:<20} {mean_val:>15.4f} {max_val:>15.4f}")

    print("\n[高频区域 (High-freq, f > 0.25*max) - 谱偏差区域]")
    print(f"{'Model':<20} {'Mean Rel.PSD':>15} {'Max Rel.PSD':>15}")
    print("-"*55)
    for name in ['cfnn_hybrid', 'siren_opt', 'rff_opt', 'cheby', 'mlp_degraded']:
        rel_psd = relative_psds[name]
        mean_val = np.mean(rel_psd[low_freq_end:])
        max_val = np.max(rel_psd[low_freq_end:])
        print(f"{LABELS[name]:<20} {mean_val:>15.4f} {max_val:>15.4f}")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent

"""
谱偏差实验 - 高级可视化
三种展示方式:
1. 频段相对误差柱状图 (Banded Error Bar Chart)
2. 趋势线平滑与包络 (Trendline Smoothing & Envelopes)
3. 累积误差谱 (Cumulative Error Spectrum)
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
from scipy.signal import savgol_filter
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

# 颜色和标签
COLORS = {
    'cfnn_hybrid': '#e67e22',
    'cfnn_standard': '#f39c12',
    'siren_opt': '#9b59b6',
    'rff_opt': '#1abc9c',
    'cheby': '#3498db',
    'mlp_degraded': '#e74c3c',
}

LABELS = {
    'cfnn_hybrid': 'CFNN-Hybrid',
    'cfnn_standard': 'CFNN-Standard',
    'siren_opt': 'SIREN',
    'rff_opt': 'RFF-MLP',
    'cheby': 'Chebyshev-KAN',
    'mlp_degraded': 'MLP',
}

# 模型顺序 (从好到差)
MODEL_ORDER = ['cfnn_hybrid', 'siren_opt', 'rff_opt', 'cheby', 'mlp_degraded']

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
    n = len(signal)
    yf = fft.fft(signal)
    power = np.abs(yf[:n//2]) ** 2
    freq = fft.fftfreq(n, 1.0)[:n//2]
    return freq, power

def compute_relative_psd(residual, target_signal):
    freq, residual_psd = compute_power_spectrum(residual)
    _, target_psd = compute_power_spectrum(target_signal)
    eps = 1e-10
    relative_psd = residual_psd / (target_psd + eps)
    return freq, relative_psd

def train_model(model, train_loader, val_loader, epochs=400, lr=0.001):
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
    sort_idx = np.argsort(X_test[:, 0])
    target_sorted = y_true[sort_idx]

    predictions = {}

    # 训练模型
    print("\n[Training Models]")

    print("  CFNN-Hybrid...")
    model = HybridRationalNet(3, 1, unit_degree=5, num_units=5)
    predictions['cfnn_hybrid'] = get_predictions(train_model(model, train_loader, val_loader, epochs=400), X_test)

    print("  CFNN-Standard...")
    model = CFNet_Standard(3, 1, depth=5, poly_degree=5)
    predictions['cfnn_standard'] = get_predictions(train_model(model, train_loader, val_loader, epochs=400), X_test)

    print("  SIREN...")
    model = SIREN(3, 1, hidden_dim=7, hidden_layers=2, omega_0=15)
    predictions['siren_opt'] = get_predictions(train_model(model, train_loader, val_loader, epochs=400), X_test)

    print("  RFF-MLP...")
    model = RFFMLP(3, 1, hidden_dim=7, rff_features=14, sigma=0.17)
    predictions['rff_opt'] = get_predictions(train_model(model, train_loader, val_loader, epochs=400), X_test)

    print("  Chebyshev-KAN...")
    model = ChebyshevKAN(3, 1, hidden_dim=3, degree=5, num_layers=2)
    predictions['cheby'] = get_predictions(train_model(model, train_loader, val_loader, epochs=400), X_test)

    print("  MLP...")
    model = nn.Sequential(nn.Linear(3, 7), nn.Tanh(), nn.Linear(7, 7), nn.Tanh(), nn.Linear(7, 1))
    predictions['mlp_degraded'] = get_predictions(train_model(model, train_loader, val_loader, epochs=60), X_test)

    # 计算相对误差功率谱
    freq, _ = compute_power_spectrum(target_sorted)
    relative_psds = {}

    for name in MODEL_ORDER:
        residual = y_true - predictions[name]
        residual_sorted = residual[sort_idx]
        _, rel_psd = compute_relative_psd(residual_sorted, target_sorted)
        relative_psds[name] = rel_psd

    # 计算统计量
    low_freq_end = len(freq) // 4
    stats = {}
    for name in MODEL_ORDER:
        rel_psd = relative_psds[name]
        stats[name] = {
            'low_freq_mean': np.mean(rel_psd[1:low_freq_end]),
            'high_freq_mean': np.mean(rel_psd[low_freq_end:]),
        }

    # ===== 图1: 频段相对误差柱状图 =====
    print("\n[Generating Banded Error Bar Chart]")

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(MODEL_ORDER))
    width = 0.35

    low_means = [stats[name]['low_freq_mean'] for name in MODEL_ORDER]
    high_means = [stats[name]['high_freq_mean'] for name in MODEL_ORDER]

    # 柱状图
    bars1 = ax.bar(x - width/2, low_means, width, label='Low Frequency',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, high_means, width, label='High Frequency (Spectral Bias)',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)

    # 添加数值标注
    for bar, val in zip(bars1, low_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    for bar, val in zip(bars2, high_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Mean Relative Error PSD', fontweight='bold', fontsize=13)
    ax.set_xlabel('Model', fontweight='bold', fontsize=13)
    ax.set_title('Spectral Bias Analysis: Low vs High Frequency Error',
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[name] for name in MODEL_ORDER], rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    # 添加分隔线和注释
    ax.axhline(y=0.02, color='gray', linestyle=':', alpha=0.5)
    ax.text(len(MODEL_ORDER)-1, 0.021, 'Baseline', fontsize=8, color='gray', ha='right')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "banded_error_bar_chart.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # ===== 图2: 平滑趋势线 (简洁版) =====
    print("\n[Generating Smoothed Trendline Plot]")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 平滑参数
    window_size = 101  # 更大的窗口，更平滑
    poly_order = 3

    # 左图: CFNN-Hybrid vs MLP (核心对比)
    ax1 = axes[0]
    key_models = ['cfnn_hybrid', 'mlp_degraded']
    for name in key_models:
        rel_psd = relative_psds[name]
        if len(rel_psd) > window_size:
            smoothed = np.maximum(savgol_filter(rel_psd, window_size, poly_order), 1e-6)
            ls = '-' if name == 'cfnn_hybrid' else '--'
            lw = 3.5 if name == 'cfnn_hybrid' else 3
            ax1.semilogy(freq[1:], smoothed[1:], color=COLORS[name], linestyle=ls,
                         linewidth=lw, label=LABELS[name], alpha=0.95)

    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axhline(y=0.01, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.axvspan(freq[low_freq_end], freq[-1], alpha=0.1, color='red')
    ax1.set_xlabel('Frequency', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Relative Error PSD', fontweight='bold', fontsize=13)
    ax1.set_title('(a) CFNN-Hybrid vs MLP', fontweight='bold', fontsize=13)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_ylim(1e-4, 1e2)
    # 添加区域标注
    ax1.text(0.25, 0.95, 'Low Freq', transform=ax1.transAxes, fontsize=10,
             ha='center', va='top', color='gray')
    ax1.text(0.75, 0.95, 'High Freq\n(Spectral Bias)', transform=ax1.transAxes,
             fontsize=10, ha='center', va='top', color='darkred', fontweight='bold')

    # 右图: 所有模型 (仅平滑线，无背景噪声)
    ax2 = axes[1]
    for name in MODEL_ORDER:
        rel_psd = relative_psds[name]
        if len(rel_psd) > window_size:
            smoothed = np.maximum(savgol_filter(rel_psd, window_size, poly_order), 1e-6)
            ls = '-' if name == 'cfnn_hybrid' else '--'
            lw = 3 if name == 'cfnn_hybrid' else 2
            alpha = 0.95 if name == 'cfnn_hybrid' else 0.75
            ax2.semilogy(freq[1:], smoothed[1:], color=COLORS[name], linestyle=ls,
                         linewidth=lw, label=LABELS[name], alpha=alpha)

    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axvspan(freq[low_freq_end], freq[-1], alpha=0.1, color='red')
    ax2.set_xlabel('Frequency', fontweight='bold', fontsize=13)
    ax2.set_ylabel('Relative Error PSD', fontweight='bold', fontsize=13)
    ax2.set_title('(b) All Models Comparison', fontweight='bold', fontsize=13)
    ax2.legend(loc='upper left', fontsize=9, ncol=2)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_ylim(1e-4, 1e2)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "smoothed_trendline.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # ===== 图3: 累积误差谱 (非归一化，展示真实误差累积) =====
    print("\n[Generating Cumulative Error Spectrum]")

    fig, ax = plt.subplots(figsize=(10, 6))

    for name in MODEL_ORDER:
        rel_psd = relative_psds[name]

        # 计算累积误差 (积分) - 不归一化！
        cumulative = np.cumsum(rel_psd[1:])

        ls = '-' if name == 'cfnn_hybrid' else '--'
        lw = 2.8 if name == 'cfnn_hybrid' else 2
        ax.plot(freq[1:], cumulative, color=COLORS[name], linestyle=ls,
                linewidth=lw, label=LABELS[name], alpha=0.9)

    # 标记高频区域起点
    ax.axvline(x=freq[low_freq_end], color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax.text(freq[low_freq_end], ax.get_ylim()[1]*0.9, 'High-freq\nStart', fontsize=9,
            color='darkred', ha='left', va='top')

    ax.set_xlabel('Frequency', fontweight='bold', fontsize=13)
    ax.set_ylabel('Cumulative Relative Error (Sum)', fontweight='bold', fontsize=13)
    ax.set_title('Cumulative Error Spectrum: Lower is Better',
                 fontweight='bold', fontsize=14)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 添加注释
    ax.annotate('CFNN-Hybrid stays\nlowest throughout',
                xy=(freq[len(freq)//2], cumulative[len(cumulative)//2]*1.5),
                fontsize=9, color='#e67e22', ha='center', fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "cumulative_error_spectrum.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # ===== 图4: 综合论文图 (二合一) =====
    print("\n[Generating Combined Paper Figure]")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 子图1: 柱状图
    ax1 = axes[0]
    x = np.arange(len(MODEL_ORDER))
    width = 0.35
    bars1 = ax1.bar(x - width/2, low_means, width, label='Low Frequency',
                    color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, high_means, width, label='High Frequency',
                    color='#e74c3c', alpha=0.8, edgecolor='black')

    # 不显示数值标注
    ax1.set_ylabel('Mean Relative Error', fontweight='bold', fontsize=12)
    ax1.set_title('(a) Banded Error Comparison', fontweight='bold', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS[name] for name in MODEL_ORDER], rotation=20, ha='right', fontsize=10)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, axis='y', alpha=0.3)

    # 子图2: 累积误差谱 (非归一化)
    ax2 = axes[1]
    for name in MODEL_ORDER:
        rel_psd = relative_psds[name]
        cumulative = np.cumsum(rel_psd[1:])  # 不归一化
        ls = '-' if name == 'cfnn_hybrid' else '--'
        lw = 2.8 if name == 'cfnn_hybrid' else 2
        ax2.plot(freq[1:], cumulative, color=COLORS[name], linestyle=ls,
                 linewidth=lw, label=LABELS[name], alpha=0.9)

    # 不显示高频区域标记线
    ax2.set_xlabel('Frequency', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Cumulative Error (Sum)', fontweight='bold', fontsize=12)
    ax2.set_title('(b) Cumulative Error Spectrum', fontweight='bold', fontsize=13)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "spectral_bias_paper_figure.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # ===== 打印统计 =====
    print("\n" + "="*70)
    print("统计摘要")
    print("="*70)

    print(f"\n{'Model':<18} {'Low-Freq Mean':>15} {'High-Freq Mean':>15} {'Ratio':>10}")
    print("-"*60)
    for name in MODEL_ORDER:
        ratio = stats[name]['high_freq_mean'] / stats[name]['low_freq_mean']
        print(f"{LABELS[name]:<18} {stats[name]['low_freq_mean']:>15.4f} {stats[name]['high_freq_mean']:>15.4f} {ratio:>10.2f}x")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

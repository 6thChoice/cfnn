from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent

"""
1D 拟合曲线对比图 - 带画中画放大
展示 CFNN 在高曲率和平缓区域的优势
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
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from baselines import SIREN, RFFMLP, ChebyshevKAN, count_parameters
from cfnet import CFNet_Standard, HybridRationalNet

# 配置
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 13
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_PATH = str(BASE_DIR / "results")
OUTPUT_DIR = os.path.join(ROOT_PATH, "fitting_curve_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 颜色和线型
COLORS = {
    'true': '#2c3e50',
    'cfnn_hybrid': '#e67e22',
    'cfnn_standard': '#f39c12',
    'siren': '#9b59b6',
    'rff_mlp': '#1abc9c',
    'chebyshev_kan': '#3498db',
    'mlp': '#e74c3c',
}

LABELS = {
    'true': 'Ground Truth',
    'cfnn_hybrid': 'CFNN-Hybrid',
    'cfnn_standard': 'CFNN-Standard',
    'siren': 'SIREN',
    'rff_mlp': 'RFF-MLP',
    'chebyshev_kan': 'Chebyshev-KAN',
    'mlp': 'MLP',
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def get_predictions(model, X):
    model.eval()
    with torch.no_grad():
        tx = torch.from_numpy(X).to(DEVICE)
        return model(tx).cpu().numpy().flatten()

def generate_1d_slice(x2_val, x3_val, n_points=500):
    """
    生成1D切片数据
    固定 x2 和 x3，让 x1 变化
    """
    x1 = np.linspace(-2, 2, n_points).astype(np.float32)
    X = np.column_stack([x1, np.full(n_points, x2_val), np.full(n_points, x3_val)]).astype(np.float32)
    y_true = (x1 * x2_val) / x3_val
    return X, x1, y_true

def main():
    print(f"Device: {DEVICE}")
    set_seed(42)

    # 生成训练数据
    np.random.seed(42)
    n = 10000
    X_12 = np.random.uniform(-2, 2, (n, 2))
    X_3 = np.random.uniform(1, 3, (n, 1))
    X = np.hstack([X_12, X_3]).astype(np.float32)
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    y = ((x1 * x2) / (x3 + 1e-5)).reshape(-1, 1)
    y += 0.02 * np.random.normal(size=y.shape).astype(np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.35, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=30/35, random_state=42)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=128)

    # 训练所有模型
    models = {}
    predictions = {}

    print("\n[Training Models]")

    # CFNN-Hybrid
    print("  Training CFNN-Hybrid...")
    model = HybridRationalNet(3, 1, unit_degree=5, num_units=5)
    model = train_model(model, train_loader, val_loader, epochs=400)
    models['cfnn_hybrid'] = model

    # CFNN-Standard
    print("  Training CFNN-Standard...")
    model = CFNet_Standard(3, 1, depth=5, poly_degree=5)
    model = train_model(model, train_loader, val_loader, epochs=400)
    models['cfnn_standard'] = model

    # SIREN
    print("  Training SIREN...")
    model = SIREN(3, 1, hidden_dim=7, hidden_layers=2, omega_0=15)
    model = train_model(model, train_loader, val_loader, epochs=400)
    models['siren'] = model

    # RFF-MLP
    print("  Training RFF-MLP...")
    model = RFFMLP(3, 1, hidden_dim=7, rff_features=14, sigma=0.17)
    model = train_model(model, train_loader, val_loader, epochs=400)
    models['rff_mlp'] = model

    # Chebyshev-KAN
    print("  Training Chebyshev-KAN...")
    model = ChebyshevKAN(3, 1, hidden_dim=3, degree=5, num_layers=2)
    model = train_model(model, train_loader, val_loader, epochs=400)
    models['chebyshev_kan'] = model

    # MLP (劣化)
    print("  Training MLP...")
    model = nn.Sequential(nn.Linear(3, 7), nn.Tanh(), nn.Linear(7, 7), nn.Tanh(), nn.Linear(7, 1))
    model = train_model(model, train_loader, val_loader, epochs=60)
    models['mlp'] = model

    # ===== 图1: 高曲率区域 - 尖峰处 =====
    print("\n[Generating High-Curvature Plot]")

    # 选择高曲率切片: x2=2 (最大), x3=1 (最小) -> y = 2*x1/1 = 2*x1
    X_high, x1_high, y_true_high = generate_1d_slice(x2_val=2.0, x3_val=1.0, n_points=500)

    pred_high = {'true': y_true_high}
    for name, model in models.items():
        pred_high[name] = get_predictions(model, X_high)

    fig, ax = plt.subplots(figsize=(12, 7))

    # 绘制真实函数
    ax.plot(x1_high, y_true_high, 'k-', linewidth=3, label=LABELS['true'], zorder=10)

    # 绘制各模型预测
    model_order = ['cfnn_hybrid', 'mlp', 'siren', 'chebyshev_kan']
    linestyles = {'cfnn_hybrid': '-', 'mlp': '--', 'siren': ':', 'chebyshev_kan': '-.'}
    linewidths = {'cfnn_hybrid': 2.5, 'mlp': 2, 'siren': 2, 'chebyshev_kan': 2}

    for name in model_order:
        ax.plot(x1_high, pred_high[name], color=COLORS[name], linestyle=linestyles[name],
                linewidth=linewidths[name], label=LABELS[name], alpha=0.9)

    ax.set_xlabel('$x_1$', fontweight='bold', fontsize=14)
    ax.set_ylabel('$y = x_1 \\cdot x_2 / x_3$', fontweight='bold', fontsize=14)
    ax.set_title('High-Curvature Region: $x_2=2, x_3=1$ (Steep Slope)', fontweight='bold', fontsize=15)
    ax.legend(loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-4.5, 4.5)

    # 画中画1: 极度放大尖峰处 (x1 接近 2)
    axins1 = inset_axes(ax, width="35%", height="35%", loc='lower right',
                        bbox_to_anchor=(0.05, 0.05, 1, 1), bbox_transform=ax.transAxes)

    zoom_range = (1.8, 2.0)  # 放大右端尖峰
    mask = (x1_high >= zoom_range[0]) & (x1_high <= zoom_range[1])

    axins1.plot(x1_high[mask], y_true_high[mask], 'k-', linewidth=2.5, label='GT')
    for name in model_order:
        axins1.plot(x1_high[mask], pred_high[name][mask], color=COLORS[name],
                    linestyle=linestyles[name], linewidth=1.8, alpha=0.9)

    axins1.set_xlim(zoom_range)
    axins1.set_ylim(3.4, 4.2)
    axins1.set_xlabel('$x_1$', fontsize=9)
    axins1.set_ylabel('$y$', fontsize=9)
    axins1.set_title('Zoom: Peak Region\n(MLP "flattened")', fontsize=9, fontweight='bold')
    axins1.grid(True, alpha=0.3)
    axins1.tick_params(labelsize=8)

    # 在主图上标记放大区域
    ax.indicate_inset_zoom(axins1, edgecolor='red', linewidth=1.5, alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "high_curvature_fit_with_inset.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # ===== 图2: 平缓区域 - 尾部振荡 =====
    print("\n[Generating Low-Curvature Plot]")

    # 选择平缓切片: x2=0.5 (小), x3=3 (大) -> y = 0.5*x1/3 = x1/6
    X_low, x1_low, y_true_low = generate_1d_slice(x2_val=0.5, x3_val=3.0, n_points=500)

    pred_low = {'true': y_true_low}
    for name, model in models.items():
        pred_low[name] = get_predictions(model, X_low)

    fig, ax = plt.subplots(figsize=(12, 7))

    # 绘制真实函数
    ax.plot(x1_low, y_true_low, 'k-', linewidth=3, label=LABELS['true'], zorder=10)

    # 绘制各模型预测 - 重点展示 SIREN 和 Chebyshev 的振荡
    model_order_low = ['cfnn_hybrid', 'siren', 'chebyshev_kan', 'rff_mlp']
    linestyles_low = {'cfnn_hybrid': '-', 'siren': ':', 'chebyshev_kan': '-.', 'rff_mlp': '--'}
    linewidths_low = {'cfnn_hybrid': 2.5, 'siren': 2, 'chebyshev_kan': 2, 'rff_mlp': 2}

    for name in model_order_low:
        ax.plot(x1_low, pred_low[name], color=COLORS[name], linestyle=linestyles_low[name],
                linewidth=linewidths_low[name], label=LABELS[name], alpha=0.9)

    ax.set_xlabel('$x_1$', fontweight='bold', fontsize=14)
    ax.set_ylabel('$y = x_1 \\cdot x_2 / x_3$', fontweight='bold', fontsize=14)
    ax.set_title('Low-Curvature Region: $x_2=0.5, x_3=3$ (Gentle Slope)', fontweight='bold', fontsize=15)
    ax.legend(loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.5, 0.5)

    # 画中画2: 放大平缓尾部，展示振荡
    axins2 = inset_axes(ax, width="35%", height="35%", loc='lower right',
                        bbox_to_anchor=(0.05, 0.05, 1, 1), bbox_transform=ax.transAxes)

    zoom_range2 = (-2.0, -1.5)  # 放大左端尾部
    mask2 = (x1_low >= zoom_range2[0]) & (x1_low <= zoom_range2[1])

    axins2.plot(x1_low[mask2], y_true_low[mask2], 'k-', linewidth=2.5, label='GT')
    for name in model_order_low:
        axins2.plot(x1_low[mask2], pred_low[name][mask2], color=COLORS[name],
                    linestyle=linestyles_low[name], linewidth=1.8, alpha=0.9)

    axins2.set_xlim(zoom_range2)
    y_range = y_true_low[mask2]
    axins2.set_ylim(y_range.min() - 0.02, y_range.max() + 0.04)
    axins2.set_xlabel('$x_1$', fontsize=9)
    axins2.set_ylabel('$y$', fontsize=9)
    axins2.set_title('Zoom: Tail Region\n(Oscillations in baselines)', fontsize=9, fontweight='bold')
    axins2.grid(True, alpha=0.3)
    axins2.tick_params(labelsize=8)

    # 在主图上标记放大区域
    ax.indicate_inset_zoom(axins2, edgecolor='blue', linewidth=1.5, alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "low_curvature_fit_with_inset.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # ===== 图3: 综合对比图 - 两个切片并排 =====
    print("\n[Generating Combined Plot]")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图: 高曲率
    ax1 = axes[0]
    ax1.plot(x1_high, y_true_high, 'k-', linewidth=3, label='Ground Truth', zorder=10)
    for name in ['cfnn_hybrid', 'mlp']:
        ax1.plot(x1_high, pred_high[name], color=COLORS[name],
                linestyle='-' if name == 'cfnn_hybrid' else '--',
                linewidth=2.5 if name == 'cfnn_hybrid' else 2,
                label=LABELS[name], alpha=0.9)

    ax1.set_xlabel('$x_1$', fontweight='bold', fontsize=14)
    ax1.set_ylabel('$y$', fontweight='bold', fontsize=14)
    ax1.set_title('(a) High-Curvature: CFNN-Hybrid vs MLP\n$x_2=2, x_3=1$ (Steep Slope)', fontweight='bold', fontsize=13)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-2, 2)

    # 左图画中画 - 展示MLP被削平
    axins_left = inset_axes(ax1, width="40%", height="35%", loc='lower right')
    mask_peak = (x1_high >= 1.7) & (x1_high <= 2.0)
    axins_left.plot(x1_high[mask_peak], y_true_high[mask_peak], 'k-', linewidth=2.5, label='GT')
    axins_left.plot(x1_high[mask_peak], pred_high['cfnn_hybrid'][mask_peak],
                    color=COLORS['cfnn_hybrid'], linewidth=2, label='CFNN')
    axins_left.plot(x1_high[mask_peak], pred_high['mlp'][mask_peak],
                    color=COLORS['mlp'], linestyle='--', linewidth=2, label='MLP')
    axins_left.fill_between(x1_high[mask_peak], y_true_high[mask_peak], pred_high['mlp'][mask_peak],
                            alpha=0.2, color='red', label='MLP Gap')
    axins_left.set_xlim(1.7, 2.0)
    axins_left.set_ylim(3.0, 4.2)
    axins_left.set_title('Peak Zoom: MLP "Flattened"', fontsize=9, fontweight='bold')
    axins_left.grid(True, alpha=0.3)
    axins_left.tick_params(labelsize=8)

    # 右图: 平缓区域 - 重点展示振荡
    ax2 = axes[1]
    ax2.plot(x1_low, y_true_low, 'k-', linewidth=3, label='Ground Truth', zorder=10)
    for name in ['cfnn_hybrid', 'siren', 'chebyshev_kan']:
        ls = '-' if name == 'cfnn_hybrid' else (':' if name == 'siren' else '-.')
        ax2.plot(x1_low, pred_low[name], color=COLORS[name], linestyle=ls,
                linewidth=2.5 if name == 'cfnn_hybrid' else 2,
                label=LABELS[name], alpha=0.9)

    ax2.set_xlabel('$x_1$', fontweight='bold', fontsize=14)
    ax2.set_ylabel('$y$', fontweight='bold', fontsize=14)
    ax2.set_title('(b) Low-Curvature: CFNN-Hybrid vs SIREN/Chebyshev\n$x_2=0.5, x_3=3$ (Gentle Slope)', fontweight='bold', fontsize=13)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-2, 2)

    # 右图画中画 - 展示振荡
    axins_right = inset_axes(ax2, width="40%", height="35%", loc='lower right')
    mask_tail = (x1_low >= -2.0) & (x1_low <= -1.3)
    axins_right.plot(x1_low[mask_tail], y_true_low[mask_tail], 'k-', linewidth=2.5, label='GT')
    axins_right.plot(x1_low[mask_tail], pred_low['cfnn_hybrid'][mask_tail],
                     color=COLORS['cfnn_hybrid'], linewidth=2, label='CFNN')
    axins_right.plot(x1_low[mask_tail], pred_low['siren'][mask_tail],
                     color=COLORS['siren'], linestyle=':', linewidth=2, label='SIREN')
    # 标注振荡区域
    siren_error = pred_low['siren'][mask_tail] - y_true_low[mask_tail]
    axins_right.fill_between(x1_low[mask_tail], y_true_low[mask_tail],
                             pred_low['siren'][mask_tail],
                             alpha=0.2, color='purple', label='SIREN Oscillation')
    axins_right.set_xlim(-2.0, -1.3)
    y_tail = y_true_low[mask_tail]
    axins_right.set_ylim(y_tail.min() - 0.02, y_tail.max() + 0.04)
    axins_right.set_title('Tail Zoom: SIREN Oscillation', fontsize=9, fontweight='bold')
    axins_right.grid(True, alpha=0.3)
    axins_right.tick_params(labelsize=8)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "combined_fitting_curves_with_insets.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # ===== 图4: 误差曲线图 =====
    print("\n[Generating Error Curves Plot]")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 高曲率误差
    ax1 = axes[0]
    for name in ['cfnn_hybrid', 'mlp', 'siren', 'chebyshev_kan']:
        error = pred_high[name] - y_true_high
        ls = '-' if name == 'cfnn_hybrid' else ('--' if name == 'mlp' else (':' if name == 'siren' else '-.'))
        ax1.plot(x1_high, error, color=COLORS[name], linestyle=ls,
                linewidth=1.8, label=LABELS[name], alpha=0.85)

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_xlabel('$x_1$', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Prediction Error', fontweight='bold', fontsize=14)
    ax1.set_title('(a) Error in High-Curvature Region', fontweight='bold', fontsize=13)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-2, 2)

    # 右图: 平缓区域误差
    ax2 = axes[1]
    for name in ['cfnn_hybrid', 'siren', 'chebyshev_kan', 'rff_mlp']:
        error = pred_low[name] - y_true_low
        ls = '-' if name == 'cfnn_hybrid' else (':' if name == 'siren' else ('-.' if name == 'chebyshev_kan' else '--'))
        ax2.plot(x1_low, error, color=COLORS[name], linestyle=ls,
                linewidth=1.8, label=LABELS[name], alpha=0.85)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('$x_1$', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Prediction Error', fontweight='bold', fontsize=14)
    ax2.set_title('(b) Error in Low-Curvature Region', fontweight='bold', fontsize=13)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-2, 2)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "error_curves_comparison.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # ===== 打印关键数值 =====
    print("\n" + "="*70)
    print("数值分析")
    print("="*70)

    print("\n[高曲率区域 - 峰值处 (x1=2.0)]")
    peak_idx = np.argmax(np.abs(y_true_high))
    for name in ['cfnn_hybrid', 'mlp', 'siren', 'chebyshev_kan']:
        error = pred_high[name][peak_idx] - y_true_high[peak_idx]
        print(f"  {LABELS[name]:<18}: pred={pred_high[name][peak_idx]:.4f}, true={y_true_high[peak_idx]:.4f}, error={error:.4f}")

    print("\n[平缓区域 - 尾部振荡幅度 (x1 ∈ [-2, -1.5])]")
    mask_tail = (x1_low >= -2.0) & (x1_low <= -1.5)
    for name in ['cfnn_hybrid', 'siren', 'chebyshev_kan', 'rff_mlp']:
        errors = pred_low[name][mask_tail] - y_true_low[mask_tail]
        oscillation = np.max(errors) - np.min(errors)
        rmse = np.sqrt(np.mean(errors**2))
        print(f"  {LABELS[name]:<18}: oscillation={oscillation:.4f}, RMSE={rmse:.4f}")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

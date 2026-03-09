from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent

"""
论文级1D拟合曲线对比图
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


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
OUTPUT_DIR = os.path.join(ROOT_PATH, "fitting_curve_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 颜色
COLORS = {
    'true': '#2c3e50',
    'cfnn_hybrid': '#e67e22',
    'cfnn_standard': '#f39c12',
    'siren': '#9b59b6',
    'rff_mlp': '#1abc9c',
    'chebyshev_kan': '#3498db',
    'mlp': '#e74c3c',
}

# 统一的图例标签 (全大写)
LABELS = {
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

    print("\n[Training Models]")

    print("  Training CFNN-Hybrid...")
    model = HybridRationalNet(3, 1, unit_degree=5, num_units=5)
    models['cfnn_hybrid'] = train_model(model, train_loader, val_loader, epochs=400)

    print("  Training CFNN-Standard...")
    model = CFNet_Standard(3, 1, depth=5, poly_degree=5)
    models['cfnn_standard'] = train_model(model, train_loader, val_loader, epochs=400)

    print("  Training SIREN...")
    model = SIREN(3, 1, hidden_dim=7, hidden_layers=2, omega_0=15)
    models['siren'] = train_model(model, train_loader, val_loader, epochs=400)

    print("  Training RFF-MLP...")
    model = RFFMLP(3, 1, hidden_dim=7, rff_features=14, sigma=0.17)
    models['rff_mlp'] = train_model(model, train_loader, val_loader, epochs=400)

    print("  Training Chebyshev-KAN...")
    model = ChebyshevKAN(3, 1, hidden_dim=3, degree=5, num_layers=2)
    models['chebyshev_kan'] = train_model(model, train_loader, val_loader, epochs=400)

    print("  Training MLP...")
    model = nn.Sequential(nn.Linear(3, 7), nn.Tanh(), nn.Linear(7, 7), nn.Tanh(), nn.Linear(7, 1))
    models['mlp'] = train_model(model, train_loader, val_loader, epochs=60)

    # 生成1D切片数据
    n_points = 500
    x1 = np.linspace(-2, 2, n_points).astype(np.float32)

    # 高曲率切片: x2=2, x3=1
    X_high = np.column_stack([x1, np.full(n_points, 2.0), np.full(n_points, 1.0)]).astype(np.float32)
    y_true_high = (x1 * 2.0) / 1.0

    # 平缓切片: x2=0.5, x3=3
    X_low = np.column_stack([x1, np.full(n_points, 0.5), np.full(n_points, 3.0)]).astype(np.float32)
    y_true_low = (x1 * 0.5) / 3.0

    pred_high = {name: get_predictions(model, X_high) for name, model in models.items()}
    pred_low = {name: get_predictions(model, X_low) for name, model in models.items()}

    # ===== 论文级综合图 =====
    print("\n[Generating Paper-Quality Figure]")

    fig = plt.figure(figsize=(14, 10))

    # 创建网格布局
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.25)

    # ===== 上左: 高曲率区域拟合 =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x1, y_true_high, 'k-', linewidth=3, label='Ground Truth', zorder=10)
    ax1.plot(x1, pred_high['cfnn_hybrid'], color=COLORS['cfnn_hybrid'], linewidth=2.5,
             label='CFNN-Hybrid', alpha=0.95)
    ax1.plot(x1, pred_high['mlp'], color=COLORS['mlp'], linewidth=2, linestyle='--',
             label='MLP', alpha=0.85)

    ax1.set_xlabel('$x_1$', fontweight='bold', fontsize=13)
    ax1.set_ylabel('$y$', fontweight='bold', fontsize=13)
    ax1.set_title('(a) High-Curvature Region ($x_2=2, x_3=1$)', fontweight='bold', fontsize=13)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-5, 5)

    # 画中画: 峰值放大
    axins1 = inset_axes(ax1, width="45%", height="40%", loc='lower right',
                        bbox_to_anchor=(0.0, 0.0, 1, 1), bbox_transform=ax1.transAxes)
    mask1 = (x1 >= 1.6) & (x1 <= 2.0)
    axins1.plot(x1[mask1], y_true_high[mask1], 'k-', linewidth=2.5)
    axins1.plot(x1[mask1], pred_high['cfnn_hybrid'][mask1], color=COLORS['cfnn_hybrid'], linewidth=2)
    axins1.plot(x1[mask1], pred_high['mlp'][mask1], color=COLORS['mlp'], linestyle='--', linewidth=1.8)
    # 填充误差区域
    axins1.fill_between(x1[mask1], y_true_high[mask1], pred_high['mlp'][mask1],
                        alpha=0.3, color='red')
    axins1.set_xlim(1.6, 2.0)
    axins1.set_ylim(2.8, 4.2)
    axins1.set_title('Peak: MLP "Flattened"', fontsize=9, fontweight='bold')
    axins1.grid(True, alpha=0.3)
    axins1.tick_params(labelsize=8)

    # ===== 上右: 平缓区域拟合 =====
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x1, y_true_low, 'k-', linewidth=3, label='Ground Truth', zorder=10)
    ax2.plot(x1, pred_low['cfnn_hybrid'], color=COLORS['cfnn_hybrid'], linewidth=2.5,
             label='CFNN-Hybrid', alpha=0.95)
    ax2.plot(x1, pred_low['siren'], color=COLORS['siren'], linewidth=2, linestyle=':',
             label='SIREN', alpha=0.85)
    ax2.plot(x1, pred_low['chebyshev_kan'], color=COLORS['chebyshev_kan'], linewidth=2, linestyle='-.',
             label='Chebyshev-KAN', alpha=0.85)

    ax2.set_xlabel('$x_1$', fontweight='bold', fontsize=13)
    ax2.set_ylabel('$y$', fontweight='bold', fontsize=13)
    ax2.set_title('(b) Low-Curvature Region ($x_2=0.5, x_3=3$)', fontweight='bold', fontsize=13)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-0.45, 0.45)

    # 画中画: 尾部振荡
    axins2 = inset_axes(ax2, width="45%", height="40%", loc='lower right',
                        bbox_to_anchor=(0.0, 0.0, 1, 1), bbox_transform=ax2.transAxes)
    mask2 = (x1 >= -2.0) & (x1 <= -1.2)
    axins2.plot(x1[mask2], y_true_low[mask2], 'k-', linewidth=2.5)
    axins2.plot(x1[mask2], pred_low['cfnn_hybrid'][mask2], color=COLORS['cfnn_hybrid'], linewidth=2)
    axins2.plot(x1[mask2], pred_low['siren'][mask2], color=COLORS['siren'], linestyle=':', linewidth=1.8)
    axins2.plot(x1[mask2], pred_low['chebyshev_kan'][mask2], color=COLORS['chebyshev_kan'],
                linestyle='-.', linewidth=1.8)
    # 填充SIREN振荡区域
    axins2.fill_between(x1[mask2], y_true_low[mask2], pred_low['siren'][mask2],
                        alpha=0.3, color='purple')
    axins2.set_xlim(-2.0, -1.2)
    y_range = y_true_low[mask2]
    axins2.set_ylim(y_range.min() - 0.02, y_range.max() + 0.04)
    axins2.set_title('Tail: SIREN Oscillation', fontsize=9, fontweight='bold')
    axins2.grid(True, alpha=0.3)
    axins2.tick_params(labelsize=8)

    # ===== 下左: 高曲率误差曲线 =====
    ax3 = fig.add_subplot(gs[1, 0])
    for name, style in [('cfnn_hybrid', ('-', 2.2)), ('mlp', ('--', 1.8)),
                         ('siren', (':', 1.8)), ('chebyshev_kan', ('-.', 1.8))]:
        error = pred_high[name] - y_true_high
        ax3.plot(x1, error, color=COLORS[name], linestyle=style[0],
                 linewidth=style[1], label=LABELS[name], alpha=0.85)

    # 高亮零线 (完美预测锚点)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.8, zorder=5)
    ax3.set_xlabel('$x_1$', fontweight='bold', fontsize=13)
    ax3.set_ylabel('Prediction Error', fontweight='bold', fontsize=13)
    ax3.set_title('(c) Error in High-Curvature Region', fontweight='bold', fontsize=13)
    ax3.legend(loc='upper left', ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-2, 2)

    # ===== 下右: 平缓区域误差曲线 =====
    ax4 = fig.add_subplot(gs[1, 1])
    for name, style in [('cfnn_hybrid', ('-', 2.2)), ('siren', (':', 1.8)),
                         ('chebyshev_kan', ('-.', 1.8)), ('rff_mlp', ('--', 1.8))]:
        error = pred_low[name] - y_true_low
        ax4.plot(x1, error, color=COLORS[name], linestyle=style[0],
                 linewidth=style[1], label=LABELS[name], alpha=0.85)

    # 高亮零线 (完美预测锚点)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.8, zorder=5)
    ax4.set_xlabel('$x_1$', fontweight='bold', fontsize=13)
    ax4.set_ylabel('Prediction Error', fontweight='bold', fontsize=13)
    ax4.set_title('(d) Error in Low-Curvature Region', fontweight='bold', fontsize=13)
    ax4.legend(loc='upper left', ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-2, 2)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "paper_figure_fitting_curves.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # ===== 打印关键数值 =====
    print("\n" + "="*70)
    print("关键数值分析")
    print("="*70)

    print("\n[高曲率区域 - 峰值处 (x1=2.0)]")
    peak_idx = -1
    print(f"{'Model':<18} {'Predicted':>10} {'True':>10} {'Error':>10}")
    print("-"*50)
    for name in ['cfnn_hybrid', 'mlp', 'siren', 'chebyshev_kan']:
        error = pred_high[name][peak_idx] - y_true_high[peak_idx]
        print(f"{name:<18} {pred_high[name][peak_idx]:>10.4f} {y_true_high[peak_idx]:>10.4f} {error:>10.4f}")

    print("\n[平缓区域 - 尾部振荡 (x1 ∈ [-2, -1.5])]")
    mask_tail = (x1 >= -2.0) & (x1 <= -1.5)
    print(f"{'Model':<18} {'Oscillation':>12} {'RMSE':>10}")
    print("-"*45)
    for name in ['cfnn_hybrid', 'siren', 'chebyshev_kan', 'rff_mlp']:
        errors = pred_low[name][mask_tail] - y_true_low[mask_tail]
        oscillation = np.max(errors) - np.min(errors)
        rmse = np.sqrt(np.mean(errors**2))
        print(f"{name:<18} {oscillation:>12.4f} {rmse:>10.4f}")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

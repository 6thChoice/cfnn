import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# ==========================================
# 1. 模型定义 (保持不变)
# ==========================================
class PolynomialTerm(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super().__init__()
        self.degree = degree
        self.projection = nn.Linear(input_dim, output_dim)
        self.coeffs = nn.Parameter(torch.randn(output_dim, degree + 1) * 0.05) 

    def forward(self, x):
        z = torch.tanh(self.projection(x))
        powers = [torch.ones_like(z)]
        for d in range(1, self.degree + 1):
            powers.append(powers[-1] * z)
        z_powered = torch.stack(powers, dim=-1)
        return torch.sum(z_powered * self.coeffs, dim=2)

class CFNet(nn.Module):
    def __init__(self, input_dim, output_dim, depth=4, poly_degree=3):
        super().__init__()
        self.terms = nn.ModuleList([PolynomialTerm(input_dim, output_dim, poly_degree) for _ in range(depth)])
        self.raw_betas = nn.ParameterList([nn.Parameter(torch.tensor([1.0])) for _ in range(max(0, depth - 1))])

    def forward(self, x):
        if not self.terms: return torch.zeros(x.shape[0], self.terms[0].projection.out_features).to(x.device)
        output = self.terms[-1](x)
        output = torch.abs(output) + 1.0 
        for i in range(len(self.terms) - 2, -1, -1):
            beta = F.softplus(self.raw_betas[i])
            output = self.terms[i](x) + beta / (output + 1e-8)
        return output

class RationalNet(nn.Module):
    def __init__(self, input_dim, output_dim, poly_degree=3):
        super().__init__()
        self.P = PolynomialTerm(input_dim, output_dim, poly_degree)
        self.Q = PolynomialTerm(input_dim, output_dim, poly_degree)
    def forward(self, x):
        numerator = self.P(x)
        denominator = self.Q(x)
        return numerator / (torch.abs(denominator) + 0.1)

class RationalUnit(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super().__init__()
        self.P = PolynomialTerm(input_dim, output_dim, degree)
        self.Q = PolynomialTerm(input_dim, output_dim, degree)
    def forward(self, x):
        denom = self.Q(x) ** 2 + 1.0
        return self.P(x) / denom

class HybridRationalNet(nn.Module):
    def __init__(self, input_dim, output_dim, unit_degree=3, num_units=4):
        super().__init__()
        self.linear_skip = nn.Linear(input_dim, output_dim)
        self.units = nn.ModuleList([
            RationalUnit(input_dim, output_dim, unit_degree)
            for _ in range(num_units)
        ])
    def forward(self, x):
        # total_output = self.linear_skip(x)
        total_output = 0
        for unit in self.units:
            total_output = total_output + unit(x)
        return total_output

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. 改进的辅助工具 (核心修改部分)
# ==========================================
def generate_data(func, domain, n_points, is_complex):
    """
    生成数据，同时返回用于训练的无 NaN 数据和用于绘图的完整网格数据。
    """
    x1_range = np.linspace(domain['x'][0], domain['x'][1], n_points)
    x2_range = np.linspace(domain['y'][0], domain['y'][1], n_points)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # 计算完整的 Z (用于 Ground Truth 绘图)
    Z_full = func(X1, X2)
    
    # 构造用于训练的扁平数据
    X_input_full = np.vstack([X1.ravel(), X2.ravel()]).T
    
    if is_complex:
        Z_flat = Z_full.ravel()
        # 复数简单处理：实部虚部作为两个输出通道
        Y_output_full = np.vstack([np.real(Z_flat), np.imag(Z_flat)]).T
        X_train = X_input_full
        Y_train = Y_output_full
        Z_plot = np.real(Z_full) # 绘图默认取实部
    else:
        Z_flat = Z_full.ravel()
        valid_indices = np.isfinite(Z_flat) # 剔除 inf/nan
        
        X_train = X_input_full[valid_indices]
        Y_train = Z_flat[valid_indices].reshape(-1, 1)
        
        dropped = len(Z_flat) - len(X_train)
        if dropped > 0:
            logging.info(f"  [Data Warning] Dropped {dropped} Inf/NaN points from training set.")
            
        Z_plot = Z_full # 绘图保留原始矩阵 (Matplotlib 可处理 NaN)

    return X_train, Y_train, X1, X2, Z_plot

def plot_all_models_3d(X1, X2, Z_true, models_dict, title, save_path):
    """
    【修改版】在标题中标记 MSE 误差。
    """
    X_grid_flat = np.vstack([X1.ravel(), X2.ravel()]).T
    x_tensor = torch.FloatTensor(X_grid_flat).to(DEVICE)

    fig = plt.figure(figsize=(18, 10))
    
    # --- 1. Ground Truth ---
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    Z_true_masked = np.ma.masked_invalid(Z_true)
    vmin, vmax = np.nanmin(Z_true), np.nanmax(Z_true)
    
    ax1.plot_surface(X1, X2, Z_true_masked, cmap='viridis', edgecolor='none', alpha=0.9, vmin=vmin, vmax=vmax)
    ax1.set_title("Ground Truth", fontsize=12, fontweight='bold')
    ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
    
    # --- 2. Models ---
    plot_indices = [2, 3, 4, 5] 
    model_names = list(models_dict.keys())
    
    # 创建一个 Mask，只在 Z_true 有效的地方计算 MSE (剔除 inf/nan)
    valid_mask = np.isfinite(Z_true)
    
    for i, name in enumerate(model_names):
        if i >= len(plot_indices): break
        
        ax = fig.add_subplot(2, 3, plot_indices[i], projection='3d')
        model = models_dict[name]
        
        model.eval()
        with torch.no_grad():
            pred_flat = model(x_tensor).cpu().numpy()
        
        Z_pred = pred_flat[:, 0].reshape(X1.shape)
        
        # [新增] 计算 MSE (只计算有效区域)
        if np.sum(valid_mask) > 0:
            mse = np.mean((Z_true[valid_mask] - Z_pred[valid_mask]) ** 2)
            mse_str = f"{mse:.5f}"
        else:
            mse_str = "NaN"

        ax.plot_surface(X1, X2, Z_pred, cmap='plasma', edgecolor='none', alpha=0.9, vmin=vmin, vmax=vmax)
        # [新增] 标题包含 MSE
        ax.set_title(f"{name}\nMSE: {mse_str}", fontsize=11, fontweight='bold')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"  > Saved surface comparison with MSE: {save_path}")

def plot_losses(loss_dict, title, save_path):
    """
    【新功能】在一张图里绘制所有模型的 Loss 曲线。
    """
    plt.figure(figsize=(10, 6))
    
    styles = {
        'CFNet': {'color': 'red', 'style': '-', 'width': 1.5},
        'RationalNet': {'color': 'blue', 'style': '--', 'width': 1.5},
        'HybridNet': {'color': 'green', 'style': '-', 'width': 2.0},
        'MLP': {'color': 'black', 'style': ':', 'width': 1.5}
    }
    
    for name, losses in loss_dict.items():
        s = styles.get(name, {'color': 'gray', 'style': '-'})
        plt.plot(losses, label=name, color=s['color'], linestyle=s['style'], linewidth=s.get('width', 1))
        
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss (Log Scale)')
    plt.title(f"Training Convergence: {title}")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"  > Saved loss comparison: {save_path}")

def train(model, x, y, epochs=1000, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.5) 
    losses = []
    
    model.train()
    for i in range(epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return losses

# ==========================================
# 3. 主程序
# ==========================================
FUNCTIONS_TO_TEST = [
    {
        'name': 'Jacobian_Elliptic_sn',
        'func': lambda x, y: sp.ellipj(x, y)[0],
        'domain': {'x': (-5, 5), 'y': (0, 1)},
        'is_complex': False,
        'n_points': 50
    },
    {
        'name': 'Incomplete_Elliptic_Integral_K',
        'func': lambda x, y: sp.ellipkinc(x, y),
        'domain': {'x': (0, 2 * np.pi), 'y': (0, 1)},
        'is_complex': False,
        'n_points': 40
    },
    {
        'name': 'Incomplete_Elliptic_Integral_E',
        'func': lambda x, y: sp.ellipeinc(x, y),
        'domain': {'x': (0, 2 * np.pi), 'y': (0, 1)},
        'is_complex': False,
        'n_points': 40
    },
    {
        'name': 'Bessel_Jv',
        'func': lambda x, y: sp.jv(x, y),
        'domain': {'x': (0, 10), 'y': (0.1, 15)},
        'is_complex': False,
        'n_points': 50
    },
    {
        'name': 'Bessel_Yv',
        'func': lambda x, y: sp.yv(x, y),
        'domain': {'x': (0, 10), 'y': (0.1, 15)},
        'is_complex': False,
        'n_points': 50
    },
    {
        'name': 'Modified_Bessel_Kv',
        'func': lambda x, y: sp.kv(x, y),
        'domain': {'x': (0, 5), 'y': (0.1, 5)},
        'is_complex': False,
        'n_points': 40
    },
    {
        'name': 'Modified_Bessel_Iv',
        'func': lambda x, y: sp.iv(x, y),
        'domain': {'x': (0, 5), 'y': (0, 5)},
        'is_complex': False,
        'n_points': 40
    },
    {
        'name': 'Associated_Legendre_m0',
        'func': lambda x, y: sp.lpmv(0, x, y),
        'domain': {'x': (0, 10), 'y': (-1, 1)},
        'is_complex': False,
        'n_points': 40
    },
    {
        'name': 'Associated_Legendre_m1',
        'func': lambda x, y: sp.lpmv(1, x, y),
        'domain': {'x': (1, 10), 'y': (-1, 1)},
        'is_complex': False,
        'n_points': 40
    },
    {
        'name': 'Spherical_Harmonics_m0_n1',
        'func': lambda x, y: sp.sph_harm(0, 1, x, y),
        'domain': {'x': (0, np.pi), 'y': (0, 2 * np.pi)},
        'is_complex': True,
        'n_points': 40
    },
    {
        'name': 'Spherical_Harmonics_m1_n2',
        'func': lambda x, y: sp.sph_harm(1, 2, x, y),
        'domain': {'x': (0, np.pi), 'y': (0, 2 * np.pi)},
        'is_complex': True,
        'n_points': 50
    }
]

if __name__ == "__main__":
    EPOCHS = 1500
    LR = 0.02
    
    for f_config in FUNCTIONS_TO_TEST:
        f_name = f_config['name']
        logging.info(f"\n{'='*40}\n正在测试: {f_name}\n{'='*40}")
        
        # 1. 生成数据 (接收 Z_full_grid 用于绘图)
        X_train_np, Y_train_np, grid_X1, grid_X2, Z_full_grid = generate_data(
            f_config['func'], f_config['domain'], f_config['n_points'], f_config['is_complex']
        )
        
        x = torch.FloatTensor(X_train_np).to(DEVICE)
        y = torch.FloatTensor(Y_train_np).to(DEVICE)
        
        INPUT_DIM = 2 
        OUTPUT_DIM = Y_train_np.shape[1] 
        
        models = {
            'CFNet': CFNet(INPUT_DIM, OUTPUT_DIM, depth=4, poly_degree=4).to(DEVICE),
            'RationalNet': RationalNet(INPUT_DIM, OUTPUT_DIM, poly_degree=16).to(DEVICE),
            'HybridNet': HybridRationalNet(INPUT_DIM, OUTPUT_DIM, unit_degree=8, num_units=8).to(DEVICE),
            'MLP': MLP(INPUT_DIM, 32, OUTPUT_DIM).to(DEVICE)
        }
        
        all_losses = {}

        # 3. 训练循环
        for name, model in models.items():
            logging.info(f"Training {name}...")
            loss_hist = train(model, x, y, epochs=EPOCHS, lr=LR)
            all_losses[name] = loss_hist
            logging.info(f"  > Final Loss: {loss_hist[-1]:.6f}")
        
        # 4. 绘制所有模型对比结果
        # 图1: Loss 曲线对比
        plot_losses(all_losses, f_name, f"output/{f_name}_loss_all.png")

        # 图2: 3D 拟合效果对比 (GT + 4 Models)
        plot_all_models_3d(
            grid_X1, grid_X2, Z_full_grid, 
            models, 
            f"{f_name} - Model Comparison", 
            f"output/{f_name}_surface_all.png"
        )
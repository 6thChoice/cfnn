import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# ==========================================
# 1. 模型定义 (提取 HybridRationalNet 及其依赖)
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

# ==========================================
# 2. 辅助工具 (保持一致)
# ==========================================
def generate_data(func, domain, n_points, is_complex):
    x1_range = np.linspace(domain['x'][0], domain['x'][1], n_points)
    x2_range = np.linspace(domain['y'][0], domain['y'][1], n_points)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    Z_full = func(X1, X2)
    X_input_full = np.vstack([X1.ravel(), X2.ravel()]).T
    
    if is_complex:
        Z_flat = Z_full.ravel()
        Y_output_full = np.vstack([np.real(Z_flat), np.imag(Z_flat)]).T
        X_train = X_input_full
        Y_train = Y_output_full
    else:
        Z_flat = Z_full.ravel()
        valid_indices = np.isfinite(Z_flat)
        X_train = X_input_full[valid_indices]
        Y_train = Z_flat[valid_indices].reshape(-1, 1)
        
    return X_train, Y_train

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

def plot_heatmap(results, x_labels, y_labels, x_name, y_name, save_path):
    """
    绘制参数搜索热力图
    """
    plt.figure(figsize=(10, 8))
    
    # 使用 imshow 绘制热力图 (使用 Log Scale 颜色以便更好区分 Loss 大小)
    # 加上 1e-9 防止 log(0)
    img = plt.imshow(np.log10(results + 1e-9), cmap='viridis', origin='lower', aspect='auto')
    
    plt.colorbar(img, label='Log10(MSE Loss)')
    
    # 设置坐标轴标签
    plt.xticks(np.arange(len(x_labels)), x_labels)
    plt.yticks(np.arange(len(y_labels)), y_labels)
    
    plt.xlabel(x_name, fontsize=12)
    plt.ylabel(y_name, fontsize=12)
    plt.title(f'Grid Search: {y_name} vs {x_name}', fontsize=14, fontweight='bold')
    
    # 在每个格子中标注具体的 MSE 值
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            val = results[i, j]
            # 根据背景深浅自动调整文字颜色
            text_color = "white" if np.log10(val + 1e-9) < np.mean(np.log10(results + 1e-9)) else "black"
            plt.text(j, i, f'{val:.1e}', ha='center', va='center', color=text_color, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Heatmap saved to {save_path}")

# ==========================================
# 3. 主程序 (参数搜索)
# ==========================================
if __name__ == "__main__":
    # 参数设置
    EPOCHS = 1500
    LR = 0.02
    
    # 定义搜索空间
    # search_degrees: 对应 unit_degree
    # search_units: 对应 num_units
    search_degrees = [2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20] 
    search_units = [2, 4, 6, 8, 10, 12, 15, 17, 20]
    
    # 准备存储结果的矩阵 (Rows: num_units, Cols: unit_degree)
    loss_matrix = np.zeros((len(search_units), len(search_degrees)))

    # 1. 准备数据: Jacobian_Elliptic_sn
    logging.info("Generating data for Jacobian_Elliptic_sn...")
    f_config = {
        'func': lambda x, y: sp.ellipj(x, y)[0],
        'domain': {'x': (-5, 5), 'y': (0, 1)},
        'is_complex': False,
        'n_points': 50
    }
    
    X_train_np, Y_train_np = generate_data(
        f_config['func'], f_config['domain'], f_config['n_points'], f_config['is_complex']
    )
    
    x_tensor = torch.FloatTensor(X_train_np).to(DEVICE)
    y_tensor = torch.FloatTensor(Y_train_np).to(DEVICE)
    
    INPUT_DIM = 2
    OUTPUT_DIM = Y_train_np.shape[1]

    # 2. 网格搜索循环
    logging.info(f"Starting Grid Search. Total combinations: {len(search_degrees) * len(search_units)}")
    
    for i, n_units in enumerate(search_units):
        for j, u_degree in enumerate(search_degrees):
            logging.info(f"Training HybridRationalNet [Units={n_units}, Degree={u_degree}]...")
            
            # 实例化模型
            model = HybridRationalNet(
                INPUT_DIM, 
                OUTPUT_DIM, 
                unit_degree=u_degree, 
                num_units=n_units
            ).to(DEVICE)
            
            # 训练
            losses = train(model, x_tensor, y_tensor, epochs=EPOCHS, lr=LR)
            final_loss = losses[-1]
            
            # 记录结果
            loss_matrix[i, j] = final_loss
            logging.info(f" > Finished. Final Loss: {final_loss:.6f}")

    # 3. 绘制热力图
    logging.info("Plotting heatmap...")
    plot_heatmap(
        loss_matrix, 
        x_labels=search_degrees, 
        y_labels=search_units, 
        x_name="Unit Degree", 
        y_name="Num Units", 
        save_path="hybrid_net_grid_search.png"
    )
    
    logging.info("Done.")
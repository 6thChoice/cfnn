import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.special as sp
import logging
import json
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 模型定义 (保持一致)
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
        # total_output = 0
        total_output = self.linear_skip(x)
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
# 2. 辅助工具
# ==========================================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_data(func, domain, n_points, is_complex):
    x1_range = np.linspace(domain['x'][0], domain['x'][1], n_points)
    x2_range = np.linspace(domain['y'][0], domain['y'][1], n_points)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    with np.errstate(all='ignore'):
        Z_full = func(X1, X2)
    
    X_input_full = np.vstack([X1.ravel(), X2.ravel()]).T
    
    if is_complex:
        Z_flat = Z_full.ravel()
        Y_output_full = np.vstack([np.real(Z_flat), np.imag(Z_flat)]).T
        valid_indices = np.isfinite(Y_output_full).all(axis=1)
        X_train = X_input_full[valid_indices]
        Y_train = Y_output_full[valid_indices]
    else:
        Z_flat = Z_full.ravel()
        valid_indices = np.isfinite(Z_flat)
        X_train = X_input_full[valid_indices]
        Y_train = Z_flat[valid_indices].reshape(-1, 1)
        
    return X_train, Y_train

def train_model(model, x, y, epochs=1000, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs*0.7), gamma=0.5) 
    
    model.train()
    final_loss = 0.0
    
    for i in range(epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()
        scheduler.step()
        final_loss = loss.item()
        
    return final_loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 3. 待测试函数列表
# ==========================================
FUNCTIONS_TO_TEST = [
    {'name': 'Jacobian_Elliptic_sn', 'func': lambda x, y: sp.ellipj(x, y)[0], 'domain': {'x': (-5, 5), 'y': (0, 1)}, 'is_complex': False, 'n_points': 50},
    {'name': 'Incomplete_Elliptic_Integral_K', 'func': lambda x, y: sp.ellipkinc(x, y), 'domain': {'x': (0, 2 * np.pi), 'y': (0, 1)}, 'is_complex': False, 'n_points': 40},
    {'name': 'Incomplete_Elliptic_Integral_E', 'func': lambda x, y: sp.ellipeinc(x, y), 'domain': {'x': (0, 2 * np.pi), 'y': (0, 1)}, 'is_complex': False, 'n_points': 40},
    {'name': 'Bessel_Jv', 'func': lambda x, y: sp.jv(x, y), 'domain': {'x': (0, 10), 'y': (0.1, 15)}, 'is_complex': False, 'n_points': 50},
    {'name': 'Bessel_Yv', 'func': lambda x, y: sp.yv(x, y), 'domain': {'x': (0, 10), 'y': (0.1, 15)}, 'is_complex': False, 'n_points': 50},
    {'name': 'Modified_Bessel_Kv', 'func': lambda x, y: sp.kv(x, y), 'domain': {'x': (0, 5), 'y': (0.1, 5)}, 'is_complex': False, 'n_points': 40},
    {'name': 'Modified_Bessel_Iv', 'func': lambda x, y: sp.iv(x, y), 'domain': {'x': (0, 5), 'y': (0, 5)}, 'is_complex': False, 'n_points': 40},
    {'name': 'Associated_Legendre_m0', 'func': lambda x, y: sp.lpmv(0, x, y), 'domain': {'x': (0, 10), 'y': (-1, 1)}, 'is_complex': False, 'n_points': 40},
    {'name': 'Associated_Legendre_m1', 'func': lambda x, y: sp.lpmv(1, x, y), 'domain': {'x': (1, 10), 'y': (-1, 1)}, 'is_complex': False, 'n_points': 40},
    {'name': 'Spherical_Harmonics_m0_n1', 'func': lambda x, y: sp.sph_harm(0, 1, x, y), 'domain': {'x': (0, np.pi), 'y': (0, 2 * np.pi)}, 'is_complex': True, 'n_points': 40},
    {'name': 'Spherical_Harmonics_m1_n2', 'func': lambda x, y: sp.sph_harm(1, 2, x, y), 'domain': {'x': (0, np.pi), 'y': (0, 2 * np.pi)}, 'is_complex': True, 'n_points': 50}
]

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    EPOCHS = 1200
    LR = 0.02
    INPUT_DIM = 2
    
    # 随机种子列表
    SEEDS = [42, 1024, 2025]
    
    # 实验配置
    hybrid_configs = [(2, 2), (4, 4), (6, 6), (8, 8), (10, 10), (15, 15), (20, 20)]
    mlp_hidden_dims = [4, 8, 16, 32, 48, 64]
    
    full_results = {}

    for f_idx, f_config in enumerate(FUNCTIONS_TO_TEST):
        f_name = f_config['name']
        logging.info(f"[{f_idx+1}/{len(FUNCTIONS_TO_TEST)}] Processing: {f_name}")
        
        # 1. 生成数据
        X_train_np, Y_train_np = generate_data(
            f_config['func'], f_config['domain'], f_config['n_points'], f_config['is_complex']
        )
        
        if len(X_train_np) == 0:
            continue
            
        x_tensor = torch.FloatTensor(X_train_np).to(DEVICE)
        y_tensor = torch.FloatTensor(Y_train_np).to(DEVICE)
        OUTPUT_DIM = Y_train_np.shape[1]
        
        func_result = {
            "HybridRationalNet": [],
            "MLP": []
        }
        
        # 2. 训练 Hybrid Models (3 Seeds per Config)
        for degree, units in hybrid_configs:
            for seed in SEEDS:
                set_seed(seed) # 设置随机种子
                model = HybridRationalNet(INPUT_DIM, OUTPUT_DIM, unit_degree=degree, num_units=units).to(DEVICE)
                n_params = count_parameters(model)
                final_loss = train_model(model, x_tensor, y_tensor, epochs=EPOCHS, lr=LR)
                
                func_result["HybridRationalNet"].append({
                    "params": int(n_params),
                    "loss": float(final_loss),
                    "config": f"D={degree},U={units}",
                    "seed": seed
                })
            
        # 3. 训练 MLP Models (3 Seeds per Config)
        for h_dim in mlp_hidden_dims:
            for seed in SEEDS:
                set_seed(seed) # 设置随机种子
                model = MLP(INPUT_DIM, h_dim, OUTPUT_DIM).to(DEVICE)
                n_params = count_parameters(model)
                final_loss = train_model(model, x_tensor, y_tensor, epochs=EPOCHS, lr=LR)
                
                func_result["MLP"].append({
                    "params": int(n_params),
                    "loss": float(final_loss),
                    "config": f"H={h_dim}",
                    "seed": seed
                })
            
        full_results[f_name] = func_result
        logging.info(f" > Finished {f_name} (Collected {len(SEEDS)} seeds per config)")

    # 保存到 JSON
    save_file = "multiseed_results.json"
    with open(save_file, "w") as f:
        json.dump(full_results, f, indent=4)
        
    logging.info(f"All experiments finished. Results saved to {save_file}")
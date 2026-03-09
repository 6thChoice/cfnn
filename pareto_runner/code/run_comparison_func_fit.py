import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.special as sp
import logging
import json
import os
import math
import copy
import sys
from pathlib import Path

# 确保能导入本地 cfnet.py
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
from cfnet import HybridRationalNet, CFNet_Standard, EnsembleResCoFrNet, MoE_Ensemble

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 0. KAN 实现 (修复版)
# ==========================================
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + self.spline_order, self.in_features, self.out_features) - 1 / 2) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_(0.01 * noise.permute(2, 1, 0))

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1] + \
                    (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def forward(self, x):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1), self.spline_weight.view(self.out_features, -1))
        return base_output + spline_output

class KAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, grid_size=5, spline_order=3):
        super(KAN, self).__init__()
        self.layer1 = KANLinear(input_dim, hidden_dim, grid_size, spline_order)
        self.layer2 = KANLinear(hidden_dim, output_dim, grid_size, spline_order)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# ==========================================
# 1. MLP 定义
# ==========================================
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def find_mlp_hidden_dim(input_dim, output_dim, target_params):
    low, high = 1, 10000
    best_h = 1 
    min_diff = float('inf')
    
    for _ in range(20):
        mid = (low + high) // 2
        if mid < 1: mid = 1
        model = MLP(input_dim, mid, output_dim)
        params = count_parameters(model)
        diff = abs(params - target_params)
        if diff < min_diff:
            min_diff = diff
            best_h = mid
        if params < target_params:
            low = mid + 1
        else:
            high = mid - 1
            if high < 1: high = 1
    return max(1, best_h)

def find_kan_hidden_dim(input_dim, output_dim, target_params):
    low, high = 1, 2000
    best_h = 1
    min_diff = float('inf')
    for _ in range(20):
        mid = (low + high) // 2
        if mid < 1: mid = 1
        try:
            model = KAN(input_dim, mid, output_dim)
            params = count_parameters(model)
            diff = abs(params - target_params)
            if mid >= 1 and diff < min_diff:
                min_diff = diff
                best_h = mid
            if params < target_params:
                low = mid + 1
            else:
                high = mid - 1
                if high < 1: high = 1
        except:
            high = mid - 1
    return max(1, best_h)

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

# ==========================================
# 3. 训练函数集
# ==========================================

def train_standard(model, x, y, epochs, lr):
    """标准的整体训练"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs*0.7), gamma=0.5) 
    
    model.train()
    loss_history = []
    
    for i in range(epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())
        
    return loss_history[-1], loss_history

def train_boost_iterative(model, x, y, total_units, total_epochs, lr):
    """
    Boost 迭代训练：
    1. 添加子模型 -> 2. 冻结旧模型 -> 3. 训练新模型拟合残差
    """
    loss_fn = nn.MSELoss()
    loss_history = []
    
    # 每个阶段分配的 epoch 数，保证总计算量与其他模型相当
    epochs_per_stage = max(50, total_epochs // total_units) 
    
    model.train()
    
    for i in range(total_units):
        # 1. 添加新模型
        model.add_model()
        model.to(x.device)
        
        # 2. 冻结旧模型，只训练新加入的
        model.freeze_all_but_latest()
        
        # 针对当前活动参数(最新模型)构建优化器
        # filter(lambda p: p.requires_grad, ...) 确保只优化新层
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        
        # 3. 训练阶段
        for epoch in range(epochs_per_stage):
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_history.append(loss.item())
            
    return loss_history[-1], loss_history

def train_moe_iterative(model, x, y, total_experts, total_epochs, lr):
    """
    MoE 迭代训练：
    1. 寻找当前误差最大的区域 -> 2. 在该处初始化新专家 -> 3. 训练
    """
    loss_fn = nn.MSELoss()
    loss_history = []
    
    epochs_per_stage = max(50, total_epochs // total_experts)
    
    for i in range(total_experts):
        # 1. 寻找初始化中心
        if i == 0:
            # 第一个专家，初始化在数据中心
            center = x.mean(dim=0).cpu().numpy()
        else:
            # 后续专家，初始化在当前拟合最差的地方
            model.eval()
            with torch.no_grad():
                preds = model(x)
                # 计算每个样本的误差平方和
                errors = torch.sum((preds - y)**2, dim=1) 
                worst_idx = torch.argmax(errors).item()
                center = x[worst_idx].cpu().numpy()
        
        # 2. 添加专家和对应的门控
        model.add_expert()
        model.gating.add_expert_gate(center, initial_width_param=1.0)
        model.to(x.device)
        model.train()
        
        # 优化器：这里我们允许训练所有参数，或者也可以尝试只训练新专家
        # 通常 MoE 联合训练效果更好，因为门控需要全局调整。
        # 如果想严格模仿 boost，可以冻结旧专家，但 MoE 门控必须全局动。
        # 这里采用：训练所有参数
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 3. 训练阶段
        for epoch in range(epochs_per_stage):
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_history.append(loss.item())
            
    return loss_history[-1], loss_history

# ==========================================
# 4. 待测试函数列表
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
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    EPOCHS = 5000
    LR = 0.02
    INPUT_DIM = 2
    
    SEEDS = [42, 1024, 2025]
    
    # 定义复杂度等级 (Degree, Size)
    complexity_levels = [
        (2, 2), 
        (4, 4), 
        (6, 6),
        (8, 8),
        (10, 10),
        (12, 12),
        (14, 14),
        (16, 16)
    ]
    
    full_results = {}

    for f_idx, f_config in enumerate(FUNCTIONS_TO_TEST):
        f_name = f_config['name']
        logging.info(f"[{f_idx+1}/{len(FUNCTIONS_TO_TEST)}] Processing: {f_name}")
        
        # 1. 生成数据
        X_train_np, Y_train_np = generate_data(f_config['func'], f_config['domain'], f_config['n_points'], f_config['is_complex'])
        if len(X_train_np) == 0: continue
        
        x_tensor = torch.FloatTensor(X_train_np).to(DEVICE)
        y_tensor = torch.FloatTensor(Y_train_np).to(DEVICE)
        OUTPUT_DIM = Y_train_np.shape[1]
        
        func_result = {}

        # 遍历复杂度等级
        for level_idx, (degree, size) in enumerate(complexity_levels):
            logging.info(f"  > Complexity Level {level_idx+1}: Degree={degree}, Size={size}")
            
            # --- Step A: 计算目标参数量 (使用 Dummy 模型) ---
            
            # 1. Hybrid Rational Net
            hybrid_dummy = HybridRationalNet(INPUT_DIM, OUTPUT_DIM, unit_degree=degree, num_units=size)
            
            # 2. CFNet Standard
            cfnet_dummy = CFNet_Standard(INPUT_DIM, OUTPUT_DIM, depth=size, poly_degree=degree)
            
            # 3. Boost Model (完全体)
            boost_dummy = EnsembleResCoFrNet(INPUT_DIM, OUTPUT_DIM, shallow_depth=2, poly_degree=degree, learning_rate=0.1)
            for _ in range(size): boost_dummy.add_model()
            
            # 4. MoE Model (完全体)
            moe_hparams = {'input_dim': INPUT_DIM, 'output_dim': OUTPUT_DIM, 'shallow_depth_per_cofrnet': 2, 'polynomial_degree': degree}
            moe_dummy = MoE_Ensemble(moe_hparams)
            moe_dummy.gating.centers = nn.Parameter(torch.randn(size, INPUT_DIM))
            moe_dummy.gating.widths = nn.Parameter(torch.ones(size, 1))
            for _ in range(size): moe_dummy.add_expert()
            
            base_model_specs = {
                "Hybrid": {"params": count_parameters(hybrid_dummy), "dummy": hybrid_dummy},
                "CFNet_Std": {"params": count_parameters(cfnet_dummy), "dummy": cfnet_dummy},
                "Boost": {"params": count_parameters(boost_dummy), "dummy": boost_dummy},
                "MoE": {"params": count_parameters(moe_dummy), "dummy": moe_dummy}
            }
            
            # --- Step B: 运行实验 ---
            for model_name, spec in base_model_specs.items():
                base_params = spec['params']
                base_params_10x = base_params * 10
                
                # 计算对应的 MLP 和 KAN 配置
                mlp_h_1x = find_mlp_hidden_dim(INPUT_DIM, OUTPUT_DIM, base_params)
                mlp_h_10x = find_mlp_hidden_dim(INPUT_DIM, OUTPUT_DIM, base_params_10x)
                kan_h_1x = find_kan_hidden_dim(INPUT_DIM, OUTPUT_DIM, base_params)
                kan_h_10x = find_kan_hidden_dim(INPUT_DIM, OUTPUT_DIM, base_params_10x)
                
                logging.info(f"    Target Params ({model_name}): {base_params}")

                for seed in SEEDS:
                    set_seed(seed)
                    
                    # 定义任务队列
                    # 注意：对于 Boost 和 MoE，我们现在初始化 *空* 模型，稍后在训练函数中增长
                    tasks = []
                    
                    # 1. 主模型 (CoFrNet Variant)
                    if model_name == "Hybrid":
                        m = HybridRationalNet(INPUT_DIM, OUTPUT_DIM, unit_degree=degree, num_units=size).to(DEVICE)
                        tasks.append({"name": model_name, "model": m, "type": "standard"})
                        
                    elif model_name == "CFNet_Std":
                        m = CFNet_Standard(INPUT_DIM, OUTPUT_DIM, depth=size, poly_degree=degree).to(DEVICE)
                        tasks.append({"name": model_name, "model": m, "type": "standard"})
                        
                    elif model_name == "Boost":
                        # 初始化空模型
                        m = EnsembleResCoFrNet(INPUT_DIM, OUTPUT_DIM, shallow_depth=2, poly_degree=degree, learning_rate=0.1)
                        tasks.append({"name": model_name, "model": m, "type": "boost_iterative"})
                        
                    elif model_name == "MoE":
                        # 初始化空模型
                        m = MoE_Ensemble(moe_hparams)
                        tasks.append({"name": model_name, "model": m, "type": "moe_iterative"})

                    # 2. 对比模型 (MLP/KAN)
                    tasks.append({"name": f"MLP_1x_for_{model_name}", "model": MLP(INPUT_DIM, mlp_h_1x, OUTPUT_DIM).to(DEVICE), "type": "standard"})
                    tasks.append({"name": f"MLP_10x_for_{model_name}", "model": MLP(INPUT_DIM, mlp_h_10x, OUTPUT_DIM).to(DEVICE), "type": "standard"})
                    tasks.append({"name": f"KAN_1x_for_{model_name}", "model": KAN(INPUT_DIM, kan_h_1x, OUTPUT_DIM).to(DEVICE), "type": "standard"})
                    tasks.append({"name": f"KAN_10x_for_{model_name}", "model": KAN(INPUT_DIM, kan_h_10x, OUTPUT_DIM).to(DEVICE), "type": "standard"})
                    
                    # 执行训练
                    for task in tasks:
                        m_obj = task["model"]
                        t_type = task["type"]
                        t_name = task["name"]
                        
                        if t_type == "standard":
                            loss, loss_hist = train_standard(m_obj, x_tensor, y_tensor, epochs=EPOCHS, lr=LR)
                            final_params = count_parameters(m_obj)
                            
                        elif t_type == "boost_iterative":
                            # 传入目标 size
                            loss, loss_hist = train_boost_iterative(m_obj, x_tensor, y_tensor, total_units=size, total_epochs=EPOCHS, lr=LR)
                            final_params = count_parameters(m_obj) # 此时模型已增长完毕
                            
                        elif t_type == "moe_iterative":
                            # 传入目标 experts
                            loss, loss_hist = train_moe_iterative(m_obj, x_tensor, y_tensor, total_experts=size, total_epochs=EPOCHS, lr=LR)
                            final_params = count_parameters(m_obj)
                        
                        # 存储结果
                        key = f"{t_name}_L{level_idx}"
                        if key not in func_result: func_result[key] = []
                        func_result[key].append({
                            "params": final_params,
                            "loss": loss,
                            "loss_history": loss_hist,
                            "seed": seed,
                            "base_target": base_params
                        })

        full_results[f_name] = func_result

    # 保存
    with open("comparison_results.json", "w") as f:
        json.dump(full_results, f, indent=4)
        
    logging.info("All experiments finished. Results saved to comparison_results.json")

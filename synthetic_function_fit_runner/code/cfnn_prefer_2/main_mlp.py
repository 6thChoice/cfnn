import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import random
import os
import matplotlib.pyplot as plt
import math # 新增
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))


from cfnet import CFNet_Standard

# ==========================================
# 1. 环境配置与工具函数
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_mlp_hidden_dim(target_params, input_dim=3, output_dim=1):
    """根据目标参数量反推 SimpleMLP 的 hidden_dim"""
    a = 1
    b = input_dim + output_dim + 2
    c = output_dim - target_params
    delta = b**2 - 4*a*c
    if delta < 0: return 8
    h = (-b + math.sqrt(delta)) / (2*a)
    return max(1, int(h))

def get_final_cfnet_params(model_type, input_dim, output_dim, d, p):
    """
    创建一个临时模型并填充到最终规模，以计算其最终参数量。
    """
    if model_type == "standard":
        from cfnet import CFNet_Standard
        m = CFNet_Standard(input_dim, output_dim, depth=d, poly_degree=p)
    elif model_type == "hybrid":
        from cfnet import HybridRationalNet
        m = HybridRationalNet(input_dim, output_dim, unit_degree=p, num_units=d)
    elif model_type == "boost":
        from cfnet import EnsembleResCoFrNet
        m = EnsembleResCoFrNet(input_dim, output_dim, shallow_depth=4, poly_degree=p, learning_rate=0.1)
        for _ in range(d): m.add_model() # 模拟添加 d 个子模型
    elif model_type == "moe":
        from cfnet import MoE_Ensemble
        hparams = {'input_dim': input_dim, 'output_dim': output_dim, 
                   'shallow_depth_per_cofrnet': 4, 'polynomial_degree': p}
        m = MoE_Ensemble(hparams)
        for _ in range(d): 
            m.add_expert() # 模拟添加 d 个专家
            m.gating.add_expert_gate(initial_center=np.zeros(input_dim)) # 添加对应的门控参数
    
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [此处保持 generate_complex_data 和 SimpleMLP 定义不变]
def generate_complex_data(n_samples=10000):
    # 1. 生成三维输入 X
    # 建议范围选择 [0.5, 2.5]，避开 0 附近以防止分母为 0 导致数值爆炸
    X = np.random.uniform(0.5, 2.5, (n_samples, 3)).astype(np.float32)
    
    # 2. 定义目标函数: f(X) = x1 + 1 / (x2 + 1/x3)
    # 我们按照嵌套顺序从内向外计算
    inner_part = 1.0 / X[:, 2]              # 1 / x3
    middle_part = X[:, 1] + inner_part      # x2 + 1/x3
    y = (X[:, 0] + 1.0 / middle_part).reshape(-1, 1)
    
    # 3. 添加少量噪声
    y += 0.02 * np.random.normal(size=y.shape).astype(np.float32)
    
    # 4. 数据集划分 (65% 训练, 5% 验证, 30% 测试)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.35, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(30/35), random_state=42
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
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

# [此处保持 train_model_adaptive 定义不变]
def train_model_adaptive(model, model_type, train_loader, val_loader, d_total, p, epochs_per_stage):
    criterion = nn.MSELoss()
    history = {"train_loss": [], "val_rmse": []}
    model.to(DEVICE)
    if model_type in ["boost", "moe"]:
        for stage in range(1, d_total + 1):
            if model_type == "boost":
                model.add_model() 
                model.to(DEVICE)
                model.freeze_all_but_latest() 
            else: # moe
                model.add_expert() 
                model.to(DEVICE)
                initial_center = np.random.uniform(-2, 2, 3)
                model.gating.add_expert_gate(initial_center=initial_center)
                model.to(DEVICE)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(trainable_params, lr=0.001)
            for epoch in range(epochs_per_stage):
                model.train()
                epoch_loss = 0
                for bx, by in train_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                model.eval()
                val_mse = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                        val_mse += nn.functional.mse_loss(model(vx), vy, reduction='sum').item()
                history["train_loss"].append(epoch_loss / len(train_loader))
                history["val_rmse"].append(np.sqrt(val_mse / len(val_loader.dataset)))
    else: # global
        total_epochs = d_total * epochs_per_stage
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(total_epochs):
            model.train()
            epoch_loss = 0
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            model.eval()
            val_mse = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                    val_mse += nn.functional.mse_loss(model(vx), vy, reduction='sum').item()
            history["train_loss"].append(epoch_loss / len(train_loader))
            history["val_rmse"].append(np.sqrt(val_mse / len(val_loader.dataset)))
    return history

# ==========================================
# 2. 执行主流程 (集成 MLP 对比)
# ==========================================
if __name__ == "__main__":
    epochs_per_stage = 300
    input_dim, output_dim = 3, 1
    
    for model_type in ["boost", "moe", "hybrid"]:
        for d in range(1, 6):
            for p in range(1, 6):
                result_path = f"mlp"
                os.makedirs(result_path, exist_ok=True)
                after_pix = f"_{model_type}_d{d}_p{p}"
                
                # 数据准备
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = generate_complex_data()
                train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=128, shuffle=True)
                val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=128)

                # 1. 初始化 CFNet 变体
                if model_type == "standard":
                    from cfnet import CFNet_Standard
                    model = CFNet_Standard(input_dim, output_dim, depth=d, poly_degree=p)
                elif model_type == "hybrid":
                    from cfnet import HybridRationalNet
                    model = HybridRationalNet(input_dim, output_dim, unit_degree=p, num_units=d)
                elif model_type == "boost":
                    from cfnet import EnsembleResCoFrNet
                    model = EnsembleResCoFrNet(input_dim, output_dim, shallow_depth=4, poly_degree=p, learning_rate=0.1)
                elif model_type == "moe":
                    from cfnet import MoE_Ensemble
                    hparams = {'input_dim': input_dim, 'output_dim': output_dim, 'shallow_depth_per_cofrnet': 4, 'polynomial_degree': p}
                    model = MoE_Ensemble(hparams)

                # 2. 初始化对标参数量的 MLP
                target_params = get_final_cfnet_params(model_type, input_dim, output_dim, d, p)
                mlp_h = get_mlp_hidden_dim(target_params)
                mlp_model = SimpleMLP(input_dim, output_dim, hidden_dim=mlp_h)
                
                print(f"[{model_type.upper()}] Final Target Params: {target_params} | MLP Hidden: {mlp_h}")

                # 3. 训练两个模型
                # history_cf = train_model_adaptive(model, model_type, train_loader, val_loader, d, p, epochs_per_stage)
                history_mlp = train_model_adaptive(mlp_model, "mlp_global", train_loader, val_loader, d, p, epochs_per_stage)

                # 4. 获取测试集预测
                # model.eval()
                mlp_model.eval()
                with torch.no_grad():
                    tx = torch.from_numpy(X_test).to(DEVICE)
                    # preds_cf = model(tx).cpu().numpy()
                    preds_mlp = mlp_model(tx).cpu().numpy()

                # 5. 保存结果
                final_output = {
                    "config": {"model_type": model_type, "d": d, "p": p, "params": target_params, "mlp_h": mlp_h},
                    "history": {"mlp": history_mlp},
                    "test_data": {
                        "y_true": y_test.flatten().tolist(),
                        # "y_cfnet": preds_cf.flatten().tolist(),
                        "y_mlp": preds_mlp.flatten().tolist(),
                    }
                }
                os.makedirs(f"{result_path}/result", exist_ok=True)
                with open(f"{result_path}/result/comparison{after_pix}.json", "w") as f:
                    json.dump(final_output, f, indent=4)

                # 6. 可视化对比
                plt.style.use('seaborn-v0_8-muted') 
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                epochs_range = range(1, d * epochs_per_stage + 1)
                label_name = f"MLP-{model_type.capitalize()}"

                # 左上: Loss
                axes[0, 0].plot(epochs_range, history_mlp["train_loss"], label=label_name, color='C0')
                axes[0, 0].set_title(f"Training Loss: {label_name}", fontweight='bold')
                axes[0, 0].set_yscale('log')
                axes[0, 0].legend()

                # 右上: RMSE
                axes[0, 1].plot(epochs_range, history_mlp["val_rmse"], label=label_name, color='C0')
                axes[0, 1].set_title(f"Validation RMSE: {label_name}", fontweight='bold')
                axes[0, 1].legend()

                # 左下: Scatter
                axes[1, 0].scatter(y_test, preds_mlp, alpha=0.5, s=15, label=label_name)
                lims = [y_test.min(), y_test.max()]
                axes[1, 0].plot(lims, lims, 'r--', label='Ideal')
                axes[1, 0].set_title("True vs Predicted", fontweight='bold')
                axes[1, 0].legend()

                # 右下: Function Fitting
                sort_idx = np.argsort(X_test[:, 0])
                axes[1, 1].scatter(X_test[sort_idx, 0], y_test[sort_idx], c='black', s=10, alpha=0.15, label='GT')
                axes[1, 1].scatter(X_test[sort_idx, 0], preds_mlp[sort_idx], s=15, alpha=0.6, label='Pred', color='C0')
                axes[1, 1].set_title(f"Fitting over Dim 0 ({model_type})", fontweight='bold')
                axes[1, 1].legend()

                plt.tight_layout()
                os.makedirs(f'{result_path}/plot', exist_ok=True)
                plt.savefig(f"{result_path}/plot/cfnet_comparison{after_pix}.png", dpi=300)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import random
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))


# 从本地 cfnet.py 导入模型
from cfnet import CFNet_Standard

# ==========================================
# 1. 环境配置与随机种子固定
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 数据准备 (3D 输入, 65:5:30 划分)
# ==========================================
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

# ==========================================
# 3. 对比模型 (MLP)
# ==========================================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(), # Tanh 在处理这种周期性/有界函数时有时比 ReLU 更好，也更接近 CFNet 的内部激活
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. 训练与评估逻辑
# ==========================================
def train_model_adaptive(model, model_type, train_loader, val_loader, d_total, p, epochs_per_stage):
    criterion = nn.MSELoss()
    history = {"train_loss": [], "val_rmse": []}
    model.to(DEVICE)

    # ------------------------------------------
    # 分支 A: 迭代训练模式 (Boost & MoE)
    # ------------------------------------------
    if model_type in ["boost", "moe"]:
        for stage in range(1, d_total + 1):
            print(f"--- Stage {stage}/{d_total}: Adding Sub-model/Expert ---")
            
            # 1. 添加组件并立即同步设备 [修复 RuntimeError]
            if model_type == "boost":
                model.add_model() 
                model.to(DEVICE) # 确保新子模型在 GPU
                model.freeze_all_but_latest() 
            else: # moe
                model.add_expert() 
                model.to(DEVICE) # 确保新专家在 GPU
                # 随机初始化中心点并指定设备
                initial_center = np.random.uniform(-2, 2, 3)
                model.gating.add_expert_gate(initial_center=initial_center)
                model.to(DEVICE) # 确保 gating 参数也在正确设备

            # 2. 更新优化器（针对新参数和冻结状态）
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(trainable_params, lr=0.001)

            # 3. 执行当前阶段训练
            for epoch in range(epochs_per_stage):
                model.train()
                epoch_loss = 0
                for bx, by in train_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(bx)
                    loss = criterion(outputs, by)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                # 验证集评估
                model.eval()
                val_mse = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                        v_pred = model(vx)
                        val_mse += nn.functional.mse_loss(v_pred, vy, reduction='sum').item()
                
                history["train_loss"].append(epoch_loss / len(train_loader))
                history["val_rmse"].append(np.sqrt(val_mse / len(val_loader.dataset)))
                
                if (epoch + 1) % 50 == 0 or epoch == epochs_per_stage - 1:
                    print(f"  Stage {stage} Epoch {epoch+1}/{epochs_per_stage} | Loss: {history['train_loss'][-1]:.6f}")

    # ------------------------------------------
    # 分支 B: 全局训练模式 (Standard & Hybrid)
    # ------------------------------------------
    else:
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
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{total_epochs} | Loss: {history['train_loss'][-1]:.6f}")

    # 【修复重点】确保函数无论走哪个分支都会返回 history 字典
    return history

def train_model(model, train_loader, val_loader, epochs=150):
    # 稍微降低学习率以获得更稳定的收敛曲线
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    history = {"train_loss": [], "val_rmse": []}
    
    model.to(DEVICE)
    for epoch in range(epochs):
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
                v_pred = model(vx)
                val_mse += nn.functional.mse_loss(v_pred, vy, reduction='sum').item()
        
        history["train_loss"].append(epoch_loss / len(train_loader))
        history["val_rmse"].append(np.sqrt(val_mse / len(val_loader.dataset)))
        
    return history

# ==========================================
# 5. 执行主流程
# ==========================================
if __name__ == "__main__":
    # --- 核心实验参数设置 ---
    model_type = "standard"  # 可选: "standard", "boost", "moe", "hybrid"
    d = 1                   # Standard: depth | Boost: models | MoE: experts | Hybrid: units
    p = 5                    # 多项式阶数 (poly_degree)
    epochs_per_stage = 300
    
    params_pair = [{"d": 10, "p": 10},
                   {"d": 15, "p": 15},
                   {"d": 20, "p": 20},
                   {"d": 25, "p": 25},
                   {"d": 10, "p": 5},
                   {"d": 15, "p": 5},
                   {"d": 20, "p": 5},
                   {"d": 25, "p": 5},
                   {"d": 5, "p": 10},
                   {"d": 5, "p": 15},
                   {"d": 5, "p": 20},
                   {"d": 5, "p": 25}]

    for model_type in ["standard", "boost", "moe", "hybrid"]:

        for item in params_pair:
            d = item['d']
            p = item['p']
        # for d in range(10, 26, 5):
        #     for p in range(10, 26, 5):
            result_path = f"{model_type}"
            os.makedirs(result_path, exist_ok=True)

            after_pix = f"_{model_type}_d{d}_p{p}"
            
            # 准备数据
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = generate_complex_data()
            train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
            val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
            train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=128)

            # --- 统一模型初始化逻辑 ---
            input_dim, output_dim = 3, 1
            
            if model_type == "standard":
                from cfnet import CFNet_Standard
                model = CFNet_Standard(input_dim, output_dim, depth=d, poly_degree=p)
                
            elif model_type == "hybrid":
                from cfnet import HybridRationalNet
                model = HybridRationalNet(input_dim, output_dim, unit_degree=p, num_units=d)
                
            elif model_type == "boost":
                from cfnet import EnsembleResCoFrNet
                # Boost 这里的 d 为子模型数量，内部子模型深度固定为 3
                model = EnsembleResCoFrNet(input_dim, output_dim, shallow_depth=4, poly_degree=p, learning_rate=0.1)
                for _ in range(d): model.add_model()
                
            elif model_type == "moe":
                from cfnet import MoE_Ensemble
                hparams = {
                    'input_dim': input_dim, 'output_dim': output_dim,
                    'shallow_depth_per_cofrnet': 4, 'polynomial_degree': p
                }
                model = MoE_Ensemble(hparams)
                for _ in range(d): model.add_expert()
                # 初始化 MoE 的门控中心（简单均匀分布或随机）
                for _ in range(d):
                    model.gating.add_expert_gate(initial_center=np.random.uniform(-1, 1, input_dim))

            print(f"Training {model_type.upper()} (d={d}, p={p}) on {DEVICE}...")
            history = train_model_adaptive(model, model_type, train_loader, val_loader, d, p, epochs_per_stage)

            # --- 获取测试集结果 ---
            model.eval()
            with torch.no_grad():
                tx = torch.from_numpy(X_test).to(DEVICE)
                preds = model(tx).cpu().numpy()

            # --- 4. 汇总结果并保存到 JSON ---
            final_output = {
                "config": {"model_type": model_type, "seed": 42, "d": d, "p": p, "epochs": d * epochs_per_stage},
                "history": {"cfnet": history},
                "test_data": {
                    "X_test_dim0": X_test[:, 0].flatten().tolist(),
                    "y_true": y_test.flatten().tolist(),
                    "y_cfnet": preds.flatten().tolist(),
                }
            }
            os.makedirs(f"{result_path}/result", exist_ok=True)
            with open(f"{result_path}/result/cfnet_full_results{after_pix}.json", "w") as f:
                json.dump(final_output, f, indent=4)

            # ==========================================
            # 6. 可视化 (修改标签显示)
            # ==========================================
            plt.style.use('seaborn-v0_8-muted') 
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            epochs_range = range(1, d * epochs_per_stage + 1)
            label_name = f"CFNet-{model_type.capitalize()}"

            # 左上: Loss
            axes[0, 0].plot(epochs_range, history["train_loss"], label=label_name, color='C0')
            axes[0, 0].set_title(f"Training Loss: {label_name}", fontweight='bold')
            axes[0, 0].set_yscale('log')
            axes[0, 0].legend()

            # 右上: RMSE
            axes[0, 1].plot(epochs_range, history["val_rmse"], label=label_name, color='C0')
            axes[0, 1].set_title(f"Validation RMSE: {label_name}", fontweight='bold')
            axes[0, 1].legend()

            # 左下: Scatter
            axes[1, 0].scatter(y_test, preds, alpha=0.5, s=15, label=label_name)
            lims = [y_test.min(), y_test.max()]
            axes[1, 0].plot(lims, lims, 'r--', label='Ideal')
            axes[1, 0].set_title("True vs Predicted", fontweight='bold')
            axes[1, 0].legend()

            # 右下: Function Fitting
            sort_idx = np.argsort(X_test[:, 0])
            axes[1, 1].scatter(X_test[sort_idx, 0], y_test[sort_idx], c='black', s=10, alpha=0.15, label='GT')
            axes[1, 1].scatter(X_test[sort_idx, 0], preds[sort_idx], s=15, alpha=0.6, label='Pred', color='C0')
            axes[1, 1].set_title(f"Fitting over Dim 0 ({model_type})", fontweight='bold')
            axes[1, 1].legend()

            plt.tight_layout()
            os.makedirs(f'{result_path}/plot', exist_ok=True)
            plt.savefig(f"{result_path}/plot/cfnet_comparison{after_pix}.png", dpi=300)
            # plt.show()
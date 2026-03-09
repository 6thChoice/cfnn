from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent

"""
谱偏差实验 - 统一训练脚本
训练所有模型: CFNN全系列 (Standard, Boost, MoE, Hybrid) + Baselines (MLP, SIREN, RFF-MLP, Chebyshev-KAN)
保存预测结果用于后续谱分析
"""
import os
import sys
sys.path.insert(0, str(CODE_DIR))
import json
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from datetime import datetime

# 路径设置

from baselines import SIREN, RFFMLP, ChebyshevKAN, count_parameters

# 导入 CFNN 模型
from cfnet import CFNet_Standard, HybridRationalNet, EnsembleResCoFrNet, MoE_Ensemble

# ==========================================
# 1. 配置
# ==========================================
ROOT_PATH = str(BASE_DIR / "results")
OUTPUT_DIR = os.path.join(ROOT_PATH, "spectral_comparison_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型配置
CFNN_TYPES = ['standard', 'boost', 'moe', 'hybrid']
BASELINE_TYPES = ['mlp', 'siren', 'rff_mlp', 'chebyshev_kan']
ALL_MODEL_TYPES = CFNN_TYPES + BASELINE_TYPES

# 训练配置
EPOCHS = 300
BATCH_SIZE = 128
LR = 0.001
INPUT_DIM = 3
OUTPUT_DIM = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CFNN 参数配置 (d=depth, p=poly_degree)
CFNN_PARAMS = [(5, 5)]  # 可扩展: [(3,3), (5,5), (10,5)]

# 随机种子
SEEDS = [42]


# ==========================================
# 2. 工具函数
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def generate_data(n_samples=10000, epsilon=1e-5, seed=42):
    """生成 nn_hard 数据: f(X) = (x1 * x2) / (x3 + ε)"""
    np.random.seed(seed)

    X_12 = np.random.uniform(-2, 2, (n_samples, 2))
    X_3 = np.random.uniform(1, 3, (n_samples, 1))
    X = np.hstack([X_12, X_3]).astype(np.float32)

    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    y = ((x1 * x2) / (x3 + epsilon)).reshape(-1, 1)
    y += 0.02 * np.random.normal(size=y.shape).astype(np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.35, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=30/35, random_state=seed)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ==========================================
# 3. 模型创建
# ==========================================
def create_cfnn_model(model_type, d, p):
    """创建 CFNN 模型"""
    if model_type == 'standard':
        return CFNet_Standard(INPUT_DIM, OUTPUT_DIM, depth=d, poly_degree=p)
    elif model_type == 'hybrid':
        return HybridRationalNet(INPUT_DIM, OUTPUT_DIM, unit_degree=p, num_units=d)
    elif model_type == 'boost':
        model = EnsembleResCoFrNet(INPUT_DIM, OUTPUT_DIM, shallow_depth=4, poly_degree=p, learning_rate=0.1)
        return model
    elif model_type == 'moe':
        hparams = {'input_dim': INPUT_DIM, 'output_dim': OUTPUT_DIM,
                   'shallow_depth_per_cofrnet': 4, 'polynomial_degree': p}
        return MoE_Ensemble(hparams)
    else:
        raise ValueError(f"Unknown CFNN type: {model_type}")


def create_baseline_model(model_type, hidden_dim=64):
    """创建 Baseline 模型"""
    if model_type == 'mlp':
        return nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, OUTPUT_DIM)
        )
    elif model_type == 'siren':
        return SIREN(INPUT_DIM, OUTPUT_DIM, hidden_dim=hidden_dim, hidden_layers=2, omega_0=30)
    elif model_type == 'rff_mlp':
        return RFFMLP(INPUT_DIM, OUTPUT_DIM, hidden_dim=hidden_dim, rff_features=hidden_dim*2, sigma=1.0)
    elif model_type == 'chebyshev_kan':
        return ChebyshevKAN(INPUT_DIM, OUTPUT_DIM, hidden_dim=hidden_dim, degree=5, num_layers=2)
    else:
        raise ValueError(f"Unknown baseline type: {model_type}")


# ==========================================
# 4. 训练函数
# ==========================================
def train_standard_model(model, train_loader, val_loader, epochs=EPOCHS):
    """训练标准模型 (非 adaptive)"""
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

    history = {'train_loss': [], 'val_rmse': []}
    best_val_rmse = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_mse = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                val_mse += nn.functional.mse_loss(model(vx), vy, reduction='sum').item()

        val_rmse = np.sqrt(val_mse / len(val_loader.dataset))
        scheduler.step(val_rmse)

        history['train_loss'].append(train_loss / len(train_loader))
        history['val_rmse'].append(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    return history, best_val_rmse


def train_boost_model(model, train_loader, val_loader, n_models=5, epochs_per_model=60):
    """训练 Boost 模型"""
    model.to(DEVICE)
    criterion = nn.MSELoss()
    history = {'train_loss': [], 'val_rmse': []}

    for stage in range(n_models):
        model.add_model()
        model.to(DEVICE)
        model.freeze_all_but_latest()

        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)

        for epoch in range(epochs_per_model):
            model.train()
            train_loss = 0
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_mse = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                    val_mse += nn.functional.mse_loss(model(vx), vy, reduction='sum').item()

            history['train_loss'].append(train_loss / len(train_loader))
            history['val_rmse'].append(np.sqrt(val_mse / len(val_loader.dataset)))

    return history, history['val_rmse'][-1]


def train_moe_model(model, train_loader, val_loader, n_experts=5, epochs_per_expert=60):
    """训练 MoE 模型"""
    model.to(DEVICE)
    criterion = nn.MSELoss()
    history = {'train_loss': [], 'val_rmse': []}

    for stage in range(n_experts):
        model.add_expert()
        initial_center = np.random.uniform(-2, 2, INPUT_DIM)
        model.gating.add_expert_gate(initial_center=initial_center)
        model.to(DEVICE)

        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)

        for epoch in range(epochs_per_expert):
            model.train()
            train_loss = 0
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_mse = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                    val_mse += nn.functional.mse_loss(model(vx), vy, reduction='sum').item()

            history['train_loss'].append(train_loss / len(train_loader))
            history['val_rmse'].append(np.sqrt(val_mse / len(val_loader.dataset)))

    return history, history['val_rmse'][-1]


# ==========================================
# 5. 主训练流程
# ==========================================
def train_all_models(data, d=5, p=5, seed=42):
    """训练所有模型并返回预测结果"""
    set_seed(seed)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                            batch_size=BATCH_SIZE)

    results = {
        'config': {'d': d, 'p': p, 'seed': seed, 'epochs': EPOCHS},
        'test_data': {
            'y_true': y_test.flatten().tolist(),
            'X_test_dim0': X_test[:, 0].tolist(),
            'X_test_dim1': X_test[:, 1].tolist(),
            'X_test_dim2': X_test[:, 2].tolist(),
        },
        'models': {}
    }

    # 训练 CFNN 模型
    print("\n===== Training CFNN Models =====")
    for model_type in CFNN_TYPES:
        print(f"\n[{model_type.upper()}] d={d}, p={p}")

        try:
            model = create_cfnn_model(model_type, d, p)

            if model_type == 'boost':
                history, val_rmse = train_boost_model(model, train_loader, val_loader,
                                                       n_models=d, epochs_per_model=EPOCHS//d)
            elif model_type == 'moe':
                history, val_rmse = train_moe_model(model, train_loader, val_loader,
                                                     n_experts=d, epochs_per_expert=EPOCHS//d)
            else:
                history, val_rmse = train_standard_model(model, train_loader, val_loader)

            # 预测
            model.eval()
            with torch.no_grad():
                tx = torch.from_numpy(X_test).to(DEVICE)
                predictions = model(tx).cpu().numpy().flatten().tolist()

            params = count_parameters(model)
            test_rmse = np.sqrt(np.mean((np.array(predictions) - y_test.flatten())**2))

            results['models'][model_type] = {
                'predictions': predictions,
                'params': params,
                'val_rmse': float(val_rmse),
                'test_rmse': float(test_rmse),
            }
            print(f"  Params: {params}, Val RMSE: {val_rmse:.6f}, Test RMSE: {test_rmse:.6f}")

        except Exception as e:
            print(f"  Error: {e}")
            results['models'][model_type] = {'error': str(e)}

    # 训练 Baseline 模型
    print("\n===== Training Baseline Models =====")
    for model_type in BASELINE_TYPES:
        print(f"\n[{model_type.upper()}]")

        try:
            # 根据 CFNN 参数量设置 hidden_dim
            if model_type == 'chebyshev_kan':
                hidden_dim = 16
            else:
                hidden_dim = 64

            model = create_baseline_model(model_type, hidden_dim)
            history, val_rmse = train_standard_model(model, train_loader, val_loader)

            # 预测
            model.eval()
            with torch.no_grad():
                tx = torch.from_numpy(X_test).to(DEVICE)
                predictions = model(tx).cpu().numpy().flatten().tolist()

            params = count_parameters(model)
            test_rmse = np.sqrt(np.mean((np.array(predictions) - y_test.flatten())**2))

            results['models'][model_type] = {
                'predictions': predictions,
                'params': params,
                'val_rmse': float(val_rmse),
                'test_rmse': float(test_rmse),
            }
            print(f"  Params: {params}, Val RMSE: {val_rmse:.6f}, Test RMSE: {test_rmse:.6f}")

        except Exception as e:
            print(f"  Error: {e}")
            results['models'][model_type] = {'error': str(e)}

    return results


def main():
    """主函数"""
    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")

    for seed in SEEDS:
        for d, p in CFNN_PARAMS:
            print(f"\n{'='*60}")
            print(f"Training: d={d}, p={p}, seed={seed}")
            print(f"{'='*60}")

            # 生成数据
            data = generate_data(seed=seed)

            # 训练所有模型
            results = train_all_models(data, d=d, p=p, seed=seed)

            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"spectral_comparison_d{d}_p{p}_seed{seed}_{timestamp}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)

            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\nResults saved: {filepath}")

            # 打印汇总
            print("\n" + "="*60)
            print("Summary")
            print("="*60)
            print(f"{'Model':<15} {'Params':>8} {'Test RMSE':>12}")
            print("-"*40)
            for model_type in ALL_MODEL_TYPES:
                if model_type in results['models'] and 'test_rmse' in results['models'][model_type]:
                    m = results['models'][model_type]
                    print(f"{model_type:<15} {m['params']:>8} {m['test_rmse']:>12.6f}")


if __name__ == "__main__":
    main()

"""
KAN 分类任务基础模块
为所有6个分类任务提供统一的KAN模型和训练框架
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import logging
import sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parents[1]
CLASSIFY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CLASSIFY_DIR))


import os
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from cfnet_complicate import CFNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SplineBasis(nn.Module):
    """B-样条基函数"""
    def __init__(self, grid_size=10, spline_order=3):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.n_basis = grid_size + spline_order

    def forward(self, x):
        batch_size, n_features = x.shape
        grid = torch.linspace(-1, 1, self.n_basis, device=x.device)
        x_expanded = x.unsqueeze(-1)
        grid_expanded = grid.view(1, 1, -1)
        distances = torch.abs(x_expanded - grid_expanded)
        width = 2.0 / (self.grid_size - 1) if self.grid_size > 1 else 1.0
        width = width * (self.spline_order + 1)
        basis = torch.exp(-0.5 * (distances / width) ** 2)
        return basis


class KANLayer(nn.Module):
    """KAN层"""
    def __init__(self, in_features, out_features, grid_size=10, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.n_basis = grid_size + spline_order

        self.spline_basis = SplineBasis(grid_size, spline_order)
        self.coeffs = nn.Parameter(torch.randn(in_features, out_features, self.n_basis) * 0.1)
        self.residual_weight = nn.Parameter(torch.ones(out_features) * 0.1)

    def forward(self, x):
        batch_size = x.shape[0]
        basis = self.spline_basis(x)
        output = torch.zeros(batch_size, self.out_features, device=x.device)

        for i in range(self.out_features):
            contribution = (basis * self.coeffs[:, i, :].unsqueeze(0)).sum(dim=[1, 2])
            output[:, i] = contribution

        residual = torch.sigmoid(x) * x
        output = output + residual[:, :self.out_features].sum(dim=1, keepdim=True) * self.residual_weight
        return output


class KAN(nn.Module):
    """KAN分类网络"""
    def __init__(self, input_dim, output_dim, hidden_dim=32, num_layers=2,
                 grid_size=10, spline_order=3):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(KANLayer(dims[i], dims[i+1], grid_size, spline_order))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i+1]))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.network(x)


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_aligned_kan(input_dim, output_dim, target_params, num_layers=2):
    """获取与目标参数量对齐的KAN模型"""
    grid_size = 10
    spline_order = 3
    n_basis = grid_size + spline_order

    def calc_params(hidden_dim):
        total = 0
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(len(dims) - 1):
            layer_params = dims[i] * dims[i+1] * n_basis + dims[i+1]
            if i < len(dims) - 2:
                layer_params += 2 * dims[i+1]
            total += layer_params
        return total

    best_h = 8
    min_diff = float('inf')
    for h in range(4, 200):
        p = calc_params(h)
        diff = abs(p - target_params)
        if diff < min_diff:
            min_diff = diff
            best_h = h
        if p > target_params * 1.5:
            break

    return KAN(input_dim, output_dim, best_h, num_layers, grid_size, spline_order), best_h


def train_model(model, train_loader, val_loader, params, run_id, device=DEVICE):
    """标准的 PyTorch 训练循环"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    pbar = tqdm(range(params['epochs']), desc=f"Run {run_id} Training", leave=False)

    for epoch in pbar:
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_val.size(0)
                val_correct += (predicted == y_val).sum().item()

        val_acc = val_correct / val_total
        pbar.set_postfix({'Loss': f"{train_loss/len(train_loader):.4f}", 'Val Acc': f"{val_acc:.4f}"})
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= params['early_stopping_patience']:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model


def evaluate_model(model, test_loader, device=DEVICE):
    """测试集评估"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.numpy())

    return accuracy_score(all_targets, all_preds)


def run_kan_experiment(dataset_name, load_data_fn, cfnet_hparams, output_filename):
    """
    运行KAN实验的通用函数

    Args:
        dataset_name: 数据集名称
        load_data_fn: 加载数据的函数，签名: load_data_fn(run_seed) -> (train_ds, val_ds, test_ds, (n_features, n_classes))
        cfnet_hparams: CFNet的超参数配置，用于获取对齐的参数量
        output_filename: 输出结果文件名
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    NUM_RUNS = 10
    START_SEED = 42

    # KAN超参数
    hparams = {
        'epochs': cfnet_hparams.get('epochs', 200),
        'batch_size': cfnet_hparams.get('batch_size', 128),
        'learning_rate': cfnet_hparams.get('learning_rate', 0.005),
        'weight_decay': cfnet_hparams.get('weight_decay', 1e-4),
        'early_stopping_patience': cfnet_hparams.get('early_stopping_patience', 20)
    }

    # 获取CFNet参数量作为锚点
    logging.info(f"[{dataset_name}] 计算 CFNet 参数量作为对齐基准...")
    temp_train_ds, _, _, (n_features, n_classes) = load_data_fn(START_SEED)

    cfnet_anchor = CFNet(
        input_dim=n_features,
        output_dim=n_classes,
        depth=cfnet_hparams.get('depth', 5),
        poly_degree=cfnet_hparams.get('poly_degree', 3)
    )
    target_params = count_parameters(cfnet_anchor)
    logging.info(f"CFNet 参数量: {target_params}")

    # 获取对齐的KAN配置
    kan_model, hidden_dim = get_aligned_kan(n_features, n_classes, target_params, num_layers=2)
    kan_params = count_parameters(kan_model)
    logging.info(f"KAN 配置: hidden_dim={hidden_dim}, 参数量={kan_params}")
    diff_pct = abs(kan_params - target_params) / target_params * 100 if target_params > 0 else 0
    logging.info(f"参数差异: {abs(kan_params - target_params)} ({diff_pct:.1f}%)")

    results = []
    all_param_counts = []

    logging.info(f"开始 {NUM_RUNS} 次重复实验。模型: KAN")

    for i in range(NUM_RUNS):
        run_seed = START_SEED + i
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)

        logging.info(f"\n{'='*20} Run {i+1}/{NUM_RUNS} (Seed: {run_seed}) {'='*20}")

        # 加载数据
        train_ds, val_ds, test_ds, dims = load_data_fn(run_seed)
        n_features, n_classes = dims

        train_loader = DataLoader(train_ds, batch_size=hparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=hparams['batch_size'], shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=hparams['batch_size'], shuffle=False)

        # 初始化模型
        model, _ = get_aligned_kan(n_features, n_classes, target_params, num_layers=2)
        all_param_counts.append(count_parameters(model))

        # 训练
        model = train_model(model, train_loader, val_loader, hparams, run_id=i+1)

        # 测试
        test_acc = evaluate_model(model, test_loader)
        results.append(test_acc)
        logging.info(f"Run {i+1} 完成。测试集准确率: {test_acc:.4f}")

    # 汇总结果
    mean_acc = np.mean(results)
    std_acc = np.std(results)
    mean_params = np.mean(all_param_counts)

    logging.info(f"\n{'='*20} 实验总结 {'='*20}")
    logging.info(f"KAN 隐藏层维度: {hidden_dim}")
    logging.info(f"平均参数量: {mean_params:.0f}")
    logging.info(f"平均准确率: {mean_acc:.4f}")
    logging.info(f"标准差: {std_acc:.4f}")
    logging.info(f"详细数据: {results}")

    # 保存结果
    output_data = {
        "model": "KAN",
        "dataset": dataset_name,
        "hyperparameters": hparams,
        "model_config": {
            "hidden_dim": hidden_dim,
            "num_layers": 2,
            "grid_size": 10,
            "spline_order": 3,
            "target_params": target_params,
            "actual_params": mean_params
        },
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "all_accuracies": results
    }

    with open(output_filename, "w") as f:
        json.dump(output_data, f, indent=4)
    logging.info(f"结果已保存至 {output_filename}")

    return output_data
"""
KAN (Kolmogorov-Arnold Network) 在 Waveform 数据集上的分类实验
与 CFNet 进行公平对比
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import json
import logging
import os
import sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parents[1]
CLASSIFY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CLASSIFY_DIR))


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# 添加项目路径
from cfnet_complicate import CFNet
from data_paths import OPENML_CACHE, HF_CACHE, TORCHVISION_DIR

# 配置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============ KAN 模型定义 ============

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


# ============ 数据加载和训练 ============

def setup_logging():
    """配置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def set_seed(seed):
    """设置随机种子以保证可复现性"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_and_preprocess_data(run_seed):
    """加载、预处理 UCI Waveform 数据集"""
    logging.info(f"正在加载 Waveform (5000) 数据集 (随机种子: {run_seed})...")

    # 1. 加载数据
    waveform = fetch_openml(name='waveform-5000', version=1, as_frame=False, parser='liac-arff', data_home=str(OPENML_CACHE))
    X = waveform.data
    y_str = waveform.target

    # 2. 标签编码
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    # 3. 数据拆分
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=run_seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=run_seed, stratify=y_temp
    )

    # 4. 特征缩放
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 转换为 Tensor
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    logging.info(f"数据准备完成: {n_features} 特征, {n_classes} 类别")
    return train_dataset, val_dataset, test_dataset, (n_features, n_classes)


def train_model(model, train_loader, val_loader, params, run_id):
    """标准的 PyTorch 训练循环"""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    # 进度条
    pbar = tqdm(range(params['epochs']), desc=f"Run {run_id} Training", leave=False)

    for epoch in pbar:
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- 验证阶段 ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                outputs = model(X_val)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_val.size(0)
                val_correct += (predicted == y_val).sum().item()

        val_acc = val_correct / val_total

        # 更新进度条信息
        pbar.set_postfix({'Loss': f"{train_loss/len(train_loader):.4f}", 'Val Acc': f"{val_acc:.4f}"})

        # 学习率调整
        scheduler.step(val_acc)

        # --- 早停机制 & 保存最佳模型 ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= params['early_stopping_patience']:
                break

    # 加载最佳权重
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model


def evaluate_model(model, test_loader):
    """测试集评估"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.numpy())

    return accuracy_score(all_targets, all_preds)


def main():
    setup_logging()

    # 实验配置
    NUM_RUNS = 10
    START_SEED = 42

    # 超参数配置
    hparams = {
        'epochs': 200,
        'batch_size': 128,
        'learning_rate': 0.005,
        'weight_decay': 1e-4,
        'early_stopping_patience': 20
    }

    # 首先获取 CFNet 参数量作为锚点
    logging.info("计算 CFNet 参数量作为对齐基准...")
    temp_train_ds, _, _, (n_features, n_classes) = load_and_preprocess_data(START_SEED)
    cfnet_anchor = CFNet(input_dim=n_features, output_dim=n_classes, depth=5, poly_degree=3)
    target_params = count_parameters(cfnet_anchor)
    logging.info(f"CFNet 参数量: {target_params}")

    # 获取对齐的 KAN
    kan_model, hidden_dim = get_aligned_kan(n_features, n_classes, target_params, num_layers=2)
    kan_params = count_parameters(kan_model)
    logging.info(f"KAN 配置: hidden_dim={hidden_dim}, 参数量={kan_params}")
    logging.info(f"参数差异: {abs(kan_params - target_params)} ({abs(kan_params - target_params)/target_params*100:.1f}%)")

    results = []
    all_param_counts = []

    logging.info(f"开始 {NUM_RUNS} 次重复实验。模型: KAN")

    for i in range(NUM_RUNS):
        run_seed = START_SEED + i
        set_seed(run_seed)

        logging.info(f"\n{'='*20} Run {i+1}/{NUM_RUNS} (Seed: {run_seed}) {'='*20}")

        # 1. 准备数据
        train_ds, val_ds, test_ds, dims = load_and_preprocess_data(run_seed)
        n_features, n_classes = dims

        train_loader = DataLoader(train_ds, batch_size=hparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=hparams['batch_size'], shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=hparams['batch_size'], shuffle=False)

        # 2. 初始化模型（每次重新创建以保证独立）
        model, _ = get_aligned_kan(n_features, n_classes, target_params, num_layers=2)
        all_param_counts.append(count_parameters(model))

        # 3. 训练
        model = train_model(model, train_loader, val_loader, hparams, run_id=i+1)

        # 4. 测试
        test_acc = evaluate_model(model, test_loader)
        results.append(test_acc)

        logging.info(f"Run {i+1} 完成。测试集准确率: {test_acc:.4f}")

    # --- 结果汇总 ---
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

    with open("kan_waveform_results.json", "w") as f:
        json.dump(output_data, f, indent=4)
    logging.info("结果已保存至 kan_waveform_results.json")


if __name__ == "__main__":
    main()
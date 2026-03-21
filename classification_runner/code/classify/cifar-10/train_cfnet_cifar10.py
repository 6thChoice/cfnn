# 文件名: train_cfnet_cifar10.py
# 描述: 使用 cfnet_complicate.py 中定义的 CFNet 模型在 CIFAR-10 数据集上执行10次重复实验

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

import numpy as np
import json
import logging
import os
import sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parents[2]
CLASSIFY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CLASSIFY_DIR))


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from data_paths import OPENML_CACHE, HF_CACHE, TORCHVISION_DIR

# --- 导入你的模型 ---
try:
    from cfnet_complicate import CFNet
except ImportError:
    print("错误: 无法导入 'CFNet'。请确保 'cfnet_complicate.py' 与此脚本位于同一目录。")
    exit()

# 配置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_and_preprocess_data(run_seed):
    """加载 CIFAR-10 数据集并展平"""
    logging.info(f"正在加载 CIFAR-10 数据集 (随机种子: {run_seed})...")
    
    # 1. 加载数据
    # CIFAR10 下载到 ./data 目录
    train_val_set = torchvision.datasets.CIFAR10(root=str(TORCHVISION_DIR / 'cifar10'), train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=str(TORCHVISION_DIR / 'cifar10'), train=False, download=True)

    X_train_val = train_val_set.data
    y_train_val = np.array(train_val_set.targets)
    
    X_test = test_set.data
    y_test = np.array(test_set.targets)
    
    # 2. 预处理
    # a. 缩放 [0, 255] -> [0, 1]
    X_train_val = X_train_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # b. 展平 (N, 32, 32, 3) -> (N, 3072)
    n_features = np.prod(X_train_val.shape[1:])
    X_train_val = X_train_val.reshape(-1, n_features)
    X_test = X_test.reshape(-1, n_features)
    
    n_classes = len(np.unique(y_train_val))

    # 3. 拆分训练集和验证集 (45k / 5k)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=run_seed, stratify=y_train_val
    )

    # 4. 转换为 Tensor
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    logging.info(f"数据准备完成: {n_features} 特征 (Flattened), {n_classes} 类别")
    return train_dataset, val_dataset, test_dataset, (n_features, n_classes)

def train_model(model, train_loader, val_loader, params, run_id):
    model = model.to(DEVICE)
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
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

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

def evaluate_model(model, test_loader):
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
    
    NUM_RUNS = 10
    START_SEED = 42
    
    # 超参数 (针对 CIFAR-10 调整)
    hparams = {
        'depth': 3,                # 深度降低，因为输入维度高，计算量大
        'poly_degree': 3,
        'epochs': 200,
        'batch_size': 256,         # 显存敏感，不要太大
        'learning_rate': 0.001,   # 降低学习率以稳定训练
        'weight_decay': 1e-4,
        'early_stopping_patience': 20
    }

    results = []

    logging.info(f"开始 {NUM_RUNS} 次重复实验。模型: CFNet (Flattened Input)")

    for i in range(NUM_RUNS):
        run_seed = START_SEED + i
        set_seed(run_seed)
        
        logging.info(f"\n{'='*20} Run {i+1}/{NUM_RUNS} (Seed: {run_seed}) {'='*20}")

        train_ds, val_ds, test_ds, dims = load_and_preprocess_data(run_seed)
        n_features, n_classes = dims

        train_loader = DataLoader(train_ds, batch_size=hparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=hparams['batch_size'], shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=hparams['batch_size'], shuffle=False)

        model = CFNet(
            input_dim=n_features, 
            output_dim=n_classes, 
            depth=hparams['depth'], 
            poly_degree=hparams['poly_degree']
        )

        model = train_model(model, train_loader, val_loader, hparams, run_id=i+1)

        test_acc = evaluate_model(model, test_loader)
        results.append(test_acc)
        
        logging.info(f"Run {i+1} 完成。测试集准确率: {test_acc:.4f}")

    mean_acc = np.mean(results)
    std_acc = np.std(results)
    
    logging.info(f"\n{'='*20} 实验总结 {'='*20}")
    logging.info(f"平均准确率: {mean_acc:.4f}")
    logging.info(f"标准差: {std_acc:.4f}")

    output_data = {
        "dataset": "CIFAR-10",
        "model": "CFNet_Complicate",
        "hyperparameters": hparams,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "all_accuracies": results
    }
    
    with open("cfnet_cifar10_results.json", "w") as f:
        json.dump(output_data, f, indent=4)
    logging.info("结果已保存至 cfnet_cifar10_results.json")

if __name__ == "__main__":
    main()

# 文件名: train_hybrid_creditcard.py
# 描述: 使用 hybrid.py 中定义的 HybridRationalNet 模型在 UCI Credit Card 数据集上执行10次重复分类实验

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import json
import logging
import os
import sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parents[2]
CLASSIFY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CLASSIFY_DIR))


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from data_paths import OPENML_CACHE, HF_CACHE, TORCHVISION_DIR

# --- 导入你的模型 ---
try:
    from hybrid import HybridRationalNet
except ImportError:
    print("错误: 无法导入 'HybridRationalNet'。请确保 'hybrid.py' 与此脚本位于同一目录。")
    exit()

# 配置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging():
    # 强制重新配置日志
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
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
    """加载并预处理 Credit Card 数据集"""
    logging.info(f"正在加载 Default of Credit Card Clients 数据集 (随机种子: {run_seed})...")
    
    # 1. 加载数据
    credit_card = fetch_openml(name='default-of-credit-card-clients', version=1, as_frame=True, parser='auto', data_home=str(OPENML_CACHE))
    df = credit_card.frame
    
    # 修复目标列名
    target_column = 'y'
    if target_column not in df.columns:
        if 'default payment next month' in df.columns:
            target_column = 'default payment next month'
        else:
            raise ValueError(f"Target column not found. Columns: {df.columns.tolist()}")

    X = df.drop(columns=target_column)
    y_raw = df[target_column]

    # 2. 标签编码
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # 3. 处理类别特征 (x2:Sex, x3:Education, x4:Marriage)
    categorical_features = ['x2', 'x3', 'x4']
    existing_cat_features = [col for col in categorical_features if col in X.columns]
    
    if existing_cat_features:
        X = pd.get_dummies(X, columns=existing_cat_features, drop_first=True, dtype=float)
    
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    # 4. 数据拆分 (70% 训练, 15% 验证, 15% 测试)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=run_seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=run_seed, stratify=y_temp
    )

    # 5. 特征缩放
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.values)
    X_val = scaler.transform(X_val.values)
    X_test = scaler.transform(X_test.values)

    # 转换为 Tensor
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    logging.info(f"数据准备完成: {n_features} 特征, {n_classes} 类别")
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
    
    # 超参数配置
    hparams = {
        'unit_degree': 3,
        'num_units': 6,             # Credit Card 数据稍微复杂些，增加单元数
        'epochs': 200,
        'batch_size': 256,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'early_stopping_patience': 25
    }

    results = []

    logging.info(f"开始 {NUM_RUNS} 次重复实验。模型: HybridRationalNet")

    for i in range(NUM_RUNS):
        run_seed = START_SEED + i
        set_seed(run_seed)
        
        logging.info(f"\n{'='*20} Run {i+1}/{NUM_RUNS} (Seed: {run_seed}) {'='*20}")

        train_ds, val_ds, test_ds, dims = load_and_preprocess_data(run_seed)
        n_features, n_classes = dims

        train_loader = DataLoader(train_ds, batch_size=hparams['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=hparams['batch_size'], shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=hparams['batch_size'], shuffle=False)

        model = HybridRationalNet(
            input_dim=n_features, 
            output_dim=n_classes, 
            unit_degree=hparams['unit_degree'], 
            num_units=hparams['num_units']
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
        "dataset": "Default of Credit Card Clients",
        "model": "HybridRationalNet",
        "hyperparameters": hparams,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "all_accuracies": results
    }
    
    with open("hybrid_creditcard_results.json", "w") as f:
        json.dump(output_data, f, indent=4)
    logging.info("结果已保存至 hybrid_creditcard_results.json")

if __name__ == "__main__":
    main()

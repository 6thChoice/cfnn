# 文件名: train_cfnet_imdb.py
# 描述: 使用 cfnet_complicate.py 中定义的 CFNet 模型在 IMDb 数据集上执行10次重复实验

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


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from datasets import load_dataset
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

def load_and_preprocess_data(run_seed, max_features=5000):
    """加载 IMDb 数据集并使用 TF-IDF 向量化"""
    logging.info(f"正在从 Hugging Face Hub 加载 IMDb 数据集 (种子: {run_seed})...")
    
    # 1. 加载
    try:
        dataset = load_dataset('imdb', cache_dir=str(HF_CACHE))
    except Exception as e:
        logging.error(f"加载数据集失败: {e}")
        exit()

    # 2. 合并 Train 和 Test
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']

    # 显式转换为 list 进行拼接
    X_text_all = list(train_texts) + list(test_texts)
    y_all = np.array(list(train_labels) + list(test_labels))

    # 3. 洗牌
    X_text_all, y_all = shuffle(X_text_all, y_all, random_state=run_seed)
    
    # 4. 拆分 Train/Val/Test (0.7 / 0.15 / 0.15)
    X_train_text, X_temp_text, y_train, y_temp = train_test_split(
        X_text_all, y_all, test_size=0.3, random_state=run_seed, stratify=y_all
    )
    X_val_text, X_test_text, y_val, y_test = train_test_split(
        X_temp_text, y_temp, test_size=0.5, random_state=run_seed, stratify=y_temp
    )

    # 5. TF-IDF 向量化
    logging.info(f"正在进行 TF-IDF 向量化 (Max Features: {max_features})...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2))
    
    X_train = vectorizer.fit_transform(X_train_text).toarray()
    X_val = vectorizer.transform(X_val_text).toarray()
    X_test = vectorizer.transform(X_test_text).toarray()

    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    # 6. 转换为 Tensor
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    logging.info(f"数据准备完成: {n_features} 特征, {n_classes} 类别")
    return train_dataset, val_dataset, test_dataset, (n_features, n_classes)

def train_model(model, train_loader, val_loader, params, run_id):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

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
    
    # 超参数
    hparams = {
        'depth': 4,
        'poly_degree': 3,
        'epochs': 100,
        'batch_size': 128,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'early_stopping_patience': 15,
        'max_features': 5000 # TF-IDF 特征数
    }

    results = []

    logging.info(f"开始 {NUM_RUNS} 次重复实验。模型: CFNet")

    for i in range(NUM_RUNS):
        run_seed = START_SEED + i
        set_seed(run_seed)
        
        logging.info(f"\n{'='*20} Run {i+1}/{NUM_RUNS} (Seed: {run_seed}) {'='*20}")

        train_ds, val_ds, test_ds, dims = load_and_preprocess_data(run_seed, max_features=hparams['max_features'])
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
        "dataset": "IMDb",
        "model": "CFNet_Complicate",
        "hyperparameters": hparams,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "all_accuracies": results
    }
    
    with open("cfnet_imdb_results.json", "w") as f:
        json.dump(output_data, f, indent=4)
    logging.info("结果已保存至 cfnet_imdb_results.json")

if __name__ == "__main__":
    main()
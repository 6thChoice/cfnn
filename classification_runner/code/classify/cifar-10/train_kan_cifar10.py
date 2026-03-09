"""KAN on CIFAR-10 Dataset"""
import sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parents[1]
CLASSIFY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CLASSIFY_DIR))


import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kan_classification_base import run_kan_experiment
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import torchvision
import torch
import numpy as np
import logging
from data_paths import OPENML_CACHE, HF_CACHE, TORCHVISION_DIR

def load_and_preprocess_data(run_seed):
    """加载 CIFAR-10 数据集"""
    logging.info(f"正在加载 CIFAR-10 数据集 (随机种子: {run_seed})...")

    train_val_set = torchvision.datasets.CIFAR10(root=str(TORCHVISION_DIR / 'cifar10'), train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=str(TORCHVISION_DIR / 'cifar10'), train=False, download=True)

    X_train_val = train_val_set.data
    y_train_val = np.array(train_val_set.targets)
    X_test = test_set.data
    y_test = np.array(test_set.targets)

    X_train_val = X_train_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    n_features = np.prod(X_train_val.shape[1:])
    X_train_val = X_train_val.reshape(-1, n_features)
    X_test = X_test.reshape(-1, n_features)

    n_classes = len(np.unique(y_train_val))

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=run_seed, stratify=y_train_val
    )

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    logging.info(f"数据准备完成: {n_features} 特征, {n_classes} 类别")
    return train_dataset, val_dataset, test_dataset, (n_features, n_classes)

if __name__ == "__main__":
    cfnet_hparams = {
        'depth': 3,
        'poly_degree': 3,
        'epochs': 200,
        'batch_size': 256,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'early_stopping_patience': 20
    }
    run_kan_experiment("CIFAR-10", load_and_preprocess_data, cfnet_hparams, "kan_cifar10_results.json")
"""KAN on MAGIC Gamma Telescope Dataset"""
import sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parents[1]
CLASSIFY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CLASSIFY_DIR))


import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kan_classification_base import run_kan_experiment
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset
import torch
import numpy as np
import logging
from data_paths import OPENML_CACHE, HF_CACHE, TORCHVISION_DIR

def load_and_preprocess_data(run_seed):
    """加载 MAGIC Telescope 数据集"""
    logging.info(f"正在加载 MAGIC Gamma Telescope 数据集 (随机种子: {run_seed})...")

    magic = fetch_openml(name='MagicTelescope', version=1, as_frame=False, parser='liac-arff', data_home=str(OPENML_CACHE))
    X = magic.data
    y_str = magic.target

    le = LabelEncoder()
    y = le.fit_transform(y_str)

    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=run_seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=run_seed, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    logging.info(f"数据准备完成: {n_features} 特征, {n_classes} 类别")
    return train_dataset, val_dataset, test_dataset, (n_features, n_classes)

if __name__ == "__main__":
    cfnet_hparams = {
        'depth': 5,
        'poly_degree': 3,
        'epochs': 200,
        'batch_size': 128,
        'learning_rate': 0.005,
        'weight_decay': 1e-4,
        'early_stopping_patience': 20
    }
    run_kan_experiment("MAGIC", load_and_preprocess_data, cfnet_hparams, "kan_magic_results.json")
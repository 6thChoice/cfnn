"""KAN on Credit Card Dataset"""
import sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parents[2]
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
import pandas as pd
import logging
from data_paths import OPENML_CACHE, HF_CACHE, TORCHVISION_DIR

def load_and_preprocess_data(run_seed):
    """加载 Credit Card 数据集"""
    logging.info(f"正在加载 Default of Credit Card Clients 数据集 (随机种子: {run_seed})...")

    credit_card = fetch_openml(name='default-of-credit-card-clients', version=1, as_frame=True, parser='auto', data_home=str(OPENML_CACHE))
    df = credit_card.frame

    target_column = 'y'
    if target_column not in df.columns:
        if 'default payment next month' in df.columns:
            target_column = 'default payment next month'
        else:
            raise ValueError(f"Target column not found.")

    X = df.drop(columns=target_column)
    y_raw = df[target_column]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    categorical_features = ['x2', 'x3', 'x4']
    existing_cat_features = [col for col in categorical_features if col in X.columns]
    if existing_cat_features:
        X = pd.get_dummies(X, columns=existing_cat_features, drop_first=True, dtype=float)

    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=run_seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=run_seed, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.values)
    X_val = scaler.transform(X_val.values)
    X_test = scaler.transform(X_test.values)

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
    run_kan_experiment("CreditCard", load_and_preprocess_data, cfnet_hparams, "kan_creditcard_results.json")

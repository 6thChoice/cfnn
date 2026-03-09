"""KAN on Quora Dataset"""
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
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import TensorDataset
from datasets import load_dataset
import torch
import numpy as np
import logging
from data_paths import OPENML_CACHE, HF_CACHE, TORCHVISION_DIR

def load_and_preprocess_data(run_seed, subset_size=50000, max_features=5000):
    """加载 Quora 数据集"""
    logging.info(f"正在从 Hugging Face Hub 加载 Quora 数据集 (种子: {run_seed})...")

    try:
        dataset = load_dataset("UKPLab/insincere-questions", cache_dir=str(HF_CACHE))
    except Exception as e:
        logging.error(f"加载数据集失败: {e}")
        exit()

    train_val_raw = dataset['train']
    if 'test' in dataset:
        test_raw = dataset['test']
        logging.info("使用官方 'test' 分割作为测试集。")
    else:
        test_raw = dataset['validation']
        logging.warning("未找到 'test' 分割，使用 'validation' 作为测试集。")

    X_train_val_text = train_val_raw['text']
    y_train_val = np.array(train_val_raw['labels'])
    X_test_text = test_raw['text']
    y_test = np.array(test_raw['labels'])

    if subset_size and subset_size < len(X_train_val_text):
        logging.info(f"使用子集: {subset_size} 条训练数据")
        indices = np.random.RandomState(run_seed).choice(len(X_train_val_text), subset_size, replace=False)
        X_train_val_text = [X_train_val_text[i] for i in indices]
        y_train_val = y_train_val[indices]

        test_subset_size = int(subset_size * 0.2)
        if test_subset_size < len(X_test_text):
            test_indices = np.random.RandomState(run_seed).choice(len(X_test_text), test_subset_size, replace=False)
            X_test_text = [X_test_text[i] for i in test_indices]
            y_test = y_test[test_indices]

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_train_val_text, y_train_val, test_size=0.15, random_state=run_seed, stratify=y_train_val
    )

    logging.info(f"正在进行 TF-IDF 向量化 (Max Features: {max_features})...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2))

    X_train = vectorizer.fit_transform(X_train_text).toarray()
    X_val = vectorizer.transform(X_val_text).toarray()
    X_test = vectorizer.transform(X_test_text).toarray()

    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    logging.info(f"数据准备完成: {n_features} 特征, {n_classes} 类别")
    return train_dataset, val_dataset, test_dataset, (n_features, n_classes)

if __name__ == "__main__":
    cfnet_hparams = {
        'depth': 4,
        'poly_degree': 3,
        'epochs': 100,
        'batch_size': 256,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'early_stopping_patience': 15
    }

    def load_data_wrapper(run_seed):
        return load_and_preprocess_data(run_seed, subset_size=50000, max_features=5000)

    run_kan_experiment("Quora", load_data_wrapper, cfnet_hparams, "kan_quora_results.json")
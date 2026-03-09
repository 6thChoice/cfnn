"""KAN on IMDB Sentiment Dataset"""
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
from sklearn.utils import shuffle
import torch
import numpy as np
import logging
from data_paths import OPENML_CACHE, HF_CACHE, TORCHVISION_DIR

def load_and_preprocess_data(run_seed, max_features=5000):
    """加载 IMDb 数据集"""
    logging.info(f"正在从 Hugging Face Hub 加载 IMDb 数据集 (种子: {run_seed})...")

    try:
        dataset = load_dataset('imdb', cache_dir=str(HF_CACHE))
    except Exception as e:
        logging.error(f"加载数据集失败: {e}")
        exit()

    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']

    X_text_all = list(train_texts) + list(test_texts)
    y_all = np.array(list(train_labels) + list(test_labels))

    X_text_all, y_all = shuffle(X_text_all, y_all, random_state=run_seed)

    X_train_text, X_temp_text, y_train, y_temp = train_test_split(
        X_text_all, y_all, test_size=0.3, random_state=run_seed, stratify=y_all
    )
    X_val_text, X_test_text, y_val, y_test = train_test_split(
        X_temp_text, y_temp, test_size=0.5, random_state=run_seed, stratify=y_temp
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
        'batch_size': 128,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'early_stopping_patience': 15
    }

    # 使用包装函数来传递max_features
    def load_data_wrapper(run_seed):
        return load_and_preprocess_data(run_seed, max_features=5000)

    run_kan_experiment("IMDB", load_data_wrapper, cfnet_hparams, "kan_imdb_results.json")
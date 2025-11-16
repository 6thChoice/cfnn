# 文件名: train_imdb_hf_repeated.py
# 描述: 修复了 Hugging Face datasets 列拼接问题的脚本

import logging
import numpy as np
import json
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import torch

import sys
sys.path.append("/home/zxc/CodeBase/cofrnet")
# --- 从您创建的文件中导入分类器类 ---
from cfnet_classifier_moe_iterative import CoFrNetClassifier_MoE

def setup_logging():
    """配置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_and_preprocess_data(run_seed, max_features=5000):
    """
    使用 Hugging Face datasets 加载、拆分和预处理 IMDb 数据集。
    
    Args:
        run_seed (int): 用于数据拆分的随机种子。
        max_features (int): TF-IDF 向量的最大维度（词汇表大小）。
    """
    logging.info(f"正在从 Hugging Face Hub 加载 'imdb' 数据集 (随机种子: {run_seed})...")
    
    # 1. 从 Hugging Face Hub 加载数据集
    dataset = load_dataset('imdb')
    
    # 2. 合并官方的训练集和测试集
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']

    # --- (关键修复) ---
    # 将 'Column' 对象显式转换为 Python 列表 (list) 才能用 '+' 拼接
    # 原代码: X_text_all = np.array(train_texts + test_texts)
    # 原代码: y_all = np.array(train_labels + test_labels)
    X_text_all = np.array(list(train_texts) + list(test_texts))
    y_all = np.array(list(train_labels) + list(test_labels))
    # --- (修复结束) ---

    # 3. 对合并后的完整数据集进行洗牌
    X_text_all, y_all = shuffle(X_text_all, y_all, random_state=run_seed)
    
    n_classes = len(np.unique(y_all))
    logging.info(f"数据集信息: {len(X_text_all)} 评论, {n_classes} 类别。")

    # 4. 数据拆分
    X_train_text, X_temp_text, y_train, y_temp = train_test_split(
        X_text_all, y_all, test_size=0.3, random_state=run_seed, stratify=y_all
    )
    X_val_text, X_test_text, y_val, y_test = train_test_split(
        X_temp_text, y_temp, test_size=0.5, random_state=run_seed, stratify=y_temp
    )
    logging.info(f"数据拆分 -> 训练: {len(X_train_text)}, 验证: {len(X_val_text)}, 测试: {len(X_test_text)}")

    # 5. 文本向量化 (TF-IDF)
    logging.info(f"正在使用 TfidfVectorizer 将文本转换为 {max_features} 维向量...")
    vectorizer = TfidfVectorizer(
        max_features=max_features, 
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    X_train = vectorizer.fit_transform(X_train_text).toarray()
    X_val = vectorizer.transform(X_val_text).toarray()
    X_test = vectorizer.transform(X_test_text).toarray()
    
    n_features = X_train.shape[1]
    logging.info("文本向量化完成。")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (n_features, n_classes)


def main():
    """主执行函数，包含10次重复实验"""
    setup_logging()
    
    num_runs = 10
    all_accuracies = []
    
    tfidf_max_features = 5000 
    
    for i in range(num_runs):
        run_seed = 42 + i
        logging.info(f"\n{'='*25} 开始第 {i+1}/{num_runs} 次实验 (种子: {run_seed}) {'='*25}")
        
        train_data, val_data, test_data, dims = load_and_preprocess_data(run_seed, max_features=tfidf_max_features)
        n_features, n_classes = dims

        hparams = {
            'input_dim': n_features,
            'output_dim': n_classes,
            'shallow_depth_per_cofrnet': 4,
            'polynomial_degree': 3,
            'learning_rate_adam': 0.001,
            'weight_decay': 1e-4,
            'epochs_per_model': 250,
            'batch_size': 128,
            'early_stopping_patience': 25,
            'max_experts': 8
        }
        
        model_save_path = "imdb_hf_classifier_temp.pth"
        log_filepath = "imdb_hf_training_log_temp.json"

        classifier = CoFrNetClassifier_MoE(hparams)
        classifier.train_iterative(
            train_data=train_data,
            val_data=val_data,
            log_filepath=log_filepath,
            model_save_path=model_save_path
        )

        logging.info(f"--- 第 {i+1} 次实验评估 ---")
        best_classifier = CoFrNetClassifier_MoE.load_model(model_save_path)
        
        X_test, y_test = test_data
        y_pred = best_classifier.predict(X_test)

        test_accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"第 {i+1}/{num_runs} 次实验 - 测试集准确率: {test_accuracy:.4f}")
        all_accuracies.append(test_accuracy)

    logging.info(f"\n{'='*25} 所有 {num_runs} 次实验已完成 {'='*25}")
    
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)

    logging.info(f"平均测试准确率: {mean_accuracy:.4f}")
    logging.info(f"准确率标准差: {std_accuracy:.4f}")

    results = {
        "dataset_name": "IMDb (from Hugging Face)",
        "accuracies": all_accuracies,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "num_runs": num_runs,
        "tfidf_max_features": tfidf_max_features,
        "hyperparameters": hparams
    }

    results_filename = "imdb_hf_experiment_results.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"所有实验结果已保存到: {results_filename}")


if __name__ == "__main__":
    main()
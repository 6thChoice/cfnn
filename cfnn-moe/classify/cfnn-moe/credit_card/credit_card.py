# 文件名: train_creditcard_repeated.py
# 描述: 修复了列名问题的 UCI Default of Credit Card Clients 分类任务脚本

import logging
import numpy as np
import json
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
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

def load_and_preprocess_data(run_seed):
    """
    加载、拆分和预处理 UCI Default of Credit Card Clients 数据集。
    
    Args:
        run_seed (int): 用于数据拆分的随机种子。
    """
    logging.info(f"正在加载 Default of Credit Card Clients 数据集 (随机种子: {run_seed})...")
    
    # 1. 从 OpenML 加载数据集
    credit_card = fetch_openml(name='default-of-credit-card-clients', version=1, as_frame=True, parser='auto')
    df = credit_card.frame
    
    # --- (关键修复 1) ---
    # 将目标列名从 'default payment next month' 修改为实际的列名 'y'
    target_column = 'y'
    if target_column not in df.columns:
        # 添加一个备用检查，以防万一
        raise ValueError(f"Target column '{target_column}' not found in DataFrame. Available columns: {df.columns.tolist()}")

    # 2. 特征和目标分离
    X = df.drop(columns=target_column)
    y_raw = df[target_column]
    # --- (修复结束 1) ---

    # 3. 标签编码
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # --- (关键修复 2) ---
    # 将类别特征的名称更新为 'x2', 'x3', 'x4'
    categorical_features = ['x2', 'x3', 'x4']
    # --- (修复结束 2) ---
    
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True, dtype=float)
    
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    logging.info(f"数据集信息: {n_samples} 样本, {n_features} 特征 (独热编码后), {n_classes} 类别。")

    # 5. 数据拆分
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=run_seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=run_seed, stratify=y_temp
    )
    logging.info(f"数据拆分 -> 训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    # 6. 特征缩放
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.values)
    X_val = scaler.transform(X_val.values)
    X_test = scaler.transform(X_test.values)
    logging.info("特征已使用 StandardScaler 进行缩放。")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (n_features, n_classes)


def main():
    """主执行函数，包含10次重复实验"""
    setup_logging()
    
    num_runs = 10
    all_accuracies = []
    
    for i in range(num_runs):
        run_seed = 42 + i
        logging.info(f"\n{'='*25} 开始第 {i+1}/{num_runs} 次实验 (种子: {run_seed}) {'='*25}")
        
        train_data, val_data, test_data, dims = load_and_preprocess_data(run_seed)
        n_features, n_classes = dims

        hparams = {
            'input_dim': n_features,
            'output_dim': n_classes,
            'shallow_depth_per_cofrnet': 4,
            'polynomial_degree': 3,
            'learning_rate_adam': 0.001,
            'weight_decay': 1e-5,
            'epochs_per_model': 250,
            'batch_size': 256,
            'early_stopping_patience': 25,
            'max_experts': 8
        }
        
        model_save_path = "creditcard_classifier_temp.pth"
        log_filepath = "creditcard_training_log_temp.json"

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
        "dataset_name": "Default of Credit Card Clients",
        "accuracies": all_accuracies,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "num_runs": num_runs,
        "hyperparameters": hparams
    }

    results_filename = "creditcard_experiment_results.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"所有实验结果已保存到: {results_filename}")


if __name__ == "__main__":
    main()
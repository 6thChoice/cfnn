# 文件名: train_magic_repeated.py
# 描述: 重复10次 UCI MAGIC Gamma Telescope 分类任务，并记录每次的测试准确率到 JSON 文件

import logging
import numpy as np
import json
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
    加载、拆分和预处理 UCI MAGIC Gamma Telescope 数据集。
    
    Args:
        run_seed (int): 用于数据拆分的随机种子，确保每次运行的数据集不同。
    """
    logging.info(f"正在加载 MAGIC Gamma Telescope 数据集 (随机种子: {run_seed})...")
    
    # 1. 从 OpenML 加载数据集
    # 'MagicTelescope' 是该数据集在 OpenML 上的标准名称
    magic = fetch_openml(name='MagicTelescope', version=1, as_frame=False, parser='liac-arff')
    X = magic.data
    y_str = magic.target

    # 2. 标签编码：将字符串标签 ('g', 'h') 转换为整数 (0, 1)
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    logging.info(f"数据集信息: {n_samples} 样本, {n_features} 特征, {n_classes} 类别。")

    # 3. 数据拆分 (70% 训练, 15% 验证, 15% 测试)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=run_seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=run_seed, stratify=y_temp
    )
    logging.info(f"数据拆分 -> 训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    # 4. 特征缩放：对于这类物理测量数据至关重要
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
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
        
        # 加载数据
        train_data, val_data, test_data, dims = load_and_preprocess_data(run_seed)
        n_features, n_classes = dims

        # 定义模型超参数 (与 Waveform 任务保持一致作为起点)
        hparams = {
            'input_dim': n_features,            # -> 10 for MAGIC dataset
            'output_dim': n_classes,            # -> 2 for MAGIC dataset
            'shallow_depth_per_cofrnet': 4,
            'polynomial_degree': 4,
            'learning_rate_adam': 0.1,
            'weight_decay': 1e-3,
            'epochs_per_model': 500,
            'batch_size': 2560,
            'early_stopping_patience': 100,
            'max_experts': 10
        }
        
        # 定义本次运行的文件路径
        model_save_path = "magic_classifier_temp.pth"
        log_filepath = "magic_training_log_temp.json"

        # 实例化并训练分类器
        classifier = CoFrNetClassifier_MoE(hparams)
        classifier.train_iterative(
            train_data=train_data,
            val_data=val_data,
            log_filepath=log_filepath,
            model_save_path=model_save_path
        )

        # --- 评估阶段 ---
        logging.info(f"--- 第 {i+1} 次实验评估 ---")
        best_classifier = CoFrNetClassifier_MoE.load_model(model_save_path)
        
        X_test, y_test = test_data
        y_pred = best_classifier.predict(X_test)

        test_accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"第 {i+1}/{num_runs} 次实验 - 测试集准确率: {test_accuracy:.4f}")
        all_accuracies.append(test_accuracy)

    # --- 所有实验结束后，汇总并保存结果 ---
    logging.info(f"\n{'='*25} 所有 {num_runs} 次实验已完成 {'='*25}")
    
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)

    logging.info(f"平均测试准确率: {mean_accuracy:.4f}")
    logging.info(f"准确率标准差: {std_accuracy:.4f}")

    results = {
        "dataset_name": "MAGIC Gamma Telescope",
        "accuracies": all_accuracies,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "num_runs": num_runs,
        "hyperparameters": hparams
    }

    # 保存到 JSON 文件
    results_filename = "magic_experiment_results.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"所有实验结果已保存到: {results_filename}")


if __name__ == "__main__":
    main()
# 文件名: train_cifar10_repeated.py
# 描述: 在 CIFAR-10 数据集上进行图像分类任务（作为模型架构对比实验）

import logging
import numpy as np
import json
import torch
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    加载、拆分和预处理 CIFAR-10 数据集。
    
    Args:
        run_seed (int): 随机种子。
    """
    logging.info(f"正在加载 CIFAR-10 数据集 (随机种子: {run_seed})...")
    
    # 1. 使用 torchvision 下载并加载数据集
    train_val_set = torchvision.datasets.CIFAR10(root='/home/zxc/CodeBase/cofrnet/data', train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root='/home/zxc/CodeBase/cofrnet/data', train=False, download=True)

    # 2. 提取数据和标签为 numpy 数组
    X_train_val = train_val_set.data
    y_train_val = np.array(train_val_set.targets)
    
    X_test = test_set.data
    y_test = np.array(test_set.targets)
    
    # 3. 预处理
    # a. 将像素值从 [0, 255] 缩放到 [0, 1]
    X_train_val = X_train_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # b. 展平图像：将 (N, 32, 32, 3) 转换为 (N, 3072)
    n_features = np.prod(X_train_val.shape[1:])
    X_train_val = X_train_val.reshape(-1, n_features)
    X_test = X_test.reshape(-1, n_features)
    
    n_classes = len(np.unique(y_train_val))
    logging.info(f"数据集信息: 图像已展平为 {n_features} 维向量, 共 {n_classes} 个类别。")

    # 4. 从官方训练集中拆分出我们自己的训练集和验证集 (45k / 5k)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=run_seed, stratify=y_train_val
    )
    logging.info(f"数据拆分 -> 训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (n_features, n_classes)


def main():
    """主执行函数，包含重复实验"""
    setup_logging()
    
    # 警告：CIFAR-10 实验会非常慢，建议先将 num_runs 设为 1
    num_runs = 10
    all_accuracies = []
    
    for i in range(num_runs):
        run_seed = 42 + i
        logging.info(f"\n{'='*25} 开始第 {i+1}/{num_runs} 次实验 (种子: {run_seed}) {'='*25}")
        
        train_data, val_data, test_data, dims = load_and_preprocess_data(run_seed)
        n_features, n_classes = dims

        hparams = {
            'input_dim': int(n_features),
            'output_dim': int(n_classes),
            'shallow_depth_per_cofrnet': 3, # 维度很高，适当降低模型复杂度
            'polynomial_degree': 3,
            'learning_rate_adam': 0.0005, # 学习率稍低可能更稳定
            'weight_decay': 1e-4,
            'epochs_per_model': 500, 
            'batch_size': 1280,      # 如果显存不足 (OOM Error)，减小此值
            'early_stopping_patience': 50,
            'max_experts': 10        # 减少专家上限，因为每个专家都很大
        }
        
        model_save_path = "cifar10_classifier_temp.pth"
        log_filepath = "cifar10_training_log_temp.json"

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
        "dataset_name": "CIFAR-10",
        "model_architecture_note": "FCN-style on flattened images (not recommended)",
        "accuracies": all_accuracies,
        "mean_accuracy": float(mean_accuracy),
        "std_accuracy": float(std_accuracy),
        "num_runs": num_runs,
        "hyperparameters": hparams
    }

    results_filename = "cifar10_experiment_results.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"所有实验结果已保存到: {results_filename}")


if __name__ == "__main__":
    main()
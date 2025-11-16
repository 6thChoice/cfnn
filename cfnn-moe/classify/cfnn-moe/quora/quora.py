# 文件名: train_quora_insincere_hf_repeated.py
# 描述: 修复了子集抽样问题的 Quora Insincere Questions 脚本

import logging
import numpy as np
import json
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
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

def load_and_preprocess_data(run_seed, use_subset=False, subset_size=50000, max_features=5000):
    """
    加载、拆分和预处理 Quora Insincere Questions 数据集。
    
    Args:
        run_seed (int): 随机种子。
        use_subset (bool): 是否仅使用数据集的一个子集进行快速测试。
        subset_size (int): 如果使用子集，其大小。
        max_features (int): TF-IDF 向量的最大维度。
    """
    logging.info(f"正在从 Hugging Face Hub 加载 'UKPLab/insincere-questions' 数据集 (种子: {run_seed})...")
    
    dataset = load_dataset("UKPLab/insincere-questions")
    
    train_val_set = dataset['train']
    final_test_set = dataset['validation'] 

    X_train_val_text = train_val_set['text']
    y_train_val = np.array(train_val_set['labels'])
    
    X_test_text = final_test_set['text']
    y_test = np.array(final_test_set['labels'])
    
    if use_subset:
        logging.warning(f"注意：正在使用 {subset_size} 条数据的子集进行快速实验！")
        
        # 确保训练子集大小不超过可用数据量
        actual_subset_size = min(subset_size, len(X_train_val_text))
        train_val_indices = np.random.choice(len(X_train_val_text), actual_subset_size, replace=False)
        X_train_val_text = [X_train_val_text[i] for i in train_val_indices]
        y_train_val = y_train_val[train_val_indices]
        
        # --- (关键修复) ---
        # 计算期望的测试子集大小
        desired_test_subset_size = int(actual_subset_size * 0.2)
        
        # 确保我们不会尝试抽取比现有测试集更多的样本
        if desired_test_subset_size >= len(X_test_text):
            logging.warning(f"期望的测试子集大小 ({desired_test_subset_size}) 大于或等于可用数量 ({len(X_test_text)})。将使用完整的测试集。")
            # 在这种情况下，我们直接使用完整的测试集，不进行抽样
        else:
            # 只有在安全的情况下才进行抽样
            test_indices = np.random.choice(len(X_test_text), desired_test_subset_size, replace=False)
            X_test_text = [X_test_text[i] for i in test_indices]
            y_test = y_test[test_indices]
        # --- (修复结束) ---
        
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_train_val_text, y_train_val, test_size=0.15, random_state=run_seed, stratify=y_train_val
    )
    
    n_classes = len(np.unique(y_train_val))
    logging.info(f"数据拆分 -> 新训练集: {len(X_train_text)}, 新验证集: {len(X_val_text)}, 最终测试集: {len(y_test)}")

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
    logging.info(f"文本向量化完成。输入维度: {n_features}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (n_features, n_classes)


def main():
    """主执行函数，包含重复实验"""
    setup_logging()
    
    USE_SUBSET_FOR_TESTING = True 
    num_runs = 10
    
    all_accuracies = []
    tfidf_max_features = 5000 
    
    for i in range(num_runs):
        run_seed = 42 + i
        logging.info(f"\n{'='*25} 开始第 {i+1}/{num_runs} 次实验 (种子: {run_seed}) {'='*25}")
        
        train_data, val_data, test_data, dims = load_and_preprocess_data(
            run_seed, 
            use_subset=USE_SUBSET_FOR_TESTING, 
            max_features=tfidf_max_features
        )
        n_features, n_classes = dims

        hparams = {
            'input_dim': n_features,
            'output_dim': n_classes,
            'shallow_depth_per_cofrnet': 4,
            'polynomial_degree': 3,
            'learning_rate_adam': 0.001,
            'weight_decay': 1e-4,
            'epochs_per_model': 250,
            'batch_size': 256,
            'early_stopping_patience': 25,
            'max_experts': 8
        }
        
        model_save_path = "quora_insincere_classifier_temp.pth"
        log_filepath = "quora_insincere_training_log_temp.json"

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
        "dataset_name": "Quora Insincere Questions (from Hugging Face)",
        "used_subset": USE_SUBSET_FOR_TESTING,
        "accuracies": all_accuracies,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "num_runs": num_runs,
        "tfidf_max_features": tfidf_max_features,
        "hyperparameters": hparams
    }

    results_filename = "quora_insincere_hf_experiment_results.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"所有实验结果已保存到: {results_filename}")


if __name__ == "__main__":
    main()
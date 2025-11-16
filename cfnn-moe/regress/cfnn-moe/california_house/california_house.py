# 文件名: run_california_experiment.py
# 描述: 使用 cfnet_regressor_moe_iterative 模型在 California Housing 数据集上执行10次回归实验
# 版本: 1.0

import numpy as np
import pandas as pd
import json
import logging
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

# --- 关键步骤: 从您提供的文件中导入模型 ---
import sys
sys.path.append("/home/zxc/CodeBase/cofrnet")
try:
    from cfnet_regressor_moe_iterative import CoFrNetRegressor_MoE
except ImportError:
    print("错误: 无法导入 'CoFrNetRegressor_MoE'。")
    print("请确保 'cfnet_regressor_moe_iterative.py' 文件与此脚本位于同一目录中。")
    exit()

# 1. 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 2. 定义模型超参数和实验设置
def get_hparams(input_dim):
    """根据输入维度生成超参数字典"""
    return {
        # 模型结构
        'input_dim': input_dim,
        'output_dim': 1,
        'shallow_depth_per_cofrnet': 3,
        'polynomial_degree': 3,
        
        # 迭代训练参数 (数据集大小适中)
        'max_experts': 10,
        'epochs_per_model': 500,
        
        # 优化器与正则化
        'batch_size': 128,
        'learning_rate_adam': 0.001,
        'weight_decay': 1e-6,
        
        # 早停
        'early_stopping_patience': 50
    }

# 3. 数据加载函数
def load_data():
    """使用 scikit-learn 加载 California Housing 数据集"""
    logging.info("正在加载 California Housing 数据集...")
    try:
        housing = fetch_california_housing()
    except Exception as e:
        logging.error(f"无法加载数据集: {e}")
        return None, None
    
    X = housing.data
    y = housing.target
    
    logging.info(f"数据加载完成。特征维度: {X.shape}")
    return X, y

# 4. 设置随机种子的函数
def set_seeds(seed):
    """为 Numpy 和 PyTorch 设置随机种子以保证可复现性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# 5. 主实验流程
def run_experiment():
    """执行完整的10次重复实验"""
    
    X, y = load_data()
    if X is None:
        return

    # 对目标变量进行对数变换以稳定训练
    y_log = np.log1p(y)

    hparams = get_hparams(X.shape[1])
    test_rmses = []
    num_runs = 10
    start_seed = 42

    logging.info(f"--- 开始进行 {num_runs} 次重复实验 (起始种子: {start_seed}) ---")
    logging.info(f"模型参数: {json.dumps(hparams, indent=2)}")

    for i in range(num_runs):
        current_seed = start_seed + i
        logging.info(f"\n{'='*50}\n--- 第 {i+1}/{num_runs} 次实验 (随机种子: {current_seed}) ---\n{'='*50}")

        set_seeds(current_seed)

        # a. 数据分割
        X_train, X_test, y_train_log, y_test_log = train_test_split(
            X, y_log, test_size=0.2, random_state=current_seed
        )

        # b. 特征缩放
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # c. 初始化并训练模型
        model = CoFrNetRegressor_MoE(hparams)
        model_save_path = f"california_moe_model_run_{i+1}.pth"
        log_filepath = "training_california.log"
        
        model.train_iterative(
            train_data=(X_train_scaled, y_train_log),
            val_data=(X_test_scaled, y_test_log),
            log_filepath=log_filepath,
            model_save_path=model_save_path
        )

        # d. 在测试集上评估
        logging.info("正在测试集上评估最终模型...")
        y_pred_log = model.predict(X_test_scaled).flatten()
        
        # 将预测值和真实值都转换回原始尺度
        y_pred_original = np.expm1(y_pred_log)
        y_test_original = np.expm1(y_test_log)
        
        # 在原始尺度上计算 RMSE (单位是 $100,000s)
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        
        test_rmses.append(rmse)
        logging.info(f"--- 实验 {i+1} 完成。测试集 RMSE (原始尺度): {rmse:.6f} ---")
        
        if os.path.exists(model_save_path):
            os.remove(model_save_path)

    # 6. 计算最终结果并保存
    mean_rmse = np.mean(test_rmses)
    std_rmse = np.std(test_rmses)

    logging.info(f"\n{'='*50}\n--- 所有实验完成 ---\n{'='*50}")
    logging.info(f"平均 RMSE (原始尺度): {mean_rmse:.6f}")
    logging.info(f"RMSE 标准差 (原始尺度): {std_rmse:.6f}")
    # 原始单位是 $100,000, 所以乘以 100,000 得到美元
    logging.info(f"换算为美元 -> 平均误差: ${mean_rmse*100000:,.2f}, 标准差: ${std_rmse*100000:,.2f}")


    results = {
        "model_name": "CoFrNetRegressor_MoE_Iterative",
        "dataset": "California Housing",
        "evaluation_metric": "RMSE on original target scale (after training on log-transformed target)",
        "hyperparameters": hparams,
        "num_runs": num_runs,
        "seeds_used": list(range(start_seed, start_seed + num_runs)),
        "test_rmses_per_run": test_rmses,
        "mean_test_rmse": mean_rmse,
        "std_test_rmse": std_rmse
    }

    results_filename = "california_housing_experiment_results.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=4)

    logging.info(f"实验结果已成功保存到: {results_filename}")

# 运行主程序
if __name__ == "__main__":
    run_experiment()
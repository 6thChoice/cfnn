# 文件名: run_wine_experiment.py
# 描述: 使用 cfnet_regressor_moe_iterative 模型在 UCI Wine Quality 数据集上执行10次回归实验
# 版本: 2.0 (已添加每次实验设置不同随机种子的功能)

import numpy as np
import pandas as pd
import json
import logging
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# --- 关键步骤: 从您提供的文件中导入模型 ---
# 确保 `cfnet_regressor_moe_iterative.py` 文件与此脚本在同一目录下
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
        
        # 迭代训练参数
        'max_experts': 10,
        'epochs_per_model': 500,
        
        # 优化器与正则化
        'batch_size': 128,
        'learning_rate_adam': 0.1,
        'weight_decay': 1e-3,
        
        # 早停
        'early_stopping_patience': 50
    }

# 3. 数据加载与预处理函数
def load_and_prepare_data():
    """下载并准备 UCI Wine Quality 数据集"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        wine_df = pd.read_csv(url, sep=';')
    except Exception as e:
        logging.error(f"无法下载或读取数据: {e}")
        return None, None

    X = wine_df.drop('quality', axis=1).values
    y = wine_df['quality'].values
    
    return X, y

# --- 新增: 设置随机种子的函数 ---
def set_seeds(seed):
    """为 Numpy 和 PyTorch 设置随机种子以保证可复现性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# 4. 主实验流程
def run_experiment():
    """执行完整的10次重复实验"""
    
    X, y = load_and_prepare_data()
    if X is None:
        return

    hparams = get_hparams(X.shape[1])
    test_rmses = []
    num_runs = 10
    start_seed = 42 # --- 新增: 定义起始种子 ---

    logging.info(f"--- 开始进行 {num_runs} 次重复实验 (起始种子: {start_seed}) ---")
    logging.info(f"模型参数: {json.dumps(hparams, indent=2)}")

    for i in range(num_runs):
        current_seed = start_seed + i
        logging.info(f"\n{'='*50}\n--- 第 {i+1}/{num_runs} 次实验 (随机种子: {current_seed}) ---\n{'='*50}")

        # --- 关键修改: 在每次循环开始时设置种子 ---
        set_seeds(current_seed)

        # a. 数据分割 (使用当前循环的种子)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=current_seed)

        # b. 特征缩放
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # c. 初始化并训练模型
        model = CoFrNetRegressor_MoE(hparams)
        model_save_path = f"wine_moe_model_run_{i+1}.pth"
        log_filepath = "training.log"
        
        model.train_iterative(
            train_data=(X_train_scaled, y_train),
            val_data=(X_test_scaled, y_test),
            log_filepath=log_filepath,
            model_save_path=model_save_path
        )

        # d. 在测试集上评估
        logging.info("正在测试集上评估最终模型...")
        y_pred = model.predict(X_test_scaled)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_rmses.append(rmse)
        logging.info(f"--- 实验 {i+1} 完成。测试集 RMSE: {rmse:.6f} ---")
        
        if os.path.exists(model_save_path):
            os.remove(model_save_path)

    # 5. 计算最终结果并保存
    mean_rmse = np.mean(test_rmses)
    std_rmse = np.std(test_rmses)

    logging.info(f"\n{'='*50}\n--- 所有实验完成 ---\n{'='*50}")
    logging.info(f"平均 RMSE: {mean_rmse:.6f}")
    logging.info(f"RMSE 标准差: {std_rmse:.6f}")

    results = {
        "model_name": "CoFrNetRegressor_MoE_Iterative",
        "dataset": "UCI Wine Quality (Red)",
        "hyperparameters": hparams,
        "num_runs": num_runs,
        "seeds_used": list(range(start_seed, start_seed + num_runs)), # 记录使用的种子
        "test_rmses_per_run": test_rmses,
        "mean_test_rmse": mean_rmse,
        "std_test_rmse": std_rmse
    }

    results_filename = "wine_quality_experiment_results.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=4)

    logging.info(f"实验结果已成功保存到: {results_filename}")

if __name__ == "__main__":
    run_experiment()
# 文件名: run_boston_experiment.py
# 描述: 使用 cfnet_regressor_moe_iterative 模型在 UCI Boston Housing 数据集上执行10次回归实验
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

# --- 关键步骤: 从您提供的文件中导入模型 ---
import sys
sys.path.append("/home/zxc/CodeBase/cofrnet")
try:
    from cfnet_regressor_moe_iterative import CoFrNetRegressor_MoE
except ImportError:
    print("错误: 无法导入 'CoFrNetRegressor_MoE'。")
    print("请确保 'cfnet_regressor_moe_iterative.py' 文件与此脚本位于同一目录中。")
    exit()

# #############################################################################
# #                              重要道德声明                               #
# # 该数据集包含基于种族的特征，这引发了严重的道德问题。                  #
# # 在 scikit-learn 中已被弃用。此脚本仅用于技术演示目的。                  #
# # 在实际应用中，请勿使用该数据集或其中包含的偏见特征。                     #
# #############################################################################


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
        
        # 迭代训练参数 (针对小数据集进行调整)
        'max_experts': 10,
        'epochs_per_model': 500,
        
        # 优化器与正则化
        'batch_size': 64,  # 数据集小，使用较小的批量大小
        'learning_rate_adam': 0.001,
        'weight_decay': 1e-5,
        
        # 早停
        'early_stopping_patience': 50
    }

# 3. 数据加载与预处理函数
def load_and_prepare_data():
    """下载并准备 Boston Housing 数据集"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    
    logging.info(f"正在从 {url} 下载数据集...")
    try:
        # 数据以空格分隔，且没有列名
        df = pd.read_csv(url, header=None, delim_whitespace=True)
        
        # 根据数据集描述手动添加列名
        column_names = [
            'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
            'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
        ]
        df.columns = column_names

    except Exception as e:
        logging.error(f"无法下载或读取数据: {e}")
        return None, None

    # 定义特征 (X) 和目标 (y)
    X = df.drop('MEDV', axis=1).values
    y = df['MEDV'].values
    
    logging.info(f"数据加载和预处理完成。特征维度: {X.shape}, 目标维度: {y.shape}")
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
    
    X, y = load_and_prepare_data()
    if X is None:
        return

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=current_seed)

        # b. 特征缩放
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # c. 初始化并训练模型
        model = CoFrNetRegressor_MoE(hparams)
        model_save_path = f"boston_moe_model_run_{i+1}.pth"
        log_filepath = "training_boston.log"
        
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
        logging.info(f"--- 实验 {i+1} 完成。测试集 RMSE: {rmse:.4f} ---")
        
        if os.path.exists(model_save_path):
            os.remove(model_save_path)

    # 6. 计算最终结果并保存
    mean_rmse = np.mean(test_rmses)
    std_rmse = np.std(test_rmses)

    logging.info(f"\n{'='*50}\n--- 所有实验完成 ---\n{'='*50}")
    logging.info(f"平均 RMSE: {mean_rmse:.4f}")
    logging.info(f"RMSE 标准差: {std_rmse:.4f}")

    results = {
        "model_name": "CoFrNetRegressor_MoE_Iterative",
        "dataset": "UCI Boston Housing",
        "ethical_warning": "This dataset contains features based on race and has been deprecated by scikit-learn due to ethical concerns.",
        "hyperparameters": hparams,
        "num_runs": num_runs,
        "seeds_used": list(range(start_seed, start_seed + num_runs)),
        "test_rmses_per_run": test_rmses,
        "mean_test_rmse": mean_rmse,
        "std_test_rmse": std_rmse
    }

    results_filename = "boston_housing_experiment_results.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=4)

    logging.info(f"实验结果已成功保存到: {results_filename}")

# 运行主程序
if __name__ == "__main__":
    run_experiment()
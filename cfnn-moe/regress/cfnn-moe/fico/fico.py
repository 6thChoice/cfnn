# 文件名: run_fico_experiment.py
# 描述: 使用 cfnet_regressor_moe_iterative 模型在 FICO Loan 数据集上执行10次回归实验
# 版本: 1.0

import numpy as np
import pandas as pd
import json
import logging
import os
import torch
import zipfile

# --- 数据集下载所需的库 ---
try:
    import kagglehub
except ImportError:
    print("错误: 'kagglehub' 库未安装。请运行 'pip install kagglehub'")
    exit()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --- 从您提供的文件中导入模型 ---
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

# 2. 定义模型超参数
def get_hparams(input_dim):
    """根据输入维度生成超参数字典"""
    return {
        'input_dim': input_dim,
        'output_dim': 1,
        'shallow_depth_per_cofrnet': 3,
        'polynomial_degree': 3,
        'max_experts': 10,
        'epochs_per_model': 1000,
        'batch_size': 128,
        'learning_rate_adam': 10,
        'weight_decay': 1e-2,
        'early_stopping_patience': 200
    }

# 3. 数据加载与准备函数
def load_and_prepare_data():
    """使用 kagglehub 下载、解压并准备 FICO Loan 数据集"""
    logging.info("正在下载 FICO Loan 数据集...")
    
    path = "/home/zxc/CodeBase/cofrnet/cfnn-moe/regress/cfnn-moe/fico/archive.zip"
    # 解压文件并读取 CSV
    csv_filename = "loan_data.csv"
    try:
        with zipfile.ZipFile(path, 'r') as zip_ref:
            with zip_ref.open(csv_filename) as f:
                df = pd.read_csv(f)
        logging.info("数据集下载和解压成功。")
    except Exception as e:
        logging.error(f"解压或读取 CSV 文件时出错: {e}")
        return None, None
    
    # 检查是否有缺失值 (这个特定数据集没有，但这是一个好习惯)
    if df.isnull().sum().sum() > 0:
        logging.warning("数据集中检测到缺失值。预处理流水线将处理它们。")

    # 定义特征 (X) 和目标 (y)
    # 目标是 'fico'
    X = df.drop('fico', axis=1)
    y = df['fico']
    
    logging.info(f"数据加载完成。原始特征维度: {X.shape}")
    return X, y

# 4. 设置随机种子的函数
def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# 5. 主实验流程
def run_experiment():
    """执行完整的10次重复实验"""
    
    X, y = load_and_prepare_data()
    if X is None: return

    test_rmses = []
    num_runs = 10
    start_seed = 42

    logging.info(f"--- 开始进行 {num_runs} 次重复实验 (起始种子: {start_seed}) ---")

    for i in range(num_runs):
        current_seed = start_seed + i
        logging.info(f"\n{'='*50}\n--- 第 {i+1}/{num_runs} 次实验 (随机种子: {current_seed}) ---\n{'='*50}")

        set_seeds(current_seed)

        # a. 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=current_seed
        )

        # b. 定义特征预处理流水线
        # 'purpose' 是唯一的类别特征
        categorical_features = ['purpose']
        numerical_features = X_train.drop(columns=categorical_features).columns

        # 为数值特征创建流水线 (补缺 + 缩放)
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # 为类别特征创建流水线 (补缺 + One-Hot编码)
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # 使用 ColumnTransformer 组合两种流水线
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
        # c. 应用预处理
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        input_dim = X_train_processed.shape[1]
        logging.info(f"预处理后输入维度: {input_dim}")
        
        # d. 初始化并训练模型
        hparams = get_hparams(input_dim)
        if i == 0:
            logging.info(f"模型参数: {json.dumps(hparams, indent=2)}")

        model = CoFrNetRegressor_MoE(hparams)
        model_save_path = f"fico_moe_model_run_{i+1}.pth"
        log_filepath = "training_fico.log"
        
        model.train_iterative(
            train_data=(X_train_processed, y_train.values),
            val_data=(X_test_processed, y_test.values),
            log_filepath=log_filepath,
            model_save_path=model_save_path
        )

        # e. 在测试集上评估
        logging.info("正在测试集上评估最终模型...")
        y_pred = model.predict(X_test_processed).flatten()
        
        # 在原始 FICO 分数尺度上计算 RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        test_rmses.append(rmse)
        logging.info(f"--- 实验 {i+1} 完成。测试集 RMSE: {rmse:.4f} FICO 分数 ---")
        
        if os.path.exists(model_save_path):
            os.remove(model_save_path)

    # 6. 计算最终结果并保存
    mean_rmse = np.mean(test_rmses)
    std_rmse = np.std(test_rmses)

    logging.info(f"\n{'='*50}\n--- 所有实验完成 ---\n{'='*50}")
    logging.info(f"平均 RMSE: {mean_rmse:.4f} FICO 分数")
    logging.info(f"RMSE 标准差: {std_rmse:.4f} FICO 分数")

    results = {
        "model_name": "CoFrNetRegressor_MoE_Iterative",
        "dataset": "FICO Loan Data",
        "evaluation_metric": "RMSE on FICO score",
        "hyperparameters": get_hparams(input_dim),
        "num_runs": num_runs,
        "seeds_used": list(range(start_seed, start_seed + num_runs)),
        "test_rmses_per_run": test_rmses,
        "mean_test_rmse": mean_rmse,
        "std_test_rmse": std_rmse
    }

    results_filename = "fico_experiment_results.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=4)

    logging.info(f"实验结果已成功保存到: {results_filename}")

if __name__ == "__main__":
    run_experiment()
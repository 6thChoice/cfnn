# 文件名: run_news_experiment.py
# 描述: 使用 cfnet_regressor_moe_iterative 模型在 UCI Online News Popularity 数据集上执行10次回归实验
# 版本: 1.0

import numpy as np
import pandas as pd
import json
import logging
import os
import torch
import requests
import zipfile
import io
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
        
        # 迭代训练参数 (针对更大数据集进行调整)
        'max_experts': 10,              # 可以尝试更多的专家
        'epochs_per_model': 500,         # 每轮训练的纪元数减少，因为数据集更大
        
        # 优化器与正则化
        'batch_size': 256,              # 增加批量大小
        'learning_rate_adam': 0.1,
        'weight_decay': 1e-3,           # 稍微调整正则化
        
        # 早停
        'early_stopping_patience': 50   # 更敏感的早停
    }

# 3. 数据加载与预处理函数
def load_and_prepare_data():
    """下载、解压并准备 Online News Popularity 数据集"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip"
    zip_filename = "OnlineNewsPopularity.zip"
    csv_filename = "OnlineNewsPopularity/OnlineNewsPopularity.csv"
    
    # 下载文件
    if not os.path.exists(zip_filename):
        logging.info(f"正在从 {url} 下载数据集...")
        try:
            r = requests.get(url)
            r.raise_for_status()
            with open(zip_filename, "wb") as f:
                f.write(r.content)
            logging.info("下载完成。")
        except Exception as e:
            logging.error(f"无法下载数据: {e}")
            return None, None
            
    # 解压文件
    logging.info("正在解压文件...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall('.')
    
    # 加载数据
    try:
        df = pd.read_csv(csv_filename)
        # 清理列名中可能存在的前导/后导空格
        df.columns = df.columns.str.strip()
    except FileNotFoundError:
        logging.error(f"错误: 未在压缩包中找到 '{csv_filename}'")
        return None, None

    # 定义特征 (X) 和目标 (y)
    # 移除 'url' (非数值) 和 'timedelta' (非预测性特征)
    X = df.drop(['url', 'timedelta', 'shares'], axis=1).values
    
    # --- 关键预处理 ---
    # 目标变量 'shares' 具有长尾分布，使用 log1p 变换使其更接近正态分布
    # 这有助于模型稳定训练并提高性能
    y = np.log1p(df['shares'].values)
    
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

        # a. 数据分割 (目标 y 已经是 log1p 变换后的)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=current_seed)

        # b. 特征缩放
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # c. 初始化并训练模型
        model = CoFrNetRegressor_MoE(hparams)
        model_save_path = f"news_moe_model_run_{i+1}.pth"
        log_filepath = "training_news.log"
        
        model.train_iterative(
            train_data=(X_train_scaled, y_train),
            val_data=(X_test_scaled, y_test), # 验证集目标也是 log 变换后的
            log_filepath=log_filepath,
            model_save_path=model_save_path
        )

        # d. 在测试集上评估
        logging.info("正在测试集上评估最终模型...")
        y_pred_log = model.predict(X_test_scaled).flatten() # 预测结果是 log(shares + 1)
        
        # --- 关键评估步骤 ---
        # 1. 将预测值和真实值都转换回原始尺度
        #    使用 np.expm1 是 np.log1p 的逆运算
        y_pred_original = np.expm1(y_pred_log)
        y_test_original = np.expm1(y_test)
        
        # 2. 在原始尺度上计算 RMSE
        #    对负预测值进行裁剪，因为分享数不能为负
        y_pred_original[y_pred_original < 0] = 0
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        
        test_rmses.append(rmse)
        logging.info(f"--- 实验 {i+1} 完成。测试集 RMSE (原始尺度): {rmse:.4f} ---")
        
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
        "dataset": "UCI Online News Popularity",
        "hyperparameters": hparams,
        "num_runs": num_runs,
        "seeds_used": list(range(start_seed, start_seed + num_runs)),
        "test_rmses_per_run": test_rmses,
        "mean_test_rmse": mean_rmse,
        "std_test_rmse": std_rmse
    }

    results_filename = "news_popularity_experiment_results.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=4)

    logging.info(f"实验结果已成功保存到: {results_filename}")

# 运行主程序
if __name__ == "__main__":
    run_experiment()
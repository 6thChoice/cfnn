#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import time
import logging
import sys
import json
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader

import sys
sys.path.append("/home/zxc/CodeBase/cofrnet")
from cfnet_regressor import CoFrNetRegressor, RegressionDataset

def setup_logging(log_file='california_housing_run.log'):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger(); root_logger.handlers = []; root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, mode='w'); file_handler.setFormatter(log_formatter); root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout); console_handler.setFormatter(log_formatter); root_logger.addHandler(console_handler)

torch.manual_seed(42)
np.random.seed(42)

def load_california_housing_data():
    logging.info("正在从 Scikit-learn 加载加州住房数据集...")
    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    logging.info(f"数据已分割和标准化。训练: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)}")

    import joblib
    scaler_filename = 'scaler.gz'
    joblib.dump(scaler, scaler_filename)
    logging.info(f"数据标准化缩放器已保存到 {scaler_filename}")

    return X_train, y_train, X_val, y_val, X_test, y_test, X.shape[1], scaler

if __name__ == '__main__':
    test_num = 5
    TASK_NAME = "California_Housing"
    LOG_FILE = f'{TASK_NAME.lower()}_run_{test_num}.log'
    JSON_LOG_FILE = f'{TASK_NAME.lower()}_log_{test_num}.json'
    MODEL_SAVE_PATH = f'best_{TASK_NAME.lower()}_regressor_{test_num}.pth'
    PREDICTION_PLOT_PATH = f'{TASK_NAME.lower()}_predictions_{test_num}.png'
    
    HYPERPARAMETERS = {
        "task_name": TASK_NAME,
        "num_models_max": 300,
        "epochs_per_model": 1000,
        "shallow_depth_per_cofrnet": 4,
        "polynomial_degree": 4,
        "boosting_learning_rate": 0.1,
        "learning_rate_adam": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 1024,
        "early_stopping_patience": 200,
        "output_dim": 1, # 输出维度
    }

    setup_logging(LOG_FILE)
    
    X_train, y_train, X_val, y_val, X_test, y_test, input_dim, scaler = load_california_housing_data()
    
    if X_train is not None:
        # 动态添加数据相关的维度到超参数中
        HYPERPARAMETERS["input_dim"] = input_dim

        logging.info(f"实验超参数: {json.dumps(HYPERPARAMETERS, indent=4)}")

        # 1. 初始化模型
        regressor = CoFrNetRegressor(hparams=HYPERPARAMETERS)

        # 2. 调用训练方法
        start_time = time.time()
        best_rmse = regressor.train(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            test_data=(X_test, y_test),
            log_filepath=JSON_LOG_FILE,
            model_save_path=MODEL_SAVE_PATH
        )
        total_time = time.time() - start_time
        logging.info(f"总训练时间: {total_time:.2f} 秒。最终最佳测试集RMSE: {best_rmse:.4f}")

        # 3. 绘制最佳模型的预测图
        regressor.plot_predictions(
            test_data=(X_test, y_test), 
            save_path=PREDICTION_PLOT_PATH
        )

        # 4. (可选) 验证加载功能
        logging.info("\n" + "="*40); logging.info("  验证已保存的最佳模型..."); logging.info("="*40)
        loaded_regressor = CoFrNetRegressor.load_model(MODEL_SAVE_PATH)
        
        # 使用加载的模型进行评估
        verification_rmse = loaded_regressor.evaluate(
            DataLoader(RegressionDataset(X_test, y_test), batch_size=HYPERPARAMETERS['batch_size'])
        )
        logging.info(f"加载后的模型在测试集上的验证 RMSE: {verification_rmse:.4f}")
        assert abs(verification_rmse - best_rmse) < 1e-4, "验证失败! RMSE 不匹配。"
        logging.info("验证成功! 模型可以被正确保存和复用。")
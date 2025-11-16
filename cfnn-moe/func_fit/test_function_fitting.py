# 文件名: test_function_fitting.py
# 描述: 使用 CoFrNetRegressor_MoE 对 scipy.special 中的函数进行拟合测试。
# v2: 增加了将详细训练日志保存到文件的功能。

import os
import logging
import json # 导入 json 模块
import numpy as np
import scipy.special as sp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 导入您提供的神经网络回归器
import sys
sys.path.append("/home/zxc/CodeBase/cofrnet")
from cfnet_regressor_moe_iterative import CoFrNetRegressor_MoE

# --- 1. 全局配置 ---
# 创建用于保存结果的目录
RESULTS_DIR = "function_fitting_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 定义要测试的函数列表 (保持不变)
FUNCTIONS_TO_TEST = [
    # {
    #     'name': 'Jacobian_Elliptic_sn',
    #     'func': lambda x, y: sp.ellipj(x, y)[0],
    #     'domain': {'x': (-5, 5), 'y': (0, 1)},
    #     'is_complex': False,
    #     'n_points': 50
    # },
    # {
    #     'name': 'Incomplete_Elliptic_Integral_K',
    #     'func': lambda x, y: sp.ellipkinc(x, y),
    #     'domain': {'x': (0, 2 * np.pi), 'y': (0, 1)},
    #     'is_complex': False,
    #     'n_points': 40
    # },
    # {
    #     'name': 'Incomplete_Elliptic_Integral_E',
    #     'func': lambda x, y: sp.ellipeinc(x, y),
    #     'domain': {'x': (0, 2 * np.pi), 'y': (0, 1)},
    #     'is_complex': False,
    #     'n_points': 40
    # },
    # {
    #     'name': 'Bessel_Jv',
    #     'func': lambda x, y: sp.jv(x, y),
    #     'domain': {'x': (0, 10), 'y': (0.1, 15)},
    #     'is_complex': False,
    #     'n_points': 50
    # },
    # {
    #     'name': 'Bessel_Yv',
    #     'func': lambda x, y: sp.yv(x, y),
    #     'domain': {'x': (0, 10), 'y': (0.1, 15)},
    #     'is_complex': False,
    #     'n_points': 50
    # },
    # {
    #     'name': 'Modified_Bessel_Kv',
    #     'func': lambda x, y: sp.kv(x, y),
    #     'domain': {'x': (0, 5), 'y': (0.1, 5)},
    #     'is_complex': False,
    #     'n_points': 40
    # },
    # {
    #     'name': 'Modified_Bessel_Iv',
    #     'func': lambda x, y: sp.iv(x, y),
    #     'domain': {'x': (0, 5), 'y': (0, 5)},
    #     'is_complex': False,
    #     'n_points': 40
    # },
    # {
    #     'name': 'Associated_Legendre_m0',
    #     'func': lambda x, y: sp.lpmv(0, x, y),
    #     'domain': {'x': (0, 10), 'y': (-1, 1)},
    #     'is_complex': False,
    #     'n_points': 40
    # },
    # {
    #     'name': 'Associated_Legendre_m1',
    #     'func': lambda x, y: sp.lpmv(1, x, y),
    #     'domain': {'x': (1, 10), 'y': (-1, 1)},
    #     'is_complex': False,
    #     'n_points': 40
    # },
    {
        'name': 'Spherical_Harmonics_m0_n1',
        'func': lambda x, y: sp.sph_harm(0, 1, x, y),
        'domain': {'x': (0, np.pi), 'y': (0, 2 * np.pi)},
        'is_complex': True,
        'n_points': 40
    },
    {
        'name': 'Spherical_Harmonics_m1_n2',
        'func': lambda x, y: sp.sph_harm(1, 2, x, y),
        'domain': {'x': (0, np.pi), 'y': (0, 2 * np.pi)},
        'is_complex': True,
        'n_points': 50
    }
]

# --- 2. 辅助函数 (保持不变) ---

def generate_data(func, domain, n_points, is_complex):
    """根据指定的函数和定义域生成网格数据。"""
    x1_range = np.linspace(domain['x'][0], domain['x'][1], n_points)
    x2_range = np.linspace(domain['y'][0], domain['y'][1], n_points)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    X_input = np.vstack([X1.ravel(), X2.ravel()]).T
    
    Z = func(X_input[:, 0], X_input[:, 1])
    
    if is_complex:
        Y_output = np.vstack([np.real(Z.ravel()), np.imag(Z.ravel())]).T
    else:
        Z_flat = Z.ravel()
        valid_indices = np.isfinite(Z_flat)
        X_input = X_input[valid_indices]
        Y_output = Z_flat[valid_indices].reshape(-1, 1)
        
    logging.info(f"生成了 {X_input.shape[0]} 个有效数据点。")
    return X_input, Y_output, X1, X2

def plot_results(X1, X2, Y_true_grid, Y_pred_grid, title, save_path):
    """绘制真实函数与模型预测结果的3D对比图。"""
    fig = plt.figure(figsize=(14, 7))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X1, X2, Y_true_grid, cmap='viridis', edgecolor='none')
    ax1.set_title("True Function")
    ax1.set_xlabel('X1'); ax1.set_ylabel('X2'); ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X1, X2, Y_pred_grid, cmap='viridis', edgecolor='none')
    ax2.set_title("Predicted Function")
    ax2.set_xlabel('X1'); ax2.set_ylabel('X2'); ax2.set_zlabel('Z')
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()
    logging.info(f"结果图已保存至: {save_path}")

# --- 3. 主执行流程 ---

def setup_logger(task_name):
    """为每个任务动态配置日志记录器。"""
    logger = logging.getLogger() # 获取根日志记录器
    logger.setLevel(logging.INFO)
    
    # 移除之前任务可能添加的所有处理器
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
        
    # 创建文件处理器，将日志写入文件
    log_file_path = os.path.join(RESULTS_DIR, f"{task_name}.log")
    file_handler = logging.FileHandler(log_file_path, mode='w') # 'w' 表示覆盖旧日志
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 创建流处理器，将日志输出到控制台
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    logging.info(f"日志将同时输出到控制台和文件: {log_file_path}")

def main():
    """主函数，循环遍历所有定义的函数并进行训练和评估。"""
    for f_info in FUNCTIONS_TO_TEST:
        task_name = f_info['name']
        
        # *** 新增：为当前任务设置独立的日志文件 ***
        setup_logger(task_name)
        
        logging.info(f"\n{'='*60}\n正在开始拟合任务: {task_name}\n{'='*60}")
        
        # --- 数据准备 ---
        X, y, X1_grid, X2_grid = generate_data(
            f_info['func'], f_info['domain'], f_info['n_points'], f_info['is_complex']
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # --- 模型超参数配置 ---
        hparams = {
            'input_dim': 2,
            'output_dim': 2 if f_info['is_complex'] else 1,
            'shallow_depth_per_cofrnet': 4,
            'polynomial_degree': 4,
            'learning_rate_adam': 1e-1,
            'weight_decay': 1e-5,
            'batch_size': 256,
            'epochs_per_model': 1000,
            'early_stopping_patience': 50,
            'max_experts': 20,
        }
        
        # --- 模型训练 ---
        regressor = CoFrNetRegressor_MoE(hparams)
        model_save_path = os.path.join(RESULTS_DIR, f"{task_name}_model.pth")
        json_log_save_path = os.path.join(RESULTS_DIR, f"{task_name}_summary.json")
        
        # *** 修改：捕获返回的日志数据 ***
        training_summary = regressor.train_iterative(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            log_filepath=json_log_save_path, # 路径现在用于 JSON
            model_save_path=model_save_path
        )
        
        # *** 新增：将返回的结构化日志保存为 JSON 文件 ***
        with open(json_log_save_path, 'w') as f:
            json.dump(training_summary, f, indent=4)
        logging.info(f"训练摘要已保存到: {json_log_save_path}")

        # --- 评估与可视化 ---
        logging.info("训练完成，正在加载最佳模型并进行可视化...")
        
        try:
            best_model = CoFrNetRegressor_MoE.load_model(model_save_path)
        except FileNotFoundError:
            logging.error(f"模型文件未找到: {model_save_path}。跳过可视化。")
            continue

        # 1. 对有效数据点 `X` 进行预测 (X 的长度是过滤后的，例如 1570)
        predictions = best_model.predict(X)
        
        # 2. 生成完整的网格输入，用于获取完整的真实 Z 值和正确的掩码
        X_full_grid_input = np.vstack([X1_grid.ravel(), X2_grid.ravel()]).T
        Z_true_full_grid = f_info['func'](X_full_grid_input[:, 0], X_full_grid_input[:, 1])

        if f_info['is_complex']:
            # 对于复数函数，没有进行数据过滤，逻辑保持不变
            pred_grid_real = np.full(X1_grid.shape, np.nan)
            pred_grid_imag = np.full(X1_grid.shape, np.nan)
            pred_grid_real.ravel()[:len(predictions)] = predictions[:, 0]
            pred_grid_imag.ravel()[:len(predictions)] = predictions[:, 1]

            plot_results(
                X1_grid, X2_grid, np.real(Z_true_full_grid).reshape(X1_grid.shape), pred_grid_real,
                f"{task_name} - Real Part",
                os.path.join(RESULTS_DIR, f"{task_name}_real_part.png")
            )
            plot_results(
                X1_grid, X2_grid, np.imag(Z_true_full_grid).reshape(X1_grid.shape), pred_grid_imag,
                f"{task_name} - Imaginary Part",
                os.path.join(RESULTS_DIR, f"{task_name}_imag_part.png")
            )
        else:
            # 3. (修复点) 基于完整的 Z 值生成一个正确尺寸的布尔掩码 (长度为 1600)
            valid_indices_for_plotting = np.isfinite(Z_true_full_grid.ravel())

            # 4. 创建一个用于绘图的空网格
            pred_grid = np.full(X1_grid.shape, np.nan)
            
            # 5. 使用正确尺寸的掩码，将预测结果 (长度 1570) 填充到绘图网格 (长度 1600) 的有效位置
            pred_grid.ravel()[valid_indices_for_plotting] = predictions.squeeze()

            # 6. 使用完整的真实Z值进行绘图
            Z_true_plotting_grid = Z_true_full_grid.reshape(X1_grid.shape)
            plot_results(
                X1_grid, X2_grid, Z_true_plotting_grid, pred_grid,
                task_name,
                os.path.join(RESULTS_DIR, f"{task_name}_fit.png")
            )

if __name__ == '__main__':
    main()
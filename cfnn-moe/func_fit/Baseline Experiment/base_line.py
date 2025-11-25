# 对比 CFNet-MoE、MLP 和 XGBoost 在函数拟合任务上的表现

import os
import sys
import json
import time
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.special as sp
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.layout import Layout

# 导入 CFNet-MoE
sys.path.append(r"D:\CFNN Moe")
from cfnet_regressor_moe_iterative import CoFrNetRegressor_MoE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f" GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print(" No GPU detected, using CPU")


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['figure.facecolor'] = 'white'


sns.set_style("whitegrid")
sns.set_context("paper")


console = Console()

# --- 全局配置 ---
RESULTS_DIR = "baseline_comparison_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


COLOR_SCHEMES = {
    'CFNet-MoE': {'cmap': 'viridis', 'color': '#440154', 'marker': 'o'},
    'MLP-PyTorch': {'cmap': 'plasma', 'color': '#B12A90', 'marker': 's'},
    'XGBoost': {'cmap': 'inferno', 'color': '#FCA636', 'marker': '^'}
}

# 测试函数列表
FUNCTIONS_TO_TEST = [
    {
        'name': 'Jacobian_Elliptic_sn',
        'func': lambda x, y: sp.ellipj(x, y)[0],
        'domain': {'x': (-5, 5), 'y': (0, 1)},
        'is_complex': False,
        'n_points': 50
    },
    {
        'name': 'Incomplete_Elliptic_Integral_K',
        'func': lambda x, y: sp.ellipkinc(x, y),
        'domain': {'x': (0, 2 * np.pi), 'y': (0, 1)},
        'is_complex': False,
        'n_points': 40
    },
    {
        'name': 'Incomplete_Elliptic_Integral_E',
        'func': lambda x, y: sp.ellipeinc(x, y),
        'domain': {'x': (0, 2 * np.pi), 'y': (0, 1)},
        'is_complex': False,
        'n_points': 40
    },
    {
        'name': 'Bessel_Jv',
        'func': lambda x, y: sp.jv(x, y),
        'domain': {'x': (0, 10), 'y': (0.1, 15)},
        'is_complex': False,
        'n_points': 50
    },
    {
        'name': 'Bessel_Yv',
        'func': lambda x, y: sp.yv(x, y),
         'domain': {'x': (0, 10), 'y': (0.1, 15)},
        'is_complex': False,
        'n_points': 50
    },
    {
        'name': 'Modified_Bessel_Kv',
        'func': lambda x, y: sp.kv(x, y),
        'domain': {'x': (0, 5), 'y': (0.1, 5)},
        'is_complex': False,
        'n_points': 40
    },
    {
        'name': 'Modified_Bessel_Iv',
        'func': lambda x, y: sp.iv(x, y),
        'domain': {'x': (0, 5), 'y': (0, 5)},
        'is_complex': False,
        'n_points': 40
    },
    {
        'name': 'Associated_Legendre_m0',
        'func': lambda x, y: sp.lpmv(0, x, y),
        'domain': {'x': (0, 10), 'y': (-1, 1)},
        'is_complex': False,
        'n_points': 40
    },
    {
        'name': 'Associated_Legendre_m1',
        'func': lambda x, y: sp.lpmv(1, x, y),
        'domain': {'x': (1, 10), 'y': (-1, 1)},
        'is_complex': False,
        'n_points': 40
    },
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

# --- PyTorch MLP ---
class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(256, 128, 64, 32)):
        super(PyTorchMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_size
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# --- 辅助函数 ---
def setup_logger():

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    
    log_file = os.path.join(RESULTS_DIR, 'baseline_experiments.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def generate_data(func, domain, n_points, is_complex):

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
    
    return X_input, Y_output, X1, X2

class ModelTrainer:

    
    def __init__(self, model_type, hparams=None):
        self.model_type = model_type
        self.hparams = hparams
        self.model = None
        self.scaler = StandardScaler()
        self.training_time = 0
        self.device = device
        
    def train(self, X_train, y_train, X_val, y_val, progress_callback=None):
        """训练模型"""
        start_time = time.time()
        
        if self.model_type == 'CFNet-MoE':
            self.model = CoFrNetRegressor_MoE(self.hparams)
    
            original_level = logging.getLogger().level
            logging.getLogger().setLevel(logging.WARNING)
            
            self.model.train_iterative(
                train_data=(X_train, y_train),
                val_data=(X_val, y_val),
                log_filepath=os.path.join(RESULTS_DIR, 'cfnet_temp.json'),
                model_save_path=os.path.join(RESULTS_DIR, 'cfnet_temp.pth')
            )
            
            logging.getLogger().setLevel(original_level)
            
        elif self.model_type == 'MLP-PyTorch':
            # 使用PyTorch MLP
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # 转换为张量
            X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            
            # 创建数据加载器
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            
            # 初始化模型
            output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
            self.model = PyTorchMLP(X_train.shape[1], output_dim).to(self.device)
            
            # 训练设置
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
            
            # 训练循环
            best_val_loss = float('inf')
            patience_counter = 0
            max_patience = 50
            
            for epoch in range(500):  # 最大轮数
                # 训练阶段
                self.model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # 验证阶段
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                
                scheduler.step(val_loss)
                
                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        break
            
            # 加载最佳模型
            self.model.load_state_dict(best_model_state)
                
        elif self.model_type == 'XGBoost':
            # XGBoost 配置 
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 8,
                'learning_rate': 0.1,
                'n_estimators': 500,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'auto',  # GPU加速
                'predictor': 'gpu_predictor' if torch.cuda.is_available() else 'auto'
            }
            
            # 处理多输出情况
            if y_train.shape[1] == 1:
                self.model = xgb.XGBRegressor(**params)
                self.model.fit(
                    X_train, y_train.ravel(),
                    eval_set=[(X_val, y_val.ravel())],
                    verbose=False
                )
            else:
                # 对于多输出，训练多个模型
                self.model = []
                for i in range(y_train.shape[1]):
                    model_i = xgb.XGBRegressor(**params)
                    model_i.fit(
                        X_train, y_train[:, i],
                        eval_set=[(X_val, y_val[:, i])],
                        verbose=False
                    )
                    self.model.append(model_i)
        
        self.training_time = time.time() - start_time
        
    def predict(self, X):
        """预测"""
        if self.model_type == 'CFNet-MoE':
            return self.model.predict(X)
        elif self.model_type == 'MLP-PyTorch':
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            self.model.eval()
            with torch.no_grad():
                pred = self.model(X_tensor).cpu().numpy()
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            return pred
        elif self.model_type == 'XGBoost':
            if isinstance(self.model, list):
                # 多输出情况
                preds = []
                for model_i in self.model:
                    preds.append(model_i.predict(X))
                return np.column_stack(preds)
            else:
                pred = self.model.predict(X)
                return pred.reshape(-1, 1)

def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # 处理 R2 score
    if y_true.ndim == 1 or y_true.shape[1] == 1:
        r2 = r2_score(y_true, y_pred)
    else:
        # 多输出情况，计算平均 R2
        r2 = r2_score(y_true, y_pred, multioutput='uniform_average')
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def plot_comparison_3d(results, func_info, save_path):
    """3D对比图"""
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    models = ['CFNet-MoE', 'MLP-PyTorch', 'XGBoost']
    
    for idx, model_name in enumerate(models):
        result = results[model_name]
        
        # True function
        ax1 = fig.add_subplot(gs[0, idx], projection='3d')
        surf1 = ax1.plot_surface(
            result['X1'], result['X2'], result['Y_true'],
            cmap=COLOR_SCHEMES[model_name]['cmap'],
            edgecolor='none', alpha=0.9, antialiased=True
        )
        ax1.set_title(f'{model_name} - True Function', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X1', fontsize=10)
        ax1.set_ylabel('X2', fontsize=10)
        ax1.set_zlabel('Z', fontsize=10)
        ax1.view_init(elev=20, azim=45)
        ax1.grid(True, alpha=0.3)
        
        # Predicted function
        ax2 = fig.add_subplot(gs[1, idx], projection='3d')
        surf2 = ax2.plot_surface(
            result['X1'], result['X2'], result['Y_pred'],
            cmap=COLOR_SCHEMES[model_name]['cmap'],
            edgecolor='none', alpha=0.9, antialiased=True
        )
        ax2.set_title(f'{model_name} - Predicted (RMSE: {result["metrics"]["RMSE"]:.4f})', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('X1', fontsize=10)
        ax2.set_ylabel('X2', fontsize=10)
        ax2.set_zlabel('Z', fontsize=10)
        ax2.view_init(elev=20, azim=45)
        ax2.grid(True, alpha=0.3)
        
        # 添加颜色条
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    plt.suptitle(f'Function Fitting Comparison: {func_info["name"]}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(all_results, save_path):
    """指标对比图"""
    # 准备数据
    metrics_data = []
    for func_name, func_results in all_results.items():
        for model_name, result in func_results.items():
            metrics_data.append({
                'Function': func_name.replace('_', ' '),
                'Model': model_name,
                **result['metrics']
            })
    
    df = pd.DataFrame(metrics_data)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['RMSE', 'MAE', 'R2', 'MSE']
    titles = ['Root Mean Square Error', 'Mean Absolute Error', 
              'R² Score', 'Mean Square Error']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        
        sns.barplot(data=df, x='Function', y=metric, hue='Model', 
                   ax=ax, palette=[COLOR_SCHEMES[m]['color'] for m in ['CFNet-MoE', 'MLP-PyTorch', 'XGBoost']])
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Function', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Model', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 调整 x 轴标签
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right')
    
    plt.suptitle('Performance Metrics Comparison Across Functions', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_time_comparison(all_results, save_path):
    """训练时间对比图"""
    time_data = []
    for func_name, func_results in all_results.items():
        for model_name, result in func_results.items():
            time_data.append({
                'Function': func_name.replace('_', ' '),
                'Model': model_name,
                'Training Time (s)': result['training_time']
            })
    
    df = pd.DataFrame(time_data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 创建分组条形图
    sns.barplot(data=df, x='Function', y='Training Time (s)', hue='Model',
               ax=ax, palette=[COLOR_SCHEMES[m]['color'] for m in ['CFNet-MoE', 'MLP-PyTorch', 'XGBoost']])
    
    ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Function', fontsize=12)
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.legend(title='Model', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 调整 x 轴标签
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha('right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(all_results):
    """汇总表格"""
    table = Table(title="Performance Summary", show_header=True, header_style="bold magenta")
    table.add_column("Function", style="cyan", no_wrap=True)
    table.add_column("Model", style="green")
    table.add_column("RMSE", justify="right", style="yellow")
    table.add_column("R²", justify="right", style="yellow")
    table.add_column("Time (s)", justify="right", style="red")
    
    for func_name, func_results in all_results.items():
        for model_name, result in func_results.items():
            table.add_row(
                func_name.replace('_', ' '),
                model_name,
                f"{result['metrics']['RMSE']:.6f}",
                f"{result['metrics']['R2']:.6f}",
                f"{result['training_time']:.2f}"
            )
    
    return table

def main():
    """主函数"""
    setup_logger()
    console.print(Panel.fit(" [bold blue]Baseline Experiments: CFNet-MoE vs MLP vs XGBoost[/bold blue]", 
                           border_style="bold green"))
    
    if torch.cuda.is_available():
        console.print(f"[bold green] Using GPU: {torch.cuda.get_device_name(0)}[/bold green]")
    else:
        console.print("[bold yellow] Using CPU (GPU not available)[/bold yellow]")
    
    all_results = {}
    
    # 主进度条
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        main_task = progress.add_task("[cyan]Overall Progress...", total=len(FUNCTIONS_TO_TEST))
        
        for func_idx, func_info in enumerate(FUNCTIONS_TO_TEST):
            func_name = func_info['name']
            console.print(f"\n[bold yellow]Testing Function {func_idx+1}/{len(FUNCTIONS_TO_TEST)}: {func_name}[/bold yellow]")
            
            # 生成数据
            X, y, X1_grid, X2_grid = generate_data(
                func_info['func'], func_info['domain'], 
                func_info['n_points'], func_info['is_complex']
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            func_results = {}
            
            # 模型训练子任务
            model_task = progress.add_task(f"[green]Training models for {func_name}...", total=3)
            
            # 训练 CFNet-MoE
            console.print("   Training CFNet-MoE...")
            hparams = {
                'input_dim': 2,
                'output_dim': 2 if func_info['is_complex'] else 1,
                'shallow_depth_per_cofrnet': 4,
                'polynomial_degree': 4,
                'learning_rate_adam': 1e-1,
                'weight_decay': 1e-5,
                'batch_size': 256,
                'epochs_per_model': 500,
                'early_stopping_patience': 30,
                'max_experts': 10,
            }
            
            cfnet_trainer = ModelTrainer('CFNet-MoE', hparams)
            cfnet_trainer.train(X_train, y_train, X_val, y_val)
            cfnet_pred = cfnet_trainer.predict(X)
            
            # 处理预测结果
            if func_info['is_complex']:
                pred_grid = cfnet_pred[:, 0].reshape(X1_grid.shape)
            else:
                pred_grid = np.full(X1_grid.shape, np.nan)
                X_full = np.vstack([X1_grid.ravel(), X2_grid.ravel()]).T
                Z_full = func_info['func'](X_full[:, 0], X_full[:, 1])
                valid_mask = np.isfinite(Z_full.ravel())
                pred_grid.ravel()[valid_mask] = cfnet_pred.squeeze()
            
            func_results['CFNet-MoE'] = {
                'X1': X1_grid,
                'X2': X2_grid,
                'Y_true': func_info['func'](X1_grid, X2_grid) if not func_info['is_complex'] 
                         else np.real(func_info['func'](X1_grid, X2_grid)),
                'Y_pred': pred_grid,
                'metrics': calculate_metrics(y_val, cfnet_trainer.predict(X_val)),
                'training_time': cfnet_trainer.training_time
            }
            progress.update(model_task, advance=1)
            
            # 训练 MLP-PyTorch
            console.print("   Training MLP ...")
            mlp_trainer = ModelTrainer('MLP-PyTorch')
            mlp_trainer.train(X_train, y_train, X_val, y_val)
            mlp_pred = mlp_trainer.predict(X)
            
            if func_info['is_complex']:
                pred_grid = mlp_pred[:, 0].reshape(X1_grid.shape)
            else:
                pred_grid = np.full(X1_grid.shape, np.nan)
                pred_grid.ravel()[valid_mask] = mlp_pred.squeeze()
            
            func_results['MLP-PyTorch'] = {
                'X1': X1_grid,
                'X2': X2_grid,
                'Y_true': func_results['CFNet-MoE']['Y_true'],
                'Y_pred': pred_grid,
                'metrics': calculate_metrics(y_val, mlp_trainer.predict(X_val)),
                'training_time': mlp_trainer.training_time
            }
            progress.update(model_task, advance=1)
            
            # 训练 XGBoost
            console.print("   Training XGBoost...")
            xgb_trainer = ModelTrainer('XGBoost')
            xgb_trainer.train(X_train, y_train, X_val, y_val)
            xgb_pred = xgb_trainer.predict(X)
            
            if func_info['is_complex']:
                pred_grid = xgb_pred[:, 0].reshape(X1_grid.shape)
            else:
                pred_grid = np.full(X1_grid.shape, np.nan)
                pred_grid.ravel()[valid_mask] = xgb_pred.squeeze()
            
            func_results['XGBoost'] = {
                'X1': X1_grid,
                'X2': X2_grid,
                'Y_true': func_results['CFNet-MoE']['Y_true'],
                'Y_pred': pred_grid,
                'metrics': calculate_metrics(y_val, xgb_trainer.predict(X_val)),
                'training_time': xgb_trainer.training_time
            }
            progress.update(model_task, advance=1)
            
            all_results[func_name] = func_results
            
            # 生成对比图
            console.print("   Generating comparison plots...")
            plot_comparison_3d(
                func_results, func_info,
                os.path.join(RESULTS_DIR, f'{func_name}_comparison_3d.png')
            )
            
            progress.update(main_task, advance=1)
    
    # 汇总图表
    console.print("\n[bold cyan]Generating summary visualizations...[/bold cyan]")
    
    plot_metrics_comparison(all_results, os.path.join(RESULTS_DIR, 'metrics_comparison.png'))
    plot_training_time_comparison(all_results, os.path.join(RESULTS_DIR, 'training_time_comparison.png'))
    
    # 汇总表格
    console.print("\n")
    console.print(create_summary_table(all_results))
    
    # 保存结果
    json_results = {}
    for func_name, func_results in all_results.items():
        json_results[func_name] = {}
        for model_name, result in func_results.items():
            json_results[func_name][model_name] = {
                'metrics': result['metrics'],
                'training_time': result['training_time']
            }
    
    with open(os.path.join(RESULTS_DIR, 'baseline_results.json'), 'w') as f:
        json.dump(json_results, f, indent=4)
    
    console.print(f"\n[bold green] All experiments completed successfully![/bold green]")
    console.print(f"[bold blue]Results saved to: {RESULTS_DIR}[/bold blue]")

if __name__ == '__main__':
    main()
"""
CFNet 优势凸显实验 - 共享工具模块
包含内存测量、收敛记录、推理速度测试、参数限制模型生成器等工具
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Callable
import sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parents[1]
CLASSIFY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CLASSIFY_DIR))


import os

try:
    from cfnet_complicate import CFNet
except ImportError:
    pass
try:
    from hybrid import HybridRationalNet
except ImportError:
    pass
try:
    from kan_classification_base import KAN, count_parameters as kan_count_parameters
    count_parameters = kan_count_parameters
except ImportError:
    pass


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==========================================
# 1. MLP 定义 (用于对比)
# ==========================================

class MLP(nn.Module):
    """标准MLP用于对比实验"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        layers = []

        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        # 隐藏层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def find_mlp_hidden_dim(input_dim, output_dim, target_params, num_layers=2):
    """找到满足参数量限制的最优MLP隐藏层维度"""
    def calc_params(hidden_dim):
        params = input_dim * hidden_dim + hidden_dim  # 第一层
        for _ in range(num_layers - 2):
            params += hidden_dim * hidden_dim + hidden_dim  # 隐藏层
        params += hidden_dim * output_dim + output_dim  # 输出层
        return params

    best_h = 8
    min_diff = float('inf')

    for h in range(1, 2000):
        p = calc_params(h)
        diff = abs(p - target_params)
        if diff < min_diff:
            min_diff = diff
            best_h = h
        if p > target_params * 1.5:
            break

    return best_h


# ==========================================
# 2. 参数对齐工具
# ==========================================

def get_cfnet_params(input_dim, output_dim, depth=5, poly_degree=3):
    """计算CFNet的参数量"""
    model = CFNet(input_dim, output_dim, depth=depth, poly_degree=poly_degree)
    return count_parameters(model)


def get_hybrid_params(input_dim, output_dim, num_units=10, unit_degree=10):
    """计算HybridRationalNet的参数量"""
    model = HybridRationalNet(input_dim, output_dim, num_units=num_units, unit_degree=unit_degree)
    return count_parameters(model)


def create_model_with_param_limit(model_type: str, param_limit: int,
                                   input_dim: int, output_dim: int) -> Tuple[nn.Module, Dict]:
    """
    创建符合参数量限制的模型

    Returns:
        model: 创建的模型
        config: 模型配置字典
    """
    if model_type == 'CFNet':
        # 尝试不同的depth和poly_degree组合
        best_model = None
        best_config = None
        best_params = 0

        for depth in [2, 3, 4, 5, 6, 7, 8]:
            for degree in [2, 3, 4, 5]:
                try:
                    model = CFNet(input_dim, output_dim, depth=depth, poly_degree=degree)
                    params = count_parameters(model)
                    if params <= param_limit and params > best_params:
                        best_model = model
                        best_config = {'depth': depth, 'poly_degree': degree, 'params': params}
                        best_params = params
                except Exception:
                    continue

        return best_model, best_config

    elif model_type == 'Hybrid':
        best_model = None
        best_config = None
        best_params = 0

        for num_units in [2, 4, 6, 8, 10, 12, 16]:
            for degree in [2, 3, 4, 5, 6, 8, 10]:
                try:
                    model = HybridRationalNet(input_dim, output_dim, num_units=num_units, unit_degree=degree)
                    params = count_parameters(model)
                    if params <= param_limit and params > best_params:
                        best_model = model
                        best_config = {'num_units': num_units, 'unit_degree': degree, 'params': params}
                        best_params = params
                except Exception:
                    continue

        return best_model, best_config

    elif model_type == 'KAN':
        # KAN参数计算：基于隐藏层维度
        grid_size = 10
        spline_order = 3
        n_basis = grid_size + spline_order

        def calc_kan_params(hidden_dim, num_layers=2):
            total = 0
            dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
            for i in range(len(dims) - 1):
                layer_params = dims[i] * dims[i+1] * n_basis + dims[i+1]
                if i < len(dims) - 2:
                    layer_params += 2 * dims[i+1]  # LayerNorm
                total += layer_params
            return total

        best_h = 4
        min_diff = float('inf')

        for h in range(4, 200):
            p = calc_kan_params(h)
            diff = abs(p - param_limit)
            if diff < min_diff:
                min_diff = diff
                best_h = h
            if p > param_limit * 1.5:
                break

        model = KAN(input_dim, output_dim, hidden_dim=best_h, num_layers=2, grid_size=grid_size, spline_order=spline_order)
        actual_params = count_parameters(model)

        return model, {'hidden_dim': best_h, 'grid_size': grid_size,
                      'spline_order': spline_order, 'params': actual_params}

    elif model_type == 'MLP':
        hidden_dim = find_mlp_hidden_dim(input_dim, output_dim, param_limit)
        model = MLP(input_dim, hidden_dim, output_dim, num_layers=2)
        actual_params = count_parameters(model)

        return model, {'hidden_dim': hidden_dim, 'num_layers': 2, 'params': actual_params}

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ==========================================
# 3. 内存测量工具
# ==========================================

def measure_model_memory(model: nn.Module, batch_size: int, input_dim: int,
                         device: str = 'cuda') -> Dict[str, float]:
    """
    测量模型的内存占用

    Returns:
        dict: 包含以下内存指标（单位MB）
            - model_params_mb: 模型参数内存
            - forward_peak_mb: 前向传播峰值内存
            - backward_peak_mb: 反向传播峰值内存
            - total_peak_mb: 总峰值内存
    """
    if not torch.cuda.is_available():
        device = 'cpu'

    model = model.to(device)
    model.train()

    # 计算模型参数内存
    model_memory = sum(p.numel() * p.element_size() for p in model.parameters())

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # 创建输入
    x = torch.randn(batch_size, input_dim, device=device)

    # 前向传播内存测量
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    output = model(x)
    forward_memory = torch.cuda.max_memory_allocated() if device == 'cuda' else 0

    # 反向传播内存测量
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    loss = output.sum()
    loss.backward()

    backward_memory = torch.cuda.max_memory_allocated() if device == 'cuda' else 0

    result = {
        'model_params_mb': model_memory / 1024**2,
        'forward_peak_mb': forward_memory / 1024**2,
        'backward_peak_mb': backward_memory / 1024**2,
        'total_peak_mb': (model_memory + backward_memory) / 1024**2,
        'device': device
    }

    return result


def measure_inference_memory(model: nn.Module, batch_size: int, input_dim: int,
                              device: str = 'cuda') -> Dict[str, float]:
    """测量推理时的内存占用（评估模式）"""
    if not torch.cuda.is_available():
        device = 'cpu'

    model = model.to(device)
    model.eval()

    model_memory = sum(p.numel() * p.element_size() for p in model.parameters())

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    x = torch.randn(batch_size, input_dim, device=device)

    with torch.no_grad():
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        output = model(x)
        inference_memory = torch.cuda.max_memory_allocated() if device == 'cuda' else 0

    return {
        'model_params_mb': model_memory / 1024**2,
        'inference_peak_mb': inference_memory / 1024**2,
        'device': device
    }


# ==========================================
# 4. 推理速度测试工具
# ==========================================

def benchmark_inference(model: nn.Module, input_dim: int,
                        batch_sizes: List[int] = [1, 32, 128, 256],
                        device: str = 'cuda', num_warmup: int = 10,
                        num_iters: int = 100) -> Dict:
    """
    基准测试模型推理速度

    Returns:
        dict: 各批次大小的推理速度指标
    """
    if not torch.cuda.is_available():
        device = 'cpu'

    model = model.to(device)
    model.eval()

    results = {}

    for bs in batch_sizes:
        x = torch.randn(bs, input_dim, device=device)

        # 预热
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(x)

        if device == 'cuda':
            torch.cuda.synchronize()

        # 测量
        start = time.time()

        with torch.no_grad():
            for _ in range(num_iters):
                _ = model(x)

        if device == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start

        results[bs] = {
            'samples_per_sec': (bs * num_iters) / elapsed,
            'ms_per_batch': (elapsed / num_iters) * 1000,
            'ms_per_sample': (elapsed / num_iters) * 1000 / bs
        }

    return results


def benchmark_inference_cpu_vs_gpu(model: nn.Module, input_dim: int,
                                    batch_size: int = 128,
                                    num_iters: int = 100) -> Dict:
    """对比CPU和GPU的推理速度"""
    results = {}

    for device_name in ['cpu', 'cuda']:
        if device_name == 'cuda' and not torch.cuda.is_available():
            continue

        device = torch.device(device_name)
        model_device = model.to(device)
        model_device.eval()

        x = torch.randn(batch_size, input_dim, device=device)

        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model_device(x)

        if device_name == 'cuda':
            torch.cuda.synchronize()

        # 测量
        start = time.time()

        with torch.no_grad():
            for _ in range(num_iters):
                _ = model_device(x)

        if device_name == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start

        results[device_name] = {
            'samples_per_sec': (batch_size * num_iters) / elapsed,
            'ms_per_batch': (elapsed / num_iters) * 1000
        }

    return results


# ==========================================
# 5. 收敛追踪器
# ==========================================

class ConvergenceTracker:
    """追踪模型训练收敛情况"""

    def __init__(self):
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'time_elapsed': [],
            'learning_rate': []
        }
        self.start_time = time.time()

    def record(self, epoch: int, train_loss: float, val_loss: float,
               val_acc: float, lr: float = None):
        """记录一个epoch的训练结果"""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['time_elapsed'].append(time.time() - self.start_time)
        self.history['learning_rate'].append(lr)

    def get_convergence_epoch(self, target_acc: float) -> Optional[int]:
        """获取达到目标准确率所需的epoch数"""
        for i, acc in enumerate(self.history['val_acc']):
            if acc >= target_acc:
                return self.history['epoch'][i]
        return None

    def get_convergence_time(self, target_acc: float) -> Optional[float]:
        """获取达到目标准确率所需的时间"""
        for i, acc in enumerate(self.history['val_acc']):
            if acc >= target_acc:
                return self.history['time_elapsed'][i]
        return None

    def get_avg_improvement_speed(self, first_n_epochs: int = 10) -> float:
        """计算前N个epoch的平均提升速度"""
        if len(self.history['val_acc']) < 2:
            return 0.0

        n = min(first_n_epochs, len(self.history['val_acc']))
        acc_diff = self.history['val_acc'][n-1] - self.history['val_acc'][0]
        return acc_diff / n

    def to_dict(self) -> Dict:
        """导出为字典"""
        return {
            'history': self.history,
            'total_epochs': len(self.history['epoch']),
            'final_val_acc': self.history['val_acc'][-1] if self.history['val_acc'] else 0,
            'best_val_acc': max(self.history['val_acc']) if self.history['val_acc'] else 0,
            'best_val_epoch': self.history['epoch'][np.argmax(self.history['val_acc'])] if self.history['val_acc'] else 0
        }


# ==========================================
# 6. 结果保存与加载
# ==========================================

def save_benchmark_results(results: Dict, filename: str):
    """保存benchmark结果到JSON文件"""
    # 转换numpy类型为Python原生类型
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results_converted = convert(results)

    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(results_converted, f, indent=2)
    logging.info(f"Results saved to {filename}")


def load_benchmark_results(filename: str) -> Dict:
    """从JSON文件加载benchmark结果"""
    with open(filename, 'r') as f:
        return json.load(f)


# ==========================================
# 7. 模型创建工厂
# ==========================================

def create_model(model_type: str, input_dim: int, output_dim: int,
                 config: Optional[Dict] = None) -> Tuple[nn.Module, Dict]:
    """
    创建指定类型的模型

    Args:
        model_type: 'CFNet', 'Hybrid', 'KAN', 'MLP'
        input_dim: 输入维度
        output_dim: 输出维度
        config: 可选的模型配置

    Returns:
        model: 创建的模型
        model_info: 包含参数量和配置的元信息
    """
    config = config or {}

    if model_type == 'CFNet':
        depth = config.get('depth', 5)
        poly_degree = config.get('poly_degree', 3)
        model = CFNet(input_dim, output_dim, depth=depth, poly_degree=poly_degree)
        params = count_parameters(model)
        return model, {'type': 'CFNet', 'depth': depth, 'poly_degree': poly_degree, 'params': params}

    elif model_type == 'Hybrid':
        num_units = config.get('num_units', 10)
        unit_degree = config.get('unit_degree', 10)
        model = HybridRationalNet(input_dim, output_dim, num_units=num_units, unit_degree=unit_degree)
        params = count_parameters(model)
        return model, {'type': 'Hybrid', 'num_units': num_units, 'unit_degree': unit_degree, 'params': params}

    elif model_type == 'KAN':
        hidden_dim = config.get('hidden_dim', 32)
        grid_size = config.get('grid_size', 10)
        spline_order = config.get('spline_order', 3)
        model = KAN(input_dim, output_dim, hidden_dim=hidden_dim, grid_size=grid_size, spline_order=spline_order)
        params = count_parameters(model)
        return model, {'type': 'KAN', 'hidden_dim': hidden_dim, 'grid_size': grid_size,
                      'spline_order': spline_order, 'params': params}

    elif model_type == 'MLP':
        hidden_dim = config.get('hidden_dim', 128)
        num_layers = config.get('num_layers', 2)
        model = MLP(input_dim, hidden_dim, output_dim, num_layers=num_layers)
        params = count_parameters(model)
        return model, {'type': 'MLP', 'hidden_dim': hidden_dim, 'num_layers': num_layers, 'params': params}

    else:
        raise ValueError(f"Unknown model type: {model_type}")
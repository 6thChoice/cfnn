"""
共享工具函数模块
包含数据生成、梯度监测、模型训练等通用功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from typing import Callable, Dict, List, Tuple, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

# ==========================================
# 1. 测试函数定义
# ==========================================

class TestFunctions:
    """测试函数集合"""

    @staticmethod
    def f1_polynomial(x):
        """简单多项式: y = 0.5*x^3 + x^2 - 2*x + 1"""
        return 0.5 * x**3 + x**2 - 2*x + 1

    @staticmethod
    def f2_trigonometric(x):
        """简单三角函数: y = sin(2*pi*x) + cos(pi*x)"""
        return np.sin(2 * np.pi * x) + np.cos(np.pi * x)

    @staticmethod
    def f3_linear_comb(x):
        """线性组合: y = 2*x + 1"""
        return 2 * x + 1

    @staticmethod
    def f4_composite(x):
        """复合函数: y = exp(x) * sin(3*pi*x)"""
        return np.exp(x) * np.sin(3 * np.pi * x)

    @staticmethod
    def f5_rational(x):
        """分式函数: y = (x^2 + 1) / (x^2 + 2)"""
        return (x**2 + 1) / (x**2 + 2)

    @staticmethod
    def f6_hyperbolic(x):
        """双曲函数: y = tanh(x) * x^2"""
        return np.tanh(x) * x**2

    @staticmethod
    def f7_high_freq(x):
        """高频振荡函数: y = sin(10*pi*x) * exp(-x^2/10)"""
        return np.sin(10 * np.pi * x) * np.exp(-x**2 / 10)

    @staticmethod
    def f8_multimodal(x):
        """多峰函数: y = x^2 * sin(5*x) + cos(3*x)"""
        return x**2 * np.sin(5 * x) + np.cos(3 * x)

    @staticmethod
    def f9_runge(x):
        """Runge函数: y = 1 / (1 + 25*x^2)"""
        return 1 / (1 + 25 * x**2)

    @staticmethod
    def f10_piecewise(x):
        """分段变化函数: y = sign(x) * |x|^0.5 + 0.5*sin(10*x)"""
        return np.sign(x) * np.abs(x)**0.5 + 0.5 * np.sin(10 * x)

    @staticmethod
    def f3d_exp_sum(X):
        """3D指数函数: f(X) = exp(x1 + x2 + x3)"""
        return np.exp(np.sum(X, axis=1)).reshape(-1, 1)

    @staticmethod
    def f3d_rational(X):
        """3D分式函数: f(X) = (x1 * x2) / (x3 + epsilon)"""
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        return ((x1 * x2) / (x3 + 1e-5)).reshape(-1, 1)

    @classmethod
    def get_function(cls, name: str) -> Optional[Callable]:
        """根据名称获取函数"""
        return getattr(cls, name, None)

    @classmethod
    def get_all_1d_functions(cls) -> Dict[str, Callable]:
        """获取所有1D函数"""
        return {
            'f1_polynomial': cls.f1_polynomial,
            'f2_trigonometric': cls.f2_trigonometric,
            'f3_linear_comb': cls.f3_linear_comb,
            'f4_composite': cls.f4_composite,
            'f5_rational': cls.f5_rational,
            'f6_hyperbolic': cls.f6_hyperbolic,
            'f7_high_freq': cls.f7_high_freq,
            'f8_multimodal': cls.f8_multimodal,
            'f9_runge': cls.f9_runge,
            'f10_piecewise': cls.f10_piecewise,
        }

    @classmethod
    def get_all_3d_functions(cls) -> Dict[str, Callable]:
        """获取所有3D函数"""
        return {
            'f3d_exp_sum': cls.f3d_exp_sum,
            'f3d_rational': cls.f3d_rational,
        }


# ==========================================
# 2. 数据生成
# ==========================================

def generate_1d_data(func: Callable, n_samples: int = 5000,
                     x_range: Tuple[float, float] = (-2, 2),
                     noise_std: float = 0.0, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成1D函数数据

    Args:
        func: 目标函数
        n_samples: 样本数量
        x_range: x的范围
        noise_std: 噪声标准差
        seed: 随机种子

    Returns:
        (X, Y) 张量元组
    """
    np.random.seed(seed)
    x = np.random.uniform(x_range[0], x_range[1], n_samples)
    y = func(x)
    if noise_std > 0:
        y += np.random.normal(0, noise_std, n_samples)

    X = torch.FloatTensor(x.reshape(-1, 1))
    Y = torch.FloatTensor(y.reshape(-1, 1))
    return X, Y


def generate_3d_data(func: Callable, n_samples: int = 10000,
                     x_range: Tuple[float, float] = (-2, 2),
                     noise_std: float = 0.0, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成3D函数数据

    Args:
        func: 目标函数
        n_samples: 样本数量
        x_range: 每个维度的范围
        noise_std: 噪声标准差
        seed: 随机种子

    Returns:
        (X, Y) 张量元组
    """
    np.random.seed(seed)
    X = np.random.uniform(x_range[0], x_range[1], (n_samples, 3))
    y = func(X)
    if noise_std > 0:
        y += np.random.normal(0, noise_std, y.shape)

    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(y)
    return X_tensor, Y_tensor


def split_data(X: torch.Tensor, Y: torch.Tensor,
               train_ratio: float = 0.65,
               val_ratio: float = 0.05,
               test_ratio: float = 0.30,
               seed: int = 42) -> Tuple[Tuple, Tuple, Tuple]:
    """
    划分数据集

    Returns:
        ((X_train, Y_train), (X_val, Y_val), (X_test, Y_test))
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    n_samples = X.shape[0]
    indices = np.random.RandomState(seed).permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    return (
        (X[train_indices], Y[train_indices]),
        (X[val_indices], Y[val_indices]),
        (X[test_indices], Y[test_indices])
    )


# ==========================================
# 3. 梯度监测
# ==========================================

class GradientMonitor:
    """
    梯度监测器
    记录训练过程中的梯度统计信息
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.grad_norms = []           # 每步的总梯度范数
        self.layer_grad_norms = {}     # 每层的梯度范数历史
        self.grad_maxs = []            # 每步的最大梯度值
        self.nan_count = 0             # NaN梯度计数
        self.inf_count = 0             # Inf梯度计数
        self.hooks = []                # 注册的hooks
        self.layer_names = []          # 层名称
        self._step_count = 0                # 步数计数

        # 为每个参数注册梯度 hook（使用 register_hook）
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.layer_names.append(name)
                self.layer_grad_norms[name] = []
                # 使用 register_hook 直接在参数的梯度上注册
                hook = param.register_hook(self._make_grad_hook(name))
                self.hooks.append(hook)

    def _make_grad_hook(self, name: str):
        """创建梯度监测 hook"""
        def hook(grad):
            """梯度 hook，接收梯度张量，返回处理后的梯度"""
            # 记录梯度范数
            if grad is not None:
                grad_norm = torch.norm(grad).item()
                self.layer_grad_norms[name].append(grad_norm)

                # 检测异常值
                if torch.isnan(grad).any():
                    self.nan_count += 1
                if torch.isinf(grad).any():
                    self.inf_count += 1

            # 返回原始梯度，不做修改
            return grad
        return hook

    def record_step(self):
        """
        记录当前步骤的梯度统计
        在 optimizer.step() 之前调用
        """
        total_norm = 0.0
        max_grad = 0.0

        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_grad = max(max_grad, torch.max(torch.abs(p.grad)).item())

        self.grad_norms.append(total_norm ** 0.5)
        self.grad_maxs.append(max_grad)

    def get_stats(self) -> Dict:
        """获取梯度统计信息"""
        grad_array = np.array(self.grad_norms)
        return {
            "grad_norms": self.grad_norms,
            "mean_grad_norm": float(np.mean(grad_array)) if len(grad_array) > 0 else 0.0,
            "std_grad_norm": float(np.std(grad_array)) if len(grad_array) > 0 else 0.0,
            "max_grad_norm": float(np.max(grad_array)) if len(grad_array) > 0 else 0.0,
            "min_grad_norm": float(np.min(grad_array)) if len(grad_array) > 0 else 0.0,
            "grad_maxs": self.grad_maxs,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
        }

    def reset(self):
        """重置统计信息"""
        self.grad_norms = []
        self.layer_grad_norms = []
        self.grad_maxs = []
        self.nan_count = 0
        self.inf_count = 0

    def remove_hooks(self):
        """移除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ==========================================
# 4. 训练函数
# ==========================================

def train_standard(model: nn.Module, X: torch.Tensor, Y: torch.Tensor,
                   epochs: int = 2000, lr: float = 0.001,
                   batch_size: int = 128, grad_clip: float = 1.0,
                   device: torch.device = torch.device('cpu'),
                   grad_monitor: Optional[GradientMonitor] = None,
                   verbose: bool = False) -> Dict:
    """
    标准训练函数（用于 CFNN 和 Hybrid）

    Returns:
        包含训练结果的字典
    """
    model = model.to(device)
    X, Y = X.to(device), Y.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.7), gamma=0.5)

    model.train()
    loss_history = []
    n_samples = X.shape[0]

    for epoch in range(epochs):
        # 小批量训练
        indices = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_X = X[batch_indices]
            batch_Y = Y[batch_indices]

            optimizer.zero_grad()
            pred = model(batch_X)
            loss = loss_fn(pred, batch_Y)
            loss.backward()

            # 记录梯度
            if grad_monitor is not None:
                grad_monitor.record_step()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        scheduler.step()

        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            logging.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")

    return {
        "final_loss": loss_history[-1],
        "loss_history": loss_history,
        "grad_stats": grad_monitor.get_stats() if grad_monitor else None,
    }


def train_boost_iterative(model: nn.Module, X: torch.Tensor, Y: torch.Tensor,
                          total_units: int, total_epochs: int,
                          lr: float = 0.001, batch_size: int = 128,
                          grad_clip: float = 1.0,
                          device: torch.device = torch.device('cpu'),
                          grad_monitor: Optional[GradientMonitor] = None,
                          verbose: bool = False) -> Dict:
    """
    Boost 迭代训练

    Returns:
        包含训练结果的字典
    """
    model = model.to(device)
    X, Y = X.to(device), Y.to(device)

    loss_fn = nn.MSELoss()
    loss_history = []

    # 每个阶段分配的 epoch 数
    epochs_per_stage = max(50, total_epochs // total_units)

    for stage in range(total_units):
        # 添加新模型
        model.add_model()

        # 冻结旧模型，只训练新加入的
        model.freeze_all_but_latest()

        # 只优化新参数
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        # 重置梯度监测器
        if grad_monitor:
            grad_monitor.reset()

        # 训练阶段
        for epoch in range(epochs_per_stage):
            indices = torch.randperm(X.shape[0])
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, X.shape[0], batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_X = X[batch_indices]
                batch_Y = Y[batch_indices]

                optimizer.zero_grad()
                pred = model(batch_X)
                loss = loss_fn(pred, batch_Y)
                loss.backward()

                if grad_monitor:
                    grad_monitor.record_step()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            loss_history.append(avg_loss)

        if verbose:
            logging.info(f"Stage {stage + 1}/{total_units}, Units: {len(model.models)}, Loss: {avg_loss:.6f}")

    return {
        "final_loss": loss_history[-1],
        "loss_history": loss_history,
        "grad_stats": grad_monitor.get_stats() if grad_monitor else None,
    }


def train_moe_iterative(model: nn.Module, X: torch.Tensor, Y: torch.Tensor,
                        total_experts: int, total_epochs: int,
                        lr: float = 0.001, batch_size: int = 128,
                        grad_clip: float = 1.0,
                        device: torch.device = torch.device('cpu'),
                        grad_monitor: Optional[GradientMonitor] = None,
                        verbose: bool = False) -> Dict:
    """
    MoE 迭代训练

    Returns:
        包含训练结果的字典
    """
    model = model.to(device)
    X, Y = X.to(device), Y.to(device)

    loss_fn = nn.MSELoss()
    loss_history = []

    epochs_per_stage = max(50, total_epochs // total_experts)

    for stage in range(total_experts):
        # 寻找初始化中心
        if stage == 0:
            center = X.mean(dim=0).cpu().numpy()
        else:
            model.eval()
            with torch.no_grad():
                preds = model(X)
                errors = torch.sum((preds - Y)**2, dim=1)
                worst_idx = torch.argmax(errors).item()
                center = X[worst_idx].cpu().numpy()

        # 添加专家和门控
        model.add_expert()
        # 获取设备用于门控网络
        device = model._device_tracker.device
        model.gating.add_expert_gate(center, initial_width_param=1.0, device=device)
        model.train()

        # 优化器（训练所有参数）
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 重置梯度监测器
        if grad_monitor:
            grad_monitor.reset()

        # 训练阶段
        for epoch in range(epochs_per_stage):
            indices = torch.randperm(X.shape[0])
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, X.shape[0], batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_X = X[batch_indices]
                batch_Y = Y[batch_indices]

                optimizer.zero_grad()
                pred = model(batch_X)
                loss = loss_fn(pred, batch_Y)
                loss.backward()

                if grad_monitor:
                    grad_monitor.record_step()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            loss_history.append(avg_loss)

        if verbose:
            logging.info(f"Stage {stage + 1}/{total_experts}, Experts: {len(model.experts)}, Loss: {avg_loss:.6f}")

    return {
        "final_loss": loss_history[-1],
        "loss_history": loss_history,
        "grad_stats": grad_monitor.get_stats() if grad_monitor else None,
    }


# ==========================================
# 5. 模型工具
# ==========================================

def count_parameters(model: nn.Module) -> int:
    """计算模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """获取计算设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 6. 结果保存
# ==========================================

def save_results(results: Dict, filepath: str):
    """保存实验结果到JSON文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # 转换numpy类型为Python原生类型
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj

    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=4)

    logging.info(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict:
    """从JSON文件加载实验结果"""
    with open(filepath, 'r') as f:
        return json.load(f)


# ==========================================
# 7. 模型样式配置（用于可视化）
# ==========================================

MODEL_STYLES = {
    "CFNN":       {"color": "#1f77b4", "label": "CFNN", "marker": "o"},
    "CFNN-Boost": {"color": "#2ca02c", "label": "CFNN-Boost", "marker": "s"},
    "CFNN-MoE":   {"color": "#9467bd", "label": "CFNN-MoE", "marker": "^"},
    "CFNN-Hybrid":{"color": "#d62728", "label": "CFNN-Hybrid", "marker": "D"},
}

# 模型名称映射
MODEL_NAME_MAP = {
    "CFNet_Standard": "CFNN",
    "EnsembleResCoFrNet": "CFNN-Boost",
    "MoE_Ensemble": "CFNN-MoE",
    "HybridRationalNet": "CFNN-Hybrid",
}

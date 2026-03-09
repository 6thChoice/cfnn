"""
基础分析器类
所有分析器都应继承此类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
import torch
import shap
from torch.utils.data import DataLoader


class BaseAnalyzer(ABC):
    """分析器基类"""

    def __init__(self, config: Dict[str, Any], save_dir: str = 'results'):
        """
        Args:
            config: 配置字典
            save_dir: 结果保存目录
        """
        self.config = config
        self.save_dir = save_dir
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # 创建保存目录
        import os
        os.makedirs(save_dir, exist_ok=True)

        # 分析结果存储
        self.results = {}

    @abstractmethod
    def compute_metrics(self, model: torch.nn.Module,
                       test_loader: DataLoader) -> Dict[str, float]:
        """
        计算性能指标

        Args:
            model: 训练好的模型
            test_loader: 测试数据加载器

        Returns:
            metrics: 指标字典
        """
        pass

    @abstractmethod
    def analyze_shap(self, model: torch.nn.Module,
                    train_loader: DataLoader,
                    test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 SHAP 值

        Args:
            model: 训练好的模型
            train_loader: 训练数据加载器（用于 SHAP 背景）
            test_loader: 测试数据加载器

        Returns:
            shap_values: SHAP 值
            test_data: 测试数据
        """
        pass

    @abstractmethod
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        生成分析报告

        Args:
            results: 分析结果

        Returns:
            report: Markdown 格式的报告
        """
        pass

    def save_results(self, results: Dict[str, Any], name: str):
        """
        保存分析结果

        Args:
            results: 分析结果
            name: 结果名称
        """
        import os
        import pickle

        # 保存为 pickle
        save_path = os.path.join(self.save_dir, f'{name}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)

        print(f"Results saved to: {save_path}")

    def compute_shap_values(self, model: torch.nn.Module,
                           train_loader: DataLoader,
                           test_loader: DataLoader,
                           background_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 SHAP 值的通用方法

        Args:
            model: 训练好的模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            background_size: 背景数据大小

        Returns:
            shap_values: SHAP 值
            test_data: 测试数据
        """
        model.eval()
        device = next(model.parameters()).device

        # 获取背景数据（训练集的子集）
        background_data = []
        for X_batch, _ in train_loader:
            background_data.append(X_batch.numpy())
            if len(background_data) * self.config['batch_size'] >= background_size:
                break
        background = np.concatenate(background_data)[:background_size]

        # 获取测试数据
        test_data_list = []
        for X_batch, _ in test_loader:
            test_data_list.append(X_batch.numpy())
            test_data = np.concatenate(test_data_list)

        # 创建预测函数
        def predict_fn(x):
            if isinstance(x, np.ndarray):
                x_tensor = torch.from_numpy(x).float()
            else:
                x_tensor = x
            x_tensor = x_tensor.to(device)
            with torch.no_grad():
                output = model(x_tensor)
            return output.cpu().numpy()

        # 使用 PermutationExplainer（更稳定，支持任何模型）
        # 减少评估次数以加快速度
        explainer = shap.explainers.PermutationExplainer(
            predict_fn,
            background,
            max_evals=500  # 限制评估次数
        )

        # 计算 SHAP 值（限制样本数以加快速度）
        max_samples = min(50, len(test_data))  # 减少到 50 个样本
        shap_values = explainer.shap_values(test_data[:max_samples])

        # 处理形状
        if shap_values.ndim == 3:
            shap_values = shap_values.squeeze(-1)

        return shap_values, test_data[:max_samples]

    def compute_performance_metrics(self, model: torch.nn.Module,
                                   test_loader: DataLoader) -> Dict[str, float]:
        """
        计算性能指标的通用方法

        Args:
            model: 训练好的模型
            test_loader: 测试数据加载器

        Returns:
            metrics: 指标字典
        """
        model.eval()
        device = next(model.parameters()).device

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        all_preds = np.concatenate(all_preds).flatten()
        all_targets = np.concatenate(all_targets).flatten()

        # 计算指标
        mse = np.mean((all_preds - all_targets) ** 2)
        mae = np.mean(np.abs(all_preds - all_targets))

        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2)
        }

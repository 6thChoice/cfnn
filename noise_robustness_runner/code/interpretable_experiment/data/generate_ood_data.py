"""
分布外数据生成器
支持多种分布偏移场景
"""

from .base import BaseDataGenerator
from typing import Dict, Any, Tuple
import numpy as np


class OODDataGenerator(BaseDataGenerator):
    """分布外数据生成器"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典
                - ood_type: OOD类型
                  * 'covariate_shift': 协变量偏移
                  * 'concept_drift': 概念漂移
                  * 'sparse_region': 稀疏区域
                - train_distribution: 训练分布配置
                - test_distribution: 测试分布配置
        """
        super().__init__(config)

        self.ood_type = config.get('ood_type', 'covariate_shift')
        self.train_dist_config = config.get('train_distribution', {'type': 'uniform', 'low': 0.5, 'high': 2.0})
        self.test_dist_config = config.get('test_distribution', {'type': 'uniform', 'low': 2.0, 'high': 3.5})

        # 真实系数（用于概念漂移）
        if self.ood_type == 'concept_drift':
            self.train_coefficients = np.array([1.5, 0.8, 0.1, 0.1])
            self.test_coefficients = np.array([0.1, 0.1, 1.5, 0.8])
        else:
            self.train_coefficients = np.array([1.5, 0.8, 0.5, 1.2])
            self.test_coefficients = self.train_coefficients

    def _generate_samples(self, n_samples: int, dist_config: Dict[str, Any]) -> np.ndarray:
        """
        根据分布配置生成样本

        Args:
            n_samples: 样本数
            dist_config: 分布配置

        Returns:
            X: (n_samples, n_features) 特征矩阵
        """
        dist_type = dist_config['type']

        if dist_type == 'uniform':
            low = dist_config.get('low', 0.5)
            high = dist_config.get('high', 2.0)
            X = np.random.uniform(low, high, (n_samples, 4))
        elif dist_type == 'normal':
            mean = dist_config.get('mean', 1.0)
            std = dist_config.get('std', 0.5)
            X = np.random.normal(mean, std, (n_samples, 4))
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

        return X

    def generate_features(self) -> np.ndarray:
        """
        生成特征矩阵（训练集和测试集分开生成）

        Returns:
            X: 特征矩阵（仅用于初始调用，实际分布不同）
        """
        # 这个方法只在初始调用时使用，返回训练分布的特征
        return self._generate_samples(self.n_samples, self.train_dist_config)

    def generate_target(self, X: np.ndarray, coefficients: np.ndarray = None) -> np.ndarray:
        """
        生成目标变量

        Args:
            X: 特征矩阵
            coefficients: 系数（用于概念漂移）

        Returns:
            y: 目标变量
        """
        if coefficients is None:
            coefficients = self.train_coefficients

        # 线性组合
        y = np.dot(X, coefficients)

        return y

    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        划分数据集（重写以支持OOD测试集）

        Args:
            X: 特征矩阵
            y: 目标变量

        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        n = len(X)
        n_test = int(n * self.test_split)
        n_val = int(n * self.val_split)
        n_train = n - n_test - n_val

        # 从训练分布划分训练集和验证集
        indices = np.random.permutation(n)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]

        # 从测试分布生成测试集（OOD）
        X_test_ood = self._generate_samples(n_test, self.test_dist_config)
        y_test_ood = self.generate_target(X_test_ood, self.test_coefficients)

        return (
            X[train_indices], y[train_indices],
            X[val_indices], y[val_indices],
            X_test_ood, y_test_ood
        )

    def create_dataloaders(self) -> Tuple:
        """
        创建数据加载器（重写以正确处理OOD）

        Returns:
            train_loader, val_loader, test_loader
        """
        # 生成训练数据
        X_train = self._generate_samples(self.n_samples, self.train_dist_config)
        y_train = self.generate_target(X_train, self.train_coefficients)
        y_train = self.add_noise(y_train)

        # 划分数据（会自动生成OOD测试集）
        X_train, y_train, X_val, y_val, X_test, y_test = self.split_data(X_train, y_train)

        # 存储为实例变量
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test

        # 创建TensorDataset
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float().unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).float().unsqueeze(1)
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X_test).float(),
            torch.from_numpy(y_test).float().unsqueeze(1)
        )

        # 创建DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return train_loader, val_loader, test_loader

    def get_feature_info(self) -> Dict[str, Any]:
        """
        获取特征信息

        Returns:
            feature_info: 特征信息字典
        """
        return {
            'n_features': 4,
            'feature_names': [f'x{i+1}' for i in range(4)],
            'ood_type': self.ood_type,
            'train_distribution': self.train_dist_config,
            'test_distribution': self.test_dist_config,
            'train_coefficients': self.train_coefficients.tolist(),
            'test_coefficients': self.test_coefficients.tolist()
        }


# 工厂函数
def create_ood_data_generator(config: Dict[str, Any]) -> OODDataGenerator:
    """
    创建OOD数据生成器

    Args:
        config: 配置字典

    Returns:
        generator: 数据生成器实例
    """
    return OODDataGenerator(config)

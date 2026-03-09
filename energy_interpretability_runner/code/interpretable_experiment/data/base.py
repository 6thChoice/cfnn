"""
基础数据类
所有数据生成器都应继承此类
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class BaseDataGenerator(ABC):
    """数据生成器基类"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典，包含：
                - n_samples: 样本数
                - noise_std: 噪声标准差
                - test_split: 测试集比例
                - val_split: 验证集比例
                - batch_size: 批大小
                - random_seed: 随机种子
        """
        self.config = config
        self.n_samples = config['n_samples']
        self.noise_std = config['noise_std']
        self.test_split = config['test_split']
        self.val_split = config['val_split']
        self.batch_size = config['batch_size']
        self.seed = config.get('random_seed', 42)

        # 设置随机种子
        self._set_seed()

        # 数据存储
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

    def _set_seed(self):
        """设置随机种子"""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

    @abstractmethod
    def generate_features(self) -> np.ndarray:
        """
        生成特征矩阵

        Returns:
            X: (n_samples, n_features) 特征矩阵
        """
        pass

    @abstractmethod
    def generate_target(self, X: np.ndarray) -> np.ndarray:
        """
        生成目标变量

        Args:
            X: 特征矩阵

        Returns:
            y: (n_samples,) 目标变量
        """
        pass

    def add_noise(self, y: np.ndarray) -> np.ndarray:
        """
        添加高斯噪声

        Args:
            y: 原始目标变量

        Returns:
            y_noisy: 添加噪声后的目标变量
        """
        noise = np.random.normal(0, self.noise_std, size=y.shape)
        return y + noise

    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        划分数据集

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

        # 随机打乱
        indices = np.random.permutation(n)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]

        return (
            X[train_indices], y[train_indices],
            X[val_indices], y[val_indices],
            X[test_indices], y[test_indices]
        )

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        创建数据加载器

        Returns:
            train_loader, val_loader, test_loader
        """
        # 生成数据
        X = self.generate_features()
        y = self.generate_target(X)
        y = self.add_noise(y)

        # 划分数据
        X_train, y_train, X_val, y_val, X_test, y_test = self.split_data(X, y)

        # 存储为实例变量
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test

        # 创建 TensorDataset
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

        # 创建 DataLoader
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
            feature_info: {
                'n_features': 特征数,
                'feature_names': 特征名称列表,
                'feature_types': 特征类型列表,
                'true_features': 真实特征索引,
                'noise_features': 噪声特征索引,
                ...其他实验特定信息
            }
        """
        return {
            'n_features': self.generate_features().shape[1],
            'feature_names': [f'x{i}' for i in range(self.generate_features().shape[1])],
            'feature_types': ['continuous'] * self.generate_features().shape[1]
        }

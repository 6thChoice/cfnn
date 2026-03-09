"""
因果数据生成器
支持多种因果图结构
"""

from .base import BaseDataGenerator
from typing import Dict, Any, List, Tuple
import numpy as np


class CausalDataGenerator(BaseDataGenerator):
    """因果数据生成器"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典
                - graph_type: 因果图类型
                  * 'chain': 链式结构 (x1 -> x2 -> x3 -> y)
                  * 'fork': 分叉结构 (x3 -> x1, x2 -> y)
                  * 'rational': 分式因果 (x3 -> (x1+x2)/x3 -> y)
                  * 'counterfactual': 反事实 (条件因果)
                - n_samples: 样本数
                - noise_std: 噪声标准差
        """
        super().__init__(config)

        self.graph_type = config.get('graph_type', 'chain')
        self.noise_std = config.get('noise_std', 0.1)

    def generate_chain_graph(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成链式因果图数据
        x1 -> x2 -> x3 -> y

        Args:
            n: 样本数

        Returns:
            X, y
        """
        # 生成根节点
        x1 = np.random.normal(0, 1, n)

        # 生成链式依赖
        x2 = 0.8 * x1 + np.random.normal(0, self.noise_std, n)
        x3 = 0.8 * x2 + np.random.normal(0, self.noise_std, n)
        y = x3 + np.random.normal(0, self.noise_std, n)

        X = np.stack([x1, x2, x3], axis=1)
        return X, y

    def generate_fork_graph(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成分叉因果图数据
              x3
             /   \
           x1     x2 -> y

        Args:
            n: 样本数

        Returns:
            X, y
        """
        # 生成混淆因子
        x3 = np.random.normal(0, 1, n)

        # 生成分支
        x1 = 0.8 * x3 + np.random.normal(0, self.noise_std, n)
        x2 = 0.8 * x3 + np.random.normal(0, self.noise_std, n)
        y = x1 + 0.5 * x2 + np.random.normal(0, self.noise_std, n)

        X = np.stack([x1, x2, x3], axis=1)
        return X, y

    def generate_rational_graph(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成分式因果图数据
        x3 -> (x1 + x2) / x3 -> y

        Args:
            n: 样本数

        Returns:
            X, y
        """
        # 生成特征（避免零值）
        x1 = np.abs(np.random.normal(1, 0.5, n))
        x2 = np.abs(np.random.normal(1, 0.5, n))
        x3 = np.abs(np.random.normal(1, 0.5, n)) + 0.5  # 避免0

        # 分式关系
        numerator = x1 + x2
        denom = x3
        y = numerator / denom + np.random.normal(0, self.noise_std, n)

        X = np.stack([x1, x2, x3], axis=1)
        return X, y

    def generate_counterfactual_graph(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成反事实因果图数据
        y = 1.5*x1 + 0.8*x2 | if x3 > 0
        y = 0.5*x1 + 0.3*x2 | if x3 < 0

        Args:
            n: 样本数

        Returns:
            X, y
        """
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        x3 = np.random.normal(0, 1, n)

        # 条件因果
        mask = x3 > 0
        y = np.zeros(n)
        y[mask] = 1.5 * x1[mask] + 0.8 * x2[mask]
        y[~mask] = 0.5 * x1[~mask] + 0.3 * x2[~mask]

        y += np.random.normal(0, self.noise_std, n)

        X = np.stack([x1, x2, x3], axis=1)
        return X, y

    def generate_features(self) -> np.ndarray:
        """
        生成特征矩阵

        Returns:
            X: 特征矩阵
        """
        if self.graph_type == 'chain':
            X, _ = self.generate_chain_graph(self.n_samples)
        elif self.graph_type == 'fork':
            X, _ = self.generate_fork_graph(self.n_samples)
        elif self.graph_type == 'rational':
            X, _ = self.generate_rational_graph(self.n_samples)
        elif self.graph_type == 'counterfactual':
            X, _ = self.generate_counterfactual_graph(self.n_samples)
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")

        return X

    def generate_target(self, X: np.ndarray) -> np.ndarray:
        """
        生成目标变量

        Args:
            X: 特征矩阵

        Returns:
            y: 目标变量
        """
        n = X.shape[0]

        if self.graph_type == 'chain':
            _, y = self.generate_chain_graph(n)
        elif self.graph_type == 'fork':
            _, y = self.generate_fork_graph(n)
        elif self.graph_type == 'rational':
            _, y = self.generate_rational_graph(n)
        elif self.graph_type == 'counterfactual':
            _, y = self.generate_counterfactual_graph(n)
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")

        return y

    def get_true_causal_order(self) -> List[str]:
        """
        获取真实的因果顺序

        Returns:
            order: 因果顺序列表（从最重要到最不重要）
        """
        if self.graph_type == 'chain':
            return ['x3', 'x2', 'x1']  # x3直接因果
        elif self.graph_type == 'fork':
            return ['x1', 'x2', 'x3']  # x1直接因果，x3是混淆
        elif self.graph_type == 'rational':
            return ['x1', 'x2', 'x3']  # x1,x2在分子，x3在分母
        elif self.graph_type == 'counterfactual':
            return ['x1', 'x2', 'x3']  # x1,x2根据x3条件
        else:
            return ['x1', 'x2', 'x3']

    def get_feature_info(self) -> Dict[str, Any]:
        """
        获取特征信息

        Returns:
            feature_info: 特征信息字典
        """
        return {
            'n_features': 3,
            'feature_names': ['x1', 'x2', 'x3'],
            'graph_type': self.graph_type,
            'true_causal_order': self.get_true_causal_order(),
            'noise_std': self.noise_std,
            'description': self._get_graph_description()
        }

    def _get_graph_description(self) -> str:
        """获取因果图描述"""
        descriptions = {
            'chain': 'x1 -> x2 -> x3 -> y',
            'fork': 'x3 -> x1, x2 -> y (x3是混淆因子)',
            'rational': 'x3 -> (x1+x2)/x3 -> y (分式因果)',
            'counterfactual': 'y = f(x1,x2) | if x3 > 0 else g(x1,x2) | if x3 < 0'
        }
        return descriptions.get(self.graph_type, 'Unknown')


# 工厂函数
def create_causal_data_generator(config: Dict[str, Any]) -> CausalDataGenerator:
    """
    创建因果数据生成器

    Args:
        config: 配置字典

    Returns:
        generator: 数据生成器实例
    """
    return CausalDataGenerator(config)

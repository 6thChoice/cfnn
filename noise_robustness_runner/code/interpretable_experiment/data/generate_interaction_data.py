"""
特征交互数据生成器
支持多种特征交互类型
"""

from .base import BaseDataGenerator
from typing import Dict, Any, List, Tuple
import numpy as np


class InteractionDataGenerator(BaseDataGenerator):
    """特征交互数据生成器"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典
                - interaction_type: 交互类型
                  * 'multiplicative': 乘法交互
                  * 'conditional': 条件交互
                  * 'rational': 分式交互
                  * 'high_order': 高阶交互
                  * 'mixed': 混合交互
                - n_features: 特征数（默认4）
                - interaction_strength: 交互强度（默认0.8）
        """
        super().__init__(config)

        self.interaction_type = config.get('interaction_type', 'multiplicative')
        self.interaction_strength = config.get('interaction_strength', 0.8)
        self.n_features = config.get('n_features', 4)

        # 定义交互对
        self.interaction_pairs = self._define_interactions()

    def _define_interactions(self) -> Dict[str, Any]:
        """定义交互类型的具体配置"""
        if self.interaction_type == 'multiplicative':
            return {
                'name': 'Multiplicative',
                'pairs': [(0, 1)],  # x1 * x2
                'base_features': [2, 3],  # x3, x4 作为基线
                'description': 'x1 * x2 乘法交互'
            }
        elif self.interaction_type == 'conditional':
            return {
                'name': 'Conditional',
                'pairs': [(0, 2)],  # x1 仅在 x3 > 0 时有影响
                'condition': {'feature': 2, 'threshold': 0},
                'base_features': [1, 3],
                'description': 'x1 条件于 x3 的交互'
            }
        elif self.interaction_type == 'rational':
            return {
                'name': 'Rational',
                'pairs': [(2, 3)],  # (x1 + x2) / (1 + x3*x4)
                'numerator_features': [0, 1],
                'denominator_features': [2, 3],
                'description': '分母交互 (CFNet 优势场景)'
            }
        elif self.interaction_type == 'high_order':
            return {
                'name': 'High_Order',
                'pairs': [(0, 1, 2)],  # x1*x2*x3
                'base_features': [3],
                'description': '三阶交互'
            }
        elif self.interaction_type == 'mixed':
            return {
                'name': 'Mixed',
                'interactions': [
                    {'type': 'multiplicative', 'features': [0, 1]},
                    {'type': 'conditional', 'features': [1, 2], 'condition': {'feature': 2, 'threshold': 0}},
                    {'type': 'rational', 'numerator': [0], 'denominator': [3]}
                ],
                'description': '混合多种交互'
            }
        else:
            raise ValueError(f"Unknown interaction type: {self.interaction_type}")

    def generate_features(self) -> np.ndarray:
        """
        生成特征矩阵

        Returns:
            X: (n_samples, n_features) 特征矩阵
        """
        n = self.n_samples

        # 生成基础特征
        if self.interaction_type == 'conditional':
            # 条件交互需要特定的分布以确保有足够的样本满足条件
            X = np.random.randn(n, self.n_features)
        else:
            # 其他交互类型使用均匀分布
            X = np.random.uniform(0.5, 2.0, (n, self.n_features))

        return X

    def generate_target(self, X: np.ndarray) -> np.ndarray:
        """
        生成目标变量（包含交互效应）

        Args:
            X: 特征矩阵

        Returns:
            y: 目标变量
        """
        n = X.shape[0]
        y = np.zeros(n)

        if self.interaction_type == 'multiplicative':
            # y = x1 * x2 + 0.5*x3 + 0.3*x4
            interaction_features = self.interaction_pairs['pairs']
            base_features = self.interaction_pairs['base_features']

            y = self.interaction_strength * X[:, 0] * X[:, 1]
            for feat in base_features:
                y += 0.5 * X[:, feat]

        elif self.interaction_type == 'conditional':
            # y = (x1 if x3 > 0 else 0) + 0.5*x2 + 0.3*x4
            condition_feat = self.interaction_pairs['condition']['feature']
            threshold = self.interaction_pairs['condition']['threshold']
            base_features = self.interaction_pairs['base_features']

            # 条件交互
            mask = X[:, condition_feat] > threshold
            y[mask] = self.interaction_strength * X[mask, 0]

            # 基线特征
            for feat in base_features:
                y += 0.5 * X[:, feat]

        elif self.interaction_type == 'rational':
            # y = (x1 + x2) / (1 + x3*x4)
            num_features = self.interaction_pairs['numerator_features']
            den_features = self.interaction_pairs['denominator_features']

            numerator = X[:, num_features[0]] + X[:, num_features[1]]
            denominator = 1 + X[:, den_features[0]] * X[:, den_features[1]]

            # 添加稳定性，避免除零
            denominator = np.maximum(denominator, 0.1)

            y = self.interaction_strength * numerator / denominator

        elif self.interaction_type == 'high_order':
            # y = x1*x2*x3 + 0.5*x4
            interaction_features = self.interaction_pairs['pairs']
            base_features = self.interaction_pairs['base_features']

            y = self.interaction_strength * X[:, 0] * X[:, 1] * X[:, 2]
            for feat in base_features:
                y += 0.5 * X[:, feat]

        elif self.interaction_type == 'mixed':
            # 混合多种交互
            interactions = self.interaction_pairs['interactions']

            for inter in interactions:
                if inter['type'] == 'multiplicative':
                    feats = inter['features']
                    y += self.interaction_strength * X[:, feats[0]] * X[:, feats[1]]
                elif inter['type'] == 'conditional':
                    feats = inter['features']
                    condition = inter['condition']
                    mask = X[:, condition['feature']] > condition['threshold']
                    y[mask] += self.interaction_strength * X[mask, feats[0]]
                elif inter['type'] == 'rational':
                    numerator = X[:, inter['numerator'][0]]
                    denominator = 1 + X[:, inter['denominator'][0]]
                    denominator = np.maximum(denominator, 0.1)
                    y += self.interaction_strength * numerator / denominator

        return y

    def get_feature_info(self) -> Dict[str, Any]:
        """
        获取特征信息

        Returns:
            feature_info: 特征信息字典
        """
        feature_names = [f'x{i+1}' for i in range(self.n_features)]

        # 标记交互特征
        interaction_features = []
        if self.interaction_type == 'multiplicative':
            interaction_features = [('x1', 'x2')]
        elif self.interaction_type == 'conditional':
            interaction_features = [('x1', 'x3')]
        elif self.interaction_type == 'rational':
            interaction_features = [('x1', 'x2', 'x3', 'x4')]
        elif self.interaction_type == 'high_order':
            interaction_features = [('x1', 'x2', 'x3')]
        elif self.interaction_type == 'mixed':
            interaction_features = [('x1', 'x2'), ('x1', 'x3'), ('x1', 'x4')]

        return {
            'n_features': self.n_features,
            'feature_names': feature_names,
            'interaction_type': self.interaction_type,
            'interaction_strength': self.interaction_strength,
            'interaction_features': interaction_features,
            'description': self.interaction_pairs['description']
        }


# 工厂函数
def create_interaction_data_generator(config: Dict[str, Any]) -> InteractionDataGenerator:
    """
    创建特征交互数据生成器

    Args:
        config: 配置字典

    Returns:
        generator: 数据生成器实例
    """
    return InteractionDataGenerator(config)

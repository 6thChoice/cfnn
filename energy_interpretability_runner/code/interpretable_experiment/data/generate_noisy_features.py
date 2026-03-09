"""
干扰特征数据生成器
生成包含无关、冗余、噪声、欺骗特征的数据
"""

from .base import BaseDataGenerator
from typing import Dict, Any
import numpy as np


class NoisyFeaturesGenerator(BaseDataGenerator):
    """干扰特征数据生成器"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典
                - n_true_features: 真实特征数（默认 4）
                - n_noise_features: 无关特征数（默认 4）
                - n_redundant_features: 冗余特征数（默认 2）
                - n_deceptive_features: 欺骗特征数（默认 1）
                - redundant_noise_std: 冗余特征噪声标准差（默认 0.05）
                - deceptive_corr: 欺骗特征相关性（默认 0.8）
        """
        super().__init__(config)

        self.n_true = config.get('n_true_features', 4)
        self.n_noise = config.get('n_noise_features', 4)
        self.n_redundant = config.get('n_redundant_features', 2)
        self.n_deceptive = config.get('n_deceptive_features', 1)

        self.redundant_noise_std = config.get('redundant_noise_std', 0.05)
        self.deceptive_corr = config.get('deceptive_corr', 0.8)

        # 总特征数
        self.n_features = (
            self.n_true +
            self.n_noise +
            self.n_redundant +
            self.n_deceptive
        )

        # 真实系数
        self.true_coefficients = np.array([1.5, 0.8, 0.5, 1.2])

    def generate_features(self) -> np.ndarray:
        """
        生成特征矩阵

        特征顺序：[真实特征, 无关特征, 冗余特征, 欺骗特征]

        Returns:
            X: (n_samples, n_features) 特征矩阵
        """
        n = self.n_samples

        # 1. 真实特征（0.5 - 2.0）
        X_true = np.random.uniform(0.5, 2.0, (n, self.n_true))

        # 2. 无关特征（0 - 1，完全随机）
        X_noise = np.random.uniform(0, 1, (n, self.n_noise))

        # 3. 冗余特征（真实特征 + 小噪声）
        X_redundant = []
        for i in range(self.n_redundant):
            # 选择前 n_redundant 个真实特征进行复制
            true_feature_idx = i % self.n_true
            redundant = X_true[:, true_feature_idx] + np.random.normal(
                0, self.redundant_noise_std, n
            )
            X_redundant.append(redundant)
        X_redundant = np.column_stack(X_redundant)

        # 4. 欺骗特征（完全随机）
        # 注意：由于数据会在split_data中重新打乱，欺骗特征在训练和测试集上的
        # 区分会在生成目标时处理
        X_deceptive = np.random.uniform(0, 1, (n, self.n_deceptive))

        # 合并所有特征
        X = np.hstack([X_true, X_noise, X_redundant, X_deceptive])

        return X

    def generate_target(self, X: np.ndarray) -> np.ndarray:
        """
        生成目标变量（仅使用真实特征）

        Args:
            X: 特征矩阵

        Returns:
            y: 目标变量
        """
        # 仅使用前 n_true 个特征
        X_true = X[:, :self.n_true]

        # 线性组合
        y = np.dot(X_true, self.true_coefficients)

        return y

    def get_feature_info(self) -> Dict[str, Any]:
        """
        获取特征信息

        Returns:
            feature_info: 特征信息字典
        """
        feature_names = []
        feature_types = []
        feature_groups = {}

        idx = 0

        # 真实特征
        for i in range(self.n_true):
            feature_names.append(f'x{i+1}_true')
            feature_types.append('true')
            feature_groups[f'x{i+1}_true'] = 'true'
            idx += 1

        # 无关特征
        for i in range(self.n_noise):
            feature_names.append(f'x{i+1}_noise')
            feature_types.append('noise')
            feature_groups[f'x{i+1}_noise'] = 'noise'
            idx += 1

        # 冗余特征
        for i in range(self.n_redundant):
            feature_names.append(f'x{i+1}_redundant')
            feature_types.append('redundant')
            # 记录它复制了哪个真实特征
            source_idx = i % self.n_true
            feature_groups[f'x{i+1}_redundant'] = f'x{source_idx+1}_true'
            idx += 1

        # 欺骗特征
        for i in range(self.n_deceptive):
            feature_names.append(f'x{i+1}_deceptive')
            feature_types.append('deceptive')
            feature_groups[f'x{i+1}_deceptive'] = 'deceptive'
            idx += 1

        return {
            'n_features': self.n_features,
            'n_true_features': self.n_true,
            'n_noise_features': self.n_noise,
            'n_redundant_features': self.n_redundant,
            'n_deceptive_features': self.n_deceptive,
            'feature_names': feature_names,
            'feature_types': feature_types,
            'feature_groups': feature_groups,
            'true_coefficients': self.true_coefficients.tolist(),
            'true_feature_indices': list(range(self.n_true)),
            'noise_feature_indices': list(range(
                self.n_true,
                self.n_true + self.n_noise
            )),
            'redundant_feature_indices': list(range(
                self.n_true + self.n_noise,
                self.n_true + self.n_noise + self.n_redundant
            )),
            'deceptive_feature_indices': list(range(
                self.n_true + self.n_noise + self.n_redundant,
                self.n_features
            ))
        }


def create_noisy_data_generator(config: Dict[str, Any]) -> NoisyFeaturesGenerator:
    """
    创建干扰特征数据生成器

    Args:
        config: 配置字典

    Returns:
        generator: 数据生成器实例
    """
    return NoisyFeaturesGenerator(config)

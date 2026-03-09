"""
方案3配置：分布外泛化稳定性实验
"""

from .base_config import BASE_CONFIG


def get_config(ood_type: str = 'covariate_shift'):
    """
    获取方案3的完整配置

    Args:
        ood_type: OOD类型
            - 'covariate_shift': 协变量偏移
            - 'concept_drift': 概念漂移
            - 'sparse_region': 稀疏区域
    """
    config = BASE_CONFIG.copy()
    config['models'] = BASE_CONFIG['models'].copy()

    # OOD场景配置
    ood_scenarios = {
        'covariate_shift': {
            'train': {'type': 'uniform', 'low': 0.5, 'high': 2.0},
            'test': {'type': 'uniform', 'low': 2.0, 'high': 3.5}
        },
        'concept_drift': {
            'train': {'type': 'uniform', 'low': 0.5, 'high': 2.0},
            'test': {'type': 'uniform', 'low': 0.5, 'high': 2.0}
        },
        'sparse_region': {
            'train': {'type': 'uniform', 'low': 0.0, 'high': 1.0},
            'test': {'type': 'uniform', 'low': 3.0, 'high': 4.0}
        }
    }

    # 数据配置
    config['data'] = {
        **{k: v for k, v in BASE_CONFIG.items()
           if k in ['device', 'n_samples', 'noise_std', 'test_split', 'val_split', 'random_seed', 'batch_size']},
        'n_features': 4,
        'ood_type': ood_type,
        'train_distribution': ood_scenarios[ood_type]['train'],
        'test_distribution': ood_scenarios[ood_type]['test']
    }

    return config


def get_all_ood_configs():
    """
    获取所有OOD类型的配置

    Returns:
        configs: OOD类型到配置的映射
    """
    ood_types = ['covariate_shift', 'concept_drift', 'sparse_region']

    return {otype: get_config(otype) for otype in ood_types}

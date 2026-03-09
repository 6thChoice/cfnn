"""
方案 5 配置：真实数据集基准测试
"""

from .base_config import BASE_CONFIG


def get_config():
    """获取方案 5 的完整配置"""
    config = BASE_CONFIG.copy()
    config['models'] = BASE_CONFIG['models'].copy()

    # 数据配置
    config['data'] = {
        'dataset_name': 'energy_efficiency',
        'target_column': 'Y1',  # 采暖负荷
        'n_samples': 5000,
        'noise_std': 0.05,
        'test_split': 0.2,
        'val_split': 0.05,
        'random_seed': 42,
        'batch_size': 64,
        'normalize': True
    }

    return config

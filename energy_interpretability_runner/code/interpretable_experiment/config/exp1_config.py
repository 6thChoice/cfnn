"""
方案 1 配置：干扰鲁棒性实验
"""

from .base_config import BASE_CONFIG


def get_config():
    """获取方案 1 的完整配置"""
    config = BASE_CONFIG.copy()
    config['models'] = BASE_CONFIG['models'].copy()

    # 数据配置
    config['data'] = {
        'n_samples': config['n_samples'],
        'noise_std': config['noise_std'],
        'test_split': config['test_split'],
        'val_split': config['val_split'],
        'random_seed': config['random_seed'],
        'batch_size': config['batch_size'],
        'n_true_features': 4,
        'n_noise_features': 4,
        'n_redundant_features': 2,
        'n_deceptive_features': 1,
        'redundant_noise_std': 0.05,
        'deceptive_corr': 0.8
    }

    # 减少训练时间以便快速验证
    config['epochs'] = 50  # 减少到 50 epochs
    config['models']['boost']['num_stages'] = 5  # 减少到 5 个阶段
    config['models']['boost']['epochs_per_stage'] = 20  # 减少到 20 epochs

    return config

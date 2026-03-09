"""
方案 1 消融实验配置：不同噪声比例下的鲁棒性测试
"""

from .base_config import BASE_CONFIG


def get_ablation_config(noise_ratio=0.5, n_total_features=20):
    """
    获取消融实验配置

    Args:
        noise_ratio: 噪声特征占总特征的比例 (0.0 - 0.9)
        n_total_features: 总特征数（默认20）

    Returns:
        config: 实验配置
    """
    config = BASE_CONFIG.copy()
    config['models'] = BASE_CONFIG['models'].copy()

    # 固定真实特征数为4
    n_true = 4

    # 根据噪声比例计算噪声特征数
    # 总特征 = 真实 + 噪声 + 冗余(2) + 欺骗(1)
    n_noise = int(n_total_features * noise_ratio) - 3
    n_noise = max(0, n_noise)  # 至少0个

    # 确保总特征数不超过设定
    n_redundant = 2
    n_deceptive = 1
    actual_total = n_true + n_noise + n_redundant + n_deceptive

    # 数据配置
    config['data'] = {
        'n_samples': config['n_samples'],
        'noise_std': config['noise_std'],
        'test_split': config['test_split'],
        'val_split': config['val_split'],
        'random_seed': config['random_seed'],
        'batch_size': config['batch_size'],
        'n_true_features': n_true,
        'n_noise_features': n_noise,
        'n_redundant_features': n_redundant,
        'n_deceptive_features': n_deceptive,
        'redundant_noise_std': 0.05,
        'deceptive_corr': 0.8,
        'noise_ratio': noise_ratio,  # 记录噪声比例
        'n_total_features': actual_total
    }

    # 训练配置（与exp1一致）
    config['epochs'] = 50
    config['models']['boost']['num_stages'] = 5
    config['models']['boost']['epochs_per_stage'] = 20

    return config


def get_noise_ratio_sweep_configs():
    """
    获取噪声比例扫描配置列表
    测试从0%到90%的噪声比例

    Returns:
        configs: 配置列表 [(noise_ratio, config), ...]
    """
    noise_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    configs = []

    for ratio in noise_ratios:
        config = get_ablation_config(noise_ratio=ratio, n_total_features=20)
        configs.append((ratio, config))

    return configs

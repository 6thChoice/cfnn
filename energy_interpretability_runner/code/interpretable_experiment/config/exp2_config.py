"""
方案2配置：特征交互捕获实验
"""

from .base_config import BASE_CONFIG


def get_config(interaction_type: str = 'multiplicative'):
    """
    获取方案2的完整配置

    Args:
        interaction_type: 交互类型
            - 'multiplicative': 乘法交互
            - 'conditional': 条件交互
            - 'rational': 分式交互（CFNet优势场景）
            - 'high_order': 高阶交互
            - 'mixed': 混合交互
    """
    config = BASE_CONFIG.copy()
    config['models'] = BASE_CONFIG['models'].copy()

    # 数据配置
    config['data'] = {
        **{k: v for k, v in BASE_CONFIG.items()
           if k in ['device', 'n_samples', 'noise_std', 'test_split', 'val_split', 'random_seed', 'batch_size']},
        'n_features': 4,
        'interaction_type': interaction_type,
        'interaction_strength': 0.8
    }

    return config


def get_all_interaction_configs():
    """
    获取所有交互类型的配置

    Returns:
        configs: 交互类型到配置的映射
    """
    interaction_types = [
        'multiplicative',
        'conditional',
        'rational',  # CFNet优势场景
        'high_order',
        'mixed'
    ]

    return {itype: get_config(itype) for itype in interaction_types}

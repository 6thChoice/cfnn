"""
方案4配置：因果结构发现实验
"""

from .base_config import BASE_CONFIG


def get_config(graph_type: str = 'rational'):
    """
    获取方案4的完整配置

    Args:
        graph_type: 因果图类型
            - 'chain': 链式结构
            - 'fork': 分叉结构
            - 'rational': 分式因果（CFNet优势场景）
            - 'counterfactual': 反事实
    """
    config = BASE_CONFIG.copy()
    config['models'] = BASE_CONFIG['models'].copy()

    # 数据配置
    config['data'] = {
        **{k: v for k, v in BASE_CONFIG.items()
           if k in ['device', 'n_samples', 'test_split', 'val_split', 'random_seed', 'batch_size']},
        'graph_type': graph_type,
        'noise_std': 0.1
    }

    return config

"""
基础配置
所有实验共享的配置
"""

import torch
from pathlib import Path

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

runner_base = Path(__file__).resolve().parents[3]
BASE_CONFIG = {
    # 设备
    'device': DEVICE,

    # 通用训练配置
    'lr': 0.001,
    'epochs': 100,
    'patience': 15,
    'batch_size': 64,

    # 数据配置
    'n_samples': 5000,
    'noise_std': 0.05,
    'test_split': 0.3,
    'val_split': 0.05,
    'random_seed': 42,

    # 结果目录
    'base_result_dir': str(runner_base / "results"),

    # 模型配置
    'models': {
        'standard': {
            'depth': 4,
            'poly_degree': 4
        },
        'hybrid': {
            'unit_degree': 4,
            'num_units': 20
        },
        'boost': {
            'shallow_depth': 4,
            'poly_degree': 4,
            'num_stages': 20,
            'epochs_per_stage': 50
        },
        'moe': {
            'shallow_depth_per_cofrnet': 2,
            'polynomial_degree': 2,
            'num_experts': 5
        }
    }
}


def get_config():
    """获取基础配置"""
    return BASE_CONFIG.copy()

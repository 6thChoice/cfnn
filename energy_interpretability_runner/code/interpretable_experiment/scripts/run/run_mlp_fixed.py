#!/usr/bin/env python3
"""
运行修复后的MLP帕累托实验
"""

import sys
import os

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'xai'))

import json
import torch
import torch.nn as nn
from datetime import datetime
from interpretable_experiment.config.exp1_pareto_config import (
    get_pareto_experiment_config,
    get_pareto_model_config
)
from interpretable_experiment.data.generate_noisy_features import NoisyFeaturesGenerator
from func_experiment.engine import BaseTrainer
from pathlib import Path
RUNNER_BASE = Path(__file__).resolve().parents[4]
BASE_RESULT_DIR = str(RUNNER_BASE / "results")

# 直接定义MLP模型，避免导入问题
class BaselineMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=2):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.Tanh())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_mlp_experiment():
    """运行完整的MLP实验（30个配置）"""

    print("="*80)
    print("MLP修复实验：验证参数对齐修复")
    print("="*80)

    # 噪声比例和参数量级别
    noise_ratios = [0.2, 0.4, 0.6, 0.8, 0.9]
    pareto_levels = [50, 100, 200, 400, 800, 1200]

    # 创建结果目录
    result_dir = os.path.join(
        BASE_RESULT_DIR,
        'exp1_pareto_mlp_fixed',
        datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    os.makedirs(result_dir, exist_ok=True)

    print(f"\n结果保存目录: {result_dir}")
    print(f"实验配置: {len(noise_ratios)}噪声 × {len(pareto_levels)}参数量 = {len(noise_ratios)*len(pareto_levels)}个实验\n")

    results = []
    total = len(noise_ratios) * len(pareto_levels)

    for i, noise_ratio in enumerate(noise_ratios):
        for j, target_params in enumerate(pareto_levels):
            idx = i * len(pareto_levels) + j + 1

            print(f"\n{'#'*80}")
            print(f"# 进度: [{idx}/{total}] 噪声{noise_ratio:.0%} 目标参数量{target_params}")
            print(f"{'#'*80}")

            try:
                # 获取配置
                exp_config = get_pareto_experiment_config(noise_ratio, target_params)
                n_features = exp_config['data']['n_total_features']

                # 获取MLP配置（使用修复后的配置生成器）
                model_config = get_pareto_model_config('MLP', target_params, n_features, 1)

                print(f"\n目标参数量: {target_params}")
                print(f"配置: hidden_dim={model_config['hidden_dim']}, num_layers={model_config.get('num_layers', 2)}")

                # 准备数据
                data_generator = NoisyFeaturesGenerator(exp_config['data'])
                train_loader, val_loader, test_loader = data_generator.create_dataloaders()

                # 创建模型
                model = BaselineMLP(
                    n_features,
                    1,
                    model_config['hidden_dim'],
                    model_config.get('num_layers', 2)
                )

                # 计算实际参数量
                actual_params = count_parameters(model)
                print(f"实际参数量: {actual_params}")
                print(f"偏差: {abs(actual_params - target_params) / target_params * 100:.1f}%")

                # 训练
                device = exp_config['device']
                model = model.to(device)
                trainer = BaseTrainer(model, device=device, lr=exp_config['lr'])

                history = trainer.fit(
                    train_loader, val_loader,
                    epochs=exp_config['epochs'],
                    patience=exp_config['patience']
                )

                # 评估
                test_mse = trainer.evaluate(test_loader)
                print(f"Test MSE: {test_mse:.6f}")

                # 保存结果
                result = {
                    'noise_ratio': noise_ratio,
                    'target_params': target_params,
                    'actual_params': actual_params,
                    'hidden_dim': model_config['hidden_dim'],
                    'test_mse': float(test_mse)
                }
                results.append(result)

                # 保存到文件
                point_dir = os.path.join(
                    result_dir,
                    f'noise_{int(noise_ratio*100)}pct',
                    f'params_{target_params}'
                )
                os.makedirs(point_dir, exist_ok=True)

                with open(os.path.join(point_dir, 'result.json'), 'w') as f:
                    json.dump(result, f, indent=2)

                torch.save(model.state_dict(), os.path.join(point_dir, 'model.pt'))

            except Exception as e:
                print(f"\n错误: {e}")
                import traceback
                traceback.print_exc()
                continue

    # 保存汇总
    summary_path = os.path.join(result_dir, 'mlp_fixed_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print(f"实验完成! 成功: {len(results)}/{total}")
    print(f"结果保存于: {result_dir}")
    print(f"汇总文件: {summary_path}")
    print("="*80)

    return results, result_dir


if __name__ == '__main__':
    results, result_dir = run_mlp_experiment()
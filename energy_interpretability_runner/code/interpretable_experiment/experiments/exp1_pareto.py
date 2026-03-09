"""
方案1帕累托前沿实验：参数量-性能权衡分析

在不同噪声比例下，测试各模型在不同参数量级别的性能表现，
绘制帕累托前沿曲线（X轴：参数量，Y轴：测试MSE）
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from datetime import datetime
from collections import defaultdict

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# 导入CFNet模型
from cfnet import (
    CFNet_Standard,
    HybridRationalNet,
    EnsembleResCoFrNet,
    MoE_Ensemble
)

# 导入其他模块
from func_experiment.models_ext import count_parameters
from func_experiment.engine import BaseTrainer, BoostTrainer
from interpretable_experiment.data.generate_noisy_features import NoisyFeaturesGenerator
from interpretable_experiment.models.kan_model import KAN, count_parameters as count_kan_params
from interpretable_experiment.config.exp1_pareto_config import (
from pathlib import Path
RUNNER_BASE = Path(__file__).resolve().parents[4]
BASE_RESULT_DIR = str(RUNNER_BASE / "results")
    get_pareto_model_config,
    get_pareto_experiment_config,
    get_all_pareto_configs,
    get_quick_test_configs,
    PARETO_LEVELS,
    NOISE_RATIOS,
    count_params_standard,
    count_params_hybrid,
    count_params_boost_per_stage,
    count_params_moe_per_expert,
    count_params_moe_gating
)


# ==========================================
# 1. 模型包装和创建
# ==========================================

class ModelWrapper(nn.Module):
    """模型包装器，统一接口"""

    def __init__(self, model):
        super().__init__()
        object.__setattr__(self, '_model', model)

    @property
    def model(self):
        return object.__getattribute__(self, '_model')

    def forward(self, x):
        output = self.model(x)
        if output.dim() == 1:
            output = output.unsqueeze(-1)
        return output

    def parameters(self, recurse=True):
        return self.model.parameters(recurse)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.model.load_state_dict(state_dict, *args, **kwargs)

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.model.to(device)
        return self

    @property
    def models(self):
        return self.model.models

    def add_model(self):
        return self.model.add_model()

    def freeze_all_but_latest(self):
        return self.model.freeze_all_but_latest()

    def add_expert(self):
        return self.model.add_expert()


def create_model_from_config(model_config: Dict[str, Any], n_features: int) -> nn.Module:
    """
    根据配置创建模型实例

    Args:
        model_config: 模型配置（来自get_pareto_model_config）
        n_features: 输入特征数

    Returns:
        model: 模型实例
    """
    model_type = model_config['type']
    output_dim = 1

    if model_type == 'mlp':
        from func_experiment.models_ext import BaselineMLP
        model = BaselineMLP(
            n_features,
            output_dim,
            model_config['hidden_dim'],
            model_config.get('num_layers', 2)
        )

    elif model_type == 'kan':
        from interpretable_experiment.models.kan_model import KANLayer
        # 构建KAN网络
        layers = []
        hidden_dim = model_config['hidden_dim']
        num_layers = model_config.get('num_layers', 2)
        grid_size = model_config['grid_size']
        spline_order = model_config['spline_order']

        dims = [n_features] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(KANLayer(dims[i], dims[i+1], grid_size, spline_order))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i+1]))

        model = nn.Sequential(*layers)
        # 包装为KAN类以兼容
        model.input_dim = n_features
        model.output_dim = output_dim

    elif model_type == 'standard':
        model = CFNet_Standard(
            n_features,
            output_dim,
            depth=model_config['depth'],
            poly_degree=model_config['poly_degree']
        )

    elif model_type == 'hybrid':
        model = HybridRationalNet(
            n_features,
            output_dim,
            unit_degree=model_config['unit_degree'],
            num_units=model_config['num_units']
        )

    elif model_type == 'boost':
        model = EnsembleResCoFrNet(
            n_features,
            output_dim,
            shallow_depth=model_config['shallow_depth'],
            poly_degree=model_config['poly_degree'],
            learning_rate=0.5
        )
        # 添加所有stages
        for _ in range(model_config['num_stages']):
            model.add_model()

    elif model_type == 'moe':
        model = MoE_Ensemble({
            'input_dim': n_features,
            'output_dim': output_dim,
            'shallow_depth_per_cofrnet': model_config['shallow_depth'],
            'polynomial_degree': model_config['poly_degree']
        })
        # 添加所有experts
        for _ in range(model_config['num_experts']):
            model.add_expert()

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return ModelWrapper(model)


# ==========================================
# 2. 单点实验执行
# ==========================================

def run_single_pareto_point(
    noise_ratio: float,
    target_params: int,
    model_name: str,
    result_dir: str,
    quick_mode: bool = False
) -> Dict[str, Any]:
    """
    运行单个帕累托点的实验

    Args:
        noise_ratio: 噪声比例
        target_params: 目标参数量
        model_name: 模型名称
        result_dir: 结果保存目录
        quick_mode: 是否快速模式

    Returns:
        result: 包含实验结果的字典
    """
    print(f"\n{'-'*60}")
    print(f"噪声比例: {noise_ratio:.0%} | 目标参数量: {target_params} | 模型: {model_name}")
    print(f"{'-'*60}")

    # 1. 获取配置
    experiment_config = get_pareto_experiment_config(noise_ratio, target_params)
    device = experiment_config['device']

    # 设置Boost/MoE的stage/expert数量
    kwargs = {}
    if model_name.lower() == 'boost':
        if target_params <= 100:
            kwargs['num_stages'] = 5
        elif target_params <= 400:
            kwargs['num_stages'] = 10
        else:
            kwargs['num_stages'] = 20
    elif model_name.lower() == 'moe':
        if target_params <= 100:
            kwargs['num_experts'] = 3
        elif target_params <= 400:
            kwargs['num_experts'] = 5
        else:
            kwargs['num_experts'] = 8

    n_features = experiment_config['data']['n_total_features']
    model_config = get_pareto_model_config(
        model_name, target_params, n_features, 1, **kwargs
    )

    # 快速模式调整
    if quick_mode:
        experiment_config['n_samples'] = 500
        experiment_config['epochs'] = 10
        experiment_config['models']['boost']['num_stages'] = 3
        experiment_config['models']['boost']['epochs_per_stage'] = 10

    # 2. 准备数据
    print("[1/3] 准备数据...")
    data_generator = NoisyFeaturesGenerator(experiment_config['data'])
    train_loader, val_loader, test_loader = data_generator.create_dataloaders()
    feature_info = data_generator.get_feature_info()

    print(f"  总特征数: {feature_info['n_features']}")
    print(f"  训练样本: {len(train_loader.dataset)}")

    # 3. 创建模型
    print("[2/3] 创建模型...")
    model = create_model_from_config(model_config, n_features).to(device)

    # 计算实际参数量
    actual_params = count_parameters(model.model)
    print(f"  目标参数量: {target_params}")
    print(f"  实际参数量: {actual_params}")
    print(f"  偏差: {abs(actual_params - target_params) / target_params * 100:.1f}%")

    # 4. 训练模型
    print("[3/3] 训练模型...")
    if model_name == 'Boost':
        trainer = BoostTrainer(model, device=device, lr=experiment_config['lr'])
        num_stages = model_config.get('num_stages', 10)
        epochs_per_stage = experiment_config['models']['boost']['epochs_per_stage']
        history = trainer.fit_boosting(
            train_loader, val_loader,
            num_stages=num_stages,
            epochs_per_stage=epochs_per_stage
        )
    else:
        trainer = BaseTrainer(model, device=device, lr=experiment_config['lr'])
        history = trainer.fit(
            train_loader, val_loader,
            epochs=experiment_config['epochs'],
            patience=experiment_config['patience']
        )

    # 5. 评估
    test_mse = trainer.evaluate(test_loader)
    print(f"  Test MSE: {test_mse:.6f}")

    # 6. 保存结果
    result = {
        'noise_ratio': noise_ratio,
        'target_params': target_params,
        'actual_params': actual_params,
        'model_name': model_name,
        'model_config': model_config,
        'test_mse': test_mse,
        'history': history,
        'feature_info': feature_info
    }

    # 保存到文件
    point_dir = os.path.join(
        result_dir,
        f'noise_{int(noise_ratio*100)}pct',
        f'params_{target_params}',
        model_name
    )
    os.makedirs(point_dir, exist_ok=True)

    with open(os.path.join(point_dir, 'result.json'), 'w') as f:
        # 保存可序列化的结果
        json.dump({
            'noise_ratio': noise_ratio,
            'target_params': target_params,
            'actual_params': actual_params,
            'model_name': model_name,
            'test_mse': float(test_mse),
            'model_config': {k: v for k, v in model_config.items() if k != 'config'}
        }, f, indent=2)

    # 保存模型
    torch.save(model.state_dict(), os.path.join(point_dir, 'model.pt'))

    return result


# ==========================================
# 3. 完整实验执行
# ==========================================

def run_pareto_experiment(quick_mode: bool = False, models: List[str] = None) -> Dict[str, Any]:
    """
    运行完整的帕累托前沿实验

    Args:
        quick_mode: 是否快速模式
        models: 要测试的模型列表，None表示所有模型

    Returns:
        all_results: 所有实验结果
    """
    print("="*80)
    print("方案1帕累托前沿实验：参数量-性能权衡分析")
    print("="*80)

    if quick_mode:
        print("\n【快速测试模式】")
        print("  - 样本数: 500")
        print("  - 训练epoch: 10")
        print("  - 仅测试部分配置")
        configs = get_quick_test_configs()
    else:
        print("\n【完整实验模式】")
        configs = get_all_pareto_configs(models=models)

    # 创建结果目录
    result_dir = os.path.join(
        BASE_RESULT_DIR,
        'exp1_pareto_quick' if quick_mode else 'exp1_pareto',
        datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    os.makedirs(result_dir, exist_ok=True)

    print(f"\n结果保存目录: {result_dir}")
    print(f"实验配置数: {len(configs)}")

    # 运行所有配置
    all_results = defaultdict(lambda: defaultdict(list))

    for i, config in enumerate(configs, 1):
        noise_ratio = config['noise_ratio']
        target_params = config['target_params']
        model_name = config['model_name']

        print(f"\n\n{'#'*80}")
        print(f"# 进度: [{i}/{len(configs)}]")
        print(f"# 噪声比例: {noise_ratio:.0%} | 参数量: {target_params} | 模型: {model_name}")
        print(f"{'#'*80}")

        try:
            result = run_single_pareto_point(
                noise_ratio, target_params, model_name, result_dir, quick_mode
            )

            # 组织结果
            all_results[noise_ratio][model_name].append({
                'target_params': target_params,
                'actual_params': result['actual_params'],
                'test_mse': result['test_mse']
            })

        except Exception as e:
            print(f"\n错误: 实验失败 - {e}")
            import traceback
            traceback.print_exc()
            continue

    # 生成汇总报告和可视化
    generate_pareto_summary(all_results, result_dir)
    plot_pareto_frontiers(all_results, result_dir)

    print("\n" + "="*80)
    print("帕累托前沿实验完成!")
    print(f"所有结果保存在: {result_dir}")
    print("="*80)

    return dict(all_results)


# ==========================================
# 4. 结果汇总和报告
# ==========================================

def generate_pareto_summary(all_results: Dict[float, Dict[str, List]], result_dir: str):
    """
    生成帕累托实验汇总报告

    Args:
        all_results: 所有实验结果
        result_dir: 结果目录
    """
    print("\n" + "="*80)
    print("生成汇总报告...")
    print("="*80)

    # 1. 创建汇总DataFrame
    summary_data = []

    for noise_ratio, models_data in all_results.items():
        for model_name, points in models_data.items():
            for point in points:
                summary_data.append({
                    'noise_ratio': noise_ratio,
                    'model': model_name,
                    'target_params': point['target_params'],
                    'actual_params': point['actual_params'],
                    'test_mse': point['test_mse']
                })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(result_dir, 'pareto_summary.csv'), index=False)

    # 2. 生成Markdown报告
    report_lines = []
    report_lines.append("# 参数量-性能帕累托前沿实验报告\n")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("="*80 + "\n\n")

    # 实验设置
    report_lines.append("## 实验设置\n")
    report_lines.append(f"- 噪声比例: {sorted(all_results.keys())}\n")
    report_lines.append(f"- 参数量级别: {PARETO_LEVELS}\n")
    report_lines.append(f"- 对比模型: MLP, KAN, Standard, Hybrid, Boost, MoE\n\n")

    # 各噪声比例下的结果表
    for noise_ratio in sorted(all_results.keys()):
        report_lines.append(f"\n## 噪声比例 {noise_ratio:.0%}\n")

        # 创建透视表
        noise_data = summary_df[summary_df['noise_ratio'] == noise_ratio]

        report_lines.append("### 测试MSE\n")
        report_lines.append("| 参数量 | MLP | KAN | Standard | Hybrid | Boost | MoE |\n")
        report_lines.append("|--------|-----|-----|----------|--------|-------|-----|\n")

        for target in PARETO_LEVELS:
            line = f"| {target} |"
            for model in ['MLP', 'KAN', 'Standard', 'Hybrid', 'Boost', 'MoE']:
                model_data = noise_data[
                    (noise_data['model'] == model) &
                    (noise_data['target_params'] == target)
                ]
                if not model_data.empty:
                    mse = model_data['test_mse'].values[0]
                    line += f" {mse:.4f} |"
                else:
                    line += " - |"
            report_lines.append(line + "\n")

    # 帕累托优势分析
    report_lines.append("\n## 帕累托优势分析\n")

    for noise_ratio in sorted(all_results.keys()):
        report_lines.append(f"\n### 噪声比例 {noise_ratio:.0%}\n")

        # 找出每个参数量级别的最佳模型
        noise_data = summary_df[summary_df['noise_ratio'] == noise_ratio]

        for target in PARETO_LEVELS:
            target_data = noise_data[noise_data['target_params'] == target]
            if not target_data.empty:
                best = target_data.loc[target_data['test_mse'].idxmin()]
                report_lines.append(f"- 参数量 {target}: **{best['model']}** (MSE={best['test_mse']:.4f})\n")

    # 保存报告
    report_path = os.path.join(result_dir, 'pareto_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)

    print(f"报告已保存到: {report_path}")


# ==========================================
# 5. 可视化
# ==========================================

def plot_pareto_frontiers(all_results: Dict[float, Dict[str, List]], result_dir: str):
    """
    绘制帕累托前沿曲线

    为每个噪声比例绘制一个子图，X轴为参数量，Y轴为测试MSE
    """
    print("\n" + "="*80)
    print("生成帕累托前沿图...")
    print("="*80)

    noise_ratios = sorted(all_results.keys())
    n_plots = len(noise_ratios)

    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # 颜色映射
    colors = {
        'MLP': '#1f77b4',
        'KAN': '#ff7f0e',
        'Standard': '#2ca02c',
        'Hybrid': '#d62728',
        'Boost': '#9467bd',
        'MoE': '#8c564b'
    }

    # 线型
    markers = {
        'MLP': 'o',
        'KAN': 's',
        'Standard': '^',
        'Hybrid': 'v',
        'Boost': 'D',
        'MoE': 'p'
    }

    for idx, noise_ratio in enumerate(noise_ratios):
        ax = axes[idx]
        models_data = all_results[noise_ratio]

        for model_name, points in models_data.items():
            if not points:
                continue

            # 按参数量排序
            sorted_points = sorted(points, key=lambda x: x['actual_params'])
            params = [p['actual_params'] for p in sorted_points]
            mses = [p['test_mse'] for p in sorted_points]

            ax.plot(params, mses,
                   label=model_name,
                   color=colors.get(model_name, 'gray'),
                   marker=markers.get(model_name, 'o'),
                   markersize=8,
                   linewidth=2)

        ax.set_xlabel('Number of Parameters', fontsize=12)
        ax.set_ylabel('Test MSE (log scale)', fontsize=12)
        ax.set_title(f'Noise Ratio {noise_ratio:.0%}', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    # 隐藏多余的子图
    for idx in range(len(noise_ratios), 6):
        axes[idx].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(result_dir, 'pareto_frontiers.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"帕累托前沿图已保存到: {plot_path}")

    # 绘制综合对比图（所有噪声比例在一起）
    plot_combined_pareto(all_results, result_dir, colors, markers)


def plot_combined_pareto(all_results: Dict[float, Dict[str, List]], result_dir: str,
                         colors: Dict[str, str], markers: Dict[str, str]):
    """
    绘制综合帕累托对比图

    每个模型一个子图，显示在不同噪声比例下的表现
    """
    print("生成综合对比图...")

    # 获取所有模型
    all_models = set()
    for noise_data in all_results.values():
        all_models.update(noise_data.keys())
    all_models = sorted(all_models)

    n_models = len(all_models)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # 噪声比例的颜色
    noise_colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))

    for idx, model_name in enumerate(all_models):
        ax = axes[idx]

        for noise_idx, (noise_ratio, noise_data) in enumerate(sorted(all_results.items())):
            if model_name not in noise_data:
                continue

            points = noise_data[model_name]
            if not points:
                continue

            sorted_points = sorted(points, key=lambda x: x['actual_params'])
            params = [p['actual_params'] for p in sorted_points]
            mses = [p['test_mse'] for p in sorted_points]

            ax.plot(params, mses,
                   label=f'{noise_ratio:.0%} noise',
                   color=noise_colors[noise_idx],
                   marker='o',
                   markersize=6,
                   linewidth=2)

        ax.set_xlabel('Number of Parameters', fontsize=12)
        ax.set_ylabel('Test MSE (log scale)', fontsize=12)
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # 隐藏多余的子图
    for idx in range(n_models, 6):
        axes[idx].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(result_dir, 'pareto_by_model.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"综合对比图已保存到: {plot_path}")


# ==========================================
# 6. 主入口
# ==========================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        run_pareto_experiment(quick_mode=False)
    else:
        print("运行快速测试（添加 --full 参数运行完整实验）")
        run_pareto_experiment(quick_mode=True)
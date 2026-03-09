"""
方案 1 消融实验：不同噪声比例下的鲁棒性测试

测试不同噪声比例（0%, 20%, 40%, 60%, 80%, 90%）下各模型的表现：
- MLP
- CFNet-Standard
- CFNet-Hybrid
- CFNet-Boost
- CFNet-MoE
- KAN
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
from datetime import datetime

# 添加项目路径（xai的父目录是paper，cfnet.py在paper下）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# 导入CFNet模型
from cfnet import (
    CFNet_Standard,
    HybridRationalNet,
    EnsembleResCoFrNet,
    MoE_Ensemble
)

# 导入其他模块（func_experiment在xai下）
from func_experiment.models_ext import get_aligned_mlp, count_parameters
from func_experiment.engine import BaseTrainer, BoostTrainer
from interpretable_experiment.data.generate_noisy_features import NoisyFeaturesGenerator
from interpretable_experiment.analysis.robustness_analyzer import RobustnessAnalyzer
from interpretable_experiment.models.kan_model import get_aligned_kan
from pathlib import Path
RUNNER_BASE = Path(__file__).resolve().parents[4]
BASE_RESULT_DIR = str(RUNNER_BASE / "results")


class ModelWrapper(nn.Module):
    """模型包装器"""
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


def setup_models(n_features: int, config: Dict[str, Any], device: str) -> Dict[str, nn.Module]:
    """
    设置所有模型

    Args:
        n_features: 输入特征数
        config: 配置字典
        device: 设备

    Returns:
        models: 模型字典
    """
    # 以 Standard CFNet 为锚点进行参数对齐
    n_true = 4  # 真实特征数
    anchor = CFNet_Standard(n_true, 1, **config['models']['standard'])
    target_params = count_parameters(anchor)

    # 创建MLP（参数对齐）
    mlp, _ = get_aligned_mlp(n_features, 1, target_params)

    # 创建KAN（参数对齐）
    kan, kan_hidden = get_aligned_kan(n_features, 1, target_params)

    # 创建所有模型
    models_dict = {
        'MLP': mlp,
        'KAN': kan,
        'Standard': CFNet_Standard(n_features, 1, **config['models']['standard']),
        'Hybrid': HybridRationalNet(n_features, 1, **config['models']['hybrid']),
        'Boost': EnsembleResCoFrNet(
            n_features, 1,
            config['models']['boost']['shallow_depth'],
            config['models']['boost']['poly_degree'],
            learning_rate=0.5
        ),
        'MoE': MoE_Ensemble({
            'input_dim': n_features,
            'output_dim': 1,
            'shallow_depth_per_cofrnet': config['models']['moe']['shallow_depth_per_cofrnet'],
            'polynomial_degree': config['models']['moe']['polynomial_degree']
        })
    }

    # 初始化 MoE 专家
    for _ in range(config['models']['moe']['num_experts']):
        models_dict['MoE'].add_expert()

    # 包装模型并移动到设备
    for name, model in models_dict.items():
        models_dict[name] = ModelWrapper(model).to(device)

    return models_dict


def train_single_model(
    model: nn.Module,
    model_name: str,
    train_loader,
    val_loader,
    config: Dict[str, Any],
    device: str
) -> Tuple[Dict, float]:
    """
    训练单个模型

    Returns:
        history: 训练历史
        test_mse: 测试MSE
    """
    print(f"\n  Training {model_name}...")

    # 创建训练器
    if model_name == 'Boost':
        trainer = BoostTrainer(model, device=device, lr=config['lr'])
        history = trainer.fit_boosting(
            train_loader, val_loader,
            num_stages=config['models']['boost']['num_stages'],
            epochs_per_stage=config['models']['boost']['epochs_per_stage']
        )
    else:
        trainer = BaseTrainer(model, device=device, lr=config['lr'])
        history = trainer.fit(
            train_loader, val_loader,
            epochs=config['epochs'],
            patience=config['patience']
        )

    return history, trainer


def evaluate_model(
    model: nn.Module,
    trainer,
    test_loader,
    train_loader,
    feature_info: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    评估模型并计算指标

    Returns:
        results: 包含metrics、shap分析等的字典
    """
    # 基础性能指标
    test_mse = trainer.evaluate(test_loader)

    # SHAP分析
    analyzer_config = {
        'feature_info': feature_info,
        'batch_size': config.get('batch_size', 64),
        'device': config.get('device', 'cpu')
    }
    analyzer = RobustnessAnalyzer(analyzer_config, save_dir='.')

    metrics = analyzer.compute_metrics(model, test_loader)
    shap_values, test_data = analyzer.analyze_shap(model, train_loader, test_loader)

    # 特征抑制分析
    suppression = analyzer.analyze_feature_suppression(shap_values)

    # 排序准确性
    ranking = analyzer.analyze_ranking_accuracy(shap_values)

    return {
        'test_mse': test_mse,
        'metrics': metrics,
        'shap_values': shap_values,
        'suppression': suppression,
        'ranking': ranking
    }


def run_single_noise_ratio(
    noise_ratio: float,
    config: Dict[str, Any],
    result_dir: str
) -> Dict[str, Any]:
    """
    运行单个噪声比例的实验

    Args:
        noise_ratio: 噪声比例
        config: 配置
        result_dir: 结果保存目录

    Returns:
        results: 该噪声比例下所有模型的结果
    """
    print(f"\n{'='*80}")
    print(f"噪声比例: {noise_ratio:.0%} (噪声特征数: {config['data']['n_noise_features']})")
    print(f"总特征数: {config['data']['n_total_features']}")
    print(f"{'='*80}")

    device = config['device']

    # 1. 数据准备
    print("\n[1/4] 准备数据...")
    data_generator = NoisyFeaturesGenerator(config['data'])
    train_loader, val_loader, test_loader = data_generator.create_dataloaders()
    feature_info = data_generator.get_feature_info()

    print(f"  - 真实特征: {feature_info['n_true_features']}")
    print(f"  - 噪声特征: {feature_info['n_noise_features']}")
    print(f"  - 冗余特征: {feature_info['n_redundant_features']}")
    print(f"  - 欺骗特征: {feature_info['n_deceptive_features']}")

    # 2. 设置模型
    print("\n[2/4] 设置模型...")
    n_features = feature_info['n_features']
    models = setup_models(n_features, config, device)

    for name, model in models.items():
        params = count_parameters(model.model)
        print(f"  - {name}: {params:,} parameters")

    # 3. 训练与评估
    print("\n[3/4] 训练与评估模型...")
    all_results = {}

    for model_name, model in models.items():
        print(f"\n  [{list(models.keys()).index(model_name)+1}/{len(models)}] {model_name}")

        # 训练
        history, trainer = train_single_model(
            model, model_name, train_loader, val_loader, config, device
        )

        # 评估
        results = evaluate_model(
            model, trainer, test_loader, train_loader, feature_info, config
        )
        results['history'] = history

        all_results[model_name] = results

        # 打印关键指标
        print(f"    Test MSE: {results['test_mse']:.6f}")
        print(f"    Noise Suppression: {results['suppression']['noise_suppression_ratio']:.4f}")
        print(f"    Top-5 Accuracy: {results['ranking'].get('top_5', 0):.1%}")

    # 4. 保存结果
    print("\n[4/4] 保存结果...")
    ratio_str = f"{noise_ratio:.0%}".replace('%', 'pct')
    ratio_dir = os.path.join(result_dir, f'noise_{ratio_str}')
    os.makedirs(ratio_dir, exist_ok=True)

    # 保存汇总CSV
    summary_data = []
    for model_name, results in all_results.items():
        summary_data.append({
            'model': model_name,
            'noise_ratio': noise_ratio,
            'n_features': n_features,
            'n_noise': feature_info['n_noise_features'],
            'test_mse': results['test_mse'],
            'mae': results['metrics']['mae'],
            'r2': results['metrics']['r2'],
            'noise_suppression_ratio': results['suppression']['noise_suppression_ratio'],
            'true_mean_importance': results['suppression']['true_mean_importance'],
            'noise_mean_importance': results['suppression']['noise_mean_importance'],
            'top_5_accuracy': results['ranking'].get('top_5', 0),
            'mean_true_rank': results['ranking'].get('mean_true_rank', 0)
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(ratio_dir, 'summary.csv'), index=False)

    # 保存详细结果（JSON）
    detailed_results = {}
    for model_name, results in all_results.items():
        detailed_results[model_name] = {
            'test_mse': float(results['test_mse']),
            'metrics': results['metrics'],
            'suppression': results['suppression'],
            'ranking': results['ranking']
        }

    with open(os.path.join(ratio_dir, 'detailed_results.json'), 'w') as f:
        json.dump(detailed_results, f, indent=2)

    print(f"  结果已保存到: {ratio_dir}")

    return all_results


def generate_ablation_report(all_results: Dict[float, Dict[str, Any]], result_dir: str):
    """
    生成消融实验综合报告

    Args:
        all_results: 所有噪声比例的结果
        result_dir: 结果目录
    """
    print("\n" + "="*80)
    print("生成综合报告...")
    print("="*80)

    # 1. 汇总表格
    all_summary = []
    for noise_ratio, ratio_results in all_results.items():
        for model_name, results in ratio_results.items():
            all_summary.append({
                'noise_ratio': noise_ratio,
                'model': model_name,
                'test_mse': results['test_mse'],
                'noise_suppression_ratio': results['suppression']['noise_suppression_ratio'],
                'top_5_accuracy': results['ranking'].get('top_5', 0),
                'mean_true_rank': results['ranking'].get('mean_true_rank', 0)
            })

    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(os.path.join(result_dir, 'ablation_summary.csv'), index=False)

    # 2. 生成Markdown报告
    report_lines = []
    report_lines.append("# 噪声抑制消融实验报告\n")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("="*80 + "\n\n")

    # 实验设置
    report_lines.append("## 实验设置\n")
    report_lines.append("- 真实特征数: 4\n")
    report_lines.append("- 冗余特征数: 2\n")
    report_lines.append("- 欺骗特征数: 1\n")
    report_lines.append("- 测试噪声比例: 0%, 20%, 40%, 60%, 80%, 90%\n")
    report_lines.append("- 对比模型: MLP, KAN, Standard, Hybrid, Boost, MoE\n\n")

    # 汇总表格
    report_lines.append("## 结果汇总\n")
    report_lines.append("### 噪声抑制率对比\n")
    report_lines.append("| 噪声比例 | MLP | KAN | Standard | Hybrid | Boost | MoE |\n")
    report_lines.append("|---------|-----|-----|----------|--------|-------|-----|\n")

    for noise_ratio in sorted(all_results.keys()):
        ratio_results = all_results[noise_ratio]
        line = f"| {noise_ratio:.0%} |"
        for model in ['MLP', 'KAN', 'Standard', 'Hybrid', 'Boost', 'MoE']:
            if model in ratio_results:
                suppression = ratio_results[model]['suppression']['noise_suppression_ratio']
                line += f" {suppression:.3f} |"
            else:
                line += " N/A |"
        report_lines.append(line + "\n")

    report_lines.append("\n### 测试MSE对比\n")
    report_lines.append("| 噪声比例 | MLP | KAN | Standard | Hybrid | Boost | MoE |\n")
    report_lines.append("|---------|-----|-----|----------|--------|-------|-----|\n")

    for noise_ratio in sorted(all_results.keys()):
        ratio_results = all_results[noise_ratio]
        line = f"| {noise_ratio:.0%} |"
        for model in ['MLP', 'KAN', 'Standard', 'Hybrid', 'Boost', 'MoE']:
            if model in ratio_results:
                mse = ratio_results[model]['test_mse']
                line += f" {mse:.4f} |"
            else:
                line += " N/A |"
        report_lines.append(line + "\n")

    report_lines.append("\n### Top-5特征识别准确率\n")
    report_lines.append("| 噪声比例 | MLP | KAN | Standard | Hybrid | Boost | MoE |\n")
    report_lines.append("|---------|-----|-----|----------|--------|-------|-----|\n")

    for noise_ratio in sorted(all_results.keys()):
        ratio_results = all_results[noise_ratio]
        line = f"| {noise_ratio:.0%} |"
        for model in ['MLP', 'KAN', 'Standard', 'Hybrid', 'Boost', 'MoE']:
            if model in ratio_results:
                acc = ratio_results[model]['ranking'].get('top_5', 0)
                line += f" {acc:.1%} |"
            else:
                line += " N/A |"
        report_lines.append(line + "\n")

    # 关键发现
    report_lines.append("\n## 关键发现\n")

    # 计算平均噪声抑制率
    avg_suppression = {}
    for model in ['MLP', 'KAN', 'Standard', 'Hybrid', 'Boost', 'MoE']:
        suppressions = []
        for noise_ratio, ratio_results in all_results.items():
            if model in ratio_results:
                suppressions.append(ratio_results[model]['suppression']['noise_suppression_ratio'])
        if suppressions:
            avg_suppression[model] = np.mean(suppressions)

    report_lines.append("\n### 平均噪声抑制率（越低越好）\n")
    for model, avg_val in sorted(avg_suppression.items(), key=lambda x: x[1]):
        report_lines.append(f"- **{model}**: {avg_val:.4f}\n")

    # 保存报告
    report_path = os.path.join(result_dir, 'ablation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)

    print(f"报告已保存到: {report_path}")


def run_ablation_experiment():
    """
    运行完整的消融实验
    """
    print("="*80)
    print("方案 1 消融实验：不同噪声比例下的鲁棒性测试")
    print("="*80)

    # 创建结果目录
    result_dir = os.path.join(
        BASE_RESULT_DIR,
        'exp1_ablation',
        datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    os.makedirs(result_dir, exist_ok=True)

    print(f"\n结果保存目录: {result_dir}")

    # 获取噪声比例配置
    from interpretable_experiment.config.exp1_ablation_config import get_noise_ratio_sweep_configs

    configs = get_noise_ratio_sweep_configs()
    all_results = {}

    # 运行每个噪声比例的实验
    for i, (noise_ratio, config) in enumerate(configs, 1):
        print(f"\n\n{'#'*80}")
        print(f"# 进度: [{i}/{len(configs)}] 噪声比例 {noise_ratio:.0%}")
        print(f"{'#'*80}")

        try:
            ratio_results = run_single_noise_ratio(noise_ratio, config, result_dir)
            all_results[noise_ratio] = ratio_results
        except Exception as e:
            print(f"\n错误: 噪声比例 {noise_ratio:.0%} 实验失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 生成综合报告
    generate_ablation_report(all_results, result_dir)

    print("\n" + "="*80)
    print("消融实验完成!")
    print(f"所有结果保存在: {result_dir}")
    print("="*80)

    return all_results


def run_ablation_experiment_quick():
    """
    运行快速测试版本的消融实验
    使用小数据集快速验证代码正确性
    """
    print("="*80)
    print("方案 1 消融实验（快速测试版）")
    print("="*80)
    print("\n快速测试配置:")
    print("  - 样本数: 500 (原5000)")
    print("  - 噪声比例: 0%, 40%, 80% (仅3个)")
    print("  - 训练epoch: 10 (原50)")
    print("  - Boost阶段: 3 (原5)")
    print("\n目的: 快速验证代码正确性\n")

    # 创建结果目录
    result_dir = os.path.join(
        BASE_RESULT_DIR,
        'exp1_ablation_quick',
        datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    os.makedirs(result_dir, exist_ok=True)

    print(f"结果保存目录: {result_dir}")

    # 获取快速测试配置
    from interpretable_experiment.config.exp1_ablation_config_quick import get_quick_test_configs

    configs = get_quick_test_configs()
    all_results = {}

    # 运行每个噪声比例的实验
    for i, (noise_ratio, config) in enumerate(configs, 1):
        print(f"\n\n{'#'*80}")
        print(f"# 快速测试 [{i}/{len(configs)}] 噪声比例 {noise_ratio:.0%}")
        print(f"{'#'*80}")

        try:
            ratio_results = run_single_noise_ratio(noise_ratio, config, result_dir)
            all_results[noise_ratio] = ratio_results
        except Exception as e:
            print(f"\n错误: 噪声比例 {noise_ratio:.0%} 实验失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 生成综合报告
    generate_ablation_report(all_results, result_dir)

    print("\n" + "="*80)
    print("快速测试完成!")
    print(f"结果保存在: {result_dir}")
    print("\n如果测试通过，运行完整实验:")
    print("  python interpretable_experiment/scripts/run/run_exp1_ablation.py")
    print("="*80)

    return all_results


if __name__ == '__main__':
    # 默认运行快速测试
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        run_ablation_experiment()
    else:
        print("运行快速测试（添加 --full 参数运行完整实验）")
        run_ablation_experiment_quick()
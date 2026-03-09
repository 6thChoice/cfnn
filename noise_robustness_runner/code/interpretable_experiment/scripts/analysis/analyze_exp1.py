#!/usr/bin/env python3
"""
方案1 分析脚本
使用已训练的模型运行SHAP分析
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
from datetime import datetime

# 添加路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from xai.interpretable.data.generate_noisy_features import NoisyFeaturesGenerator
from xai.interpretable.analysis.robustness_analyzer import RobustnessAnalyzer
from xai.interpretable.config.exp1_config import get_config
from cfnet import CFNet_Standard, HybridRationalNet, EnsembleResCoFrNet, MoE_Ensemble
from xai.interpretable.experiments.exp1_robustness import ModelWrapper


def load_models(checkpoint_dir, config, device):
    """加载所有训练好的模型"""
    models = {}
    n_features = 11  # 从配置中获取

    # 创建模型实例
    model_configs = {
        'Standard': CFNet_Standard(n_features, 1, **config['models']['standard']),
        'Hybrid': HybridRationalNet(n_features, 1, **config['models']['hybrid']),
        'Boost': EnsembleResCoFrNet(n_features, 1,
                                   config['models']['boost']['shallow_depth'],
                                   config['models']['boost']['poly_degree'],
                                   learning_rate=0.5),
        'MoE': MoE_Ensemble({
            'input_dim': n_features,
            'output_dim': 1,
            'shallow_depth_per_cofrnet': config['models']['moe']['shallow_depth_per_cofrnet'],
            'polynomial_degree': config['models']['moe']['polynomial_degree']
        })
    }

    # 加载 MLP
    from func.models_ext import get_aligned_mlp
    anchor = CFNet_Standard(4, 1, **config['models']['standard'])
    target_params = sum(p.numel() for p in anchor.parameters())
    model_configs['MLP'] = get_aligned_mlp(n_features, 1, target_params)[0]

    # 初始化 MoE 专家
    for _ in range(config['models']['moe']['num_experts']):
        model_configs['MoE'].add_expert()

    # 加载检查点
    for name, model in model_configs.items():
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoints', f'{name}.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # 检查点可能是空的，这种情况下跳过加载
            if len(checkpoint) > 0:
                model.load_state_dict(checkpoint)
                print(f"✓ Loaded {name} from {checkpoint_path}")
            else:
                print(f"⚠ Warning: Empty checkpoint for {name}, using initialized weights")
            models[name] = ModelWrapper(model).to(device)
        else:
            print(f"✗ Missing checkpoint: {checkpoint_path}")

    return models


def main():
    """主函数"""
    print("=" * 80)
    print("方案 1: 分析阶段")
    print("=" * 80)

    # 加载配置
    config = get_config()
    device = config['device']

    # 指定结果目录（使用最新的）
    runner_base = Path(__file__).resolve().parents[4]
    results_dir = runner_base / "results"
    result_dir = str(results_dir / "exp1_robustness" / "20260114_122551")

    print(f"\n使用结果目录: {result_dir}")

    # 1. 创建数据生成器
    print("\n[Phase 1] 创建数据生成器...")
    data_generator = NoisyFeaturesGenerator(config['data'])
    train_loader, val_loader, test_loader = data_generator.create_dataloaders()
    feature_info = data_generator.get_feature_info()

    print(f"  - True features: {data_generator.n_true}")
    print(f"  - Noise features: {data_generator.n_noise}")
    print(f"  - Redundant features: {data_generator.n_redundant}")
    print(f"  - Deceptive features: {data_generator.n_deceptive}")
    print(f"  - Total features: {data_generator.n_features}")

    # 2. 加载模型
    print("\n[Phase 2] 加载训练好的模型...")
    models = load_models(result_dir, config, device)
    print(f"  - 已加载 {len(models)} 个模型")

    # 3. 创建分析器
    print("\n[Phase 3] 创建分析器...")
    analyzer_config = {**config, 'feature_info': feature_info}
    analyzer = RobustnessAnalyzer(analyzer_config, save_dir=os.path.join(result_dir, 'metrics'))

    # 4. 分析每个模型
    print("\n[Phase 4] 运行 SHAP 分析...")
    analysis_results = {}

    for model_name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"分析 {model_name}...")
        print(f"{'=' * 60}\n")

        # 计算性能指标
        print("  [1/2] 计算性能指标...")
        metrics = analyzer.compute_metrics(model, test_loader)
        print(f"    - MSE: {metrics['mse']:.6f}")
        print(f"    - MAE: {metrics['mae']:.6f}")
        print(f"    - R²:  {metrics['r2']:.4f}")

        # 计算 SHAP 值
        print("  [2/2] 计算 SHAP 值...")
        shap_values, test_data = analyzer.analyze_shap(model, train_loader, test_loader)
        print(f"    - SHAP values shape: {shap_values.shape}")

        # 保存结果
        result_key = f"{model_name}_robustness"
        analysis_results[result_key] = {
            'metrics': metrics,
            'shap_values': shap_values,
            'test_data': test_data
        }

        analyzer.save_results(analysis_results[result_key], result_key)

    # 5. 生成报告
    print("\n[Phase 5] 生成分析报告...")
    report = analyzer.generate_report(analysis_results)

    report_path = os.path.join(result_dir, 'summary', 'robustness_report.md')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\n报告已保存到: {report_path}")

    # 6. 打印总结
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)

    print("\n性能对比:")
    print("| 模型 | MSE | MAE | R² |")
    print("|------|-----|-----|-----|")
    for key, result in analysis_results.items():
        model_name = key.replace('_robustness', '')
        metrics = result['metrics']
        print(f"| {model_name} | {metrics['mse']:.4f} | {metrics['mae']:.4f} | {metrics['r2']:.4f} |")

    return analysis_results


if __name__ == '__main__':
    results = main()

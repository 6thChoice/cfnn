#!/usr/bin/env python3
"""
方案2 分析脚本
使用已训练的模型运行交互分析
"""

import sys
import os
from pathlib import Path
import torch
import pickle
import numpy as np

# 添加路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from xai.interpretable.data.generate_interaction_data import InteractionDataGenerator
from xai.interpretable.analysis.interaction_analyzer import InteractionAnalyzer
from xai.interpretable.config.exp2_config import get_config
from cfnet import CFNet_Standard, HybridRationalNet, MoE_Ensemble
from func.models_ext import get_aligned_mlp, count_parameters
from xai.interpretable.experiments.exp1_robustness import ModelWrapper


def main():
    """主函数"""
    print("=" * 80)
    print("方案 2: 分析阶段（使用已训练模型）")
    print("=" * 80)

    # 加载配置
    config = get_config('rational')

    # 指定结果目录
    runner_base = Path(__file__).resolve().parents[4]
    results_dir = runner_base / "results"
    result_dir = str(results_dir / "exp2_interaction_rational" / "20260114_132618")

    print(f"\n使用结果目录: {result_dir}")

    # 1. 创建数据生成器
    print("\n[Phase 1] 创建数据生成器...")
    data_generator = InteractionDataGenerator(config['data'])
    train_loader, val_loader, test_loader = data_generator.create_dataloaders()
    feature_info = data_generator.get_feature_info()

    print(f"  - Interaction type: {data_generator.interaction_type}")
    print(f"  - Features: {data_generator.n_features}")

    # 2. 加载模型
    print("\n[Phase 2] 加载训练好的模型...")
    device = config['device']
    n_features = data_generator.n_features

    models_dict = {
        'Standard': CFNet_Standard(n_features, 1, **config['models']['standard']),
        'Hybrid': HybridRationalNet(n_features, 1, **config['models']['hybrid']),
    }

    # 添加MLP
    anchor = CFNet_Standard(n_features, 1, **config['models']['standard'])
    target_params = count_parameters(anchor)
    mlp, _ = get_aligned_mlp(n_features, 1, target_params)
    models_dict['MLP'] = mlp

    # 添加MoE
    moe = MoE_Ensemble({
        'input_dim': n_features,
        'output_dim': 1,
        'shallow_depth_per_cofrnet': config['models']['moe']['shallow_depth_per_cofrnet'],
        'polynomial_degree': config['models']['moe']['polynomial_degree']
    })
    for _ in range(config['models']['moe']['num_experts']):
        moe.add_expert()
    models_dict['MoE'] = moe

    models = {}
    for name, model in models_dict.items():
        checkpoint_path = os.path.join(result_dir, 'checkpoints', f'{name}.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint)
            print(f"  ✓ Loaded {name}")
            models[name] = ModelWrapper(model).to(device)
        else:
            print(f"  ✗ Missing {name}")

    # 3. 创建分析器
    print("\n[Phase 3] 创建分析器...")
    analyzer_config = {**config, 'feature_info': feature_info}
    analyzer = InteractionAnalyzer(analyzer_config, save_dir=os.path.join(result_dir, 'metrics'))

    # 4. 分析每个模型
    print("\n[Phase 4] 运行交互分析...")
    analysis_results = {}

    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"分析 {model_name}...")
        print(f"{'='*60}\n")

        # 计算性能指标
        print("  [1/3] 计算性能指标...")
        metrics = analyzer.compute_metrics(model, test_loader)
        print(f"    - MSE: {metrics['mse']:.6f}")
        print(f"    - R²: {metrics['r2']:.4f}")

        # 计算SHAP值
        print("  [2/3] 计算SHAP值...")
        shap_values, test_data = analyzer.analyze_shap(model, train_loader, test_loader)
        print(f"    - SHAP values shape: {shap_values.shape}")

        # 计算交互强度
        print("  [3/3] 计算交互强度...")
        interaction_strength = analyzer.analyze_interaction_strength(shap_values)
        print(f"    - Total variance: {interaction_strength['total_variance']:.4f}")

        # 评估交互检测
        detection_metrics = analyzer.evaluate_interaction_detection(shap_values, test_data)
        print(f"    - Detection F1: {detection_metrics['f1_score']:.4f}")

        # 保存结果
        result_key = f"{model_name}_interaction"
        analysis_results[result_key] = {
            'metrics': metrics,
            'shap_values': shap_values,
            'test_data': test_data,
            'interaction_strength': interaction_strength,
            'detection_metrics': detection_metrics
        }

        save_path = os.path.join(result_dir, 'metrics', f'{result_key}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(analysis_results[result_key], f)
        print(f"  ✓ Saved to {save_path}")

    # 5. 生成报告
    print("\n[Phase 5] 生成报告...")
    report = analyzer.generate_report(analysis_results)

    report_path = os.path.join(result_dir, 'summary', 'interaction_report.md')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\n报告已保存到: {report_path}")

    # 6. 打印总结
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)

    print("\n性能对比:")
    print("| 模型 | MSE | R² | 交互检测F1 |")
    print("|------|-----|-----|-----------|")
    for key, result in analysis_results.items():
        model_name = key.replace('_interaction', '')
        metrics = result['metrics']
        f1 = result['detection_metrics']['f1_score']
        print(f"| {model_name} | {metrics['mse']:.4f} | {metrics['r2']:.4f} | {f1:.4f} |")

    return analysis_results


if __name__ == '__main__':
    results = main()

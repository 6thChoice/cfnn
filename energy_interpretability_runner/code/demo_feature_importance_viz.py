#!/usr/bin/env python3
"""
演示特征重要性可视化功能
"""

import numpy as np
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interpretable_experiment.analysis.domain_analyzer import DomainAnalyzer


def create_mock_results():
    """
    创建模拟的实验结果数据
    模拟能源数据集上的多个模型特征重要性结果
    """
    # 特征名称（能源数据集）
    feature_names = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']

    # 领域知识（真实重要性）
    # X1, X2: 高重要性 (3)
    # X3, X4, X5, X7: 中重要性 (2)
    # X6, X8: 低重要性 (1)
    domain_importance = {
        'X1': 3, 'X2': 3,  # 高重要性
        'X3': 2, 'X4': 2, 'X5': 2, 'X7': 2,  # 中重要性
        'X6': 1, 'X8': 1  # 低重要性
    }

    # 模拟不同模型的SHAP重要性
    np.random.seed(42)
    n_samples = 50
    n_features = len(feature_names)

    # Standard CFNet - 较好的识别能力
    shap_standard = np.random.randn(n_samples, n_features) * 0.1
    shap_standard[:, 0] += 0.8  # X1
    shap_standard[:, 1] += 0.7  # X2
    shap_standard[:, 2] += 0.4  # X3
    shap_standard[:, 3] += 0.3  # X4
    shap_standard[:, 4] += 0.35  # X5
    shap_standard[:, 6] += 0.4  # X7

    # Hybrid CFNet - 更强的识别能力
    shap_hybrid = np.random.randn(n_samples, n_features) * 0.08
    shap_hybrid[:, 0] += 0.9  # X1
    shap_hybrid[:, 1] += 0.85  # X2
    shap_hybrid[:, 2] += 0.45  # X3
    shap_hybrid[:, 3] += 0.4  # X4
    shap_hybrid[:, 4] += 0.4  # X5
    shap_hybrid[:, 6] += 0.45  # X7

    # MLP - 较弱的识别能力，可能误判
    shap_mlp = np.random.randn(n_samples, n_features) * 0.15
    shap_mlp[:, 0] += 0.6  # X1
    shap_mlp[:, 1] += 0.55  # X2
    shap_mlp[:, 2] += 0.5  # X3 (高估)
    shap_mlp[:, 5] += 0.3  # X6 (误判)
    shap_mlp[:, 6] += 0.35  # X7

    # Boost CFNet - 最强的识别能力
    shap_boost = np.random.randn(n_samples, n_features) * 0.05
    shap_boost[:, 0] += 0.95  # X1
    shap_boost[:, 1] += 0.9  # X2
    shap_boost[:, 2] += 0.5  # X3
    shap_boost[:, 3] += 0.45  # X4
    shap_boost[:, 4] += 0.45  # X5
    shap_boost[:, 6] += 0.5  # X7

    results = {
        'Standard_domain': {
            'shap_values': shap_standard,
            'metrics': {'mse': 0.0234, 'mae': 0.1234, 'r2': 0.9234}
        },
        'Hybrid_domain': {
            'shap_values': shap_hybrid,
            'metrics': {'mse': 0.0189, 'mae': 0.1098, 'r2': 0.9389}
        },
        'MLP_domain': {
            'shap_values': shap_mlp,
            'metrics': {'mse': 0.0345, 'mae': 0.1456, 'r2': 0.8876}
        },
        'Boost_domain': {
            'shap_values': shap_boost,
            'metrics': {'mse': 0.0156, 'mae': 0.0987, 'r2': 0.9489}
        }
    }

    config = {
        'feature_info': {
            'feature_names': feature_names,
            'domain_importance': domain_importance,
            'dataset_name': 'energy_efficiency'
        },
        'batch_size': 32,
        'device': 'cpu'
    }

    return results, config


def main():
    """主函数"""
    print("=" * 80)
    print("特征重要性可视化演示")
    print("=" * 80)

    # 创建模拟结果
    print("\n创建模拟实验数据...")
    results, config = create_mock_results()

    # 创建输出目录
    output_dir = 'demo_viz_output'
    os.makedirs(output_dir, exist_ok=True)
    config['base_result_dir'] = output_dir

    # 创建分析器
    print(f"\n初始化 DomainAnalyzer...")
    analyzer = DomainAnalyzer(config, save_dir=output_dir)

    # 生成可视化
    print("\n" + "-" * 80)
    print("生成可视化图表...")
    print("-" * 80)

    viz_paths = analyzer.generate_visualizations(results)

    print("\n" + "=" * 80)
    print("可视化完成！")
    print("=" * 80)
    print("\n生成的图表：")
    for name, path in viz_paths.items():
        print(f"  - {name}: {os.path.abspath(path)}")

    # 生成报告
    print("\n生成分析报告...")
    report = analyzer.generate_report(results)

    # 保存报告
    report_path = os.path.join(output_dir, 'domain_analysis_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"报告已保存: {os.path.abspath(report_path)}")

    # 打印部分报告内容
    print("\n" + "=" * 80)
    print("报告预览（前2000字符）：")
    print("=" * 80)
    print(report[:2000])
    print("...")

    print("\n" + "=" * 80)
    print("演示完成！请查看输出目录中的图表和报告。")
    print(f"输出目录: {os.path.abspath(output_dir)}")
    print("=" * 80)


if __name__ == '__main__':
    main()

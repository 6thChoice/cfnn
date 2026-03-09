#!/usr/bin/env python3
"""
方案 1 报告生成脚本
使用已保存的分析结果生成报告
"""

import sys
import os
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

# 添加路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from xai.interpretable.analysis.robustness_analyzer import RobustnessAnalyzer
from xai.interpretable.config.exp1_config import get_config


def main():
    """主函数"""
    print("=" * 80)
    print("方案 1: 报告生成")
    print("=" * 80)

    # 加载配置
    config = get_config()

    # 指定结果目录（使用最新的）
    runner_base = Path(__file__).resolve().parents[4]
    results_dir = runner_base / "results"
    result_dir = str(results_dir / "exp1_robustness" / "20260114_124745")

    print(f"\n使用结果目录: {result_dir}")

    # 创建分析器（用于使用报告生成方法）
    from xai.interpretable.data.generate_noisy_features import NoisyFeaturesGenerator
    data_generator = NoisyFeaturesGenerator(config['data'])
    feature_info = data_generator.get_feature_info()

    analyzer_config = {**config, 'feature_info': feature_info}
    analyzer = RobustnessAnalyzer(analyzer_config, save_dir=os.path.join(result_dir, 'metrics'))

    # 加载所有保存的结果
    print("\n[Phase 1] 加载分析结果...")
    analysis_results = {}

    model_names = ['Standard', 'Hybrid', 'MLP', 'Boost', 'MoE']
    for model_name in model_names:
        result_path = os.path.join(result_dir, f'{model_name}_robustness.pkl')
        if os.path.exists(result_path):
            with open(result_path, 'rb') as f:
                analysis_results[f'{model_name}_robustness'] = pickle.load(f)
            print(f"  ✓ Loaded {model_name}")
        else:
            print(f"  ✗ Missing {result_path}")

    # 生成报告
    print("\n[Phase 2] 生成报告...")
    report = analyzer.generate_report(analysis_results)

    # 保存报告
    report_path = os.path.join(result_dir, 'summary', 'robustness_report.md')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\n报告已保存到: {report_path}")

    # 打印报告内容
    print("\n" + "=" * 80)
    print("报告预览")
    print("=" * 80)
    print(report)

    return report


if __name__ == '__main__':
    report = main()

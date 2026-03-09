#!/usr/bin/env python3
"""
方案1帕累托前沿实验运行脚本

参数量-性能帕累托前沿实验的入口脚本，支持快速测试和完整实验两种模式。

用法:
    # 快速测试（推荐首次运行验证）
    python run_exp1_pareto.py --quick

    # 完整实验（所有噪声比例 × 所有参数量级别 × 所有模型）
    python run_exp1_pareto.py --full

    # 指定特定模型进行测试
    python run_exp1_pareto.py --quick --models MLP Standard Boost

    # 验证配置生成（不运行实验）
    python run_exp1_pareto.py --verify-config

输出:
    - 实验结果保存于: interpretable/results/exp1_pareto/YYYYMMDD_HHMMSS/
    - pareto_summary.csv: 所有实验结果的汇总表格
    - pareto_report.md: 详细的实验报告（包含各噪声比例下的对比表）
    - pareto_frontiers.png: 帕累托前沿曲线图（每个噪声比例一个子图）
    - pareto_by_model.png: 按模型分组的对比图
"""

import sys
import os
import argparse
from datetime import datetime

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from interpretable_experiment.experiments.exp1_pareto import (
    run_pareto_experiment
)
from interpretable_experiment.config.exp1_pareto_config import (
    PARETO_LEVELS,
    NOISE_RATIOS,
    get_all_pareto_configs,
    get_quick_test_configs,
    print_pareto_config_table
)


def print_experiment_info():
    """打印实验信息"""
    print("="*80)
    print("方案1帕累托前沿实验：参数量-性能权衡分析")
    print("="*80)
    print()
    print("实验设计:")
    print(f"  - 噪声比例: {[f'{r:.0%}' for r in NOISE_RATIOS]}")
    print(f"  - 参数量级别: {PARETO_LEVELS}")
    print("  - 对比模型: MLP, KAN, CFNet-Standard, CFNet-Hybrid, CFNet-Boost, CFNet-MoE")
    print()
    print("实验目的:")
    print("  理解模型规模与鲁棒性之间的权衡关系，绘制参数量-性能帕累托前沿")
    print()


def verify_config():
    """验证配置生成是否正确"""
    print("="*80)
    print("验证帕累托配置生成")
    print("="*80)
    print()

    # 打印配置表
    print_pareto_config_table()

    print("\n\n验证快速测试配置...")
    quick_configs = get_quick_test_configs()
    print(f"快速测试配置数: {len(quick_configs)}")

    print("\n快速测试配置示例:")
    for i, cfg in enumerate(quick_configs[:3]):
        print(f"\n配置 {i+1}:")
        print(f"  噪声比例: {cfg['noise_ratio']:.0%}")
        print(f"  目标参数量: {cfg['target_params']}")
        print(f"  模型: {cfg['model_name']}")
        print(f"  实际参数量: {cfg['model_config'].get('actual_params', 'N/A')}")

    print("\n\n验证完整配置...")
    full_configs = get_all_pareto_configs()
    print(f"完整实验配置数: {len(full_configs)}")
    print(f"  = {len(NOISE_RATIOS)} 噪声比例 × {len(PARETO_LEVELS)} 参数量级别 × 6 模型")

    print("\n" + "="*80)
    print("配置验证完成!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='运行方案1帕累托前沿实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速测试（小数据集，快速验证代码正确性）
  python run_exp1_pareto.py --quick

  # 完整实验（完整配置，耗时较长）
  python run_exp1_pareto.py --full

  # 只测试特定模型
  python run_exp1_pareto.py --quick --models MLP Standard

  # 验证配置（不运行实验）
  python run_exp1_pareto.py --verify-config
        """
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='运行快速测试模式（样本数500，epoch=10，仅部分配置）'
    )

    parser.add_argument(
        '--full',
        action='store_true',
        help='运行完整实验模式（完整数据集，50 epochs，所有配置）'
    )

    parser.add_argument(
        '--verify-config',
        action='store_true',
        help='验证配置生成是否正确（不运行实验）'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        choices=['MLP', 'KAN', 'Standard', 'Hybrid', 'Boost', 'MoE', 'all'],
        default=['all'],
        help='指定要测试的模型（默认：all）'
    )

    args = parser.parse_args()

    # 验证配置模式
    if args.verify_config:
        verify_config()
        return

    # 如果没有指定模式，默认快速测试
    if not args.quick and not args.full:
        args.quick = True

    # 打印实验信息
    print_experiment_info()

    # 确定要运行的模型
    if 'all' in args.models:
        models = None  # None表示所有模型
    else:
        models = args.models

    if models:
        print(f"指定模型: {', '.join(models)}")
    else:
        print("测试所有模型: MLP, KAN, Standard, Hybrid, Boost, MoE")
    print()

    # 运行实验
    if args.full:
        print("【完整实验模式】")
        print("注意: 完整实验可能需要较长时间（数小时），请耐心等待。")
        print()
        confirm = input("确认开始完整实验? [y/N]: ")
        if confirm.lower() != 'y':
            print("已取消。如需快速测试，请使用 --quick 参数。")
            return

        run_pareto_experiment(quick_mode=False, models=models)
    else:
        print("【快速测试模式】")
        print("使用小数据集和减少的配置快速验证代码正确性。")
        print()
        run_pareto_experiment(quick_mode=True, models=models)


if __name__ == '__main__':
    main()

"""
运行消融实验完整版

使用完整配置运行所有噪声比例的实验
预期运行时间: 较长（建议先运行快速测试版）
"""

import os
import sys

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from interpretable_experiment.experiments.exp1_ablation import run_ablation_experiment


if __name__ == '__main__':
    print("="*80)
    print("消融实验完整版")
    print("="*80)
    print("\n完整配置:")
    print("  - 样本数: 5000")
    print("  - 噪声比例: 0%, 20%, 40%, 60%, 80%, 90% (6个)")
    print("  - 训练epoch: 50")
    print("  - 模型: MLP, KAN, Standard, Hybrid, Boost, MoE")
    print("\n注意: 运行时间较长，建议先运行快速测试版验证代码")
    print("  快速测试: python run_exp1_ablation_quick.py")
    print("="*80)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--confirm', action='store_true',
                        help='确认运行完整实验')
    args = parser.parse_args()

    if not args.confirm:
        print("\n警告: 完整实验运行时间较长!")
        print("建议先运行快速测试版:")
        print("  python run_exp1_ablation_quick.py")
        print("\n如果确认要运行完整实验，请添加 --confirm 参数:")
        print("  python run_exp1_ablation.py --confirm")
        sys.exit(0)

    results = run_ablation_experiment()

    print("\n✓ 完整实验完成！")

#!/usr/bin/env python3
"""
方案2 简化分析脚本
"""

import sys
import os

# 确保正确的项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

# 使用绝对路径导入
from xai.interpretable.experiments.exp2_interaction import run_interaction_experiment


if __name__ == '__main__':
    print("="*80)
    print("方案 2：特征交互捕获实验（分析阶段）")
    print("="*80)
    print("\n交互类型: rational (分式交互)")
    print("\n开始运行实验（完整流程）...\n")

    # 直接运行完整实验（会使用已有的检查点）
    results = run_interaction_experiment(interaction_type='rational')

    print("\n实验完成！")
    print("结果保存在: interpretable/results/exp2_interaction_rational/")

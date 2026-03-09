#!/usr/bin/env python3
"""
运行方案 3：OOD泛化稳定性实验（完整版）
"""

import os
import sys

# 确保正确的项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

# 使用绝对路径导入
from xai.interpretable.experiments.exp3_ood import run_ood_experiment


if __name__ == '__main__':
    print("="*80)
    print("方案 3：分布外泛化稳定性实验")
    print("="*80)
    print("\nOOD类型: covariate_shift (协变量偏移)")
    print("\n开始运行实验...\n")

    # 运行协变量偏移实验
    results = run_ood_experiment(ood_type='covariate_shift')

    print("\n实验完成！")
    print("结果保存在: interpretable/results/exp3_ood_covariate_shift/")

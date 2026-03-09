"""
运行方案 3：分布外泛化稳定性实验
"""

import os
import sys

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# 从 xai.interpretable.experiments 导入
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

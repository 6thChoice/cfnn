"""
运行方案 1：特征干扰鲁棒性实验
"""

import os
import sys

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# 从 xai.interpretable.experiments 导入
from xai.interpretable_experiment.experiments.exp1_robustness import run_robustness_experiment


if __name__ == '__main__':
    print("="*80)
    print("方案 1：特征干扰鲁棒性实验")
    print("="*80)
    print("\n开始运行实验...\n")

    results = run_robustness_experiment()

    print("\n实验完成！")
    print("结果保存在: interpretable/results/exp1_robustness/")

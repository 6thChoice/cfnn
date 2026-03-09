"""
运行方案 5：真实数据集基准测试实验
"""

import os
import sys

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# 从 xai.interpretable.experiments 导入
from xai.interpretable.experiments.exp5_real_data import run_real_data_experiment


if __name__ == '__main__':
    print("="*80)
    print("方案 5：真实数据集基准测试实验")
    print("="*80)
    print("\n开始运行实验...\n")

    results = run_real_data_experiment()

    print("\n实验完成！")
    print("结果保存在: interpretable/results/exp5_real_data/")

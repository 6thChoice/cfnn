#!/usr/bin/env python3
"""
运行方案 4：因果结构发现实验（完整版）
"""

import os
import sys

# 确保正确的项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

# 使用绝对路径导入
from xai.interpretable.experiments.exp4_causal import run_causal_experiment


if __name__ == '__main__':
    print("="*80)
    print("方案 4：因果结构发现实验")
    print("="*80)
    print("\n因果图类型: rational (分式因果 - CFNet优势场景)")
    print("\n开始运行实验...\n")

    # 运行分式因果实验
    results = run_causal_experiment(graph_type='rational')

    print("\n实验完成！")
    print("结果保存在: interpretable/results/exp4_causal_rational/")

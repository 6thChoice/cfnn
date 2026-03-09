"""
运行方案 4：因果结构发现实验
"""

import os
import sys

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# 从 xai.interpretable.experiments 导入
from xai.interpretable.experiments.exp4_causal import run_causal_experiment


if __name__ == '__main__':
    print("="*80)
    print("方案 4：因果结构发现实验")
    print("="*80)
    print("\n因果图类型: rational (分式因果 - CFNet优势场景)")
    print("\n开始运行实验...\n")

    # 运行分式因果实验（CFNet优势场景）
    results = run_causal_experiment(graph_type='rational')

    print("\n实验完成！")
    print("结果保存在: interpretable/results/exp4_causal_rational/")

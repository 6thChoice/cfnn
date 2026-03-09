#!/usr/bin/env python3
"""
运行真实的exp5实验（能源数据集）
保存分析数据以便后续多次绘图
"""

import os
import sys
import pickle
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # 添加父目录以导入cfnet

from interpretable_experiment.experiments.exp5_real_data import run_real_data_experiment


def main():
    """主函数"""
    print("=" * 80)
    print("运行方案5：真实数据集（能源效率）实验")
    print("=" * 80)

    # 运行实验
    results = run_real_data_experiment()

    # 打印结果路径
    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)

    # 保存完整结果到单独文件，方便后续加载
    runner_base = Path(__file__).resolve().parents[1]
    output_dir = runner_base / 'results' / 'exp5_saved_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 提取关键数据用于绘图
    plot_data = {
        'analysis_results': results['analysis_results'],
        'feature_info': {
            'feature_names': ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'],
            'domain_importance': {
                'X1': 3, 'X2': 3,  # 高重要性
                'X3': 2, 'X4': 2, 'X5': 2, 'X7': 2,  # 中重要性
                'X6': 1, 'X8': 1  # 低重要性
            }
        }
    }

    # 从实验目录复制数据
    # 找到最新的实验目录
    base_dir = runner_base / 'results' / 'exp5_real_data'
    if base_dir.exists():
        exp_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith('202')])
        if exp_dirs:
            latest_dir = os.path.join(str(base_dir), exp_dirs[-1])
            print(f"\n实验结果保存在: {latest_dir}")

            # 复制关键数据
            saved_data_path = output_dir / 'exp5_analysis_data.pkl'
            with open(saved_data_path, 'wb') as f:
                pickle.dump(plot_data, f)
            print(f"分析数据已保存到: {saved_data_path}")

            # 同时创建一个易于加载的JSON版本（不包含numpy数组）
            import json
            json_data = {}
            for key, value in results['analysis_results'].items():
                json_data[key] = {
                    'metrics': value['metrics'],
                    # 不包含shap_values数组，因为JSON不支持
                }
            json_path = output_dir / 'exp5_metrics.json'
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"指标数据已保存到: {json_path}")

    print("\n你可以使用以下命令重新绘图：")
    print("  python plot_from_saved_data.py")
    print("=" * 80)

    return results


if __name__ == '__main__':
    main()

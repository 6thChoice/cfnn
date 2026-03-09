#!/usr/bin/env python3
"""
从保存的exp5实验数据重新绘图
无需重新运行实验，可多次调用生成不同样式的图表
"""

import os
from pathlib import Path
import sys
import pickle
import glob

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # 添加父目录以导入cfnet

from interpretable_experiment.analysis.domain_analyzer import DomainAnalyzer


def find_latest_experiment():
    """找到最新的exp5实验目录"""
    runner_base = Path(__file__).resolve().parents[1]
    base_dir = runner_base / 'results' / 'exp5_real_data'
    if not os.path.exists(base_dir):
        return None

    exp_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith('20')])
    if not exp_dirs:
        return None

    return os.path.join(base_dir, exp_dirs[-1])


def load_experiment_data(exp_dir=None):
    """
    加载实验数据

    Args:
        exp_dir: 实验目录路径，如果为None则自动查找最新实验

    Returns:
        results: 分析结果字典
        feature_info: 特征信息字典
    """
    if exp_dir is None:
        exp_dir = find_latest_experiment()

    if exp_dir is None:
        raise FileNotFoundError("未找到实验数据，请先运行 run_exp5_real.py")

    print(f"加载实验数据: {exp_dir}")

    # 查找所有 .pkl 文件
    pkl_files = glob.glob(os.path.join(exp_dir, '*.pkl'))

    if not pkl_files:
        raise FileNotFoundError(f"在 {exp_dir} 中未找到 .pkl 数据文件")

    # 加载所有结果
    results = {}
    for pkl_file in pkl_files:
        name = os.path.basename(pkl_file).replace('.pkl', '')
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            results[name] = data

    # 从第一个结果中提取特征信息
    # 注意：这里假设所有模型使用相同的特征
    feature_names = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
    domain_importance = {
        'X1': 3, 'X2': 3,  # 高重要性 - 直接影响热交换
        'X3': 2, 'X4': 2, 'X5': 2, 'X7': 2,  # 中重要性
        'X6': 1, 'X8': 1  # 低重要性
    }

    feature_info = {
        'feature_names': feature_names,
        'domain_importance': domain_importance,
        'dataset_name': 'energy_efficiency'
    }

    print(f"  加载了 {len(results)} 个模型的分析结果")
    return results, feature_info, exp_dir


def replot_from_experiment(exp_dir=None, output_dir=None):
    """
    从保存的实验数据重新绘图

    Args:
        exp_dir: 实验目录，None则使用最新实验
        output_dir: 图表输出目录，None则使用实验目录
    """
    # 加载数据
    results, feature_info, loaded_exp_dir = load_experiment_data(exp_dir)

    if output_dir is None:
        output_dir = os.path.join(loaded_exp_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n图表将保存到: {output_dir}")

    # 创建分析器配置
    config = {
        'feature_info': feature_info,
        'device': 'cpu',
        'batch_size': 32
    }

    # 创建分析器
    analyzer = DomainAnalyzer(config, save_dir=output_dir)

    # 生成所有可视化
    print("\n" + "=" * 60)
    print("生成可视化图表...")
    print("=" * 60)

    viz_paths = analyzer.generate_visualizations(results)

    print("\n" + "=" * 60)
    print("图表生成完成!")
    print("=" * 60)
    for name, path in viz_paths.items():
        print(f"  - {name}: {path}")

    # 生成完整的分析报告
    print("\n生成分析报告...")
    report = analyzer.generate_report(results)
    report_path = os.path.join(output_dir, 'domain_analysis_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"报告已保存: {report_path}")

    return viz_paths


def quick_plot(style='comparison', exp_dir=None, output_path=None):
    """
    快速生成指定类型的图表

    Args:
        style: 图表类型 ('comparison', 'radar', 'heatmap', 'correlation')
        exp_dir: 实验目录
        output_path: 输出路径
    """
    results, feature_info, loaded_exp_dir = load_experiment_data(exp_dir)

    if output_path is None:
        output_dir = os.path.join(loaded_exp_dir, 'plots')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'feature_importance_{style}.png')

    config = {
        'feature_info': feature_info,
        'device': 'cpu',
        'batch_size': 32
    }

    analyzer = DomainAnalyzer(config, save_dir=os.path.dirname(output_path))

    print(f"生成 {style} 图表...")
    if style == 'comparison':
        path = analyzer.plot_feature_importance_comparison(results, output_path)
    elif style == 'radar':
        path = analyzer.plot_importance_radar(results, output_path)
    elif style == 'heatmap':
        path = analyzer.plot_importance_heatmap(results, output_path)
    elif style == 'correlation':
        path = analyzer.plot_importance_correlation(results, output_path)
    else:
        raise ValueError(f"未知的图表类型: {style}")

    print(f"图表已保存: {path}")
    return path


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='从保存的exp5实验数据重新绘图')
    parser.add_argument('--exp-dir', type=str, default=None,
                       help='实验目录路径（默认使用最新实验）')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='图表输出目录')
    parser.add_argument('--style', type=str, default='all',
                       choices=['all', 'comparison', 'radar', 'heatmap', 'correlation'],
                       help='图表类型')

    args = parser.parse_args()

    print("=" * 80)
    print("从保存的实验数据重新绘图")
    print("=" * 80)

    if args.style == 'all':
        replot_from_experiment(args.exp_dir, args.output_dir)
    else:
        quick_plot(args.style, args.exp_dir)

    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()

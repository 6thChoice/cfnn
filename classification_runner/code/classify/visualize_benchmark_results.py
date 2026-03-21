import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
CLASSIFY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CLASSIFY_DIR))

"""
Benchmark结果可视化脚本
生成收敛曲线、内存对比图、参数效率图等
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def load_results(filename):
    """加载benchmark结果"""
    with open(filename, 'r') as f:
        return json.load(f)


def plot_convergence_curves(results, output_dir='benchmark_results/figures'):
    """绘制收敛曲线对比图"""
    os.makedirs(output_dir, exist_ok=True)

    summaries = results.get('summaries', [])
    if not summaries:
        print("No convergence data found")
        return

    # 按数据集分组
    datasets = {}
    for summary in summaries:
        ds = summary['dataset']
        if ds not in datasets:
            datasets[ds] = []
        datasets[ds].append(summary)

    for dataset, ds_summaries in datasets.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = {'CFNet': '#1f77b4', 'Hybrid': '#ff7f0e', 'KAN': '#2ca02c', 'MLP': '#d62728'}

        # 左图：验证准确率 vs Epoch
        ax1 = axes[0]
        for summary in ds_summaries:
            model_type = summary['model_type']
            # 取第一次运行的历史
            if summary['all_results']:
                history = summary['all_results'][0]['convergence_history']
                ax1.plot(history['epoch'], history['val_acc'],
                        label=model_type, color=colors.get(model_type, 'gray'),
                        linewidth=2, alpha=0.8)

        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Validation Accuracy', fontsize=12)
        ax1.set_title(f'{dataset} - Convergence Curves', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 右图：验证准确率 vs Time
        ax2 = axes[1]
        for summary in ds_summaries:
            model_type = summary['model_type']
            if summary['all_results']:
                history = summary['all_results'][0]['convergence_history']
                ax2.plot(history['time_elapsed'], history['val_acc'],
                        label=model_type, color=colors.get(model_type, 'gray'),
                        linewidth=2, alpha=0.8)

        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Validation Accuracy', fontsize=12)
        ax2.set_title(f'{dataset} - Convergence vs Time', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/convergence_{dataset.lower()}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Convergence plots saved to {output_dir}/")


def plot_convergence_speed_comparison(results, output_dir='benchmark_results/figures'):
    """绘制达到目标准确率所需epoch对比图"""
    os.makedirs(output_dir, exist_ok=True)

    summaries = results.get('summaries', [])
    if not summaries:
        return

    datasets = list(set(s['dataset'] for s in summaries))
    model_types = list(set(s['model_type'] for s in summaries))

    # 收集数据
    for dataset in datasets:
        ds_data = [s for s in summaries if s['dataset'] == dataset]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(model_types))
        width = 0.25

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        target_keys = []

        # 找到所有目标key
        for s in ds_data:
            for key in s.get('convergence_stats', {}).keys():
                if key.startswith('epochs_to_') and key not in target_keys:
                    target_keys.append(key)

        target_keys = sorted(target_keys)

        for i, key in enumerate(target_keys):
            means = []
            stds = []
            for model in model_types:
                model_data = next((s for s in ds_data if s['model_type'] == model), None)
                if model_data and key in model_data.get('convergence_stats', {}):
                    stat = model_data['convergence_stats'][key]
                    if stat['mean'] is not None:
                        means.append(stat['mean'])
                        stds.append(stat['std'])
                    else:
                        means.append(0)
                        stds.append(0)
                else:
                    means.append(0)
                    stds.append(0)

            label = f"Target {key.replace('epochs_to_', '')}%"
            ax.bar(x + i*width, means, width, yerr=stds, label=label,
                   color=colors[i % len(colors)], alpha=0.8)

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Epochs to Target', fontsize=12)
        ax.set_title(f'{dataset} - Convergence Speed Comparison', fontsize=14)
        ax.set_xticks(x + width * (len(target_keys) - 1) / 2)
        ax.set_xticklabels(model_types)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/convergence_speed_{dataset.lower()}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Convergence speed comparison plots saved to {output_dir}/")


def plot_memory_comparison(results, output_dir='benchmark_results/figures'):
    """绘制内存占用对比图"""
    os.makedirs(output_dir, exist_ok=True)

    analysis = results.get('analysis', [])
    if not analysis:
        return

    import pandas as pd
    df = pd.DataFrame(analysis)

    datasets = df['dataset'].unique()

    for dataset in datasets:
        ds_df = df[df['dataset'] == dataset]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左图：参数内存对比
        ax1 = axes[0]
        models = ds_df['model'].unique()
        param_mbs = [ds_df[ds_df['model'] == m]['param_mb'].iloc[0] for m in models]

        colors = {'CFNet': '#1f77b4', 'Hybrid': '#ff7f0e', 'KAN': '#2ca02c', 'MLP': '#d62728'}
        bar_colors = [colors.get(m, 'gray') for m in models]

        ax1.bar(models, param_mbs, color=bar_colors, alpha=0.8)
        ax1.set_ylabel('Model Parameters (MB)', fontsize=12)
        ax1.set_title(f'{dataset} - Parameter Memory', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')

        # 右图：不同batch size的总内存
        ax2 = axes[1]
        batch_sizes = sorted(ds_df['batch_size'].unique())
        x = np.arange(len(batch_sizes))
        width = 0.2

        for i, model in enumerate(models):
            model_df = ds_df[ds_df['model'] == model]
            totals = [model_df[model_df['batch_size'] == bs]['total_mb'].iloc[0]
                     if len(model_df[model_df['batch_size'] == bs]) > 0 else 0
                     for bs in batch_sizes]
            ax2.bar(x + i*width, totals, width, label=model,
                   color=colors.get(model, 'gray'), alpha=0.8)

        ax2.set_xlabel('Batch Size', fontsize=12)
        ax2.set_ylabel('Total Memory (MB)', fontsize=12)
        ax2.set_title(f'{dataset} - Training Memory Usage', fontsize=14)
        ax2.set_xticks(x + width * (len(models) - 1) / 2)
        ax2.set_xticklabels(batch_sizes)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/memory_{dataset.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Memory comparison plots saved to {output_dir}/")


def plot_param_efficiency(results, output_dir='benchmark_results/figures'):
    """绘制参数效率图（Pareto前沿）"""
    os.makedirs(output_dir, exist_ok=True)

    all_results = results.get('results', [])
    if not all_results:
        return

    datasets = list(set(r['dataset'] for r in all_results))

    colors = {'CFNet': '#1f77b4', 'Hybrid': '#ff7f0e', 'KAN': '#2ca02c', 'MLP': '#d62728'}

    for dataset in datasets:
        ds_results = [r for r in all_results if r['dataset'] == dataset]

        fig, ax = plt.subplots(figsize=(10, 6))

        for result in ds_results:
            model_type = result['model_type']
            limits = []
            accs = []

            for limit, stats in result.get('results_by_limit', {}).items():
                limits.append(stats['mean_params'])
                accs.append(stats['mean_test_acc'])

            if limits:
                ax.plot(limits, accs, 'o-', label=model_type,
                       color=colors.get(model_type, 'gray'),
                       linewidth=2, markersize=8, alpha=0.8)

        ax.set_xlabel('Number of Parameters', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title(f'{dataset} - Parameter Efficiency (Pareto Frontier)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/param_efficiency_{dataset.lower()}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Parameter efficiency plots saved to {output_dir}/")


def plot_inference_speed(results, output_dir='benchmark_results/figures'):
    """绘制推理速度对比图"""
    os.makedirs(output_dir, exist_ok=True)

    analysis = results.get('analysis', [])
    if not analysis:
        return

    import pandas as pd
    df = pd.DataFrame(analysis)

    datasets = df['dataset'].unique()
    colors = {'CFNet': '#1f77b4', 'Hybrid': '#ff7f0e', 'KAN': '#2ca02c', 'MLP': '#d62728'}

    for dataset in datasets:
        ds_df = df[df['dataset'] == dataset]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左图：单样本延迟
        ax1 = axes[0]
        single = ds_df[ds_df['batch_size'] == 1]
        models = single['model'].unique()

        for model in models:
            model_df = single[single['model'] == model]
            if not model_df.empty:
                ax1.bar(model, model_df['ms_per_sample'].iloc[0] * 1000,
                       color=colors.get(model, 'gray'), alpha=0.8)

        ax1.set_ylabel('Latency (ms/sample)', fontsize=12)
        ax1.set_title(f'{dataset} - Single Sample Latency', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')

        # 右图：不同batch size的吞吐量
        ax2 = axes[1]
        batch_sizes = sorted(ds_df['batch_size'].unique())

        for model in models:
            model_df = ds_df[ds_df['model'] == model]
            throughputs = [model_df[model_df['batch_size'] == bs]['samples_per_sec'].iloc[0]
                          if len(model_df[model_df['batch_size'] == bs]) > 0 else 0
                          for bs in batch_sizes]
            ax2.plot(batch_sizes, throughputs, 'o-', label=model,
                    color=colors.get(model, 'gray'), linewidth=2, markersize=8)

        ax2.set_xlabel('Batch Size', fontsize=12)
        ax2.set_ylabel('Throughput (samples/sec)', fontsize=12)
        ax2.set_title(f'{dataset} - Inference Throughput', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/inference_{dataset.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Inference speed plots saved to {output_dir}/")


def generate_summary_table(convergence_file, memory_file, param_file, inference_file,
                           output_file='benchmark_results/summary_table.md'):
    """生成汇总表格"""

    lines = []
    lines.append("# CFNet Benchmark Results Summary\n")
    lines.append("\n## 1. Convergence Speed\n")
    lines.append("\n| Dataset | Model | Test Acc | Epochs to 80% | Epochs to 90% | Avg Speed |\n")
    lines.append("|---------|-------|----------|---------------|---------------|-----------|\n")

    try:
        conv_results = load_results(convergence_file)
        for summary in conv_results.get('summaries', []):
            ds = summary['dataset']
            model = summary['model_type']
            acc = f"{summary['mean_test_acc']:.4f}±{summary['std_test_acc']:.4f}"

            stats = summary.get('convergence_stats', {})
            e80 = stats.get('epochs_to_80', {}).get('mean', 'N/A')
            e80 = f"{e80:.1f}" if isinstance(e80, (int, float)) else e80

            e90 = stats.get('epochs_to_90', {}).get('mean', 'N/A')
            e90 = f"{e90:.1f}" if isinstance(e90, (int, float)) else e90

            speed = summary['all_results'][0]['avg_improvement_speed'] if summary['all_results'] else 0
            speed_str = f"{speed:.4f}"

            lines.append(f"| {ds} | {model} | {acc} | {e80} | {e90} | {speed_str} |\n")
    except Exception as e:
        lines.append(f"Error loading convergence results: {e}\n")

    lines.append("\n## 2. Memory Usage\n")
    lines.append("\n| Dataset | Model | Params | Param MB | Forward MB | Backward MB | Total MB |\n")
    lines.append("|---------|-------|--------|----------|------------|-------------|----------|\n")

    try:
        mem_results = load_results(memory_file)
        for item in mem_results.get('analysis', []):
            if item['batch_size'] == 128:  # 只显示一个batch size
                lines.append(f"| {item['dataset']} | {item['model']} | {item['params']:,} | "
                           f"{item['param_mb']:.2f} | {item['forward_mb']:.2f} | "
                           f"{item['backward_mb']:.2f} | {item['total_mb']:.2f} |\n")
    except Exception as e:
        lines.append(f"Error loading memory results: {e}\n")

    lines.append("\n## 3. Inference Speed (GPU)\n")
    lines.append("\n| Dataset | Model | Latency (ms) | Throughput (samples/s) |\n")
    lines.append("|---------|-------|--------------|------------------------|\n")

    try:
        inf_results = load_results(inference_file)
        for item in inf_results.get('analysis', []):
            if item['batch_size'] == 1:  # 单样本延迟
                lines.append(f"| {item['dataset']} | {item['model']} | "
                           f"{item['ms_per_sample']*1000:.3f} | "
                           f"{item['samples_per_sec']:.1f} |\n")
    except Exception as e:
        lines.append(f"Error loading inference results: {e}\n")

    # 写入文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.writelines(lines)

    print(f"Summary table saved to {output_file}")


def main():
    """主函数：生成所有可视化"""
    result_dir = 'benchmark_results'
    figure_dir = f'{result_dir}/figures'

    os.makedirs(figure_dir, exist_ok=True)

    # 生成各实验的可视化
    try:
        conv_results = load_results(f'{result_dir}/convergence_results.json')
        plot_convergence_curves(conv_results, figure_dir)
        plot_convergence_speed_comparison(conv_results, figure_dir)
    except Exception as e:
        print(f"Error plotting convergence results: {e}")

    try:
        mem_results = load_results(f'{result_dir}/memory_results.json')
        plot_memory_comparison(mem_results, figure_dir)
    except Exception as e:
        print(f"Error plotting memory results: {e}")

    try:
        param_results = load_results(f'{result_dir}/param_efficiency_results.json')
        plot_param_efficiency(param_results, figure_dir)
    except Exception as e:
        print(f"Error plotting param efficiency results: {e}")

    try:
        inf_results = load_results(f'{result_dir}/inference_results.json')
        plot_inference_speed(inf_results, figure_dir)
    except Exception as e:
        print(f"Error plotting inference results: {e}")

    # 生成汇总表格
    generate_summary_table(
        f'{result_dir}/convergence_results.json',
        f'{result_dir}/memory_results.json',
        f'{result_dir}/param_efficiency_results.json',
        f'{result_dir}/inference_results.json',
        f'{result_dir}/summary_table.md'
    )

    print("\nAll visualizations generated successfully!")


if __name__ == "__main__":
    main()

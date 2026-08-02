import json
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict

def aggregate_results(data_list):
    """
    将原始结果按配置分组，计算平均 Loss。
    Input: [{'params': 100, 'loss': 0.1, 'config': 'A'}, {'params': 100, 'loss': 0.2, 'config': 'A'}, ...]
    Output: [(params, mean_loss, config_label), ...]
    """
    grouped = defaultdict(lambda: {'params': 0, 'losses': []})
    
    for entry in data_list:
        cfg = entry['config']
        grouped[cfg]['params'] = entry['params']
        grouped[cfg]['losses'].append(entry['loss'])
        
    aggregated_list = []
    for cfg, val in grouped.items():
        mean_loss = np.mean(val['losses'])
        aggregated_list.append((val['params'], mean_loss, cfg))
        
    # 按参数量排序
    aggregated_list.sort(key=lambda x: x[0])
    return aggregated_list

def plot_single_function(func_name, data, save_dir):
    plt.figure(figsize=(10, 6))
    
    colors = {'HybridRationalNet': 'blue', 'MLP': 'red'}
    markers = {'HybridRationalNet': 'o', 'MLP': 's'}
    
    # 遍历两种模型类型
    for model_type, points in data.items():
        if not points: continue
        
        # 聚合数据：计算多次运行的平均值
        agg_points = aggregate_results(points)
        
        params = [p[0] for p in agg_points]
        mean_losses = [p[1] for p in agg_points]
        labels = [p[2] for p in agg_points]
        
        color = colors.get(model_type, 'gray')
        marker = markers.get(model_type, 'x')
        
        # 绘制趋势线
        plt.plot(params, mean_losses, linestyle='--', color=color, alpha=0.5, label=f'{model_type} Trend (Avg)')
        # 绘制散点
        plt.scatter(params, mean_losses, c=color, marker=marker, s=80, label=f'{model_type} Mean', zorder=5)
        
        # 标注文本
        for i, txt in enumerate(labels):
            offset = (0, 10) if model_type == 'HybridRationalNet' else (0, -15)
            plt.annotate(txt, (params[i], mean_losses[i]), 
                         textcoords="offset points", xytext=offset, ha='center', 
                         fontsize=8, color=color, alpha=0.8)

    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel('Number of Parameters (Log Scale)', fontsize=12)
    plt.ylabel('Mean MSE Loss (Log Scale, 3 Seeds)', fontsize=12)
    plt.title(f'Pareto Frontier (Avg of 3 Runs): {func_name}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    # 保存
    file_path = os.path.join(save_dir, f"{func_name}_pareto_avg.png")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    print(f"Saved plot for {func_name} to {file_path}")

def main():
    json_file = "multiseed_results.json"
    output_dir = "results_plots_avg"
    
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found. Please run train_multiseed.py first.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(json_file, "r") as f:
        full_results = json.load(f)
        
    print(f"Loaded results for {len(full_results)} functions.")
    
    for func_name, model_data in full_results.items():
        plot_single_function(func_name, model_data, output_dir)
        
    print("All average plots generated.")

if __name__ == "__main__":
    main()
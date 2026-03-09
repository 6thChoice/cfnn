from pathlib import Path
import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置路径与参数
# ==========================================
FILE_PATHS = [
    str((Path(__file__).resolve().parent / "nn_hard_1/plot/model_lead_metrics_1.json")),
    str((Path(__file__).resolve().parent / "nn_hard_2/plot/model_lead_metrics_2.json")),
    str((Path(__file__).resolve().parent / "cfnn_prefer_1/plot/model_lead_metrics_3.json")),
    str((Path(__file__).resolve().parent / "cfnn_prefer_2/plot/model_lead_metrics_4.json"))
]

MODEL_TYPES = ["standard", "boost", "moe", "hybrid"]
RUNNER_BASE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = str(RUNNER_BASE / "results" / "unified_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. 数据聚合与统计
# ==========================================
# 结构: aggregated_data[model_type][(d,p)] = {'loss': [], 'rmse': []}
aggregated_data = {m: {} for m in MODEL_TYPES}

for path in FILE_PATHS:
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        continue
    
    with open(path, 'r') as f:
        data = json.load(f)
        for m_type in MODEL_TYPES:
            if m_type not in data: continue
            for entry in data[m_type]:
                if entry['status'] != 'success': continue
                
                key = (entry['d'], entry['p'])
                if key not in aggregated_data[m_type]:
                    aggregated_data[m_type][key] = {'loss': [], 'rmse': []}
                
                aggregated_data[m_type][key]['loss'].append(entry['loss_lead'])
                aggregated_data[m_type][key]['rmse'].append(entry['rmse_lead'])

# ==========================================
# 3. 绘图函数
# ==========================================
for m_type in MODEL_TYPES:
    print(f"Plotting unified chart for {m_type}...")
    
    # 提取排序后的标签和统计量
    sorted_keys = sorted(aggregated_data[m_type].keys())
    labels = [f"({k[0]},{k[1]})" for k in sorted_keys]
    
    loss_means = [np.mean(aggregated_data[m_type][k]['loss']) for k in sorted_keys]
    loss_stds = [np.std(aggregated_data[m_type][k]['loss']) for k in sorted_keys]
    
    rmse_means = [np.mean(aggregated_data[m_type][k]['rmse']) for k in sorted_keys]
    rmse_stds = [np.std(aggregated_data[m_type][k]['rmse']) for k in sorted_keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(24, 10))
    ax2 = ax1.twinx()

    # --- 绘制 Loss 均值柱状图及误差棒 ---
    colors_loss = ['#3498db' if val >= 0 else '#e74c3c' for val in loss_means]
    bars1 = ax1.bar(x - width/2, loss_means, width, yerr=loss_stds, 
                    label='Avg Loss Lead %', color=colors_loss, alpha=0.6, 
                    edgecolor='black', capsize=5, error_kw={'elinewidth': 2, 'ecolor': 'gray'})

    # --- 绘制 RMSE 均值柱状图及误差棒 ---
    colors_rmse = ['#2980b9' if val >= 0 else '#c0392b' for val in rmse_means]
    bars2 = ax2.bar(x + width/2, rmse_means, width, yerr=rmse_stds, 
                    label='Avg RMSE Lead %', color=colors_rmse, alpha=0.4, 
                    edgecolor='black', hatch='//', capsize=5, error_kw={'elinewidth': 2, 'ecolor': 'gray'})

    # --- 绘制趋势连线 (连接均值) ---
    ax1.plot(x - width/2, loss_means, color='#aa635c', marker='o', markersize=10, 
             linewidth=3, label='Loss Mean Trend', zorder=5)
    ax2.plot(x + width/2, rmse_means, color='#326283', marker='s', markersize=10, 
             linewidth=3, linestyle='--', label='RMSE Mean Trend', zorder=5)

    # --- 样式设置 ---
    for ax in [ax1, ax2]:
        ax.set_ylim(-110, 110) # 稍微宽一点以容纳误差棒
        ax.axhline(0, color='black', linewidth=1.5)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', labelsize=28)

    ax1.set_ylabel('Mean Loss Lead (%)', fontsize=40)
    ax2.set_ylabel('Mean RMSE Lead (%)', fontsize=40)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, ha='center', fontsize=32)
    
    # plt.title(f"Unified Performance Lead: CFNet-{m_type.upper()} (Mean ± Std)", fontsize=32, pad=20)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    if m_type == "standard":
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=28, ncol=2)
    else:
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=28, ncol=2)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{m_type}_unified_lead_std.pdf")
    plt.savefig(save_path)
    plt.close()

print(f"All unified plots have been saved in {OUTPUT_DIR}/")

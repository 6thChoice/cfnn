import os
import json
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 基础配置与路径设置
# ==========================================
MODEL_TYPES = ["standard", "boost", "moe", "hybrid"]
D_RANGE = range(1, 6, 2)
P_RANGE = range(1, 6, 2)

N_LAST_STEPS = 300

# 假设当前运行路径包含 mlp/, standard/ 等文件夹
BASE_PATH = ".." 
PLOT_TYPE = "pdf"
os.makedirs(f"{PLOT_TYPE}_{N_LAST_STEPS}", exist_ok=True)

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def get_pareto_front(x, y):
    """计算帕累托前沿（用于最小化误差问题）"""
    sorted_indices = np.argsort(x)
    x_sorted = np.array(x)[sorted_indices]
    y_sorted = np.array(y)[sorted_indices]
    
    pareto_x, pareto_y = [x_sorted[0]], [y_sorted[0]]
    current_min_y = y_sorted[0]
    
    for i in range(1, len(x_sorted)):
        if y_sorted[i] < current_min_y:
            pareto_x.append(x_sorted[i])
            pareto_y.append(y_sorted[i])
            current_min_y = y_sorted[i]
    return pareto_x, pareto_y

# ==========================================
# 2. 核心绘图函数
# ==========================================
def plot_trajectory_atlas(m_type, metric_key, title_suffix):
    """
    绘制 5x5 的轨迹图矩阵
    metric_key: 'train_loss' 或 'val_rmse'
    """
    fig, axes = plt.subplots(3, 3, figsize=(25, 25), sharex=True)
    # fig.suptitle(f"{m_type.upper()} {title_suffix} Comparison Atlas (Rows: d, Cols: p)", fontsize=24, fontweight='bold')
    
    all_configs_found = False
    
    for d_idx, d in enumerate(D_RANGE):
        for p_idx, p in enumerate(P_RANGE):
            ax = axes[d_idx, p_idx]
            
            # 加载数据
            mlp_path = os.path.join(BASE_PATH, "mlp", "result", f"comparison_{m_type}_d{d}_p{p}.json")
            cf_path = os.path.join(BASE_PATH, m_type, "result", f"cfnet_full_results_{m_type}_d{d}_p{p}.json")
            
            mlp_data = load_json(mlp_path)
            cf_data = load_json(cf_path)
            
            if mlp_data and cf_data:
                all_configs_found = True
                cf_hist = cf_data["history"]["cfnet"][metric_key][-N_LAST_STEPS:]
                mlp_hist = mlp_data["history"]["mlp"][metric_key][-N_LAST_STEPS:]
                
                line_cf, = ax.plot(cf_hist, label="CFNN", color='C0', lw=15)
                line_mlp, = ax.plot(mlp_hist, label="MLP", color='C1', linestyle='--', lw=15)
                
                if metric_key == 'train_loss':
                    ax.set_yscale('log')
                
                # 标注参数量信息
                params = mlp_data["config"]["params"]
                # ax.set_title(f"d={d}, p={p} (Params: {params})", fontsize=10)
                ax.set_title(f"d={d}, p={p}", fontsize=70)
                # if d_idx == 0 and p_idx == 0:
                #     ax.legend(fontsize=8)
                ax.tick_params(axis='x', labelsize=70)
                ax.tick_params(axis='y', labelsize=70)
            else:
                ax.text(0.5, 0.5, "Data Missing", ha='center', va='center', alpha=0.5)
            
            ax.grid(True, alpha=0.2)

    # --- 核心修改部分：创建全局横向图例 ---
    if line_cf and line_mlp:
        # loc='upper center' 居中，ncol=2 设置为两列（一行），bbox_to_anchor 精确定位
        fig.legend(handles=[line_cf, line_mlp], 
                   labels=['CFNN', 'MLP'],
                   loc='upper center', 
                   bbox_to_anchor=(0.5, 1.02), 
                   ncol=2, 
                   fontsize=70, 
                   frameon=False,
                   edgecolor='black')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_path = f"{PLOT_TYPE}_{N_LAST_STEPS}/{m_type}_{metric_key}_atlas.{PLOT_TYPE}"
    plt.savefig(save_path)
    plt.close()
    return all_configs_found

# ==========================================
# 3. 执行主循环
# ==========================================
for m_type in MODEL_TYPES:
    print(f"Processing {m_type}...")
    
    # A. 绘制训练损失全集 (5x5)
    plot_trajectory_atlas(m_type, "train_loss", "Training Loss")
    
    # B. 绘制验证精度全集 (5x5)
    plot_trajectory_atlas(m_type, "val_rmse", "Validation RMSE")

print("All plots generated successfully.")
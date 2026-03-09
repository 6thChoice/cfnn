import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

# ==========================================
# 1. 配置参数与路径
# ==========================================
MODEL_TYPES = ["standard", "boost", "moe", "hybrid"]
D_RANGE = range(1, 6) 
P_RANGE = range(1, 6) 
BASE_PATH = ".."
PLOT_TYPE = "pdf"

os.makedirs(PLOT_TYPE, exist_ok=True)

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def load_json(path):
    if not os.path.exists(path): return None
    with open(path, 'r') as f: return json.load(f)

# 设置字体大小变量，方便统一调整
LABEL_FONT_SIZE = 80  # 轴标题字体
TICK_FONT_SIZE = 70   # 轴刻度数字字体
CBAR_LABEL_SIZE = 80  # 颜色条标题字体
CBAR_TICK_SIZE = 70   # 颜色条刻度数字字体

# ==========================================
# 2. 数据处理与热力图绘制
# ==========================================
for m_type in MODEL_TYPES:
    rmse_matrix = np.full((5, 5), np.nan)
    
    for d_idx, d in enumerate(D_RANGE):
        for p_idx, p in enumerate(P_RANGE):
            cf_path = os.path.join(BASE_PATH, m_type, "result", f"cfnet_full_results_{m_type}_d{d}_p{p}.json")
            data = load_json(cf_path)
            
            if data and "test_data" in data:
                y_true = data["test_data"]["y_true"]
                y_cf = data["test_data"]["y_cfnet"]
                rmse = calculate_rmse(y_true, y_cf)
                rmse_matrix[d-1, p-1] = rmse

    # --- 优化绘图设置 ---
    plt.figure(figsize=(21, 19)) # 增加尺寸以适应大字体
    
    norm = LogNorm(vmin=np.nanmin(rmse_matrix), vmax=np.nanmax(rmse_matrix))
    
    ax = sns.heatmap(rmse_matrix, 
                     fmt=".4f", 
                     norm=norm, 
                     cmap="YlGnBu_r", 
                     yticklabels=[f"{d}" for d in D_RANGE], 
                     xticklabels=[f"{p}" for p in P_RANGE],
                     cbar_kws={'label': 'Test RMSE'},
                     )
    
    plt.ylabel("Continued Fraction Depth ($d$)", fontsize=LABEL_FONT_SIZE)
    plt.xlabel("Polynomial Degree ($p$)", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    
    # 颜色条处理
    cbar = ax.collections[0].colorbar
    
    # 1. 设置颜色条标题
    cbar.set_label('Test RMSE', fontsize=CBAR_LABEL_SIZE)
    
    # 2. 彻底隐去所有刻度信息（主要和次要）
    # which='both' 作用于主要和次要刻度
    # length=0 隐藏刻度线，width=0 确保彻底不可见
    cbar.ax.tick_params(axis='y', which='both', length=0, width=0, labelsize=0)
    
    # 3. 再次确保刻度位置列表为空
    cbar.ax.yaxis.set_ticks([])
    cbar.set_ticks([])

    plt.tight_layout(rect=[0, 0, 1, 1])
    
    save_name = f"{PLOT_TYPE}/{m_type}_test_rmse_heatmap.{PLOT_TYPE}"
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"Generated optimized heatmap for {m_type}: {save_name}")

print("All optimized heatmaps generated successfully.")
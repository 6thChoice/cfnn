import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置参数与路径
# ==========================================
MODEL_TYPES = ["standard"]
BASE_PATH = ".." 

# 定义需要读取的参数对
params_pair = [
    # {"d": 10, "p": 10}, {"d": 15, "p": 15}, {"d": 20, "p": 20}, {"d": 25, "p": 25},
    {"d": 5, "p": 5}, {"d": 10, "p": 5},  {"d": 15, "p": 5},  {"d": 20, "p": 5},  {"d": 25, "p": 5},
    {"d": 5, "p": 5}, {"d": 5, "p": 10},  {"d": 5, "p": 15},  {"d": 5, "p": 20},  {"d": 5, "p": 25}
]

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def load_json(path):
    if not os.path.exists(path): return None
    with open(path, 'r') as f: return json.load(f)

# ==========================================
# 2. 核心绘图逻辑
# ==========================================
for m_type in MODEL_TYPES:
    both_vary = ([], [])
    depth_vary = ([], [])
    degree_vary = ([], [])

    for pair in params_pair:
        d, p = pair["d"], pair["p"]
        cf_path = os.path.join(BASE_PATH, m_type, "result", f"cfnet_full_results_{m_type}_d{d}_p{p}.json")
        data = load_json(cf_path)
        
        if data and "test_data" in data:
            rmse = calculate_rmse(data["test_data"]["y_true"], data["test_data"]["y_cfnet"])
            
            # 基于数值关系分类
            if d == p:
                both_vary[0].append(d)
                both_vary[1].append(rmse)
            elif p == 5:
                depth_vary[0].append(d)
                depth_vary[1].append(rmse)
            elif d == 5:
                degree_vary[0].append(p)
                degree_vary[1].append(rmse)

    # 排序
    for data_list in [both_vary, depth_vary, degree_vary]:
        if data_list[0]:
            sort_idx = np.argsort(data_list[0])
            data_list[0][:] = [data_list[0][i] for i in sort_idx]
            data_list[1][:] = [data_list[1][i] for i in sort_idx]

    # --- 开始绘图 ---
    plt.figure(figsize=(10, 7))
    
    if degree_vary[0]:
        plt.plot(degree_vary[0], degree_vary[1], 'o-', label=r"Varying Degree $p$ (Fixed $d=5$)", linewidth=2.5, markersize=10)
    if depth_vary[0]:
        plt.plot(depth_vary[0], depth_vary[1], 's-', label=r"Varying Depth $d$ (Fixed $p=5$)", linewidth=2.5, markersize=10)
    if both_vary[0]:
        plt.plot(both_vary[0], both_vary[1], 'D-', label=r"Varying Both ($d=p$)", linewidth=2.5, markersize=10, color='forestgreen')

    # --- 关键修改：设置 y 轴范围与对数坐标 ---
    plt.yscale('log') 
    # plt.ylim(0.02, 0.1)  # 限制范围以聚焦底部竞争关系

    # 样式美化
    plt.title(f"Model Scaling Performance (Focused): {m_type.upper()}", fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Varying Parameter Value ($d$ or $p$)", fontsize=15)
    plt.ylabel("Test RMSE (Log Scale)", fontsize=15)
    
    # 网格线设置（包含次要刻度网格）
    plt.grid(True, which="both", ls="--", alpha=0.6)
    
    plt.legend(fontsize=12, frameon=True, shadow=True, loc='best')
    plt.xticks([10, 15, 20, 25], fontsize=12)
    plt.yticks(fontsize=12)
    
    save_path = f"scaling_focused_{m_type}_log.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated focused log-scale plot for {m_type}")
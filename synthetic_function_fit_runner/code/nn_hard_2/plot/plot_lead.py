import os
import json
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 配置与工具函数
# ==========================================
MODEL_TYPES = ["standard", "boost", "moe", "hybrid"]
D_RANGE = range(1, 6, 2)
P_RANGE = range(1, 6, 2)
BASE_PATH = ".."  # JSON 文件根目录

PLOT_TYPE = "pdf"
os.makedirs(PLOT_TYPE, exist_ok=True)

def load_json(path):
    if not os.path.exists(path): return None
    with open(path, 'r') as f: return json.load(f)

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

# ==========================================
# 2. 绘制领先量柱状图矩阵
# ==========================================
for m_type in MODEL_TYPES:
    print(f"Generating Lead Atlas for {m_type}...")
    
    # 创建 5x5 布局
    fig, axes = plt.subplots(3, 3, figsize=(25, 25), sharex=True)
    # fig.suptitle(f"Advantage of CFNet-{m_type.upper()} over MLP (%)\n"
    #              r"Formula: $\frac{MLP - CFNet}{MLP} \times 100\%$ (Positive is Better)", 
    #              fontsize=24, fontweight='bold')

    for d_idx, d in enumerate(D_RANGE):
        for p_idx, p in enumerate(P_RANGE):
            ax = axes[d_idx, p_idx]
            
            # 定义文件路径
            mlp_path = os.path.join(BASE_PATH, "mlp", "result", f"comparison_{m_type}_d{d}_p{p}.json")
            cf_path = os.path.join(BASE_PATH, m_type, "result", f"cfnet_full_results_{m_type}_d{d}_p{p}.json")
            
            mlp_data = load_json(mlp_path)
            cf_data = load_json(cf_path)
            
            if mlp_data and cf_data:
                # --- A. 训练损失领先量 ---
                cf_loss = cf_data["history"]["cfnet"]["train_loss"][-1]
                mlp_loss = mlp_data["history"]["mlp"]["train_loss"][-1]
                loss_lead = ((mlp_loss - cf_loss) / mlp_loss) * 100
                
                # --- B. 测试集 RMSE 领先量 ---
                y_true = np.array(cf_data["test_data"]["y_true"])
                y_cf = np.array(cf_data["test_data"]["y_cfnet"])
                y_mlp = np.array(mlp_data["test_data"]["y_mlp"])
                
                cf_rmse = calculate_rmse(y_true, y_cf)
                mlp_rmse = calculate_rmse(y_true, y_mlp)
                rmse_lead = ((mlp_rmse - cf_rmse) / mlp_rmse) * 100
                
                # --- C. 绘制柱状图 ---
                metrics = ['Loss', 'RMSE']
                leads = [loss_lead, rmse_lead]
                # 蓝色代表领先，红色代表落后
                colors = ['#3498db' if x >= 0 else '#e74c3c' for x in leads]
                
                bars = ax.bar(metrics, leads, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
                ax.axhline(0, color='black', linewidth=1)
                
                # 在柱子上标注具体百分比
                # for bar in bars:
                #     height = bar.get_height()
                #     ax.text(bar.get_x() + bar.get_width()/2, height + (2 if height >= 0 else -6),
                #             f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                #             fontsize=10, fontweight='bold')
                
                ax.set_title(f"d={d}, p={p}", fontsize=70)
                # 设置纵轴范围，确保标注可见
                ax.set_ylim(min(-25, min(leads)*1.3), max(100, max(leads)*1.2))
                ax.tick_params(axis='x', labelsize=70)
                ax.tick_params(axis='y', labelsize=70)
            else:
                ax.text(0.5, 0.5, "Data Missing", ha='center', va='center', color='gray')
            
            ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_path = f"{PLOT_TYPE}/{m_type}_lead_comparison_atlas.{PLOT_TYPE}"
    plt.savefig(save_path)
    plt.close()

print("All lead comparison plots have been generated.")
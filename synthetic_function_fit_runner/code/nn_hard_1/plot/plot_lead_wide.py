import os
import json
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 配置与工具函数 (保持不变)
# ==========================================
MODEL_TYPES = ["standard", "boost", "moe", "hybrid"]
D_RANGE = range(1, 6, 2)
P_RANGE = range(1, 6, 2)
BASE_PATH = ".." 

PLOT_TYPE = "pdf"
os.makedirs(PLOT_TYPE+"_wide", exist_ok=True)

def load_json(path):
    if not os.path.exists(path): return None
    with open(path, 'r') as f: return json.load(f)

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

# ==========================================
# 2. 绘制极宽的领先量对比图
# ==========================================
for m_type in MODEL_TYPES:
    print(f"Generating Comprehensive Lead Chart for {m_type}...")
    
    labels = []
    loss_leads = []
    rmse_leads = []

    for d in D_RANGE:
        for p in P_RANGE:
            labels.append(f"({d},{p})")
            mlp_path = os.path.join(BASE_PATH, "mlp", "result", f"comparison_{m_type}_d{d}_p{p}.json")
            cf_path = os.path.join(BASE_PATH, m_type, "result", f"cfnet_full_results_{m_type}_d{d}_p{p}.json")
            
            mlp_data = load_json(mlp_path)
            cf_data = load_json(cf_path)
            
            if mlp_data and cf_data:
                cf_loss = cf_data["history"]["cfnet"]["train_loss"][-1]
                mlp_loss = mlp_data["history"]["mlp"]["train_loss"][-1]
                loss_lead = max(-100, ((mlp_loss - cf_loss) / mlp_loss) * 100)
                
                y_true = np.array(cf_data["test_data"]["y_true"])
                cf_rmse = calculate_rmse(y_true, cf_data["test_data"]["y_cfnet"])
                mlp_rmse = calculate_rmse(y_true, mlp_data["test_data"]["y_mlp"])
                rmse_lead = max(-100, ((mlp_rmse - cf_rmse) / mlp_rmse) * 100)
                
                loss_leads.append(loss_lead)
                rmse_leads.append(rmse_lead)
            else:
                loss_leads.append(0)
                rmse_leads.append(0)

    # --- 绘图逻辑开始 ---
    x = np.arange(len(labels))
    width = 0.35 
    
    fig, ax1 = plt.subplots(figsize=(24, 10)) # 略微加宽加高以适应大字体
    ax2 = ax1.twinx() 

    # 1. 绘制柱状图
    colors_loss = ['#3498db' if val >= 0 else '#e74c3c' for val in loss_leads]
    bars1 = ax1.bar(x - width/2, loss_leads, width, label='Loss Lead %', 
                    color=colors_loss, alpha=0.6, edgecolor='black')

    colors_rmse = ['#2980b9' if val >= 0 else '#c0392b' for val in rmse_leads]
    bars2 = ax2.bar(x + width/2, rmse_leads, width, label='RMSE Lead %', 
                    color=colors_rmse, alpha=0.4, edgecolor='black', hatch='//')

    # 2. 绘制顶部连线 (关键改动)
    # Loss 连线对齐 ax1 的 X 偏移
    ax1.plot(x - width/2, loss_leads, color="#aa635c", marker='o', markersize=10, 
             linewidth=3, label='Loss Trend', zorder=5)
    
    # RMSE 连线对齐 ax2 的 X 偏移
    ax2.plot(x + width/2, rmse_leads, color="#326283", marker='s', markersize=10, 
             linewidth=3, linestyle='--', label='RMSE Trend', zorder=5)

    # 3. 统一设置
    for ax in [ax1, ax2]:
        ax.set_ylim(-105, 105) 
        ax.axhline(0, color='black', linewidth=1.5)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', labelsize=32)

    ax1.set_ylabel('Loss Lead (%)', fontsize=40, labelpad=15)
    ax2.set_ylabel('RMSE Lead (%)', fontsize=40, labelpad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, ha='center', fontsize=28)

    # 4. 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=28, ncol=2)

    plt.tight_layout()
    save_path = f"{PLOT_TYPE}_wide/{m_type}_combined_lead_chart.{PLOT_TYPE}"
    plt.savefig(save_path)
    plt.close()

print("All combined lead charts with trend lines have been generated.")
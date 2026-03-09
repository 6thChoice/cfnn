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

    d = 5
    p = 5

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

        print(f"MLP RMSE: {mlp_rmse} | {m_type} RMSE: {cf_rmse}")

print("All lead comparison plots have been generated.")
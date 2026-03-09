import os
import json
import numpy as np

# ==========================================
# 1. 配置与工具函数
# ==========================================
MODEL_TYPES = ["standard", "boost", "moe", "hybrid"]
D_RANGE = range(1, 6, 2)
P_RANGE = range(1, 6, 2)
BASE_PATH = ".." 
OUTPUT_FILE = "model_lead_metrics_2.json"

def load_json(path):
    if not os.path.exists(path): return None
    with open(path, 'r') as f: return json.load(f)

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

# ==========================================
# 2. 执行计算并保存数据
# ==========================================
results_db = {}

for m_type in MODEL_TYPES:
    print(f"Processing metrics for {m_type}...")
    results_db[m_type] = []
    print("=" * 50)

    for d in D_RANGE:
        for p in P_RANGE:
            # 路径构建
            mlp_path = os.path.join(BASE_PATH, "mlp", "result", f"comparison_{m_type}_d{d}_p{p}.json")
            cf_path = os.path.join(BASE_PATH, m_type, "result", f"cfnet_full_results_{m_type}_d{d}_p{p}.json")
            
            mlp_data = load_json(mlp_path)
            cf_data = load_json(cf_path)
            
            entry = {
                "d": d,
                "p": p,
                "loss_lead": None,
                "rmse_lead": None,
                "status": "missing"
            }

            if mlp_data and cf_data:
                # 1. 计算 Loss 领先量 (截断到 -100)
                cf_loss = cf_data["history"]["cfnet"]["train_loss"][-1]
                mlp_loss = mlp_data["history"]["mlp"]["train_loss"][-1]
                loss_lead = ((mlp_loss - cf_loss) / mlp_loss) * 100
                entry["loss_lead"] = round(max(-100, loss_lead), 4)
                
                # 2. 计算 RMSE 领先量 (截断到 -100)
                y_true = np.array(cf_data["test_data"]["y_true"])
                cf_rmse = calculate_rmse(y_true, cf_data["test_data"]["y_cfnet"])
                mlp_rmse = calculate_rmse(y_true, mlp_data["test_data"]["y_mlp"])
                rmse_lead = ((mlp_rmse - cf_rmse) / mlp_rmse) * 100
                entry["rmse_lead"] = round(max(-100, rmse_lead), 4)
                print(f"CF RMSE: {cf_rmse}")
                print(f"MLP RMSE: {mlp_rmse}")
                
                entry["status"] = "success"

            results_db[m_type].append(entry)

# 保存到 JSON 文件
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(results_db, f, indent=4, ensure_ascii=False)

print(f"Successfully saved metrics to {OUTPUT_FILE}")
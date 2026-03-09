import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# 配置
RESULTS_FILE = "comparison_results.json"
OUTPUT_DIR = "loss_plots/pdf"
SMOOTHING_WINDOW = 1  # 设置大于1可进行滑动平均平滑

# 颜色和样式配置
MODEL_STYLES = {
    "Hybrid":      {"color": "red", "label": "CFNN-Hybrid"},
    "CFNet_Std":   {"color": "blue", "label": "CFNN"},
    "Boost":       {"color": "green", "label": "CFNN-Boost"},
    "MoE":         {"color": "purple", "label": "CFNN-MoE"},
}

BASELINE_STYLES = {
    "MLP_1x":  {"color": "orange", "style": "--", "label": "MLP (1x)"},
    "MLP_10x": {"color": "darkorange", "style": "-",  "label": "MLP (10x)"},
    "KAN_1x":  {"color": "cyan",   "style": "--", "label": "KAN (1x)"},
    "KAN_10x": {"color": "darkcyan", "style": "-",  "label": "KAN (10x)"},
}

def moving_average(data, window_size):
    if window_size <= 1: return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def get_model_category(model_name):
    """解析模型名称，返回 (BaseModelName, Type, Multiplier)"""
    # 检查是否是 Baseline
    if "_for_" in model_name:
        parts = model_name.split("_for_")
        baseline_type = parts[0] # e.g., MLP_1x
        base_model = parts[1]    # e.g., Hybrid
        return base_model, baseline_type, None
    else:
        # 是 Base Model
        return model_name, "Base", None

def plot_loss_curves():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found. Please run the training script first.")
        return

    print(f"Loading results from {RESULTS_FILE}...")
    with open(RESULTS_FILE, "r") as f:
        full_results = json.load(f)

    # 创建输出主目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 遍历每个函数
    for func_name, experiments in full_results.items():
        print(f"Processing function: {func_name}")
        func_dir = os.path.join(OUTPUT_DIR, func_name)
        if not os.path.exists(func_dir):
            os.makedirs(func_dir)

        # 整理数据结构: grouped_data[level][base_model_name] = { 'Base': ..., 'MLP_1x': ..., ... }
        grouped_data = {}

        for key, runs in experiments.items():
            # 解析 Key: e.g., "MLP_1x_for_Hybrid_L0" -> name="MLP_1x_for_Hybrid", level=0
            if "_L" not in key: continue
            name_part, level_part = key.rsplit("_L", 1)
            level = int(level_part)

            if level not in grouped_data: grouped_data[level] = {}

            # 解析模型类型
            base_model, m_type, _ = get_model_category(name_part)

            if "kan" in m_type.lower():
                continue
            
            if base_model not in grouped_data[level]:
                grouped_data[level][base_model] = {}

            # 聚合多个 Seed 的 Loss History
            # runs 是一个列表，包含多个 seed 的结果
            all_histories = []
            for r in runs:
                if "loss_history" in r:
                    all_histories.append(r["loss_history"])
            
            if not all_histories: continue

            # 计算平均曲线和标准差
            # 假设所有 runs 长度相同，取最短的以防万一
            min_len = min(len(h) for h in all_histories)
            all_histories = [h[:min_len] for h in all_histories]
            
            arr = np.array(all_histories)
            mean_curve = np.mean(arr, axis=0)
            std_curve = np.std(arr, axis=0)

            grouped_data[level][base_model][m_type] = {
                "mean": mean_curve,
                "std": std_curve,
                "params": runs[0]["params"] # 取第一个 run 的参数量作为参考
            }

        # 开始绘图：按 Level 遍历
        for level in sorted(grouped_data.keys()):
            level_data = grouped_data[level]
            
            # --- 图表 1: 核心模型概览 (Overview) ---
            plt.figure(figsize=(10, 6))
            for base_name in MODEL_STYLES.keys():
                if base_name in level_data and "Base" in level_data[base_name]:
                    data = level_data[base_name]["Base"]
                    style = MODEL_STYLES[base_name]
                    
                    x = np.arange(len(data["mean"]))
                    y = moving_average(data["mean"], SMOOTHING_WINDOW)
                    y_std = moving_average(data["std"], SMOOTHING_WINDOW)
                    
                    # 调整 x 长度匹配移动平均后
                    x = x[:len(y)]
                    
                    p = plt.semilogy(x, y, label=f"{style['label']}", color=style['color'], linewidth=2)
                    plt.fill_between(x, y-y_std, y+y_std, color=style['color'], alpha=0.1)

            # plt.title(f"Overview: Core Models Comparison\nFunction: {func_name} | Level: {level}")
            plt.xlabel("Epochs", fontsize=20)
            plt.ylabel("MSE Loss", fontsize=20)
            plt.tick_params(axis="both", which="major", labelsize=15)
            plt.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=2,
                fontsize=20,
                frameon=False,
                borderaxespad=0
            )
            plt.grid(True, which="both", ls="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(func_dir, f"L{level}_0_Overview.pdf"))
            plt.close()

            # --- 图表 2-5: 每个核心模型 vs Baselines ---
            for base_name in MODEL_STYLES.keys():
                if base_name not in level_data: continue
                
                cat_data = level_data[base_name]
                if "Base" not in cat_data: continue

                plt.figure(figsize=(10, 6))
                
                # 绘制 Base Model
                base_info = cat_data["Base"]
                base_style = MODEL_STYLES[base_name]
                x_base = np.arange(len(base_info["mean"]))
                y_base = moving_average(base_info["mean"], SMOOTHING_WINDOW)
                x_base = x_base[:len(y_base)]
                
                plt.semilogy(x_base, y_base, label=f"{base_style['label']}", 
                             color=base_style['color'], linewidth=2.5)

                # 绘制 Baselines (MLP 1x, 10x, KAN 1x, 10x)
                for bl_key, bl_style in BASELINE_STYLES.items():
                    if bl_key in cat_data:
                        bl_info = cat_data[bl_key]
                        x_bl = np.arange(len(bl_info["mean"]))
                        y_bl = moving_average(bl_info["mean"], SMOOTHING_WINDOW)
                        x_bl = x_bl[:len(y_bl)]
                        
                        plt.semilogy(x_bl, y_bl, 
                                     label=f"{bl_style['label']}", 
                                     color=bl_style['color'], 
                                     linestyle=bl_style['style'],
                                     alpha=0.8)

                # plt.title(f"Detail: {base_style['label']} vs Baselines\nFunction: {func_name} | Level: {level}")
                plt.xlabel("Epochs", fontsize=20)
                plt.ylabel("MSE Loss", fontsize=20)
                plt.tick_params(axis="both", which="major", labelsize=15)
                plt.legend(
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.02),
                    ncol=3,
                    fontsize=20,
                    frameon=False,
                    borderaxespad=0
                )
                plt.grid(True, which="both", ls="--", alpha=0.4)
                plt.tight_layout()
                plt.savefig(os.path.join(func_dir, f"L{level}_Detail_{base_name}.pdf"))
                plt.close()

    print(f"All plots saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    plot_loss_curves()
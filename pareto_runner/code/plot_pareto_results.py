import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_results(filename="comparison_results.json"):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return None
    with open(filename, "r") as f:
        return json.load(f)

def parse_model_type(key):
    """
    解析键名，返回 (模型大类, 是否为10x变体)
    例如: 
    "Hybrid_L0" -> ("Hybrid", False)
    "MLP_1x_for_Hybrid_L0" -> ("MLP", False)
    "KAN_10x_for_Hybrid_L0" -> ("KAN", True)
    """
    if "MLP" in key:
        return "MLP", "10x" in key
    elif "KAN" in key:
        return "KAN", "10x" in key
    elif "Hybrid" in key:
        return "HybridRationalNet", False
    elif "CFNet_Std" in key:
        return "CFNet_Standard", False
    elif "Boost" in key:
        return "EnsembleResCoFrNet", False
    elif "MoE" in key:
        return "MoE_Ensemble", False
    else:
        return "Unknown", False

def plot_pareto_frontiers(results):
    # 定义绘图样式
    # 连分式模型使用实线 + 鲜艳颜色
    # MLP/KAN 使用虚线/点线 + 灰色/深色系，作为背景参考
    styles = {
        "HybridRationalNet": {"color": "#e74c3c", "marker": "o", "label": "CFNN-Hybrid", "ls": "-"},
        "CFNet_Standard":    {"color": "#3498db", "marker": "s", "label": "CFNet Standard", "ls": "-"},
        "EnsembleResCoFrNet":{"color": "#2ecc71", "marker": "^", "label": "Boost CFNet", "ls": "-"},
        "MoE_Ensemble":      {"color": "#9b59b6", "marker": "D", "label": "CFNN-MoE", "ls": "-"},
        
        "MLP": {"color": "#34495e", "marker": "x", "label": "MLP", "ls": "--"}, # 1x
        "MLP_10x": {"color": "#34495e", "marker": "*", "label": "MLP (10x Params)", "ls": ":"}, 
        
        "KAN": {"color": "#e67e22", "marker": "v", "label": "KAN", "ls": "--"}, # 1x
        "KAN_10x": {"color": "#d35400", "marker": "P", "label": "KAN (10x Params)", "ls": ":"}
    }

    for func_name, func_data in results.items():
        print(f"Plotting results for function: {func_name}")
        plt.figure(figsize=(12, 10))
        
        # 容器：存储各类模型的 (params, loss) 点集
        points_map = {k: [] for k in styles.keys()} 
        # 加上 10x 的键
        points_map["MLP_10x"] = []
        points_map["KAN_10x"] = []

        # 1. 提取数据
        for exp_key, runs in func_data.items():
            # 计算平均参数量和平均 Loss
            if not runs: continue
            
            avg_params = np.mean([r['params'] for r in runs])
            avg_loss = np.mean([r['loss'] for r in runs])
            
            model_cat, is_10x = parse_model_type(exp_key)

            # print(model_cat)

            if is_10x or "KAN" in model_cat or 'Res' in model_cat or "CFNet" in model_cat:
                continue
            
            # 将数据放入对应容器
            target_key = model_cat
            if model_cat in ["MLP", "KAN"] and is_10x:
                target_key = f"{model_cat}_10x"
            
            if target_key in points_map:
                points_map[target_key].append((avg_params, avg_loss))

        # 2. 绘制曲线
        for model_name, points in points_map.items():
            if not points: continue
            print(model_name)

            # 按参数量排序，以便绘制连线
            points.sort(key=lambda x: x[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            
            style = styles.get(model_name, {})
            label = style.get("label", model_name)
            
            # 绘制线条和点
            plt.plot(xs, ys, 
                     color=style.get("color"), 
                     marker=style.get("marker"), 
                     linestyle=style.get("ls"), 
                     linewidth=5 if "10x" not in model_name else 1.5,
                     markersize=8,
                     label=label,
                     alpha=0.9)

        # 3. 图表美化
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of Parameters (Log Scale)', fontsize=40)
        plt.ylabel('MSE Loss (Log Scale)', fontsize=40)
        # --- 核心修改：设置刻度字体大小 ---
        # 方法 A：同时设置两个轴
        plt.tick_params(axis='both', which='major', labelsize=35)
        # plt.title(f'Pareto Frontier Analysis: {func_name}', fontsize=14, fontweight='bold')
        
        plt.grid(True, which="major", ls="-", alpha=0.4, color='gray')
        plt.grid(True, which="minor", ls=":", alpha=0.2, color='gray')
        
        # --- 核心修改部分 ---
        plt.legend(
            loc='lower center',          # 图例的锚点在图例自身的底部中央
            bbox_to_anchor=(0.5, 1.02), # 将锚点放置在坐标系 (0.5, 1.02) 的位置（即图正上方）
            ncol=3,                      # 强制设置为 3 列
            fontsize=30,
            frameon=False,                # 是否显示边框
            borderaxespad=0.
        )
        # -------------------
        plt.tight_layout()
        
        # 保存图片
        save_path = f"plot/pdf/pareto_{func_name}.pdf"
        plt.savefig(save_path, bbox_inches='tight')
        print(f" > Saved plot to {save_path}")
        # plt.show() # 如果在服务器运行，请注释此行
        plt.close()

if __name__ == "__main__":
    data = load_results()
    if data:
        plot_pareto_frontiers(data)
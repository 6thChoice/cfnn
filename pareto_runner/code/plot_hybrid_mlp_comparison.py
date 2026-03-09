import json
import matplotlib.pyplot as plt
import numpy as np
import os

# ==================== 参数化配置区域 ====================

# 文件路径配置
RESULTS_FILE = "comparison_results.json"
OUTPUT_FILE = "hybrid_mlp_comparison.pdf"

# 字体大小配置（参数化可调）
FONT_SIZES = {
    "legend": 20,      # 图例字体大小
    "axis_label": 20,  # 轴标签字体大小
    "tick": 15,        # 刻度字体大小
    "title": 18,       # 标题字体大小（如果需要）
}

# 字体族配置
FONT_FAMILY = 'serif'  # 可选: 'serif', 'sans-serif', 'monospace' 等

# 折线样式配置（参数化可调）
LINE_STYLES = {
    "Hybrid": {
        "color": "#d62728",  # 红色
        "linestyle": "-",
        "linewidth": 2.5,
        "label": "CFNN-Hybrid",
        "alpha": 1.0,
    },
    "MLP_1x": {
        "color": "#1f77b4",  # 蓝色
        "linestyle": "-",
        "linewidth": 2.0,
        "label": "MLP (1x)",
        "alpha": 0.8,
    },
    "MLP_10x": {
        "color": "#1f77b4",  # 蓝色（与 MLP_1x 相同）
        "linestyle": "--",   # 虚线区分
        "linewidth": 2.0,
        "label": "MLP (10x)",
        "alpha": 0.8,
    },
}


# 子图布局配置
SUBPLOT_CONFIG = {
    "nrows": 2,        # 行数
    "ncols": 2,        # 列数
    "figsize": (16, 12),  # 整体图尺寸 (宽, 高)
    "wspace": 0.25,    # 子图间水平间距
    "hspace": 0.35,    # 子图间垂直间距
}

# 图例配置
LEGEND_CONFIG = {
    "loc": "lower center",
    "bbox_to_anchor": (0.5, 0.97),  # 相对于图的位置
    "ncol": 3,         # 图例列数
    "frameon": False,  # 无边框
    "borderaxespad": 0,
}

# 网格配置
GRID_CONFIG = {
    "visible": True,
    "which": "both",
    "linestyle": "--",
    "alpha": 0.4,
}

# 平滑窗口配置
SMOOTHING_WINDOW = 1  # 设置大于1可进行滑动平均平滑

# 子图选择配置：指定函数和对应的Level
# 格式: (函数名, Level, 子图标题)
SUBPLOT_SELECTION = [
    ("Incomplete_Elliptic_Integral_E", 7, "Incomplete Elliptic E"),
    ("Incomplete_Elliptic_Integral_K", 1, "Incomplete Elliptic K"),
    ("Jacobian_Elliptic_sn", 2, "Jacobian sn"),
    ("Modified_Bessel_Iv", 7, "Modified Bessel Iv"),
]

# ==================== 函数定义 ====================

def moving_average(data, window_size):
    """计算移动平均"""
    if window_size <= 1:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def setup_fonts():
    """设置字体"""
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def get_model_category(model_name):
    """解析模型名称，返回 (BaseModelName, Type, Multiplier)"""
    if "_for_" in model_name:
        parts = model_name.split("_for_")
        baseline_type = parts[0]  # e.g., MLP_1x
        base_model = parts[1]     # e.g., Hybrid
        return base_model, baseline_type, None
    else:
        return model_name, "Base", None


def load_experiment_data(results_file):
    """加载实验结果数据"""
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found.")
        return None

    print(f"Loading results from {results_file}...")
    with open(results_file, "r") as f:
        return json.load(f)


def extract_loss_data(experiments, target_level, target_base_model="Hybrid"):
    """
    提取指定 level 和 base model 的损失数据
    返回: dict with keys 'Hybrid', 'MLP_1x', 'MLP_10x'
    """
    result = {
        "Hybrid": None,
        "MLP_1x": None,
        "MLP_10x": None,
    }

    for key, runs in experiments.items():
        if "_L" not in key:
            continue

        name_part, level_part = key.rsplit("_L", 1)
        level = int(level_part)

        if level != target_level:
            continue

        base_model, m_type, _ = get_model_category(name_part)

        if base_model != target_base_model:
            continue

        # 聚合多个 Seed 的 Loss History
        all_histories = []
        for r in runs:
            if "loss_history" in r:
                all_histories.append(r["loss_history"])

        if not all_histories:
            continue

        # 计算平均曲线和标准差
        min_len = min(len(h) for h in all_histories)
        all_histories = [h[:min_len] for h in all_histories]

        arr = np.array(all_histories)
        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)

        data = {
            "mean": mean_curve,
            "std": std_curve,
            "params": runs[0]["params"] if runs else 0
        }

        # 分类存储
        if m_type == "Base":
            result["Hybrid"] = data
        elif m_type == "MLP_1x":
            result["MLP_1x"] = data
        elif m_type == "MLP_10x":
            result["MLP_10x"] = data

    return result


def plot_single_subplot(ax, data_dict, title, show_xlabel=True, show_ylabel=True):
    """
    在指定的 axes 上绘制单个子图

    Args:
        ax: matplotlib axes 对象
        data_dict: 包含 Hybrid, MLP_1x, MLP_10x 数据的字典
        title: 子图标题
        show_xlabel: 是否显示 X 轴标签
        show_ylabel: 是否显示 Y 轴标签
    """
    # 绘制每种模型
    for model_key, style in LINE_STYLES.items():
        if model_key not in data_dict or data_dict[model_key] is None:
            continue

        data = data_dict[model_key]
        mean_curve = data["mean"]

        x = np.arange(len(mean_curve))
        y = moving_average(mean_curve, SMOOTHING_WINDOW)

        # 调整 x 长度匹配移动平均后
        x = x[:len(y)]

        # 绘制主线
        ax.semilogy(
            x, y,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
        )

    # 设置轴标签（仅在边缘子图显示）
    if show_xlabel:
        ax.set_xlabel("Epochs", fontsize=FONT_SIZES["axis_label"])
    if show_ylabel:
        ax.set_ylabel("MSE Loss", fontsize=FONT_SIZES["axis_label"])

    # 设置刻度字体大小
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZES["tick"])

    # 设置网格
    ax.grid(
        GRID_CONFIG["visible"],
        which=GRID_CONFIG["which"],
        linestyle=GRID_CONFIG["linestyle"],
        alpha=GRID_CONFIG["alpha"],
    )

    # 隐藏上方和右侧框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 设置标题（如果需要）
    if title:
        ax.set_title(title, fontsize=FONT_SIZES["title"])


def create_comparison_plot():
    """创建 CFNN-Hybrid vs MLP 对比图"""
    # 设置字体
    setup_fonts()

    # 加载数据
    full_results = load_experiment_data(RESULTS_FILE)
    if full_results is None:
        return

    # 创建图形和子图
    fig, axes = plt.subplots(
        SUBPLOT_CONFIG["nrows"],
        SUBPLOT_CONFIG["ncols"],
        figsize=SUBPLOT_CONFIG["figsize"],
    )

    # 调整子图间距
    plt.subplots_adjust(
        wspace=SUBPLOT_CONFIG["wspace"],
        hspace=SUBPLOT_CONFIG["hspace"],
    )

    # 展平 axes 以便迭代
    axes_flat = axes.flatten()

    # 绘制每个子图
    ncols = SUBPLOT_CONFIG["ncols"]
    nrows = SUBPLOT_CONFIG["nrows"]
    for idx, (func_name, level, subplot_title) in enumerate(SUBPLOT_SELECTION):
        ax = axes_flat[idx]

        # 判断是否为边缘子图
        row = idx // ncols  # 行索引 (0 或 1)
        col = idx % ncols   # 列索引 (0 或 1)
        show_ylabel = (col == 0)  # 第一列显示 Y 轴标签
        show_xlabel = (row == nrows - 1)  # 最后一行显示 X 轴标签

        print(f"Processing: {func_name} | Level {level}")

        if func_name not in full_results:
            print(f"  Warning: Function {func_name} not found in results")
            continue

        experiments = full_results[func_name]
        data_dict = extract_loss_data(experiments, level, target_base_model="Hybrid")

        # 检查是否有数据
        has_data = any(v is not None for v in data_dict.values())
        if not has_data:
            print(f"  Warning: No data found for {func_name} L{level}")
            continue

        plot_single_subplot(ax, data_dict, subplot_title, show_xlabel, show_ylabel)

    # 添加统一的图例在正上方
    # 创建代理句柄用于图例
    from matplotlib.lines import Line2D
    legend_handles = []
    for model_key, style in LINE_STYLES.items():
        handle = Line2D(
            [0], [0],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=style["label"],
        )
        legend_handles.append(handle)

    # 在图上方添加图例
    fig.legend(
        handles=legend_handles,
        loc=LEGEND_CONFIG["loc"],
        bbox_to_anchor=LEGEND_CONFIG["bbox_to_anchor"],
        ncol=LEGEND_CONFIG["ncol"],
        fontsize=FONT_SIZES["legend"],
        frameon=LEGEND_CONFIG["frameon"],
        borderaxespad=LEGEND_CONFIG["borderaxespad"],
    )

    # 调整布局，为顶部的图例留出空间
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存图片
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {OUTPUT_FILE}")

    plt.close()


if __name__ == "__main__":
    create_comparison_plot()

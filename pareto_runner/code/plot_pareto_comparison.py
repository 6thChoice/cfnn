import json
import matplotlib.pyplot as plt
import numpy as np
import os

# ==================== 参数化配置区域 ====================

# 文件路径配置
RESULTS_FILE = "comparison_results.json"
OUTPUT_FILE = "pareto_comparison.pdf"

# 字体大小配置（参数化可调）
FONT_SIZES = {
    "legend": 20,      # 图例字体大小
    "axis_label": 20,  # 轴标签字体大小
    "tick": 15,        # 刻度字体大小
    "title": 18,       # 子图标题字体大小
}

# 字体族配置
FONT_FAMILY = 'serif'  # 可选: 'serif', 'sans-serif', 'monospace' 等

# 折线样式配置（参数化可调）
LINE_STYLES = {
    "Hybrid": {
        "color": "#d62728",  # 红色
        "marker": "v",       # 下三角
        "linestyle": "-",
        "linewidth": 2.5,
        "markersize": 8,
        "label": "CFNN-Hybrid",
        "alpha": 0.9,
    },
    "MoE": {
        "color": "#8c564b",  # 棕色
        "marker": "p",       # 五边形
        "linestyle": "-",
        "linewidth": 2.5,
        "markersize": 8,
        "label": "CFNN-MoE",
        "alpha": 0.9,
    },
    "MLP": {
        "color": "#1f77b4",  # 蓝色
        "marker": "x",
        "linestyle": "--",
        "linewidth": 2.0,
        "markersize": 8,
        "label": "MLP",
        "alpha": 0.8,
    },
}

# 子图布局配置
SUBPLOT_CONFIG = {
    "nrows": 2,        # 行数
    "ncols": 2,        # 列数
    "figsize": (16, 14),  # 整体图尺寸 (宽, 高)
    "wspace": 0.25,    # 子图间水平间距
    "hspace": 0.30,    # 子图间垂直间距
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
    "which": "major",
    "linestyle": "-",
    "alpha": 0.4,
    "color": "gray",
}

# 次要网格配置
MINOR_GRID_CONFIG = {
    "visible": True,
    "which": "minor",
    "linestyle": ":",
    "alpha": 0.2,
    "color": "gray",
}

# 子图选择配置：指定函数和对应的标题
# 格式: (函数名, 子图标题)
SUBPLOT_SELECTION = [
    ("Bessel_Jv", "Bessel Jv"),
    ("Incomplete_Elliptic_Integral_E", "Incomplete Elliptic E"),
    ("Incomplete_Elliptic_Integral_K", "Incomplete Elliptic K"),
    ("Jacobian_Elliptic_sn", "Jacobian sn"),
]

# ==================== 函数定义 ====================

def setup_fonts():
    """设置字体"""
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_experiment_data(results_file):
    """加载实验结果数据"""
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found.")
        return None

    print(f"Loading results from {results_file}...")
    with open(results_file, "r") as f:
        return json.load(f)


def parse_model_type(key):
    """
    解析键名，返回模型类型
    例如:
    "Hybrid_L0" -> "Hybrid"
    "MoE_L1" -> "MoE"
    "MLP_1x_for_Hybrid_L0" -> "MLP"
    """
    if "MLP" in key:
        return "MLP"
    elif "Hybrid" in key:
        return "Hybrid"
    elif "MoE" in key:
        return "MoE"
    else:
        return None


def extract_pareto_data(experiments):
    """
    提取帕累托前沿数据
    返回: dict with keys 'Hybrid', 'MoE', 'MLP'
    每个值为 [(params, loss), ...] 列表

    注意：MLP 使用与 Hybrid 对应的 MLP_1x_for_Hybrid_L* 数据
    """
    result = {
        "Hybrid": [],
        "MoE": [],
        "MLP": [],
    }

    for exp_key, runs in experiments.items():
        if not runs:
            continue

        # 跳过 10x 模型
        if "10x" in exp_key:
            continue

        # 计算平均参数量和平均 Loss
        avg_params = np.mean([r['params'] for r in runs])
        avg_loss = np.mean([r['loss'] for r in runs])

        # Hybrid: 直接匹配 Hybrid_L*
        if exp_key.startswith("Hybrid_L"):
            result["Hybrid"].append((avg_params, avg_loss))
        # MoE: 直接匹配 MoE_L*
        elif exp_key.startswith("MoE_L"):
            result["MoE"].append((avg_params, avg_loss))
        # MLP: 只使用与 Hybrid 对应的 MLP_1x_for_Hybrid_L*
        elif exp_key.startswith("MLP_1x_for_Hybrid_L"):
            result["MLP"].append((avg_params, avg_loss))

    # 按参数量排序
    for model_type in result:
        result[model_type].sort(key=lambda x: x[0])

    return result


def plot_single_subplot(ax, data_dict, title, show_xlabel=True, show_ylabel=True):
    """
    在指定的 axes 上绘制单个子图

    Args:
        ax: matplotlib axes 对象
        data_dict: 包含各模型 (params, loss) 列表的字典
        title: 子图标题
        show_xlabel: 是否显示 X 轴标签
        show_ylabel: 是否显示 Y 轴标签
    """
    # 绘制每种模型
    for model_key, style in LINE_STYLES.items():
        if model_key not in data_dict or not data_dict[model_key]:
            continue

        points = data_dict[model_key]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        ax.plot(
            xs, ys,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            label=style["label"],
            alpha=style["alpha"],
        )

    # 设置对数坐标（仅Y轴）
    ax.set_yscale('log')

    # 设置横坐标为线性坐标，10个刻度
    ax.set_xscale('linear')
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))

    # 设置轴标签（仅在边缘子图显示）
    if show_xlabel:
        ax.set_xlabel("Number of Parameters", fontsize=FONT_SIZES["axis_label"])
    if show_ylabel:
        ax.set_ylabel("MSE Loss", fontsize=FONT_SIZES["axis_label"])

    # 设置刻度字体大小
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZES["tick"])

    # 设置网格（仅横向网格）
    ax.grid(
        GRID_CONFIG["visible"],
        which=GRID_CONFIG["which"],
        linestyle=GRID_CONFIG["linestyle"],
        alpha=GRID_CONFIG["alpha"],
        color=GRID_CONFIG["color"],
        axis='y',  # 仅Y轴方向
    )
    ax.grid(
        MINOR_GRID_CONFIG["visible"],
        which=MINOR_GRID_CONFIG["which"],
        linestyle=MINOR_GRID_CONFIG["linestyle"],
        alpha=MINOR_GRID_CONFIG["alpha"],
        color=MINOR_GRID_CONFIG["color"],
        axis='y',  # 仅Y轴方向
    )

    # 隐藏上方和右侧框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 设置标题
    if title:
        ax.set_title(title, fontsize=FONT_SIZES["title"])


def create_pareto_comparison_plot():
    """创建帕累托前沿对比图"""
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
    for idx, (func_name, subplot_title) in enumerate(SUBPLOT_SELECTION):
        ax = axes_flat[idx]

        # 判断是否为边缘子图
        row = idx // ncols
        col = idx % ncols
        show_ylabel = (col == 0)
        show_xlabel = (row == nrows - 1)

        print(f"Processing: {func_name}")

        if func_name not in full_results:
            print(f"  Warning: Function {func_name} not found in results")
            continue

        experiments = full_results[func_name]
        data_dict = extract_pareto_data(experiments)

        # 检查是否有数据
        has_data = any(len(v) > 0 for v in data_dict.values())
        if not has_data:
            print(f"  Warning: No data found for {func_name}")
            continue

        plot_single_subplot(ax, data_dict, subplot_title, show_xlabel, show_ylabel)

    # 添加统一的图例在正上方
    from matplotlib.lines import Line2D
    legend_handles = []
    for model_key, style in LINE_STYLES.items():
        handle = Line2D(
            [0], [0],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
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
    create_pareto_comparison_plot()

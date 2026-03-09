#!/usr/bin/env python3
"""
帕累托前沿绘图脚本
绘制不同噪声水平下的参数量-性能帕累托前沿图
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
runner_base = Path(__file__).resolve().parents[4]
results_dir = runner_base / "results"

# ==================== 配置参数 ====================

# 输入数据路径
DATA_PATH = str(results_dir / "exp1_pareto_merged_mlp_fixed" / "merged_summary.csv")

# 输出图片路径
OUTPUT_DIR = str(results_dir / "exp1_pareto_merged_mlp_fixed")
OUTPUT_FILENAME = 'pareto_frontier_2x2.pdf'

# 要绘制的噪声水平
NOISE_RATIOS = [0.2, 0.4, 0.6, 0.8]

# ==================== 图形样式配置 ====================

# 图形尺寸 (英寸)
FIGURE_SIZE = (10, 8)

# 子图布局参数
SUBPLOT_NROWS = 2
SUBPLOT_NCOLS = 2
SUBPLOT_ADJUST_LEFT = 0.1
SUBPLOT_ADJUST_RIGHT = 0.95
SUBPLOT_ADJUST_BOTTOM = 0.1
SUBPLOT_ADJUST_TOP = 0.95
SUBPLOT_ADJUST_WSPACE = 0.3
SUBPLOT_ADJUST_HSPACE = 0.35

# 字体大小配置
FONT_SIZE_TITLE = 14          # 子图标题字体大小
FONT_SIZE_AXIS_LABEL = 12     # 轴标签字体大小
FONT_SIZE_TICK = 10           # 刻度字体大小
FONT_SIZE_LEGEND = 9          # 图例字体大小

# 线条样式配置
LINE_WIDTH = 2.0              # 线条粗细
LINE_STYLE = '-'              # 线条样式 ('-', '--', '-.', ':')
MARKER_SIZE = 6               # 标记大小
MARKER_EDGE_WIDTH = 1.0       # 标记边缘粗细

# 颜色配置 (模型名称 -> 颜色)
COLORS = {
    'MLP': '#1f77b4',         # 蓝色
    'KAN': '#ff7f0e',         # 橙色
    'Standard': '#2ca02c',    # 绿色
    'Hybrid': '#d62728',      # 红色
    'Boost': '#9467bd',       # 紫色
    'MoE': '#8c564b'          # 棕色
}

# 标记样式配置 (模型名称 -> 标记)
MARKERS = {
    'MLP': 'o',               # 圆形
    'KAN': 's',               # 方形
    'Standard': '^',          # 三角形上
    'Hybrid': 'v',            # 三角形下
    'Boost': 'D',             # 菱形
    'MoE': 'p'                # 五边形
}

# 轴配置
X_AXIS_LABEL = 'Number of Parameters'
Y_AXIS_LABEL = 'Test MSE'
Y_AXIS_SCALE = 'log'          # Y轴刻度类型 ('linear' 或 'log')

# 网格线配置
GRID_VISIBLE = True
GRID_ALPHA = 0.3
GRID_LINESTYLE = '--'
GRID_LINEWIDTH = 0.5

# 图例配置（全局图例，放在图上方，两行三列）
LEGEND_LOC = 'upper center'
LEGEND_FRAMEON = False  # 去掉边框
LEGEND_FRAMEALPHA = 0.9
LEGEND_EDGECOLOR = 'gray'
LEGEND_NCOL = 3  # 3个模型一列，共两行
LEGEND_BBOX_TO_ANCHOR = (0.5, 0.95)  # 放在图上方，预留两行空间
LEGEND_FONTSIZE = 14  # 放大图例字体

# 模型名称映射（用于图例显示）
MODEL_NAME_MAPPING = {
    'MLP': 'MLP',
    'KAN': 'KAN',
    'Standard': 'CFNN',
    'Hybrid': 'CFNN-Hybrid',
    'Boost': 'CFNN-Boost',
    'MoE': 'CFNN-MoE'
}

# ==================== 绘图函数 ====================

def load_data(data_path):
    """加载实验数据"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    df = pd.read_csv(data_path)
    return df


def setup_plot_style():
    """设置绘图样式"""
    rcParams['font.size'] = FONT_SIZE_TICK
    rcParams['axes.titlesize'] = FONT_SIZE_TITLE
    rcParams['axes.labelsize'] = FONT_SIZE_AXIS_LABEL
    rcParams['xtick.labelsize'] = FONT_SIZE_TICK
    rcParams['ytick.labelsize'] = FONT_SIZE_TICK
    rcParams['legend.fontsize'] = FONT_SIZE_LEGEND
    rcParams['lines.linewidth'] = LINE_WIDTH
    rcParams['lines.markersize'] = MARKER_SIZE
    rcParams['grid.alpha'] = GRID_ALPHA
    rcParams['grid.linestyle'] = GRID_LINESTYLE
    rcParams['grid.linewidth'] = GRID_LINEWIDTH


def plot_pareto_frontier(df, noise_ratios, output_path):
    """
    绘制帕累托前沿图

    Args:
        df: DataFrame containing experiment results
        noise_ratios: List of noise ratios to plot
        output_path: Path to save the figure
    """
    # 创建图形
    fig, axes = plt.subplots(SUBPLOT_NROWS, SUBPLOT_NCOLS, figsize=FIGURE_SIZE)
    axes = axes.flatten()

    # 获取所有模型列表
    models = ['MLP', 'KAN', 'Standard', 'Hybrid', 'Boost', 'MoE']

    # 遍历每个噪声比例
    for idx, noise_ratio in enumerate(noise_ratios):
        ax = axes[idx]

        # 筛选当前噪声比例的数据
        noise_data = df[df['noise_ratio'] == noise_ratio]

        # 遍历每个模型
        for model in models:
            model_data = noise_data[noise_data['model'] == model]

            if len(model_data) == 0:
                continue

            # 按实际参数量排序
            model_data = model_data.sort_values('actual_params')

            # 绘制线条（使用映射后的名称作为图例标签）
            display_name = MODEL_NAME_MAPPING.get(model, model)
            ax.plot(
                model_data['actual_params'],
                model_data['test_mse'],
                label=display_name,
                color=COLORS.get(model, 'gray'),
                linestyle=LINE_STYLE,
                linewidth=LINE_WIDTH,
                marker=MARKERS.get(model, 'o'),
                markersize=MARKER_SIZE,
                markeredgecolor='white',
                markeredgewidth=MARKER_EDGE_WIDTH,
                alpha=0.9
            )

        # 设置子图标题 (噪声比例)
        ax.set_title(
            f'{int(noise_ratio * 100)}% Noise',
            fontsize=FONT_SIZE_TITLE,
            fontweight='normal'
        )

        # 设置轴标签
        if idx >= 2:  # 底部子图显示X轴标签
            ax.set_xlabel(X_AXIS_LABEL, fontsize=FONT_SIZE_AXIS_LABEL)
        if idx % 2 == 0:  # 左侧子图显示Y轴标签
            ax.set_ylabel(Y_AXIS_LABEL, fontsize=FONT_SIZE_AXIS_LABEL)

        # 设置刻度
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)

        # 设置Y轴为对数刻度
        if Y_AXIS_SCALE == 'log':
            ax.set_yscale('log')

        # 添加网格线
        if GRID_VISIBLE:
            ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH)

        # 去除上方和右侧框线
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 不添加子图图例，使用全局图例

    # 添加全局图例（位于图上方，两行三列）
    handles, labels = axes[0].get_legend_handles_labels()
    # 去重并保持顺序
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    fig.legend(
        unique_handles,
        unique_labels,
        loc=LEGEND_LOC,
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        ncol=LEGEND_NCOL,
        frameon=LEGEND_FRAMEON,
        framealpha=LEGEND_FRAMEALPHA,
        edgecolor=LEGEND_EDGECOLOR,
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.5,
        handletextpad=0.5,
        columnspacing=1.5
    )

    # 调整子图布局（给上方图例留出空间）
    plt.subplots_adjust(
        left=SUBPLOT_ADJUST_LEFT,
        right=SUBPLOT_ADJUST_RIGHT,
        bottom=SUBPLOT_ADJUST_BOTTOM,
        top=0.82,  # 留出空间给两行图例
        wspace=SUBPLOT_ADJUST_WSPACE,
        hspace=SUBPLOT_ADJUST_HSPACE
    )

    # 保存图片
    output_full_path = os.path.join(output_path, OUTPUT_FILENAME)
    plt.savefig(
        output_full_path,
        format='pdf',
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.1
    )
    plt.close()

    print(f"帕累托前沿图已保存: {output_full_path}")
    return output_full_path


def main():
    """主函数"""
    print("=" * 80)
    print("帕累托前沿绘图脚本")
    print("=" * 80)

    # 检查输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"创建输出目录: {OUTPUT_DIR}")

    # 加载数据
    print(f"\n加载数据: {DATA_PATH}")
    try:
        df = load_data(DATA_PATH)
        print(f"成功加载 {len(df)} 条记录")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)

    # 显示数据概览
    print(f"\n数据概览:")
    print(f"  - 噪声比例: {sorted(df['noise_ratio'].unique())}")
    print(f"  - 模型: {sorted(df['model'].unique())}")
    print(f"  - 参数量范围: {df['actual_params'].min():.0f} - {df['actual_params'].max():.0f}")
    print(f"  - MSE范围: {df['test_mse'].min():.6f} - {df['test_mse'].max():.6f}")

    # 设置绘图样式
    print("\n设置绘图样式...")
    setup_plot_style()

    # 绘制帕累托前沿
    print(f"\n绘制帕累托前沿图 (噪声比例: {NOISE_RATIOS})...")
    output_path = plot_pareto_frontier(df, NOISE_RATIOS, OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("绘图完成!")
    print(f"输出文件: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
"""
绘制 CFNN 与三个变体的参数扩展性比较图
横坐标：参数量
纵坐标：测试损失 (Test Loss)
所有样式参数可配置调整
"""
import os
import warnings
import sys
from io import StringIO

# 禁用 matplotlib 的 Unicode 减号警告
os.environ['MPLCONFIGDIR'] = os.path.expanduser('~/.config/matplotlib')

# 捕获 matplotlib 的字体警告到空
class WarningCapture:
    def __init__(self):
        self.original_stderr = sys.stderr

    def __enter__(self):
        sys.stderr = StringIO()
        return self

    def __exit__(self, *args):
        sys.stderr = self.original_stderr

import json
import numpy as np
import matplotlib
# 配置 matplotlib 使用 ASCII 减号（避免 Unicode 警告）
matplotlib.use('Agg')  # 使用非交互式后端
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.formatter.use_mathtext'] = False

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 忽略所有 matplotlib 字体相关警告
warnings.filterwarnings('ignore', message='.*Font.*')
warnings.filterwarnings('ignore', message='.*glyph.*')

plt.rcParams['axes.unicode_minus'] = False

# 在导入时抑制警告
with WarningCapture():
    pass

# ==========================================
# 样式配置参数（可调整）
# ==========================================
class PlotConfig:
    """绘图样式配置"""

    # --- 图形尺寸 ---
    FIGURE_WIDTH = 12
    FIGURE_HEIGHT = 8
    DPI = 300

    # --- 字号设置 ---
    TITLE_FONT_SIZE = 18
    LABEL_FONT_SIZE = 20
    LEGEND_FONT_SIZE = 20
    ANNOTATION_FONT_SIZE = 20

    # --- 线条设置 ---
    LINE_WIDTH = 2.5
    LINE_ALPHA = 0.9
    MARKER_SIZE = 8

    # --- 颜色设置 ---
    COLOR_CFNN = '#444444'      # 灰色 - 基准模型
    COLOR_BOOST = '#2ca02c'     # 绿色
    COLOR_MOE = '#9467bd'       # 紫色
    COLOR_HYBRID = '#d62728'    # 红色

    # --- 线型和标记 ---
    CFNN_LINE_STYLE = '--'      # CFNN 虚线
    CFNN_MARKER = 'o'           # CFNN 圆圈标记

    VARIANT_LINE_STYLE = '-'    # 变体实线
    BOOST_MARKER = 's'          # Boost 方块标记
    MOE_MARKER = '^'            # MoE 三角标记
    HYBRID_MARKER = 'D'         # Hybrid 菱形标记

    # --- 坐标轴设置 ---
    TICK_SIZE = 12
    X_LABEL = 'Continued Fraction Depth'
    Y_LABEL = 'Test Loss (MSE)'

    # --- 网格设置 ---
    GRID_ALPHA = 0.3
    GRID_LINE_STYLE = '-'
    GRID_LINE_WIDTH = 0.5

    # --- Y 轴显示范围限制 ---
    # 限制 Y 轴最大值，曲线可以超出此范围继续绘制
    # 设为 None 表示自动调整
    Y_AXIS_MAX_LIMIT = 100  # Y 轴最大显示值

    # --- 超出阈值标注样式 ---
    EXCEEDED_ANNOTATE = True          # 是否标注超出 Y 轴限制的点
    EXCEEDED_COLOR = 'red'            # 标注颜色
    EXCEEDED_MARKER = 'v'             # 标记样式（向下箭头）
    EXCEEDED_MARKER_SIZE = 10         # 标记大小
    EXCEEDED_FONT_SIZE = 10           # 标注文字大小
    EXCEEDED_FORMAT = '{:.3f}'        # 数值格式


# ==========================================
# 数据加载函数
# ==========================================
def calculate_n_parameters(model_type: str, depth: int) -> int:
    """
    根据模型类型和深度计算参数量
    （用于修复实验数据中缺失的参数量）
    """
    if model_type == 'CFNN':
        # CFNN: depth * 9 - 1 (approximately)
        return depth * 9 - 1
    elif model_type == 'CFNN-Boost':
        # CFNN-Boost: 每个单元约为 55 参数（shallow_depth=2, poly_degree=3）
        return depth * 55
    elif model_type == 'CFNN-MoE':
        # CFNN-MoE: 基础约 55 + 每个专家约 96
        return 55 + (depth - 1) * 96
    elif model_type == 'CFNN-Hybrid':
        # CFNN-Hybrid: depth * 16 + 4
        return depth * 16 + 4
    return depth * 50  # 默认估算


def load_scalability_data() -> Dict[str, List[Dict]]:
    """
    加载参数扩展性实验数据

    Returns:
        字典，key 为模型名称，value 为该模型的实验数据列表
        每个实验数据包含: depth, n_parameters, test_loss, seed
    """
    data = {
        'CFNN': [],
        'CFNN-Boost': [],
        'CFNN-MoE': [],
        'CFNN-Hybrid': []
    }

    # 加载扩展性实验数据
    scalability_path = Path('results/scalability/scalability_test_results.json')
    if scalability_path.exists():
        with open(scalability_path, 'r') as f:
            scal_data = json.load(f)

        for exp in scal_data.get('experiments', []):
            model_type = exp.get('model_type', exp.get('model_type_raw', ''))
            depth = exp.get('depth', 0)
            n_params = exp.get('n_parameters', 0)

            # 简化模型名称
            if 'CFNN' in model_type and 'Boost' not in model_type and 'MoE' not in model_type and 'Hybrid' not in model_type:
                key = 'CFNN'
                if n_params == 0:
                    n_params = calculate_n_parameters('CFNN', depth)
            elif 'Boost' in model_type:
                key = 'CFNN-Boost'  # 先加载，后面会被改进数据覆盖
                if n_params == 0:
                    n_params = calculate_n_parameters('CFNN-Boost', depth)
            elif 'MoE' in model_type:
                key = 'CFNN-MoE'
                if n_params == 0:
                    n_params = calculate_n_parameters('CFNN-MoE', depth)
            elif 'Hybrid' in model_type:
                key = 'CFNN-Hybrid'
                if n_params == 0:
                    n_params = calculate_n_parameters('CFNN-Hybrid', depth)
            else:
                continue

            data[key].append({
                'depth': depth,
                'n_parameters': n_params,
                'test_loss': exp.get('test_loss', float('inf')),
                'seed': exp.get('seed', 0),
                'training_successful': exp.get('training_successful', True)
            })

    # 加载改进的 CFNN-Boost 数据（覆盖原有数据）
    boost_improved_path = Path('results/boost_improved/boost_improved_results.json')
    if boost_improved_path.exists():
        print(f"使用改进的 CFNN-Boost 数据: {boost_improved_path}")
        with open(boost_improved_path, 'r') as f:
            boost_data = json.load(f)

        # 清空原有的 Boost 数据
        data['CFNN-Boost'] = []

        for exp in boost_data.get('experiments', []):
            depth = exp.get('depth', 0)
            # 改进实验的参数量计算
            n_params = calculate_n_parameters('CFNN-Boost', depth)

            data['CFNN-Boost'].append({
                'depth': depth,
                'n_parameters': n_params,
                'test_loss': exp.get('test_loss', float('inf')),
                'seed': exp.get('seed', 0),
                'training_successful': exp.get('training_successful', True)
            })

    return data


# ==========================================
# 数据处理函数
# ==========================================
def aggregate_by_depth(experiments: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    按深度聚合实验数据，计算均值和标准差

    Args:
        experiments: 实验数据列表

    Returns:
        (深度数组, 平均测试损失, 标准差)
    """
    # 按深度分组
    depth_groups = {}
    for exp in experiments:
        if not exp.get('training_successful', True):
            continue
        depth = exp['depth']
        loss = exp['test_loss']
        if depth not in depth_groups:
            depth_groups[depth] = []
        depth_groups[depth].append(loss)

    if not depth_groups:
        return np.array([]), np.array([]), np.array([])

    # 排序并计算统计量
    sorted_depths = sorted(depth_groups.keys())
    mean_losses = []
    std_losses = []

    for depth in sorted_depths:
        losses = depth_groups[depth]
        mean_losses.append(np.mean(losses))
        std_losses.append(np.std(losses))

    return np.array(sorted_depths), np.array(mean_losses), np.array(std_losses)


# ==========================================
# 绘图函数
# ==========================================
def plot_scalability_comparison(
    data: Dict[str, List[Dict]],
    config: PlotConfig = PlotConfig,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制参数扩展性比较图

    Args:
        data: 模型数据字典
        config: 绘图配置
        save_path: 保存路径
    """
    # 设置字体（使用 ASCII 减号避免 Unicode 警告）
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.unicode_minus'] = False  # 使用 ASCII 减号

    # 创建图形
    fig, ax = plt.subplots(figsize=(config.FIGURE_WIDTH, config.FIGURE_HEIGHT), dpi=config.DPI)

    # 模型配置
    model_configs = {
        'CFNN': {
            'color': config.COLOR_CFNN,
            'label': 'CFNN',
            'linestyle': config.CFNN_LINE_STYLE,
            'marker': config.CFNN_MARKER,
            'linewidth': config.LINE_WIDTH,
            'alpha': config.CFNN_ALPHA if hasattr(config, 'CFNN_ALPHA') else config.LINE_ALPHA,
        },
        'CFNN-Boost': {
            'color': config.COLOR_BOOST,
            'label': 'CFNN-Boost',
            'linestyle': config.VARIANT_LINE_STYLE,
            'marker': config.BOOST_MARKER,
            'linewidth': config.LINE_WIDTH,
            'alpha': config.LINE_ALPHA,
        },
        'CFNN-MoE': {
            'color': config.COLOR_MOE,
            'label': 'CFNN-MoE',
            'linestyle': config.VARIANT_LINE_STYLE,
            'marker': config.MOE_MARKER,
            'linewidth': config.LINE_WIDTH,
            'alpha': config.LINE_ALPHA,
        },
        'CFNN-Hybrid': {
            'color': config.COLOR_HYBRID,
            'label': 'CFNN-Hybrid',
            'linestyle': config.VARIANT_LINE_STYLE,
            'marker': config.HYBRID_MARKER,
            'linewidth': config.LINE_WIDTH,
            'alpha': config.LINE_ALPHA,
        }
    }

    # 绘制顺序
    draw_order = ['CFNN', 'CFNN-Hybrid', 'CFNN-MoE', 'CFNN-Boost']

    # 绘制每条曲线
    for model_name in draw_order:
        experiments = data.get(model_name, [])
        if not experiments:
            continue

        depths, mean_loss, std_loss = aggregate_by_depth(experiments)

        if len(depths) == 0:
            continue

        cfg = model_configs.get(model_name, {})

        # 绘制折线
        ax.plot(
            depths,
            mean_loss,
            color=cfg['color'],
            linewidth=cfg['linewidth'],
            linestyle=cfg['linestyle'],
            alpha=cfg['alpha'],
            label=cfg['label'],
            marker=cfg['marker'],
            markersize=config.MARKER_SIZE,
            markeredgecolor='white',
            markeredgewidth=0.5
        )

        # 绘制误差带
        ax.fill_between(
            depths,
            np.maximum(0, mean_loss - std_loss),
            mean_loss + std_loss,
            color=cfg['color'],
            alpha=0.15,
            linewidth=0
        )

        # 标注超出 Y 轴限制的点
        if config.EXCEEDED_ANNOTATE:
            exceeded_mask = mean_loss > config.Y_AXIS_MAX_LIMIT

            if np.any(exceeded_mask):
                # 标注所有超出的点
                for idx in np.where(exceeded_mask)[0]:
                    depth_val = depths[idx]
                    loss_val = mean_loss[idx]

                    # 绘制标记
                    ax.scatter(
                        [depth_val],
                        [config.Y_AXIS_MAX_LIMIT * 0.95],
                        marker=config.EXCEEDED_MARKER,
                        s=config.EXCEEDED_MARKER_SIZE ** 2,
                        color=config.EXCEEDED_COLOR,
                        zorder=5,
                        edgecolors='black',
                        linewidths=0.5
                    )

                    # 标注数值
                    ax.annotate(
                        f'{config.EXCEEDED_FORMAT.format(loss_val)}',
                        xy=(depth_val, config.Y_AXIS_MAX_LIMIT * 0.95),
                        xytext=(depth_val, config.Y_AXIS_MAX_LIMIT * 0.85),
                        fontsize=config.EXCEEDED_FONT_SIZE,
                        color=config.EXCEEDED_COLOR,
                        fontweight='bold',
                        ha='center',
                        va='top',
                        arrowprops=dict(arrowstyle='->', color=config.EXCEEDED_COLOR, lw=1)
                    )

    # 设置坐标轴
    ax.set_xlabel(config.X_LABEL, fontsize=config.LABEL_FONT_SIZE, fontweight='bold')
    ax.set_ylabel(config.Y_LABEL, fontsize=config.LABEL_FONT_SIZE, fontweight='bold')
    # ax.set_title(
    #     'Parameter Scalability',
    #     fontsize=config.TITLE_FONT_SIZE,
    #     fontweight='bold'
    # )
    ax.tick_params(labelsize=config.TICK_SIZE)

    # 设置 Y 轴范围（对数刻度）
    # 使用 LogFormatter 而非 LogFormatterMathtext 避免 Unicode 负号问题
    from matplotlib.ticker import LogFormatter, LogLocator
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-5, top=config.Y_AXIS_MAX_LIMIT)

    # 使用不依赖 mathtext 的 LogFormatter
    formatter = LogFormatter()
    formatter.format_str = '%.0e'  # 科学计数法格式，使用 ASCII 减号
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(LogLocator())

    # 网格
    ax.grid(True, alpha=config.GRID_ALPHA, linestyle=config.GRID_LINE_STYLE, linewidth=config.GRID_LINE_WIDTH)
    ax.grid(True, alpha=config.GRID_ALPHA, which='minor', linestyle=config.GRID_LINE_STYLE, linewidth=config.GRID_LINE_WIDTH * 0.5)

    # 图例
    ax.legend(loc='upper right', fontsize=config.LEGEND_FONT_SIZE, framealpha=0.95)

    plt.tight_layout()

    # 保存图形
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存 PNG
        fig.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")

        # 保存 PDF
        pdf_path = save_path.with_suffix('.pdf')
        fig.savefig(pdf_path, dpi=config.DPI, bbox_inches='tight', format='pdf')
        print(f"PDF 已保存至: {pdf_path}")

    return fig


# ==========================================
# 主函数
# ==========================================
def main():
    """主函数"""

    print("=" * 70)
    print("CFNN 与变体参数扩展性比较图")
    print("横坐标：连分式深度，纵坐标：测试损失")
    print("=" * 70)

    # 加载数据
    data = load_scalability_data()

    print("\n数据加载统计:")
    for model, experiments in data.items():
        if experiments:
            depths = [e['depth'] for e in experiments]
            print(f"  {model}: {len(experiments)} 条记录, 深度范围 {min(depths)} - {max(depths)}")

    # 生成图形
    print("\n生成参数扩展性比较图...")
    with WarningCapture():
        fig = plot_scalability_comparison(
            data,
            config=PlotConfig,
            save_path='results/plots/scalability_comparison.png'
        )

    plt.close(fig)

    print("\n" + "=" * 70)
    print("绘图完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

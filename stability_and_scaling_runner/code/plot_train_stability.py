"""
绘制 CFNN 与三个变体的训练损失下降稳定性对比图
使用双面板设计：
- 左图：训练损失曲线对比
- 右图：梯度标准差对比（来自梯度稳定性实验）
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
matplotlib.use('Agg')
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore', message='.*Font.*')
warnings.filterwarnings('ignore', message='.*glyph.*')

# ==========================================
# 样式配置参数（可调整）
# ==========================================
class PlotConfig:
    """绘图样式配置"""

    # --- 图形尺寸 ---
    FIGURE_WIDTH = 16
    FIGURE_HEIGHT = 7
    DPI = 300

    # --- 字号设置 ---
    TITLE_FONT_SIZE = 32
    LABEL_FONT_SIZE = 20
    LEGEND_FONT_SIZE = 20
    ANNOTATION_FONT_SIZE = 20
    BAR_LABEL_FONT_SIZE = 20

    # --- 损失曲线线条设置 ---
    LINE_WIDTH_VARIANT = 2.5
    LINE_WIDTH_CFNN = 2.5
    LINE_ALPHA = 0.9
    CFNN_ALPHA = 0.7

    # --- 颜色设置 ---
    COLOR_CFNN = '#444444'      # 灰色 - 不稳定
    COLOR_BOOST = '#2ca02c'     # 绿色 - 稳定
    COLOR_MOE = '#9467bd'       # 紫色 - 稳定
    COLOR_HYBRID = '#d62728'    # 红色 - 稳定

    # --- 线型设置 ---
    CFNN_LINE_STYLE = '--'      # CFNN 虚线
    VARIANT_LINE_STYLE = '-'    # 变体实线

    # --- 坐标轴设置 ---
    TICK_SIZE = 12

    # --- 网格设置 ---
    GRID_ALPHA = 0.3
    GRID_LINE_STYLE = '-'
    GRID_LINE_WIDTH = 0.5

    # --- 柱状图设置 ---
    BAR_WIDTH = 0.6
    BAR_ALPHA = 0.85
    CFNN_BAR_HATCH = '////'     # CFNN 柱状图斜线填充

    # --- 数据平滑 ---
    SMOOTHING_WINDOW = 20

    # --- Y 轴显示范围限制 ---
    # 限制 Y 轴最大值，曲线可以超出此范围继续绘制
    # 设为 None 表示自动调整
    Y_AXIS_MAX_LIMIT = 100000000  # Y 轴最大显示值，超出部分继续向上延伸

    # --- 最大损失标注样式 ---
    # 标注每条曲线的最大损失位置和数值
    MAX_LOSS_ANNOTATE = True          # 是否标注最大损失
    MAX_LOSS_COLOR = 'red'            # 标注颜色
    MAX_LOSS_MARKER = 'v'             # 最大值标记样式（向下箭头）
    MAX_LOFF_MARKER_SIZE = 10         # 标记大小
    MAX_LOSS_FONT_SIZE = 20           # 标注文字大小
    MAX_LOSS_FORMAT = '{:.1f}'        # 数值格式
    MAX_LOSS_OFFSET_Y = 2             # 标注文字垂直偏移


# ==========================================
# 数据加载函数
# ==========================================
def load_all_data() -> Tuple[Dict[str, List[List[float]]], Dict[str, List[float]]]:
    """
    加载所有数据

    Returns:
        (loss_data, grad_std_data)
    """
    # 加载损失数据
    loss_data = {
        'CFNN': [],
        'CFNN-Boost': [],
        'CFNN-MoE': [],
        'CFNN-Hybrid': []
    }

    boost_path = Path('results/boost_improved/boost_improved_results.json')
    if boost_path.exists():
        with open(boost_path, 'r') as f:
            boost_data = json.load(f)
        for exp in boost_data.get('experiments', []):
            if 'loss_history' in exp and exp.get('training_successful', True):
                loss_data['CFNN-Boost'].append(exp['loss_history'])

    scalability_path = Path('results/scalability/scalability_test_results.json')
    if scalability_path.exists():
        with open(scalability_path, 'r') as f:
            scal_data = json.load(f)
        for exp in scal_data.get('experiments', []):
            model_type = exp.get('model_type', exp.get('model_type_raw', ''))
            if 'loss_history' in exp and exp.get('training_successful', True):
                if 'MoE' in model_type:
                    loss_data['CFNN-MoE'].append(exp['loss_history'])
                elif 'Hybrid' in model_type:
                    loss_data['CFNN-Hybrid'].append(exp['loss_history'])
                elif 'CFNN' in model_type and 'Boost' not in model_type and 'MoE' not in model_type and 'Hybrid' not in model_type:
                    loss_data['CFNN'].append(exp['loss_history'])

    # 加载梯度标准差数据
    grad_std_data = {
        'CFNN': [],
        'CFNN-Boost': [],
        'CFNN-MoE': [],
        'CFNN-Hybrid': []
    }

    grad_path = Path('results/gradient_stability/gradient_test_results.json')
    if grad_path.exists():
        with open(grad_path, 'r') as f:
            g_data = json.load(f)
        for exp in g_data.get('experiments', []):
            model_type = exp.get('model_type', exp.get('model_type_raw', ''))
            grad_stats = exp.get('grad_stats', {})
            std_grad = grad_stats.get('std_grad_norm', 0)

            # 简化模型名称
            if 'CFNN' in model_type and 'Boost' not in model_type and 'MoE' not in model_type and 'Hybrid' not in model_type:
                key = 'CFNN'
            elif 'Boost' in model_type:
                key = 'CFNN-Boost'
            elif 'MoE' in model_type:
                key = 'CFNN-MoE'
            elif 'Hybrid' in model_type:
                key = 'CFNN-Hybrid'
            else:
                continue

            if std_grad > 0:  # 只记录有效值
                grad_std_data[key].append(std_grad)

    return loss_data, grad_std_data


# ==========================================
# 数据处理函数
# ==========================================
def smooth_curve(data: List[float], window: int = 20) -> List[float]:
    """使用移动平均平滑数据曲线"""
    if window <= 1:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    return smoothed


def normalize_curves(histories: List[List[float]], max_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将多条损失曲线归一化到相同长度并计算均值和标准差

    Args:
        histories: 损失历史列表
        max_length: 目标长度

    Returns:
        epochs: epoch 序列
        mean_loss: 平均损失
        std_loss: 损失标准差
    """
    if not histories:
        return np.array([]), np.array([]), np.array([])

    if max_length is None:
        max_length = max(len(h) for h in histories)

    normalized = []
    for hist in histories:
        if len(hist) < max_length:
            x_old = np.linspace(0, 1, len(hist))
            x_new = np.linspace(0, 1, max_length)
            interpolated = np.interp(x_new, x_old, hist)
            normalized.append(interpolated)
        else:
            normalized.append(hist[:max_length])

    normalized = np.array(normalized)
    epochs = np.arange(1, max_length + 1)
    mean_loss = np.mean(normalized, axis=0)
    std_loss = np.std(normalized, axis=0)

    return epochs, mean_loss, std_loss


# ==========================================
# 绘图函数
# ==========================================
def plot_training_stability(
    loss_data: Dict[str, List[List[float]]],
    grad_std_data: Dict[str, List[float]],
    config: PlotConfig = PlotConfig,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制训练损失下降稳定性对比图
    左图：损失曲线对比
    右图：梯度标准差柱状图（稳定性指标）
    """
    # 设置字体（使用 ASCII 减号避免 Unicode 警告）
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.unicode_minus'] = False  # 使用 ASCII 减号

    # 创建双面板图形
    fig = plt.figure(figsize=(config.FIGURE_WIDTH, config.FIGURE_HEIGHT), dpi=config.DPI)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.25)

    ax_loss = fig.add_subplot(gs[0])
    ax_stab = fig.add_subplot(gs[1])

    # 模型配置
    model_configs = {
        'CFNN': {
            'color': config.COLOR_CFNN,
            'label': 'CFNN',
            'linestyle': config.CFNN_LINE_STYLE,
            'linewidth': config.LINE_WIDTH_CFNN,
            'alpha': config.CFNN_ALPHA,
            'is_variant': False
        },
        'CFNN-Boost': {
            'color': config.COLOR_BOOST,
            'label': 'CFNN-Boost',
            'linestyle': config.VARIANT_LINE_STYLE,
            'linewidth': config.LINE_WIDTH_VARIANT,
            'alpha': config.LINE_ALPHA,
            'is_variant': True
        },
        'CFNN-MoE': {
            'color': config.COLOR_MOE,
            'label': 'CFNN-MoE',
            'linestyle': config.VARIANT_LINE_STYLE,
            'linewidth': config.LINE_WIDTH_VARIANT,
            'alpha': config.LINE_ALPHA,
            'is_variant': True
        },
        'CFNN-Hybrid': {
            'color': config.COLOR_HYBRID,
            'label': 'CFNN-Hybrid',
            'linestyle': config.VARIANT_LINE_STYLE,
            'linewidth': config.LINE_WIDTH_VARIANT,
            'alpha': config.LINE_ALPHA,
            'is_variant': True
        }
    }

    # 统一训练轮数
    standard_epochs = 200

    # 绘制顺序
    draw_order = ['CFNN', 'CFNN-Hybrid', 'CFNN-MoE', 'CFNN-Boost']

    # ==================== 左图：损失曲线 ====================
    for model_name in draw_order:
        histories = loss_data.get(model_name, [])
        if not histories:
            continue

        epochs, mean_loss, std_loss = normalize_curves(
            histories, max_length=standard_epochs
        )
        mean_smooth = smooth_curve(mean_loss, config.SMOOTHING_WINDOW)

        cfg = model_configs.get(model_name, {})

        # 绘制损失曲线
        ax_loss.plot(
            epochs,
            mean_smooth,
            color=cfg['color'],
            linewidth=cfg['linewidth'],
            linestyle=cfg['linestyle'],
            alpha=cfg['alpha'],
            label=cfg['label'].replace('\n', ' ')
        )

        # 标注超出 Y 轴限制的损失点
        if config.MAX_LOSS_ANNOTATE:
            # 找出所有超出阈值的点
            exceeded_mask = np.array(mean_smooth) > config.Y_AXIS_MAX_LIMIT

            if np.any(exceeded_mask):
                # 找到第一个超出的点（通常是最重要的）
                first_exceeded_idx = np.where(exceeded_mask)[0][0]
                first_exceeded_epoch = epochs[first_exceeded_idx]
                first_exceeded_value = mean_smooth[first_exceeded_idx]

                # 绘制标记（在 Y 轴上限边缘）
                ax_loss.scatter(
                    [first_exceeded_epoch],
                    [config.Y_AXIS_MAX_LIMIT * 0.95],
                    marker=config.MAX_LOSS_MARKER,
                    s=config.MAX_LOFF_MARKER_SIZE ** 2,
                    color=config.MAX_LOSS_COLOR,
                    zorder=5,
                    edgecolors='black',
                    linewidths=0.5
                )

                # 标注数值（在 Y 轴上限上方）
                ax_loss.annotate(
                    f'{cfg["label"]}: {config.MAX_LOSS_FORMAT.format(first_exceeded_value)}',
                    xy=(first_exceeded_epoch, config.Y_AXIS_MAX_LIMIT * 0.95),
                    xytext=(first_exceeded_epoch, config.Y_AXIS_MAX_LIMIT * 0.85 - config.MAX_LOSS_OFFSET_Y),
                    fontsize=config.MAX_LOSS_FONT_SIZE,
                    color=config.MAX_LOSS_COLOR,
                    fontweight='bold',
                    ha='center',
                    va='top',
                    arrowprops=dict(arrowstyle='->', color=config.MAX_LOSS_COLOR, lw=1)
                )

        # 变体绘制方差带
        if cfg.get('is_variant', False):
            ax_loss.fill_between(
                epochs,
                np.maximum(0, np.array(mean_smooth) - np.array(std_loss)),
                np.array(mean_smooth) + np.array(std_loss),
                color=cfg['color'],
                alpha=0.15,
                linewidth=0
            )

    # 左图设置
    ax_loss.set_xlabel('Training Epoch', fontsize=config.LABEL_FONT_SIZE, fontweight='bold')
    ax_loss.set_ylabel('Training Loss', fontsize=config.LABEL_FONT_SIZE, fontweight='bold')
    ax_loss.set_title('(a) Training Loss Curves', fontsize=config.TITLE_FONT_SIZE, fontweight='bold', loc='left')
    ax_loss.tick_params(labelsize=config.TICK_SIZE)
    ax_loss.grid(True, alpha=config.GRID_ALPHA, linestyle=config.GRID_LINE_STYLE, linewidth=config.GRID_LINE_WIDTH)
    ax_loss.legend(loc='lower right', fontsize=config.LEGEND_FONT_SIZE, framealpha=0.95)
    ax_loss.set_xlim(left=0)
    ax_loss.set_ylim(bottom=-0.1, top=config.Y_AXIS_MAX_LIMIT)  # 设置 Y 轴上限

    # ==================== 右图：梯度标准差对比 ====================
    # 计算每个模型的平均梯度标准差
    grad_std_means = {}
    for model_name in ['CFNN', 'CFNN-Boost', 'CFNN-MoE', 'CFNN-Hybrid']:
        values = grad_std_data.get(model_name, [])
        if values:
            grad_std_means[model_name] = np.mean(values)
        else:
            grad_std_means[model_name] = 0.0

    # 按梯度标准差排序（不稳定的在前）
    models_sorted = sorted(grad_std_means.keys(), key=lambda x: grad_std_means[x], reverse=True)

    positions = np.arange(len(models_sorted))
    values = [grad_std_means[m] for m in models_sorted]
    colors = [model_configs[m]['color'] for m in models_sorted]
    hatches = [config.CFNN_BAR_HATCH if not model_configs[m]['is_variant'] else None for m in models_sorted]

    # 绘制柱状图
    for i, (pos, val, color, hatch) in enumerate(zip(positions, values, colors, hatches)):
        # 判断是否为 CFNN 并需要截断
        is_cfnn = models_sorted[i] == 'CFNN'
        display_val = min(val, 1.0) if is_cfnn else val

        ax_stab.bar(
            pos,
            display_val,
            width=config.BAR_WIDTH,
            color=color,
            alpha=config.BAR_ALPHA,
            hatch=hatch if hatch else None,
            edgecolor='black',
            linewidth=2 if hatch else 0.5,
            label=model_configs[models_sorted[i]]['label'].replace('\n', ' ')
        )

        # 如果被截断，添加截断标记
        if is_cfnn and val > 1.0:
            ax_stab.plot(pos, 1.0, marker='v', color='red', markersize=12, zorder=5)

        # 添加数值标签
        ax_stab.text(
            pos,
            display_val,
            f'{val:.2f}',
            ha='center',
            va='bottom',
            fontsize=config.BAR_LABEL_FONT_SIZE,
            fontweight='bold'
        )

    # # CFNN 不稳定标注
    # if models_sorted[0] == 'CFNN':
    #     cfnn_val = grad_std_means['CFNN']
    #     min_variant_val = min(grad_std_means[m] for m in models_sorted if model_configs[m]['is_variant'] and grad_std_means[m] > 0)
    #     improvement = cfnn_val / min_variant_val if min_variant_val > 0 else 0

    #     # 添加不稳定性标注
    #     ax_stab.annotate(
    #         f'梯度不稳定!\nGradStd 高达 {cfnn_val:.1f}\n比变体差 {improvement:.0f}×',
    #         xy=(0, cfnn_val),
    #         xytext=(0.5, 0.92),
    #         textcoords='axes fraction',
    #         fontsize=config.ANNOTATION_FONT_SIZE,
    #         color='red',
    #         fontweight='bold',
    #         arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
    #         bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', edgecolor='red', alpha=0.9, linewidth=2)
    #     )

    # 右图设置
    ax_stab.set_xlabel('Model Architecture', fontsize=config.LABEL_FONT_SIZE, fontweight='bold')
    ax_stab.set_ylabel(
        'Gradient Standard Deviation',
        fontsize=config.LABEL_FONT_SIZE,
        fontweight='bold'
    )
    ax_stab.set_title('(b) Gradient Instability', fontsize=config.TITLE_FONT_SIZE, fontweight='bold', loc='left')
    ax_stab.set_xticks(positions)
    ax_stab.set_xticklabels([model_configs[m]['label'] for m in models_sorted], fontsize=config.LEGEND_FONT_SIZE)
    ax_stab.tick_params(labelsize=config.TICK_SIZE, axis='y')
    ax_stab.grid(True, alpha=config.GRID_ALPHA, axis='y', linestyle=config.GRID_LINE_STYLE, linewidth=config.GRID_LINE_WIDTH)
    ax_stab.set_ylim(bottom=0)

    # 整体标题
    fig.suptitle(
        'Training Stability: CFNN Gradient Instability vs. Improved Variants',
        fontsize=config.TITLE_FONT_SIZE + 2,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout()

    # 保存图形
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存合并图形
        fig.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"合并图形已保存至: {save_path}")

        # ===== 保存左图（损失曲线）为独立文件 =====
        fig_left, ax_left_only = plt.subplots(figsize=(12, 7), dpi=config.DPI)

        # 重新绘制左图内容
        for model_name in draw_order:
            histories = loss_data.get(model_name, [])
            if not histories:
                continue

            epochs, mean_loss, std_loss = normalize_curves(
                histories, max_length=standard_epochs
            )
            mean_smooth = smooth_curve(mean_loss, config.SMOOTHING_WINDOW)
            cfg = model_configs.get(model_name, {})

            ax_left_only.plot(
                epochs,
                mean_smooth,
                color=cfg['color'],
                linewidth=cfg['linewidth'],
                linestyle=cfg['linestyle'],
                alpha=cfg['alpha'],
                label=cfg['label']
            )

            # 标注超出 Y 轴限制的损失点
            if config.MAX_LOSS_ANNOTATE:
                # 找出所有超出阈值的点
                exceeded_mask = np.array(mean_smooth) > config.Y_AXIS_MAX_LIMIT

                if np.any(exceeded_mask):
                    # 找到第一个超出的点（通常是最重要的）
                    first_exceeded_idx = np.where(exceeded_mask)[0][0]
                    first_exceeded_epoch = epochs[first_exceeded_idx]
                    first_exceeded_value = mean_smooth[first_exceeded_idx]

                    # 绘制标记（在 Y 轴上限边缘）
                    ax_left_only.scatter(
                        [first_exceeded_epoch],
                        [config.Y_AXIS_MAX_LIMIT * 0.95],
                        marker=config.MAX_LOSS_MARKER,
                        s=config.MAX_LOFF_MARKER_SIZE ** 2,
                        color=config.MAX_LOSS_COLOR,
                        zorder=5,
                        edgecolors='black',
                        linewidths=0.5
                    )

                    # 标注数值（在 Y 轴上限上方）
                    ax_left_only.annotate(
                        f'{cfg["label"]}: {config.MAX_LOSS_FORMAT.format(first_exceeded_value)}',
                        xy=(first_exceeded_epoch, config.Y_AXIS_MAX_LIMIT * 0.95),
                        xytext=(first_exceeded_epoch, config.Y_AXIS_MAX_LIMIT * 0.85 - config.MAX_LOSS_OFFSET_Y),
                        fontsize=config.MAX_LOSS_FONT_SIZE,
                        color=config.MAX_LOSS_COLOR,
                        fontweight='bold',
                        ha='center',
                        va='top',
                        arrowprops=dict(arrowstyle='->', color=config.MAX_LOSS_COLOR, lw=1)
                    )

            if cfg.get('is_variant', False):
                ax_left_only.fill_between(
                    epochs,
                    np.maximum(0, np.array(mean_smooth) - np.array(std_loss)),
                    np.array(mean_smooth) + np.array(std_loss),
                    color=cfg['color'],
                    alpha=0.15,
                    linewidth=0
                )

        # ax_left_only.set_xlabel('Training Epoch', fontsize=config.LABEL_FONT_SIZE, fontweight='bold')
        # ax_left_only.set_ylabel('Training Loss (MSE)', fontsize=config.LABEL_FONT_SIZE, fontweight='bold')
        # ax_left_only.set_title('Training Loss Curves', fontsize=config.TITLE_FONT_SIZE, fontweight='bold')
        ax_left_only.tick_params(labelsize=config.TICK_SIZE)
        ax_left_only.grid(True, alpha=config.GRID_ALPHA, linestyle=config.GRID_LINE_STYLE, linewidth=config.GRID_LINE_WIDTH)
        ax_left_only.legend(loc='lower right', fontsize=config.LEGEND_FONT_SIZE, framealpha=0.95)
        ax_left_only.set_xlim(left=0)
        ax_left_only.set_ylim(bottom=-0.1, top=config.Y_AXIS_MAX_LIMIT)  # 设置 Y 轴上限

        # 保存左图
        left_path = save_path.parent / f"{save_path.stem}_loss_curves.png"
        fig_left.savefig(left_path, dpi=config.DPI, bbox_inches='tight')
        print(f"左图（损失曲线）已保存至: {left_path}")

        # 保存左图 PDF
        left_path_pdf = save_path.parent / f"{save_path.stem}_loss_curves.pdf"
        fig_left.savefig(left_path_pdf, dpi=config.DPI, bbox_inches='tight', format='pdf')
        print(f"左图 PDF 已保存至: {left_path_pdf}")

        plt.close(fig_left)

        # ===== 保存右图（梯度标准差）为独立文件 =====
        fig_right, ax_right_only = plt.subplots(figsize=(10, 7), dpi=config.DPI)

        # 重新绘制右图内容
        positions = np.arange(len(models_sorted))
        values = [grad_std_means[m] for m in models_sorted]
        colors = [model_configs[m]['color'] for m in models_sorted]
        hatches = [config.CFNN_BAR_HATCH if not model_configs[m]['is_variant'] else None for m in models_sorted]

        for i, (pos, val, color, hatch) in enumerate(zip(positions, values, colors, hatches)):
            # 判断是否为 CFNN 并需要截断
            is_cfnn = models_sorted[i] == 'CFNN'
            display_val = min(val, 1.0) if is_cfnn else val

            ax_right_only.bar(
                pos,
                display_val,
                width=config.BAR_WIDTH,
                color=color,
                alpha=config.BAR_ALPHA,
                hatch=hatch if hatch else None,
                edgecolor='black',
                linewidth=2 if hatch else 0.5
            )

            # 如果被截断，添加截断标记
            if is_cfnn and val > 1.0:
                ax_right_only.plot(pos, 1.0, marker='v', color='red', markersize=12, zorder=5)

            ax_right_only.text(
                pos,
                display_val,
                f'{val:.2f}',
                ha='center',
                va='bottom',
                fontsize=config.BAR_LABEL_FONT_SIZE,
                fontweight='bold'
            )

        ax_right_only.set_xlabel('Model Architecture', fontsize=config.LABEL_FONT_SIZE, fontweight='bold')
        ax_right_only.set_ylabel(
            'Gradient Standard Deviation',
            fontsize=config.LABEL_FONT_SIZE,
            fontweight='bold'
        )
        ax_right_only.set_title('Gradient Instability', fontsize=config.TITLE_FONT_SIZE, fontweight='bold')
        ax_right_only.set_xticks(positions)
        ax_right_only.set_xticklabels([model_configs[m]['label'] for m in models_sorted], fontsize=config.LEGEND_FONT_SIZE)
        ax_right_only.tick_params(labelsize=config.TICK_SIZE, axis='y')
        ax_right_only.grid(True, alpha=config.GRID_ALPHA, axis='y', linestyle=config.GRID_LINE_STYLE, linewidth=config.GRID_LINE_WIDTH)
        ax_right_only.set_ylim(bottom=0)

        # 保存右图
        right_path = save_path.parent / f"{save_path.stem}_gradient_std.png"
        fig_right.savefig(right_path, dpi=config.DPI, bbox_inches='tight')
        print(f"右图（梯度标准差）已保存至: {right_path}")

        # 保存右图 PDF
        right_path_pdf = save_path.parent / f"{save_path.stem}_gradient_std.pdf"
        fig_right.savefig(right_path_pdf, dpi=config.DPI, bbox_inches='tight', format='pdf')
        print(f"右图 PDF 已保存至: {right_path_pdf}")

        plt.close(fig_right)

    return fig


# ==========================================
# 主函数
# ==========================================
def main():
    """主函数"""

    print("=" * 70)
    print("CFNN 与变体训练损失下降稳定性对比图")
    print("双面板设计：左图损失曲线，右图梯度标准差对比")
    print("=" * 70)

    # 加载数据
    loss_data, grad_std_data = load_all_data()

    print("\n数据加载统计:")
    for model, histories in loss_data.items():
        print(f"  {model}: {len(histories)} 条损失记录")

    print("\n梯度标准差统计:")
    for model, values in grad_std_data.items():
        if values:
            print(f"  {model}: 平均 GradStd = {np.mean(values):.4f}")

    # 生成图形（抑制 matplotlib 字体警告）
    print("\n生成训练稳定性对比图...")
    with WarningCapture():
        fig = plot_training_stability(
            loss_data,
            grad_std_data,
            config=PlotConfig,
            save_path='results/plots/training_stability_comparison.png'
        )

    # 同时生成 PDF 版本
    fig.savefig(
        'results/plots/training_stability_comparison.pdf',
        dpi=PlotConfig.DPI,
        bbox_inches='tight',
        format='pdf'
    )
    print("PDF 版本已保存至: results/plots/training_stability_comparison.pdf")

    plt.close(fig)

    print("\n" + "=" * 70)
    print("绘图完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

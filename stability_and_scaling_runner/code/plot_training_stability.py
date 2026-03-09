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
    LABEL_FONT_SIZE = 28
    LEGEND_FONT_SIZE = 20
    ANNOTATION_FONT_SIZE = 28
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
    TICK_SIZE = 16

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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.ticker as ticker

def plot_training_stability(
    loss_data: Dict[str, List[List[float]]],
    grad_std_data: Dict[str, List[float]],  # 虽然不画右图，但保留参数以兼容原有调用
    config: PlotConfig = PlotConfig,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    单面板优化版：只绘制左侧损失曲线
    - 小窗内双对数轴：左侧度量变体，右侧度量 CFNN
    - 移除右侧梯度稳定性柱状图
    """
    # 基础样式设置
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 创建单面板图形，调整宽高比使其更协调 (例如从 16:7 改为 12:8)
    fig, ax_loss = plt.subplots(figsize=(12, 8), dpi=config.DPI)
    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.1, right=0.9)

    # 模型配置
    model_configs = {
        'CFNN': {'color': config.COLOR_CFNN, 'label': 'CFNN', 'linestyle': '--', 'lw': 1.5, 'alpha': 0.5},
        'CFNN-Hybrid': {'color': config.COLOR_HYBRID, 'label': 'CFNN-Hybrid', 'linestyle': '-', 'lw': 2.5, 'alpha': 1.0},
        'CFNN-MoE': {'color': config.COLOR_MOE, 'label': 'CFNN-MoE', 'linestyle': '-', 'lw': 2.5, 'alpha': 1.0},
        'CFNN-Boost': {'color': config.COLOR_BOOST, 'label': 'CFNN-Boost', 'linestyle': '-', 'lw': 2.5, 'alpha': 1.0}
    }
    
    # 创建小窗 (由于变宽了，位置微调)
    ax_inset = inset_axes(ax_loss, width="40%", height="35%", loc='upper left', 
                          bbox_to_anchor=(0.4, -0.05, 1, 1), bbox_transform=ax_loss.transAxes)
    ax_inset_right = ax_inset.twinx()

    standard_epochs = 200
    lines = []
    draw_order = ['CFNN', 'CFNN-Hybrid', 'CFNN-MoE', 'CFNN-Boost']
    
    for model_name in draw_order:
        histories = loss_data.get(model_name, [])
        if not histories: continue

        epochs, mean_loss, std_loss = normalize_curves(histories, max_length=standard_epochs)
        mean_smooth = smooth_curve(mean_loss, config.SMOOTHING_WINDOW)
        cfg = model_configs[model_name]

        # 主图绘制
        l, = ax_loss.plot(epochs, mean_smooth, color=cfg['color'], lw=cfg['lw'], 
                         ls=cfg['linestyle'], alpha=cfg['alpha'], label=cfg['label'])
        lines.append(l)
        
        # 小窗绘制
        if model_name == 'CFNN':
            ax_inset_right.plot(epochs, mean_smooth, color=cfg['color'], lw=cfg['lw'], ls=cfg['linestyle'])
        else:
            ax_inset.plot(epochs, mean_smooth, color=cfg['color'], lw=cfg['lw'], ls=cfg['linestyle'])

    # --- 小窗双对数轴设置 ---
    ax_inset.set_xlim(160, 200)
    
    # 左对数轴 (变体)
    ax_inset.set_yscale('log')
    ax_inset.set_ylim(2e-1, 1e1) 
    ax_inset.tick_params(axis='y', labelcolor=config.COLOR_HYBRID, labelsize=config.TICK_SIZE - 4)
    ax_inset.set_ylabel('Variants Loss', fontsize=18)

    # 2. 右侧对数轴 (CFNN)
    ax_inset_right.set_yscale('log')
    ax_inset_right.set_ylim(1e1, 1e7)
    ax_inset_right.tick_params(axis='y', labelcolor=config.COLOR_CFNN, labelsize=config.TICK_SIZE - 4)
    ax_inset_right.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax_inset_right.set_ylabel('CFNN Loss', fontsize=18, labelpad=15)

    # 辅助线和网格
    ax_inset.grid(True, which="both", alpha=0.1, ls='-')
    mark_inset(ax_loss, ax_inset, loc1=1, loc2=3, fc="none", ec="0.5", ls=':', lw=0.8)

    # --- 主图修饰 ---
    ax_loss.set_ylim(-0.1e7, 5.5e7) 
    ax_loss.set_xlabel('Training Epoch', fontsize=config.LABEL_FONT_SIZE, fontweight='bold')
    ax_loss.set_ylabel('Training Loss', fontsize=config.LABEL_FONT_SIZE, fontweight='bold')
    # ax_loss.set_title('Training Loss Convergence Analysis', loc='center', fontsize=config.TITLE_FONT_SIZE, pad=40)
    ax_loss.grid(True, alpha=0.2, ls='-')

    # --- 顶部横向图例 ---
    # 由于只有一张图，图例位置可以放得更显眼一点
    ax_loss.legend(handles=lines, loc='upper center', bbox_to_anchor=(0.5, 1.12),
                   ncol=4, fontsize=config.LEGEND_FONT_SIZE, frameon=False)

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

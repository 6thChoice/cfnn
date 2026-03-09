"""
领域知识分析器
评估模型特征重要性是否与领域知识一致
"""

from .base_analyzer import BaseAnalyzer
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import pandas as pd
import os


class DomainAnalyzer(BaseAnalyzer):
    """领域知识分析器"""

    def __init__(self, config: Dict[str, Any], save_dir: str = 'results'):
        """
        Args:
            config: 配置字典
                - feature_info: 特征信息（包含 domain_importance）
            save_dir: 结果保存目录
        """
        super().__init__(config, save_dir)

        self.feature_info = config['feature_info']
        self.domain_importance = self.feature_info.get('domain_importance', {})
        self.feature_names = self.feature_info['feature_names']

    def compute_metrics(self, model: torch.nn.Module,
                       test_loader: DataLoader) -> Dict[str, float]:
        """
        计算性能指标

        Args:
            model: 训练好的模型
            test_loader: 测试数据加载器

        Returns:
            metrics: 指标字典
        """
        return self.compute_performance_metrics(model, test_loader)

    def analyze_shap(self, model: torch.nn.Module,
                    train_loader: DataLoader,
                    test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 SHAP 值

        Args:
            model: 训练好的模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器

        Returns:
            shap_values: SHAP 值
            test_data: 测试数据
        """
        return self.compute_shap_values(model, train_loader, test_loader)

    def compute_domain_consistency(self, shap_values: np.ndarray) -> Dict[str, float]:
        """
        计算与领域知识的一致性

        Args:
            shap_values: SHAP 值

        Returns:
            consistency_metrics: {
                'spearman_correlation': Spearman 相关系数
                'rank_correlation': 排序相关性
                'top_k_consistency': Top K 一致性
            }
        """
        # 计算 SHAP 重要性（平均绝对值）
        mean_importance = np.mean(np.abs(shap_values), axis=0)

        # 获取领域知识重要性
        domain_scores = np.array([
            self.domain_importance.get(fname, 2)
            for fname in self.feature_names
        ])

        # 计算 Spearman 相关系数
        spearman_corr, p_value = spearmanr(mean_importance, domain_scores)

        # 计算排序
        shap_ranking = np.argsort(-mean_importance)
        domain_ranking = np.argsort(-domain_scores)

        # 计算排序一致性 (Kendall's tau 的简化版本)
        n = len(mean_importance)
        concordant_pairs = 0
        total_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                # 比较 SHAP 排序和领域排序
                shap_order = (mean_importance[i] > mean_importance[j])
                domain_order = (domain_scores[i] > domain_scores[j])

                if shap_order == domain_order:
                    concordant_pairs += 1
                total_pairs += 1

        rank_consistency = concordant_pairs / total_pairs if total_pairs > 0 else 0

        # Top K 一致性
        top_k_values = [3, 5, min(10, len(mean_importance))]
        top_k_consistency = {}

        for k in top_k_values:
            if k > len(mean_importance):
                continue

            # SHAP Top K 特征
            shap_top_k = set(shap_ranking[:k])

            # 领域知识 Top K 特征
            domain_top_k_indices = np.argsort(-domain_scores)[:k]
            domain_top_k = set(domain_top_k_indices)

            # 计算重叠率
            overlap = len(shap_top_k.intersection(domain_top_k))
            consistency = overlap / k
            top_k_consistency[f'top_{k}'] = float(consistency)

        return {
            'spearman_correlation': float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
            'spearman_p_value': float(p_value) if not np.isnan(p_value) else 1.0,
            'rank_consistency': float(rank_consistency),
            **top_k_consistency
        }

    def generate_feature_comparison_table(self, shap_values: np.ndarray) -> Dict[str, Any]:
        """
        生成特征对比表格

        Args:
            shap_values: SHAP 值

        Returns:
            comparison_data: 对比数据
        """
        # 计算 SHAP 重要性
        mean_importance = np.mean(np.abs(shap_values), axis=0)
        std_importance = np.std(shap_values, axis=0)

        # 获取领域知识
        comparison_data = []
        for i, fname in enumerate(self.feature_names):
            domain_score = self.domain_importance.get(fname, 2)
            domain_label = {1: 'Low', 2: 'Medium', 3: 'High'}.get(domain_score, 'Unknown')

            comparison_data.append({
                'feature': fname,
                'shap_importance': float(mean_importance[i]),
                'shap_std': float(std_importance[i]),
                'domain_importance': domain_score,
                'domain_label': domain_label,
                'shap_rank': int(np.argsort(-mean_importance).tolist().index(i) + 1),
                'domain_rank': int(np.argsort([-self.domain_importance.get(fn, 2)
                                               for fn in self.feature_names]).tolist().index(i) + 1)
            })

        return comparison_data

    def _sort_models(self, results: Dict[str, Any]) -> List[Tuple[str, Any, str]]:
        """
        排序模型，按指定顺序：CFNN、CFNN-Boost、CFNN-MoE、CFNN-Hybrid、MLP

        Args:
            results: 分析结果字典

        Returns:
            sorted_items: 排序后的(key, value, display_name)列表
        """
        # 定义模型顺序和显示名称映射
        model_order = {
            'Standard': 0,
            'Boost': 1,
            'MoE': 2,
            'Hybrid': 3,
            'MLP': 4
        }

        model_display_names = {
            'Standard': 'CFNN',
            'Boost': 'CFNN-Boost',
            'MoE': 'CFNN-MoE',
            'Hybrid': 'CFNN-Hybrid',
            'MLP': 'MLP'
        }

        items = []
        for key, value in results.items():
            model_name = key.replace('_domain', '')
            order = model_order.get(model_name, 99)
            display_name = model_display_names.get(model_name, model_name)
            items.append((key, value, display_name, order))

        # 按顺序排序
        items.sort(key=lambda x: x[3])

        # 返回 (key, value, display_name)
        return [(item[0], item[1], item[2]) for item in items]

    def plot_feature_importance_comparison(self, results: Dict[str, Any],
                                           save_path: str = None) -> str:
        """
        绘制特征重要性对比图（水平条形图）

        直观展示各模型判断的特征重要性与领域知识（真实重要性）之间的关系
        特征按X1-X8顺序排列（不按重要性排序）

        Args:
            results: 分析结果字典
            save_path: 保存路径

        Returns:
            save_path: 图片保存路径
        """
        # 字号配置（明确化）
        FONT_SIZES = {
            'model_name': 22,      # 模型名称
            'value_label': 14,     # 数值标签
            'y_tick': 18,          # Y轴刻度
            'x_tick': 18,
            'x_label': 21,         # X轴标签
            'legend': 21,          # 图例
            'stars': 17,           # 星级标记
        }

        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 准备数据（按指定顺序：CFNN、CFNN-Boost、CFNN-MoE、CFNN-Hybrid、MLP）
        sorted_results = self._sort_models(results)
        model_names = []
        all_importance_data = []

        for key, result, display_name in sorted_results:
            model_names.append(display_name)
            shap_values = result['shap_values']
            mean_importance = np.mean(np.abs(shap_values), axis=0)
            all_importance_data.append(mean_importance)

        # 转换为numpy数组
        importance_matrix = np.array(all_importance_data)  # [n_models, n_features]

        # 获取领域知识重要性
        domain_scores = np.array([
            self.domain_importance.get(fname, 2)
            for fname in self.feature_names
        ])

        # 创建图形
        n_models = len(model_names)
        n_features = len(self.feature_names)

        # 计算图表布局
        fig_height = max(8, n_features * 0.6)
        fig, axes = plt.subplots(1, n_models + 1, figsize=(4 * (n_models + 1), fig_height))

        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # 颜色映射：根据领域知识重要性给特征着色
        feature_colors = []
        for score in domain_scores:
            if score == 3:  # 高重要性
                feature_colors.append('#d62728')  # 红色
            elif score == 2:  # 中重要性
                feature_colors.append('#ff7f0e')  # 橙色
            else:  # 低重要性
                feature_colors.append('#2ca02c')  # 绿色

        # 绘制每个模型的特征重要性
        for idx, (model_name, importance) in enumerate(zip(model_names, importance_matrix)):
            ax = axes[idx]

            # 归一化重要性用于展示（0-100分制）
            if importance.max() > 0:
                normalized_importance = (importance / importance.max()) * 100
            else:
                normalized_importance = importance

            # 按特征名称排序（X1, X2, ..., X8）
            sorted_indices = list(range(n_features))  # 保持原有顺序 [0, 1, 2, ..., 7]
            sorted_features = [self.feature_names[i] for i in sorted_indices]
            sorted_values = normalized_importance[sorted_indices]
            sorted_colors = [feature_colors[i] for i in sorted_indices]

            # 绘制水平条形图
            bars = ax.barh(range(n_features), sorted_values, color=sorted_colors, alpha=0.8, edgecolor='white')

            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, sorted_values)):
                ax.text(val + 2, bar.get_y() + bar.get_height()/2,
                       f'{val:.1f}',
                       va='center', ha='left', fontsize=FONT_SIZES['value_label'], fontweight='bold')
                
            ax.tick_params(axis='x', labelsize=FONT_SIZES['x_tick'])

            # 添加模型名称在子图上方
            ax.text(0.5, 1.02, model_name, transform=ax.transAxes,
                   ha='center', va='bottom', fontsize=FONT_SIZES['model_name'], fontweight='bold')

            # 设置标签（无标题）
            ax.set_yticks(range(n_features))
            ax.set_yticklabels(sorted_features, fontsize=FONT_SIZES['y_tick'])
            # ax.set_xlabel('Importance Score (0-100)', fontsize=FONT_SIZES['x_label'])
            ax.set_xlim(0, 110)

            # 添加网格线
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)

            # 移除上方和右侧框线
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # 绘制领域知识（Ground Truth）
        ax = axes[-1]
        domain_values = domain_scores * 33.33  # 转换为0-100分制 (1->33, 2->67, 3->100)

        # 按特征名称排序（X1, X2, ..., X8）
        sorted_indices = list(range(n_features))
        sorted_features = [self.feature_names[i] for i in sorted_indices]
        sorted_values = domain_values[sorted_indices]
        sorted_colors = [feature_colors[i] for i in sorted_indices]

        bars = ax.barh(range(n_features), sorted_values, color=sorted_colors, alpha=0.8, edgecolor='white')

        # 添加 Ground Truth 标签
        ax.text(0.5, 1.02, 'Ground Truth', transform=ax.transAxes,
               ha='center', va='bottom', fontsize=FONT_SIZES['model_name'], fontweight='bold')

        # for i, (bar, val) in enumerate(zip(bars, sorted_values)):
        #     stars = {33: '★', 67: '★★', 100: '★★★'}.get(int(val), '')
        #     ax.text(val + 2, bar.get_y() + bar.get_height()/2,
        #            stars,
        #            va='center', ha='left', fontsize=FONT_SIZES['stars'], fontweight='bold')

        ax.set_yticks(range(n_features))
        ax.set_yticklabels(sorted_features, fontsize=10)
        ax.tick_params(axis='x', labelsize=FONT_SIZES['x_tick'])
        ax.tick_params(axis='y', labelsize=FONT_SIZES['y_tick'])
        # ax.set_xlabel('Domain Knowledge Score', fontsize=11)
        ax.set_xlim(0, 110)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 添加图例
        legend_elements = [
            mpatches.Patch(color='#d62728', alpha=0.8, label='High Importance (★★★)'),
            mpatches.Patch(color='#ff7f0e', alpha=0.8, label='Medium Importance (★★)'),
            mpatches.Patch(color='#2ca02c', alpha=0.8, label='Low Importance (★)')
        ]
        fig.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=FONT_SIZES['legend'],
                  frameon=False, fancybox=False, shadow=False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # 保存图片
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'feature_importance_comparison.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Feature importance comparison plot saved to: {save_path}")
        return save_path

    def plot_importance_radar(self, results: Dict[str, Any],
                              save_path: str = None) -> str:
        """
        绘制特征重要性雷达图

        展示各模型在关键特征上的重要性打分差异
        图例放在图片正下方，一行五列，无边框
        Y轴范围根据真实特征重要性设置，目标值在三分之二位置

        Args:
            results: 分析结果字典
            save_path: 保存路径

        Returns:
            save_path: 图片保存路径
        """
        # 字号配置（明确化）
        FONT_SIZES = {
            'feature_label': 14,   # 特征标签（X1-X8）
            'radial_tick': 12,     # 径向刻度（Under/Target/Over）
            'legend': 13,          # 图例
            'star': 10,            # Target星号标记
        }

        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 准备数据（按指定顺序：CFNN、CFNN-Boost、CFNN-MoE、CFNN-Hybrid、MLP）
        sorted_results = self._sort_models(results)
        model_names = []
        all_importance_data = []

        for key, result, display_name in sorted_results:
            model_names.append(display_name)
            shap_values = result['shap_values']
            mean_importance = np.mean(np.abs(shap_values), axis=0)
            all_importance_data.append(mean_importance)

        importance_matrix = np.array(all_importance_data)
        n_models = len(model_names)
        n_features = len(self.feature_names)

        # 获取领域知识分数 (1, 2, 3) 转换为 (33.3, 66.7, 100)
        domain_scores = np.array([
            self.domain_importance.get(fname, 2) * 33.33
            for fname in self.feature_names
        ])

        # 第一步：对每个模型单独归一化到 0-1（保持模型内部相对重要性）
        model_normalized = np.zeros_like(importance_matrix)
        for i in range(n_models):
            if importance_matrix[i].max() > 0:
                model_normalized[i] = importance_matrix[i] / importance_matrix[i].max()

        # 第二步：根据真实重要性调整比例
        # 目标：真实重要性在雷达图上位于约2/3处
        # 每个特征的归一化重要性乘以 (1/0.67) 的因子，使典型值接近0.67
        # 但为了展示过强调和过轻视，我们需要动态范围

        # 为每个特征设置一个参考最大值
        # 真实重要性应位于2/3处，所以参考最大值 = 真实重要性 / 0.67
        reference_max = domain_scores / 0.67

        # 但如果模型有过度强调的情况，需要更大的范围
        # 找出每个特征在所有模型中的最大原始归一化值
        max_per_feature = model_normalized.max(axis=0)

        # 最终归一化：模型值 * (真实重要性 / 0.67) / 参考最大值
        # 简化：模型值 / 0.67，但不超过1.0（Over区域）
        normalized_matrix = np.zeros_like(model_normalized)
        for i in range(n_models):
            for j in range(n_features):
                # 将模型归一化值映射到 0-1.5 范围，使0.67对应目标
                val = model_normalized[i, j] / 0.67
                # 限制在0-1.5范围内，超过1.0表示"Over"
                normalized_matrix[i, j] = min(val, 1.5)

        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # 颜色配置（使用tab10确保颜色区分明显）
        colors = plt.cm.tab10(np.linspace(0, 1, n_models))

        for i, (model_name, importance) in enumerate(zip(model_names, normalized_matrix)):
            values = importance.tolist()
            values += values[:1]  # 闭合

            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])

        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.feature_names, fontsize=FONT_SIZES['feature_label'])

        # 设置径向范围 0-1.0，Target在0.67位置
        # 超过1.0的区域表示模型过度强调
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.33, 0.67, 1.0])
        ax.set_yticklabels(['Under', 'Target', 'Over'], fontsize=FONT_SIZES['radial_tick'])
        ax.grid(True, linestyle='--', alpha=0.5)

        # 添加虚线圆圈标记Target位置（0.67）
        circle = plt.Circle((0, 0), 0.67, transform=ax.transData._b,
                           fill=False, linestyle='--', alpha=0.5, color='gray')
        ax.add_patch(circle)

        # 在每个轴上标记Target位置
        for angle in angles[:-1]:
            ax.plot(angle, 0.67, 'k*', markersize=FONT_SIZES['star'], alpha=0.6)

        # 图例放在正下方，一行五列，无边框
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                   fontsize=FONT_SIZES['legend'], frameon=False, ncol=5)

        # 调整布局，为下方图例留出空间
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        if save_path is None:
            save_path = os.path.join(self.save_dir, 'feature_importance_radar.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Feature importance radar plot saved to: {save_path}")
        return save_path

    def plot_importance_heatmap(self, results: Dict[str, Any],
                                save_path: str = None) -> str:
        """
        绘制特征重要性热力图

        直观比较所有模型在各特征上的重要性打分
        MLP放在最后

        Args:
            results: 分析结果字典
            save_path: 保存路径

        Returns:
            save_path: 图片保存路径
        """
        # 字号配置（明确化）
        FONT_SIZES = {
            'annot': 12,           # 热力图数值标签
            'axis_label': 14,      # 轴标签（Features/Models）
            'cbar_label': 13,      # 颜色条标签
            'tick': 12,            # 刻度标签
        }

        import seaborn as sns

        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 准备数据（按指定顺序）
        sorted_results = self._sort_models(results)
        model_names = []
        all_importance_data = []

        for key, result, display_name in sorted_results:
            model_names.append(display_name)
            shap_values = result['shap_values']
            mean_importance = np.mean(np.abs(shap_values), axis=0)
            # 归一化到 0-100
            if mean_importance.max() > 0:
                mean_importance = (mean_importance / mean_importance.max()) * 100
            all_importance_data.append(mean_importance)

        # 添加领域知识作为参考
        domain_scores = np.array([
            self.domain_importance.get(fname, 2)
            for fname in self.feature_names
        ]) * 33.33

        model_names.append('Ground Truth')
        all_importance_data.append(domain_scores)

        # 创建DataFrame
        importance_matrix = np.array(all_importance_data)
        df = pd.DataFrame(importance_matrix, index=model_names, columns=self.feature_names)

        # 创建热力图
        fig, ax = plt.subplots(figsize=(12, max(6, len(model_names) * 0.5)))

        # 使用不同颜色映射突出 Ground Truth
        heatmap = sns.heatmap(df, annot=True, fmt='.1f', cmap='YlOrRd',
                   cbar_kws={'label': 'Importance Score (0-100)'},
                   annot_kws={'fontsize': FONT_SIZES['annot']},
                   linewidths=0.5, ax=ax, vmin=0, vmax=100)

        # 设置颜色条标签字号
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=FONT_SIZES['cbar_label'])
        cbar.set_label('Importance Score (0-100)', fontsize=FONT_SIZES['cbar_label'])

        # 高亮 Ground Truth 行
        ax.add_patch(plt.Rectangle((0, len(model_names)-1), len(self.feature_names), 1,
                                   fill=False, edgecolor='#1f77b4', lw=3))

        # 设置刻度标签字号
        ax.tick_params(axis='both', labelsize=FONT_SIZES['tick'])
        ax.set_xlabel('Features', fontsize=FONT_SIZES['axis_label'])
        ax.set_ylabel('Models', fontsize=FONT_SIZES['axis_label'])

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.save_dir, 'feature_importance_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Feature importance heatmap saved to: {save_path}")
        return save_path

    def plot_importance_correlation(self, results: Dict[str, Any],
                                    save_path: str = None) -> str:
        """
        绘制特征重要性与领域知识的相关性散点图
        五个子图并排放置，形成一行五列
        MLP放在最后

        Args:
            results: 分析结果字典
            save_path: 保存路径

        Returns:
            save_path: 图片保存路径
        """
        # 字号配置（明确化）
        FONT_SIZES = {
            'model_name': 13,      # 模型名称和相关系数
            'axis_label': 12,      # 轴标签（Domain Knowledge/Model Importance）
            'tick': 11,            # 刻度标签（Low/Med/High）
            'feature_name': 10,    # 特征名称标注
            'legend': 12,          # 图例
        }

        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 获取领域知识分数
        domain_scores = np.array([
            self.domain_importance.get(fname, 2)
            for fname in self.feature_names
        ])

        # 准备数据（按指定顺序）
        sorted_results = self._sort_models(results)

        # 创建子图 - 一行五列
        n_models = len(sorted_results)
        n_cols = n_models  # 所有模型并排
        n_rows = 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 5))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        colors = {'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#2ca02c'}

        for idx, (key, result, display_name) in enumerate(sorted_results):
            ax = axes[idx]
            model_name = display_name

            shap_values = result['shap_values']
            mean_importance = np.mean(np.abs(shap_values), axis=0)

            # 归一化
            if mean_importance.max() > 0:
                normalized_importance = (mean_importance / mean_importance.max()) * 3
            else:
                normalized_importance = mean_importance

            # 为每个特征标注重要性等级
            for i, fname in enumerate(self.feature_names):
                domain_level = domain_scores[i]
                level_name = {3: 'High', 2: 'Medium', 1: 'Low'}.get(int(domain_level), 'Unknown')

                ax.scatter(domain_level, normalized_importance[i],
                          c=colors[level_name], s=200, alpha=0.7,
                          edgecolors='white', linewidth=2, zorder=3)
                ax.annotate(fname, (domain_level, normalized_importance[i]),
                           textcoords="offset points", xytext=(5, 5),
                           ha='left', fontsize=FONT_SIZES['feature_name'], fontweight='bold')

            # 添加理想对角线
            ax.plot([1, 3], [1, 3], 'k--', alpha=0.3, linewidth=1, label='Perfect Match')

            # 计算并显示相关系数
            corr, p_value = spearmanr(domain_scores, normalized_importance)

            ax.set_xlabel('Domain Knowledge', fontsize=FONT_SIZES['axis_label'])
            ax.set_ylabel('Model Importance', fontsize=FONT_SIZES['axis_label'])
            # 在子图上方添加模型名称和相关系数（不是标题）
            ax.text(0.5, 1.02, f'{model_name}\nρ = {corr:.3f}{"*" if p_value < 0.05 else ""}',
                   transform=ax.transAxes, ha='center', va='bottom',
                   fontsize=FONT_SIZES['model_name'], fontweight='bold')
            ax.set_xlim(0.5, 3.5)
            ax.set_ylim(0, 3.5)
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(['Low\n(★)', 'Med\n(★★)', 'High\n(★★★)'], fontsize=FONT_SIZES['tick'])
            ax.set_yticks([1, 2, 3])
            ax.set_yticklabels(['Low', 'Med', 'High'], fontsize=FONT_SIZES['tick'])
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # 添加图例（放在下方）
        legend_elements = [
            plt.scatter([], [], c=colors['High'], s=100, label='High (★★★)', edgecolors='white'),
            plt.scatter([], [], c=colors['Medium'], s=100, label='Medium (★★)', edgecolors='white'),
            plt.scatter([], [], c=colors['Low'], s=100, label='Low (★)', edgecolors='white')
        ]
        fig.legend(handles=legend_elements, loc='lower center',
                  ncol=3, fontsize=FONT_SIZES['legend'], frameon=False,
                  bbox_to_anchor=(0.5, -0))

        plt.tight_layout(rect=[0, 0.08, 1, 1])

        if save_path is None:
            save_path = os.path.join(self.save_dir, 'feature_importance_correlation.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Feature importance correlation plot saved to: {save_path}")
        return save_path

    def generate_visualizations(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        生成所有可视化图表

        Args:
            results: 分析结果字典

        Returns:
            viz_paths: 图表路径字典
        """
        print("\nGenerating feature importance visualizations...")

        viz_paths = {}

        # 1. 水平条形图对比
        print("  - Creating feature importance comparison chart...")
        viz_paths['comparison'] = self.plot_feature_importance_comparison(results)

        # 2. 雷达图
        print("  - Creating radar chart...")
        viz_paths['radar'] = self.plot_importance_radar(results)

        # 3. 热力图
        print("  - Creating heatmap...")
        viz_paths['heatmap'] = self.plot_importance_heatmap(results)

        # 4. 相关性散点图
        print("  - Creating correlation scatter plot...")
        viz_paths['correlation'] = self.plot_importance_correlation(results)

        print("All visualizations generated successfully!")
        return viz_paths

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        生成分析报告

        Args:
            results: 分析结果字典

        Returns:
            report: Markdown 格式的报告
        """
        from datetime import datetime

        lines = []
        lines.append(f"# 领域知识一致性分析报告\n")
        lines.append(f"数据集: {self.feature_info.get('dataset_name', 'Unknown')}\n")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("="*80 + "\n\n")

        # 遍历每个模型
        for key, result in results.items():
            model_name = key.replace('_domain', '')
            lines.append(f"## {model_name}\n")

            # 性能指标
            metrics = result['metrics']
            lines.append("### 性能指标\n")
            lines.append("| 指标 | 值 |\n")
            lines.append("|------|------|\n")
            lines.append(f"| MSE | {metrics['mse']:.6f} |\n")
            lines.append(f"| MAE | {metrics['mae']:.6f} |\n")
            lines.append(f"| R² | {metrics['r2']:.4f} |\n\n")

            # 领域知识一致性
            shap_values = result['shap_values']
            consistency = self.compute_domain_consistency(shap_values)

            lines.append("### 领域知识一致性分析\n")
            lines.append(f"- **Spearman 相关系数**: {consistency['spearman_correlation']:.4f}")
            if consistency['spearman_p_value'] < 0.05:
                lines.append(" ( statistically significant, p < 0.05 )")
            lines.append("\n")
            lines.append(f"- **排序一致性**: {consistency['rank_consistency']:.1%}\n")
            for k, val in consistency.items():
                if k.startswith('top_'):
                    lines.append(f"- **Top {k[4:]} 一致性**: {val:.1%}\n")
            lines.append("\n")

            # 特征对比表格
            lines.append("### 特征重要性对比\n")
            lines.append("| 特征 | SHAP 重要性 | 标准差 | 领域重要性 | SHAP 排名 | 领域排名 |\n")
            lines.append("|------|-------------|--------|-----------|----------|----------|\n")

            comparison_data = self.generate_feature_comparison_table(shap_values)
            for item in sorted(comparison_data, key=lambda x: x['shap_importance'], reverse=True):
                domain_stars = {1: '★', 2: '★★', 3: '★★★'}.get(item['domain_importance'], '')
                lines.append(
                    f"| {item['feature']} | "
                    f"{item['shap_importance']:.4f} | "
                    f"{item['shap_std']:.4f} | "
                    f"{item['domain_label']} {domain_stars} | "
                    f"{item['shap_rank']} | "
                    f"{item['domain_rank']} |\n"
                )
            lines.append("\n")

        # 模型对比
        lines.append("## 模型对比\n")
        lines.append("| 模型 | MSE | R² | Spearman 相关系数 | 排序一致性 | Top-3 一致性 |\n")
        lines.append("|------|-----|-----|------------------|-----------|-------------|\n")

        for key, result in results.items():
            model_name = key.replace('_domain', '')
            metrics = result['metrics']
            shap_values = result['shap_values']

            consistency = self.compute_domain_consistency(shap_values)

            lines.append(
                f"| {model_name} | "
                f"{metrics['mse']:.4f} | "
                f"{metrics['r2']:.4f} | "
                f"{consistency['spearman_correlation']:.4f} | "
                f"{consistency['rank_consistency']:.1%} | "
                f"{consistency.get('top_3', 0):.1%} |\n"
            )

        lines.append("\n")
        lines.append("### 领域知识说明\n")
        lines.append("- **★★★ 高重要性**: 对目标变量有直接且重要的影响\n")
        lines.append("- **★★ 中重要性**: 对目标变量有间接影响\n")
        lines.append("- **★ 低重要性**: 对目标变量影响较小\n")

        # 生成可视化图表
        lines.append("\n## 可视化分析\n")
        viz_paths = self.generate_visualizations(results)
        for viz_name, viz_path in viz_paths.items():
            lines.append(f"- **{viz_name}**: `{viz_path}`\n")

        return ''.join(lines)

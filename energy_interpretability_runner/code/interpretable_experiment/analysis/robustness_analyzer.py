"""
鲁棒性分析器
评估模型在干扰特征下的表现
"""

from .base_analyzer import BaseAnalyzer
from typing import Dict, Any, Tuple
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader


class RobustnessAnalyzer(BaseAnalyzer):
    """鲁棒性分析器"""

    def __init__(self, config: Dict[str, Any], save_dir: str = 'results'):
        """
        Args:
            config: 配置字典
                - feature_info: 特征信息（从 data_generator.get_feature_info() 获得）
            save_dir: 结果保存目录
        """
        super().__init__(config, save_dir)

        self.feature_info = config['feature_info']
        self.true_indices = self.feature_info['true_feature_indices']
        self.noise_indices = self.feature_info['noise_feature_indices']
        self.redundant_indices = self.feature_info['redundant_feature_indices']
        self.deceptive_indices = self.feature_info['deceptive_feature_indices']

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

    def analyze_feature_suppression(self, shap_values: np.ndarray) -> Dict[str, float]:
        """
        分析干扰特征抑制率

        Args:
            shap_values: SHAP 值

        Returns:
            suppression_metrics: {
                'noise_suppression_ratio': 噪声抑制率
                'true_mean_importance': 真实特征平均重要性
                'noise_mean_importance': 噪声特征平均重要性
                'redundant_corr': 冗余特征与源特征的相关性
            }
        """
        # 计算平均重要性（绝对值）
        mean_importance = np.mean(np.abs(shap_values), axis=0)

        # 真实特征平均
        true_mean = np.mean(mean_importance[self.true_indices])

        # 噪声特征平均
        noise_mean = np.mean(mean_importance[self.noise_indices])

        # 抑制率
        suppression_ratio = noise_mean / true_mean if true_mean > 0 else float('inf')

        # 冗余特征分析
        redundant_corrs = []
        for i, redundant_idx in enumerate(self.redundant_indices):
            # 找到对应的真实特征索引
            source_true_idx = i % len(self.true_indices)
            source_idx = self.true_indices[source_true_idx]

            # 计算相关性
            corr = np.corrcoef(
                shap_values[:, redundant_idx],
                shap_values[:, source_idx]
            )[0, 1]
            redundant_corrs.append(corr)

        return {
            'noise_suppression_ratio': float(suppression_ratio),
            'true_mean_importance': float(true_mean),
            'noise_mean_importance': float(noise_mean),
            'redundant_corr': float(np.mean(redundant_corrs)) if redundant_corrs else 0.0,
            'redundant_mean_importance': float(np.mean(mean_importance[self.redundant_indices])) if self.redundant_indices else 0.0,
            'deceptive_mean_importance': float(np.mean(mean_importance[self.deceptive_indices])) if self.deceptive_indices else 0.0
        }

    def analyze_ranking_accuracy(self, shap_values: np.ndarray) -> Dict[str, float]:
        """
        分析特征重要性排序准确性

        Args:
            shap_values: SHAP 值

        Returns:
            ranking_metrics: {
                'true_features_in_top_k': 真实特征在 Top K 中的比例,
                'mean_true_rank': 真实特征的平均排名
            }
        """
        # 计算平均重要性
        mean_importance = np.mean(np.abs(shap_values), axis=0)

        # 排序（降序）
        ranking = np.argsort(-mean_importance)

        # 检查真实特征在 Top K 中的比例
        n_true = len(self.true_indices)
        top_k_values = [5, 10, min(20, len(mean_importance))]

        top_k_accuracy = {}
        for k in top_k_values:
            if k > len(ranking):
                continue
            top_k_indices = set(ranking[:k])
            true_in_top_k = len(top_k_indices.intersection(set(self.true_indices)))
            top_k_accuracy[f'top_{k}'] = true_in_top_k / n_true

        # 真实特征的平均排名
        true_ranks = []
        for true_idx in self.true_indices:
            rank = np.where(ranking == true_idx)[0]
            if len(rank) > 0:
                true_ranks.append(rank[0] + 1)

        if not true_ranks:
            return top_k_accuracy

        return {
            **top_k_accuracy,
            'mean_true_rank': float(np.mean(true_ranks)),
            'median_true_rank': float(np.median(true_ranks))
        }

    def generate_feature_importance_table(self, shap_values: np.ndarray) -> pd.DataFrame:
        """
        生成特征重要性表格

        Args:
            shap_values: SHAP 值

        Returns:
            df: 特征重要性表格
        """
        # 计算平均重要性
        mean_importance = np.mean(np.abs(shap_values), axis=0)

        # 获取特征信息
        feature_names = self.feature_info['feature_names']
        feature_types = self.feature_info['feature_types']

        # 创建 DataFrame
        df = pd.DataFrame({
            'Feature': feature_names,
            'Type': feature_types,
            'Mean_Importance': mean_importance,
            'Std_Importance': np.std(shap_values, axis=0)
        })

        # 添加排名
        df['Rank'] = df['Mean_Importance'].rank(ascending=False)

        # 按重要性排序
        df = df.sort_values('Mean_Importance', ascending=False)

        return df

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
        lines.append(f"# 鲁棒性分析报告\n")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("="*80 + "\n\n")

        # 遍历每个模型
        for key, result in results.items():
            model_name = key.replace('_robustness', '')
            lines.append(f"## {model_name}\n")

            # 性能指标
            metrics = result['metrics']
            lines.append("### 性能指标\n")
            lines.append("| 指标 | 值 |\n")
            lines.append("|------|------|\n")
            lines.append(f"| MSE | {metrics['mse']:.6f} |\n")
            lines.append(f"| MAE | {metrics['mae']:.6f} |\n")
            lines.append(f"| R² | {metrics['r2']:.4f} |\n\n")

            # 特征抑制分析
            shap_values = result['shap_values']
            suppression = self.analyze_feature_suppression(shap_values)

            lines.append("### 干扰特征抑制分析\n")
            lines.append(f"- **噪声抑制率**: {suppression['noise_suppression_ratio']:.3f}\n")
            lines.append(f"  - 真实特征平均重要性: {suppression['true_mean_importance']:.4f}\n")
            lines.append(f"  - 噪声特征平均重要性: {suppression['noise_mean_importance']:.4f}\n")
            lines.append(f"  - 冗余特征平均重要性: {suppression['redundant_mean_importance']:.4f}\n")
            lines.append(f"  - 冗余特征与源特征相关性: {suppression['redundant_corr']:.3f}\n")
            lines.append(f"  - 欺骗特征平均重要性: {suppression['deceptive_mean_importance']:.4f}\n\n")

            # 排序准确性
            ranking = self.analyze_ranking_accuracy(shap_values)
            lines.append("### 特征重要性排序准确性\n")
            lines.append(f"- **真实特征平均排名**: {ranking.get('mean_true_rank', 'N/A'):.1f}\n")
            lines.append(f"- **真实特征中位数排名**: {ranking.get('median_true_rank', 'N/A'):.1f}\n")
            for k, acc in ranking.items():
                if k.startswith('top_'):
                    lines.append(f"- **Top {k[4:]} 准确率**: {acc:.1%}\n")
            lines.append("\n")

            # 特征重要性表格
            lines.append("### 特征重要性详情\n")
            df = self.generate_feature_importance_table(shap_values)
            lines.append(df.to_markdown(index=False))
            lines.append("\n\n")

        # 对比总结
        lines.append("## 模型对比\n")
        lines.append("| 模型 | MSE | MAE | R² | 噪声抑制率 | Top-5 准确率 | 平均真实排名 |\n")
        lines.append("|------|-----|-----|-----|-----------|-------------|-------------|\n")

        for key, result in results.items():
            model_name = key.replace('_robustness', '')
            metrics = result['metrics']
            shap_values = result['shap_values']

            suppression = self.analyze_feature_suppression(shap_values)
            ranking = self.analyze_ranking_accuracy(shap_values)

            lines.append(
                f"| {model_name} | {metrics['mse']:.4f} | "
                f"{metrics['mae']:.4f} | {metrics['r2']:.4f} | "
                f"{suppression['noise_suppression_ratio']:.3f} | "
                f"{ranking.get('top_5', 0):.1%} | "
                f"{ranking.get('mean_true_rank', 0):.1f} |\n"
            )

        return ''.join(lines)

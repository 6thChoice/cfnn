"""
因果分析器
评估模型特征重要性与真实因果结构的一致性
"""

from .base_analyzer import BaseAnalyzer
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader


class CausalAnalyzer(BaseAnalyzer):
    """因果分析器"""

    def __init__(self, config: Dict[str, Any], save_dir: str = 'results'):
        """
        Args:
            config: 配置字典
                - feature_info: 特征信息（包含真实因果顺序）
            save_dir: 结果保存目录
        """
        super().__init__(config, save_dir)

        self.feature_info = config['feature_info']
        self.true_causal_order = self.feature_info['true_causal_order']
        self.graph_type = self.feature_info['graph_type']

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
        model.eval()
        device = next(model.parameters()).device

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        all_preds = np.concatenate(all_preds).flatten()
        all_targets = np.concatenate(all_targets).flatten()

        # 计算指标
        mse = np.mean((all_preds - all_targets) ** 2)
        mae = np.mean(np.abs(all_preds - all_targets))

        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        return {
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2)
        }

    def analyze_shap(self, model: torch.nn.Module,
                    train_loader: DataLoader,
                    test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算SHAP值

        Args:
            model: 训练好的模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器

        Returns:
            shap_values: SHAP值
            test_data: 测试数据
        """
        # 获取背景数据
        background_data = []
        for X_batch, _ in train_loader:
            background_data.append(X_batch.numpy())
            if len(background_data) >= 10:
                break
        background = np.concatenate(background_data)[:100]

        # 获取测试数据
        test_data_list = []
        for X_batch, _ in test_loader:
            test_data_list.append(X_batch.numpy())
        test_data = np.concatenate(test_data_list)

        # 创建预测函数
        device = next(model.parameters()).device
        def predict_fn(x):
            if isinstance(x, np.ndarray):
                x_tensor = torch.from_numpy(x).float()
            else:
                x_tensor = x
            x_tensor = x_tensor.to(device)
            with torch.no_grad():
                output = model(x_tensor)
            return output.cpu().numpy()

        # 使用 PermutationExplainer
        import shap
        explainer = shap.explainers.PermutationExplainer(
            predict_fn,
            background,
            max_evals=500
        )

        # 计算SHAP值
        max_samples = min(50, len(test_data))
        shap_values = explainer.shap_values(test_data[:max_samples])

        if shap_values.ndim == 3:
            shap_values = shap_values.squeeze(-1)

        return shap_values, test_data[:max_samples]

    def evaluate_causal_detection(self, shap_values: np.ndarray) -> Dict[str, Any]:
        """
        评估因果特征检测准确性

        Args:
            shap_values: SHAP值

        Returns:
            detection_metrics: 检测准确性指标
        """
        # 计算平均重要性
        mean_importance = np.mean(np.abs(shap_values), axis=0)

        # 获取SHAP排名
        shap_ranking = np.argsort(-mean_importance)
        shap_ranking_names = [self.feature_info['feature_names'][i] for i in shap_ranking]

        # 获取真实因果排名
        true_ranking_names = self.true_causal_order

        # Top-K 准确率
        top_k_accuracies = {}
        for k in [1, 2, 3]:
            top_k_predicted = set(shap_ranking_names[:k])
            top_k_true = set(true_ranking_names[:k])
            accuracy = len(top_k_predicted & top_k_true) / k
            top_k_accuracies[f'top_{k}_accuracy'] = float(accuracy)

        # Spearman相关系数
        from scipy.stats import spearmanr
        # 将特征名转换为排名索引
        name_to_idx = {name: i for i, name in enumerate(self.feature_info['feature_names'])}
        true_ranks = [name_to_idx[name] for name in true_ranking_names]
        shap_ranks = list(shap_ranking)

        spearman_corr, _ = spearmanr(true_ranks, shap_ranks)

        return {
            **top_k_accuracies,
            'spearman_correlation': float(spearman_corr),
            'mean_importance': mean_importance.tolist(),
            'shap_ranking': shap_ranking_names,
            'true_ranking': true_ranking_names
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        生成因果分析报告

        Args:
            results: 分析结果字典

        Returns:
            report: Markdown格式的报告
        """
        from datetime import datetime

        lines = []
        lines.append(f"# 因果结构发现分析报告\n")
        lines.append(f"因果图类型: {self.graph_type}\n")
        lines.append(f"真实因果结构: {self.feature_info['description']}\n")
        lines.append(f"生成时间: {datetime.now()}\n")
        lines.append("="*80 + "\n\n")

        # 遍历每个模型
        for key, result in results.items():
            model_name = key.replace('_causal', '')
            lines.append(f"## {model_name}\n")

            # 性能指标
            metrics = result['metrics']
            lines.append("### 性能指标\n")
            lines.append("| 指标 | 值 |\n")
            lines.append("|------|------|\n")
            lines.append(f"| MSE | {metrics['mse']:.6f} |\n")
            lines.append(f"| MAE | {metrics['mae']:.6f} |\n")
            lines.append(f"| R² | {metrics['r2']:.4f} |\n\n")

            # 因果检测准确性
            detection = result.get('causal_detection', {})
            if 'top_1_accuracy' in detection:
                lines.append("### 因果特征检测准确性\n")
                lines.append(f"- **Top-1 准确率**: {detection['top_1_accuracy']:.1%}\n")
                lines.append(f"- **Top-2 准确率**: {detection['top_2_accuracy']:.1%}\n")
                lines.append(f"- **Top-3 准确率**: {detection['top_3_accuracy']:.1%}\n")
                lines.append(f"- **Spearman相关性**: {detection['spearman_correlation']:.4f}\n\n")

            # 排名对比
            if 'shap_ranking' in detection:
                lines.append("### 因果排名对比\n")
                lines.append("| 排名 | SHAP排名 | 真实因果排名 |\n")
                lines.append("|------|----------|-------------|\n")
                for i in range(len(detection['shap_ranking'])):
                    shap_name = detection['shap_ranking'][i]
                    true_name = detection['true_ranking'][i] if i < len(detection['true_ranking']) else '-'
                    match = "✓" if shap_name == true_name else ""
                    lines.append(f"| {i+1} | {shap_name} | {true_name} {match} |\n")
                lines.append("\n")

            # 特征重要性
            if 'mean_importance' in detection:
                lines.append("### 特征重要性\n")
                lines.append("| 特征 | 平均重要性 |\n")
                lines.append("|------|-----------|\n")
                for i, name in enumerate(self.feature_info['feature_names']):
                    importance = detection['mean_importance'][i]
                    lines.append(f"| {name} | {importance:.4f} |\n")
                lines.append("\n")

        # 模型对比
        lines.append("## 模型对比\n")
        lines.append("| 模型 | MSE | R² | Top-1准确率 | Top-3准确率 | Spearman相关性 |\n")
        lines.append("|------|-----|-----|-----------|-----------|---------------|\n")

        for key, result in results.items():
            model_name = key.replace('_causal', '')
            metrics = result['metrics']
            detection = result.get('causal_detection', {})

            lines.append(
                f"| {model_name} | {metrics['mse']:.4f} | "
                f"{metrics['r2']:.4f} | "
                f"{detection.get('top_1_accuracy', 0):.1%} | "
                f"{detection.get('top_3_accuracy', 0):.1%} | "
                f"{detection.get('spearman_correlation', 0):.4f} |\n"
            )

        return ''.join(lines)

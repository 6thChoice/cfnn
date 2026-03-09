"""
特征交互分析器
计算SHAP交互值并分析特征交互效应
"""

from .base_analyzer import BaseAnalyzer
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


class InteractionAnalyzer(BaseAnalyzer):
    """特征交互分析器"""

    def __init__(self, config: Dict[str, Any], save_dir: str = 'results'):
        """
        Args:
            config: 配置字典
                - feature_info: 特征信息
            save_dir: 结果保存目录
        """
        super().__init__(config, save_dir)

        self.feature_info = config['feature_info']
        self.interaction_type = self.feature_info['interaction_type']
        self.expected_interactions = self.feature_info['interaction_features']

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
        计算SHAP值（标准）

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

    def analyze_interaction_strength(self, shap_values: np.ndarray) -> Dict[str, float]:
        """
        分析交互强度

        通过观察SHAP值的非线性模式推断交互

        Args:
            shap_values: SHAP值

        Returns:
            interaction_metrics: 交互强度指标
        """
        # 计算SHAP值的方差（高方差可能表明交互效应）
        shap_var = np.var(shap_values, axis=0)

        # 计算SHAP值的绝对值平均
        shap_abs_mean = np.mean(np.abs(shap_values), axis=0)

        return {
            'shap_variance': shap_var.tolist(),
            'shap_abs_mean': shap_abs_mean.tolist(),
            'total_variance': float(np.sum(shap_var)),
            'max_variance': float(np.max(shap_var))
        }

    def detect_top_interactions(self, shap_values: np.ndarray,
                                test_data: np.ndarray,
                                k: int = 3) -> List[Dict[str, Any]]:
        """
        检测Top K交互对

        基于特征相关性推断交互

        Args:
            shap_values: SHAP值
            test_data: 测试数据
            k: Top K数量

        Returns:
            top_interactions: Top K交互对列表
        """
        # 计算SHAP值与特征值的相关性
        # 如果一个特征的SHAP值依赖于另一个特征的值，则存在交互
        interactions = []

        n_features = shap_values.shape[1]

        for i in range(n_features):
            for j in range(i+1, n_features):
                # 计算特征i的SHAP值与特征j的值的相关性
                correlation = np.corrcoef(np.abs(shap_values[:, i]), test_data[:, j])[0, 1]

                interactions.append({
                    'feature_pair': (f'x{i+1}', f'x{j+1}'),
                    'correlation': float(correlation),
                    'abs_correlation': float(abs(correlation))
                })

        # 按相关性绝对值排序
        interactions.sort(key=lambda x: x['abs_correlation'], reverse=True)

        return interactions[:k]

    def evaluate_interaction_detection(self, shap_values: np.ndarray,
                                      test_data: np.ndarray) -> Dict[str, float]:
        """
        评估交互检测准确性

        Args:
            shap_values: SHAP值
            test_data: 测试数据

        Returns:
            detection_metrics: 检测准确性指标
        """
        # 检测Top K交互对
        k = min(3, len(self.expected_interactions))
        detected_interactions = self.detect_top_interactions(shap_values, test_data, k)

        # 转换预期交互为集合
        expected_set = set()
        for inter in self.expected_interactions:
            if isinstance(inter, tuple):
                if len(inter) == 2:
                    expected_set.add(tuple(sorted(inter)))
                elif len(inter) > 2:
                    # 对于高阶交互，检查所有两两组合
                    for i in range(len(inter)):
                        for j in range(i+1, len(inter)):
                            expected_set.add(tuple(sorted([inter[i], inter[j]])))

        # 转换检测到的交互为集合
        detected_set = set()
        for inter in detected_interactions:
            detected_set.add(tuple(sorted(inter['feature_pair'])))

        # 计算召回率和精确率
        true_positives = len(expected_set & detected_set)
        recall = true_positives / len(expected_set) if len(expected_set) > 0 else 0
        precision = true_positives / k if k > 0 else 0

        return {
            'recall': float(recall),
            'precision': float(precision),
            'f1_score': float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0,
            'true_positives': int(true_positives),
            'top_interactions': detected_interactions
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        生成交互分析报告

        Args:
            results: 分析结果字典

        Returns:
            report: Markdown格式的报告
        """
        from datetime import datetime

        lines = []
        lines.append(f"# 特征交互分析报告\n")
        lines.append(f"交互类型: {self.interaction_type}\n")
        lines.append(f"生成时间: {datetime.now()}\n")
        lines.append("="*80 + "\n\n")

        # 遍历每个模型
        for key, result in results.items():
            model_name = key.replace('_interaction', '')
            lines.append(f"## {model_name}\n")

            # 性能指标
            metrics = result['metrics']
            lines.append("### 性能指标\n")
            lines.append("| 指标 | 值 |\n")
            lines.append("|------|------|\n")
            lines.append(f"| MSE | {metrics['mse']:.6f} |\n")
            lines.append(f"| MAE | {metrics['mae']:.6f} |\n")
            lines.append(f"| R² | {metrics['r2']:.4f} |\n\n")

            # 交互强度分析
            interaction_strength = result.get('interaction_strength', {})
            lines.append("### 交互强度分析\n")
            lines.append(f"- **总方差**: {interaction_strength.get('total_variance', 0):.4f}\n")
            lines.append(f"- **最大方差**: {interaction_strength.get('max_variance', 0):.4f}\n\n")

            # SHAP值统计
            if 'shap_abs_mean' in interaction_strength:
                lines.append("### SHAP值统计\n")
                lines.append("| 特征 | 平均绝对SHAP值 | 方差 |\n")
                lines.append("|------|---------------|-----|\n")
                for i, (mean_val, var) in enumerate(zip(
                    interaction_strength['shap_abs_mean'],
                    interaction_strength['shap_variance']
                )):
                    lines.append(f"| x{i+1} | {mean_val:.4f} | {var:.4f} |\n")
                lines.append("\n")

            # 交互检测
            detection = result.get('detection_metrics', {})
            if 'top_interactions' in detection:
                lines.append("### Top交互对\n")
                lines.append("| 排名 | 交互对 | 相关性 |\n")
                lines.append("|------|--------|--------|\n")
                for i, inter in enumerate(detection['top_interactions']):
                    pair_str = ', '.join(inter['feature_pair'])
                    lines.append(f"| {i+1} | {pair_str} | {inter['correlation']:.4f} |\n")
                lines.append("\n")

            # 检测准确性
            if 'recall' in detection:
                lines.append("### 交互检测准确性\n")
                lines.append(f"- **召回率**: {detection['recall']:.1%}\n")
                lines.append(f"- **精确率**: {detection['precision']:.1%}\n")
                lines.append(f"- **F1分数**: {detection['f1_score']:.4f}\n")
                lines.append(f"- **真正例**: {detection['true_positives']}\n\n")

        # 模型对比
        lines.append("## 模型对比\n")
        lines.append("| 模型 | MSE | R² | 交互检测F1 |\n")
        lines.append("|------|-----|-----|------------|\n")

        for key, result in results.items():
            model_name = key.replace('_interaction', '')
            metrics = result['metrics']
            detection = result.get('detection_metrics', {})
            f1_score = detection.get('f1_score', 0)

            lines.append(
                f"| {model_name} | {metrics['mse']:.4f} | "
                f"{metrics['r2']:.4f} | {f1_score:.4f} |\n"
            )

        return ''.join(lines)

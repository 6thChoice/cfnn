"""
分布外分析器
评估模型在OOD场景下的表现和特征重要性稳定性
"""

from .base_analyzer import BaseAnalyzer
from typing import Dict, Any, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader


class OODAnalyzer(BaseAnalyzer):
    """分布外分析器"""

    def __init__(self, config: Dict[str, Any], save_dir: str = 'results'):
        """
        Args:
            config: 配置字典
                - feature_info: 特征信息（包含OOD类型）
            save_dir: 结果保存目录
        """
        super().__init__(config, save_dir)

        self.feature_info = config['feature_info']
        self.ood_type = self.feature_info['ood_type']

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

    def compute_importance_stability(self, shap_train: np.ndarray,
                                    shap_test: np.ndarray) -> Dict[str, float]:
        """
        计算特征重要性稳定性

        Args:
            shap_train: 训练集的SHAP值
            shap_test: 测试集（OOD）的SHAP值

        Returns:
            stability_metrics: 稳定性指标
        """
        # 计算平均重要性
        mean_importance_train = np.mean(np.abs(shap_train), axis=0)
        mean_importance_test = np.mean(np.abs(shap_test), axis=0)

        # Spearman相关系数
        from scipy.stats import spearmanr
        spearman_corr, _ = spearmanr(mean_importance_train, mean_importance_test)

        # Pearson相关系数
        pearson_corr = np.corrcoef(mean_importance_train, mean_importance_test)[0, 1]

        # 排序稳定性（排名相关性）
        ranking_train = np.argsort(-mean_importance_train)
        ranking_test = np.argsort(-mean_importance_test)
        ranking_corr, _ = spearmanr(ranking_train, ranking_test)

        # 计算每个特征的变化率
        change_rate = np.abs(mean_importance_test - mean_importance_train) / (np.abs(mean_importance_train) + 1e-8)

        return {
            'spearman_corr': float(spearman_corr),
            'pearson_corr': float(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
            'ranking_corr': float(ranking_corr),
            'mean_change_rate': float(np.mean(change_rate)),
            'max_change_rate': float(np.max(change_rate)),
            'mean_importance_train': mean_importance_train.tolist(),
            'mean_importance_test': mean_importance_test.tolist()
        }

    def compute_performance_drop(self, metrics_train: Dict[str, float],
                                metrics_test: Dict[str, float]) -> Dict[str, float]:
        """
        计算性能下降

        Args:
            metrics_train: 训练集性能指标
            metrics_test: 测试集性能指标

        Returns:
            drop_metrics: 性能下降指标
        """
        mse_train = metrics_train['mse']
        mse_test = metrics_test['mse']

        r2_train = metrics_train['r2']
        r2_test = metrics_test['r2']

        # 绝对下降
        mse_drop = mse_test - mse_train
        r2_drop = r2_train - r2_test

        # 相对下降
        mse_relative_drop = mse_drop / mse_train if mse_train > 0 else float('inf')

        return {
            'mse_drop': float(mse_drop),
            'mse_relative_drop': float(mse_relative_drop),
            'r2_drop': float(r2_drop),
            'mse_train': float(mse_train),
            'mse_test': float(mse_test),
            'r2_train': float(r2_train),
            'r2_test': float(r2_test)
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        生成OOD分析报告

        Args:
            results: 分析结果字典

        Returns:
            report: Markdown格式的报告
        """
        from datetime import datetime

        lines = []
        lines.append(f"# 分布外泛化稳定性分析报告\n")
        lines.append(f"OOD类型: {self.ood_type}\n")
        lines.append(f"生成时间: {datetime.now()}\n")
        lines.append("="*80 + "\n\n")

        # 遍历每个模型
        for key, result in results.items():
            model_name = key.replace('_ood', '')
            lines.append(f"## {model_name}\n")

            # 性能指标
            metrics = result['metrics']
            lines.append("### 性能指标（OOD测试集）\n")
            lines.append("| 指标 | 值 |\n")
            lines.append("|------|------|\n")
            lines.append(f"| MSE | {metrics['mse']:.6f} |\n")
            lines.append(f"| MAE | {metrics['mae']:.6f} |\n")
            lines.append(f"| R² | {metrics['r2']:.4f} |\n\n")

            # 性能下降
            perf_drop = result.get('performance_drop', {})
            if 'mse_drop' in perf_drop:
                lines.append("### 性能下降（训练集 → OOD测试集）\n")
                lines.append(f"- **MSE 下降**: {perf_drop['mse_drop']:.6f}\n")
                lines.append(f"- **MSE 相对下降**: {perf_drop['mse_relative_drop']:.1%}\n")
                lines.append(f"- **R² 下降**: {perf_drop['r2_drop']:.4f}\n\n")

            # 特征重要性稳定性
            stability = result.get('stability', {})
            if 'spearman_corr' in stability:
                lines.append("### 特征重要性稳定性\n")
                lines.append(f"- **Spearman相关系数**: {stability['spearman_corr']:.4f}\n")
                lines.append(f"- **Pearson相关系数**: {stability['pearson_corr']:.4f}\n")
                lines.append(f"- **排名相关性**: {stability['ranking_corr']:.4f}\n")
                lines.append(f"- **平均变化率**: {stability['mean_change_rate']:.1%}\n")
                lines.append(f"- **最大变化率**: {stability['max_change_rate']:.1%}\n\n")

            # 特征重要性对比
            if 'mean_importance_train' in stability:
                lines.append("### 特征重要性对比\n")
                lines.append("| 特征 | 训练集 | 测试集 | 变化率 |\n")
                lines.append("|------|--------|--------|--------|\n")
                for i in range(len(stability['mean_importance_train'])):
                    train_val = stability['mean_importance_train'][i]
                    test_val = stability['mean_importance_test'][i]
                    change = abs(test_val - train_val) / (abs(train_val) + 1e-8)
                    lines.append(f"| x{i+1} | {train_val:.4f} | {test_val:.4f} | {change:.1%} |\n")
                lines.append("\n")

        # 模型对比
        lines.append("## 模型对比\n")
        lines.append("| 模型 | MSE(OOD) | R²(OOD) | Spearman稳定性 | MSE相对下降 |\n")
        lines.append("|------|----------|---------|----------------|------------|\n")

        for key, result in results.items():
            model_name = key.replace('_ood', '')
            metrics = result['metrics']
            stability = result.get('stability', {})
            perf_drop = result.get('performance_drop', {})

            lines.append(
                f"| {model_name} | {metrics['mse']:.4f} | "
                f"{metrics['r2']:.4f} | "
                f"{stability.get('spearman_corr', 0):.4f} | "
                f"{perf_drop.get('mse_relative_drop', 0):.1%} |\n"
            )

        return ''.join(lines)

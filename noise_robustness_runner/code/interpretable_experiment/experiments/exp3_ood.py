"""
方案3：分布外泛化稳定性实验
"""

from .base_experiment import BaseExperiment
from ..data.generate_ood_data import OODDataGenerator
from ..analysis.ood_analyzer import OODAnalyzer
from ..experiments.exp1_robustness import ModelWrapper
import sys
import os
import torch
from typing import Dict, Any

# 导入 CFNet 模型
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from cfnet import CFNet_Standard, HybridRationalNet, MoE_Ensemble
from func.models_ext import get_aligned_mlp, count_parameters


class OODExperiment(BaseExperiment):
    """分布外泛化实验"""

    def setup_data(self):
        """设置数据生成器"""
        from ..config.exp3_config import get_config
        config = get_config()

        # 创建数据生成器
        data_generator = OODDataGenerator(config['data'])

        # 创建数据加载器
        train_loader, val_loader, test_loader = data_generator.create_dataloaders()

        self.data_generator = {
            'generator': data_generator,
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'feature_info': data_generator.get_feature_info()
        }

        self.logger.info(f"Data setup complete:")
        self.logger.info(f"  - OOD type: {data_generator.ood_type}")
        self.logger.info(f"  - Features: {data_generator.X_train.shape[1]}")
        self.logger.info(f"  - Train distribution: {data_generator.train_dist_config}")
        self.logger.info(f"  - Test distribution: {data_generator.test_dist_config}")

    def setup_models(self):
        """设置模型"""
        from ..config.exp3_config import get_config
        config = get_config()
        device = config['device']

        n_features = self.data_generator['generator'].X_train.shape[1]

        # 以 Standard CFNet 为锚点
        anchor = CFNet_Standard(n_features, 1, **config['models']['standard'])
        target_params = count_parameters(anchor)

        # 创建所有模型（排除Boost和Hybrid，简化实验）
        self.models = {
            'Standard': CFNet_Standard(
                n_features,
                1,
                **config['models']['standard']
            ),
            'MoE': MoE_Ensemble({
                'input_dim': n_features,
                'output_dim': 1,
                'shallow_depth_per_cofrnet': config['models']['moe']['shallow_depth_per_cofrnet'],
                'polynomial_degree': config['models']['moe']['polynomial_degree']
            })
        }

        # 添加MLP
        mlp, _ = get_aligned_mlp(n_features, 1, target_params)
        self.models['MLP'] = mlp

        # 初始化 MoE 专家
        for _ in range(config['models']['moe']['num_experts']):
            self.models['MoE'].add_expert()

        # 移动到设备并包装
        for name, model in self.models.items():
            self.models[name] = ModelWrapper(model).to(device)

        self.logger.info(f"Models setup complete: {list(self.models.keys())}")

    def setup_analyzers(self):
        """设置分析器"""
        # 创建OOD分析器
        analyzer_config = {
            **self.config,
            'feature_info': self.data_generator['feature_info']
        }

        self.analyzers = {
            'ood': OODAnalyzer(analyzer_config, self.result_dir)
        }

        self.logger.info(f"Analyzers setup complete: {list(self.analyzers.keys())}")

    def analyze(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析实验结果（重写以添加稳定性分析）

        Args:
            training_results: 训练结果

        Returns:
            analysis_results: 分析结果
        """
        analysis_results = {}

        for model_name in self.models.keys():
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Analyzing {model_name}...")
            self.logger.info(f"{'='*60}\n")

            # 加载模型
            model = self.models[model_name]
            checkpoint = torch.load(
                training_results[model_name]['model_path'],
                map_location=self.config['device']
            )
            model.load_state_dict(checkpoint)
            model.eval()

            # 运行分析器
            for analyzer_name, analyzer in self.analyzers.items():
                self.logger.info(f"  Running {analyzer_name}...")

                # 计算测试集（OOD）指标
                metrics_test = analyzer.compute_metrics(
                    model,
                    self.data_generator['test']
                )

                # 计算训练集指标（用于性能下降分析）
                metrics_train = analyzer.compute_metrics(
                    model,
                    self.data_generator['train']
                )

                # 计算SHAP值（测试集）
                shap_test, test_data = analyzer.analyze_shap(
                    model,
                    self.data_generator['train'],
                    self.data_generator['test']
                )

                # 计算SHAP值（训练集，用于稳定性分析）
                shap_train, _ = analyzer.analyze_shap(
                    model,
                    self.data_generator['train'],
                    self.data_generator['train']
                )

                # 计算稳定性
                stability = analyzer.compute_importance_stability(shap_train, shap_test)

                # 计算性能下降
                performance_drop = analyzer.compute_performance_drop(metrics_train, metrics_test)

                # 保存结果
                result_key = f"{model_name}_{analyzer_name}"
                analysis_results[result_key] = {
                    'metrics': metrics_test,
                    'shap_values': shap_test,
                    'test_data': test_data,
                    'stability': stability,
                    'performance_drop': performance_drop,
                    'metrics_train': metrics_train
                }

                # 保存到文件
                import pickle
                save_path = os.path.join(self.result_dir, 'metrics', f'{result_key}.pkl')
                with open(save_path, 'wb') as f:
                    pickle.dump(analysis_results[result_key], f)

        return analysis_results


# 运行函数
def run_ood_experiment(ood_type: str = 'covariate_shift'):
    """
    运行OOD泛化实验

    Args:
        ood_type: OOD类型
    """
    from ..config.exp3_config import get_config

    # 加载配置
    config = get_config(ood_type)

    # 创建实验
    experiment = OODExperiment(config, f'exp3_ood_{ood_type}')

    # 运行实验
    results = experiment.run()

    return results


if __name__ == '__main__':
    # 运行协变量偏移实验
    results = run_ood_experiment(ood_type='covariate_shift')

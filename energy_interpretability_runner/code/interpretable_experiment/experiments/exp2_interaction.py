"""
方案2：特征交互捕获实验
"""

from .base_experiment import BaseExperiment
from ..data.generate_interaction_data import InteractionDataGenerator
from ..analysis.interaction_analyzer import InteractionAnalyzer
from ..experiments.exp1_robustness import ModelWrapper
import sys
import os
import torch
from typing import Dict, Any

# 导入 CFNet 模型
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from cfnet import CFNet_Standard, HybridRationalNet, EnsembleResCoFrNet, MoE_Ensemble
from func.models_ext import get_aligned_mlp, count_parameters


class InteractionExperiment(BaseExperiment):
    """特征交互捕获实验"""

    def setup_data(self):
        """设置数据生成器"""
        from ..config.exp2_config import get_config
        config = get_config()

        # 创建数据生成器
        data_generator = InteractionDataGenerator(config['data'])

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
        self.logger.info(f"  - Interaction type: {data_generator.interaction_type}")
        self.logger.info(f"  - Features: {data_generator.n_features}")
        self.logger.info(f"  - Interaction strength: {data_generator.interaction_strength}")

    def setup_models(self):
        """设置模型"""
        from ..config.exp2_config import get_config
        config = get_config()
        device = config['device']

        n_features = self.data_generator['generator'].n_features

        # 以 Standard CFNet 为锚点
        anchor = CFNet_Standard(n_features, 1, **config['models']['standard'])
        target_params = count_parameters(anchor)

        # 创建所有模型（排除Boost，因为训练不稳定）
        self.models = {
            'Standard': CFNet_Standard(
                n_features,
                1,
                **config['models']['standard']
            ),
            'Hybrid': HybridRationalNet(
                n_features,
                1,
                **config['models']['hybrid']
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
        # 创建交互分析器
        analyzer_config = {
            **self.config,
            'feature_info': self.data_generator['feature_info']
        }

        self.analyzers = {
            'interaction': InteractionAnalyzer(analyzer_config, self.result_dir)
        }

        self.logger.info(f"Analyzers setup complete: {list(self.analyzers.keys())}")

    def analyze(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析实验结果（重写以添加交互强度分析）

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

                # 计算指标
                metrics = analyzer.compute_metrics(
                    model,
                    self.data_generator['test']
                )

                # 计算SHAP
                shap_values, test_data = analyzer.analyze_shap(
                    model,
                    self.data_generator['train'],
                    self.data_generator['test']
                )

                # 计算交互强度
                interaction_strength = analyzer.analyze_interaction_strength(shap_values)

                # 评估交互检测
                detection_metrics = analyzer.evaluate_interaction_detection(shap_values, test_data)

                # 保存结果
                result_key = f"{model_name}_{analyzer_name}"
                analysis_results[result_key] = {
                    'metrics': metrics,
                    'shap_values': shap_values,
                    'test_data': test_data,
                    'interaction_strength': interaction_strength,
                    'detection_metrics': detection_metrics
                }

                # 保存到文件
                import pickle
                save_path = os.path.join(self.result_dir, 'metrics', f'{result_key}.pkl')
                with open(save_path, 'wb') as f:
                    pickle.dump(analysis_results[result_key], f)

        return analysis_results


# 运行函数
def run_interaction_experiment(interaction_type: str = 'rational'):
    """
    运行特征交互捕获实验

    Args:
        interaction_type: 交互类型（默认'rational'，CFNet优势场景）
    """
    import torch
    from ..config.exp2_config import get_config

    # 加载配置
    config = get_config(interaction_type)

    # 创建实验
    experiment = InteractionExperiment(config, f'exp2_interaction_{interaction_type}')

    # 运行实验
    results = experiment.run()

    return results


if __name__ == '__main__':
    import torch
    # 运行分式交互实验（CFNet优势场景）
    results = run_interaction_experiment(interaction_type='rational')

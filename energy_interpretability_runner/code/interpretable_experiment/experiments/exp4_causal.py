"""
方案4：因果结构发现实验
"""

from .base_experiment import BaseExperiment
from ..data.generate_causal_data import CausalDataGenerator
from ..analysis.causal_analyzer import CausalAnalyzer
from ..experiments.exp1_robustness import ModelWrapper
import sys
import os
import torch
from typing import Dict, Any

# 导入 CFNet 模型
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from cfnet import CFNet_Standard, HybridRationalNet, MoE_Ensemble
from func.models_ext import get_aligned_mlp, count_parameters


class CausalExperiment(BaseExperiment):
    """因果结构发现实验"""

    def setup_data(self):
        """设置数据生成器"""
        from ..config.exp4_config import get_config
        config = get_config()

        # 创建数据生成器
        data_generator = CausalDataGenerator(config['data'])

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
        self.logger.info(f"  - Graph type: {data_generator.graph_type}")
        self.logger.info(f"  - Features: {data_generator.get_feature_info()['n_features']}")
        self.logger.info(f"  - True causal order: {data_generator.get_true_causal_order()}")

    def setup_models(self):
        """设置模型"""
        from ..config.exp4_config import get_config
        config = get_config()
        device = config['device']

        n_features = self.data_generator['feature_info']['n_features']

        # 以 Standard CFNet 为锚点
        anchor = CFNet_Standard(n_features, 1, **config['models']['standard'])
        target_params = count_parameters(anchor)

        # 创建所有模型（简化）
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
            )
        }

        # 添加MLP
        mlp, _ = get_aligned_mlp(n_features, 1, target_params)
        self.models['MLP'] = mlp

        # 移动到设备并包装
        for name, model in self.models.items():
            self.models[name] = ModelWrapper(model).to(device)

        self.logger.info(f"Models setup complete: {list(self.models.keys())}")

    def setup_analyzers(self):
        """设置分析器"""
        # 创建因果分析器
        analyzer_config = {
            **self.config,
            'feature_info': self.data_generator['feature_info']
        }

        self.analyzers = {
            'causal': CausalAnalyzer(analyzer_config, self.result_dir)
        }

        self.logger.info(f"Analyzers setup complete: {list(self.analyzers.keys())}")

    def analyze(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析实验结果（重写以添加因果检测分析）

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

                # 评估因果检测
                causal_detection = analyzer.evaluate_causal_detection(shap_values)

                # 保存结果
                result_key = f"{model_name}_{analyzer_name}"
                analysis_results[result_key] = {
                    'metrics': metrics,
                    'shap_values': shap_values,
                    'test_data': test_data,
                    'causal_detection': causal_detection
                }

                # 保存到文件
                import pickle
                save_path = os.path.join(self.result_dir, 'metrics', f'{result_key}.pkl')
                with open(save_path, 'wb') as f:
                    pickle.dump(analysis_results[result_key], f)

        return analysis_results


# 运行函数
def run_causal_experiment(graph_type: str = 'rational'):
    """
    运行因果发现实验

    Args:
        graph_type: 因果图类型
    """
    from ..config.exp4_config import get_config

    # 加载配置
    config = get_config(graph_type)

    # 创建实验
    experiment = CausalExperiment(config, f'exp4_causal_{graph_type}')

    # 运行实验
    results = experiment.run()

    return results


if __name__ == '__main__':
    # 运行分式因果实验（CFNet优势场景）
    results = run_causal_experiment(graph_type='rational')

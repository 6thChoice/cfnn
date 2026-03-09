"""
方案 1：特征干扰鲁棒性实验
"""

from .base_experiment import BaseExperiment
from ..data.generate_noisy_features import NoisyFeaturesGenerator
from ..analysis.robustness_analyzer import RobustnessAnalyzer
import sys
import os
import torch
import torch.nn as nn

# 将父目录添加到路径以导入 cfnet 和 func
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from cfnet import (
    CFNet_Standard,
    HybridRationalNet,
    EnsembleResCoFrNet,
    MoE_Ensemble
)
from func.models_ext import get_aligned_mlp, count_parameters


class ModelWrapper(nn.Module):
    """模型包装器，确保输出维度正确并转发特殊方法"""
    def __init__(self, model):
        super().__init__()
        # 将模型存储为一个普通的属性，但使用字符串键避免 PyTorch 自动注册
        object.__setattr__(self, '_model', model)

    @property
    def model(self):
        """获取被包装的模型"""
        return object.__getattribute__(self, '_model')

    def forward(self, x):
        output = self.model(x)
        # 确保输出是 [batch_size, 1]
        if output.dim() == 1:
            output = output.unsqueeze(-1)
        return output

    def parameters(self, recurse=True):
        """转发参数访问"""
        return self.model.parameters(recurse)

    def state_dict(self, *args, **kwargs):
        """转发 state_dict"""
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """转发 load_state_dict"""
        return self.model.load_state_dict(state_dict, *args, **kwargs)

    def train(self, mode=True):
        """转发 train 方法"""
        self.model.train(mode)
        return self

    def eval(self):
        """转发 eval 方法"""
        self.model.eval()
        return self

    def to(self, device):
        """转发 to 方法"""
        self.model.to(device)
        return self

    @property
    def models(self):
        """转发 models 属性（用于 Boost 模型）"""
        return self.model.models

    def add_model(self):
        """转发 add_model 方法（用于 Boost 模型）"""
        return self.model.add_model()

    def freeze_all_but_latest(self):
        """转发 freeze_all_but_latest 方法（用于 Boost 模型）"""
        return self.model.freeze_all_but_latest()

    def add_expert(self):
        """转发 add_expert 方法（用于 MoE 模型）"""
        return self.model.add_expert()


class RobustnessExperiment(BaseExperiment):
    """鲁棒性实验"""

    def setup_data(self):
        """设置数据生成器"""
        from ..config.exp1_config import get_config
        config = get_config()

        # 创建数据生成器
        data_generator = NoisyFeaturesGenerator(config['data'])

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
        self.logger.info(f"  - True features: {data_generator.n_true}")
        self.logger.info(f"  - Noise features: {data_generator.n_noise}")
        self.logger.info(f"  - Redundant features: {data_generator.n_redundant}")
        self.logger.info(f"  - Deceptive features: {data_generator.n_deceptive}")
        self.logger.info(f"  - Total features: {data_generator.n_features}")

    def setup_models(self):
        """设置模型"""
        from ..config.exp1_config import get_config
        config = get_config()
        device = config['device']

        # 获取真实特征数（用于对齐参数）
        n_true = self.data_generator['feature_info']['n_true_features']
        n_features = self.data_generator['generator'].n_features

        # 以 Standard CFNet 为锚点
        anchor = CFNet_Standard(
            n_true,
            1,
            **config['models']['standard']
        )
        target_params = count_parameters(anchor)

        # 创建 MLP（参数对齐）
        mlp, _ = get_aligned_mlp(n_features, 1, target_params)

        # 创建所有模型
        models_dict = {
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
            'MLP': mlp,
            'Boost': EnsembleResCoFrNet(
                n_features,
                1,
                config['models']['boost']['shallow_depth'],
                config['models']['boost']['poly_degree'],
                learning_rate=0.5
            ),
            'MoE': MoE_Ensemble({
                'input_dim': n_features,
                'output_dim': 1,
                'shallow_depth_per_cofrnet': config['models']['moe']['shallow_depth_per_cofrnet'],
                'polynomial_degree': config['models']['moe']['polynomial_degree']
            })
        }

        # 初始化 MoE 专家
        for _ in range(config['models']['moe']['num_experts']):
            models_dict['MoE'].add_expert()

        # 包装模型并移动到设备
        for name, model in models_dict.items():
            self.models[name] = ModelWrapper(model).to(device)
            param_count = count_parameters(self.models[name])
            self.logger.info(f"  {name}: {param_count:,} parameters")

    def setup_analyzers(self):
        """设置分析器"""
        # 创建鲁棒性分析器
        analyzer_config = {
            **self.config,
            'feature_info': self.data_generator['feature_info']
        }

        self.analyzers = {
            'robustness': RobustnessAnalyzer(analyzer_config, self.result_dir)
        }

        self.logger.info(f"Analyzers setup complete: {list(self.analyzers.keys())}")


# 运行函数
def run_robustness_experiment(config_path: str = None):
    """
    运行鲁棒性实验

    Args:
        config_path: 配置文件路径（可选）
    """
    # 加载配置
    if config_path:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        from ..config.exp1_config import get_config
        config = get_config()

    # 创建实验
    experiment = RobustnessExperiment(config, 'exp1_config')

    # 运行实验
    results = experiment.run()

    return results


if __name__ == '__main__':
    run_robustness_experiment()

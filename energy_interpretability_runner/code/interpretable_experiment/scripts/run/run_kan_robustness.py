"""
KAN (Kolmogorov-Arnold Network) 噪声特征干扰实验
与 CFNet 系列模型进行公平对比
"""

import os
import sys

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime

from cfnet import CFNet_Standard

# 添加 func 路径
func_path = os.path.abspath(os.path.join(project_root, 'xai', 'func'))
if func_path not in sys.path:
    sys.path.insert(0, func_path)

from models_ext import count_parameters
from engine import BaseTrainer

# 动态导入 interpretable 模块
interpretable_path = os.path.abspath(os.path.dirname(__file__))
if interpretable_path not in sys.path:
    sys.path.insert(0, interpretable_path)

# 使用 sys.modules 技巧来确保 data 作为包被识别
import importlib.util
spec = importlib.util.spec_from_file_location("data_module", os.path.join(interpretable_path, "data", "__init__.py"))
data_module = importlib.util.module_from_spec(spec)
sys.modules["data_module"] = data_module

spec2 = importlib.util.spec_from_file_location("generate_noisy_features", os.path.join(interpretable_path, "data", "generate_noisy_features.py"))
generate_noisy_features = importlib.util.module_from_spec(spec2)
sys.modules["generate_noisy_features"] = generate_noisy_features
spec2.loader.exec_module(generate_noisy_features)
NoisyFeaturesGenerator = generate_noisy_features.NoisyFeaturesGenerator

spec3 = importlib.util.spec_from_file_location("robustness_analyzer", os.path.join(interpretable_path, "analysis", "robustness_analyzer.py"))
robustness_analyzer = importlib.util.module_from_spec(spec3)
sys.modules["robustness_analyzer"] = robustness_analyzer
spec3.loader.exec_module(robustness_analyzer)
RobustnessAnalyzer = robustness_analyzer.RobustnessAnalyzer


class SplineBasis(nn.Module):
    """B-样条基函数"""
    def __init__(self, grid_size=10, spline_order=3):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

    def forward(self, x):
        """计算B-样条基函数值"""
        batch_size, n_features = x.shape
        grid = torch.linspace(-1, 1, self.grid_size + 1, device=x.device)
        x_expanded = x.unsqueeze(-1)
        grid_expanded = grid.view(1, 1, -1)
        distances = torch.abs(x_expanded - grid_expanded)
        width = (grid[1] - grid[0]) * (self.spline_order + 1)
        basis = torch.exp(-0.5 * (distances / width) ** 2)
        return basis


class KANLayer(nn.Module):
    """KAN层"""
    def __init__(self, in_features, out_features, grid_size=10, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.n_basis = grid_size + spline_order

        self.spline_basis = SplineBasis(grid_size, spline_order)
        self.coeffs = nn.Parameter(
            torch.randn(in_features, out_features, self.n_basis) * 0.1
        )
        self.residual_weight = nn.Parameter(torch.ones(out_features) * 0.1)

    def forward(self, x):
        batch_size = x.shape[0]
        basis = self.spline_basis(x)
        output = torch.zeros(batch_size, self.out_features, device=x.device)

        for i in range(self.out_features):
            contribution = (basis * self.coeffs[:, i, :].unsqueeze(0)).sum(dim=[1, 2])
            output[:, i] = contribution

        residual = torch.sigmoid(x) * x
        output = output + residual[:, :self.out_features].sum(dim=1, keepdim=True) * self.residual_weight
        return output


class KAN(nn.Module):
    """完整KAN网络"""
    def __init__(self, input_dim, output_dim, hidden_dim=32, num_layers=2,
                 grid_size=10, spline_order=3):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(KANLayer(dims[i], dims[i+1], grid_size, spline_order))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i+1]))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        output = self.network(x)
        if output.shape[-1] == 1:
            output = output.squeeze(-1)
        return output


def get_aligned_kan(input_dim, output_dim, target_params, num_layers=2):
    """获取与目标参数量对齐的KAN模型"""
    grid_size = 10
    spline_order = 3
    n_basis = grid_size + spline_order

    def calc_params(hidden_dim):
        total = 0
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(len(dims) - 1):
            layer_params = dims[i] * dims[i+1] * n_basis + dims[i+1]
            if i < len(dims) - 2:
                layer_params += 2 * dims[i+1]
            total += layer_params
        return total

    best_h = 8
    min_diff = float('inf')
    for h in range(4, 200):
        p = calc_params(h)
        diff = abs(p - target_params)
        if diff < min_diff:
            min_diff = diff
            best_h = h
        if p > target_params * 1.5:
            break

    return KAN(input_dim, output_dim, best_h, num_layers, grid_size, spline_order), best_h


class KANRobustnessExperiment:
    """KAN鲁棒性实验"""

    def __init__(self, config):
        self.config = config
        self.device = config['device']

        # 创建结果目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.result_dir = os.path.join(
            config['base_result_dir'], 'exp1_robustness_kan', timestamp
        )
        os.makedirs(self.result_dir, exist_ok=True)
        for subdir in ['metrics', 'shap', 'plots', 'summary', 'checkpoints']:
            os.makedirs(os.path.join(self.result_dir, subdir), exist_ok=True)

        # 保存配置
        with open(os.path.join(self.result_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        self.data_generator = None
        self.model = None
        self.analyzer = None

    def setup_data(self):
        """设置数据"""
        data_config = self.config['data']
        data_generator = NoisyFeaturesGenerator(data_config)
        train_loader, val_loader, test_loader = data_generator.create_dataloaders()

        self.data_generator = {
            'generator': data_generator,
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'feature_info': data_generator.get_feature_info()
        }

        print(f"Data setup complete:")
        print(f"  - True features: {data_generator.n_true}")
        print(f"  - Noise features: {data_generator.n_noise}")
        print(f"  - Total features: {data_generator.n_features}")

    def setup_model(self):
        """设置KAN模型（与CFNet参数量对齐）"""
        n_features = self.data_generator['generator'].n_features
        n_true = self.data_generator['feature_info']['n_true_features']

        # 以 Standard CFNet 为锚点获取目标参数量
        anchor = CFNet_Standard(n_true, 1, depth=4, poly_degree=4)
        target_params = count_parameters(anchor)

        # 创建参数量对齐的 KAN
        self.model, hidden_dim = get_aligned_kan(
            n_features, 1, target_params, num_layers=2
        )
        self.model = self.model.to(self.device)

        actual_params = count_parameters(self.model)
        print(f"\nModel setup complete:")
        print(f"  - Target parameters: {target_params:,}")
        print(f"  - KAN parameters: {actual_params:,}")
        print(f"  - Hidden dim: {hidden_dim}")
        print(f"  - Difference: {abs(actual_params - target_params)} ({abs(actual_params - target_params)/target_params*100:.1f}%)")

    def setup_analyzer(self):
        """设置分析器"""
        analyzer_config = {
            **self.config,
            'feature_info': self.data_generator['feature_info']
        }
        self.analyzer = RobustnessAnalyzer(analyzer_config, self.result_dir)

    def train(self):
        """训练模型"""
        print("\n" + "="*60)
        print("Training KAN...")
        print("="*60)

        trainer = BaseTrainer(self.model, device=self.device, lr=self.config['lr'])

        history = trainer.fit(
            self.data_generator['train'],
            self.data_generator['val'],
            epochs=self.config['epochs'],
            patience=self.config['patience']
        )

        test_mse = trainer.evaluate(self.data_generator['test'])

        # 保存模型
        model_path = os.path.join(self.result_dir, 'checkpoints', 'KAN.pt')
        torch.save(self.model.state_dict(), model_path)

        print(f"\nKAN - Test MSE: {test_mse:.6f}")

        return {
            'history': history,
            'test_mse': test_mse,
            'model_path': model_path
        }

    def analyze(self, training_results):
        """分析结果"""
        print("\n" + "="*60)
        print("Analyzing KAN...")
        print("="*60)

        # 加载模型
        checkpoint = torch.load(training_results['model_path'], map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        # 计算指标
        metrics = self.analyzer.compute_metrics(
            self.model, self.data_generator['test']
        )

        # 计算SHAP
        shap_values, test_data = self.analyzer.analyze_shap(
            self.model,
            self.data_generator['train'],
            self.data_generator['test']
        )

        # 保存结果
        results = {
            'metrics': metrics,
            'shap_values': shap_values,
            'test_data': test_data
        }

        self.analyzer.save_results(results, 'KAN_robustness')

        return results

    def generate_report(self, analysis_results, training_results):
        """生成报告"""
        report_data = {
            'KAN_robustness': analysis_results
        }
        report = self.analyzer.generate_report(report_data)

        report_path = os.path.join(self.result_dir, 'summary', 'kan_robustness_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\nReport saved to: {report_path}")
        print(f"\n{report}")

        # 保存关键指标供对比
        summary = {
            'model': 'KAN',
            'test_mse': training_results['test_mse'],
            'parameters': count_parameters(self.model),
            'feature_selection_accuracy': analysis_results['metrics'].get('feature_selection_accuracy', {}),
            'noise_suppression_ratio': analysis_results['metrics'].get('noise_suppression_ratio', {}),
            'result_dir': self.result_dir
        }

        summary_path = os.path.join(self.result_dir, 'kan_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary

    def run(self):
        """运行完整实验"""
        print("\n" + "="*60)
        print("KAN Robustness Experiment")
        print("="*60)
        print(f"Result Directory: {self.result_dir}\n")

        self.setup_data()
        self.setup_model()
        self.setup_analyzer()

        training_results = self.train()
        analysis_results = self.analyze(training_results)
        summary = self.generate_report(analysis_results, training_results)

        print("\n" + "="*60)
        print("Experiment Complete!")
        print(f"Results saved to: {self.result_dir}")
        print("="*60)

        return summary


def run_kan_experiment():
    """运行KAN实验"""
    # 使用与其他实验相同的配置
    from pathlib import Path
    runner_base = Path(__file__).resolve().parents[4]
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': 0.001,
        'epochs': 50,
        'patience': 15,
        'batch_size': 64,
        'base_result_dir': str(runner_base / "results"),
        'data': {
            'n_samples': 5000,
            'noise_std': 0.05,
            'test_split': 0.3,
            'val_split': 0.05,
            'random_seed': 42,
            'batch_size': 64,
            'n_true_features': 4,
            'n_noise_features': 4,
            'n_redundant_features': 2,
            'n_deceptive_features': 1,
            'redundant_noise_std': 0.05,
            'deceptive_corr': 0.8
        }
    }

    experiment = KANRobustnessExperiment(config)
    return experiment.run()


if __name__ == '__main__':
    print("="*80)
    print("KAN 噪声特征干扰鲁棒性实验")
    print("="*80)
    print("\n开始运行实验...\n")

    summary = run_kan_experiment()

    print("\n实验完成！")
    print(f"结果保存在: {summary['result_dir']}")
    print(f"\n关键指标:")
    print(f"  - Test MSE: {summary['test_mse']:.6f}")
    print(f"  - Parameters: {summary['parameters']:,}")

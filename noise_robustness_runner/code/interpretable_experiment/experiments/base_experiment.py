"""
基础实验类
所有实验都应继承此类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import os
import json
from datetime import datetime


class BaseExperiment(ABC):
    """实验基类"""

    def __init__(self, config: Dict[str, Any], exp_name: str):
        """
        Args:
            config: 配置字典
            exp_name: 实验名称
        """
        self.config = config
        self.exp_name = exp_name

        # 创建结果目录
        self.result_dir = os.path.join(
            config['base_result_dir'],
            exp_name,
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        os.makedirs(self.result_dir, exist_ok=True)

        # 创建子目录
        for subdir in ['metrics', 'shap', 'plots', 'summary', 'checkpoints']:
            os.makedirs(os.path.join(self.result_dir, subdir), exist_ok=True)

        # 保存配置
        self._save_config()

        # 初始化组件
        self.data_generator = None
        self.models = {}
        self.analyzers = {}

        # 设置日志
        from ..utils.logger import setup_logger
        self.logger = setup_logger(self.result_dir, exp_name)

    def _save_config(self):
        """保存配置到文件"""
        config_path = os.path.join(self.result_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Config saved to: {config_path}")

    @abstractmethod
    def setup_data(self):
        """设置数据生成器"""
        pass

    @abstractmethod
    def setup_models(self):
        """设置模型"""
        pass

    @abstractmethod
    def setup_analyzers(self):
        """设置分析器"""
        pass

    def train_models(self) -> Dict[str, Any]:
        """
        训练所有模型

        Returns:
            training_results: 训练结果
        """
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        from func.engine import BaseTrainer, BoostTrainer

        training_results = {}

        for model_name, model in self.models.items():
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Training {model_name}...")
            self.logger.info(f"{'='*60}\n")

            # 创建训练器
            if model_name == 'Boost':
                trainer = BoostTrainer(
                    model,
                    device=self.config['device'],
                    lr=self.config['lr']
                )
            else:
                trainer = BaseTrainer(
                    model,
                    device=self.config['device'],
                    lr=self.config['lr']
                )

            # 训练
            if model_name == 'Boost':
                history = trainer.fit_boosting(
                    self.data_generator['train'],
                    self.data_generator['val'],
                    num_stages=self.config['models']['boost']['num_stages'],
                    epochs_per_stage=self.config['models']['boost']['epochs_per_stage']
                )
            else:
                history = trainer.fit(
                    self.data_generator['train'],
                    self.data_generator['val'],
                    epochs=self.config['epochs'],
                    patience=self.config['patience']
                )

            # 评估
            test_mse = trainer.evaluate(self.data_generator['test'])

            # 保存模型
            model_path = os.path.join(
                self.result_dir,
                'checkpoints',
                f'{model_name}.pt'
            )
            torch.save(model.state_dict(), model_path)

            training_results[model_name] = {
                'history': history,
                'test_mse': test_mse,
                'model_path': model_path
            }

            self.logger.info(f"{model_name} - Test MSE: {test_mse:.6f}")

        return training_results

    def analyze(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析实验结果

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

                # 计算 SHAP
                shap_values, test_data = analyzer.analyze_shap(
                    model,
                    self.data_generator['train'],
                    self.data_generator['test']
                )

                # 保存结果
                result_key = f"{model_name}_{analyzer_name}"
                analysis_results[result_key] = {
                    'metrics': metrics,
                    'shap_values': shap_values,
                    'test_data': test_data
                }

                # 保存到文件
                analyzer.save_results(
                    analysis_results[result_key],
                    result_key
                )

        return analysis_results

    def generate_reports(self, analysis_results: Dict[str, Any]):
        """
        生成分析报告

        Args:
            analysis_results: 分析结果
        """
        for analyzer_name, analyzer in self.analyzers.items():
            # 筛选该分析器的结果
            analyzer_results = {
                k: v for k, v in analysis_results.items()
                if k.endswith(analyzer_name)
            }

            # 生成报告
            report = analyzer.generate_report(analyzer_results)

            # 保存报告
            report_path = os.path.join(
                self.result_dir,
                'summary',
                f'{analyzer_name}_report.md'
            )
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)

            self.logger.info(f"Report saved to: {report_path}")

            # 打印报告到控制台
            self.logger.info(f"\n{report}")

    def run(self):
        """
        运行完整实验流程

        Returns:
            results: 所有结果
        """
        from ..utils.logger import ProgressLogger
        progress = ProgressLogger(self.logger, 6)

        progress.start_phase("Initializing Experiment")
        self.logger.info(f"Experiment: {self.exp_name}")
        self.logger.info(f"Result Directory: {self.result_dir}")

        progress.start_phase("Setting up Data")
        self.setup_data()

        progress.start_phase("Setting up Models")
        self.setup_models()

        progress.start_phase("Setting up Analyzers")
        self.setup_analyzers()

        progress.start_phase("Training Models")
        training_results = self.train_models()

        progress.start_phase("Analyzing Results")
        analysis_results = self.analyze(training_results)

        progress.start_phase("Generating Reports")
        self.generate_reports(analysis_results)

        progress.finish()

        self.logger.info(f"\nExperiment Complete: {self.exp_name}")
        self.logger.info(f"Results saved to: {self.result_dir}\n")

        return {
            'training_results': training_results,
            'analysis_results': analysis_results
        }

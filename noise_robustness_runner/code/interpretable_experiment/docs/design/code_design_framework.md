# CFNet 可解释性实验代码设计框架

**文档版本**: v1.0
**创建时间**: 2026-01-14
**目标**: 为 5 个可解释性实验方案提供详细的代码实现架构

---

## 目录

1. [整体架构设计](#1-整体架构设计)
2. [目录结构设计](#2-目录结构设计)
3. [核心模块设计](#3-核心模块设计)
4. [数据层设计](#4-数据层设计)
5. [分析层设计](#5-分析层设计)
6. [实验层设计](#6-实验层设计)
7. [配置管理](#7-配置管理)
8. [接口规范](#8-接口规范)
9. [实现路线图](#9-实现路线图)

---

## 1. 整体架构设计

### 1.1 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                    实验层 (Experiments)                      │
│  exp1_robustness.py  exp2_interaction.py  exp3_ood.py      │
│  exp4_causal.py     exp5_real_data.py                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    分析层 (Analysis)                         │
│  robustness_analyzer.py  interaction_analyzer.py            │
│  ood_analyzer.py         causal_analyzer.py                 │
│  domain_analyzer.py                                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    数据层 (Data)                             │
│  generate_noisy_features.py  generate_interaction_data.py    │
│  generate_ood_data.py        generate_causal_data.py         │
│  load_real_datasets.py                                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    基础层 (Foundation)                       │
│  models/  (CFNet variants)                                  │
│  engine/  (Training engine)                                  │
│  analysis/ (SHAP, structural)                                │
│  config.py (Configuration)                                   │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 设计原则

1. **模块化**: 每个实验独立，但共享基础组件
2. **可扩展**: 易于添加新实验和新分析工具
3. **可复用**: 数据生成、训练、分析流程复用现有代码
4. **可测试**: 每个模块有单元测试
5. **文档化**: 完整的 docstring 和类型标注

### 1.3 技术栈

```python
# 核心依赖
numpy >= 1.21.0
pandas >= 1.3.0
torch >= 1.10.0
shap >= 0.40.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0

# 可选依赖（方案 4）
cdt >= 0.6  # Causal Discovery Toolbox
networkx >= 2.6

# 开发依赖
pytest >= 6.2.0
black >= 21.0.0
mypy >= 0.910
```

---

## 2. 目录结构设计

### 2.1 完整目录树

```
xai/
├── README.md                           # 项目说明
├── requirements.txt                    # 依赖列表
├── setup.py                           # 安装配置
│
├── config/                            # 配置管理
│   ├── __init__.py
│   ├── base_config.py                 # 基础配置
│   ├── exp1_config.py                 # 方案 1 配置
│   ├── exp2_config.py                 # 方案 2 配置
│   ├── exp3_config.py                 # 方案 3 配置
│   ├── exp4_config.py                 # 方案 4 配置
│   └── exp5_config.py                 # 方案 5 配置
│
├── data/                              # 数据生成模块
│   ├── __init__.py
│   ├── base.py                        # 基础数据类
│   ├── generate_noisy_features.py     # 方案 1: 干扰特征
│   ├── generate_interaction_data.py   # 方案 2: 交互数据
│   ├── generate_ood_data.py           # 方案 3: OOD 数据
│   ├── generate_causal_data.py        # 方案 4: 因果数据
│   ├── load_real_datasets.py          # 方案 5: 真实数据
│   └── utils.py                       # 数据工具函数
│
├── analysis/                          # 分析模块
│   ├── __init__.py
│   ├── base_analyzer.py               # 基础分析器
│   ├── robustness_analyzer.py         # 方案 1: 鲁棒性分析
│   ├── interaction_analyzer.py        # 方案 2: 交互分析
│   ├── ood_analyzer.py                # 方案 3: OOD 分析
│   ├── causal_analyzer.py             # 方案 4: 因果分析
│   ├── domain_analyzer.py             # 方案 5: 领域分析
│   └── metrics.py                     # 评估指标
│
├── visualization/                     # 可视化模块
│   ├── __init__.py
│   ├── base_plotter.py                # 基础绘图
│   ├── robustness_plots.py            # 方案 1: 鲁棒性图
│   ├── interaction_plots.py           # 方案 2: 交互图
│   ├── ood_plots.py                   # 方案 3: OOD 图
│   ├── causal_plots.py                # 方案 4: 因果图
│   ├── domain_plots.py                # 方案 5: 领域图
│   └── utils.py                       # 绘图工具
│
├── experiments/                       # 实验入口
│   ├── __init__.py
│   ├── base_experiment.py             # 基础实验类
│   ├── exp1_robustness.py             # 方案 1 实验
│   ├── exp2_interaction.py            # 方案 2 实验
│   ├── exp3_ood.py                    # 方案 3 实验
│   ├── exp4_causal.py                 # 方案 4 实验
│   ├── exp5_real_data.py              # 方案 5 实验
│   └── runner.py                      # 批量运行工具
│
├── models/                            # 模型定义（已存在）
│   ├── cfnet.py
│   └── ...
│
├── engine/                            # 训练引擎（已存在）
│   ├── base_trainer.py
│   └── ...
│
├── utils/                             # 工具函数
│   ├── __init__.py
│   ├── random.py                      # 随机种子
│   ├── logger.py                      # 日志工具
│   ├── checkpoint.py                  # 检查点管理
│   └── metrics.py                     # 指标计算
│
├── tests/                             # 单元测试
│   ├── __init__.py
│   ├── test_data.py                   # 测试数据生成
│   ├── test_analysis.py               # 测试分析器
│   └── test_integration.py            # 集成测试
│
└── results/                           # 结果输出
    ├── exp1_robustness/
    │   ├── metrics/
    │   ├── shap/
    │   ├── plots/
    │   └── summary/
    ├── exp2_interaction/
    ├── exp3_ood/
    ├── exp4_causal/
    └── exp5_real_data/
```

### 2.2 文件命名规范

```
数据生成: generate_<experiment>_data.py
分析器: <experiment>_analyzer.py
可视化: <experiment>_plots.py
实验入口: exp<index>_<name>.py
配置: exp<index>_config.py
```

---

## 3. 核心模块设计

### 3.1 基础数据类（data/base.py）

```python
"""
基础数据类
所有数据生成器都应继承此类
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class BaseDataGenerator(ABC):
    """数据生成器基类"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典，包含：
                - n_samples: 样本数
                - noise_std: 噪声标准差
                - test_split: 测试集比例
                - val_split: 验证集比例
                - batch_size: 批大小
                - random_seed: 随机种子
        """
        self.config = config
        self.n_samples = config['n_samples']
        self.noise_std = config['noise_std']
        self.test_split = config['test_split']
        self.val_split = config['val_split']
        self.batch_size = config['batch_size']
        self.seed = config.get('random_seed', 42)

        # 设置随机种子
        self._set_seed()

        # 数据存储
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

    def _set_seed(self):
        """设置随机种子"""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

    @abstractmethod
    def generate_features(self) -> np.ndarray:
        """
        生成特征矩阵

        Returns:
            X: (n_samples, n_features) 特征矩阵
        """
        pass

    @abstractmethod
    def generate_target(self, X: np.ndarray) -> np.ndarray:
        """
        生成目标变量

        Args:
            X: 特征矩阵

        Returns:
            y: (n_samples,) 目标变量
        """
        pass

    def add_noise(self, y: np.ndarray) -> np.ndarray:
        """
        添加高斯噪声

        Args:
            y: 原始目标变量

        Returns:
            y_noisy: 添加噪声后的目标变量
        """
        noise = np.random.normal(0, self.noise_std, size=y.shape)
        return y + noise

    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        划分数据集

        Args:
            X: 特征矩阵
            y: 目标变量

        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        n = len(X)
        n_test = int(n * self.test_split)
        n_val = int(n * self.val_split)
        n_train = n - n_test - n_val

        # 随机打乱
        indices = np.random.permutation(n)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]

        return (
            X[train_indices], y[train_indices],
            X[val_indices], y[val_indices],
            X[test_indices], y[test_indices]
        )

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        创建数据加载器

        Returns:
            train_loader, val_loader, test_loader
        """
        # 生成数据
        X = self.generate_features()
        y = self.generate_target(X)
        y = self.add_noise(y)

        # 划分数据
        X_train, y_train, X_val, y_val, X_test, y_test = self.split_data(X, y)

        # 存储为实例变量
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test

        # 创建 TensorDataset
        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float().unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).float().unsqueeze(1)
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X_test).float(),
            torch.from_numpy(y_test).float().unsqueeze(1)
        )

        # 创建 DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return train_loader, val_loader, test_loader

    def get_feature_info(self) -> Dict[str, Any]:
        """
        获取特征信息

        Returns:
            feature_info: {
                'n_features': 特征数,
                'feature_names': 特征名称列表,
                'feature_types': 特征类型列表,
                'true_features': 真实特征索引,
                'noise_features': 噪声特征索引,
                ...其他实验特定信息
            }
        """
        return {
            'n_features': self.generate_features().shape[1],
            'feature_names': [f'x{i}' for i in range(self.generate_features().shape[1])],
            'feature_types': ['continuous'] * self.generate_features().shape[1]
        }
```

### 3.2 基础分析器（analysis/base_analyzer.py）

```python
"""
基础分析器类
所有分析器都应继承此类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
import shap


class BaseAnalyzer(ABC):
    """分析器基类"""

    def __init__(self, config: Dict[str, Any], save_dir: str = 'results'):
        """
        Args:
            config: 配置字典
            save_dir: 结果保存目录
        """
        self.config = config
        self.save_dir = save_dir
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # 创建保存目录
        import os
        os.makedirs(save_dir, exist_ok=True)

        # 分析结果存储
        self.results = {}

    @abstractmethod
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
        pass

    @abstractmethod
    def analyze_shap(self, model: torch.nn.Module,
                    train_loader: DataLoader,
                    test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 SHAP 值

        Args:
            model: 训练好的模型
            train_loader: 训练数据加载器（用于 SHAP 背景）
            test_loader: 测试数据加载器

        Returns:
            shap_values: SHAP 值
            test_data: 测试数据
        """
        pass

    @abstractmethod
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        生成分析报告

        Args:
            results: 分析结果

        Returns:
            report: Markdown 格式的报告
        """
        pass

    def save_results(self, results: Dict[str, Any], name: str):
        """
        保存分析结果

        Args:
            results: 分析结果
            name: 结果名称
        """
        import os
        import pickle

        # 保存为 pickle
        save_path = os.path.join(self.save_dir, f'{name}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)

        print(f"Results saved to: {save_path}")
```

### 3.3 基础实验类（experiments/base_experiment.py）

```python
"""
基础实验类
所有实验都应继承此类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
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
        from engine import BaseTrainer

        training_results = {}

        for model_name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Training {model_name}...")
            print(f"{'='*60}\n")

            # 创建训练器
            trainer = BaseTrainer(
                model,
                device=self.config['device'],
                lr=self.config['lr']
            )

            # 训练
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

            print(f"{model_name} - Test MSE: {test_mse:.6f}")

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
            print(f"\n{'='*60}")
            print(f"Analyzing {model_name}...")
            print(f"{'='*60}\n")

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
                print(f"  Running {analyzer_name}...")

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
            with open(report_path, 'w') as f:
                f.write(report)

            print(f"Report saved to: {report_path}")

    def run(self):
        """
        运行完整实验流程

        Returns:
            results: 所有结果
        """
        print(f"\n{'='*80}")
        print(f"Starting Experiment: {self.exp_name}")
        print(f"Result Directory: {self.result_dir}")
        print(f"{'='*80}\n")

        # 1. 设置数据
        print("[Phase 1] Setting up data...")
        self.setup_data()

        # 2. 设置模型
        print("[Phase 2] Setting up models...")
        self.setup_models()

        # 3. 设置分析器
        print("[Phase 3] Setting up analyzers...")
        self.setup_analyzers()

        # 4. 训练模型
        print("[Phase 4] Training models...")
        training_results = self.train_models()

        # 5. 分析结果
        print("[Phase 5] Analyzing results...")
        analysis_results = self.analyze(training_results)

        # 6. 生成报告
        print("[Phase 6] Generating reports...")
        self.generate_reports(analysis_results)

        print(f"\n{'='*80}")
        print(f"Experiment Complete: {self.exp_name}")
        print(f"Results saved to: {self.result_dir}")
        print(f"{'='*80}\n")

        return {
            'training_results': training_results,
            'analysis_results': analysis_results
        }
```

---

## 4. 数据层设计

### 4.1 方案 1：干扰特征数据生成器

**文件**: `data/generate_noisy_features.py`

```python
"""
干扰特征数据生成器
生成包含无关、冗余、噪声、欺骗特征的数据
"""

from .base import BaseDataGenerator
from typing import Dict, Any
import numpy as np


class NoisyFeaturesGenerator(BaseDataGenerator):
    """干扰特征数据生成器"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典
                - n_true_features: 真实特征数（默认 4）
                - n_noise_features: 无关特征数（默认 4）
                - n_redundant_features: 冗余特征数（默认 2）
                - n_deceptive_features: 欺骗特征数（默认 1）
                - redundant_noise_std: 冗余特征噪声标准差（默认 0.05）
                - deceptive_corr: 欺骗特征相关性（默认 0.8）
        """
        super().__init__(config)

        self.n_true = config.get('n_true_features', 4)
        self.n_noise = config.get('n_noise_features', 4)
        self.n_redundant = config.get('n_redundant_features', 2)
        self.n_deceptive = config.get('n_deceptive_features', 1)

        self.redundant_noise_std = config.get('redundant_noise_std', 0.05)
        self.deceptive_corr = config.get('deceptive_corr', 0.8)

        # 总特征数
        self.n_features = (
            self.n_true +
            self.n_noise +
            self.n_redundant +
            self.n_deceptive
        )

        # 真实系数
        self.true_coefficients = np.array([1.5, 0.8, 0.5, 1.2])

    def generate_features(self) -> np.ndarray:
        """
        生成特征矩阵

        特征顺序：[真实特征, 无关特征, 冗余特征, 欺骗特征]

        Returns:
            X: (n_samples, n_features) 特征矩阵
        """
        n = self.n_samples

        # 1. 真实特征（0.5 - 2.0）
        X_true = np.random.uniform(0.5, 2.0, (n, self.n_true))

        # 2. 无关特征（0 - 1，完全随机）
        X_noise = np.random.uniform(0, 1, (n, self.n_noise))

        # 3. 冗余特征（真实特征 + 小噪声）
        # 选择前 n_redundant 个真实特征进行复制
        X_redundant = []
        for i in range(self.n_redundant):
            true_feature_idx = i % self.n_true
            redundant = X_true[:, true_feature_idx] + np.random.normal(
                0, self.redundant_noise_std, n
            )
            X_redundant.append(redundant)
        X_redundant = np.column_stack(X_redundant)

        # 4. 欺骗特征（在训练集上与 y 相关，测试集上不相关）
        X_deceptive = []
        for i in range(self.n_deceptive):
            # 训练集：生成与 y 相关的特征
            deceptive_train = np.random.uniform(0, 1, int(n * 0.8))
            # 测试集：随机特征
            deceptive_test = np.random.uniform(0, 1, int(n * 0.2))
            X_deceptive.append(np.concatenate([deceptive_train, deceptive_test]))
        X_deceptive = np.column_stack(X_deceptive)

        # 合并所有特征
        X = np.hstack([X_true, X_noise, X_redundant, X_deceptive])

        # 打乱训练/测试分割的欺骗特征
        # 注意：这在 split_data 中会被重新打乱，
        # 所以需要在生成目标时处理欺骗特征

        return X

    def generate_target(self, X: np.ndarray) -> np.ndarray:
        """
        生成目标变量（仅使用真实特征）

        Args:
            X: 特征矩阵

        Returns:
            y: 目标变量
        """
        # 仅使用前 n_true 个特征
        X_true = X[:, :self.n_true]

        # 线性组合
        y = np.dot(X_true, self.true_coefficients)

        return y

    def get_feature_info(self) -> Dict[str, Any]:
        """
        获取特征信息

        Returns:
            feature_info: 特征信息字典
        """
        feature_names = []
        feature_types = []
        feature_groups = {}

        # 真实特征
        for i in range(self.n_true):
            feature_names.append(f'x{i+1}_true')
            feature_types.append('true')
            feature_groups[f'x{i+1}_true'] = 'true'

        # 无关特征
        for i in range(self.n_noise):
            feature_names.append(f'x{i+1}_noise')
            feature_types.append('noise')
            feature_groups[f'x{i+1}_noise'] = 'noise'

        # 冗余特征
        for i in range(self.n_redundant):
            feature_names.append(f'x{i+1}_redundant')
            feature_types.append('redundant')
            # 记录它复制了哪个真实特征
            source_idx = i % self.n_true
            feature_groups[f'x{i+1}_redundant'] = f'x{source_idx+1}_true'

        # 欺骗特征
        for i in range(self.n_deceptive):
            feature_names.append(f'x{i+1}_deceptive')
            feature_types.append('deceptive')
            feature_groups[f'x{i+1}_deceptive'] = 'deceptive'

        return {
            'n_features': self.n_features,
            'n_true_features': self.n_true,
            'n_noise_features': self.n_noise,
            'n_redundant_features': self.n_redundant,
            'n_deceptive_features': self.n_deceptive,
            'feature_names': feature_names,
            'feature_types': feature_types,
            'feature_groups': feature_groups,
            'true_coefficients': self.true_coefficients.tolist(),
            'true_feature_indices': list(range(self.n_true)),
            'noise_feature_indices': list(range(
                self.n_true,
                self.n_true + self.n_noise
            )),
            'redundant_feature_indices': list(range(
                self.n_true + self.n_noise,
                self.n_true + self.n_noise + self.n_redundant
            )),
            'deceptive_feature_indices': list(range(
                self.n_true + self.n_noise + self.n_redundant,
                self.n_features
            ))
        }


# 工厂函数
def create_noisy_data_generator(config: Dict[str, Any]) -> NoisyFeaturesGenerator:
    """
    创建干扰特征数据生成器

    Args:
        config: 配置字典

    Returns:
        generator: 数据生成器实例
    """
    return NoisyFeaturesGenerator(config)
```

### 4.2 方案 2-5 的数据生成器接口

**接口规范**:

```python
# generate_interaction_data.py
class InteractionDataGenerator(BaseDataGenerator):
    """
    特征交互数据生成器

    支持的交互类型：
    - multiplicative: 乘法交互 (x1 * x2)
    - conditional: 条件交互 (x1 if x3 > 0)
    - rational: 分式交互 ((x1 + x2) / x3)
    - high_order: 高阶交互 (x1 * x2 * x3)
    """
    pass

# generate_ood_data.py
class OODDataGenerator(BaseDataGenerator):
    """
    分布外数据生成器

    支持的场景：
    - covariate_shift: 协变量偏移
    - concept_drift: 概念漂移
    - sparse_region: 稀疏区域
    """
    pass

# generate_causal_data.py
class CausalDataGenerator(BaseDataGenerator):
    """
    因果数据生成器

    支持的因果图：
    - chain: x1 → x2 → x3 → y
    - fork: x3 → (x1, x2) → y
    - rational_causal: 分式因果
    - counterfactual: 反事实
    """
    pass

# load_real_datasets.py
class RealDatasetLoader(BaseDataGenerator):
    """
    真实数据集加载器

    支持的数据集：
    - energy_efficiency: 建筑能耗
    - boston: 房价
    - diabetes: 糖尿病
    - wine: 葡萄酒质量
    """
    pass
```

---

## 5. 分析层设计

### 5.1 方案 1：鲁棒性分析器

**文件**: `analysis/robustness_analyzer.py`

```python
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
            metrics: {
                'mse': 均方误差,
                'mae': 平均绝对误差,
                'r2': R² 分数
            }
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
        计算 SHAP 值

        Args:
            model: 训练好的模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器

        Returns:
            shap_values: (n_samples, n_features) SHAP 值
            test_data: (n_samples, n_features) 测试数据
        """
        import shap

        model.eval()
        device = next(model.parameters()).device

        # 获取背景数据（训练集的子集）
        background_data = []
        for X_batch, _ in train_loader:
            background_data.append(X_batch.numpy())
            if len(background_data) >= 10:  # 100 个样本
                break
        background = np.concatenate(background_data)[:100]

        # 获取测试数据
        test_data_list = []
        for X_batch, _ in test_loader:
            test_data_list.append(X_batch.numpy())
        test_data = np.concatenate(test_data_list)

        # 创建 SHAP 解释器
        explainer = shap.DeepExplainer(model,
                                       torch.from_numpy(background).float().to(device))

        # 计算 SHAP 值
        shap_values = explainer.shap_values(
            torch.from_numpy(test_data).float().to(device)
        )

        # 处理形状
        if shap_values.ndim == 3:
            shap_values = shap_values.squeeze(-1)

        return shap_values, test_data

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
            'redundant_corr': float(np.mean(redundant_corrs)) if redundant_corrs else 0.0
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
        top_k = [5, 10, min(20, len(mean_importance))]

        top_k_accuracy = {}
        for k in top_k:
            top_k_indices = set(ranking[:k])
            true_in_top_k = len(top_k_indices.intersection(set(self.true_indices)))
            top_k_accuracy[f'top_{k}'] = true_in_top_k / n_true

        # 真实特征的平均排名
        true_ranks = []
        for true_idx in self.true_indices:
            rank = np.where(ranking == true_idx)[0][0] + 1
            true_ranks.append(rank)

        return {
            **top_k_accuracy,
            'mean_true_rank': float(np.mean(true_ranks)),
            'median_true_rank': float(np.median(true_ranks))
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        生成分析报告

        Args:
            results: 分析结果字典
                - {model_name}_robustness: {
                    'metrics': 性能指标,
                    'shap_values': SHAP 值,
                    'test_data': 测试数据
                  }

        Returns:
            report: Markdown 格式的报告
        """
        from datetime import datetime

        lines = []
        lines.append(f"# 鲁棒性分析报告\n")
        lines.append(f"生成时间: {datetime.now()}\n")
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
            lines.append(f"  - 冗余特征相关性: {suppression['redundant_corr']:.3f}\n\n")

            # 排序准确性
            ranking = self.analyze_ranking_accuracy(shap_values)
            lines.append("### 特征重要性排序准确性\n")
            lines.append(f"- **真实特征平均排名**: {ranking['mean_true_rank']:.1f}\n")
            lines.append(f"- **真实特征中位数排名**: {ranking['median_true_rank']:.1f}\n")
            for k, acc in ranking.items():
                if k.startswith('top_'):
                    lines.append(f"- **Top {k[4:]} 准确率**: {acc:.1%}\n")
            lines.append("\n")

        # 对比总结
        lines.append("## 模型对比\n")
        lines.append("| 模型 | MSE | 噪声抑制率 | Top-5 准确率 |\n")
        lines.append("|------|-----|-----------|-------------|\n")

        for key, result in results.items():
            model_name = key.replace('_robustness', '')
            metrics = result['metrics']
            shap_values = result['shap_values']

            suppression = self.analyze_feature_suppression(shap_values)
            ranking = self.analyze_ranking_accuracy(shap_values)

            lines.append(
                f"| {model_name} | {metrics['mse']:.4f} | "
                f"{suppression['noise_suppression_ratio']:.3f} | "
                f"{ranking['top_5']:.1%} |\n"
            )

        return ''.join(lines)
```

### 5.2 方案 2-5 的分析器接口

```python
# analysis/interaction_analyzer.py
class InteractionAnalyzer(BaseAnalyzer):
    """
    特征交互分析器

    主要方法：
    - compute_shap_interaction_values(): 计算 SHAP 交互值
    - analyze_interaction_strength(): 分析交互强度
    - detect_top_interactions(): 检测 Top K 交互对
    """
    pass

# analysis/ood_analyzer.py
class OODAnalyzer(BaseAnalyzer):
    """
    分布外分析器

    主要方法：
    - compute_distribution_shift(): 计算分布偏移
    - analyze_importance_stability(): 分析重要性稳定性
    - evaluate_ood_performance(): 评估 OOD 性能
    """
    pass

# analysis/causal_analyzer.py
class CausalAnalyzer(BaseAnalyzer):
    """
    因果分析器

    主要方法：
    - discover_causal_structure(): 发现因果结构
    - compare_with_shap(): 与 SHAP 对比
    - generate_counterfactuals(): 生成反事实解释
    """
    pass

# analysis/domain_analyzer.py
class DomainAnalyzer(BaseAnalyzer):
    """
    领域知识分析器

    主要方法：
    - load_domain_knowledge(): 加载领域知识
    - compute_domain_consistency(): 计算一致性
    - bootstrap_stability(): Bootstrap 稳定性分析
    """
    pass
```

---

## 6. 实验层设计

### 6.1 方案 1：鲁棒性实验

**文件**: `experiments/exp1_robustness.py`

```python
"""
方案 1：特征干扰鲁棒性实验
"""

from .base_experiment import BaseExperiment
from ..data.generate_noisy_features import NoisyFeaturesGenerator
from ..analysis.robustness_analyzer import RobustnessAnalyzer
from ..models_ext import get_aligned_mlp
import sys
import os

# 导入 CFNet 模型
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from cfnet import CFNet_Standard, HybridRationalNet, EnsembleResCoFrNet, MoE_Ensemble


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

        print(f"Data setup complete:")
        print(f"  - True features: {data_generator.n_true}")
        print(f"  - Noise features: {data_generator.n_noise}")
        print(f"  - Redundant features: {data_generator.n_redundant}")
        print(f"  - Deceptive features: {data_generator.n_deceptive}")
        print(f"  - Total features: {data_generator.n_features}")

    def setup_models(self):
        """设置模型"""
        config = self.config
        device = config['device']

        # 获取真实特征数（用于对齐参数）
        n_true = self.data_generator['feature_info']['n_true_features']

        # 以 Standard CFNet 为锚点
        anchor = CFNet_Standard(
            n_true,  # 仅使用真实特征维度
            1,
            **config['models']['standard']
        )
        target_params = count_parameters(anchor)

        # 创建所有模型
        self.models = {
            'Standard': CFNet_Standard(
                self.data_generator['generator'].n_features,  # 使用全部特征
                1,
                **config['models']['standard']
            ),
            'Hybrid': HybridRationalNet(
                self.data_generator['generator'].n_features,
                1,
                **config['models']['hybrid']
            ),
            'MLP': get_aligned_mlp(
                self.data_generator['generator'].n_features,
                1,
                target_params
            )[0],
            'Boost': EnsembleResCoFrNet(
                self.data_generator['generator'].n_features,
                1,
                config['models']['boost']['shallow_depth'],
                config['models']['boost']['poly_degree'],
                learning_rate=0.5
            ),
            'MoE': MoE_Ensemble({
                'input_dim': self.data_generator['generator'].n_features,
                'output_dim': 1,
                'shallow_depth_per_cofrnet': config['models']['moe']['shallow_depth_per_cofrnet'],
                'polynomial_degree': config['models']['moe']['polynomial_degree']
            })
        }

        # 初始化 MoE 专家
        for _ in range(config['models']['moe']['num_experts']):
            self.models['MoE'].add_expert()

        # 移动到设备并包装
        for name, model in self.models.items():
            self.models[name] = ModelWrapper(model).to(device)

        print(f"Models setup complete: {list(self.models.keys())}")

    def setup_analyzers(self):
        """设置分析器"""
        # 创建鲁棒性分析器
        analyzer_config = {
            **self.config,
            'feature_info': self.data_generator['feature_info']
        }

        self.analyzers = {
            'robustness': RobustnessAnalyzer(analyzer_config, self.save_dir)
        }

        print(f"Analyzers setup complete: {list(self.analyzers.keys())}")


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
    experiment = RobustnessExperiment(config, 'exp1_robustness')

    # 运行实验
    results = experiment.run()

    return results


if __name__ == '__main__':
    run_robustness_experiment()
```

### 6.2 方案 2-5 的实验接口

```python
# experiments/exp2_interaction.py
class InteractionExperiment(BaseExperiment):
    """特征交互捕获实验"""
    pass

# experiments/exp3_ood.py
class OODExperiment(BaseExperiment):
    """分布外泛化实验"""
    pass

# experiments/exp4_causal.py
class CausalExperiment(BaseExperiment):
    """因果结构发现实验"""
    pass

# experiments/exp5_real_data.py
class RealDataExperiment(BaseExperiment):
    """真实数据集基准测试"""
    pass
```

---

## 7. 配置管理

### 7.1 基础配置（config/base_config.py）

```python
"""
基础配置
所有实验共享的配置
"""

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_CONFIG = {
    # 设备
    'device': DEVICE,

    # 通用训练配置
    'lr': 0.001,
    'epochs': 100,
    'patience': 15,
    'batch_size': 64,

    # 数据配置
    'n_samples': 5000,
    'noise_std': 0.05,
    'test_split': 0.3,
    'val_split': 0.05,
    'random_seed': 42,

    # 结果目录
    'base_result_dir': 'results',

    # 模型配置
    'models': {
        'standard': {
            'depth': 4,
            'poly_degree': 4
        },
        'hybrid': {
            'unit_degree': 4,
            'num_units': 20
        },
        'boost': {
            'shallow_depth': 4,
            'poly_degree': 4,
            'num_stages': 20,
            'epochs_per_stage': 50
        },
        'moe': {
            'shallow_depth_per_cofrnet': 2,
            'polynomial_degree': 2,
            'num_experts': 5
        }
    }
}
```

### 7.2 实验 1 配置（config/exp1_config.py）

```python
"""
方案 1 配置：干扰鲁棒性实验
"""

from .base_config import BASE_CONFIG

def get_config():
    """获取方案 1 的完整配置"""
    config = BASE_CONFIG.copy()
    config['models'] = BASE_CONFIG['models'].copy()

    # 数据配置
    config['data'] = {
        **{k: v for k, v in BASE_CONFIG.items() if k in ['device', 'n_samples', 'noise_std', 'test_split', 'val_split', 'random_seed', 'batch_size']},
        'n_true_features': 4,
        'n_noise_features': 4,
        'n_redundant_features': 2,
        'n_deceptive_features': 1,
        'redundant_noise_std': 0.05,
        'deceptive_corr': 0.8
    }

    return config
```

---

## 8. 接口规范

### 8.1 数据生成器接口

```python
class BaseDataGenerator(ABC):
    """数据生成器基类"""

    @abstractmethod
    def generate_features(self) -> np.ndarray:
        """生成特征矩阵"""
        pass

    @abstractmethod
    def generate_target(self, X: np.ndarray) -> np.ndarray:
        """生成目标变量"""
        pass

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建数据加载器"""
        pass

    def get_feature_info(self) -> Dict[str, Any]:
        """获取特征信息"""
        pass
```

### 8.2 分析器接口

```python
class BaseAnalyzer(ABC):
    """分析器基类"""

    @abstractmethod
    def compute_metrics(self, model, test_loader) -> Dict[str, float]:
        """计算性能指标"""
        pass

    @abstractmethod
    def analyze_shap(self, model, train_loader, test_loader) -> Tuple[np.ndarray, np.ndarray]:
        """计算 SHAP 值"""
        pass

    @abstractmethod
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成分析报告"""
        pass
```

### 8.3 实验接口

```python
class BaseExperiment(ABC):
    """实验基类"""

    @abstractmethod
    def setup_data(self):
        """设置数据"""
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
        """训练模型"""
        pass

    def analyze(self, training_results) -> Dict[str, Any]:
        """分析结果"""
        pass

    def run(self):
        """运行完整实验"""
        pass
```

---

## 9. 实现路线图

### Phase 1: 基础架构（1-2 天）

**任务**:
- [ ] 创建目录结构
- [ ] 实现基础类（BaseDataGenerator, BaseAnalyzer, BaseExperiment）
- [ ] 实现配置管理系统
- [ ] 编写单元测试框架

**验收标准**:
- 基础类可以实例化
- 配置可以正确加载
- 单元测试通过

### Phase 2: 方案 1 实现（2-3 天）

**任务**:
- [ ] 实现 NoisyFeaturesGenerator
- [ ] 实现 RobustnessAnalyzer
- [ ] 实现 RobustnessExperiment
- [ ] 运行完整实验并生成报告
- [ ] 调试和优化

**验收标准**:
- 实验可以成功运行
- 报告正确生成
- CFNet 优势可以观察到

### Phase 3: 方案 5 实现（2-3 天）

**任务**:
- [ ] 实现 RealDatasetLoader
- [ ] 实现 DomainAnalyzer
- [ ] 实现 RealDataExperiment
- [ ] 加载 Energy Efficiency 数据集
- [ ] 运行实验并对比

**验收标准**:
- 真实数据可以正确加载
- 与领域知识对比正确
- 多模型对比结果合理

### Phase 4: 方案 2 实现（3-4 天）

**任务**:
- [ ] 实现 InteractionDataGenerator
- [ ] 实现 InteractionAnalyzer（SHAP 交互值）
- [ ] 实现 InteractionExperiment
- [ ] 可视化交互热力图和依赖图

**验收标准**:
- SHAP 交互值正确计算
- 分式交互优势明显

### Phase 5: 方案 3 实现（4-5 天）

**任务**:
- [ ] 实现 OODDataGenerator
- [ ] 实现 OODAnalyzer
- [ ] 实现 OODExperiment
- [ ] 可视化稳定性对比

**验收标准**:
- OOD 场景正确创建
- 稳定性指标合理

### Phase 6: 方案 4 实现（5-7 天，可选）

**任务**:
- [ ] 实现 CausalDataGenerator
- [ ] 实现 CausalAnalyzer
- [ ] 集成因果发现库
- [ ] 实现 CausalExperiment

**验收标准**:
- 因果图正确生成
- 与 SHAP 对比有意义

### Phase 7: 整合与优化（3-5 天）

**任务**:
- [ ] 批量运行工具（experiments/runner.py）
- [ ] 统一结果格式
- [ ] 生成综合报告
- [ ] 代码优化和文档完善

**验收标准**:
- 所有实验可以一键运行
- 结果格式统一
- 文档完整

---

## 附录

### A. 类型标注规范

```python
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

# 数据生成器返回类型
DataSplit = Tuple[
    np.ndarray, np.ndarray,  # X_train, y_train
    np.ndarray, np.ndarray,  # X_val, y_val
    np.ndarray, np.ndarray   # X_test, y_test
]

DataloaderTriple = Tuple[DataLoader, DataLoader, DataLoader]

# SHAP 分析结果
SHAPResult = Tuple[np.ndarray, np.ndarray]  # shap_values, test_data

# 性能指标
MetricsDict = Dict[str, float]

# 分析结果
AnalysisResult = Dict[str, Any]
```

### B. 错误处理规范

```python
class DataGenerationError(Exception):
    """数据生成错误"""
    pass

class AnalysisError(Exception):
    """分析错误"""
    pass

class ExperimentError(Exception):
    """实验错误"""
    pass
```

### C. 日志规范

```python
import logging

def setup_logger(save_dir: str, exp_name: str):
    """设置日志器"""
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)

    # 文件处理器
    fh = logging.FileHandler(f'{save_dir}/{exp_name}.log')
    fh.setLevel(logging.INFO)

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
```

---

**文档版本**: v1.0
**最后更新**: 2026-01-14
**维护者**: CFNet XAI 实验组

---

**END OF CODE DESIGN FRAMEWORK**

# 可解释性实验实现快速开始指南

**目标**: 帮助开发者快速上手实现 5 个可解释性实验方案

**文档版本**: v1.0
**创建时间**: 2026-01-14

---

## 📋 文档导航

| 文档 | 大小 | 内容 |
|------|------|------|
| **interpretable_experiment_proposals.md** | 30KB | 5 个实验方案的设计理论 |
| **code_design_framework.md** | 54KB | 详细的代码架构设计 |
| **quick_start_guide.md** (本文件) | - | 实施路线图和快速开始 |

---

## 🚀 快速开始（5 分钟）

### Step 1: 理解整体架构

**三层架构**：
```
实验层 (Experiments)  →  调度实验流程
        ↓
分析层 (Analysis)      →  分析结果（SHAP、鲁棒性等）
        ↓
数据层 (Data)          →  生成数据（干扰、交互、因果等）
        ↓
基础层 (Foundation)    →  CFNet 模型、训练引擎（已存在）
```

### Step 2: 查看核心文件

**设计文档**：
```bash
cd /home/zxc/CodeBase/cofrnet/experiments/paper/xai/docs/interpretable

# 查看方案设计
cat interpretable_experiment_proposals.md

# 查看代码架构
cat code_design_framework.md
```

### Step 3: 选择实施方案

**推荐顺序**：

| 阶段 | 方案 | 预计时间 | 优先级 |
|------|------|---------|--------|
| **Phase 1** | 方案 1（干扰鲁棒性） | 2-3 天 | ⭐⭐⭐⭐⭐ |
| **Phase 1** | 方案 5（真实数据） | 2-3 天 | ⭐⭐⭐⭐⭐ |
| **Phase 2** | 方案 2（交互捕获） | 3-4 天 | ⭐⭐⭐⭐ |
| **Phase 2** | 方案 3（OOD 泛化） | 4-5 天 | ⭐⭐⭐⭐ |
| **Phase 3** | 方案 4（因果发现） | 5-7 天 | ⭐⭐⭐ |

---

## 📂 目录结构创建

### 创建新的目录结构

```bash
cd /home/zxc/CodeBase/cofrnet/experiments/paper/xai

# 创建新的模块目录
mkdir -p config data analysis visualization experiments utils tests

# 在 config 中创建配置文件
touch config/__init__.py
touch config/base_config.py
touch config/exp1_config.py

# 在 data 中创建数据生成器
touch data/__init__.py
touch data/base.py
touch data/generate_noisy_features.py

# 在 analysis 中创建分析器
touch analysis/__init__.py
touch analysis/base_analyzer.py
touch analysis/robustness_analyzer.py

# 在 experiments 中创建实验入口
touch experiments/__init__.py
touch experiments/base_experiment.py
touch experiments/exp1_robustness.py

# 创建结果目录
mkdir -p results/exp1_robustness/{metrics,shap,plots,summary,checkpoints}
```

### 完整目录树

```
xai/
├── config/                     # [新增] 配置管理
│   ├── __init__.py
│   ├── base_config.py          # 基础配置
│   └── exp1_config.py          # 方案 1 配置
│
├── data/                       # [新增] 数据生成模块
│   ├── __init__.py
│   ├── base.py                 # 基础数据类
│   └── generate_noisy_features.py  # 方案 1 数据生成器
│
├── analysis/                   # [新增] 分析模块
│   ├── __init__.py
│   ├── base_analyzer.py        # 基础分析器
│   └── robustness_analyzer.py  # 方案 1 分析器
│
├── experiments/                # [新增] 实验入口
│   ├── __init__.py
│   ├── base_experiment.py      # 基础实验类
│   └── exp1_robustness.py      # 方案 1 实验
│
├── models/                     # [已存在] CFNet 模型
├── engine/                     # [已存在] 训练引擎
├── analysis.py                 # [已存在] SHAP 分析
├── data.py                     # [已存在] 原数据生成
└── run_experiment.py           # [已存在] 原实验入口
```

---

## 💻 最小可行实现（MVP）

### 方案 1 的最小版本（1 天实现）

#### 1. 基础数据类（data/base.py）

```python
# 创建最简版本
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

class BaseDataGenerator(ABC):
    """数据生成器基类"""

    def __init__(self, config):
        self.config = config
        self.n_samples = config['n_samples']
        np.random.seed(42)

    @abstractmethod
    def generate_features(self):
        pass

    @abstractmethod
    def generate_target(self, X):
        pass

    def create_dataloaders(self):
        """创建数据加载器"""
        X = self.generate_features()
        y = self.generate_target(X)

        # 简单划分 70% / 15% / 15%
        n = len(X)
        n_train = int(0.7 * n)
        n_val = int(0.85 * n)

        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:n_val], y[n_train:n_val]
        X_test, y_test = X[n_val:], y[n_val:]

        # 创建 DataLoader
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y_train).float().unsqueeze(1)
            ),
            batch_size=64,
            shuffle=True
        )

        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_val).float(),
                torch.from_numpy(y_val).float().unsqueeze(1)
            ),
            batch_size=64,
            shuffle=False
        )

        test_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_test).float(),
                torch.from_numpy(y_test).float().unsqueeze(1)
            ),
            batch_size=64,
            shuffle=False
        )

        return train_loader, val_loader, test_loader
```

#### 2. 干扰特征生成器（data/generate_noisy_features.py）

```python
from .base import BaseDataGenerator
import numpy as np

class NoisyFeaturesGenerator(BaseDataGenerator):
    """干扰特征数据生成器"""

    def __init__(self, config):
        super().__init__(config)
        self.n_true = config.get('n_true_features', 4)
        self.n_noise = config.get('n_noise_features', 4)
        self.n_features = self.n_true + self.n_noise

    def generate_features(self):
        """生成特征矩阵"""
        n = self.n_samples

        # 真实特征
        X_true = np.random.uniform(0.5, 2.0, (n, self.n_true))

        # 噪声特征
        X_noise = np.random.uniform(0, 1, (n, self.n_noise))

        # 合并
        X = np.hstack([X_true, X_noise])
        return X

    def generate_target(self, X):
        """生成目标变量（仅使用真实特征）"""
        # 仅使用前 n_true 个特征
        X_true = X[:, :self.n_true]
        coefficients = np.array([1.5, 0.8, 0.5, 1.2])
        y = np.dot(X_true, coefficients)
        return y
```

#### 3. 运行实验（scripts/run_exp1_simple.py）

```python
"""
方案 1 最小化运行脚本
"""
import sys
sys.path.append('..')

from data.generate_noisy_features import NoisyFeaturesGenerator
from cfnet import CFNet_Standard, HybridRationalNet
from models_ext import get_aligned_mlp, count_parameters
import torch

# 配置
config = {
    'n_samples': 5000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# 1. 生成数据
print("Generating data...")
data_gen = NoisyFeaturesGenerator(config)
train_loader, val_loader, test_loader = data_gen.create_dataloaders()

print(f"Total features: {data_gen.n_features}")
print(f"True features: {data_gen.n_true}")
print(f"Noise features: {data_gen.n_noise}")

# 2. 创建模型
print("\nCreating models...")
device = config['device']

# CFNet Standard
cfnet = CFNet_Standard(data_gen.n_features, 1, depth=4, poly_degree=4).to(device)

# MLP（参数对齐）
anchor = CFNet_Standard(data_gen.n_true, 1, depth=4, poly_degree=4)
target_params = count_parameters(anchor)
mlp, _ = get_aligned_mlp(data_gen.n_features, 1, target_params)
mlp = mlp.to(device)

print(f"CFNet params: {count_parameters(cfnet)}")
print(f"MLP params: {count_parameters(mlp)}")

# 3. 训练模型
print("\nTraining CFNet...")
from engine import BaseTrainer
trainer_cfnet = BaseTrainer(cfnet, device=device, lr=0.001)
history_cfnet = trainer_cfnet.fit(train_loader, val_loader, epochs=50, patience=10)
mse_cfnet = trainer_cfnet.evaluate(test_loader)

print("\nTraining MLP...")
trainer_mlp = BaseTrainer(mlp, device=device, lr=0.001)
history_mlp = trainer_mlp.fit(train_loader, val_loader, epochs=50, patience=10)
mse_mlp = trainer_mlp.evaluate(test_loader)

print(f"\nResults:")
print(f"CFNet MSE: {mse_cfnet:.6f}")
print(f"MLP MSE: {mse_mlp:.6f}")
print(f"Match Ratio: {mse_cfnet / mse_mlp:.3f}")

# 4. 计算 SHAP
print("\nComputing SHAP...")
import shap

# CFNet SHAP
cfnet.eval()
background = next(iter(train_loader))[0][:100].to(device)
test_data = next(iter(test_loader))[0][:200].to(device)

explainer_cfnet = shap.DeepExplainer(cfnet, background)
shap_cfnet = explainer_cfnet.shap_values(test_data)

# MLP SHAP
mlp.eval()
explainer_mlp = shap.DeepExplainer(mlp, background)
shap_mlp = explainer_mlp.shap_values(test_data)

# 5. 分析结果
print("\nFeature Importance Analysis:")
print("-" * 60)

# 计算平均重要性（绝对值）
mean_importance_cfnet = np.mean(np.abs(shap_cfnet), axis=0)
mean_importance_mlp = np.mean(np.abs(shap_mlp), axis=0)

print(f"{'Feature':<15} {'CFNet':<15} {'MLP':<15}")
print("-" * 60)
for i in range(data_gen.n_features):
    if i < data_gen.n_true:
        feature_type = "True"
    else:
        feature_type = "Noise"
    print(f"x{i+1} ({feature_type}):    {mean_importance_cfnet[i]:<15.4f} {mean_importance_mlp[i]:<15.4f}")

# 计算抑制率
true_mean_cfnet = np.mean(mean_importance_cfnet[:data_gen.n_true])
noise_mean_cfnet = np.mean(mean_importance_cfnet[data_gen.n_true:])
true_mean_mlp = np.mean(mean_importance_mlp[:data_gen.n_true])
noise_mean_mlp = np.mean(mean_importance_mlp[data_gen.n_true:])

suppression_cfnet = noise_mean_cfnet / true_mean_cfnet
suppression_mlp = noise_mean_mlp / true_mean_mlp

print("\nNoise Suppression Ratio:")
print(f"CFNet: {suppression_cfnet:.3f}")
print(f"MLP: {suppression_mlp:.3f}")
print(f"Lower is better → CFNet is {'better' if suppression_cfnet < suppression_mlp else 'worse'}")
```

#### 4. 运行脚本

```bash
cd /home/zxc/CodeBase/cofrnet/experiments/paper/xai

# 创建 scripts 目录
mkdir scripts

# 将上述代码保存为 scripts/run_exp1_simple.py

# 运行
python scripts/run_exp1_simple.py
```

**预期输出**：
```
Generating data...
Total features: 8
True features: 4
Noise features: 4

Creating models...
CFNet params: 43
MLP params: 45

Training CFNet...
[Epoch 1/50] Train Loss: 1.234567, Val Loss: 1.123456
...
[Epoch 50/50] Train Loss: 0.002345, Val Loss: 0.002123
Evaluating...
CFNet MSE: 0.002123

Training MLP...
...
MLP MSE: 0.002456

Results:
CFNet MSE: 0.002123
MLP MSE: 0.002456
Match Ratio: 0.864

Feature Importance Analysis:
------------------------------------------------------------
Feature         CFNet           MLP
------------------------------------------------------------
x1 (True):      0.2345          0.2234
x2 (True):      0.1234          0.1345
x3 (True):      0.0987          0.0876
x4 (True):      0.1567          0.1456
x5 (Noise):     0.0123          0.0234
x6 (Noise):     0.0109          0.0198
x7 (Noise):     0.0112          0.0212
x8 (Noise):     0.0098          0.0223

Noise Suppression Ratio:
CFNet: 0.087
MLP: 0.154
Lower is better → CFNet is better
```

---

## 📚 完整实施步骤

### Step 1: 环境准备（30 分钟）

```bash
# 1. 创建目录结构
cd /home/zxc/CodeBase/cofrnet/experiments/paper/xai
mkdir -p config data analysis experiments utils

# 2. 创建 __init__.py 文件
touch config/__init__.py
touch data/__init__.py
touch analysis/__init__.py
touch experiments/__init__.py
touch utils/__init__.py

# 3. 验证环境
python -c "import torch; import shap; print('Environment OK')"
```

### Step 2: 基础框架实现（2-3 小时）

按照 `code_design_framework.md` 中的设计实现：

1. **data/base.py** - 基础数据类
2. **analysis/base_analyzer.py** - 基础分析器
3. **experiments/base_experiment.py** - 基础实验类

### Step 3: 方案 1 实现（1-2 天）

1. **数据层**：
   - `data/generate_noisy_features.py`

2. **分析层**：
   - `analysis/robustness_analyzer.py`

3. **实验层**：
   - `experiments/exp1_robustness.py`

4. **配置**：
   - `config/exp1_config.py`

5. **运行实验**：
   ```bash
   python -m experiments.exp1_robustness
   ```

### Step 4: 方案 5 实现（1-2 天）

1. **数据层**：
   - `data/load_real_datasets.py`

2. **分析层**：
   - `analysis/domain_analyzer.py`

3. **实验层**：
   - `experiments/exp5_real_data.py`

### Step 5: 方案 2-4 实现（2-3 周）

按照相同模式实现其他方案。

---

## 🔧 常用命令

### 运行单个实验

```bash
# 方案 1
python -m experiments.exp1_robustness

# 方案 5
python -m experiments.exp5_real_data
```

### 批量运行

```python
# scripts/run_all_experiments.py
from experiments.exp1_robustness import run_robustness_experiment
from experiments.exp5_real_data import run_real_data_experiment

experiments = [
    ('exp1_robustness', run_robustness_experiment),
    ('exp5_real_data', run_real_data_experiment)
]

for exp_name, run_func in experiments:
    print(f"\nRunning {exp_name}...")
    try:
        results = run_func()
        print(f"{exp_name} completed successfully")
    except Exception as e:
        print(f"{exp_name} failed: {e}")
```

### 查看结果

```bash
# 查看报告
cat results/exp1_robustness/summary/robustness_report.md

# 查看 SHAP 图
ls results/exp1_robustness/shap/
```

---

## 📊 代码复用清单

### 从现有代码复用

| 模块 | 已有文件 | 新文件 | 复用内容 |
|------|---------|--------|---------|
| 数据加载 | `data.py` | `data/base.py` | DataLoader 创建逻辑 |
| 模型训练 | `engine.py` | `analysis/base_analyzer.py` | 训练循环、评估 |
| SHAP 分析 | `analysis.py` | `analysis/base_analyzer.py` | SHAP 计算 |
| 实验运行 | `run_experiment.py` | `experiments/base_experiment.py` | 流程框架 |

### 需要新增的代码

| 模块 | 新增代码量 | 关键类/函数 |
|------|-----------|-----------|
| 数据生成 | ~500 行/方案 | `generate_noisy_features.py` |
| 分析器 | ~400 行/方案 | `robustness_analyzer.py` |
| 实验 | ~300 行/方案 | `exp1_robustness.py` |
| 配置 | ~100 行/方案 | `exp1_config.py` |

---

## 🐛 调试技巧

### 1. 单元测试

```python
# tests/test_data.py
def test_noisy_features_generator():
    config = {'n_samples': 100, 'n_true_features': 4, 'n_noise_features': 4}
    gen = NoisyFeaturesGenerator(config)

    # 测试特征形状
    X = gen.generate_features()
    assert X.shape == (100, 8)

    # 测试目标变量
    y = gen.generate_target(X)
    assert y.shape == (100,)

    print("Data generation test passed!")
```

### 2. 打印调试

```python
# 在关键位置添加打印
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Feature info: {data_gen.get_feature_info()}")
```

### 3. 可视化检查

```python
import matplotlib.pyplot as plt

# 检查特征分布
plt.hist(X[:, 0], bins=50)
plt.title('Feature x1 Distribution')
plt.show()
```

---

## 📖 参考资源

### 内部文档

1. **方案设计**: `docs/interpretable/interpretable_experiment_proposals.md`
2. **代码架构**: `docs/interpretable/code_design_framework.md`
3. **原有实验**: `docs/func/experimental_results_report.md`

### 外部资源

1. **SHAP 文档**: https://shap.readthedocs.io/
2. **PyTorch 文档**: https://pytorch.org/docs/
3. **Scikit-learn**: https://scikit-learn.org/

### 代码示例

1. **SHAP 深度学习示例**: https://github.com/slundberg/shap#deep-learning-example
2. **PyTorch 数据加载**: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

---

## ✅ 实施检查清单

### Phase 1: 基础架构（1-2 天）

- [ ] 创建目录结构
- [ ] 实现 `data/base.py`
- [ ] 实现 `analysis/base_analyzer.py`
- [ ] 实现 `experiments/base_experiment.py`
- [ ] 实现 `config/base_config.py`
- [ ] 编写单元测试
- [ ] 测试基础功能

### Phase 2: 方案 1 实现（2-3 天）

- [ ] 实现 `data/generate_noisy_features.py`
- [ ] 实现 `analysis/robustness_analyzer.py`
- [ ] 实现 `experiments/exp1_robustness.py`
- [ ] 实现 `config/exp1_config.py`
- [ ] 运行完整实验
- [ ] 生成报告
- [ ] 验证 CFNet 优势

### Phase 3: 方案 5 实现（2-3 天）

- [ ] 实现 `data/load_real_datasets.py`
- [ ] 实现 `analysis/domain_analyzer.py`
- [ ] 实现 `experiments/exp5_real_data.py`
- [ ] 实现 `config/exp5_config.py`
- [ ] 加载 Energy Efficiency 数据集
- [ ] 运行实验并对比
- [ ] 分析与领域知识一致性

### Phase 4: 整合与文档（1-2 天）

- [ ] 创建批量运行脚本
- [ ] 统一结果格式
- [ ] 生成综合报告
- [ ] 完善文档和注释
- [ ] 代码 review 和优化

---

## 🆘 获取帮助

### 遇到问题？

1. **查看文档**：
   - 首先查看 `code_design_framework.md` 中的详细设计
   - 参考 `interpretable_experiment_proposals.md` 中的方案说明

2. **检查现有代码**：
   - 查看 `data.py`, `analysis.py`, `run_experiment.py` 的实现
   - 复用相似的代码模式

3. **单元测试**：
   - 对每个模块编写单元测试
   - 逐步验证功能

### 常见问题

**Q1: SHAP 计算内存不足**
```python
# 解决：减少背景数据数量
background = next(iter(train_loader))[0][:50].to(device)  # 从 100 减少到 50
```

**Q2: 训练时间过长**
```python
# 解决：减少 epoch 或提前停止
history = trainer.fit(train_loader, val_loader, epochs=50, patience=5)
```

**Q3: 特征信息丢失**
```python
# 解决：在数据生成器中保存 feature_info
feature_info = data_generator.get_feature_info()
print(feature_info)
```

---

## 📞 联系方式

**项目维护者**: CFNet XAI 实验组
**文档位置**: `/home/zxc/CodeBase/cofrnet/experiments/paper/xai/docs/interpretable/`

**更新日志**:
- v1.0 (2026-01-14): 初始版本

---

**END OF QUICK START GUIDE**

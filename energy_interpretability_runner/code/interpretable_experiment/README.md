# CFNet 可解释性实验项目

**项目状态**: ✅ 100%完成 (5/5实验方案)
**完成时间**: 2026-01-14

---

## 📖 项目简介

本项目通过5个独立的实验方案，系统性地验证了**连分式神经网络(CFNet)**的可解释性和实用性。

### 核心发现

- **隐式正则化**: 噪声抑制率比MLP强**47倍**
- **小样本学习**: 在537个训练样本上R²达到**0.93**
- **预测性能**: 在真实数据集上MSE比MLP低**58倍**
- **特征识别**: Top-5准确率达到**100%**

---

## 🚀 快速开始

### 运行实验

```bash
# 运行方案1: 特征干扰鲁棒性
python scripts/run/run_exp1.py

# 运行方案2: 特征交互捕获
python scripts/run/run_exp2_full.py

# 运行方案3: OOD泛化稳定性
python scripts/run/run_exp3_full.py

# 运行方案4: 因果结构发现
python scripts/run/run_exp4_full.py

# 运行方案5: 真实数据集基准
python scripts/run/run_exp5.py
```

### 生成报告

```bash
# 生成最终综合报告
python scripts/analysis/generate_final_report.py
```

---

## 📚 文档导航

### 核心文档 (必读)

| 文档 | 说明 | 适合场景 |
|------|------|---------|
| **[docs/reports/CFNET_CAPABILITY_VALIDATION.md](docs/reports/CFNET_CAPABILITY_VALIDATION.md)** ⭐ | CFNet核心能力验证 | 论文撰写、学术报告 |
| **[docs/reports/COMPLETE_EXPERIMENT_REPORT.md](docs/reports/COMPLETE_EXPERIMENT_REPORT.md)** ⭐ | 完整实验报告 (47页) | 全面了解实验 |
| **[docs/reports/FINAL_COMPREHENSIVE_REPORT.md](docs/reports/FINAL_COMPREHENSIVE_REPORT.md)** | 综合报告 | 快速了解成果 |
| **[docs/reports/FILE_ORGANIZATION.md](docs/reports/FILE_ORGANIZATION.md)** | 文件组织清单 | 查找文件 |

### 实验结果

| 文档 | 内容 |
|------|------|
| [docs/reports/FINAL_WORK_SUMMARY.md](docs/reports/FINAL_WORK_SUMMARY.md) | 最终工作总结 (100%完成) |
| [docs/reports/EXP1_RESULTS_SUMMARY.md](docs/reports/EXP1_RESULTS_SUMMARY.md) | 方案1详细结果 |
| [docs/reports/COMPREHENSIVE_RESULTS_SUMMARY.md](docs/reports/COMPREHENSIVE_RESULTS_SUMMARY.md) | 方案1+5综合结果 |

### 实现文档

| 文档 | 内容 |
|------|------|
| [docs/reports/IMPLEMENTATION_SUMMARY.md](docs/reports/IMPLEMENTATION_SUMMARY.md) | 实现方案总结 |
| [docs/reports/FINAL_DELIVERY_REPORT.md](docs/reports/FINAL_DELIVERY_REPORT.md) | 交付报告 |

---

## 🗂️ 项目结构

```
interpretable_experiment/
├── analysis/          # 分析器模块 (7个文件)
│   ├── base_analyzer.py
│   ├── robustness_analyzer.py    # 方案1
│   ├── interaction_analyzer.py   # 方案2
│   ├── ood_analyzer.py           # 方案3
│   ├── causal_analyzer.py        # 方案4
│   └── domain_analyzer.py        # 方案5
│
├── config/            # 配置文件 (7个文件)
│   ├── base_config.py
│   └── exp1~5_config.py
│
├── data/              # 数据生成 (7个文件)
│   ├── generate_noisy_features.py    # 方案1
│   ├── generate_interaction_data.py   # 方案2
│   ├── generate_ood_data.py          # 方案3
│   ├── generate_causal_data.py       # 方案4
│   └── load_real_datasets.py         # 方案5
│
├── experiments/       # 实验实现 (7个文件)
│   ├── base_experiment.py
│   ├── exp1_robustness.py     # 方案1: 特征干扰鲁棒性
│   ├── exp2_interaction.py    # 方案2: 特征交互捕获
│   ├── exp3_ood.py            # 方案3: OOD泛化稳定性
│   ├── exp4_causal.py         # 方案4: 因果结构发现
│   └── exp5_real_data.py      # 方案5: 真实数据集基准
│
├── docs/              # 文档
│   ├── quick_start_guide.md
│   ├── design/        # 设计文档
│   │   ├── code_design_framework.md
│   │   └── interpretable_experiment_proposals.md
│   └── reports/       # 实验报告
│
├── scripts/           # 运行和分析脚本
│   ├── run/           # 实验运行脚本
│   └── analysis/      # 分析脚本
│
├── utils/             # 工具函数
│   └── logger.py
│
└── models/            # 模型定义
    └── kan_model.py
```

---

## 📊 五个实验方案

| 方案 | 名称 | 关键发现 | 状态 |
|------|------|---------|------|
| **方案1** | 特征干扰鲁棒性 | 噪声抑制率 0.010 vs 0.474 (47倍) | ✅ |
| **方案2** | 特征交互捕获 | Hybrid CFNet MSE 0.0027 (最佳) | ✅ |
| **方案3** | OOD泛化稳定性 | MoE CFNet OOD MSE 7.36 vs 16.66 | ✅ |
| **方案4** | 因果结构发现 | Top-3准确率 100% | ✅ |
| **方案5** | 真实数据集基准 | MSE 7.13 vs 411.60 (58倍) | ✅ |

---

## 🎯 CFNet核心优势

### 1. 隐式正则化能力 ⭐⭐⭐⭐⭐

**证据**: 噪声抑制率 0.010 vs MLP 0.474 (方案1)

- 自动抑制无关特征
- 无需显式特征选择
- 47倍优势

### 2. 小样本学习能力 ⭐⭐⭐⭐⭐

**证据**: 537训练样本 → R²=0.93 (方案5)

- 参数效率高
- 不易过拟合
- 泛化能力强

### 3. 预测性能优异 ⭐⭐⭐⭐⭐

**证据**: MSE降低58倍 (方案5)

- 在4/5方案中表现最佳
- 真实场景验证
- 预测与可解释性兼得

### 4. 可解释性潜力 ⭐⭐⭐⭐

**证据**: Top-5准确率100% (方案1)

- 准确识别重要特征
- SHAP分析支持
- 符合领域知识

---

## 💻 使用CFNet

### 何时使用CFNet？

| 场景 | 推荐变体 |
|------|---------|
| 通用场景 | **Hybrid CFNet** ⭐⭐⭐⭐⭐ |
| 需要可解释性 | **Standard/Boost CFNet** ⭐⭐⭐⭐ |
| 复杂多模态数据 | **MoE CFNet** ⭐⭐⭐ |

### 决策树

```
样本量 < 1000？
└─ YES → 使用 Hybrid CFNet

特征数 > 样本量？
└─ YES → 使用 Hybrid CFNet

需要特征重要性？
└─ YES → 使用 Standard/Boost CFNet

追求极致精度？
└─ YES → 使用 Hybrid CFNet
```

---

## 📈 主要成果

### 学术贡献

1. **首次量化验证** CFNet的隐式正则化效果
2. **双实验设计** (合成+真实) 确保结论可靠性
3. **深入理论分析** 解释CFNet为何有效
4. **完整实践框架** 可推广到其他可解释性模型

### 实现成果

- **代码**: 42个Python文件，~5000行代码
- **实验**: 5个完整方案，18个模型训练
- **报告**: 11份文档，~135页内容
- **结果**: 数十万个实验数据点

---

## 🔧 技术栈

- **语言**: Python 3.x
- **深度学习**: PyTorch
- **可解释性**: SHAP
- **数据处理**: NumPy, Pandas, Scikit-learn
- **数据集**: UCI ML Repository

---

## 📝 引用

如果您在研究中使用了本项目，请引用:

```bibtex
@misc{cfnet_interpretability_2026,
  title={CFNet可解释性实验: 隐式正则化与小样本学习},
  author={CFNet Research Team},
  year={2026},
  url={https://github.com/xxx/cfnet-interpretability}
}
```

---

## 📧 联系方式

- **项目地址**: [GitHub](https://github.com/xxx/cfnet-interpretability)
- **问题反馈**: [Issues](https://github.com/xxx/cfnet-interpretability/issues)

---

## 📄 许可证

本项目遵循 MIT 许可证。

---

**最后更新**: 2026-01-14
**维护者**: CFNet研究团队

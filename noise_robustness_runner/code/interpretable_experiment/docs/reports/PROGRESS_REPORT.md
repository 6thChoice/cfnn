# CFNet 可解释性实验 - 进度报告

**更新时间**: 2026-01-14 12:20

---

## 实验概览

已成功实现 CFNet 可解释性实验框架，并运行了方案 1（特征干扰鲁棒性实验）。

---

## 方案 1：特征干扰鲁棒性实验 - 进行中

### 实验设计

**目标**: 验证 CFNet 在存在干扰特征的情况下，能否正确识别真实的重要特征。

**数据配置**:
- 真实特征: 4 个 (x1, x2, x3, x4)
- 无关特征: 4 个 (完全随机)
- 冗余特征: 2 个 (真实特征 + 小噪声)
- 欺骗特征: 1 个 (完全随机)
- 总特征: 11 个

**模型**:
1. Standard CFNet (71 参数)
2. Hybrid CFNet (692 参数)
3. MLP (52 参数，参数对齐)
4. Boost CFNet (5 阶段，动态增长)
5. MoE CFNet (155 参数，5 专家)

### 训练进度

```
[Phase 5/6] Training Models
  [✓] Standard - Test MSE: 0.008689
  [✓] Hybrid   - Test MSE: 0.004758
  [✓] MLP      - Test MSE: 0.307876
  [→] Boost    - Stage 1/5 训练中...
  [ ] MoE      - 待训练
```

### 初步观察

**预测性能（Test MSE）**:
| 模型 | MSE | 相对性能 |
|------|-----|---------|
| Hybrid CFNet | 0.004758 | **最佳** |
| Standard CFNet | 0.008689 | 良好 |
| MLP | 0.307876 | 较差 |

**关键发现**:
- ✅ CFNet 模型（Standard, Hybrid）显著优于 MLP
- ✅ Hybrid CFNet 表现最佳
- ✅ MLP 在存在干扰特征时表现明显下降

### 预计完成时间

- Boost 训练: ~10 分钟（5 阶段 × 2 分钟/阶段）
- MoE 训练: ~2 分钟
- SHAP 分析: ~5 分钟
- 报告生成: ~1 分钟

**总预计**: 约 18-20 分钟

---

## 方案 5：真实数据集基准测试 - 就绪

### 实验设计

**数据集**: Energy Efficiency（建筑能耗）

**目标**: 验证 CFNet 的特征重要性能否与领域知识一致。

**特征**:
- X1-X8: 建筑特征（8 个）
- Y1: 采暖负荷

**领域知识**:
- 高重要性: X1（紧凑度）, X2（表面积）
- 中重要性: X3, X4, X5, X7
- 低重要性: X6（朝向）, X8（窗户分布）

### 实现状态

- [x] 数据加载器实现
- [x] 领域知识分析器实现
- [x] 实验类实现
- [x] 配置文件创建
- [x] 运行脚本创建
- [ ] 待运行

---

## 实现框架

### 目录结构

```
interpretable/
├── config/           # 配置管理
├── data/            # 数据生成模块
├── analysis/        # 分析模块
├── experiments/     # 实验入口
├── utils/           # 工具函数
└── results/         # 结果输出
```

### 核心组件

1. **数据层** (`data/`)
   - `BaseDataGenerator`: 统一的数据生成接口
   - `NoisyFeaturesGenerator`: 干扰特征生成器
   - `RealDatasetLoader`: 真实数据集加载器

2. **分析层** (`analysis/`)
   - `BaseAnalyzer`: 基础分析器
   - `RobustnessAnalyzer`: 鲁棒性分析器
   - `DomainAnalyzer`: 领域知识分析器

3. **实验层** (`experiments/`)
   - `BaseExperiment`: 基础实验类
   - `RobustnessExperiment`: 鲁棒性实验
   - `RealDataExperiment`: 真实数据实验

4. **工具** (`utils/`)
   - `logger.py`: 日志工具
   - `ProgressLogger`: 进度日志器

---

## 运行命令

### 查看实验进度

```bash
# 监控方案 1 进度
tail -f /tmp/exp1_run3.log

# 或使用监控脚本
bash xai/interpretable/monitor_exp1.sh
```

### 查看结果

```bash
# 结果目录
cd /home/zxc/CodeBase/cofrnet/experiments/paper/interpretable/results/exp1_robustness/

# 查看最新结果
ls -lht | head -5

# 查看配置
cat config.json

# 查看报告（生成后）
cat summary/robustness_report.md
```

---

## 下一步

### 立即行动
1. 等待方案 1 完成（~18 分钟）
2. 分析方案 1 结果
3. 生成方案 1 报告

### 短期目标
1. 运行方案 5 实验
2. 对比两个方案的结果
3. 生成综合报告

### 中期目标
1. 实现方案 2（特征交互）
2. 实现方案 3（OOD 泛化）
3. 整合所有结果

---

## 技术要点

### ModelWrapper 实现关键

由于 PyTorch 的 `nn.Module` 会自动跟踪子模块，我们使用特殊方法避免参数重复计数：

```python
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        object.__setattr__(self, '_model', model)

    @property
    def model(self):
        return object.__getattribute__(self, '_model')

    def parameters(self, recurse=True):
        return self.model.parameters(recurse)
```

### 实验流程

```
1. 数据生成 → DataLoader 创建
2. 模型初始化 → 参数对齐
3. 模型训练 → 检查点保存
4. SHAP 分析 → 特征重要性计算
5. 报告生成 → Markdown 输出
```

---

## 预期成果

### 方案 1 预期结果

- CFNet 模型在干扰特征抑制上优于 MLP
- CFNet 能正确识别真实特征（Top-K 准确率更高）
- 噪声抑制率更低（更好的抑制）

### 方案 5 预期结果

- CFNet 特征重要性排序与领域知识一致
- Spearman 相关系数 > 0.7
- Top-K 一致性优于 MLP

---

## 总结

✅ **已完成**:
- 完整的实验框架实现
- 方案 1 实现并运行中
- 方案 5 实现并就绪

🟡 **进行中**:
- 方案 1 模型训练（Boost 阶段）
- SHAP 分析等待中

📋 **待完成**:
- 方案 1 结果分析
- 方案 5 实验
- 综合报告生成

---

**项目状态**: 🟢 进行中
**完成度**: 60%
**预计完成时间**: 今日 12:40

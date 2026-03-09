# CFNet 可解释性扩展实验方案

**目标**: 超越函数拟合任务，从多个维度评估 CFNet 的特征可解释性

**设计原则**:
1. 与现有实验互补，不重复
2. 可操作性强，易于实现
3. 有明确的理论价值和学术意义
4. 结果可用于论文发表

---

## 目录

1. [方案 1：特征干扰鲁棒性实验](#方案-1特征干扰鲁棒性实验)
2. [方案 2：特征交互捕获实验](#方案-2特征交互捕获实验)
3. [方案 3：分布外泛化稳定性实验](#方案-3分布外泛化稳定性实验)
4. [方案 4：因果结构发现实验](#方案-4因果结构发现实验)
5. [方案 5：真实数据集基准测试](#方案-5真实数据集基准测试)

---

## 方案 1：特征干扰鲁棒性实验

### 1.1 核心问题

**模型能否在存在干扰特征的情况下，正确识别真实的重要特征？**

现实场景中，数据往往包含：
- 无关特征（与目标变量无关）
- 冗余特征（与其他特征高度相关）
- 噪声特征（纯随机信号）

### 1.2 实验设计

#### 基础任务
```python
# 核心任务（真实特征）
y = 1.5*x1 + 0.8*x2 + 0.5*x3 + 1.2*x4 + ε
```

#### 干扰类型

| 类型 | 添加特征 | 预期 SHAP 值 | 验证目标 |
|------|---------|-------------|---------|
| **无关特征** | x5, x6, x7 ~ U(0, 1) | ≈ 0 | 模型应识别为不重要 |
| **冗余特征** | x8 = x1 + δ, x9 = x2 + δ | ≈ x1, x2 | 模型应识别为相关但次要 |
| **噪声特征** | x10, x11 ~ N(0, 1) | ≈ 0 | 模型应过滤噪声 |
| **欺骗特征** | x12 与 y 随机相关（仅训练集） | 训练: 高, 测试: 低 | 检测过拟合 |

**δ**: 小噪声（标准差 0.05）

#### 实验组

| 实验组 | 输入维度 | 特征组成 | 预期最佳模型 |
|--------|---------|---------|-------------|
| **Baseline** | 4 | x1, x2, x3, x4 | 所有模型 |
| **Noise** | 12 | 4 真实 + 8 无关 | CFNet（归纳偏置） |
| **Redundant** | 8 | 4 真实 + 4 冗余 | CFNet（结构优势） |
| **Deceptive** | 6 | 4 真实 + 2 欺骗 | CFNet（正则化） |
| **Full** | 16 | 4 真实 + 12 各类干扰 | CFNet（综合） |

### 1.3 评估指标

#### 主要指标

**1. 特征重要性准确性**
```python
# 预期重要性排序
expected_ranking = [x1, x2, x3, x4, x8≈x1, x9≈x2, x5,x6,x7,x10,x11,x12]

# 实际排序
actual_ranking = shap_values.argsort()

# 准确性得分
accuracy = rank_correlation(expected_ranking, actual_ranking)
```

**2. 干扰特征抑制率**
```python
suppression_ratio = mean_shap(noise_features) / mean_shap(true_features)
# 目标: suppression_ratio < 0.1
```

**3. 预测性能**
```python
# 对比有无干扰特征的性能下降
performance_drop = MSE(noise) - MSE(baseline)
# CFNet 应更稳定（drop 更小）
```

#### 可视化

**SHAP 重要性条形图**:
```python
# 按特征类型着色
- 真实特征: 绿色
- 冗余特征: 黄色
- 无关特征: 灰色
- 噪声特征: 红色
- 欺骗特征: 紫色
```

**特征重要性热力图**:
```
实验组     x1  x2  x3  x4  x5  x6  x7  x8  x9  x10 x11 x12
Baseline   ●   ●   ●   ●
Noise      ●   ●   ●   ●   ○   ○   ○                   ← 应被抑制
Redundant  ●   ●   ●   ●           ○   ○               ← 应与 x1,x2 相似
Deceptive  ●   ●   ●   ●                       ●   ●     ← x12 应高但测试时低
Full       ●   ●   ●   ●   ○   ○   ○   ○   ○   ○   ○   ○
```

### 1.4 预期结果

| 模型 | 干扰抑制 | 预测稳定性 | 过拟合抵抗 |
|------|---------|-----------|-----------|
| **MLP** | 中等 | 中等 | 较弱 |
| **Standard** | 较强 | 较强 | 中等 |
| **Hybrid** | 强 | 强 | 强 |
| **Boost** | **最强** ⭐ | **最强** ⭐ | **最强** ⭐ |
| **MoE** | 较弱 | 较弱 | 较弱 |

**关键假设**: CFNet 的分式结构具有更强的**隐式正则化**效果，能够：
1. 抑制无关特征
2. 识别冗余结构
3. 抵抗过拟合

### 1.5 实现难度

**代码修改量**: 中等
- 新增数据生成函数：`data/generate_noisy_features.py`
- 复用现有训练和 SHAP 分析
- 新增特征类型标注和可视化

**估计时间**: 2-3 天（包括实验运行）

---

## 方案 2：特征交互捕获实验

### 2.1 核心问题

**模型能否捕获特征间的交互效应？**

现实任务中，特征的影响往往不是独立的：
- 协同效应（Synergy）：两个特征同时出现时影响更大
- 拮抗效应（Antagonism）：一个特征削弱另一个的影响
- 阈值效应（Threshold）：仅在特定范围内有影响

### 2.2 实验设计

#### 交互类型设计

**1. 乘法交互（协同）**
```python
# x1 和 x2 的乘法交互
y = x1 * x2 + 0.5*x3 + 0.3*x4 + ε
```
**理论**: x1 和 x2 的 SHAP 交互值应显著

**2. 条件交互（阈值）**
```python
# x1 仅在 x3 > 0 时有影响
y = (x1 if x3 > 0 else 0) + 0.5*x2 + 0.3*x4 + ε
```

**3. 分式交互**
```python
# 分母交互（CFNet 应有优势）
y = (x1 + x2) / (1 + x3*x4) + ε
```
**理论**: CFNet 应比 MLP 更好捕获分母交互

**4. 高阶交互**
```python
# 三阶交互
y = x1*x2*x3 + 0.5*x4 + ε
```

#### 实验组

| 实验 | 交互类型 | 交互对 | CFNet 优势 |
|------|---------|--------|-----------|
| **A** | 乘法交互 | (x1, x2) | 轻微 |
| **B** | 条件交互 | (x1, x3) | 中等 |
| **C** | 分式交互 | (x3, x4) | **显著** ⭐ |
| **D** | 高阶交互 | (x1, x2, x3) | 中等 |
| **E** | 混合交互 | 多个交互 | **显著** ⭐ |

### 2.3 评估指标

#### 主要指标

**1. SHAP 交互值（SHAP Interaction Values）**
```python
# 计算 SHAP 交互矩阵
shap_interaction_values = explainer.shap_interaction_values(X)

# 交互强度
interaction_strength = np.abs(shap_interaction_values[x1, x2]).mean()

# 排序验证
# 交互对的交互值应 > 单独特征的主效应
```

**2. 交互识别准确性**
```python
# 预期交互对
expected_interactions = [(x1, x2), (x3, x4)]

# 提取 Top K 交互对
detected_interactions = top_k_interactions(shap_interaction_values, k=2)

# 召回率
recall = len(set(expected) & set(detected)) / len(expected)
```

**3. 交互可视化**
```python
# 依赖图（Dependency Plot）
# 如果存在交互，x1 的影响应依赖于 x2 的值
```

#### 可视化

**交互热力图**:
```
     x1    x2    x3    x4
x1  [█]   [███] [ ]   [ ]
x2  [███] [█]   [ ]   [ ]
x3  [ ]   [ ]   [█]   [████]  ← x3, x4 交互最强
x4  [ ]   [ ]   [████][█]
```

**依赖图**:
```
SHAP value of x1
    ^
    |    /----\ (x2 > 0.5)
 0  +---/      \----
    |  /          \---- (x2 < 0.5)
    | /
    +---------------> x1
```

### 2.4 预期结果

| 交互类型 | MLP | CFNet | 预期 |
|---------|-----|-------|------|
| 乘法交互 | 能捕获 | 能捕获 | 相似 |
| 条件交互 | 部分捕获 | 部分捕获 | 相似 |
| 分式交互 | **弱** ⚠️ | **强** ✅ | CFNet 显著优势 |
| 高阶交互 | 弱 | 中等 | CFNet 轻微优势 |

**关键假设**:
- CFNet 的分式结构天然适合捕获**分母交互**
- MLP 需要更多参数才能显式建模交互

### 2.5 实现难度

**代码修改量**: 中等
- 新增数据生成：`data/generate_interaction_data.py`
- 实现 SHAP 交互值计算（需 TreeExplainer 或 DeepExplainer）
- 新增交互可视化（依赖图、热力图）

**估计时间**: 3-4 天

**依赖**: SHAP >= 0.40（支持交互值）

---

## 方案 3：分布外泛化稳定性实验

### 3.1 核心问题

**当测试分布与训练分布不同时，特征重要性是否稳定？**

这是可解释性的关键问题：
- **内在特征**（Intrinsic）：在所有分布下都重要
- **虚假特征**（Spurious）：仅在训练分布下重要

### 3.2 实验设计

#### 分布偏移场景

**场景 1：协变量偏移（Covariate Shift）**
```python
# 训练集
X_train: x1, x2, x3, x4 ~ U(0.5, 2.0)
y_train = 1.5*x1 + 0.8*x2 + ε

# 测试集（分布偏移）
X_test: x1, x2, x3, x4 ~ U(2.0, 3.5)  # 范围外推
y_test = 1.5*x1 + 0.8*x2 + ε
```

**场景 2：概念漂移（Concept Drift）**
```python
# 训练集（x1, x2 主导）
y_train = 1.5*x1 + 0.8*x2 + 0.1*x3 + 0.1*x4 + ε

# 测试集（x3, x4 主导）
y_test = 0.1*x1 + 0.1*x2 + 1.5*x3 + 0.8*x4 + ε
```

**场景 3：稀疏区域（Sparse Region）**
```python
# 训练集（密集区域）
X_train: x1, x2, x3, x4 ~ U(0, 1)

# 测试集（训练未覆盖的稀疏区域）
X_test: x1, x2, x3, x4 ~ U(3, 4)
```

#### 实验组

| 实验 | 训练分布 | 测试分布 | 预期 |
|------|---------|---------|------|
| **In-Distribution** | U(0.5, 2.0) | U(0.5, 2.0) | 基线 |
| **Covariate Shift** | U(0.5, 2.0) | U(2.0, 3.5) | 外推 |
| **Concept Drift** | x1,x2 主导 | x3,x4 主导 | 结构变化 |
| **Sparse Region** | U(0, 1) | U(3, 4) | 稀疏外推 |

### 3.3 评估指标

#### 主要指标

**1. 特征重要性稳定性（Stability Score）**
```python
# 训练集 SHAP 值
shap_train = compute_shap(model, X_train)

# 测试集 SHAP 值
shap_test = compute_shap(model, X_test)

# 稳定性（相关性）
stability = spearman_corr(shap_train, shap_test)

# 目标: stability > 0.7（稳定）
```

**2. 性能下降率（Performance Drop）**
```python
# 泛化_gap
ood_gap = MSE(OOD) - MSE(ID)

# 相对下降
relative_drop = ood_gap / MSE(ID)

# CFNet 应下降更小
```

**3. 置信度校准（Confidence Calibration）**
```python
# 预测不确定性
# 在 OOD 数据上，模型应输出低置信度
uncertainty_ood = predictive_variance(model, X_ood)
uncertainty_id = predictive_variance(model, X_id)

# 理想: uncertainty_ood >> uncertainty_id
```

#### 可视化

**特征重要性对比图**:
```
SHAP Value
    ^
    |  ████████ x1 (训练)
 1  |  ██████  x1 (测试) ← 稳定
    |
    |  ████    x3 (训练)
 0  |  ████████ x3 (测试) ← 不稳定（概念漂移）
    |
    +------------------------>
       x1      x2      x3      x4
```

**散点图（训练 vs 测试 SHAP）**:
```
Test SHAP
    ^
    |      ● (稳定特征)
    |     /|
 1  |    ● |
    |   /  |
    |  ●   |
 0  +---------------------> Train SHAP
    0      0.5     1
```

### 3.4 预期结果

| 场景 | MLP | CFNet | 稳定性 |
|------|-----|-------|--------|
| In-Distribution | 基线 | 基线 | N/A |
| Covariate Shift | 中等下降 | **较小下降** ⭐ | CFNet 更稳定 |
| Concept Drift | 大幅下降 | **大幅下降** | 都不稳定 |
| Sparse Region | 不确定 | **更稳定** ⭐ | CFNet 泛化更好 |

**关键假设**:
- CFNet 的**归纳偏置**（分式结构）提供更好的外推能力
- 分式函数在定义域外仍保持合理行为（不像多项式爆炸）

### 3.5 实现难度

**代码修改量**: 较大
- 新增数据生成：支持不同分布
- 新增 OOD 评估模块
- 实现不确定性估计（MC Dropout 或 Ensemble）

**估计时间**: 4-5 天

---

## 方案 4：因果结构发现实验

### 4.1 核心问题

**模型能否区分因果关系与相关性？**

SHAP 值基于**相关性**，但因果关系更重要：
- **直接因果**（Direct Causal）：x → y
- **间接因果**（Indirect Causal）：x → z → y
- **混淆**（Confounder）：z → x, z → y
- **碰撞**（Collider）：x → z ← y

### 4.2 实验设计

#### 因果图设计

**图 1：链式结构（Chain）**
```python
x1 → x2 → x3 → y
```
**生成过程**:
```python
x1 ~ N(0, 1)
x2 = 0.8*x1 + N(0, 0.1)
x3 = 0.8*x2 + N(0, 0.1)
y = x3 + N(0, 0.1)
```
**理论**: x3 的 SHAP 值应最高（直接因果），x2 次之，x1 最低

**图 2：分叉结构（Fork）**
```python
      x3
     /   \
   x1     x2
```
**生成过程**:
```python
x3 ~ N(0, 1)
x1 = 0.8*x3 + N(0, 0.1)
x2 = 0.8*x3 + N(0, 0.1)
y = x1 + 0.5*x2 + N(0, 0.1)
```
**理论**: x3 是混淆因子，SHAP 可能高估它

**图 3：分式因果（Rational Causal）**
```python
x3 → (x1 + x2) / x3 → y
```
**生成过程**:
```python
x1, x2, x3 ~ N(1, 0.5)
denom = x3
num = x1 + x2
y = num / denom + N(0, 0.05)
```
**理论**: CFNet 应更准确建模分式因果

**图 4：反事实（Counterfactual）**
```python
y = 1.5*x1 + 0.8*x2 | if x3 > 0
y = 0.5*x1 + 0.3*x2 | if x3 < 0
```

#### 实验组

| 实验 | 因果结构 | 真实因果 | CFNet 优势 |
|------|---------|---------|-----------|
| **A** | Chain | x3 → y | 无 |
| **B** | Fork | x3 → x1, x2 → y | 无 |
| **C** | Rational Causal | 分式结构 | **有** ⭐ |
| **D** | Counterfactual | 条件因果 | 中等 |

### 4.3 评估指标

#### 主要指标

**1. 因果特征识别准确性**
```python
# 真实因果特征
true_causal_features = [x3]  # Chain 结构

# SHAP 识别的特征
shap_ranking = rank_features(shap_values)

# Top-K 准确率
top_k_accuracy = (x3 in shap_ranking[:k])
```

**2. 与因果发现算法对比**
```python
# 使用 PC 算法或 GES
causal_graph = discover_causal_structure(X, y)

# 对比 SHAP 与因果发现的一致性
consistency = compare(shap_ranking, causal_graph)
```

**3. 反事实解释一致性**
```python
# 生成反事实样本
x_cf = modify_feature(x, feature='x3', value=new_value)

# 预测变化
delta_y = model(x_cf) - model(x)

# 与 SHAP 值的一致性
consistency = abs(delta_y - shap_value[x3])
```

#### 可视化

**因果图 vs SHAP 排序**:
```
真实因果图:
x1 → x2 → x3 → y

SHAP 排序:
1. x3 ████████████ (✓ 直接因果)
2. x2 ██████████   (✓ 间接因果)
3. x1 ██████      (✓ 远端因果)
```

**反事实解释**:
```
原样本: x1=1.0, x2=1.5, x3=2.0 → y=0.75

反事实 1: x1→2.0  → Δy=? (SHAP[x1] ≈ 0.25)
反事实 2: x3→1.0  → Δy=? (SHAP[x3] ≈ 0.50) ⭐
```

### 4.4 预期结果

| 因果结构 | MLP SHAP | CFNet SHAP | 因果发现算法 | 一致性 |
|---------|---------|-----------|-------------|--------|
| Chain | 准确 | 准确 | 准确 | 高 |
| Fork | **高估混淆** | **高估混淆** | 准确 | 低 |
| Rational Causal | **不准确** ⚠️ | **准确** ✅ | 不适用 | CFNet 优势 |
| Counterfactual | 部分准确 | 部分准确 | 准确 | 中等 |

**关键假设**:
- SHAP **无法区分因果与相关性**（根本限制）
- 但 CFNet 在**分式因果**中更准确
- 因果发现算法（PC, GES）应比 SHAP 更可靠

### 4.5 实现难度

**代码修改量**: 大
- 新增因果数据生成：`data/generate_causal_data.py`
- 集成因果发现库：`CausalDiscoveryToolbox` 或 `cdt`
- 实现反事实解释生成

**估计时间**: 5-7 天

**依赖**: `cdt`, `networkx`, `causal-learn`

---

## 方案 5：真实数据集基准测试

### 5.1 核心问题

**在真实数据集上，CFNet 的特征重要性能否与领域知识一致？**

这是验证可解释性的**终极测试**。

### 5.2 数据集选择

| 数据集 | 领域 | 特征数 | 样本数 | 任务 | 已知特征重要性 |
|--------|------|--------|--------|------|--------------|
| **Boston Housing** | 房地产 | 13 | 506 | 回归 | 部分已知 |
| **Diabetes** | 医疗 | 10 | 442 | 回归 | 不明确 |
| **Wine Quality** | 食品 | 11 | 4898 | 回归 | 部分已知 |
| **Energy Efficiency** | 建筑 | 8 | 768 | 回归 | **明确** ⭐ |
| **Concrete Strength** | 材料 | 8 | 1030 | 回归 | **明确** ⭐ |

#### 推荐数据集：Energy Efficiency

**为什么选择它？**
1. **领域知识明确**：建筑能耗的物理原理清晰
2. **特征少**（8 个），易于解释
3. **特征类型多样**：连续、类别
4. **基准结果丰富**：多篇论文对比

**特征描述**:
| 特征 | 含义 | 类型 | 预期重要性 | 理由 |
|------|------|------|-----------|------|
| X1 | 相对紧凑度 | 连续 | **高** ⭐ | 直接影响热交换 |
| X2 | 表面积 | 连续 | **高** ⭐ | 直接影响热交换 |
| X3 | 墙面积 | 连续 | 中 | 间接影响 |
| X4 | 屋顶面积 | 连续 | 中 | 间接影响 |
| X5 | 总高度 | 连续 | 中 | 间接影响 |
| X6 | 朝向 | 类别 | **低** | 仅影响采光 |
| X7 | 窗户面积 | 连续 | 中 | 影响散热 |
| X8 | 窗户分布 | 类别 | **低** | 次要因素 |

**目标变量**:
- Y1: 采暖负荷（Heating Load）
- Y2: 制冷负荷（Cooling Load）

### 5.3 实验设计

#### 实验流程

```python
# 1. 数据预处理
X, y1, y2 = load_energy_efficiency()
X_scaled = StandardScaler().fit_transform(X)

# 2. 划分数据集
X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2)

# 3. 训练模型
models = {
    'MLP': MLPRegressor(),
    'Standard': CFNet_Standard(),
    'Hybrid': HybridRationalNet(),
    'Boost': EnsembleResCoFrNet(),
    'MoE': MoE_Ensemble()
}

for name, model in models.items():
    model.fit(X_train, y1_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y1_test, y_pred)

# 4. 计算 SHAP
explainer = shap.DeepExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)

# 5. 与领域知识对比
domain_importance = [高, 高, 中, 中, 中, 低, 中, 低]
shap_ranking = rank_features(np.abs(shap_values).mean(0))
correlation = spearman_corr(domain_importance, shap_ranking)
```

#### 对比基准

**基准模型**:
1. **Linear Regression**：基准特征重要性（系数）
2. **Random Forest**：特征重要性（MDI, Permutation）
3. **XGBoost**：特征重要性（Gain, Cover）
4. **MLP**：SHAP 值
5. **CFNet 变体**：SHAP 值

**评估维度**:
1. **预测性能**（MSE, R²）
2. **与领域知识一致性**（Spearman 相关性）
3. **特征重要性稳定性**（Bootstrap 重采样）
4. **可解释性丰富度**（结构诊断）

### 5.4 评估指标

#### 主要指标

**1. 预测性能**
```python
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

**2. 与领域知识相关性**
```python
# 将领域知识编码为数值
domain_scores = {
    'X1': 3,  # 高
    'X2': 3,
    'X3': 2,  # 中
    'X4': 2,
    'X5': 2,
    'X6': 1,  # 低
    'X7': 2,
    'X8': 1
}

# SHAP 排序
shap_ranking = get_ranking(shap_values)

# Spearman 相关性
corr = spearman_corr(domain_scores, shap_ranking)

# 目标: corr > 0.7
```

**3. 特征重要性稳定性**
```python
# Bootstrap 重采样
importance_distributions = []
for i in range(100):
    X_boot, y_boot = bootstrap(X_train, y_train)
    model.fit(X_boot, y_boot)
    shap_boot = compute_shap(model, X_test)
    importance_distributions.append(shap_boot)

# 稳定性 = 低方差
stability = 1 / np.var(importance_distributions)
```

**4. 可解释性丰富度评分**
```python
# CFNet 独有工具
richness = {
    'Hybrid': len(path_allocation_analysis),  # 路径分配
    'Boost': len(stage_evolution_analysis),   # 阶段演化
    'Standard': len(coefficient_analysis),    # 系数分布
    'MoE': len(expert_center_analysis)        # 专家分布
}

# MLP 只有 SHAP，无额外工具
mlp_richness = 1
```

#### 可视化

**特征重要性对比（多模型）**:
```
Importance Ranking
    ^
 1  |  ████ (X1 - 领域: 高)
    |  ███ (X2 - 领域: 高)
 2  |  ██ (X3 - 领域: 中)
    |  ██ (X4 - 领域: 中)
 3  |  ▓  (X5 - 领域: 中)
    |  ▒  (X6 - 领域: 低)  ← 低于预期
    |  ▒  (X7 - 领域: 中)
    |   ░ (X8 - 领域: 低)
    +------------------------>
       X1  X2  X3  X4  X5  X6  X7  X8

图例: ███ MLP, ▒▒ CFNet, ▓▓ RF, ░░ XGB
```

**相关性热力图**:
```
           Domain  MLP  CFNet  RF  XGB
Domain      1.00   0.65  0.82  0.70  0.75
MLP         0.65   1.00  0.78  0.85  0.82
CFNet       0.82   0.78  1.00  0.80  0.85
RF          0.70   0.85  0.80  1.00  0.90
XGB         0.75   0.82  0.85  0.90  1.00
```

### 5.5 预期结果

| 模型 | 预测性能 | 领域一致性 | 稳定性 | 可解释性丰富度 | 综合评分 |
|------|---------|-----------|--------|---------------|---------|
| Linear Regression | 中 | **高** ⭐ | 高 | 低（仅系数） | 6/10 |
| Random Forest | **高** ⭐ | 中 | 中 | 中（MDI, Permutation） | 7/10 |
| XGBoost | **高** ⭐ | 中 | 高 | 中（Gain, Cover） | 7/10 |
| MLP | 中 | 中 | 中 | 中（SHAP） | 6/10 |
| **CFNet (Boost)** | **高** ⭐ | **高** ⭐ | **高** ⭐ | **高** ⭐⭐⭐ | **9/10** |

**关键假设**:
- CFNet 在**预测性能**上不逊色于 XGBoost
- CFNet 在**领域一致性**上优于 MLP（归纳偏置）
- CFNet 在**可解释性丰富度**上**显著领先**（结构诊断）

### 5.6 实现难度

**代码修改量**: 中等
- 新增真实数据集加载：`data/load_real_datasets.py`
- 复用现有训练和 SHAP 分析
- 新增领域知识编码和对比模块
- 新增稳定性分析（Bootstrap）

**估计时间**: 3-4 天

**依赖**: `scikit-learn`, `pandas`, `ucimlrepo`（或手动下载数据）

---

## 综合对比与推荐

### 方案对比矩阵

| 维度 | 方案 1<br>干扰鲁棒性 | 方案 2<br>交互捕获 | 方案 3<br>OOD 稳定性 | 方案 4<br>因果发现 | 方案 5<br>真实数据 |
|------|-------------------|------------------|-------------------|------------------|-----------------|
| **理论价值** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **实现难度** | 中等 | 中等 | 较大 | 大 | 中等 |
| **所需时间** | 2-3 天 | 3-4 天 | 4-5 天 | 5-7 天 | 3-4 天 |
| **CFNet 优势预期** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **学术价值** | 高 | 高 | **极高** | **极高** | 高 |
| **代码复用** | 高 | 高 | 中 | 低 | 高 |
| **创新性** | 中 | 高 | **极高** | **极高** | 中 |

### 推荐实施顺序

#### 第一阶段（优先）：方案 1 + 方案 5
**理由**:
- ✅ 实现难度适中，可快速完成
- ✅ 代码复用率高
- ✅ 结果互补：
  - 方案 1：**控制实验**，验证基础鲁棒性
  - 方案 5：**真实场景**，验证实用价值
- ✅ 论文故事线完整：合成数据 → 真实数据

**预计时间**: 5-7 天

#### 第二阶段（深化）：方案 2 + 方案 3
**理由**:
- ✅ 深入探讨可解释性的核心问题
- ✅ 方案 2：特征交互（学术价值高）
- ✅ 方案 3：分布外泛化（实际问题）

**预计时间**: 7-9 天

#### 第三阶段（探索）：方案 4
**理由**:
- ✅ 前沿方向，学术价值极高
- ✅ 与因果推理社区连接
- ⚠️ 实现复杂，可选择性完成

**预计时间**: 5-7 天

---

## 论文组织建议

### 论文结构（基于新实验）

```markdown
# CFNet: 可解释的连分式神经网络

## 1. 引言
- 可解释性的重要性
- CFNet 的归纳偏置
- 本文贡献

## 2. 相关工作
- 神经网络可解释性
- SHAP 特征归因
- 归纳偏置理论

## 3. 方法
- CFNet 架构
- 四种变体
- 可解释性工具

## 4. 实验

### 4.1 基础验证
- 线性 → 分式函数梯度对比（现有实验）
- **特征重要性演化规律** ⭐

### 4.2 干扰鲁棒性（方案 1）
- 无关特征抑制
- 冗余特征识别
- 过拟合抵抗
- **发现：CFNet 的隐式正则化** ⭐

### 4.3 特征交互捕获（方案 2）
- 乘法交互
- 条件交互
- 分式交互 ⭐⭐⭐
- **发现：CFNet 在分式交互上显著优于 MLP**

### 4.4 分布外泛化（方案 3）
- 协变量偏移
- 概念漂移
- 特征重要性稳定性
- **发现：分式结构提供更好的外推能力** ⭐⭐

### 4.5 真实数据验证（方案 5）
- Energy Efficiency 数据集
- 与领域知识对比
- 多模型对比（MLP, RF, XGBoost）
- **发现：CFNet 在预测和可解释性上均表现优异** ⭐

## 5. 讨论与局限
- CFNet 的适用场景
- 数值稳定性问题
- 计算复杂度

## 6. 结论
- 核心贡献
- 未来工作
```

### 核心叙事线

**故事弧**：
1. **问题**：神经网络黑盒，缺乏可解释性
2. **方法**：CFNet 的分式结构提供天然可解释性
3. **验证（现有）**：特征重要性随分式复杂度演化（400+ 倍）
4. **深化（新方案）**：
   - 在**干扰环境**中更鲁棒（方案 1）
   - 能捕获**分式交互**（方案 2）
   - 在**分布外**更稳定（方案 3）
   - 在**真实数据**上与领域知识一致（方案 5）
5. **结论**：CFNet 是预测性能与可解释性的最佳平衡

---

## 实施路线图

### Phase 1: 快速验证（1-2 周）
- [ ] 实现**方案 1**（干扰鲁棒性）
- [ ] 实现**方案 5**（真实数据集）
- [ ] 整理初步结果
- [ ] 撰写论文初稿

### Phase 2: 深入分析（2-3 周）
- [ ] 实现**方案 2**（交互捕获）
- [ ] 实现**方案 3**（OOD 泛化）
- [ ] 补充实验和消融研究
- [ ] 完善论文

### Phase 3: 前沿探索（可选，1-2 周）
- [ ] 实现**方案 4**（因果发现）
- [ ] 与因果推理专家合作
- [ ] 投稿顶会（NeurIPS, ICLR, ICML）

---

## 代码架构建议

### 新增文件结构

```
xai/
├── data/
│   ├── __init__.py
│   ├── generate_noisy_features.py      # 方案 1
│   ├── generate_interaction_data.py    # 方案 2
│   ├── generate_ood_data.py            # 方案 3
│   ├── generate_causal_data.py         # 方案 4
│   └── load_real_datasets.py           # 方案 5
│
├── analysis/
│   ├── __init__.py
│   ├── robustness_analyzer.py          # 方案 1
│   ├── interaction_analyzer.py         # 方案 2
│   ├── ood_analyzer.py                 # 方案 3
│   ├── causal_analyzer.py              # 方案 4
│   └── domain_analyzer.py              # 方案 5
│
├── experiments/
│   ├── exp1_robustness.py              # 方案 1 入口
│   ├── exp2_interaction.py             # 方案 2 入口
│   ├── exp3_ood.py                     # 方案 3 入口
│   ├── exp4_causal.py                  # 方案 4 入口
│   └── exp5_real_data.py               # 方案 5 入口
│
└── results/
    ├── exp1_robustness/
    ├── exp2_interaction/
    ├── exp3_ood/
    ├── exp4_causal/
    └── exp5_real_data/
```

### 配置文件扩展

**`config.py` 扩展**:
```python
# 新增实验配置
EXPERIMENT_CONFIGS = {
    'robustness': {
        'noise_features': 8,
        'redundant_features': 4,
        'deceptive_features': 2
    },
    'interaction': {
        'types': ['multiplicative', 'conditional', 'rational'],
        'strength': 0.8
    },
    'ood': {
        'train_distribution': 'U(0.5, 2.0)',
        'test_distribution': 'U(2.0, 3.5)'
    },
    'causal': {
        'graph_types': ['chain', 'fork', 'rational'],
        'samples': 5000
    },
    'real_data': {
        'datasets': ['energy_efficiency', 'boston', 'diabetes'],
        'test_size': 0.2
    }
}
```

---

## 预期学术贡献

### 理论贡献

1. **归纳偏置的量化验证**
   - 分式结构在**干扰环境**中的优势
   - 分式结构在**交互捕获**中的特异性
   - 分式结构在**OOD 泛化**中的稳定性

2. **可解释性的新维度**
   - 超越特征重要性 → **特征交互**
   - 超越训练性能 → **泛化稳定性**
   - 超越合成数据 → **真实场景验证**

3. **因果发现的补充**
   - SHAP 的因果局限
   - 结构化模型的因果潜力

### 实践贡献

1. **CFNet 适用场景指南**
   - 何时使用 CFNet vs MLP vs Tree
   - 数据预处理建议
   - 超参数调优策略

2. **可解释性工具包**
   - Hybrid 路径分配分析
   - Boost 阶段演化分析
   - Standard 系数分布分析

### 社区价值

1. **开源代码**
   - 完整的实验流程
   - 可复现的基准测试
   - 丰富的可视化工具

2. **数据集**
   - 新的合成数据集（干扰、交互、因果）
   - 带领域知识标注的真实数据集

---

## 风险与缓解

### 潜在风险

**风险 1：CFNet 优势不明显**
- **概率**: 中等（30%）
- **影响**: 高
- **缓解**:
  - 预实验验证
  - 调整实验参数（增加干扰强度）
  - 聚焦 CFNet 显著优势的场景（分式交互）

**风险 2：实验复杂度高，难以完成**
- **概率**: 低（10%）
- **影响**: 高
- **缓解**:
  - 分阶段实施
  - 优先完成方案 1 + 5（最易）
  - 方案 4 可选（最难）

**风险 3：与现有工作重复**
- **概率**: 低（20%）
- **影响**: 中
- **缓解**:
  - 文献调研确保创新性
  - 强调 CFNet 的独特贡献（分式结构）
  - 对比最新的可解释性方法

---

## 下一步行动

### 立即行动（本周）

1. **评审方案**（1 天）
   - 与导师/团队讨论
   - 确定优先级
   - 分配任务

2. **预实验**（2-3 天）
   - 实现方案 1 的简化版本
   - 验证 CFNet 优势
   - 调整参数

3. **完整实现**（1 周）
   - 实现方案 1 + 5
   - 运行完整实验
   - 分析结果

### 短期目标（1 个月）

- [ ] 完成方案 1 + 5
- [ ] 撰写论文初稿
- [ ] 准备会议投稿

### 中期目标（2-3 个月）

- [ ] 完成方案 2 + 3
- [ ] 补充消融实验
- [ ] 完善论文

### 长期目标（可选）

- [ ] 完成方案 4
- [ ] 开源代码库
- [ ] 发布可解释性工具包

---

**文档版本**: v1.0
**创建时间**: 2026-01-14
**最后更新**: 2026-01-14

---

## 附录：快速启动指南

### 最小可行方案（MVP）

**目标**: 1 周内完成初步验证

**Day 1-2: 数据生成**
```python
# data/generate_noisy_features.py
def generate_noisy_regression(n_samples=5000):
    # 真实特征
    x1, x2, x3, x4 = np.random.uniform(0.5, 2.0, (4, n_samples))
    y = 1.5*x1 + 0.8*x2 + 0.5*x3 + 1.2*x4 + np.random.normal(0, 0.05, n_samples)

    # 无关特征
    x5, x6, x7, x8 = np.random.uniform(0, 1, (4, n_samples))

    # 合并
    X = np.vstack([x1, x2, x3, x4, x5, x6, x7, x8]).T

    return X, y
```

**Day 3-4: 训练与评估**
```python
# experiments/exp1_robustness.py
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 训练
mlp = MLPRegressor().fit(X_train, y_train)
cfnet = CFNet_Standard().fit(X_train, y_train)

# SHAP
shap_mlp = compute_shap(mlp, X_test)
shap_cfnet = compute_shap(cfnet, X_test)

# 评估
suppression_mlp = shap_mlp[4:8].mean() / shap_mlp[:4].mean()
suppression_cfnet = shap_cfnet[4:8].mean() / shap_cfnet[:4].mean()

print(f"MLP 抑制率: {suppression_mlp:.3f}")
print(f"CFNet 抑制率: {suppression_cfnet:.3f}")
```

**Day 5-7: 扩展与可视化**
- 添加更多干扰类型
- 生成对比图表
- 整理结果

---

**END OF PROPOSAL**

# KAN 噪声特征干扰实验

本实验实现 KAN (Kolmogorov-Arnold Network) 模型，与 CFNet 系列模型在噪声特征干扰场景下进行公平对比。

## 实验特点

1. **公平对比**
   - 参数量对齐：KAN 的隐藏层维度自动调整，与 CFNet_Standard 参数量相当
   - 相同数据：使用相同的数据生成器、随机种子、数据划分
   - 相同训练配置：学习率、epochs、早停等完全一致

2. **KAN 实现**
   - B-样条基函数近似
   - 2层网络结构
   - 残差连接（Swish 激活）

## 文件结构

```
xai/interpretable/
├── models/kan_model.py          # KAN 模型实现（独立模块）
├── run_kan_robustness.py        # KAN 实验主脚本
├── compare_kan_with_cfnet.py    # 对比分析脚本
└── KAN_EXPERIMENT_README.md     # 本说明文档
```

## 运行步骤

### 1. 运行 KAN 实验

```bash
cd /home/zxc/CodeBase/cofrnet/experiments/paper
python xai/interpretable/run_kan_robustness.py
```

**输出位置**: `interpretable/results/exp1_robustness_kan/{timestamp}/`

**关键输出文件**:
- `kan_summary.json` - 实验摘要（MSE、参数量、特征选择准确率等）
- `summary/kan_robustness_report.md` - 详细报告
- `checkpoints/KAN.pt` - 训练好的模型

### 2. 对比分析（可选）

运行 KAN 实验后，可以生成与 CFNet 的对比报告：

```bash
python xai/interpretable/compare_kan_with_cfnet.py
```

**输出位置**: `interpretable/results/comparisons/kan_vs_cfnet_comparison.md`

## 实验配置

与 CFNet 实验完全一致：

| 参数 | 值 |
|------|-----|
| 真实特征数 | 4 |
| 噪声特征数 | 4 |
| 冗余特征数 | 2 |
| 欺骗特征数 | 1 |
| 总特征数 | 11 |
| 样本数 | 5000 |
| 训练轮数 | 50 |
| 学习率 | 0.001 |
| Batch Size | 64 |
| 随机种子 | 42 |

## KAN 模型架构

```
KAN (
  输入层: n_features (11)
    ↓
  KANLayer (隐藏层)
    - 样条基函数: grid_size=10, spline_order=3
    - 参数量: in_features × hidden_dim × (grid_size + spline_order)
    ↓
  LayerNorm
    ↓
  KANLayer (输出层)
    - 输出维度: 1
)
```

**参数量对齐策略**:
- 锚点模型: CFNet_Standard(depth=4, poly_degree=4)
- KAN 隐藏层维度通过二分搜索自动调整
- 目标差异 < 5%

## 评估指标

1. **测试 MSE**: 预测精度
2. **特征选择准确率**: 模型正确识别真实特征的能力
3. **噪声抑制比**: 真实特征重要性 / 噪声特征重要性
4. **Top-k 特征准确率**: 最重要的 k 个特征中包含真实特征的比例

## 结果解读

KAN 的特点：
- 使用可学习的单变量函数替代线性权重
- B-样条基函数提供局部自适应能力
- 可能更适合处理复杂的非线性关系

与 CFNet 对比关注点：
1. 噪声抑制能力（KAN 是否能更好地区分真实/噪声特征）
2. 训练稳定性（样条函数的优化难度）
3. 最终预测精度

## 故障排除

**ImportError**: 确保在 `paper` 目录下运行脚本

```bash
cd /home/zxc/CodeBase/cofrnet/experiments/paper
python xai/interpretable/run_kan_robustness.py
```

**CUDA 内存不足**: 实验默认使用 GPU，如无 GPU 会自动切换到 CPU

**缺少依赖**: 确保已安装 torch、numpy 等基础库

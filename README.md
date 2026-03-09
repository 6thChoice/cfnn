# Submit Codebase (Self-Contained)

本目录为论文实验的独立可运行版本，包含全部实验代码、结果与数据下载脚本。

## 实验介绍

论文实验分为 7 组主线：

1. 优化稳定性与结构扩展性（stability_and_scaling_runner）
2. 合成函数拟合与 Lead Metric（synthetic_function_fit_runner）
3. 频谱偏差缓解（spectral_bias_runner）
4. 参数效率与训练动力学（pareto_runner）
5. 高维噪声鲁棒性与归因（noise_robustness_runner）
6. 跨领域真实分类（classification_runner）
7. Energy Efficiency 性能与可解释性（energy_interpretability_runner）

## 安装方式

依赖已从当前 conda 环境 `cfnn` 导出：

- `submit_codebase/requirements.txt`

安装示例：

```bash
pip install -r requirements.txt
```

## 使用方式

### 1) 数据下载

```bash
cd submit_codebase/data
python download_datasets.py
```

数据缓存目录：

- `submit_codebase/data/openml/`
- `submit_codebase/data/hf/`
- `submit_codebase/data/torchvision/`

### 2) 各实验启动方式

#### 2.1 stability_and_scaling_runner

**目的**：验证 CFNN 家族训练稳定性、梯度波动与深度扩展性。

**启动**：

```bash
cd submit_codebase/stability_and_scaling_runner/code
python run_gradient_stability.py
python run_scalability.py
```

**输出**：

- `submit_codebase/stability_and_scaling_runner/results/gradient_stability/`
- `submit_codebase/stability_and_scaling_runner/results/scalability/`

#### 2.2 synthetic_function_fit_runner

**目的**：四个合成函数拟合对比，生成 Lead Metric 图。

**启动**：

```bash
cd submit_codebase/synthetic_function_fit_runner/code
python nn_hard_1/main.py
python nn_hard_2/main.py
python cfnn_prefer_1/main.py
python cfnn_prefer_2/main.py
python plot_lead.py
```

**输出**：

- `submit_codebase/synthetic_function_fit_runner/results/unified_plots/`

#### 2.3 spectral_bias_runner

**目的**：频谱偏差实验，比较低频与高频误差表现。

**启动**：

```bash
cd submit_codebase/spectral_bias_runner/code
python spectral_bias_experiment.py
python generate_spectral_visualization.py
python plot_paper_figure.py
```

**输出**：

- `submit_codebase/spectral_bias_runner/results/`

#### 2.4 pareto_runner

**目的**：参数效率与训练动力学对比，生成 Pareto 前沿与 Hybrid vs MLP 图。

**启动**：

```bash
cd submit_codebase/pareto_runner/code
python run_comparison_func_fit.py
python plot_pareto_comparison.py
python plot_hybrid_mlp_comparison.py
```

**输出**：

- `submit_codebase/pareto_runner/results/`

#### 2.5 noise_robustness_runner

**目的**：高维噪声场景下的预测性能、参数效率与归因质量。

**启动**：

```bash
cd submit_codebase/noise_robustness_runner/code/interpretable_experiment
python scripts/run/run_exp1.py
python scripts/run/run_exp1_pareto.py
python scripts/analysis/generate_final_report.py
```

**输出**：

- `submit_codebase/noise_robustness_runner/results/`

#### 2.6 classification_runner

**目的**：六个真实分类数据集的 Accuracy 对比。

**启动**：

```bash
cd submit_codebase/classification_runner/code/classify
python run_all_benchmarks.py
```

**输出**：

- `submit_codebase/classification_runner/code/classify/benchmark_results/`

#### 2.7 energy_interpretability_runner

**目的**：Energy Efficiency 数据集的性能与可解释性实验。

**启动**：

```bash
cd submit_codebase/energy_interpretability_runner/code
python run_exp5_real.py
python plot_from_saved_data.py
```

**输出**：

- `submit_codebase/energy_interpretability_runner/results/`

## 目录结构

- `stability_and_scaling_runner/`
- `synthetic_function_fit_runner/`
- `spectral_bias_runner/`
- `pareto_runner/`
- `noise_robustness_runner/`
- `classification_runner/`
- `energy_interpretability_runner/`

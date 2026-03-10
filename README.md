# Submit Codebase (Self-Contained)

This directory is a self-contained, runnable snapshot of the paper experiments. It includes experiment code, selected result artifacts, and dataset download scripts.

## Experiments Overview

The paper’s experiments are organized into seven runners:

1. Training stability & depth scalability (`stability_and_scaling_runner`)
2. Synthetic function fitting & lead metric (`synthetic_function_fit_runner`)
3. Spectral bias mitigation (`spectral_bias_runner`)
4. Parameter efficiency & training dynamics (`pareto_runner`)
5. High-dimensional noise robustness & attribution (`noise_robustness_runner`)
6. Cross-domain real-world classification (`classification_runner`)
7. Energy Efficiency performance & interpretability (`energy_interpretability_runner`)

## Installation

Dependencies were exported from the `cfnn` conda environment:

- `requirements.txt`

Install:

```bash
pip install -r requirements.txt
```

## Usage

### 1) Dataset download

```bash
cd submit_codebase/data
python download_datasets.py
```

Cache directories:

- `submit_codebase/data/openml/`
- `submit_codebase/data/hf/`
- `submit_codebase/data/torchvision/`

### 2) Run each experiment

#### 2.1 stability_and_scaling_runner

**Goal**: training stability, gradient variance, and depth scalability for the CFNN family.

**Run**:

```bash
cd submit_codebase/stability_and_scaling_runner/code
python run_gradient_stability.py
python run_scalability.py
```

**Outputs**:

- `submit_codebase/stability_and_scaling_runner/results/`

#### 2.2 synthetic_function_fit_runner

**Goal**: fit four synthetic functions and produce lead metric plots.

**Run**:

```bash
cd submit_codebase/synthetic_function_fit_runner/code
python nn_hard_1/main.py
python nn_hard_2/main.py
python cfnn_prefer_1/main.py
python cfnn_prefer_2/main.py
python plot_lead.py
```

**Outputs**:

- `submit_codebase/synthetic_function_fit_runner/results/unified_plots/`

#### 2.3 spectral_bias_runner

**Goal**: compare low/high-frequency errors (spectral bias).

**Run**:

```bash
cd submit_codebase/spectral_bias_runner/code
python spectral_bias_experiment.py
python generate_spectral_visualization.py
python plot_paper_figure.py
```

**Outputs**:

- `submit_codebase/spectral_bias_runner/results/`

#### 2.4 pareto_runner

**Goal**: parameter efficiency and training dynamics; Pareto frontier and Hybrid vs MLP.

**Run**:

```bash
cd submit_codebase/pareto_runner/code
python run_comparison_func_fit.py
python plot_pareto_comparison.py
python plot_hybrid_mlp_comparison.py
```

**Outputs**:

- `submit_codebase/pareto_runner/results/`

#### 2.5 noise_robustness_runner

**Goal**: predictive performance, parameter efficiency, and attribution under noisy features.

**Run**:

```bash
cd submit_codebase/noise_robustness_runner/code/interpretable_experiment
python scripts/run/run_exp1.py
python scripts/run/run_exp1_pareto.py
python scripts/analysis/generate_final_report.py
```

**Outputs**:

- `submit_codebase/noise_robustness_runner/results/`

#### 2.6 classification_runner

**Goal**: accuracy across six real-world classification datasets.

**Run**:

```bash
cd submit_codebase/classification_runner/code/classify
python run_all_benchmarks.py
```

**Outputs**:

- `submit_codebase/classification_runner/code/classify/benchmark_results/`

#### 2.7 energy_interpretability_runner

**Goal**: Energy Efficiency regression performance and interpretability.

**Run**:

```bash
cd submit_codebase/energy_interpretability_runner/code
python run_exp5_real.py
python plot_from_saved_data.py
```

**Outputs**:

- `submit_codebase/energy_interpretability_runner/results/`

## Directory Layout

- `stability_and_scaling_runner/`
- `synthetic_function_fit_runner/`
- `spectral_bias_runner/`
- `pareto_runner/`
- `noise_robustness_runner/`
- `classification_runner/`
- `energy_interpretability_runner/`

# Experiment To Code Map

This file maps the paper experiments to the runnable code in this repository.

## Core Implementations

- Foundational CFNN / Standard: `*/code/cfnet.py` -> `CFNet_Standard`
- CFNN-Hybrid: `*/code/cfnet.py` -> `HybridRationalNet`
- CFNN-Boost: `*/code/cfnet.py` -> `EnsembleResCoFrNet`
- CFNN-MoE: `*/code/cfnet.py` -> `MoE_Ensemble`

## Paper Experiment Mapping

### 1. Optimization Stability And Structural Scalability

- Runner: `stability_and_scaling_runner/`
- Main scripts:
  - `stability_and_scaling_runner/code/run_gradient_stability.py`
  - `stability_and_scaling_runner/code/run_scalability.py`
- Typical outputs:
  - `stability_and_scaling_runner/results/gradient_stability/`
  - `stability_and_scaling_runner/results/scalability/`

### 2. Synthetic Function Fitting And Lead Metric

- Runner: `synthetic_function_fit_runner/`
- Main scripts:
  - `synthetic_function_fit_runner/code/nn_hard_1/main.py`
  - `synthetic_function_fit_runner/code/nn_hard_2/main.py`
  - `synthetic_function_fit_runner/code/cfnn_prefer_1/main.py`
  - `synthetic_function_fit_runner/code/cfnn_prefer_2/main.py`
  - `synthetic_function_fit_runner/code/plot_lead.py`
- KAN baselines:
  - `synthetic_function_fit_runner/code/*/main_kan.py`
  - `synthetic_function_fit_runner/code/kan_model.py`

### 3. Spectral Bias Mitigation

- Runner: `spectral_bias_runner/`
- Main scripts:
  - `spectral_bias_runner/code/spectral_bias_experiment.py`
  - `spectral_bias_runner/code/generate_spectral_visualization.py`
  - `spectral_bias_runner/code/plot_paper_figure.py`
- External baselines:
  - `spectral_bias_runner/code/baselines.py`

### 4. Parameter Efficiency And Training Dynamics

- Runner: `pareto_runner/`
- Main scripts:
  - `pareto_runner/code/run_comparison_func_fit.py`
  - `pareto_runner/code/plot_pareto_comparison.py`
  - `pareto_runner/code/plot_hybrid_mlp_comparison.py`

### 5. Noisy-Feature Robustness, Attribution, And Threshold Analysis

- Runner: `noise_robustness_runner/`
- Main scripts:
  - `noise_robustness_runner/code/interpretable_experiment/scripts/run/run_exp1.py`
  - `noise_robustness_runner/code/interpretable_experiment/scripts/run/run_exp1_pareto.py`
  - `noise_robustness_runner/code/interpretable_experiment/scripts/analysis/generate_final_report.py`
- Core experiment files:
  - `noise_robustness_runner/code/interpretable_experiment/experiments/exp1_robustness.py`
  - `noise_robustness_runner/code/interpretable_experiment/experiments/exp1_pareto.py`
  - `noise_robustness_runner/code/interpretable_experiment/config/exp1_config.py`
  - `noise_robustness_runner/code/interpretable_experiment/config/exp1_pareto_config.py`

### 6. Cross-Domain Classification

- Runner: `classification_runner/`
- Main script:
  - `classification_runner/code/classify/run_all_benchmarks.py`
- Dataset-specific scripts:
  - `classification_runner/code/classify/waveform/`
  - `classification_runner/code/classify/magic/`
  - `classification_runner/code/classify/credit_card/`
  - `classification_runner/code/classify/cifar-10/`
  - `classification_runner/code/classify/sentiment/`
  - `classification_runner/code/classify/quora/`
- CoFrNet baselines:
  - `classification_runner/code/CoFrNet_D.py`
  - `classification_runner/code/CoFrNet_DL.py`

### 7. Energy Efficiency Regression And Interpretability

- Runner: `energy_interpretability_runner/`
- Main scripts:
  - `energy_interpretability_runner/code/run_exp5_real.py`
  - `energy_interpretability_runner/code/plot_from_saved_data.py`
- Core experiment files:
  - `energy_interpretability_runner/code/interpretable_experiment/experiments/exp5_real_data.py`
  - `energy_interpretability_runner/code/interpretable_experiment/config/exp5_config.py`
  - `energy_interpretability_runner/code/interpretable_experiment/data/load_real_datasets.py`

## Practical Guidance

- If you want the exact paper-style experiment grouping, start from the runner directories, not from the original workspace history.
- If you only need one paper section, do not install the full workspace. Use the corresponding runner only.
- If a runner contains cached results, treat them as reference artifacts, not as the only source of truth. The executable scripts above are the authoritative entry points.

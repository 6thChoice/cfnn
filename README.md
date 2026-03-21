# CFNN Paper Reproduction Codebase

This directory is a GitHub-ready reproduction package for the experiments in `sn-article.pdf` ("CFNN: Continued Fraction Neural Network").

The repository is organized by experiment runner rather than by model file. This keeps each paper experiment self-contained and makes it easier for other teams to reproduce a specific table or figure without first understanding the full research workspace.

## Included Experiment Groups

1. `stability_and_scaling_runner/`
   Training stability, gradient variance, and depth scalability of the CFNN family.
2. `synthetic_function_fit_runner/`
   Synthetic function fitting, RMSE comparison, and lead-metric plots.
3. `spectral_bias_runner/`
   Spectral bias evaluation against MLP, SIREN, RFF-MLP, and Chebyshev-KAN.
4. `pareto_runner/`
   Parameter efficiency, Pareto frontier analysis, and Hybrid-vs-MLP training dynamics.
5. `noise_robustness_runner/`
   High-dimensional noisy-feature robustness, SHAP attribution, and parameter-threshold analysis.
6. `classification_runner/`
   Cross-domain classification benchmarks on six datasets.
7. `energy_interpretability_runner/`
   UCI Energy Efficiency regression and domain-aligned interpretability analysis.

## Repository Layout

```text
git_codebase/
├── data/                          Dataset download helpers and cache locations
├── docs/                          Experiment mapping and reproduction notes
├── requirements.txt               Python dependencies
├── stability_and_scaling_runner/  Fig. 1-3 style experiments
├── synthetic_function_fit_runner/ Synthetic fitting and lead metric
├── spectral_bias_runner/          Fig. 4-5 style experiments
├── pareto_runner/                 Parameter-efficiency experiments
├── noise_robustness_runner/       Table 5-7 style experiments
├── classification_runner/         Table 3 style experiments
└── energy_interpretability_runner/ Table 8-9 style experiments
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download datasets

```bash
cd data
python download_datasets.py
```

Downloaded caches are expected under:

- `data/openml/`
- `data/hf/`
- `data/torchvision/`

Repository note:

- CIFAR-10 is already bundled under `data/torchvision/cifar10/cifar-10-batches-py/`
- OpenML and HuggingFace caches are not bundled, so those datasets may download on first use

### 3. Run a target experiment

Training stability and scalability:

```bash
cd stability_and_scaling_runner/code
python run_gradient_stability.py
python run_scalability.py
```

Synthetic function fitting:

```bash
cd synthetic_function_fit_runner/code
python nn_hard_1/main.py
python nn_hard_2/main.py
python cfnn_prefer_1/main.py
python cfnn_prefer_2/main.py
python plot_lead.py
```

Spectral bias:

```bash
cd spectral_bias_runner/code
python spectral_bias_experiment.py
python generate_spectral_visualization.py
python plot_paper_figure.py
```

Pareto and training dynamics:

```bash
cd pareto_runner/code
python run_comparison_func_fit.py
python plot_pareto_comparison.py
python plot_hybrid_mlp_comparison.py
```

Noise robustness and attribution:

```bash
cd noise_robustness_runner/code/interpretable_experiment
python scripts/run/run_exp1.py
python scripts/run/run_exp1_pareto.py
python scripts/analysis/generate_final_report.py
```

Classification:

```bash
cd classification_runner/code/classify
python run_all_benchmarks.py
```

Energy Efficiency interpretability:

```bash
cd energy_interpretability_runner/code
python run_exp5_real.py
python plot_from_saved_data.py
```

## Core Model Implementations

The most important shared implementation files are:

- `stability_and_scaling_runner/code/cfnet.py`
- `synthetic_function_fit_runner/code/cfnet.py`
- `spectral_bias_runner/code/cfnet.py`
- `pareto_runner/code/cfnet.py`
- `noise_robustness_runner/code/cfnet.py`
- `energy_interpretability_runner/code/cfnet.py`

These define the paper variants:

- `CFNet_Standard`: foundational CFNN
- `HybridRationalNet`: CFNN-Hybrid
- `EnsembleResCoFrNet`: CFNN-Boost
- `MoE_Ensemble`: CFNN-MoE

## Recommended Reading Order

If you are reproducing the paper from scratch, read these files first:

1. `docs/EXPERIMENT_TO_CODE_MAP.md`
2. `docs/REPRODUCTION_NOTES.md`
3. The runner README-equivalent scripts in the experiment you want to execute

## Notes

- This package is intentionally separated from the original research workspace. Early exploratory scripts from the larger workspace were not used as the primary public entry points here.
- Some runners include selected result artifacts to document expected outputs and make sanity-checking easier.
- Large caches and regenerated outputs are ignored by `.gitignore`.

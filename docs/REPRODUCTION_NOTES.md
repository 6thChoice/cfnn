# Reproduction Notes

## What This Package Is

This repository is a curated reproduction package extracted from a larger research workspace. It is organized to reduce ambiguity for external users.

The public entry points are the runner directories at the repository root. Those are the directories other teams should execute.

## What Was Intentionally Not Used As The Main Entry

The original workspace contained many exploratory scripts and intermediate result directories. Those were useful during research, but they are not the preferred public interface for reproduction.

This package keeps the runner-based structure because:

- import paths are already consistent inside each runner
- experiments are grouped by paper section
- selected artifacts are already colocated with the code that generated them

## Data And Caches

Run:

```bash
cd data
python download_datasets.py
```

Expected cache locations:

- `data/openml/`
- `data/hf/`
- `data/torchvision/`

This repository already includes:

- CIFAR-10 extracted batches under `data/torchvision/cifar10/cifar-10-batches-py/`
- synthetic lead-metric JSON files required by `synthetic_function_fit_runner/code/plot_lead.py`
- saved Energy interpretability artifacts for offline re-plotting

The repository does not currently include project-local OpenML or HuggingFace caches, because
those caches were not present inside the source project tree. Those datasets may still need a
first-time download.

## Environment

- Python version should match the environment used to export `requirements.txt`
- GPU is recommended for the full experiments
- Some datasets require network access during first download

## Results Included In The Repository

Selected result artifacts are retained because they help users verify that a runner executed correctly. They should be treated as reference outputs, not as a replacement for running the code.

Included convenience artifacts:

- `synthetic_function_fit_runner/code/*/plot/model_lead_metrics_*.json`
- `energy_interpretability_runner/results/exp5_saved_results/exp5_analysis_data.pkl`
- `energy_interpretability_runner/results/exp5_real_data/20260216_115826/*.pkl`

Known omission for GitHub compatibility:

- `pareto_runner` plotting scripts expect `pareto_runner/results/comparison_results.json`
- the source file in the research workspace is about 220MB, so it is not bundled here
- regenerate it with `pareto_runner/code/run_comparison_func_fit.py` before running the Pareto plotting scripts

## Known Caveat About Historical Result Versions

The original workspace contained multiple historical runs for some experiments, especially robustness and interpretability analyses. This package keeps the runner code as the primary source of truth. If you regenerate results, prefer the outputs produced by the runner scripts in this repository rather than older logs from the original workspace.

## Recommended Reproduction Order

1. Install dependencies
2. Download datasets
3. Run `stability_and_scaling_runner`
4. Run `synthetic_function_fit_runner`
5. Run `spectral_bias_runner`
6. Run `pareto_runner`
7. Run `noise_robustness_runner`
8. Run `classification_runner`
9. Run `energy_interpretability_runner`

## Minimal Sanity Checks

- Every runner should create outputs under its own `results/` directory
- Classification benchmarks should write into `classification_runner/code/classify/benchmark_results/`
- Energy interpretability should generate a domain report under `energy_interpretability_runner/results/`

## Suggested GitHub Presentation

- Keep the root `README.md` short and task-oriented
- Put paper-to-code mapping in `docs/EXPERIMENT_TO_CODE_MAP.md`
- Do not commit newly downloaded dataset caches
- Do not overwrite included reference results unless you intentionally want to refresh them

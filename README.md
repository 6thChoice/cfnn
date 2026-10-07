# CFNN: Continued Fraction Neural Networks

This repository contains the code and reproducibility artifacts for the manuscript **“Continued fraction neural networks for sharp scientific response modelling”**, submitted to *Communications AI & Computing*. The frozen submission release is `v1.0.0-caic`.

The repository includes CFNN and CFNN-Hybrid implementations, experiment runners, frozen configurations, public-data metadata, result manifests, minimal verification data and figure-generation scripts.

## Layout

- `src/cfnn/`: CFNN, CFNN-Hybrid, CoFrNet comparison implementations and the lightweight legacy baseline code used by the spectral figure.
- `experiments/`: training, benchmark, downstream microwave and analysis scripts.
- `figures/`: manuscript figure-generation scripts.
- `configs/`: frozen experiment configurations and result schemas.
- `data/`: public-data source manifests and download tooling.
- `results/`: selected manifests, summaries, model selections and figure-source arrays.
- `docs/`: experiment-to-code mapping and minimal verification data documentation.
- `scripts/`: environment setup, package checks, figure reproduction and verification utilities.

## Environments

The repository records two environments because the legacy benchmark/figure workflow and the downstream confirmatory runs were executed separately.

### Core and legacy figure environment

- Python 3.12.3
- CUDA 12.4
- PyTorch 2.5.1+cu124

Create this environment with:

```bash
bash scripts/install_env.sh
source .venv/bin/activate
```

Direct dependencies are pinned in `requirements-lock.txt`; the transitive snapshot is `requirements-server-freeze.txt`.

### Downstream confirmatory environment

- NVIDIA GeForce RTX 3080
- Python 3.13.5
- CUDA 12.8
- PyTorch 2.10.0+cu128

Direct dependencies are recorded in `requirements-downstream-confirmatory.txt`. The corresponding hardware and runtime versions are also stored in the downstream run manifests.

For CPU-only source inspection, install an appropriate CPU PyTorch wheel and then install `requirements.txt`.

## Data

The measured datasets are publicly available from their original repositories:

- Microwave Fano response: Zenodo record 7767046.
- Microstrip resonator response: Zenodo record 14175959.
- Additional source metadata: `data/downstream_sources.json`.

Download public raw data with:

```bash
python data/download_downstream_datasets.py   --config data/downstream_sources.json   --raw-root data/raw
```

Audit source metadata without downloading raw files with:

```bash
python data/download_downstream_datasets.py --metadata-only
```

## Minimal verification data

The archived JSON and NPZ files listed in `results/figures/minimal_verification_manifest.json` are the minimal verification data for the submission package. Validate their checksums and basic schemas with:

```bash
python scripts/verify_minimal_reproduction.py
```

Field descriptions and provenance are documented in `docs/MINIMAL_VERIFICATION_DATA.md`. The upstream datasets retain their original licences; the Apache-2.0 software licence does not relicense third-party data.

## Figure reproduction

The main package-relative figure scripts write to `results/figures/`:

```bash
export PYTHONPATH="$PWD/experiments:$PWD/src:${PYTHONPATH:-}"
bash scripts/reproduce_main_figures.sh
```

## Experiment entry points

Representative commands include:

```bash
python experiments/run_stability_fair_check.py
python experiments/peak_sensitive_benchmark.py --config configs/peak_sensitive_config.json
python experiments/downstream_benchmark.py --config configs/downstream_config.json
python experiments/run_ai4science_fano_a2_pilot.py --help
python experiments/run_ai4science_microstrip_low_calibration_event_transfer.py --help
python experiments/run_independent_benchmark.py --smoke --out results/generated/independent_smoke.json
```

## Package check

Run the existing lightweight package check with:

```bash
bash scripts/check_package.sh
```

It compiles the Python sources and redraws the four bundled main-figure workflows.

## Licence

The software is released under the Apache License 2.0. See `LICENSE` and `NOTICE`. Dataset licences remain with their original providers.

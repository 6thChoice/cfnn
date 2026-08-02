# CFNN NMI Reproduction Package

This repository is the curated code package for the manuscript "Continued fraction neural networks for sharp scientific response modelling".

It contains the CFNN and CFNN-Hybrid implementations, experiment runners, public-data download metadata, selected result manifests, and cached figure-source arrays needed to reproduce the main NMI figures without committing raw datasets or large training logs.

## Layout

- `src/cfnn_nmi/`: CFNN, CFNN-Hybrid, CoFrNet comparison implementations, and lightweight legacy baseline code needed by the spectral figure.
- `experiments/`: training, benchmark, downstream microwave, and analysis scripts extracted from the formal experiment workspace.
- `figures/`: manuscript figure-generation scripts.
- `configs/`: frozen experiment configurations and result schemas.
- `data/`: source manifests and the downloader for public datasets.
- `results/`: selected manifests, summaries, selections, figure caches, and final manuscript figure outputs.
- `server_provenance/`: audit notes for remote execution artifacts that were inspected but not bundled.

## Environment

The formal training environment used Python 3.12.3, CUDA 12.4, and PyTorch 2.5.1+cu124. To create a matching local environment:

```bash
cd submit_codebase_nmi
bash scripts/install_env.sh
source .venv/bin/activate
```

For CPU-only inspection, install a CPU PyTorch wheel manually, then run:

```bash
python -m pip install -r requirements.txt
```

The full transitive snapshots are retained as `requirements-lock.txt` and `requirements-server-freeze.txt`.

## Data

Raw public datasets are not committed. Download metadata is included for:

- Microwave Fano response: Zenodo record 7767046.
- Microstrip resonator response: Zenodo record 14175959.
- Additional audit/extension sources listed in `data/downstream_sources.json`.

To download or refresh public raw data:

```bash
python data/download_downstream_datasets.py --config data/downstream_sources.json --raw-root data/raw
```

To audit metadata without downloading raw files:

```bash
python data/download_downstream_datasets.py --metadata-only
```

## Main Figure Reproduction

The four manuscript figure scripts are package-relative and write to `results/figures/`:

```bash
export PYTHONPATH="$PWD/experiments:$PWD/src:${PYTHONPATH:-}"
bash scripts/reproduce_main_figures.sh
```

These scripts use bundled caches where available. Rebuilding caches for the microwave figures requires the public raw data under `data/raw/`.

## Experiment Entry Points

Representative runners:

```bash
python experiments/run_stability_fair_check.py
python experiments/peak_sensitive_benchmark.py --config configs/peak_sensitive_config.json
python experiments/downstream_benchmark.py --config configs/downstream_config.json
python experiments/run_ai4science_fano_a2_pilot.py --help
python experiments/run_ai4science_microstrip_low_calibration_event_transfer.py --help
python experiments/run_independent_benchmark.py --smoke --out results/generated/independent_smoke.json
```

Full experiments are GPU-oriented and may take many hours. The included `results/` files are reference artifacts and provenance records, not a substitute for rerunning when exact hardware replication is required.

## Verification

Run a lightweight package check:

```bash
bash scripts/check_package.sh
```

This compiles the Python files and redraws the four main manuscript figures from bundled caches.

## Release Notes

Before public submission, replace manuscript placeholders with the GitHub URL and archive DOI. Do not commit downloaded raw data, remote logs, tarballs, credentials, or local virtual environments.

# Reproduction Notes

## Scope

This is a curated release package extracted from a larger research workspace. It is designed for public review and GitHub publication, not for preserving every exploratory run.

Bundled:

- Model implementations and experiment scripts.
- Frozen configs, schemas, candidate selections, and run manifests.
- Lightweight result summaries.
- Main manuscript figure caches and rendered outputs.

Not bundled:

- Raw public datasets.
- Full remote execution logs and tarballs.
- The 428 MB full gradient-stability JSON; a reduced figure cache is included instead.
- Local virtual environments, compiled bytecode, and credentials.

## Recommended Order

1. Create the environment with `bash scripts/install_env.sh`.
2. Run `bash scripts/check_package.sh`.
3. Download data with `python data/download_downstream_datasets.py --config data/downstream_sources.json --raw-root data/raw`.
4. Redraw the main figures with `bash scripts/reproduce_main_figures.sh`.
5. Run full GPU experiments only after confirming the target configs and output paths.

## Output Paths

For new runs, prefer writing under `results/generated/` or a separate scratch directory. Do not overwrite bundled reference artifacts unless you are intentionally refreshing the public release.

## Hardware

The formal environment used an RTX 2080 Ti class CUDA setup. CPU execution is suitable for import checks and cached figure redraws, but full training runs are expected to be slow.

## Optional Text Benchmarks

Historical fixed-vector text benchmarks may require `tensorflow` and `nltk`; install `requirements-optional.txt` only if those scripts are needed.

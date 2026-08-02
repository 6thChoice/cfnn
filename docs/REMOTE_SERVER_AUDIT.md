# Remote Server Audit

Remote execution artifacts were inspected on 2026-08-01 through the user-provided SSH endpoint. Credentials are intentionally not recorded in this repository.

## Inspected Paths

- `/root/cofrnet/experiment_refine`
- `/root/autodl-tmp/cfnn_confirmatory`
- `/root/autodl-tmp/cfnn_explore`
- `/root/autodl-tmp/cfnn_explore_v2`

## Relevant Remote Content

The remote workspace contained downstream and confirmatory execution files, including:

- `downstream_benchmark.py`
- `downstream_training.py`
- `downstream_protocol.py`
- `downstream_metrics.py`
- `downstream_domain_baselines.py`
- `downstream_config.json`
- `downstream_extension_config.json`
- `downstream_group_transfer_config.json`
- execution manifests under downstream confirmatory, group-transfer, and seed-extension result folders.

## Inclusion Decision

Large remote logs, tarballs, and full execution archives are not bundled in this GitHub package. The public package includes local equivalents, frozen configs, selected summaries, figure caches, and provenance manifests that are directly tied to the manuscript.

If a reviewer requests full remote logs, export them as a separate archival supplement instead of committing them to Git.

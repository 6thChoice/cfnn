# Experiment To Code Map

This file maps manuscript evidence to the public reproduction package.

## Core Implementations

- Foundational projected CFNN: `src/cfnn/cfnn/cfnn_poly.py`, `src/cfnn/cfnet_complicate.py`.
- Stabilized CFNN-Hybrid / projected additive rational network: `src/cfnn/hybrid.py`, `experiments/confirmatory_models.py`, `experiments/eis_models.py`.
- Published CoFrNet comparison implementations: `src/cfnn/CoFrNet_D.py`, `src/cfnn/CoFrNet_DL.py`.

## Manuscript Figures

- Figure 1, training stability: `figures/create_cfnn_hybrid_training_stability.py`.
  - Cache: `results/figures/cfnn_hybrid_training_stability_cache.json`.
  - Reference outputs: `results/figures/cfnn_hybrid_training_stability.{pdf,png}`.
- Figure 2, sharp high-curvature response: `figures/create_sharp_response_mechanism.py`.
  - Cache: `results/figures/sharp_response_mechanism_cache.npz`.
  - Reference outputs: `results/figures/sharp_response_mechanism.{pdf,png}`.
- Figure 3, Microwave Fano recovery: `figures/create_ai4science_fano_3d_response_recovery.py`.
  - Source cache: `results/figures/fano_response_recovery_cache.npz`.
  - Reference outputs: `results/figures/fano_3d_response_recovery.{pdf,png}`.
- Figure 4, Microstrip low-calibration recovery: `figures/create_ai4science_microstrip_recovery.py`.
  - Cache: `results/figures/microstrip_low_calibration_recovery_cache.npz`.
  - Metrics: `results/figures/microstrip_low_calibration_recovery_metrics.json`.
  - Reference outputs: `results/figures/microstrip_low_calibration_recovery.{pdf,png}`.

## Manuscript Tables

- Clean synthetic function fitting table:
  - Runner family: `experiments/run_independent_benchmark.py`, `experiments/run_parameter_matched_synthetic.py`.
  - Summaries: `results/summaries/independent_benchmark/`, `results/summaries/core_common_budget/`.
- Common ML comparability table:
  - Implementations: `src/cfnn/CoFrNet_D.py`, `src/cfnn/CoFrNet_DL.py`.
  - Downstream scripts: `experiments/downstream_benchmark.py`, `experiments/downstream_domain_baselines.py`.
  - Configs/manifests: `configs/downstream_config.json`, `results/manifests/downstream_confirmatory_run_manifest.json`.
- Microwave Fano and Microstrip numerical claims:
  - Fano runners: `experiments/ai4science_fano.py`, `experiments/ai4science_fano_a2.py`, `experiments/run_ai4science_fano_a2_pilot.py`.
  - Microstrip runners: `experiments/ai4science_microstrip_b.py`, `experiments/run_ai4science_microstrip_low_calibration_event_transfer.py`.
  - Selection manifests: `results/selections/`.
  - Summaries: `results/ai4science/microwave_fano/`, `results/ai4science/microstrip/`.

## Protocol And Provenance

- Frozen peak-sensitive protocol: `configs/peak_sensitive_config.json`, `results/manifests/protocol_manifest.json`.
- Downstream protocol: `experiments/downstream_protocol.py`, `experiments/downstream_training.py`, `experiments/downstream_metrics.py`.
- Public data sources: `data/downstream_sources.json`.
- Remote execution audit: `docs/REMOTE_SERVER_AUDIT.md`.

## Notes

Several exploratory scripts are preserved in `experiments/` and `figures/` for traceability. The package-relative scripts listed under "Manuscript Figures" are the public figure entry points.

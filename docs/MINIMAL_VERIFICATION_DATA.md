# Minimal verification data

The files listed in `results/figures/minimal_verification_manifest.json` form the minimal verification data bundled with the CFNN submission release. They contain archived arrays and structured summaries consumed by the package-relative figure scripts.

This is not a retraining dataset. The complete measured datasets remain available from their original Zenodo records and can be downloaded with `data/download_downstream_datasets.py`.

## Files

- `cfnn_hybrid_training_stability_cache.json`: reduced training-stability trajectories and summaries.
- `sharp_response_mechanism_cache.npz`: target, prediction and residual-spectrum arrays for the sharp-response figure.
- `fano_response_recovery_cache.npz`: measured Fano target and reconstruction arrays used by the response-recovery figure.
- `microstrip_low_calibration_recovery_cache.npz`: microstrip target, calibration and reconstruction arrays used by the low-calibration figure.
- `microstrip_low_calibration_recovery_metrics.json`: event-level metrics accompanying the microstrip cache.

## Integrity verification

Run:

```bash
python scripts/verify_minimal_reproduction.py
```

The verifier checks each file against its recorded size and SHA-256 digest, parses JSON files and opens NPZ archives without enabling pickle deserialization.

## Licence and provenance

The Apache-2.0 licence applies to the software in this repository. Source measurements retain the terms of their original providers. Dataset source records are listed in `data/downstream_sources.json` and the manuscript Data Availability statement.

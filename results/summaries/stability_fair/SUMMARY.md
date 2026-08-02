# Stability fair-comparison result

Task: 1D Runge peak + high-frequency component; depth/units=6; 5 seeds.

## FIXED
| Model | fail rate | test loss | grad median | grad IQR | grad p95 | grad max |
|---|---:|---:|---:|---:|---:|---:|
| CFNN | 0% | 0.7996 ± 0.0858 | 0.517 | 2.047 | 6.497 | 8.715 |
| CFNN-Hybrid | 0% | 0.6412 ± 0.0680 | 0.490 | 1.030 | 3.428 | 4.098 |

## TUNED
Best configs: `{'CFNN': {'lr': 0.01, 'weight_decay': 0.0, 'grad_clip': 1.0}, 'CFNN-Hybrid': {'lr': 0.01, 'weight_decay': 0.0, 'grad_clip': 1.0}}`

| Model | fail rate | test loss | grad median | grad IQR | grad p95 | grad max |
|---|---:|---:|---:|---:|---:|---:|
| CFNN | 0% | 0.1392 ± 0.0184 | 0.237 | 0.429 | 1.504 | 13.412 |
| CFNN-Hybrid | 0% | 0.1231 ± 0.0183 | 0.070 | 0.177 | 0.890 | 4.098 |

## Interpretation

This is a compact reviewer-facing check, not a replacement for the full stability runner. It gives both models the same fixed recipe and then the same small tuning budget. If CFNN remains higher in gradient dispersion after tuning, the instability claim is less likely to be just an under-regularized recipe artifact.
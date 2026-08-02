# Production-model stability comparison

Task: 1D Runge peak + high-frequency component; degree=3; 5 seeds.

## Parameter Matched

### Fixed
| Model | params | fail rate | test loss | grad std | grad p95 |
|---|---:|---:|---:|---:|---:|
| CFNN | 41 | 0% | 0.7996 ± 0.0858 | 2.063 ± 0.459 | 6.497 ± 1.346 |
| CFNN-Hybrid | 38 | 0% | 0.2982 ± 0.0406 | 4.003 ± 2.159 | 7.320 ± 5.469 |

### Tuned
Best configs: `{'CFNN': {'lr': 0.01, 'weight_decay': 0.0, 'grad_clip': 1.0}, 'CFNN-Hybrid': {'lr': 0.01, 'weight_decay': 0.0001, 'grad_clip': 1.0}}`
| Model | params | fail rate | test loss | grad std | grad p95 |
|---|---:|---:|---:|---:|---:|
| CFNN | 41 | 0% | 0.1392 ± 0.0184 | 1.299 ± 0.773 | 1.504 ± 1.218 |
| CFNN-Hybrid | 38 | 0% | 0.1312 ± 0.0131 | 2.599 ± 0.857 | 3.763 ± 1.136 |


## Component Matched

### Fixed
| Model | params | fail rate | test loss | grad std | grad p95 |
|---|---:|---:|---:|---:|---:|
| CFNN | 41 | 0% | 0.7996 ± 0.0858 | 2.063 ± 0.459 | 6.497 ± 1.346 |
| CFNN-Hybrid | 74 | 0% | 0.2601 ± 0.0502 | 4.040 ± 3.108 | 7.053 ± 4.239 |

### Tuned
Best configs: `{'CFNN': {'lr': 0.01, 'weight_decay': 0.0, 'grad_clip': 1.0}, 'CFNN-Hybrid': {'lr': 0.01, 'weight_decay': 0.0, 'grad_clip': 1.0}}`
| Model | params | fail rate | test loss | grad std | grad p95 |
|---|---:|---:|---:|---:|---:|
| CFNN | 41 | 0% | 0.1392 ± 0.0184 | 1.299 ± 0.773 | 1.504 ± 1.218 |
| CFNN-Hybrid | 74 | 0% | 0.1366 ± 0.0251 | 5.973 ± 1.529 | 12.435 ± 5.322 |

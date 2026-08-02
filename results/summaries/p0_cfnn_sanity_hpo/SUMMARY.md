# P0 CFNN-side small-budget sanity HPO

Device: `cuda`; seeds: `[42, 123, 456]`; n_samples=3000; max_epochs=500.

This is a limited reviewer-facing sanity check, not a full CFNN-family HPO sweep.

## Rational_Interaction

| Model | params | RMSE mean±std | R² mean±std | best cfg | tuned MLP RMSE | old best CFNN RMSE |
|---|---:|---:|---:|---|---:|---:|
| Hybrid | 244 | 0.034579 ± 0.001020 | 0.998769 ± 0.000091 | `{'degree': 5, 'size': 12, 'lr': 0.01, 'weight_decay': 0.0, 'grad_clip': 5.0}` | 0.021159 | 0.030800 |
| MoE | 200 | 0.043775 ± 0.001393 | 0.998025 ± 0.000185 | `{'degree': 5, 'size': 8, 'lr': 0.01, 'weight_decay': 0.0, 'grad_clip': 5.0}` | 0.021159 | 0.030800 |

## Nested_Rational

| Model | params | RMSE mean±std | R² mean±std | best cfg | tuned MLP RMSE | old best CFNN RMSE |
|---|---:|---:|---:|---|---:|---:|
| Hybrid | 196 | 0.021656 ± 0.000907 | 0.998685 ± 0.000112 | `{'degree': 3, 'size': 12, 'lr': 0.01, 'weight_decay': 0.0, 'grad_clip': 5.0}` | 0.020980 | 0.019800 |
| MoE | 200 | 0.024791 ± 0.000930 | 0.998279 ± 0.000106 | `{'degree': 5, 'size': 8, 'lr': 0.01, 'weight_decay': 0.0, 'grad_clip': 5.0}` | 0.020980 | 0.019800 |

## Conservative use in paper

Use this table to state that a small CFNN-side tuning check was run for the sensitive clean-fitting functions. If the tuned CFNN variants remain close to tuned MLP rather than decisively ahead, Table 1 should be framed as a fairness calibration, not as the main evidence for CFNN superiority.
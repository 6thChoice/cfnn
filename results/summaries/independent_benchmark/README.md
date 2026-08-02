# Independent common-budget benchmark

`run_independent_benchmark.py` evaluates CFNN-Hybrid against MLP, the fixed-centre
Gaussian RBF baseline, official `pykan`, CoFrNet-Standard, a trainable rational
activation network, and SIREN. The targets are deliberately independent of the
Lorentzian/pole construction used by the legacy sharpness experiments:

- `gaussian_peaks`: mixtures of Gaussian peaks with heterogeneous widths;
- `fourier`: a high-frequency periodic sum;
- `nonseparable`: a three-dimensional, two-output target containing a coupled
  Gaussian/cosine component and a multiplicative/periodic component.

The split is 60% train, 20% validation, and 20% test. Targets are standardized
using training-set mean and standard deviation for optimization; reported
$R^2$ is invariant to this affine scaling. Configuration selection
uses validation loss only. The raw record stores the realized trainable
parameter count, validation loss, selected epoch count, wall time, seed, and
model metadata. The final benchmark command is:

```bash
python experiment_refine/run_independent_benchmark.py \
  --out experiment_refine/independent_benchmark_results/raw.json \
  --epochs 250
```

The official KAN implementation is `pykan==0.2.8`; its spline order, grid, and
grid-adaptation metadata are recorded in each raw record. The current runner is
an experiment implementation and should be run to completion before its values
are cited in the manuscript.

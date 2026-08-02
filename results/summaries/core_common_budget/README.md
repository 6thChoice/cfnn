# Core common-budget benchmark

`raw_normalized.json` contains the revised five-seed comparison on a broad
regularized pole, a sharp regularized pole, and a Lorentzian resonance. The
same runner and protocol are used for the independent benchmark. Nominal
budget anchors are 32, 64, 128, 256, and 512; actual trainable parameter counts
are recorded per run. Target normalization is fitted on training observations
only. Validation selects the configuration and epoch count, after which a fresh
model is refit on training plus validation data for the fixed epoch count.

The baseline matrix is CFNN-Hybrid, MLP, fixed-centre Gaussian RBF, official
PyKAN 0.2.8, CoFrNet-Standard, rational activation NN, and SIREN. The raw
records are summarized by `summarize_independent_benchmark.py`.

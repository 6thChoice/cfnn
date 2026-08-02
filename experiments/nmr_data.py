"""Idealized semi-synthetic NMR line-shape stress tests.

The environment tables are manually specified approximate values informed by
common assignments. They are not digitizations of named SDBS records or raw
instrument spectra. The strychnine centers are approximate values informed by
Carter--Luther--Long (1974). Every environment is expanded with idealized
first-order splitting and a pure Lorentzian forward model.
"""
from __future__ import annotations

import numpy as np

SPECTROMETER_MHZ = 400.0   # only for J[Hz] -> ppm conversion
FWHM_PPM = 0.02            # ~8 Hz @ 400 MHz; real & sampling-tractable
GAMMA_PPM = FWHM_PPM / 2.0

# env = {"delta": ppm, "n_h": integration, "J": [Hz, ...]}  ([]=singlet)
ETHANOL = {
    "name": "ethanol", "ppm_lo": 0.5, "ppm_hi": 4.2,
    "source": "manually specified approximate common assignments; no SDBS accession",
    "envs": [
        {"delta": 1.22, "n_h": 3.0, "J": [7.0, 7.0]},        # CH3 triplet
        {"delta": 3.69, "n_h": 2.0, "J": [7.0, 7.0, 7.0]},   # CH2 quartet
        {"delta": 2.60, "n_h": 1.0, "J": []},                # OH singlet
    ],
}
CAFFEINE = {
    "name": "caffeine", "ppm_lo": 3.0, "ppm_hi": 8.0,
    "source": "manually specified approximate common assignments; no SDBS accession",
    "envs": [
        {"delta": 7.51, "n_h": 1.0, "J": []},   # H-8 aromatic
        {"delta": 3.99, "n_h": 3.0, "J": []},   # N-CH3
        {"delta": 3.58, "n_h": 3.0, "J": []},   # N-CH3
        {"delta": 3.40, "n_h": 3.0, "J": []},   # N-CH3
    ],
}
STRYCHNINE = {
    "name": "strychnine", "ppm_lo": 1.0, "ppm_hi": 8.5,
    "source": "approximate table informed by Carter--Luther--Long (1974)",
    "envs": [
        {"delta": 8.10, "n_h": 1.0, "J": [8.0]},        # H-1
        {"delta": 7.24, "n_h": 1.0, "J": [7.5, 7.5]},   # H-3
        {"delta": 7.14, "n_h": 1.0, "J": [7.5, 7.5]},   # H-2
        {"delta": 7.08, "n_h": 1.0, "J": [8.0]},        # H-4
        {"delta": 5.89, "n_h": 1.0, "J": [6.5]},        # H-22 vinyl
        {"delta": 4.28, "n_h": 1.0, "J": [8.5]},        # H-12
        {"delta": 4.14, "n_h": 1.0, "J": []},           # H-23a
        {"delta": 4.06, "n_h": 1.0, "J": []},           # H-23b
        {"delta": 3.93, "n_h": 1.0, "J": []},           # H-8
        {"delta": 3.84, "n_h": 1.0, "J": []},           # H-16
        {"delta": 3.70, "n_h": 1.0, "J": []},           # H-20a
        {"delta": 3.20, "n_h": 1.0, "J": []},           # H-11a
        {"delta": 3.10, "n_h": 1.0, "J": []},           # H-18a
        {"delta": 2.88, "n_h": 1.0, "J": []},           # H-11b
        {"delta": 2.72, "n_h": 1.0, "J": []},           # H-20b
        {"delta": 2.66, "n_h": 1.0, "J": []},           # H-18b
        {"delta": 2.35, "n_h": 2.0, "J": []},           # H-14 + H-15b
        {"delta": 1.88, "n_h": 2.0, "J": []},           # H-17a/b
        {"delta": 1.45, "n_h": 1.0, "J": []},           # H-15a
        {"delta": 1.27, "n_h": 1.0, "J": []},           # H-13
    ],
}
MOLECULES = {"ethanol": ETHANOL, "caffeine": CAFFEINE, "strychnine": STRYCHNINE}


def expand_multiplet(delta, J, spectrometer_mhz):
    """First-order splitting: each J splits every line into two at +-(J/2)/MHz
    ppm with half weight. Equal J's -> binomial pattern. Returns [(ppm, w)]
    with sum(w)==1. Coincident lines are merged."""
    lines = [(float(delta), 1.0)]
    for j in J:
        dppm = (j / 2.0) / spectrometer_mhz
        nxt = []
        for p, w in lines:
            nxt.append((p - dppm, w / 2.0))
            nxt.append((p + dppm, w / 2.0))
        lines = nxt
    merged = {}
    for p, w in lines:
        key = round(p, 9)
        merged[key] = merged.get(key, 0.0) + w
    return [(k, merged[k]) for k in sorted(merged)]


def nmr_forward(x_ppm, mol, gamma_ppm):
    """Clean spectrum = sum over envs, over split lines, of a Lorentzian with
    height ∝ n_h*line_weight (equal gamma -> area ∝ n_h). Returns float32."""
    x = np.asarray(x_ppm, dtype=np.float64)
    y = np.zeros_like(x)
    g2 = gamma_ppm * gamma_ppm
    for env in mol["envs"]:
        for mu, w in expand_multiplet(env["delta"], env["J"], SPECTROMETER_MHZ):
            A = env["n_h"] * w
            y = y + A * g2 / ((x - mu) ** 2 + g2)
    return y.astype(np.float32)


def make_nmr_dataset(mol_key, n_points, noise, seed, n_dense=4000):
    """Real-molecule NMR spectrum as a 1D regression task. x = ppm normalized
    to [-1,1]; multiplicative noise on training Y; clean dense Yeval for R^2."""
    mol = MOLECULES[mol_key]
    lo, hi = mol["ppm_lo"], mol["ppm_hi"]
    rng = np.random.default_rng(seed)

    xppm = np.linspace(lo, hi, n_points).astype(np.float64)
    y = nmr_forward(xppm, mol, GAMMA_PPM).astype(np.float64)
    yn = (y * (1.0 + noise * rng.standard_normal(y.shape))).astype(np.float32)

    xd = np.linspace(lo, hi, n_dense).astype(np.float64)
    yd = nmr_forward(xd, mol, GAMMA_PPM)

    def norm(a):
        return ((a - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32)

    peaks = np.array(sorted(e["delta"] for e in mol["envs"]), dtype=np.float64)
    return {
        "X": norm(xppm).reshape(-1, 1), "Y": yn.reshape(-1, 1),
        "Xeval": norm(xd).reshape(-1, 1), "Yeval": yd.reshape(-1, 1),
        "out_dim": 1, "kind": "scalar",
        "ppm_lo": lo, "ppm_hi": hi, "true_peaks_ppm": peaks, "mol": mol_key,
        "parameter_source": mol["source"],
    }

"""EIS pilot utilities: equivalent-circuit forward models, noise, representation,
metrics, and classical rational baseline (nonlinear least squares circuit fit)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

TWO_PI = 2.0 * np.pi


def _omega(f):
    return TWO_PI * np.asarray(f, dtype=float)


def randles_impedance(f, Rs, Rct, Cdl):
    w = _omega(f)
    return Rs + Rct / (1.0 + 1j * w * Rct * Cdl)


def two_tc_impedance(f, Rs, R1, C1, R2, C2):
    w = _omega(f)
    return Rs + R1 / (1.0 + 1j * w * R1 * C1) + R2 / (1.0 + 1j * w * R2 * C2)


def randles_warburg_impedance(f, Rs, Rct, Cdl, sigma):
    w = _omega(f)
    Zw = sigma * (1.0 - 1j) / np.sqrt(w)          # semi-infinite Warburg (non-rational)
    return Rs + (Rct + Zw) / (1.0 + 1j * w * Cdl * (Rct + Zw))


def randles_cpe_impedance(f, Rs, Rct, Q, alpha):
    w = _omega(f)
    Zcpe = 1.0 / (Q * (1j * w) ** alpha)          # CPE (non-rational, fractional order)
    return Rs + (Rct * Zcpe) / (Rct + Zcpe)


CIRCUITS = {
    "randles": {
        "func": randles_impedance,
        "param_names": ["Rs", "Rct", "Cdl"],
        "pole_names": ["tau1"],
    },
    "two_tc": {
        "func": two_tc_impedance,
        "param_names": ["Rs", "R1", "C1", "R2", "C2"],
        "pole_names": ["tau1", "tau2"],
    },
    "randles_warburg": {
        "func": randles_warburg_impedance,
        "param_names": ["Rs", "Rct", "Cdl", "sigma"],
        "pole_names": ["tau1"],
    },
    "randles_cpe": {
        "func": randles_cpe_impedance,
        "param_names": ["Rs", "Rct", "Q", "alpha"],
        "pole_names": ["tau1"],
    },
}


def ground_truth_poles(circuit_name, params):
    """Return RC time constants (= pole locations) for a circuit's parameters."""
    p = params
    if circuit_name == "two_tc":
        return {"tau1": p["R1"] * p["C1"], "tau2": p["R2"] * p["C2"]}
    if circuit_name == "randles":
        return {"tau1": p["Rct"] * p["Cdl"]}
    if circuit_name == "randles_warburg":
        return {"tau1": p["Rct"] * p["Cdl"]}
    if circuit_name == "randles_cpe":
        # effective time constant for CPE: tau = (Rct*Q)^(1/alpha)
        return {"tau1": (p["Rct"] * p["Q"]) ** (1.0 / p["alpha"])}
    raise ValueError(f"unknown circuit {circuit_name}")


def freq_grid(f_min, f_max, n):
    """Generate logarithmic frequency grid."""
    return np.logspace(np.log10(f_min), np.log10(f_max), int(n))


def add_proportional_noise(Z, sigma_rel, rng):
    """Add proportional (relative) noise to impedance values.

    Noise magnitude is proportional to impedance magnitude.
    """
    if sigma_rel <= 0:
        return Z.copy()
    mag = np.abs(Z)
    noise = rng.normal(0.0, sigma_rel, size=Z.shape) * mag \
        + 1j * rng.normal(0.0, sigma_rel, size=Z.shape) * mag
    return Z + noise


def omega_features(f, mode, f_min, f_max):
    """Normalize frequency to [0, 1] representation.

    Parameters
    ----------
    f : array-like
        Frequencies (Hz).
    mode : str
        One of 'linear', 'sqrt', 'log'.
    f_min, f_max : float
        Min/max frequency bounds.

    Returns
    -------
    np.ndarray
        Normalized features of shape (n, 1) as float32.
    """
    f = np.asarray(f, dtype=float)
    if mode == "linear":
        w, wmin, wmax = _omega(f), _omega(f_min), _omega(f_max)
        x = (w - wmin) / (wmax - wmin)
    elif mode == "sqrt":
        w, wmin, wmax = _omega(f), _omega(f_min), _omega(f_max)
        x = (np.sqrt(w) - np.sqrt(wmin)) / (np.sqrt(wmax) - np.sqrt(wmin))
    elif mode == "log":
        lo, hi = np.log10(f_min), np.log10(f_max)
        x = (np.log10(f) - lo) / (hi - lo)
    else:
        raise ValueError(f"unknown mode {mode}")
    return x.reshape(-1, 1).astype(np.float32)


def make_dataset(circuit_name, params, f_min, f_max, n_points, sigma_rel, seed, mode):
    """Generate a complete dataset for a circuit with noisy impedance measurements.

    Parameters
    ----------
    circuit_name : str
        Name of circuit (key in CIRCUITS).
    params : dict
        Circuit parameters.
    f_min, f_max : float
        Frequency range.
    n_points : int
        Number of frequency points.
    sigma_rel : float
        Relative noise standard deviation.
    seed : int
        Random seed.
    mode : str
        Feature normalization mode.

    Returns
    -------
    dict
        Contains X (n, 1), Y (n, 2), f, Z_clean, Z_noisy, circuit, params, mode.
    """
    rng = np.random.default_rng(seed)
    f = freq_grid(f_min, f_max, n_points)
    Z_clean = CIRCUITS[circuit_name]["func"](f, **params)
    Z_noisy = add_proportional_noise(Z_clean, sigma_rel, rng)
    X = omega_features(f, mode, f_min, f_max)
    Y = np.stack([Z_noisy.real, Z_noisy.imag], axis=1).astype(np.float32)
    return {"X": X, "Y": Y, "f": f, "Z_clean": Z_clean, "Z_noisy": Z_noisy,
            "circuit": circuit_name, "params": params, "mode": mode}


# reasonable initial guesses per circuit
_INIT = {
    "randles": dict(Rs=10.0, Rct=200.0, Cdl=1e-5),
    "two_tc": dict(Rs=10.0, R1=200.0, C1=1e-4, R2=200.0, C2=1e-6),
    "randles_warburg": dict(Rs=10.0, Rct=200.0, Cdl=1e-5, sigma=50.0),
    "randles_cpe": dict(Rs=10.0, Rct=200.0, Q=1e-5, alpha=0.8),
}
# params fit in log-space except alpha (fit in logit-like linear, clipped to (0,1))
_LOG_PARAMS = {"Rs", "Rct", "Cdl", "R1", "C1", "R2", "C2", "sigma", "Q"}


def _pack(params, names):
    return np.array([np.log(params[n]) if n in _LOG_PARAMS else params[n] for n in names])


def _unpack(theta, names):
    out = {}
    for n, t in zip(names, theta):
        out[n] = float(np.exp(t)) if n in _LOG_PARAMS else float(np.clip(t, 1e-3, 0.999))
    return out


def fit_equivalent_circuit(f, Z, circuit_name, init=None):
    spec = CIRCUITS[circuit_name]
    names = spec["param_names"]
    init = dict(_INIT[circuit_name], **(init or {}))
    scale = np.median(np.abs(Z)) + 1e-9

    def resid(theta):
        p = _unpack(theta, names)
        Zm = spec["func"](f, **p)
        r = (Zm - Z) / scale
        return np.concatenate([r.real, r.imag])

    sol = least_squares(resid, _pack(init, names), method="lm", max_nfev=5000)
    return {"params": _unpack(sol.x, names), "success": bool(sol.success),
            "cost": float(sol.cost)}


def recover_params_from_curve(f_dense, Z_pred, circuit_name):
    return fit_equivalent_circuit(f_dense, Z_pred, circuit_name)


def relative_param_error(recovered, truth):
    errs = {}
    for k, v in truth.items():
        if k in recovered:
            errs[k] = abs(recovered[k] - v) / (abs(v) + 1e-12)
    errs["median"] = float(np.median(list(errs.values()))) if errs else float("nan")
    return errs


def pole_recovery_error(circuit_name, recovered_params, truth_params):
    """Order-invariant pole recovery error.

    Matches recovered time constants to true time constants by sorted value
    (rather than by fixed key name) so that branch-swapped fits for multi-pole
    circuits (e.g. two_tc's tau1/tau2) are not spuriously penalized. Returned
    dict keys are still the true pole names.
    """
    tr = ground_truth_poles(circuit_name, truth_params)      # {name: tau}
    rc = ground_truth_poles(circuit_name, recovered_params)
    tr_items = sorted(tr.items(), key=lambda kv: kv[1])       # sort true poles by tau
    rc_vals = sorted(rc.values())                            # sort recovered taus
    out = {}
    for (name, tv), rv in zip(tr_items, rc_vals):
        out[name] = abs(rv - tv) / (abs(tv) + 1e-12)
    out["median"] = float(np.median([out[n] for n, _ in tr_items])) if tr else float("nan")
    return out


# ============================================================================
# Phase B: Variable-order Voigt circuit + NLSQ fitters + prediction metric
# ============================================================================

def voigt_impedance(f, Rs, R_list, C_list):
    """Voigt (parallel RC) impedance with K branches.

    Z = Rs + Σ R_i / (1 + jω R_i C_i)
    """
    w = _omega(f)
    Z = np.full(w.shape, Rs, dtype=complex)
    for R, C in zip(R_list, C_list):
        Z = Z + R / (1.0 + 1j * w * R * C)
    return Z


def sample_voigt_params(K, rng):
    """Sample K time constants spread across literature-ish ranges.

    Returns dict with Rs, R_list, C_list, taus (sorted time constants).
    """
    Rs = float(rng.uniform(5.0, 30.0))
    R_list = [float(rng.uniform(80.0, 1000.0)) for _ in range(K)]
    # base time constants spread across the measurable band, jittered
    base_tau = np.logspace(-3.5, 0.0, K) if K > 1 else np.array([10 ** rng.uniform(-3.0, -0.5)])
    C_list = []
    for i in range(K):
        tau = float(base_tau[i]) * float(rng.uniform(0.6, 1.6))
        C_list.append(tau / R_list[i])
    taus = [R_list[i] * C_list[i] for i in range(K)]
    return {"Rs": Rs, "R_list": R_list, "C_list": C_list, "taus": sorted(taus)}


def make_voigt_dataset(K, f_min, f_max, n_points, sigma_rel, seed, mode):
    """Generate a complete Voigt dataset with K branches.

    Returns dict with X, Y, f, Z_clean, Z_noisy, params, taus, mode, K.
    """
    rng = np.random.default_rng(seed)
    p = sample_voigt_params(K, rng)
    f = freq_grid(f_min, f_max, n_points)
    Z_clean = voigt_impedance(f, p["Rs"], p["R_list"], p["C_list"])
    Z_noisy = add_proportional_noise(Z_clean, sigma_rel, rng)
    X = omega_features(f, mode, f_min, f_max)
    Y = np.stack([Z_noisy.real, Z_noisy.imag], axis=1).astype(np.float32)
    return {"X": X, "Y": Y, "f": f, "Z_clean": Z_clean, "Z_noisy": Z_noisy,
            "params": p, "taus": p["taus"], "mode": mode, "K": K}


def _voigt_pack(Rs, R_list, C_list):
    """Pack Voigt parameters into log-space vector."""
    return np.log(np.concatenate([[Rs], R_list, C_list]))


def _voigt_unpack(theta, K):
    """Unpack log-space vector to Voigt parameters."""
    v = np.exp(theta)
    return {"Rs": float(v[0]), "R_list": [float(x) for x in v[1:1 + K]],
            "C_list": [float(x) for x in v[1 + K:1 + 2 * K]]}


def fit_voigt(f, Z, K):
    """Fit K-branch Voigt model using log-space NLSQ.

    Returns dict with params, success, cost, rss, K.
    """
    scale = np.median(np.abs(Z)) + 1e-9
    # init: spread taus across band, mid resistances
    init = {"Rs": 10.0, "R_list": [200.0] * K,
            "C_list": [float(t) / 200.0 for t in np.logspace(-3.5, 0.0, K)]}

    def resid(theta):
        p = _voigt_unpack(theta, K)
        Zm = voigt_impedance(f, p["Rs"], p["R_list"], p["C_list"])
        r = (Zm - Z) / scale
        return np.concatenate([r.real, r.imag])

    sol = least_squares(resid, _voigt_pack(init["Rs"], init["R_list"], init["C_list"]),
                        method="lm", max_nfev=5000)
    p = _voigt_unpack(sol.x, K)
    rss = float(np.sum(resid(sol.x) ** 2)) * (scale ** 2)
    return {"params": p, "success": bool(sol.success), "cost": float(sol.cost),
            "rss": rss, "K": K}


def fit_voigt_aic(f, Z, Kmax=4):
    """Fit K=1..Kmax Voigt models, select via AIC.

    Returns dict with params, success, cost, rss, K, aic.
    """
    n = 2 * len(f)
    best = None
    for K in range(1, Kmax + 1):
        r = fit_voigt(f, Z, K)
        p = 1 + 2 * K
        rss = max(r["rss"], 1e-12)
        aic = n * np.log(rss / n) + 2 * p
        cand = dict(r, aic=float(aic))
        if best is None or cand["aic"] < best["aic"]:
            best = cand
    return best


def complex_r2(Z_true, Z_pred):
    """R² metric on stacked [Re, Im] of complex impedance.

    Captures agreement on both magnitude and phase.
    """
    y = np.concatenate([np.asarray(Z_true).real, np.asarray(Z_true).imag])
    p = np.concatenate([np.asarray(Z_pred).real, np.asarray(Z_pred).imag])
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot

# experiment_refine/nmr_peak_recovery.py
"""Peak-recovery metrics for NMR reconstructions: does a model that fits the
spectrum also put the peaks in the right ppm positions? Order-invariant greedy
nearest matching within a ppm tolerance."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def seedwise_min_param_crossing(rows, metric, threshold):
    """Return minimum crossing params per seed and separately list censored seeds."""
    seeds = sorted({row["seed"] for row in rows})
    values, failures = [], []
    for seed in seeds:
        candidates = [row["params"] for row in rows
                      if row["seed"] == seed and row[metric] >= threshold]
        if candidates:
            values.append(min(candidates))
        else:
            failures.append(seed)
    return values, failures


def pick_peaks(x_ppm, y, height_frac=0.05, min_dist_ppm=0.03):
    x = np.asarray(x_ppm, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    ymax = float(np.max(y)) if y.size else 0.0
    if ymax <= 0:
        return np.array([], dtype=np.float64)
    dx = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    dist = max(1, int(round(min_dist_ppm / max(dx, 1e-12))))
    idx, _ = find_peaks(y, height=height_frac * ymax, distance=dist)
    return np.sort(x[idx])


def match_peaks(true_ppm, pred_ppm, tol_ppm):
    true = list(np.asarray(true_ppm, dtype=np.float64).ravel())
    pred = sorted(np.asarray(pred_ppm, dtype=np.float64).ravel().tolist())
    used = [False] * len(pred)
    hits, errs = 0, []
    for t in true:
        best_j, best_d = -1, tol_ppm
        for j, p in enumerate(pred):
            if used[j]:
                continue
            d = abs(p - t)
            if d <= best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            used[best_j] = True
            hits += 1
            errs.append(best_d)
    n_false = used.count(False)
    return {"n_hit": hits, "n_miss": len(true) - hits, "n_false": n_false,
            "abs_errors": errs}


def peak_recovery_metrics(true_ppm, x_ppm, y_pred, tol_ppm=0.04,
                          height_frac=0.05, min_dist_ppm=0.03):
    pred_ppm = pick_peaks(x_ppm, y_pred, height_frac, min_dist_ppm)
    m = match_peaks(true_ppm, pred_ppm, tol_ppm)
    n_true = len(np.asarray(true_ppm).ravel())
    n_pred = len(pred_ppm)
    tp = m["n_hit"]
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    ppm_err = float(np.mean(m["abs_errors"])) if m["abs_errors"] else float("nan")
    return {"ppm_error_mean": ppm_err, "precision": precision, "recall": recall,
            "f1": f1, "n_true": n_true, "n_pred": n_pred}

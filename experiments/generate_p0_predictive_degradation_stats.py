#!/usr/bin/env python3
"""P0-2: statistical summary for predictive degradation curves.

Reads the completed hard-noise degradation raw/summary files and writes paper-facing
AUC, high-noise mean, paired differences, bootstrap CIs, and paired t-test style
summaries. This does not rerun models.
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path("/home/zxc/CodeBase/cofrnet")
IN_DIR = ROOT / "experiment_refine" / "new_experiments_20260701_150828" / "hard_noise_degradation_results_projected_full"
OUT_DIR = ROOT / "experiment_refine" / "p0_predictive_degradation_stats"
MODELS = ["MLP", "CFNN-Hybrid", "KAN"]
METRICS = ["R2", "RMSE", "MSE", "MAE"]
HIGH_NOISE_MIN = 0.5
BOOT = 20000
RNG = np.random.default_rng(20260702)


def load_raw():
    raw_path = IN_DIR / "raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    return raw


def iter_records(raw):
    if isinstance(raw, list):
        for r in raw:
            yield r
    elif isinstance(raw, dict):
        # tolerate either:
        #   {noise: {seed: {model: metrics}}}
        #   {noise: {model: {metric: [seed_values...]}}}
        #   {noise: [records...]}
        for noise_key, payload in raw.items():
            if isinstance(payload, list):
                for r in payload:
                    yield r
            elif isinstance(payload, dict):
                if all(isinstance(v, dict) and any(isinstance(vv, list) for vv in v.values()) for v in payload.values()):
                    for model, metrics_by_name in payload.items():
                        max_len = max((len(v) for v in metrics_by_name.values() if isinstance(v, list)), default=0)
                        for idx in range(max_len):
                            rr = {"noise_level": float(noise_key), "seed": idx, "model": model}
                            for metric, vals in metrics_by_name.items():
                                if isinstance(vals, list) and idx < len(vals):
                                    rr[metric] = vals[idx]
                            yield rr
                else:
                    for seed_key, seed_payload in payload.items():
                        if isinstance(seed_payload, dict):
                            for model, metrics in seed_payload.items():
                                if isinstance(metrics, dict):
                                    rr = {"noise_level": float(noise_key), "seed": int(seed_key), "model": model}
                                    rr.update(metrics)
                                    yield rr


def normalize_records(raw):
    records = []
    for r in iter_records(raw):
        model = r.get("model") or r.get("model_name")
        noise = r.get("noise_level", r.get("noise", r.get("noise_ratio")))
        seed = r.get("seed")
        if model is None or noise is None or seed is None:
            continue
        rec = {"model": model, "noise": float(noise), "seed": int(seed)}
        for m in METRICS:
            if m in r:
                rec[m] = float(r[m])
            elif m.lower() in r:
                rec[m] = float(r[m.lower()])
        records.append(rec)
    if not records:
        raise RuntimeError(f"Could not parse records from {IN_DIR / 'raw.json'}")
    return records


def pivot(records, metric):
    noises = sorted({r["noise"] for r in records})
    seeds = sorted({r["seed"] for r in records})
    data = {m: np.full((len(seeds), len(noises)), np.nan, dtype=float) for m in MODELS}
    si = {s: i for i, s in enumerate(seeds)}
    ni = {n: i for i, n in enumerate(noises)}
    for r in records:
        if r["model"] in data and metric in r:
            data[r["model"]][si[r["seed"]], ni[r["noise"]]] = r[metric]
    return noises, seeds, data


def auc_by_seed(values, noises):
    x = np.asarray(noises, dtype=float)
    return np.trapezoid(values, x=x, axis=1) / (x.max() - x.min())


def mean_high_by_seed(values, noises):
    mask = np.asarray(noises) >= HIGH_NOISE_MIN
    return np.nanmean(values[:, mask], axis=1)


def ci_bootstrap(vals, func=np.mean):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return [float("nan"), float("nan")]
    idx = RNG.integers(0, vals.size, size=(BOOT, vals.size))
    samples = func(vals[idx], axis=1)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def paired_t(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = vals.size
    if n < 2:
        return {"n": int(n), "mean": float(vals.mean()) if n else float("nan"), "std": float("nan"), "t": float("nan"), "p_norm_approx": float("nan")}
    mean = vals.mean()
    std = vals.std(ddof=1)
    t = mean / (std / math.sqrt(n)) if std > 0 else float("inf")
    # Normal approximation is enough for a descriptive reviewer-facing note; avoid scipy dependency assumptions.
    p = 2 * (1 - NormalDist().cdf(abs(t))) if math.isfinite(t) else 0.0
    return {"n": int(n), "mean": float(mean), "std": float(std), "t": float(t), "p_norm_approx": float(p)}


def summarize_metric(records, metric):
    noises, seeds, data = pivot(records, metric)
    out = {"noise_levels": noises, "seeds": seeds, "models": {}}
    per_seed_auc = {}
    per_seed_high = {}
    for model in MODELS:
        vals = data[model]
        auc = auc_by_seed(vals, noises)
        high = mean_high_by_seed(vals, noises)
        per_seed_auc[model] = auc
        per_seed_high[model] = high
        out["models"][model] = {
            "auc_mean": float(np.nanmean(auc)),
            "auc_std": float(np.nanstd(auc, ddof=1)),
            "auc_ci95_bootstrap": ci_bootstrap(auc),
            "high_noise_mean": float(np.nanmean(high)),
            "high_noise_std": float(np.nanstd(high, ddof=1)),
            "high_noise_ci95_bootstrap": ci_bootstrap(high),
            "per_seed_auc": [float(x) for x in auc],
            "per_seed_high_noise": [float(x) for x in high],
        }
    out["paired_differences"] = {}
    for a, b in combinations(MODELS, 2):
        auc_diff = per_seed_auc[a] - per_seed_auc[b]
        high_diff = per_seed_high[a] - per_seed_high[b]
        out["paired_differences"][f"{a}_minus_{b}"] = {
            "auc_diff_mean": float(np.nanmean(auc_diff)),
            "auc_diff_ci95_bootstrap": ci_bootstrap(auc_diff),
            "auc_diff_t_summary": paired_t(auc_diff),
            "high_noise_diff_mean": float(np.nanmean(high_diff)),
            "high_noise_diff_ci95_bootstrap": ci_bootstrap(high_diff),
            "high_noise_diff_t_summary": paired_t(high_diff),
        }
    return out


def write_markdown(stats):
    lines = ["# P0 predictive degradation statistical summary", ""]
    lines.append(f"Input: `{IN_DIR}`")
    lines.append("")
    lines.append("AUC is normalized over noise levels 0.0--0.9. High-noise mean uses noise >= 50%.")
    lines.append("Bootstrap CIs resample seeds; p-values are normal-approx paired summaries for description only.")
    lines.append("")
    for metric, s in stats.items():
        lines.append(f"## {metric}")
        lines.append("")
        direction = "higher is better" if metric == "R2" else "lower is better"
        lines.append(f"Metric direction: {direction}.")
        lines.append("")
        lines.append("| Model | AUC mean±std | AUC 95% CI | High-noise mean±std | High-noise 95% CI |")
        lines.append("|---|---:|---:|---:|---:|")
        for model in MODELS:
            m = s["models"][model]
            lines.append(
                f"| {model} | {m['auc_mean']:.6f} ± {m['auc_std']:.6f} | "
                f"[{m['auc_ci95_bootstrap'][0]:.6f}, {m['auc_ci95_bootstrap'][1]:.6f}] | "
                f"{m['high_noise_mean']:.6f} ± {m['high_noise_std']:.6f} | "
                f"[{m['high_noise_ci95_bootstrap'][0]:.6f}, {m['high_noise_ci95_bootstrap'][1]:.6f}] |"
            )
        lines.append("")
        if metric == "R2":
            lines.append("### Key paired differences for paper wording")
            lines.append("")
            wanted = [
                ("CFNN-Hybrid", "MLP"),
                ("CFNN-Hybrid", "KAN"),
            ]
            for left, right in wanted:
                direct = s["paired_differences"].get(f"{left}_minus_{right}")
                reverse = s["paired_differences"].get(f"{right}_minus_{left}")
                if direct:
                    auc_mean = direct["auc_diff_mean"]
                    auc_ci = direct["auc_diff_ci95_bootstrap"]
                    high_mean = direct["high_noise_diff_mean"]
                    high_ci = direct["high_noise_diff_ci95_bootstrap"]
                elif reverse:
                    auc_mean = -reverse["auc_diff_mean"]
                    auc_ci = [-reverse["auc_diff_ci95_bootstrap"][1], -reverse["auc_diff_ci95_bootstrap"][0]]
                    high_mean = -reverse["high_noise_diff_mean"]
                    high_ci = [-reverse["high_noise_diff_ci95_bootstrap"][1], -reverse["high_noise_diff_ci95_bootstrap"][0]]
                else:
                    continue
                lines.append(
                    f"- {left}_minus_{right}: AUC Δ={auc_mean:.6f} "
                    f"CI [{auc_ci[0]:.6f}, {auc_ci[1]:.6f}], "
                    f"high-noise Δ={high_mean:.6f} "
                    f"CI [{high_ci[0]:.6f}, {high_ci[1]:.6f}]."
                )
            lines.append("")
    lines.append("## Conservative conclusion")
    lines.append("")
    lines.append("For R², CFNN-Hybrid should be described as comparable to tuned MLP on predictive degradation, with at most a small medium/high-noise advantage. The clearer separation is against KAN. This supports replacing the old 47-fold predictive-robustness language with a distinction between predictive degradation and attribution-level noise suppression.")
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = normalize_records(load_raw())
    stats = {metric: summarize_metric(records, metric) for metric in METRICS}
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_markdown(stats)
    print((OUT_DIR / "SUMMARY.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

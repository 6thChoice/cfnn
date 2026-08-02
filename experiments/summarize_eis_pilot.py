"""Aggregate EIS pilot raw results into summary.json + human-readable SUMMARY.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiment_refine" / "eis_pilot_results"

RAW_FILES = {
    "E0": "E0_representation_raw.json", "E1": "E1_sharpness_raw.json",
    "E2": "E2_data_efficiency_raw.json", "E3": "E3_param_efficiency_raw.json",
    "E4": "E4_extrapolation_raw.json", "E5": "E5_nonrational_raw.json",
}
GROUPS = {
    "E0": (["mode", "model"], ["pole_err_median", "curve_mse"]),
    "E1": (["tau_ratio", "model"], ["pole_err_median", "curve_mse"]),
    "E2": (["n_points", "model"], ["pole_err_median"]),
    "E3": (["family", "model"], ["pole_err_median", "params_count"]),
    "E4": (["model"], ["low_f_mse", "high_f_mse"]),
    "E5": (["circuit", "model"], ["pole_err_median", "curve_mse"]),
}


def aggregate(rows, group_keys, metric_keys):
    groups = {}
    for r in rows:
        key = tuple(r.get(k) for k in group_keys)
        groups.setdefault(key, []).append(r)
    out = []
    for key, items in groups.items():
        row = {k: v for k, v in zip(group_keys, key)}
        row["n"] = len(items)
        for m in metric_keys:
            vals = [i[m] for i in items if m in i and i[m] is not None and np.isfinite(i[m])]
            row[f"{m}_mean"] = float(np.mean(vals)) if vals else float("nan")
            row[f"{m}_std"] = float(np.std(vals)) if vals else float("nan")
        out.append(row)
    return out


def main():
    summary, md = {}, ["# EIS Pilot Summary\n"]
    for exp, fname in RAW_FILES.items():
        path = OUT_DIR / fname
        if not path.exists():
            continue
        rows = json.load(open(path))
        gk, mk = GROUPS[exp]
        agg = aggregate(rows, gk, mk)
        summary[exp] = agg
        md.append(f"\n## {exp}\n")
        header = gk + ["n"] + [f"{m}_mean" for m in mk]
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "|".join(["---"] * len(header)) + "|")
        for r in sorted(agg, key=lambda x: [str(x.get(k)) for k in gk]):
            cells = [str(r.get(k)) for k in gk] + [str(r["n"])] + \
                    [f"{r.get(f'{m}_mean', float('nan')):.4g}" for m in mk]
            md.append("| " + " | ".join(cells) + " |")
    json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(md))
    print(f"wrote {OUT_DIR/'summary.json'} and SUMMARY.md")


if __name__ == "__main__":
    main()

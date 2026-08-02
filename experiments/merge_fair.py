"""Merge the per-(task,family) server results into combined raw/summary and
print the fair 'min params to R2>=0.99' tables vs the old (pre-fairness-fix)
numbers. Writes combined raw/summary back into each results dir."""
import json, glob, os
from collections import defaultdict

ER = os.path.dirname(os.path.abspath(__file__))


def load_concat(pattern):
    rows = []
    for fp in sorted(glob.glob(pattern)):
        rows.extend(json.load(open(fp)))
    return rows


def summarize(raw):
    g = defaultdict(list)
    for r in raw:
        g[(r["task"], r["family"], r["params"])].append(r["r2"])
    return [{"task": k[0], "family": k[1], "params": k[2],
             "r2_median": float(sorted(v)[len(v)//2]) if len(v) % 2 else
             float((sorted(v)[len(v)//2-1]+sorted(v)[len(v)//2])/2), "n": len(v)}
            for k, v in g.items()]


def minparams(summ, task, fam):
    pts = sorted((s["params"], s["r2_median"]) for s in summ
                 if s["task"] == task and s["family"] == fam)
    ok = [p for p, r in pts if r >= 0.99]
    best = max((r for _, r in pts), default=float("nan"))
    return (min(ok) if ok else None), best, pts


EXP = {
    "lowbudget": dict(
        raw=f"{ER}/lowbudget_pareto_results/server_fair/pareto_raw_*.json",
        outdir=f"{ER}/lowbudget_pareto_results",
        rawname="pareto_raw_fair.json", sumname="pareto_summary_fair.json",
        tasks=["pole_eps0.001", "pole_eps0.01", "eis_two_tc"],
        old={"pole_eps0.001": (14, 53, "NEVER"), "pole_eps0.01": (14, 19, "NEVER"),
             "eis_two_tc": (36, 22, 83)}),
    "resonance": dict(
        raw=f"{ER}/resonance_pareto_results/server_fair/resonance_raw_*.json",
        outdir=f"{ER}/resonance_pareto_results",
        rawname="resonance_raw_fair.json", sumname="resonance_summary_fair.json",
        tasks=["res_Q_broad", "res_Q_med", "res_Q_sharp", "res_2peak"],
        old={"res_Q_broad": (14, 13, "NEVER"), "res_Q_med": (14, 19, "NEVER"),
             "res_Q_sharp": (14, 19, "NEVER"), "res_2peak": (26, 46, "NEVER")}),
    "nmr": dict(
        raw=f"{ER}/nmr_pareto_results/server_fair/raw_*.json",
        outdir=f"{ER}/nmr_pareto_results",
        rawname="raw_fair.json", sumname="summary_fair.json",
        tasks=["ethanol", "caffeine", "strychnine"],
        old={"ethanol": None, "caffeine": None, "strychnine": None}),
}

for name, cfg in EXP.items():
    raw = load_concat(cfg["raw"])
    summ = summarize(raw)
    json.dump(raw, open(f"{cfg['outdir']}/{cfg['rawname']}", "w"), indent=1)
    json.dump(summ, open(f"{cfg['outdir']}/{cfg['sumname']}", "w"), indent=1)
    print(f"\n===== {name.upper()}  ({len(raw)} rows, {len(summ)} summary pts) =====")
    print(f"{'task':>16} | {'CFNN':>16} {'MLP':>16} {'KAN':>16}   (old: C/M/K)")
    for t in cfg["tasks"]:
        cells = []
        for fam in ["CFNN", "MLP", "KAN"]:
            mp, best, pts = minparams(summ, t, fam)
            cells.append(f"{mp}" if mp is not None else f"NEVER({best:.3f})")
        old = cfg["old"].get(t)
        olds = "/".join(str(x) for x in old) if old else "-"
        print(f"{t:>16} | {cells[0]:>16} {cells[1]:>16} {cells[2]:>16}   (old {olds})")

# NMR peak recovery merge
print("\n===== NMR PEAK RECOVERY =====")
pk = []
for fp in sorted(glob.glob(f"{ER}/nmr_pareto_results/server_fair/peak_recovery_*.json")):
    pk.extend(json.load(open(fp)))
json.dump(pk, open(f"{ER}/nmr_pareto_results/peak_recovery_fair.json", "w"), indent=1)
print(f"{'task':>12} {'family':>6} {'params':>7} {'F1':>6} {'ppmErr':>8} {'pass099':>8} {'nseed':>6}")
for r in sorted(pk, key=lambda r: (r["task"], r["family"])):
    print(f"{r['task']:>12} {r['family']:>6} {r['params']:>7} {r['f1_median']:>6.3f} "
          f"{r['ppm_error_median']:>8.4f} {str(r['passed_r2_099']):>8} {r['n_seeds']:>6}")

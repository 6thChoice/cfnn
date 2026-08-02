"""EIS Phase B: unknown/mismatched model order.
B1a: prediction quality vs true Voigt order K (neural vs classical fixed/AIC/oracle).
B1b: ill-posed sweep (high K, few points, high noise) + catastrophic-failure rate.
Metric = prediction R2/curve_mse on a dense clean grid (NOT pole recovery)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiment_refine"))
import eis_utils as U
import eis_models as M
from param_matched_utils import NumpySafeEncoder, count_trainable_params, find_mlp_config_for_budget, comparison_type

OUT = ROOT / "experiment_refine" / "eis_phaseB_results"; OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 123, 456, 789, 1024]
F_MIN, F_MAX = 1e-2, 1e5


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def _std(Y):
    m = Y.mean(0, keepdims=True); s = Y.std(0, keepdims=True) + 1e-8
    return ((Y - m) / s).astype("float32"), m, s


def neural_predict(builder, ds, dense_f):
    Yb, m, s = _std(ds["Y"])
    Xt = torch.tensor(ds["X"], device=M.DEVICE); Yt = torch.tensor(Yb, device=M.DEVICE)
    model = builder()
    if model is None: return None, None
    model = model.to(M.DEVICE)
    model = M.train_model(model, Xt, Yt, Xt, Yt, 600, 5e-3, patience=80)
    Xd = U.omega_features(dense_f, ds["mode"], F_MIN, F_MAX)
    pred = M.predict_numpy(model, Xd) * s + m
    return pred[:, 0] + 1j * pred[:, 1], count_trainable_params(model)


def eval_row(exp, cfg_extra, model, Zpred, Ztrue, params_count, extra=None):
    r2 = U.complex_r2(Ztrue, Zpred); mse = float(np.mean(np.abs(Zpred - Ztrue) ** 2))
    row = {"exp": exp, "model": model, "params_count": params_count,
           "r2": r2, "curve_mse": mse, "failed": bool(r2 < 0)}
    row.update(cfg_extra)
    if extra: row.update(extra)
    return row


def neural_builders(cfnn_params):
    mlp = find_mlp_config_for_budget(1, cfnn_params, 2, (1, 2, 3), 256)
    return {"CFNN-Hybrid": lambda: M.CFNNHybridEIS(),
            "MLP-matched": lambda c=mlp: M.BudgetMLP(1, 2, c["hidden_dim"], c["num_layers"]),
            "KAN": lambda: M.build_kan(1, 2)}


def classical_rows(ds, dense_f, Ztrue):
    rows = []
    Zmeas = ds["Y"][:, 0] + 1j * ds["Y"][:, 1]

    def predict(fitres):
        p = fitres["params"]
        return U.voigt_impedance(dense_f, p["Rs"], p["R_list"], p["C_list"])

    f2 = U.fit_voigt(ds["f"], Zmeas, 2)
    rows.append(("Classical-fixed2", predict(f2), 1 + 2 * 2, {"fit_K": 2, "fit_success": f2["success"]}))
    fa = U.fit_voigt_aic(ds["f"], Zmeas, 4)
    rows.append(("Classical-AIC", predict(fa), 1 + 2 * fa["K"], {"fit_K": fa["K"], "fit_success": fa["success"]}))
    fo = U.fit_voigt(ds["f"], Zmeas, ds["K"])
    rows.append(("Classical-oracle", predict(fo), 1 + 2 * ds["K"], {"fit_K": ds["K"], "fit_success": fo["success"]}))
    return rows


def run_configs(exp, cfgs, args):
    cfnn_params = count_trainable_params(M.CFNNHybridEIS())
    builders = neural_builders(cfnn_params)
    dense_f = U.freq_grid(F_MIN, F_MAX, 400)
    seeds = SEEDS[:2] if args.smoke else SEEDS
    raw = []
    for cfg in cfgs:
        for seed in seeds:
            set_seed(seed)
            ds = U.make_voigt_dataset(cfg["K"], F_MIN, F_MAX, cfg["n_points"], cfg["noise"], seed, "log")
            Ztrue = U.voigt_impedance(dense_f, ds["params"]["Rs"], ds["params"]["R_list"], ds["params"]["C_list"])
            for name, b in builders.items():
                set_seed(seed)  # identical init state per model (E3 lesson)
                Zp, pc = neural_predict(b, ds, dense_f)
                if Zp is None: continue
                raw.append(eval_row(exp, {**cfg, "seed": seed}, name, Zp, Ztrue, pc,
                                    {"comparison_type": comparison_type(pc, cfnn_params)}))
            for name, Zp, pc, extra in classical_rows(ds, dense_f, Ztrue):
                raw.append(eval_row(exp, {**cfg, "seed": seed}, name, Zp, Ztrue, pc,
                                    {**extra, "comparison_type": "classical_reference"}))
    return raw


def exp_order_mismatch(args):
    npt = 20 if args.smoke else 40
    cfgs = [{"K": K, "n_points": npt, "noise": 0.03} for K in [1, 2, 3, 4]]
    raw = run_configs("B1a", cfgs, args)
    json.dump(raw, open(OUT / "B1a_order_mismatch_raw.json", "w"), indent=2, cls=NumpySafeEncoder)
    return raw


def exp_illposed(args):
    cfgs = [{"K": K, "n_points": npt, "noise": nz}
            for K in [3, 4] for npt in ([12, 40] if args.smoke else [12, 20, 40]) for nz in [0.02, 0.05, 0.10]]
    raw = run_configs("B1b", cfgs, args)
    json.dump(raw, open(OUT / "B1b_illposed_raw.json", "w"), indent=2, cls=NumpySafeEncoder)
    return raw


EXPERIMENTS = {"B1a": exp_order_mismatch, "B1b": exp_illposed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="B1a", choices=["B1a", "B1b", "all"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    for k in (list(EXPERIMENTS) if args.exp == "all" else [args.exp]):
        print(f"[phaseB] running {k}"); EXPERIMENTS[k](args)
    print("[phaseB] done")


if __name__ == "__main__":
    main()

"""EIS pilot: CFNN vs MLP/KAN vs classical circuit fit, four evaluation axes.
Sub-experiments: E0 representation, E1 sharpness recovery, E2 data efficiency,
E3 param efficiency, E4 extrapolation, E5 non-rational stress."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiment_refine"))

import eis_utils as U
import eis_models as M
from param_matched_utils import (NumpySafeEncoder, count_trainable_params,
                                  find_mlp_config_for_budget, budget_metadata,
                                  comparison_type)

OUT_DIR = ROOT / "experiment_refine" / "eis_pilot_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 123, 456, 789, 1024]
F_MIN, F_MAX = 1e-2, 1e5


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def standardize_fit(Y):
    mean = Y.mean(axis=0, keepdims=True)
    std = Y.std(axis=0, keepdims=True) + 1e-8
    return ((Y - mean) / std).astype("float32"), mean, std


def inv_std(Yb, mean, std):
    return Yb * std + mean


def _to_tensor(a):
    return torch.tensor(np.asarray(a, dtype="float32"), device=M.DEVICE)


def fit_one_model(builder, ds, dense_f, circuit, args, target_params=None):
    """Train on the (noisy, standardized) dataset; predict a dense clean curve;
    fit the circuit to that curve to recover physical parameters."""
    X, Y = ds["X"], ds["Y"]
    Yb, mean, std = standardize_fit(Y)
    Xt, Yt = _to_tensor(X), _to_tensor(Yb)
    model = builder()
    if model is None:
        return None
    model = model.to(M.DEVICE)
    model = M.train_model(model, Xt, Yt, Xt, Yt, args.epochs, args.lr,
                          wd=1e-5, patience=args.patience)
    params_count = count_trainable_params(model)

    Xd = U.omega_features(dense_f, ds["mode"], F_MIN, F_MAX)
    pred_b = M.predict_numpy(model, Xd)
    pred = inv_std(pred_b, mean, std)
    Z_pred = pred[:, 0] + 1j * pred[:, 1]

    Z_true_dense = U.CIRCUITS[circuit]["func"](dense_f, **ds["params"])
    curve_mse = float(np.mean(np.abs(Z_pred - Z_true_dense) ** 2))

    rec = U.recover_params_from_curve(dense_f, Z_pred, circuit)
    pole_err = U.pole_recovery_error(circuit, rec["params"], ds["params"])
    param_err = U.relative_param_error(rec["params"], ds["params"])
    result = {"params_count": params_count, "curve_mse": curve_mse,
              "pole_err_median": pole_err["median"], "param_err_median": param_err["median"],
              "pole_err": pole_err, "recovered": rec["params"]}
    if target_params is not None:
        result["target_params"] = int(target_params)
        result["param_ratio"] = round(params_count / target_params, 6)
        result["comparison_type"] = comparison_type(params_count, target_params)
    return result


def classical_baseline(ds, dense_f, circuit):
    """Domain-standard: fit circuit directly to the noisy measured points."""
    Z_meas = ds["Y"][:, 0] + 1j * ds["Y"][:, 1]
    rec = U.fit_equivalent_circuit(ds["f"], Z_meas, circuit)
    pole_err = U.pole_recovery_error(circuit, rec["params"], ds["params"])
    param_err = U.relative_param_error(rec["params"], ds["params"])
    return {"params_count": len(U.CIRCUITS[circuit]["param_names"]),
            "curve_mse": float("nan"),
            "pole_err_median": pole_err["median"],
            "param_err_median": param_err["median"], "pole_err": pole_err,
            "recovered": rec["params"], "comparison_type": "classical_reference",
            "fit_success": rec["success"], "fit_cost": rec["cost"]}


def MODEL_BUILDERS(cfnn_params):
    mlp_m = find_mlp_config_for_budget(1, cfnn_params, 2, (1, 2, 3), 256)
    mlp_l = find_mlp_config_for_budget(1, max(4 * cfnn_params, 1000), 2, (2, 3), 512)
    return {
        "CFNN-Hybrid": lambda: M.CFNNHybridEIS(),
        "MLP-matched": lambda c=mlp_m: M.BudgetMLP(1, 2, c["hidden_dim"], c["num_layers"]),
        "MLP-large": lambda c=mlp_l: M.BudgetMLP(1, 2, c["hidden_dim"], c["num_layers"]),
        "KAN": lambda: M.build_kan(1, 2),
    }


def save_json(obj, name):
    with open(OUT_DIR / name, "w") as fh:
        json.dump(obj, fh, indent=2, cls=NumpySafeEncoder)


TWO_TC_BASE = dict(Rs=10.0, R1=300.0, C1=1e-4, R2=800.0, C2=1e-6)


def exp_representation(args):
    """E0: linear vs sqrt vs log omega representation on the two-time-constant circuit."""
    dense_f = U.freq_grid(F_MIN, F_MAX, 400)
    cfnn_params = count_trainable_params(M.CFNNHybridEIS())
    builders = MODEL_BUILDERS(cfnn_params)
    raw = []
    modes = ["linear", "sqrt", "log"]
    circuit = "two_tc"
    for mode in modes:
        for seed in (SEEDS[:2] if args.smoke else SEEDS):
            set_seed(seed)
            ds = U.make_dataset(circuit, TWO_TC_BASE, F_MIN, F_MAX,
                                n_points=(40 if args.smoke else 80),
                                sigma_rel=0.02, seed=seed, mode=mode)
            for name, b in builders.items():
                res = fit_one_model(b, ds, dense_f, circuit, args, target_params=cfnn_params)
                if res is None:
                    continue
                raw.append({"exp": "E0", "mode": mode, "model": name, "seed": seed, **res})
    save_json(raw, "E0_representation_raw.json")
    return raw


TAU_RATIOS = [1.5, 3.0, 10.0, 30.0, 100.0]  # tau1/tau2: close (sharp) -> separated
N_POINTS_SWEEP = [10, 20, 40, 80]


def exp_sharpness(args):
    """E1: recover time constants as the two RC arcs merge (tau ratio sweep)."""
    dense_f = U.freq_grid(F_MIN, F_MAX, 400)
    cfnn_params = count_trainable_params(M.CFNNHybridEIS())
    builders = MODEL_BUILDERS(cfnn_params)
    raw = []
    base_tau2 = TWO_TC_BASE["R2"] * TWO_TC_BASE["C2"]
    for ratio in TAU_RATIOS:
        # keep R1,R2 fixed; set C1 so that tau1 = ratio * tau2
        C1 = ratio * base_tau2 / TWO_TC_BASE["R1"]
        params = dict(TWO_TC_BASE, C1=C1)
        for seed in (SEEDS[:2] if args.smoke else SEEDS):
            set_seed(seed)
            ds = U.make_dataset("two_tc", params, F_MIN, F_MAX,
                                n_points=(40 if args.smoke else 80),
                                sigma_rel=0.02, seed=seed, mode=args.rep)
            for name, b in builders.items():
                res = fit_one_model(b, ds, dense_f, "two_tc", args, target_params=cfnn_params)
                if res is None:
                    continue
                raw.append({"exp": "E1", "tau_ratio": ratio, "model": name, "seed": seed, **res})
            cb = classical_baseline(ds, dense_f, "two_tc")
            raw.append({"exp": "E1", "tau_ratio": ratio, "model": "Classical-NLSQ", "seed": seed, **cb})
    save_json(raw, "E1_sharpness_raw.json")
    return raw


def exp_data_efficiency(args):
    """E2: recovery vs number of measured frequencies (rapid-EIS regime)."""
    dense_f = U.freq_grid(F_MIN, F_MAX, 400)
    cfnn_params = count_trainable_params(M.CFNNHybridEIS())
    builders = MODEL_BUILDERS(cfnn_params)
    raw = []
    for npt in N_POINTS_SWEEP:
        for seed in (SEEDS[:2] if args.smoke else SEEDS):
            set_seed(seed)
            ds = U.make_dataset("two_tc", TWO_TC_BASE, F_MIN, F_MAX,
                                n_points=npt, sigma_rel=0.02, seed=seed, mode=args.rep)
            for name, b in builders.items():
                res = fit_one_model(b, ds, dense_f, "two_tc", args, target_params=cfnn_params)
                if res is None:
                    continue
                raw.append({"exp": "E2", "n_points": npt, "model": name, "seed": seed, **res})
            cb = classical_baseline(ds, dense_f, "two_tc")
            raw.append({"exp": "E2", "n_points": npt, "model": "Classical-NLSQ", "seed": seed, **cb})
    save_json(raw, "E2_data_efficiency_raw.json")
    return raw


CFNN_UNIT_SWEEP = [2, 4, 6, 8]
MLP_BUDGET_SWEEP = [60, 150, 400, 1000]


def exp_param_efficiency(args):
    """E3: recovery vs trainable params — CFNN capacity sweep vs MLP budget sweep."""
    dense_f = U.freq_grid(F_MIN, F_MAX, 400)
    cfnn_params = count_trainable_params(M.CFNNHybridEIS())
    raw = []
    seeds = SEEDS[:2] if args.smoke else SEEDS
    for seed in seeds:
        ds = U.make_dataset("two_tc", TWO_TC_BASE, F_MIN, F_MAX,
                            n_points=(40 if args.smoke else 80),
                            sigma_rel=0.02, seed=seed, mode=args.rep)
        for nu in CFNN_UNIT_SWEEP:
            set_seed(seed)
            res = fit_one_model(lambda nu=nu: M.CFNNHybridEIS(n_units=nu), ds, dense_f, "two_tc", args, target_params=cfnn_params)
            if res is None:
                continue
            raw.append({"exp": "E3", "family": "CFNN", "knob": nu, "model": f"CFNN-u{nu}",
                        "seed": seed, **res})
        for tgt in MLP_BUDGET_SWEEP:
            set_seed(seed)
            cfg = find_mlp_config_for_budget(1, tgt, 2, (1, 2, 3), 256)
            res = fit_one_model(lambda c=cfg: M.BudgetMLP(1, 2, c["hidden_dim"], c["num_layers"]),
                                ds, dense_f, "two_tc", args, target_params=cfnn_params)
            if res is None:
                continue
            raw.append({"exp": "E3", "family": "MLP", "knob": tgt, "model": f"MLP-{tgt}",
                        "seed": seed, **res})
    save_json(raw, "E3_param_efficiency_raw.json")
    return raw


def exp_extrapolation(args):
    """E4: train on mid-band only, predict held-out low-f and high-f tails."""
    cfnn_params = count_trainable_params(M.CFNNHybridEIS())
    builders = MODEL_BUILDERS(cfnn_params)
    raw = []
    f_lo_hi = (F_MIN, F_MAX)
    f_mid = (1e0, 1e3)  # training window (Hz)
    circuit = "randles_warburg"
    base = dict(Rs=10.0, Rct=400.0, Cdl=1e-5, sigma=80.0)
    dense_full = U.freq_grid(*f_lo_hi, 400)
    low_mask = dense_full < f_mid[0]
    high_mask = dense_full > f_mid[1]
    for seed in (SEEDS[:2] if args.smoke else SEEDS):
        set_seed(seed)
        ds = U.make_dataset(circuit, base, f_mid[0], f_mid[1],
                            n_points=(40 if args.smoke else 80),
                            sigma_rel=0.02, seed=seed, mode=args.rep)
        Z_full = U.CIRCUITS[circuit]["func"](dense_full, **base)
        for name, b in builders.items():
            model_builder = b
            # reuse fit_one_model's training, but evaluate on full band manually:
            X, Y = ds["X"], ds["Y"]
            Yb, mean, std = standardize_fit(Y)
            Xt, Yt = _to_tensor(X), _to_tensor(Yb)
            model = model_builder()
            if model is None:
                continue
            model = model.to(M.DEVICE)
            model = M.train_model(model, Xt, Yt, Xt, Yt, args.epochs, args.lr, patience=args.patience)
            Xd = U.omega_features(dense_full, args.rep, f_mid[0], f_mid[1])
            pred = inv_std(M.predict_numpy(model, Xd), mean, std)
            Zp = pred[:, 0] + 1j * pred[:, 1]
            low_mse = float(np.mean(np.abs(Zp[low_mask] - Z_full[low_mask]) ** 2))
            high_mse = float(np.mean(np.abs(Zp[high_mask] - Z_full[high_mask]) ** 2))
            raw.append({"exp": "E4", "model": name, "seed": seed,
                        "params_count": count_trainable_params(model),
                        "low_f_mse": low_mse, "high_f_mse": high_mse})
        # Classical extrapolation baseline: fit circuit to mid-band noisy data,
        # then predict the full band from the recovered params.
        rec = U.fit_equivalent_circuit(ds["f"], ds["Y"][:, 0] + 1j * ds["Y"][:, 1], circuit)
        Zc = U.CIRCUITS[circuit]["func"](dense_full, **rec["params"])
        low_mse = float(np.mean(np.abs(Zc[low_mask] - Z_full[low_mask]) ** 2))
        high_mse = float(np.mean(np.abs(Zc[high_mask] - Z_full[high_mask]) ** 2))
        raw.append({"exp": "E4", "model": "Classical-NLSQ", "seed": seed,
                    "params_count": len(U.CIRCUITS[circuit]["param_names"]),
                    "low_f_mse": low_mse, "high_f_mse": high_mse,
                    "fit_success": rec["success"]})
    save_json(raw, "E4_extrapolation_raw.json")
    return raw


NONRATIONAL_CASES = [
    ("randles_warburg", dict(Rs=10.0, Rct=400.0, Cdl=1e-5, sigma=80.0)),
    ("randles_cpe", dict(Rs=10.0, Rct=400.0, Q=1e-5, alpha=0.75)),
]


def exp_nonrational(args):
    """E5: fractional-order (Warburg / CPE) stress — graceful degradation or blow-up?"""
    cfnn_params = count_trainable_params(M.CFNNHybridEIS())
    builders = MODEL_BUILDERS(cfnn_params)
    dense_f = U.freq_grid(F_MIN, F_MAX, 400)
    raw = []
    for circuit, base in NONRATIONAL_CASES:
        for seed in (SEEDS[:2] if args.smoke else SEEDS):
            set_seed(seed)
            ds = U.make_dataset(circuit, base, F_MIN, F_MAX,
                                n_points=(40 if args.smoke else 80),
                                sigma_rel=0.02, seed=seed, mode=args.rep)
            for name, b in builders.items():
                res = fit_one_model(b, ds, dense_f, circuit, args, target_params=cfnn_params)
                if res is None:
                    continue
                raw.append({"exp": "E5", "circuit": circuit, "model": name, "seed": seed, **res})
            cb = classical_baseline(ds, dense_f, circuit)
            raw.append({"exp": "E5", "circuit": circuit, "model": "Classical-NLSQ", "seed": seed, **cb})
    save_json(raw, "E5_nonrational_raw.json")
    return raw


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E0",
                    choices=["E0", "E1", "E2", "E3", "E4", "E5", "all"])
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rep", default="linear", choices=["linear", "sqrt", "log"])
    return ap


EXPERIMENTS = {"E0": exp_representation, "E1": exp_sharpness, "E2": exp_data_efficiency, "E3": exp_param_efficiency, "E4": exp_extrapolation, "E5": exp_nonrational}


def main():
    args = build_argparser().parse_args()
    if args.smoke:
        args.epochs = 60
    targets = list(EXPERIMENTS) if args.exp == "all" else [args.exp]
    for key in targets:
        print(f"[eis-pilot] running {key} ...")
        EXPERIMENTS[key](args)
    print("[eis-pilot] done")


if __name__ == "__main__":
    main()

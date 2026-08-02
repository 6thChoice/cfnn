"""Common-budget benchmark for independent and controlled core targets.

This runner is intentionally small and deterministic enough for smoke tests.
It keeps train/validation/test roles separate and writes raw per-seed records.
The full matrix is enabled by increasing --seeds and --epochs.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import r2_score

from fair_baselines import (build_mlp, build_gaussian_rbf, build_pykan,
                            build_cofrnet, build_rational_nn, build_siren)
from eis_models import CFNNHybridEIS


def targets(name, n=512, d=1, seed=7):
    if name.startswith("nmr_"):
        import nmr_data as nmr
        molecule = name.removeprefix("nmr_")
        ds = nmr.make_nmr_dataset(molecule, n, 0.03, seed, n_dense=n)
        return ds["X"], ds["Y"]
    if name == "energy":
        import pandas as pd
        frame = pd.read_csv(Path(__file__).with_name("energy_efficiency.csv"))
        cols = [c for c in frame.columns if c.upper().startswith("X")][:8]
        ycol = [c for c in frame.columns if c.upper() == "Y1"][0]
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(frame))
        return (frame[cols].to_numpy(dtype="float32")[order],
                frame[[ycol]].to_numpy(dtype="float32")[order])
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, d)).astype("float32")
    if name == "gaussian_peaks":
        centers = np.array([-.68, -.15, .42], dtype="float32")[:d+2]
        widths = np.array([.07, .16, .045], dtype="float32")[:len(centers)]
        y = sum(np.exp(-((x[:, 0] - c) / w) ** 2) for c, w in zip(centers, widths))
        y = y[:, None]
    elif name == "fourier":
        z = x[:, 0]
        y = (0.55*np.sin(3*np.pi*z) + 0.30*np.cos(11*np.pi*z)
             + 0.15*np.sin(19*np.pi*z))[:, None]
    elif name == "nonseparable":
        if d < 3: raise ValueError("nonseparable requires d>=3")
        y1 = np.exp(-3*(x[:, 0]**2 + x[:, 1]**2)) * np.cos(5*x[:, 2])
        y2 = x[:, 0]*x[:, 1] + .25*np.sin(7*x[:, 2])
        y = np.stack([y1, y2], axis=1)
    elif name.startswith("pole"):
        eps = 1e-3 if name.endswith("sharp") else 1e-2
        z = x[:, 0]
        y = (1.0 / (z*z + eps))[:, None]
    elif name == "resonance":
        z = x[:, 0]
        y = (1.0 / ((z - 0.15)**2 + 0.04**2))[:, None]
    elif name == "cusp":
        z = x[:, 0]
        y = (np.abs(z - 0.13) ** 0.35)[:, None]
    elif name == "kink":
        z = x[:, 0]
        y = np.maximum(0.0, z + 0.18)[:, None]
    elif name == "compact_bump":
        z = (x[:, 0] - 0.2) / 0.32
        y = np.where(np.abs(z) < 1.0, (1.0 - z*z) ** 3, 0.0)[:, None]
    elif name == "skew_peak":
        z = x[:, 0] - 0.1
        y = (np.exp(-((z / 0.12) ** 2)) * (1.0 + 1.8 * np.tanh(5.0*z)))[:, None]
    elif name == "manifold_bump":
        if d < 3: raise ValueError("manifold_bump requires d>=3")
        u = x[:, 0] + 0.45 * x[:, 1] ** 2
        v = x[:, 2] - 0.35 * np.sin(3 * x[:, 0])
        y = np.exp(-((u - 0.1) ** 2 / 0.06 + (v + 0.2) ** 2 / 0.12))[:, None]
    else: raise ValueError(name)
    return x, y.astype("float32")


def build(family, d, out, budget, seed):
    if family == "CFNN-Hybrid": return CFNNHybridEIS(d, out, n_units=max(1, budget//40), degree=3), {"family": family}
    if family == "MLP": return build_mlp(d, out, max(4, budget//(d+out+2)), 2)
    if family == "Gaussian RBF": return build_gaussian_rbf(d, out, max(4, budget//(d+out+1)))
    if family == "KAN": return build_pykan(d, out, max(3, budget//20), grid=5, seed=seed)
    if family == "CoFrNet-Standard": return build_cofrnet(d, out, max(1, budget//20), 3)
    if family == "Rational activation NN": return build_rational_nn(d, out, max(4, budget//(d+out+2)), 2)
    if family == "SIREN": return build_siren(d, out, max(4, budget//(d+out+2)), 2)
    raise ValueError(family)


def train(model, xtr, ytr, xv, yv, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.float()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    loss = nn.MSELoss(); best = None; best_val = float("inf"); best_epoch = 1; patience = 0
    for epoch in range(epochs):
        model.train(); opt.zero_grad(); pred = model(xtr); l = loss(pred, ytr)
        l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        with torch.no_grad(): val = loss(model(xv), yv).item()
        if val < best_val:
            best_val, best, best_epoch, patience = val, {k: v.detach().clone() for k,v in model.state_dict().items()}, epoch + 1, 0
        else: patience += 1
        if patience >= 50: break
    if best is not None: model.load_state_dict(best)
    return model, best_epoch, best_val


def refit(model, x, y, epochs, seed):
    torch.manual_seed(seed + 1000003); np.random.seed(seed + 1000003)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    loss = nn.MSELoss()
    for _ in range(max(1, epochs)):
        model.train(); opt.zero_grad(); value = loss(model(x), y)
        value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiment_refine/independent_benchmark_results/raw.json")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--seeds", default="42,123,456,789,1024")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tasks", default="independent", choices=["independent", "core", "all"])
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.smoke: seeds, args.epochs = seeds[:1], 5
    independent = [("gaussian_peaks",1), ("fourier",1), ("nonseparable",3),
                   ("cusp",1), ("kink",1), ("compact_bump",1),
                   ("skew_peak",1), ("manifold_bump",3)]
    core = [("pole_sharp",1), ("pole_broad",1), ("resonance",1)]
    tasks = independent if args.tasks == "independent" else core if args.tasks == "core" else independent + core
    families = ["CFNN-Hybrid","MLP","Gaussian RBF","KAN","CoFrNet-Standard","Rational activation NN","SIREN"]
    budgets = [32,64,128,256,512] if not args.smoke else [64]
    records=[]
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("[]")
    for task,d in tasks:
        x,y=targets(task, d=d); n=len(x); i=int(.6*n); j=int(.8*n)
        xt, xv, xe = map(torch.from_numpy, (x[:i],x[i:j],x[j:]))
        # Fit target normalization on training observations only.  This is
        # essential for pole-like targets whose raw peak height dominates MSE.
        ymean, ystd = y[:i].mean(axis=0, keepdims=True), y[:i].std(axis=0, keepdims=True) + 1e-6
        yn = (y - ymean) / ystd
        yt, yv, ye = map(torch.from_numpy, (yn[:i],yn[i:j],yn[j:]))
        for family in families:
            for budget in budgets:
                for seed in seeds:
                    torch.manual_seed(seed); t=time.time(); model, meta=build(family,d,y.shape[1],budget,seed)
                    model, selected_epochs, val=train(model,xt,yt,xv,yv,args.epochs,seed)
                    # Selection and final fitting use separate model instances.
                    final_model, _ = build(family,d,y.shape[1],budget,seed + 1000003)
                    final_model = refit(final_model, torch.cat([xt, xv]), torch.cat([yt, yv]), selected_epochs, seed)
                    final_model.eval()
                    with torch.no_grad(): pred=final_model(xe).detach().cpu().numpy()
                    with torch.no_grad(): final_val = float(nn.MSELoss()(final_model(torch.cat([xt,xv])), torch.cat([yt,yv])))
                    record={"task":task,"family":family,"budget":budget,"seed":seed,
                            "actual_parameters":sum(p.numel() for p in final_model.parameters() if p.requires_grad),
                            "r2":float(r2_score(ye.numpy(),pred,multioutput="variance_weighted")),
                            "val_mse":float(final_val), "selection_val_mse":float(val),
                            "selected_epochs":selected_epochs,"seconds":time.time()-t,
                            "target_normalization":"training mean/std; R2 invariant to affine target scaling",
                            "metadata":meta}
                    records.append(record); out_path.write_text(json.dumps(records,indent=2))
                    print(task, family, budget, seed, record["r2"], flush=True)

if __name__ == "__main__": main()

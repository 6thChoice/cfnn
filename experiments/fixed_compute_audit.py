"""Fixed-optimizer-step audit for already matched architectures.

This intentionally reports a separate curve from validation-selected early
stopping. It uses the architecture/configuration selected by the matched
benchmark and evaluates at predetermined step budgets.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import r2_score

from budget_matched_benchmark import DEVICE, build
from run_independent_benchmark import targets


def run_steps(model, x, y, steps, lr, wd, seed):
    torch.manual_seed(seed + 7000001)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss = nn.MSELoss()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True); value = loss(model(x), y)
        value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    return model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--matched", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default="20,40,80"); ap.add_argument("--task", default="core")
    args = ap.parse_args(); payload = json.loads(Path(args.matched).read_text())
    records = payload["records"] if isinstance(payload, dict) else payload
    steps = [int(x) for x in args.steps.split(",")]; out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    result = []
    for rec in records:
        if rec["task"] not in {"pole_sharp", "pole_broad", "resonance", "gaussian_peaks", "fourier", "nonseparable"}:
            continue
        task = rec["task"]; d = 3 if task == "nonseparable" else 1; x, y = targets(task, d=d); n=len(x); i,j=int(.6*n),int(.8*n)
        ym,ys=y[:i].mean(0,keepdims=True),y[:i].std(0,keepdims=True)+1e-6; yn=(y-ym)/ys
        xt=torch.from_numpy(x[:i]).to(DEVICE); yt=torch.from_numpy(yn[:i]).to(DEVICE); xe=torch.from_numpy(x[j:]).to(DEVICE); ye=yn[j:]
        lr=rec["selected_hyperparameters"]["lr"]; wd=rec["selected_hyperparameters"]["weight_decay"]
        for step in steps:
            started=time.time(); model,_=build(rec["family"],d,y.shape[1],rec["config"],rec["seed"]+1000003); model=run_steps(model,xt,yt,step,lr,wd,rec["seed"])
            with torch.no_grad(): pred=model(xe).detach().cpu().numpy()
            result.append({"task":task,"family":rec["family"],"target_budget":rec["target_budget"],"actual_parameters":rec["actual_parameters"],"seed":rec["seed"],"steps":step,"r2":float(r2_score(ye,pred,multioutput="variance_weighted")),"wall_seconds":time.time()-started})
    out.write_text(json.dumps(result,indent=2)); print(f"wrote {len(result)} fixed-compute records to {out}")


if __name__ == "__main__": main()

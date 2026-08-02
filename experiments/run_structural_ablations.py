"""Small preregistered ablation of the projected additive rational realization."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import r2_score

from eis_models import CFNNHybridEIS, DEVICE


def count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def target(x):
    z = x[:, :1]
    return torch.cat([1.0 / (z*z + 0.03), torch.exp(-18*z*z)], dim=1)


def run(seed, variant):
    torch.manual_seed(seed); np.random.seed(seed)
    x = torch.linspace(-1, 1, 240).reshape(-1, 1)
    y = target(x)
    tr, va, te = slice(0, 144), slice(144, 192), slice(192, 240)
    variant_name = variant.pop('name', 'unknown')
    model = CFNNHybridEIS(output_dim=2, n_units=6, degree=3, **variant).to(DEVICE)
    x, y = x.to(DEVICE), y.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    best, state = float('inf'), None
    for _ in range(80):
        model.train(); opt.zero_grad()
        loss = torch.mean((model(x[tr]) - y[tr]) ** 2)
        if not torch.isfinite(loss): break
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        with torch.no_grad():
            val = torch.mean((model(x[va]) - y[va]) ** 2).item()
        if np.isfinite(val) and val < best:
            best, state = val, {k: v.detach().clone() for k, v in model.state_dict().items()}
    if state is not None: model.load_state_dict(state)
    with torch.no_grad(): pred = model(x[te]).cpu().numpy()
    return {'seed': seed, 'variant': variant_name, 'parameters': count(model),
            'r2': float(np.mean([r2_score(y[te].cpu()[:, j], pred[:, j]) for j in range(2)])),
            'status': 'ok' if state is not None else 'optimization_failure'}


if __name__ == '__main__':
    variants = {
        'full': dict(shared_projection=False, skip=True, squared_denominator=True, epsilon=0.1),
        'shared_pq': dict(shared_projection=True, skip=True, squared_denominator=True, epsilon=0.1),
        'no_skip': dict(shared_projection=False, skip=False, squared_denominator=True, epsilon=0.1),
        'abs_denominator': dict(shared_projection=False, skip=True, squared_denominator=False, epsilon=0.1),
        'epsilon_01': dict(shared_projection=False, skip=True, squared_denominator=True, epsilon=0.01),
    }
    out = [run(seed, cfg | {'name': name}) for name, cfg in variants.items() for seed in [42,123,456,789,1024,2048,4096,8192,16384,32768]]
    Path('experiment_refine/structural_ablation_results.json').write_text(json.dumps(out, indent=2))
    print('wrote', len(out), 'records')

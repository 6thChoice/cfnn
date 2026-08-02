"""EIS pilot models: rational CFNN-Hybrid, budget MLP, optional KAN, training loop.
Adapted from run_pole_sharpness_sweep.py for 1-input -> 2-output regression."""
from __future__ import annotations

import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BudgetMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.Tanh()]
            d = hidden_dim
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RationalUnit(nn.Module):
    def __init__(self, input_dim, output_dim, degree, shared_projection=False,
                 epsilon=0.1, squared_denominator=True):
        super().__init__()
        self.P_proj = nn.Linear(input_dim, output_dim)
        self.P_coeffs = nn.Parameter(torch.randn(output_dim, degree + 1) * 0.05)
        self.Q_proj = self.P_proj if shared_projection else nn.Linear(input_dim, output_dim)
        self.Q_coeffs = nn.Parameter(torch.randn(output_dim, degree + 1) * 0.05)
        self.degree = degree
        self.epsilon = epsilon
        self.squared_denominator = squared_denominator

    def forward(self, x):
        def _poly(proj, coeffs):
            z = torch.tanh(proj(x))
            powers = [torch.ones_like(z)]
            for _ in range(1, self.degree + 1):
                powers.append(powers[-1] * z)
            return torch.sum(torch.stack(powers, dim=-1) * coeffs, dim=2)
        q = _poly(self.Q_proj, self.Q_coeffs)
        denominator = q ** 2 if self.squared_denominator else q.abs()
        return _poly(self.P_proj, self.P_coeffs) / (denominator + self.epsilon)


class CFNNHybridEIS(nn.Module):
    def __init__(self, input_dim=1, output_dim=2, n_units=6, degree=5,
                 shared_projection=False, epsilon=0.1, squared_denominator=True,
                 skip=True):
        super().__init__()
        self.linear_skip = nn.Linear(input_dim, output_dim) if skip else None
        self.units = nn.ModuleList([RationalUnit(input_dim, output_dim, degree,
            shared_projection=shared_projection, epsilon=epsilon,
            squared_denominator=squared_denominator) for _ in range(n_units)])

    def forward(self, x):
        out = self.linear_skip(x) if self.linear_skip is not None else torch.zeros(
            x.shape[0], self.units[0].P_coeffs.shape[0], device=x.device, dtype=x.dtype)
        for u in self.units:
            out = out + u(x)
        return out


class CFNNMoE(nn.Module):
    """Gated mixture of rational experts:
        out = skip(x) + sum_e softmax(gate(x))_e * expert_e(x).
    The plain CFNN sums rational units ungated -- all units see the global loss
    and fail to specialize as the number of sharp peaks grows (credit
    assignment). Here a small MLP gate over the input can localize each expert
    to a peak neighborhood. Tests whether gating extends CFNN's few-peak sweet
    spot to more peaks -- and whether it stays parameter-economical vs MLP once
    the gate is paid for."""

    def __init__(self, input_dim=1, output_dim=1, n_experts=4, degree=5, gate_hidden=8):
        super().__init__()
        self.linear_skip = nn.Linear(input_dim, output_dim)
        self.experts = nn.ModuleList(
            [RationalUnit(input_dim, output_dim, degree) for _ in range(n_experts)])
        self.gate = nn.Sequential(nn.Linear(input_dim, gate_hidden), nn.Tanh(),
                                  nn.Linear(gate_hidden, n_experts))
        self.n_experts = n_experts

    def forward(self, x):
        w = torch.softmax(self.gate(x), dim=-1)              # (B, n_experts)
        outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, n_experts, out_dim)
        mixed = (w.unsqueeze(-1) * outs).sum(dim=1)          # (B, out_dim)
        return self.linear_skip(x) + mixed


_KAN_PATH = ROOT / "git_codebase" / "noise_robustness_runner" / "code" / "interpretable_experiment" / "models"
if _KAN_PATH.exists():
    sys.path.insert(0, str(_KAN_PATH))
try:
    from kan_model import KAN as _RepoKAN
except Exception:
    _RepoKAN = None


class _KanShapeFix(nn.Module):
    """Wrap the vendored KAN so forward always returns (n, output_dim). The
    vendored KAN squeezes the last dim when output_dim==1, yielding (n,), which
    then broadcasts to (n,n) against a (n,1) target in nn.MSELoss -- a silent
    loss corruption. Re-add the trailing dim."""

    def __init__(self, kan, output_dim):
        super().__init__()
        self.kan = kan
        self.output_dim = output_dim

    def forward(self, x):
        out = self.kan(x)
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        return out


def build_kan(input_dim=1, output_dim=2, hidden=8, layers=2, grid=8, order=3):
    if _RepoKAN is None:
        return None
    kan = _RepoKAN(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden,
                   num_layers=layers, grid_size=grid, spline_order=order)
    return _KanShapeFix(kan, output_dim)


def train_model(model, X_tr, y_tr, X_val, y_val, epochs, lr, wd=1e-5, patience=80):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_sd = deepcopy(model.state_dict())
    best_val, stale = float("inf"), 0
    crit = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(X_tr)
        loss = crit(pred, y_tr)
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                v = crit(model(X_val), y_val).item()
            if math.isfinite(v) and v < best_val - 1e-6:
                best_val, stale = v, 0
                best_sd = deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    break
    model.load_state_dict(best_sd)
    return model


def predict_numpy(model, X_np):
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(np.asarray(X_np, dtype="float32"), device=DEVICE)
        out = model(xt)
    return out.cpu().numpy()

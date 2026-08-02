"""Baselines used by the revised CFNN comparisons.

The module deliberately keeps model construction separate from experiment
selection.  Every builder returns a model plus a metadata dictionary so that
the reported parameter count and implementation identity are serialized with
the raw result.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _meta(model, family: str, **kwargs) -> Dict:
    return {"family": family, "actual_parameters": parameter_count(model), **kwargs}


def build_mlp(input_dim, output_dim, hidden=32, depth=2, activation="tanh"):
    act = nn.Tanh if activation == "tanh" else nn.GELU
    layers, d = [], input_dim
    for _ in range(depth):
        layers.extend([nn.Linear(d, hidden), act()])
        d = hidden
    layers.append(nn.Linear(d, output_dim))
    model = nn.Sequential(*layers)
    return model, _meta(model, "MLP", hidden=hidden, depth=depth, activation=activation)


def build_fourier_mlp(input_dim, output_dim, frequencies=16, hidden=32,
                      depth=2, scale=4.0, activation="gelu"):
    """MLP over a fixed random Fourier feature map."""
    class FourierFeatureMLP(nn.Module):
        def __init__(self):
            super().__init__()
            projection = torch.randn(input_dim, frequencies) * float(scale)
            self.register_buffer("projection", projection)
            act = nn.GELU if activation == "gelu" else nn.Tanh
            layers, current = [], 2 * frequencies
            for _ in range(depth):
                layers.extend([nn.Linear(current, hidden), act()])
                current = hidden
            layers.append(nn.Linear(current, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            projected = 2.0 * torch.pi * x @ self.projection
            features = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
            return self.network(features)

    model = FourierFeatureMLP()
    return model, _meta(
        model, "Fourier-feature MLP", frequencies=frequencies, hidden=hidden,
        depth=depth, scale=float(scale), activation=activation,
        feature_projection="fixed Gaussian",
    )


def build_gaussian_rbf(input_dim, output_dim, hidden=32, grid_range=(-1.0, 1.0)):
    """Fixed-centre Gaussian RBF network; this is not a KAN implementation."""
    class GaussianRBF(nn.Module):
        def __init__(self):
            super().__init__()
            centers = torch.linspace(grid_range[0], grid_range[1], hidden).repeat(input_dim, 1)
            self.centers = nn.Parameter(centers, requires_grad=False)
            self.log_width = nn.Parameter(torch.zeros(input_dim))
            self.coeff = nn.Parameter(torch.randn(input_dim * hidden, output_dim) * 0.02)
            self.bias = nn.Parameter(torch.zeros(output_dim))

        def forward(self, x):
            basis = []
            for j in range(x.shape[-1]):
                width = F.softplus(self.log_width[j]) + 1e-4
                basis.append(torch.exp(-0.5 * ((x[:, j:j+1] - self.centers[j:j+1]) / width) ** 2))
            return torch.cat(basis, dim=1) @ self.coeff + self.bias
    model = GaussianRBF()
    return model, _meta(model, "Gaussian RBF", hidden=hidden, centers="fixed uniform grid")


def build_pykan(input_dim, output_dim, hidden=8, grid=5, k=3, device="cpu", seed=1):
    """Official PyKAN baseline (pykan 0.2.8 in the experiment environment)."""
    from kan import KAN
    model = KAN(width=[input_dim, hidden, output_dim], grid=grid, k=k,
                base_fun="silu", symbolic_enabled=False, save_act=False,
                auto_save=False, device=device, seed=seed)
    return model, _meta(model, "KAN", hidden=hidden, grid=grid, spline_order=k,
                        implementation="pykan", grid_adaptation="disabled",
                        symbolic_enabled=False)


def build_cofrnet(input_dim, output_dim, depth=4, degree=3):
    """Build the repository's local nested control, not the Puri et al. code."""
    sys.path.insert(0, str(ROOT / "submit_codebase" / "classification_runner" / "code"))
    from cfnet import CFNet_Standard
    model = CFNet_Standard(input_dim, output_dim, depth=depth, poly_degree=degree)
    return model, _meta(
        model, "CoFrNet-Standard", depth=depth, degree=degree,
        implementation="local projected-polynomial nested CF control",
        relation_to_puri_2021="conceptual recursion only; not an official reproduction",
        terminal_denominator="abs(term)+1", recursive_scale="softplus(beta)",
    )


class RationalActivation(nn.Module):
    """Trainable rational activation, used as a controlled rational-NN baseline."""
    def __init__(self, degree=3, epsilon=1e-3):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(degree + 1))
        self.q = nn.Parameter(torch.zeros(degree + 1))
        self.p.data[1] = 1.0
        self.epsilon = epsilon
        self.degree = degree

    def forward(self, x):
        powers = [torch.ones_like(x)]
        for _ in range(self.degree):
            powers.append(powers[-1] * x)
        p = sum(a * b for a, b in zip(self.p, powers))
        q = sum(a * b for a, b in zip(self.q, powers))
        return p / (1.0 + q.abs() + self.epsilon)


def build_rational_nn(input_dim, output_dim, hidden=32, depth=2, degree=3):
    layers, d = [], input_dim
    for _ in range(depth):
        layers.extend([nn.Linear(d, hidden), RationalActivation(degree)])
        d = hidden
    layers.append(nn.Linear(d, output_dim))
    model = nn.Sequential(*layers)
    return model, _meta(model, "Rational activation NN", hidden=hidden, depth=depth,
                        degree=degree, denominator="1+abs(Q(x))+epsilon")


class _SineLayer(nn.Module):
    """SIREN layer with the principled initialization from Sitzmann et al."""
    def __init__(self, in_features, out_features, omega_0=30.0, first=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0
        self.first = first
        with torch.no_grad():
            bound = (1.0 / in_features) if first else ((6.0 / in_features) ** 0.5 / omega_0)
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


def build_siren(input_dim, output_dim, hidden=32, depth=2, omega=30.0):
    layers = [_SineLayer(input_dim, hidden, omega_0=omega, first=True)]
    for _ in range(max(0, depth - 1)):
        layers.append(_SineLayer(hidden, hidden, omega_0=omega, first=False))
    final = nn.Linear(hidden, output_dim)
    with torch.no_grad():
        bound = (6.0 / hidden) ** 0.5 / omega
        final.weight.uniform_(-bound, bound)
        final.bias.uniform_(-bound, bound)
    layers.append(final)
    model = nn.Sequential(*layers)
    return model, _meta(model, "SIREN", hidden=hidden, depth=depth, omega=omega,
                        initialization="SIREN principled initialization")

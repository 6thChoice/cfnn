"""Model registry and frozen architecture grids for confirmatory experiments."""
from __future__ import annotations

import json
import math
from itertools import product

import torch
from torch import nn

from budget_matching import Candidate
from fair_baselines import (
    build_cofrnet,
    build_fourier_mlp,
    build_gaussian_rbf,
    build_mlp,
    build_pykan,
    build_rational_nn,
    build_siren,
    parameter_count,
)


FAMILIES = (
    "CFNN",
    "PARN",
    "MLP",
    "Fourier-feature MLP",
    "Gaussian RBF",
    "KAN",
    "Local nested CF control",
    "Rational activation NN",
    "SIREN",
)

PARN_VARIANTS = {
    "CFNN": {},
    "PARN": {},
    "PARN-shared-PQ": {"shared_projection": True},
    "PARN-no-skip": {"skip": False},
    "PARN-epsilon-0.01": {"epsilon": 0.01},
    "PARN-epsilon-0.03": {"epsilon": 0.03},
    "PARN-epsilon-0.3": {"epsilon": 0.3},
    "PARN-abs-denominator": {"denominator": "absolute"},
}


class VectorizedPARN(nn.Module):
    """Batched implementation of the reference additive rational units."""
    def __init__(self, input_dim: int, output_dim: int, n_units: int, degree: int,
                 *, shared_projection: bool = False, epsilon: float = 0.1,
                 squared_denominator: bool = True, skip: bool = True,
                 numerator_init_std: float = 0.05,
                 denominator_init_std: float = 0.05):
        super().__init__()
        self.linear_skip = nn.Linear(input_dim, output_dim) if skip else None
        self.p_weight = nn.Parameter(torch.empty(n_units, output_dim, input_dim))
        self.p_bias = nn.Parameter(torch.empty(n_units, output_dim))
        self.p_coeffs = nn.Parameter(
            torch.randn(n_units, output_dim, degree + 1) * numerator_init_std
        )
        if shared_projection:
            self.q_weight = self.p_weight
            self.q_bias = self.p_bias
        else:
            self.q_weight = nn.Parameter(torch.empty(n_units, output_dim, input_dim))
            self.q_bias = nn.Parameter(torch.empty(n_units, output_dim))
        self.q_coeffs = nn.Parameter(
            torch.randn(n_units, output_dim, degree + 1) * denominator_init_std
        )
        self.degree = degree
        self.epsilon = float(epsilon)
        self.squared_denominator = squared_denominator
        self.output_dim = output_dim
        self.numerator_init_std = float(numerator_init_std)
        self.denominator_init_std = float(denominator_init_std)
        self._reset_projection_parameters(input_dim, shared_projection)

    def _reset_projection_parameters(self, input_dim: int, shared_projection: bool) -> None:
        nn.init.kaiming_uniform_(self.p_weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(input_dim)
        nn.init.uniform_(self.p_bias, -bound, bound)
        if not shared_projection:
            nn.init.kaiming_uniform_(self.q_weight, a=math.sqrt(5))
            nn.init.uniform_(self.q_bias, -bound, bound)

    def _poly(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
              coefficients: torch.Tensor) -> torch.Tensor:
        projection = torch.tanh(torch.einsum("bd,uod->buo", x, weight) + bias.unsqueeze(0))
        powers = [torch.ones_like(projection)]
        for _ in range(self.degree):
            powers.append(powers[-1] * projection)
        return torch.sum(torch.stack(powers, dim=-1) * coefficients.unsqueeze(0), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numerator = self._poly(x, self.p_weight, self.p_bias, self.p_coeffs)
        denominator_base = self._poly(x, self.q_weight, self.q_bias, self.q_coeffs)
        denominator = denominator_base.square() if self.squared_denominator else denominator_base.abs()
        rational_sum = torch.sum(numerator / (denominator + self.epsilon), dim=1)
        if self.linear_skip is not None:
            rational_sum = rational_sum + self.linear_skip(x)
        return rational_sum


class VectorizedCoFrNet(nn.Module):
    """Vectorized local nested control; not an official Puri et al. reproduction."""
    def __init__(self, input_dim: int, output_dim: int, depth: int, degree: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(depth, output_dim, input_dim))
        self.bias = nn.Parameter(torch.empty(depth, output_dim))
        self.coefficients = nn.Parameter(torch.randn(depth, output_dim, degree + 1) * 0.05)
        self.raw_betas = nn.Parameter(torch.ones(max(0, depth - 1)))
        self.depth = depth
        self.degree = degree
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(input_dim)
        nn.init.uniform_(self.bias, -bound, bound)

    def _terms(self, x: torch.Tensor) -> torch.Tensor:
        projection = torch.tanh(torch.einsum("bd,tod->bto", x, self.weight) + self.bias.unsqueeze(0))
        powers = [torch.ones_like(projection)]
        for _ in range(self.degree):
            powers.append(powers[-1] * projection)
        return torch.sum(
            torch.stack(powers, dim=-1) * self.coefficients.unsqueeze(0), dim=-1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        terms = self._terms(x)
        output = torch.abs(terms[:, -1]) + 1.0
        for index in range(self.depth - 2, -1, -1):
            beta = torch.nn.functional.softplus(self.raw_betas[index])
            output = terms[:, index] + beta / (output + 1e-8)
        return output


def build_model(family: str, input_dim: int, output_dim: int, config: dict, seed: int):
    if family in PARN_VARIANTS:
        config = {**config, **PARN_VARIANTS[family]}
        initialization_mode = config.get("initialization_mode", "fixed")
        if initialization_mode == "fixed":
            numerator_init_std = 0.05
        elif initialization_mode == "numerator_scaled":
            numerator_init_std = 0.05 / math.sqrt(config["units"])
        else:
            raise ValueError(f"unknown CFNN initialization mode: {initialization_mode}")
        denominator_init_std = 0.05
        model = VectorizedPARN(
            input_dim=input_dim,
            output_dim=output_dim,
            n_units=config["units"],
            degree=config["degree"],
            shared_projection=config.get("shared_projection", False),
            epsilon=config.get("epsilon", 0.1),
            squared_denominator=config.get("denominator", "squared") == "squared",
            skip=config.get("skip", True),
            numerator_init_std=numerator_init_std,
            denominator_init_std=denominator_init_std,
        )
        return model, {
            "family": family,
            "actual_parameters": parameter_count(model),
            "initialization_mode": initialization_mode,
            "numerator_init_std": numerator_init_std,
            "denominator_init_std": denominator_init_std,
            **config,
            "implementation": "projected additive rational network",
        }
    if family == "MLP":
        return build_mlp(input_dim, output_dim, config["width"], config["depth"], config["activation"])
    if family == "Fourier-feature MLP":
        return build_fourier_mlp(
            input_dim, output_dim, config["frequencies"], config["width"],
            config["depth"], config["scale"], config["activation"],
        )
    if family == "Gaussian RBF":
        return build_gaussian_rbf(input_dim, output_dim, config["basis"])
    if family == "KAN":
        return build_pykan(
            input_dim, output_dim, config["width"], config["grid"], config["k"],
            device="cpu", seed=seed,
        )
    if family == "Local nested CF control":
        model = VectorizedCoFrNet(input_dim, output_dim, config["depth"], config["degree"])
        return model, {
            "family": family,
            "actual_parameters": parameter_count(model),
            **config,
            "implementation": "vectorized local projected-polynomial nested CF control",
            "relation_to_puri_2021": "conceptual recursion only; not an official reproduction",
            "terminal_denominator": "abs(term)+1",
            "recursive_scale": "softplus(beta)",
        }
    if family == "Rational activation NN":
        return build_rational_nn(
            input_dim, output_dim, config["width"], config["depth"], config["degree"]
        )
    if family == "SIREN":
        return build_siren(
            input_dim, output_dim, config["width"], config["depth"], config["omega"]
        )
    raise ValueError(f"unknown family: {family}")


def architecture_grid(family: str, protocol_grid: str | None = None):
    if protocol_grid not in {None, "v2"}:
        raise ValueError(f"unknown protocol grid: {protocol_grid}")
    if protocol_grid == "v2" and family != "CFNN":
        raise ValueError("protocol_grid='v2' is only defined for CFNN")
    if family == "CFNN" and protocol_grid == "v2":
        return (
            {"units": units, "degree": degree, "epsilon": epsilon,
             "denominator": "squared", "shared_projection": False, "skip": True,
             "initialization_mode": initialization_mode,
             "numerator_init_std": (
                 0.05 / math.sqrt(units)
                 if initialization_mode == "numerator_scaled" else 0.05
             ),
             "denominator_init_std": 0.05}
            for units, degree, epsilon, initialization_mode in product(
                range(1, 161), (1, 2, 3, 4, 5, 8),
                (0.05, 0.1, 0.2, 0.3, 0.5),
                ("fixed", "numerator_scaled"),
            )
        )
    if family in {"CFNN", "PARN"}:
        return (
            {"units": units, "degree": degree, "epsilon": epsilon,
             "denominator": "squared", "shared_projection": False, "skip": True}
            for units, degree, epsilon in product(
                range(1, 161), (1, 2, 3, 4, 5, 8), (0.03, 0.1, 0.3)
            )
        )
    if family in PARN_VARIANTS:
        overrides = PARN_VARIANTS[family]
        return (
            {"units": units, "degree": degree, "epsilon": 0.1,
             "denominator": "squared", "shared_projection": False, "skip": True,
             **overrides}
            for units, degree in product(range(1, 161), (1, 2, 3, 4, 5, 8))
        )
    if family == "MLP":
        return (
            {"width": width, "depth": depth, "activation": activation}
            for width, depth, activation in product(range(2, 513), (1, 2, 3), ("tanh", "gelu"))
        )
    if family == "Fourier-feature MLP":
        return (
            {"frequencies": frequencies, "width": width, "depth": depth,
             "scale": scale, "activation": "gelu"}
            for frequencies, width, depth, scale in product(
                (2, 4, 8, 16, 32, 64), range(2, 129), (1, 2), (2.0, 4.0, 8.0)
            )
        )
    if family == "Gaussian RBF":
        return ({"basis": basis} for basis in range(2, 1101))
    if family == "KAN":
        return (
            {"width": width, "grid": grid, "k": order}
            for width, grid, order in product(range(1, 129), range(2, 11), range(2, 6))
        )
    if family == "Local nested CF control":
        return (
            {"depth": depth, "degree": degree}
            for depth, degree in product(range(1, 257), (1, 2, 3, 4, 5, 8))
        )
    if family == "Rational activation NN":
        return (
            {"width": width, "depth": depth, "degree": degree}
            for width, depth, degree in product(range(2, 513), (1, 2, 3), (1, 2, 3, 5))
        )
    if family == "SIREN":
        return (
            {"width": width, "depth": depth, "omega": omega}
            for width, depth, omega in product(range(2, 513), (1, 2, 3), (10.0, 30.0, 60.0))
        )
    raise ValueError(f"unknown family: {family}")


def enumerate_candidates(
    family: str, input_dim: int, output_dim: int, protocol_grid: str | None = None
) -> list[Candidate]:
    candidates = []
    for config in architecture_grid(family, protocol_grid=protocol_grid):
        key = json.dumps(config, sort_keys=True, separators=(",", ":"))
        model, _ = build_model(family, input_dim, output_dim, config, seed=0)
        candidates.append(Candidate(key, parameter_count(model), config))
    return candidates

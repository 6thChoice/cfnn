"""Utilities for parameter-matched CFNN baseline experiments.

The key policy is that CFNN-family trainable parameter counts define the
comparison budget. Larger baselines may be kept as references, but must not be
labeled parameter-matched.
"""
from __future__ import annotations

import json
from typing import Iterable

import numpy as np


class NumpySafeEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types and common torch/Python types."""
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def count_trainable_params(model) -> int:
    """Return the number of trainable parameters in a PyTorch-style model."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def mlp_param_count(input_dim: int, hidden_dim: int, num_layers: int, output_dim: int = 1) -> int:
    """Parameter count for a fully connected MLP with `num_layers` hidden layers."""
    if input_dim <= 0 or hidden_dim <= 0 or num_layers <= 0 or output_dim <= 0:
        raise ValueError("input_dim, hidden_dim, num_layers, and output_dim must be positive")

    params = input_dim * hidden_dim + hidden_dim
    params += (num_layers - 1) * (hidden_dim * hidden_dim + hidden_dim)
    params += hidden_dim * output_dim + output_dim
    return int(params)


def find_mlp_config_for_budget(
    input_dim: int,
    target_params: int,
    output_dim: int = 1,
    depths: Iterable[int] = (1, 2, 3),
    max_hidden: int = 512,
) -> dict:
    """Find the MLP depth/width whose parameter count is closest to a target budget."""
    if target_params <= 0:
        raise ValueError("target_params must be positive")

    best = None
    for num_layers in depths:
        for hidden_dim in range(1, max_hidden + 1):
            actual = mlp_param_count(input_dim, hidden_dim, num_layers, output_dim)
            diff = abs(actual - target_params)
            candidate = {
                "num_layers": int(num_layers),
                "hidden_dim": int(hidden_dim),
                "actual_params": int(actual),
                "target_params": int(target_params),
                "param_ratio": round(actual / target_params, 6),
                "relative_error": round(diff / target_params, 6),
            }
            if best is None or candidate["relative_error"] < best["relative_error"]:
                best = candidate
    return best


def comparison_type(trainable_params: int, target_params: int, tolerance: float = 0.20) -> str:
    """Classify whether a baseline is parameter-matched to the target budget."""
    if target_params <= 0:
        raise ValueError("target_params must be positive")
    ratio = trainable_params / target_params
    if abs(ratio - 1.0) <= tolerance:
        return "parameter_matched"
    return "capacity_unmatched_reference"


def budget_metadata(model: str, trainable_params: int, target_params: int, tolerance: float = 0.20) -> dict:
    """Return standard metadata attached to every parameter-budgeted result row."""
    if target_params <= 0:
        raise ValueError("target_params must be positive")
    return {
        "model": model,
        "trainable_params": int(trainable_params),
        "target_params": int(target_params),
        "param_ratio": round(trainable_params / target_params, 6),
        "comparison_type": comparison_type(trainable_params, target_params, tolerance),
    }

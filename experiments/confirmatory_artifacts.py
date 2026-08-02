"""Frozen candidate manifests and reproducibility metadata."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch

from budget_matching import Candidate, within_budget


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _structural_signature(config: dict) -> str:
    capacity_keys = {"units", "width", "basis"}
    structural = {key: value for key, value in config.items() if key not in capacity_keys}
    return json.dumps(structural, sort_keys=True, separators=(",", ":"))


def _balanced_signature_order(signatures: Sequence[str]) -> list[str]:
    """Order signatures to cover each structural level before repeats dominate."""
    remaining = sorted(signatures)
    selected = []
    level_counts = defaultdict(int)
    decoded = {signature: json.loads(signature) for signature in remaining}
    while remaining:
        scores = {}
        for signature in remaining:
            levels = [
                (key, json.dumps(value, sort_keys=True, separators=(",", ":")))
                for key, value in decoded[signature].items()
            ]
            unseen = sum(level_counts[level] == 0 for level in levels)
            balance = sum(1.0 / (1.0 + level_counts[level]) for level in levels)
            scores[signature] = (unseen, balance)
        best_score = max(scores.values())
        chosen = next(signature for signature in remaining if scores[signature] == best_score)
        selected.append(chosen)
        for key, value in decoded[chosen].items():
            level = (key, json.dumps(value, sort_keys=True, separators=(",", ":")))
            level_counts[level] += 1
        remaining.remove(chosen)
    return selected


def freeze_budget_grid(
    candidates: Sequence[Candidate],
    budgets: Sequence[int],
    *,
    tolerance: float,
    minimum_per_budget: int = 1,
    maximum_per_budget: int,
) -> dict[str, list[dict]]:
    """Deterministically retain a diverse, bounded set before any training."""
    frozen = {}
    for budget in budgets:
        eligible = [c for c in candidates if within_budget(c.actual_parameters, budget, tolerance)]
        groups = defaultdict(list)
        for candidate in eligible:
            groups[_structural_signature(candidate.config)].append(candidate)
        for values in groups.values():
            values.sort(key=lambda c: (abs(c.actual_parameters - budget), c.key))
        selected = []
        signatures = _balanced_signature_order(groups)
        index = 0
        while len(selected) < maximum_per_budget:
            added = False
            for signature in signatures:
                if index < len(groups[signature]):
                    selected.append(groups[signature][index])
                    added = True
                    if len(selected) == maximum_per_budget:
                        break
            if not added:
                break
            index += 1
        records = [
            {
                "candidate_key": candidate.key,
                "actual_parameters": candidate.actual_parameters,
                "budget_relative_error": abs(candidate.actual_parameters - budget) / budget,
                "config": candidate.config,
            }
            for candidate in selected
        ]
        frozen[str(budget)] = records if len(records) >= minimum_per_budget else []
    return frozen


def _git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True,
        capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_run_manifest(config: dict, command: Sequence[str], code_paths: Sequence[Path]) -> dict:
    files = []
    for path in sorted((Path(item).resolve() for item in code_paths), key=str):
        files.append({"path": str(path), "sha256": _sha256_bytes(path.read_bytes())})
    hardware = {"device": "cpu", "gpu_name": None, "gpu_memory_bytes": None}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        hardware = {
            "device": "cuda",
            "gpu_name": props.name,
            "gpu_memory_bytes": props.total_memory,
            "cuda_version": torch.version.cuda,
        }
    return {
        "config_sha256": _sha256_bytes(_canonical_json(config)),
        "command": list(command),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "reproducibility": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "hardware": hardware,
        "git_commit": _git_commit(Path.cwd()),
        "code_files": files,
    }

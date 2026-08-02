"""Exact realized-parameter budget matching utilities.

Nominal budgets are only labels. A candidate is eligible after construction
when its actual trainable parameter count falls inside the requested tolerance.
The assignment helper prevents the same architecture from being reused at
multiple budget anchors.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import permutations
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Candidate:
    key: str
    actual_parameters: int
    config: dict[str, Any]


def within_budget(actual: int, target: int, relative_tolerance: float = 0.05) -> bool:
    if actual <= 0 or target <= 0:
        return False
    return abs(actual - target) / target <= relative_tolerance


def assign_unique_candidates(
    candidates: Sequence[Candidate],
    budgets: Sequence[int],
    relative_tolerance: float = 0.05,
) -> dict[int, Candidate | None]:
    """Assign at most one candidate to each budget and each candidate once.

    The minimum-cost one-to-one assignment is selected, where cost is relative
    parameter mismatch. Missing budgets are returned as ``None`` rather than
    silently borrowing a neighboring architecture.
    """
    budgets = tuple(budgets)
    eligible = {
        b: [c for c in candidates if within_budget(c.actual_parameters, b, relative_tolerance)]
        for b in budgets
    }
    best = None
    # The grids used by the experiments are small. Exhaustive assignments are
    # preferable here because the matching rule is part of the evidence.
    def visit(i, used, chosen, cost):
        nonlocal best
        if i == len(budgets):
            score = (sum(c is None for c in chosen), cost)
            if best is None or score < best[0]:
                best = (score, tuple(chosen))
            return
        b = budgets[i]
        visit(i + 1, used, chosen + [None], cost)
        for c in eligible[b]:
            if c.key in used:
                continue
            visit(i + 1, used | {c.key}, chosen + [c], cost + abs(c.actual_parameters - b) / b)
    visit(0, set(), [], 0.0)
    assert best is not None
    return {b: c for b, c in zip(budgets, best[1])}


def candidate_record(key: str, model, config: dict[str, Any]) -> Candidate:
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return Candidate(key=key, actual_parameters=actual, config=config)


def serialize_assignment(assignment: dict[int, Candidate | None]) -> list[dict[str, Any]]:
    return [
        {"target_budget": b, "matched": c is not None,
         **({"candidate_key": c.key, "actual_parameters": c.actual_parameters,
             "budget_relative_error": abs(c.actual_parameters - b) / b,
             "config": c.config} if c else {})}
        for b, c in sorted(assignment.items())
    ]

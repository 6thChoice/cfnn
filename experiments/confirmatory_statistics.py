"""Robust summaries and paired inference for confirmatory records."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from itertools import product

import numpy as np


def _finite_metric(record: dict, metric: str) -> float | None:
    raw = record["test_metrics"].get(metric)
    if raw is None:
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def summarize_records(records: list[dict], *, max_steps: int | None = None) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        groups[(record["task"], record["family"], int(record["target_budget"]))].append(record)
    summaries = []
    for (task, family, budget), values in sorted(groups.items()):
        nrmse = [value for record in values if (value := _finite_metric(record, "nrmse")) is not None]
        r2 = [value for record in values if (value := _finite_metric(record, "r2")) is not None]
        numerical_failures = sum(
            record["status"].startswith("nonfinite") or _finite_metric(record, "nrmse") is None
            for record in values
        )
        underfit = sum(record["test_metrics"].get("status") == "underfit" for record in values)
        cap_hits = (
            sum(int(record["optimizer_steps"]) >= max_steps for record in values)
            if max_steps is not None else 0
        )
        summaries.append({
            "task": task,
            "family": family,
            "target_budget": budget,
            "actual_parameters_median": float(np.median([r["actual_parameters"] for r in values])),
            "n": len(values),
            "finite_n": len(nrmse),
            "nrmse_median": float(np.median(nrmse)) if nrmse else float("nan"),
            "nrmse_q1": round(float(np.quantile(nrmse, 0.25)), 12) if nrmse else float("nan"),
            "nrmse_q3": round(float(np.quantile(nrmse, 0.75)), 12) if nrmse else float("nan"),
            "r2_median": float(np.median(r2)) if r2 else float("nan"),
            "r2_q1": float(np.quantile(r2, 0.25)) if r2 else float("nan"),
            "r2_q3": float(np.quantile(r2, 0.75)) if r2 else float("nan"),
            "numerical_failure_rate": numerical_failures / len(values),
            "underfit_rate": underfit / len(values),
            "training_cap_rate": cap_hits / len(values) if max_steps is not None else None,
            "wall_seconds_median": float(np.median([r["training_wall_seconds"] for r in values])),
            "optimizer_steps_median": float(np.median([r["optimizer_steps"] for r in values])),
        })
    return summaries


def paired_hierarchical_bootstrap(
    records: list[dict],
    family_a: str,
    family_b: str,
    *,
    metric: str,
    draws: int = 10000,
    seed: int = 2027,
) -> dict:
    values = defaultdict(dict)
    for record in records:
        if record["family"] not in {family_a, family_b}:
            continue
        metric_value = _finite_metric(record, metric)
        if metric_value is not None:
            values[(record["data_seed"], record["init_seed"])][record["family"]] = metric_value
    paired = {
        key: item[family_a] - item[family_b]
        for key, item in values.items()
        if family_a in item and family_b in item
    }
    if not paired:
        raise ValueError("no finite paired observations")
    by_data = defaultdict(list)
    for (data_seed, _), difference in paired.items():
        by_data[data_seed].append(difference)
    data_seeds = sorted(by_data)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(draws):
        sampled_differences = []
        for sampled_seed in rng.choice(data_seeds, size=len(data_seeds), replace=True):
            within = by_data[int(sampled_seed)]
            sampled_differences.extend(rng.choice(within, size=len(within), replace=True))
        bootstrap.append(float(np.median(sampled_differences)))
    return {
        "family_a": family_a,
        "family_b": family_b,
        "metric": metric,
        "paired_n": len(paired),
        "data_instance_n": len(data_seeds),
        "median_difference": round(float(np.median(list(paired.values()))), 12),
        "ci_low": round(float(np.quantile(bootstrap, 0.025)), 12),
        "ci_high": round(float(np.quantile(bootstrap, 0.975)), 12),
        "bootstrap_draws": draws,
    }


def paired_instance_sign_flip_test(
    records: list[dict], family_a: str, family_b: str, *, metric: str
) -> dict:
    """Exact paired test using one mean effect per independently generated instance."""
    values = defaultdict(dict)
    for record in records:
        if record["family"] not in {family_a, family_b}:
            continue
        value = _finite_metric(record, metric)
        if value is not None:
            values[(record["data_seed"], record["init_seed"])][record["family"]] = value
    by_instance = defaultdict(list)
    for (data_seed, _), paired in values.items():
        if family_a in paired and family_b in paired:
            by_instance[data_seed].append(paired[family_a] - paired[family_b])
    effects = np.asarray([
        np.mean(by_instance[data_seed]) for data_seed in sorted(by_instance) if by_instance[data_seed]
    ], dtype=np.float64)
    if len(effects) == 0:
        raise ValueError("no paired data-instance effects")
    observed = abs(float(np.mean(effects)))
    permutations = [
        abs(float(np.mean(effects * np.asarray(signs))))
        for signs in product((-1.0, 1.0), repeat=len(effects))
    ]
    p_value = sum(value >= observed - 1e-15 for value in permutations) / len(permutations)
    return {
        "instance_n": len(effects),
        "mean_instance_difference": float(np.mean(effects)),
        "p_value": float(p_value),
        "test": "exact paired sign-flip on instance-mean effects",
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _macro_token(value: str) -> str:
    if value.isupper() and value.isalpha():
        return value
    digit_words = {
        "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
        "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
    }
    parts = []
    for part in re.split(r"[^A-Za-z0-9]+", value):
        capitalized = part.capitalize()
        parts.append("".join(digit_words.get(character, character) for character in capitalized))
    return re.sub(r"[^A-Za-z]", "", "".join(parts))


def _budget_token(budget: int) -> str:
    known = {128: "OneTwentyEight", 256: "TwoFiftySix", 512: "FiveTwelve", 1024: "OneZeroTwoFour"}
    return known.get(int(budget), "B" + str(int(budget)))


def _step_token(step: int) -> str:
    known = {20: "Twenty", 80: "Eighty", 200: "TwoHundred", 500: "FiveHundred"}
    return known.get(int(step), _macro_token(str(int(step))))


def render_latex_macros(
    summaries: list[dict], *, comparisons: list[dict] | None = None,
    profiles: list[dict] | None = None, costs: list[dict] | None = None,
    namespace: str = "",
) -> str:
    lines = ["% Generated from confirmatory evaluation records; do not edit manually."]
    namespace_token = _macro_token(namespace) if namespace else ""
    for summary in sorted(summaries, key=lambda item: (item["task"], item["family"], item["target_budget"])):
        prefix = (
            namespace_token + _macro_token(summary["task"]) + _macro_token(summary["family"])
            + "B" + _budget_token(summary["target_budget"])
        )
        lines.append(f"\\newcommand{{\\{prefix}NRMSE}}{{{summary['nrmse_median']:.3f}}}")
        lines.append(f"\\newcommand{{\\{prefix}NRMSEQOne}}{{{summary['nrmse_q1']:.3f}}}")
        lines.append(f"\\newcommand{{\\{prefix}NRMSEQThree}}{{{summary['nrmse_q3']:.3f}}}")
        lines.append(
            f"\\newcommand{{\\{prefix}Parameters}}"
            f"{{{int(round(summary['actual_parameters_median']))}}}"
        )
        lines.append(f"\\newcommand{{\\{prefix}N}}{{{summary['n']}}}")
        lines.append(f"\\newcommand{{\\{prefix}FailureRate}}{{{summary['numerical_failure_rate']:.3f}}}")
        lines.append(f"\\newcommand{{\\{prefix}UnderfitRate}}{{{summary['underfit_rate']:.3f}}}")
        if summary.get("training_cap_rate") is not None:
            lines.append(
                f"\\newcommand{{\\{prefix}TrainingCapRate}}"
                f"{{{summary['training_cap_rate']:.3f}}}"
            )
    for comparison in sorted(
        comparisons or [],
        key=lambda item: (
            item["task"], item["family_a"], item["family_b"], item["target_budget"]
        ),
    ):
        prefix = (
            namespace_token + _macro_token(comparison["task"])
            + _macro_token(comparison["family_a"])
            + "Vs"
            + _macro_token(comparison["family_b"])
            + "B"
            + _budget_token(comparison["target_budget"])
        )
        lines.append(
            f"\\newcommand{{\\{prefix}NRMSEDifference}}"
            f"{{\\ensuremath{{{comparison['median_difference']:.3f}}}}}"
        )
        lines.append(
            f"\\newcommand{{\\{prefix}NRMSECI}}"
            f"{{\\ensuremath{{[{comparison['ci_low']:.3f}, {comparison['ci_high']:.3f}]}}}}"
        )
        lines.append(f"\\newcommand{{\\{prefix}PairedN}}{{{comparison['paired_n']}}}")
        lines.append(f"\\newcommand{{\\{prefix}RawP}}{{{comparison['p_value']:.4f}}}")
        if comparison.get("p_holm") is not None:
            lines.append(
                f"\\newcommand{{\\{prefix}HolmP}}{{{comparison['p_holm']:.4f}}}"
            )
    for profile in sorted(
        profiles or [],
        key=lambda item: (item["task"], item["family"], item["target_budget"], item["step"]),
    ):
        prefix = (
            namespace_token + _macro_token(profile["task"]) + _macro_token(profile["family"])
            + "B" + _budget_token(profile["target_budget"])
            + "Step" + _step_token(profile["step"])
        )
        lines.append(f"\\newcommand{{\\{prefix}NRMSE}}{{{profile['nrmse_median']:.3f}}}")
        lines.append(f"\\newcommand{{\\{prefix}N}}{{{profile['n']}}}")
        lines.append(
            f"\\newcommand{{\\{prefix}Coverage}}{{{profile['checkpoint_coverage']:.3f}}}"
        )
    if costs:
        stage_totals: dict[str, list[float]] = {}
        for row in costs:
            stage = str(row["stage"])
            totals = stage_totals.setdefault(stage, [0.0, 0.0])
            totals[0] += int(row["run_count"])
            totals[1] += float(row["process_hours"])
        total_runs = 0
        total_hours = 0.0
        for stage, (run_count, process_hours) in sorted(stage_totals.items()):
            prefix = namespace_token + "Confirmatory" + _macro_token(stage)
            lines.append(f"\\newcommand{{\\{prefix}Runs}}{{{int(run_count)}}}")
            lines.append(
                f"\\newcommand{{\\{prefix}ProcessHours}}{{{process_hours:.2f}}}"
            )
            total_runs += int(run_count)
            total_hours += process_hours
        prefix = namespace_token + "ConfirmatoryTotal"
        lines.append(f"\\newcommand{{\\{prefix}Runs}}{{{total_runs}}}")
        lines.append(f"\\newcommand{{\\{prefix}ProcessHours}}{{{total_hours:.2f}}}")
    return "\n".join(lines) + "\n"

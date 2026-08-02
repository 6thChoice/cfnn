"""Validate confirmatory records and generate all paper-facing artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from jsonschema import Draft202012Validator

from confirmatory_statistics import (
    holm_adjust,
    paired_hierarchical_bootstrap,
    paired_instance_sign_flip_test,
    render_latex_macros,
    summarize_records,
)


ROOT = Path(__file__).resolve().parent

TASK_LABELS = {
    "pole_sharp": "sharp regularized pole",
    "pole_broad": "broad regularized pole",
    "lorentzian_two": "two-component Lorentzian",
    "cusp": "cusp",
    "compact_bump": "compact-support bump",
    "asymmetric_peak": "asymmetric peak",
    "manifold_bump": "non-ridge manifold bump",
    "nmr_ethanol": "semi-synthetic ethanol NMR",
    "nmr_caffeine": "semi-synthetic caffeine NMR",
    "nmr_strychnine": "semi-synthetic strychnine NMR",
    "energy": "Energy Efficiency",
}

FAMILY_LABELS = {"PARN": "CFNN"}


def _family_label(family: str) -> str:
    """Map immutable result-family keys to manuscript-facing names."""
    if family == "PARN":
        return "CFNN"
    if family.startswith("PARN-"):
        return "CFNN-" + family.removeprefix("PARN-")
    return FAMILY_LABELS.get(family, family)


def _ablation_family_label(family: str) -> str:
    return "CFNN (offset tuned)" if family == "PARN" else _family_label(family)


def load_records(input_root: Path) -> list[dict]:
    records = []
    for path in sorted(input_root.glob("**/evaluation/**/*.jsonl")):
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return records


def compute_cost_rows(input_root: Path) -> list[dict]:
    """Aggregate recorded process time without calling it elapsed wall-clock time."""
    grouped = defaultdict(list)
    stage_directories = {
        "tuning_screen": "screening",
        "tuning_final": "tuning",
        "evaluation": "evaluation",
    }
    for directory, stage in stage_directories.items():
        for path in sorted(input_root.glob(f"**/{directory}/**/*.jsonl")):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (
                    stage, record["task"], record["family"],
                    int(record["target_budget"]),
                )
                grouped[key].append(float(record["training_wall_seconds"]))
    return [
        {
            "stage": stage,
            "task": task,
            "family": family,
            "target_budget": budget,
            "run_count": len(seconds),
            "process_seconds": float(sum(seconds)),
            "process_hours": float(sum(seconds) / 3600.0),
        }
        for (stage, task, family, budget), seconds in sorted(grouped.items())
    ]


def load_selection_rows(input_root: Path) -> list[dict]:
    """Flatten validation-selected configurations for direct protocol auditing."""
    rows = []
    for path in sorted(input_root.glob("**/selections/**/*.json")):
        selection = json.loads(path.read_text())
        candidate = selection.get("candidate", {})
        optimizer = selection.get("optimizer", {})
        rows.append({
            "task": selection["task"],
            "family": selection["family"],
            "target_budget": int(selection["target_budget"]),
            "status": selection["status"],
            "candidate_key": candidate.get("candidate_key"),
            "actual_parameters": candidate.get("actual_parameters"),
            "candidate_config": (
                json.dumps(candidate["config"], sort_keys=True, separators=(",", ":"))
                if "config" in candidate else None
            ),
            "optimizer_key": optimizer.get("key"),
            "learning_rate": optimizer.get("lr"),
            "weight_decay": optimizer.get("weight_decay"),
            "selected_epoch": selection.get("selected_epoch"),
            "median_validation_nrmse": selection.get("median_validation_nrmse"),
            "screened_candidate_count": selection.get("screened_candidate_count"),
            "finalist_count": selection.get("finalist_count"),
            "optimizer_count": selection.get("optimizer_count"),
            "tuning_seed_count": selection.get("tuning_seed_count"),
            "source_path": str(path.relative_to(input_root)),
        })
    return rows


def load_candidate_inventory(input_root: Path) -> dict:
    paths = sorted(input_root.glob("**/candidate_manifest.json"))
    if not paths:
        raise ValueError(f"no candidate manifest found under {input_root}")
    inventories = [json.loads(path.read_text()) for path in paths]
    canonical = {json.dumps(value, sort_keys=True, separators=(",", ":")) for value in inventories}
    if len(canonical) != 1:
        raise ValueError("shards contain non-identical candidate manifests")
    return inventories[0]


def validate_completeness(
    records: list[dict], config: dict, inventory: dict, *, task_set: str
) -> None:
    if task_set == "primary":
        tasks = config["primary_tasks"]
    elif task_set == "supplementary":
        tasks = config["supplementary_tasks"]
    elif task_set == "all":
        tasks = config["primary_tasks"] + config["supplementary_tasks"]
    else:
        raise ValueError(f"unknown task set: {task_set}")
    by_cell = defaultdict(list)
    for record in records:
        by_cell[(record["task"], record["family"], int(record["target_budget"]))].append(record)
    errors = []
    for task in tasks:
        pairs = [
            (int(data_seed), int(init_seed))
            for data_seed in config["evaluation_data_seeds"]
            for init_seed in config["evaluation_init_seeds"]
        ]
        if task in config["supplementary_tasks"]:
            pairs = pairs[: int(config["auxiliary_pairs"])]
        expected_pairs = set(pairs)
        for family in config["families"]:
            for budget in config["budgets"]:
                candidates = inventory["tasks"][task][family][str(budget)]
                expected = expected_pairs if candidates else set()
                values = by_cell.get((task, family, int(budget)), [])
                actual = {(int(row["data_seed"]), int(row["init_seed"])) for row in values}
                if len(actual) != len(values):
                    errors.append(f"{task}/{family}/{budget}: duplicate evaluation pairs")
                if actual != expected:
                    errors.append(
                        f"{task}/{family}/{budget}: expected {len(expected)} records, "
                        f"found {len(actual)}"
                    )
    allowed = {
        (task, family, int(budget))
        for task in tasks for family in config["families"] for budget in config["budgets"]
    }
    extras = sorted(set(by_cell) - allowed)
    if extras:
        errors.append(f"records outside requested task set: {extras[:5]}")
    if errors:
        raise ValueError("incomplete confirmatory matrix:\n" + "\n".join(errors[:30]))


def _file_fingerprint(path: Path, *, relative_to: Path) -> dict:
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(relative_to.resolve()).as_posix()
    except ValueError:
        display_path = path.name
    return {"path": display_path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def build_artifact_manifest(
    input_root: Path,
    schema_path: Path,
    *,
    config_path: Path,
    record_count: int,
    command: list[str],
) -> dict:
    raw_paths = sorted(input_root.glob("**/evaluation/**/*.jsonl"))
    supporting = sorted(
        path for path in input_root.rglob("*.json")
        if path.name in {"run_manifest.json", "candidate_manifest.json"}
        or "selections" in path.parts
    )
    supporting.extend(sorted(
        path for path in input_root.rglob("*.jsonl")
        if "tuning_screen" in path.parts or "tuning_final" in path.parts
    ))
    generator_paths = [
        Path(__file__), ROOT / "confirmatory_statistics.py", schema_path,
        config_path, ROOT / "requirements-server-freeze.txt",
    ]
    return {
        "command": list(command),
        "record_count": int(record_count),
        "raw_inputs": [
            _file_fingerprint(path, relative_to=input_root) for path in raw_paths
        ],
        "supporting_inputs": [
            _file_fingerprint(path, relative_to=input_root) for path in supporting
        ],
        "generator_sources": [
            _file_fingerprint(path, relative_to=ROOT.parent) for path in generator_paths
        ],
    }


def validate_records(records: list[dict], schema_path: Path) -> None:
    validator = Draft202012Validator(json.loads(schema_path.read_text()))
    errors = []
    for index, record in enumerate(records):
        for error in validator.iter_errors(record):
            errors.append(f"record {index} at {list(error.path)}: {error.message}")
    if errors:
        raise ValueError("schema validation failed:\n" + "\n".join(errors[:20]))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_comparisons(records: list[dict], primary_family: str = "PARN") -> list[dict]:
    comparisons = []
    groups = {}
    for record in records:
        groups.setdefault((record["task"], record["target_budget"]), []).append(record)
    for (task, budget), values in sorted(groups.items()):
        families = sorted({record["family"] for record in values} - {primary_family})
        for family in families:
            try:
                comparison = paired_hierarchical_bootstrap(
                    values, primary_family, family, metric="nrmse", draws=10000,
                    seed=2027 + int(budget),
                )
                significance = paired_instance_sign_flip_test(
                    values, primary_family, family, metric="nrmse"
                )
            except ValueError:
                continue
            comparisons.append({
                "task": task, "target_budget": budget, **comparison,
                "p_value": significance["p_value"],
                "instance_n_for_test": significance["instance_n"],
                "p_holm": None,
            })
    primary_indices = [
        index for index, comparison in enumerate(comparisons)
        if comparison["target_budget"] == 512
    ]
    adjusted = holm_adjust([comparisons[index]["p_value"] for index in primary_indices])
    for index, value in zip(primary_indices, adjusted):
        comparisons[index]["p_holm"] = value
    return comparisons


def render_primary_latex_table(
    summaries: list[dict],
    primary_budget: int = 512,
    families: tuple[str, ...] = ("PARN", "MLP", "KAN", "SIREN"),
) -> str:
    """Render the prespecified-budget table directly from validated summaries."""
    rows = [row for row in summaries if int(row["target_budget"]) == primary_budget]
    by_key = {(row["task"], row["family"]): row for row in rows}
    lines = [
        "% Generated from confirmatory evaluation records; Do not edit manually.",
        "\\begin{tabular}{l" + "c" * len(families) + "}",
        "\\toprule",
        "Task & " + " & ".join(_family_label(family) for family in families) + " \\\\",
        "\\midrule",
    ]
    for task_key in sorted({row["task"] for row in rows}):
        cells = []
        for family in families:
            row = by_key.get((task_key, family))
            if row is None:
                cells.append("--")
                continue
            nrmse = f"{row['nrmse_median']:.3f} [{row['nrmse_q1']:.3f}, {row['nrmse_q3']:.3f}]"
            details = (
                f"$p={int(round(row['actual_parameters_median']))}$, $n={row['n']}$, "
                f"$f={100.0 * row['numerical_failure_rate']:.1f}\\%$, "
                f"$c={100.0 * row['training_cap_rate']:.1f}\\%$"
            )
            cells.append(f"\\shortstack{{{nrmse}\\\\{{\\scriptsize {details}}}}}")
        label = TASK_LABELS.get(task_key, task_key.replace("_", " "))
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def render_ablation_latex_table(summaries: list[dict], primary_budget: int = 512) -> str:
    rows = [
        row for row in summaries
        if int(row["target_budget"]) == primary_budget
        and (row["family"] == "PARN" or row["family"].startswith("PARN-"))
    ]
    families = sorted({row["family"] for row in rows}, key=lambda value: (value != "PARN", value))
    tasks = sorted({row["task"] for row in rows})
    if len(families) < 2 or not tasks:
        return ""
    by_key = {(row["family"], row["task"]): row for row in rows}
    lines = [
        "% Generated from confirmatory ablation records; do not edit manually.",
        "\\begin{tabular}{l" + "c" * len(tasks) + "}",
        "\\toprule",
        "Variant & " + " & ".join(TASK_LABELS.get(task, task.replace("_", " ")) for task in tasks) + " \\\\",
        "\\midrule",
    ]
    for family in families:
        cells = []
        for task in tasks:
            row = by_key.get((family, task))
            cells.append(
                "--" if row is None else
                f"{row['nrmse_median']:.3f} [{row['nrmse_q1']:.3f}, {row['nrmse_q3']:.3f}]"
            )
        lines.append(_ablation_family_label(family) + " & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def render_resource_latex_table(records: list[dict], primary_budget: int = 512) -> str:
    """Summarize hardware-dependent evaluation costs at the primary budget."""
    grouped = defaultdict(list)
    for record in records:
        if int(record["target_budget"]) == primary_budget:
            grouped[record["family"]].append(record)
    lines = [
        "% Generated from confirmatory evaluation records; do not edit manually.",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Family & $n$ & wall time (s) & optimizer steps & peak memory (MiB) \\\\",
        "\\midrule",
    ]
    for family in sorted(grouped):
        values = grouped[family]
        wall = np.asarray([float(row["training_wall_seconds"]) for row in values])
        steps = np.asarray([float(row["optimizer_steps"]) for row in values])
        memory = np.asarray([float(row["peak_memory_bytes"]) / (1024 ** 2) for row in values])
        interval = lambda array: (
            f"{np.median(array):.1f} "
            f"[{np.quantile(array, 0.25):.1f}, {np.quantile(array, 0.75):.1f}]"
        )
        lines.append(
            f"{_family_label(family)} & {len(values)} & {interval(wall)} & {interval(steps)} & "
            f"{interval(memory)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def step_profile_rows(records: list[dict], primary_budget: int = 512) -> list[dict]:
    """Aggregate only synchronized, prespecified optimizer-step checkpoints."""
    grouped = defaultdict(list)
    wall = defaultdict(list)
    cell_sizes = defaultdict(int)
    for record in records:
        if int(record["target_budget"]) != primary_budget:
            continue
        cell_sizes[(record["task"], record["family"])] += 1
        for point in record.get("curve", []):
            if point.get("label") == "selected_checkpoint" or point.get("test_nrmse") is None:
                continue
            value = float(point["test_nrmse"])
            if not np.isfinite(value):
                continue
            key = (record["task"], record["family"], int(point["step"]))
            grouped[key].append(value)
            wall[key].append(float(point["wall_seconds"]))
    rows = []
    for (task, family, step), values in sorted(grouped.items()):
        rows.append({
            "task": task,
            "family": family,
            "target_budget": primary_budget,
            "step": step,
            "n": len(values),
            "cell_n": cell_sizes[(task, family)],
            "checkpoint_coverage": len(values) / cell_sizes[(task, family)],
            "nrmse_median": float(np.median(values)),
            "nrmse_q1": float(np.quantile(values, 0.25)),
            "nrmse_q3": float(np.quantile(values, 0.75)),
            "wall_seconds_median": float(np.median(wall[(task, family, step)])),
        })
    return rows


def plot_step_profiles(
    rows: list[dict], output: Path,
    families: tuple[str, ...] = ("PARN", "MLP", "KAN", "SIREN"),
) -> None:
    rows = [row for row in rows if row["family"] in families]
    tasks = sorted({row["task"] for row in rows})
    if not tasks:
        return
    columns = min(2, len(tasks)); nrows = int(np.ceil(len(tasks) / columns))
    figure, axes = plt.subplots(nrows, columns, figsize=(5.0 * columns, 3.6 * nrows), squeeze=False)
    for axis, task in zip(axes.flat, tasks):
        task_rows = [row for row in rows if row["task"] == task]
        for family in sorted({row["family"] for row in task_rows}):
            values = sorted((row for row in task_rows if row["family"] == family), key=lambda row: row["step"])
            axis.plot(
                [row["step"] for row in values], [row["nrmse_median"] for row in values],
                marker="o", label=_family_label(family),
            )
            incomplete = [row for row in values if row["n"] < row["cell_n"]]
            if incomplete:
                axis.scatter(
                    [row["step"] for row in incomplete],
                    [row["nrmse_median"] for row in incomplete],
                    marker="s", facecolors="none", edgecolors="black", s=42, zorder=5,
                )
        axis.set_xscale("log"); axis.set_yscale("log")
        axis.set_title(TASK_LABELS.get(task, task.replace("_", " ")))
        axis.set_xlabel("optimizer steps"); axis.set_ylabel("test NRMSE (median)")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(tasks):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_wall_profiles(
    rows: list[dict], output: Path,
    families: tuple[str, ...] = ("PARN", "MLP", "KAN", "SIREN"),
) -> None:
    """Plot observed hardware time without presenting it as FLOP matching."""
    rows = [row for row in rows if row["family"] in families]
    tasks = sorted({row["task"] for row in rows})
    if not tasks:
        return
    columns = min(2, len(tasks)); nrows = int(np.ceil(len(tasks) / columns))
    figure, axes = plt.subplots(nrows, columns, figsize=(5.0 * columns, 3.6 * nrows), squeeze=False)
    for axis, task in zip(axes.flat, tasks):
        task_rows = [row for row in rows if row["task"] == task]
        for family in sorted({row["family"] for row in task_rows}):
            values = sorted((row for row in task_rows if row["family"] == family), key=lambda row: row["step"])
            axis.plot(
                [row["wall_seconds_median"] for row in values],
                [row["nrmse_median"] for row in values], marker="o",
                label=_family_label(family),
            )
            incomplete = [row for row in values if row["n"] < row["cell_n"]]
            if incomplete:
                axis.scatter(
                    [row["wall_seconds_median"] for row in incomplete],
                    [row["nrmse_median"] for row in incomplete],
                    marker="s", facecolors="none", edgecolors="black", s=42, zorder=5,
                )
        axis.set_xscale("log"); axis.set_yscale("log")
        axis.set_title(TASK_LABELS.get(task, task.replace("_", " ")))
        axis.set_xlabel("synchronized wall time (s, median)"); axis.set_ylabel("test NRMSE (median)")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(tasks):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def robust_log_upper_limit(values: list[float]) -> float | None:
    """Return a Tukey-style log fence only when finite outliers distort a panel."""
    positive = np.asarray([value for value in values if np.isfinite(value) and value > 0])
    if len(positive) < 5:
        return None
    if float(np.max(positive) / np.median(positive)) < 100.0:
        return None
    logs = np.log10(positive)
    q1, q3 = np.quantile(logs, [0.25, 0.75])
    fence_log = float(q3 + 1.5 * max(q3 - q1, 0.15))
    if float(np.max(logs)) <= fence_log + 0.5:
        return None
    return 10.0 ** fence_log


def plot_budget_summary(summaries: list[dict], output: Path) -> None:
    tasks = sorted({row["task"] for row in summaries})
    if not tasks:
        return
    columns = min(2, len(tasks)); rows = int(np.ceil(len(tasks) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 3.6 * rows), squeeze=False)
    for axis, task in zip(axes.flat, tasks):
        task_rows = [row for row in summaries if row["task"] == task]
        upper_limit = robust_log_upper_limit([row["nrmse_median"] for row in task_rows])
        for family in sorted({row["family"] for row in task_rows}):
            values = sorted((row for row in task_rows if row["family"] == family), key=lambda row: row["target_budget"])
            x = [row["actual_parameters_median"] for row in values]
            raw_y = [row["nrmse_median"] for row in values]
            y = [min(value, upper_limit * 0.94) if upper_limit else value for value in raw_y]
            lower = [
                min(max(0.0, row["nrmse_median"] - row["nrmse_q1"]), plotted * 0.95)
                for row, plotted in zip(values, y)
            ]
            upper = [
                max(0.0, min(row["nrmse_q3"], upper_limit * 0.94) - plotted)
                if upper_limit else max(0.0, row["nrmse_q3"] - row["nrmse_median"])
                for row, plotted in zip(values, y)
            ]
            container = axis.errorbar(
                x, y, yerr=[lower, upper], marker="o", capsize=2,
                label=_family_label(family),
            )
            if upper_limit:
                color = container.lines[0].get_color()
                for row, raw_value in zip(values, raw_y):
                    if raw_value <= upper_limit:
                        continue
                    marker_y = upper_limit * 0.94
                    axis.scatter(row["actual_parameters_median"], marker_y, marker="^", color=color, zorder=5)
                    axis.annotate(
                        f"{raw_value:.1g}",
                        (row["actual_parameters_median"], marker_y),
                        xytext=(0, -11), textcoords="offset points",
                        ha="center", va="top", fontsize=6, color=color,
                    )
            failures = [row for row in values if row["numerical_failure_rate"] > 0]
            if failures:
                axis.scatter(
                    [row["actual_parameters_median"] for row in failures],
                    [row["nrmse_median"] for row in failures], marker="x", s=70, color="black",
                )
        axis.set_xscale("log", base=2); axis.set_yscale("log")
        if upper_limit:
            axis.set_ylim(top=upper_limit * 1.15)
        axis.set_title(TASK_LABELS.get(task, task.replace("_", " ")))
        axis.set_xlabel("realized trainable parameters"); axis.set_ylabel("test NRMSE (median [IQR])")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(tasks):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_paired_effects(
    comparisons: list[dict], output: Path, primary_budget: int = 512
) -> int:
    """Plot paired PARN-minus-baseline effects at the prespecified budget."""
    rows = [
        row for row in comparisons
        if int(row["target_budget"]) == primary_budget
        and all(np.isfinite(float(row[key])) for key in ("median_difference", "ci_low", "ci_high"))
    ]
    tasks = sorted({row["task"] for row in rows})
    if not tasks:
        return 0
    columns = min(2, len(tasks)); nrows = int(np.ceil(len(tasks) / columns))
    figure, axes = plt.subplots(
        nrows, columns, figsize=(5.2 * columns, 3.8 * nrows), squeeze=False
    )
    for axis, task in zip(axes.flat, tasks):
        values = sorted(
            (row for row in rows if row["task"] == task),
            key=lambda row: _family_label(row["family_b"]),
        )
        positions = np.arange(len(values))
        effects = np.asarray([float(row["median_difference"]) for row in values])
        low = effects - np.asarray([float(row["ci_low"]) for row in values])
        high = np.asarray([float(row["ci_high"]) for row in values]) - effects
        axis.errorbar(effects, positions, xerr=[low, high], fmt="o", capsize=3)
        axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_yticks(
            positions,
            [_family_label(row["family_b"]) for row in values],
        )
        axis.set_xlabel("paired NRMSE difference (CFNN - baseline)")
        axis.set_title(TASK_LABELS.get(task, task.replace("_", " ")))
        axis.grid(axis="x", alpha=0.2)
    for axis in axes.flat[len(tasks):]:
        axis.set_visible(False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)
    return len(rows)


def plot_ablation_summary(summaries: list[dict], output: Path, primary_budget: int = 512) -> None:
    rows = [
        row for row in summaries
        if int(row["target_budget"]) == primary_budget
        and (row["family"] == "PARN" or row["family"].startswith("PARN-"))
    ]
    families = sorted({row["family"] for row in rows})
    tasks = sorted({row["task"] for row in rows})
    if len(families) < 2 or not tasks:
        return
    figure, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 4.4), squeeze=False)
    for axis, task in zip(axes.flat, tasks):
        values = sorted(
            (row for row in rows if row["task"] == task),
            key=lambda row: (row["nrmse_median"], row["family"]), reverse=True,
        )
        positions = np.arange(len(values))
        medians = np.asarray([row["nrmse_median"] for row in values])
        lower = medians - np.asarray([row["nrmse_q1"] for row in values])
        upper = np.asarray([row["nrmse_q3"] for row in values]) - medians
        axis.errorbar(medians, positions, xerr=[lower, upper], fmt="o", capsize=3)
        axis.set_yticks(positions, [_ablation_family_label(row["family"]) for row in values])
        axis.set_xscale("log")
        axis.set_xlabel("test NRMSE, median [IQR]")
        axis.set_title(TASK_LABELS.get(task, task.replace("_", " ")))
        axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--schema", default=str(ROOT / "confirmatory_results_schema.json"))
    parser.add_argument("--config", default=str(ROOT / "confirmatory_config.json"))
    parser.add_argument("--task-set", choices=("primary", "supplementary", "all"), default="primary")
    parser.add_argument(
        "--macro-namespace", default="",
        help="Prefix generated LaTeX macros to avoid collisions between studies.",
    )
    args = parser.parse_args()
    input_root, output = Path(args.input_root), Path(args.output_dir)
    records = load_records(input_root)
    if not records:
        raise ValueError(f"no evaluation records found under {input_root}")
    validate_records(records, Path(args.schema))
    config = json.loads(Path(args.config).read_text())
    inventory = load_candidate_inventory(input_root)
    validate_completeness(records, config, inventory, task_set=args.task_set)
    summaries = summarize_records(records, max_steps=int(config["training"]["max_steps"]))
    comparisons = paired_comparisons(records)
    profiles = step_profile_rows(records)
    costs = compute_cost_rows(input_root)
    selections = load_selection_rows(input_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True))
    (output / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2, sort_keys=True))
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "paired_comparisons.csv", comparisons)
    write_csv(output / "optimizer_step_profiles.csv", profiles)
    write_csv(output / "compute_costs.csv", costs)
    write_csv(output / "selected_configurations.csv", selections)
    (output / "confirmatory_numbers.tex").write_text(
        render_latex_macros(
            summaries, comparisons=comparisons, profiles=profiles,
            costs=costs, namespace=args.macro_namespace,
        )
    )
    (output / "confirmatory_primary_table.tex").write_text(render_primary_latex_table(summaries))
    (output / "confirmatory_full_table.tex").write_text(
        render_primary_latex_table(summaries, families=tuple(config["families"]))
    )
    (output / "confirmatory_resource_table.tex").write_text(
        render_resource_latex_table(records)
    )
    ablation_table = render_ablation_latex_table(summaries)
    if ablation_table:
        (output / "confirmatory_ablation_table.tex").write_text(ablation_table)
    (output / "artifact_manifest.json").write_text(json.dumps(
        build_artifact_manifest(
            input_root, Path(args.schema), config_path=Path(args.config),
            record_count=len(records), command=sys.argv,
        ),
        indent=2, sort_keys=True,
    ))
    plot_budget_summary(summaries, output / "confirmatory_budget_summary")
    plot_paired_effects(comparisons, output / "confirmatory_paired_effects")
    plot_ablation_summary(summaries, output / "confirmatory_ablation_summary")
    plot_step_profiles(profiles, output / "confirmatory_optimizer_steps")
    plot_wall_profiles(profiles, output / "confirmatory_wall_time")
    print(f"validated {len(records)} records and wrote {len(summaries)} summaries")


if __name__ == "__main__":
    main()

"""Validate downstream records and generate all paper-facing artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
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
)
from downstream_benchmark import make_task_bundle


ROOT = Path(__file__).resolve().parent


def load_records(input_root: Path) -> list[dict]:
    records = []
    for path in sorted((Path(input_root) / "evaluation").rglob("*.jsonl")):
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return records


def load_domain_records(input_root: Path) -> list[dict]:
    path = Path(input_root) / "domain_references" / "records.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_records(records: list[dict], schema_path: Path) -> None:
    validator = Draft202012Validator(json.loads(Path(schema_path).read_text()))
    errors = []
    for index, record in enumerate(records):
        errors.extend(
            f"record {index} at {list(error.path)}: {error.message}"
            for error in validator.iter_errors(record)
        )
    if errors:
        raise ValueError("downstream schema validation failed:\n" + "\n".join(errors[:30]))


def validate_completeness(
    records: list[dict],
    config: dict,
    inventory: dict,
    *,
    tasks: list[str],
    families: list[str],
    budgets: list[int],
) -> None:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["task"], record["family"], int(record["target_budget"]))].append(record)
    expected_pairs = {
        (int(data_seed), int(init_seed))
        for data_seed in config["evaluation_data_seeds"]
        for init_seed in config["evaluation_init_seeds"]
    }
    errors = []
    for task in tasks:
        for family in families:
            for budget in budgets:
                candidates = inventory["tasks"][task][family][str(int(budget))]
                expected = expected_pairs if candidates else set()
                values = grouped.get((task, family, int(budget)), [])
                actual = {
                    (int(record["data_seed"]), int(record["init_seed"]))
                    for record in values
                }
                if len(actual) != len(values):
                    errors.append(f"duplicate pairs {task}/{family}/{budget}")
                if actual != expected:
                    errors.append(
                        f"{task}/{family}/{budget}: expected {len(expected)}, found {len(actual)}"
                    )
    allowed = {
        (task, family, int(budget))
        for task in tasks for family in families for budget in budgets
    }
    extras = sorted(set(grouped) - allowed)
    if extras:
        errors.append(f"records outside frozen matrix: {extras[:5]}")
    if errors:
        raise ValueError("incomplete downstream matrix:\n" + "\n".join(errors[:30]))


def _comparison_records(records: list[dict]) -> list[dict]:
    transformed = []
    for record in records:
        value = record.get("primary_metric_value")
        if value is None or not math.isfinite(float(value)):
            loss = None
        elif record["problem_type"] == "classification":
            loss = 1.0 - float(value)
        else:
            loss = float(value)
        transformed.append({
            **record,
            "test_metrics": {**record.get("test_metrics", {}), "comparison_loss": loss},
        })
    return transformed


def paired_comparisons(
    records: list[dict],
    primary_family: str,
    primary_budget: int,
    *,
    draws: int = 10000,
) -> list[dict]:
    transformed = _comparison_records(records)
    comparisons = []
    grouped = defaultdict(list)
    for record in transformed:
        grouped[(record["task"], int(record["target_budget"]))].append(record)
    for (task, budget), values in sorted(grouped.items()):
        baselines = sorted({record["family"] for record in values} - {primary_family})
        for baseline in baselines:
            try:
                comparison = paired_hierarchical_bootstrap(
                    values,
                    primary_family,
                    baseline,
                    metric="comparison_loss",
                    draws=int(draws),
                    seed=2027 + int(budget),
                )
                significance = paired_instance_sign_flip_test(
                    values, primary_family, baseline, metric="comparison_loss"
                )
            except ValueError:
                continue
            comparisons.append({
                "task": task,
                "target_budget": int(budget),
                **comparison,
                "p_value": significance["p_value"],
                "instance_n_for_test": significance["instance_n"],
                "p_holm": None,
                "difference_interpretation": "negative_favors_CFNN",
            })
    primary_indices = [
        index for index, comparison in enumerate(comparisons)
        if comparison["target_budget"] == int(primary_budget)
    ]
    adjusted = holm_adjust([comparisons[index]["p_value"] for index in primary_indices])
    for index, value in zip(primary_indices, adjusted):
        comparisons[index]["p_holm"] = float(value)
    return comparisons


def summarize_records(records: list[dict], max_steps: int) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["task"], record["family"], int(record["target_budget"]))].append(record)
    rows = []
    for (task, family, budget), values in sorted(grouped.items()):
        finite = [
            float(record["primary_metric_value"])
            for record in values
            if record.get("primary_metric_value") is not None
            and math.isfinite(float(record["primary_metric_value"]))
        ]
        rows.append({
            "task": task,
            "family": family,
            "target_budget": budget,
            "metric_name": values[0]["primary_metric_name"],
            "n": len(values),
            "finite_n": len(finite),
            "metric_median": float(np.median(finite)) if finite else None,
            "metric_q1": float(np.quantile(finite, 0.25)) if finite else None,
            "metric_q3": float(np.quantile(finite, 0.75)) if finite else None,
            "actual_parameters_median": float(np.median([
                record["actual_parameters"] for record in values
            ])),
            "failure_rate": float(sum(
                record["status"] != "ok" or record.get("primary_metric_value") is None
                for record in values
            ) / len(values)),
            "training_cap_rate": float(sum(
                int(record["optimizer_steps"]) >= int(max_steps) for record in values
            ) / len(values)),
            "optimizer_steps_median": float(np.median([
                record["optimizer_steps"] for record in values
            ])),
            "training_wall_seconds_median": float(np.median([
                record["training_wall_seconds"] for record in values
            ])),
        })
    return rows


def summarize_domain_records(records: list[dict]) -> list[dict]:
    """Summarize physical references outside the neural parameter ranking."""
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["task"], record["method"])].append(record)
    rows = []
    for (task, method), values in sorted(grouped.items()):
        finite = [
            float(record["test_metrics"]["complex_nrmse"])
            for record in values
            if record.get("test_metrics", {}).get("complex_nrmse") is not None
            and math.isfinite(float(record["test_metrics"]["complex_nrmse"]))
        ]
        selected_orders = [
            int(record["selected_max_poles"])
            for record in values if record.get("selected_max_poles") is not None
        ]
        rows.append({
            "task": task,
            "method": method,
            "n": len(values),
            "finite_n": len(finite),
            "metric_name": "complex_nrmse",
            "metric_median": float(np.median(finite)) if finite else None,
            "metric_q1": float(np.quantile(finite, 0.25)) if finite else None,
            "metric_q3": float(np.quantile(finite, 0.75)) if finite else None,
            "failure_rate": float(sum(
                record.get("status") != "ok"
                or record.get("test_metrics", {}).get("complex_nrmse") is None
                for record in values
            ) / len(values)),
            "convergence_warning_rate": float(sum(
                int(record.get("test_fit_metadata", {}).get(
                    "convergence_warning_count", 0
                )) > 0 for record in values
            ) / len(values)),
            "selected_order_median": (
                float(np.median(selected_orders)) if selected_orders else None
            ),
        })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _macro_token(value: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", value)
    token = "".join(part[:1].upper() + part[1:] for part in parts if part)
    digit_names = {
        "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
        "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
    }
    return "".join(digit_names.get(character, character) for character in token)


def render_latex_macros(summaries: list[dict], comparisons: list[dict]) -> str:
    lines = ["% Generated from validated downstream records; do not edit manually."]
    for row in summaries:
        prefix = (
            "Downstream" + _macro_token(row["task"]) + _macro_token(row["family"])
            + "B" + _macro_token(str(row["target_budget"]))
        )
        value = "NA" if row["metric_median"] is None else f"{row['metric_median']:.4f}"
        lines.append(f"\\newcommand{{\\{prefix}Metric}}{{{value}}}")
        lines.append(f"\\newcommand{{\\{prefix}N}}{{{row['n']}}}")
        lines.append(f"\\newcommand{{\\{prefix}Parameters}}{{{int(round(row['actual_parameters_median']))}}}")
    for row in comparisons:
        if row["target_budget"] != 256:
            continue
        prefix = "Downstream" + _macro_token(row["task"]) + _macro_token(row["family_b"])
        lines.append(f"\\newcommand{{\\{prefix}Difference}}{{{row['median_difference']:.4f}}}")
        lines.append(f"\\newcommand{{\\{prefix}CI}}{{[{row['ci_low']:.4f}, {row['ci_high']:.4f}]}}")
    return "\n".join(lines) + "\n"


def _metric_uses_log_scale(metric_name: str) -> bool:
    return metric_name in {"nrmse", "complex_nrmse"}


def _parameter_curve_figure(
    summaries: list[dict], output: Path, *, columns: int | None = None
) -> None:
    tasks = sorted({row["task"] for row in summaries})
    if not tasks:
        return
    columns = min(columns or 2, len(tasks))
    rows_count = int(math.ceil(len(tasks) / columns))
    compact_three_column = columns >= 3
    column_width = 4.5 if compact_three_column else 6.2
    figure, axes = plt.subplots(
        rows_count, columns,
        figsize=(column_width * columns, 3.8 * rows_count), squeeze=False,
    )
    for axis, task in zip(axes.flat, tasks):
        task_rows = [row for row in summaries if row["task"] == task]
        for family in sorted({row["family"] for row in task_rows}):
            values = sorted(
                (row for row in task_rows if row["family"] == family),
                key=lambda row: row["actual_parameters_median"],
            )
            finite = [row for row in values if row["metric_median"] is not None]
            if not finite:
                continue
            x_values = [row["actual_parameters_median"] for row in finite]
            y_values = [row["metric_median"] for row in finite]
            lower = [row["metric_median"] - row["metric_q1"] for row in finite]
            upper = [row["metric_q3"] - row["metric_median"] for row in finite]
            axis.errorbar(
                x_values, y_values, yerr=np.asarray([lower, upper]),
                marker="o", capsize=2, linewidth=1.2, label=family,
            )
        axis.set_xscale("log", base=2)
        axis.set_title(task.replace("_", " "), fontsize=12 if compact_three_column else None)
        axis.set_xlabel(
            "Realized trainable parameters",
            fontsize=11 if compact_three_column else None,
        )
        axis.set_ylabel(
            task_rows[0]["metric_name"],
            fontsize=11 if compact_three_column else None,
        )
        if compact_three_column:
            axis.tick_params(labelsize=10)
        if _metric_uses_log_scale(task_rows[0]["metric_name"]):
            axis.set_yscale("log")
        axis.grid(alpha=0.25, which="both")
    for axis in axes.flat[len(tasks):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles, labels, loc="upper center", ncol=min(4, len(labels)),
            fontsize=11 if compact_three_column else None,
        )
    figure.tight_layout(rect=(0, 0, 1, 0.92 if handles else 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def render_primary_effects(
    comparisons: list[dict],
    primary_budget: int,
    output: Path,
    *,
    columns: int | None = None,
) -> None:
    """Plot paired CFNN-minus-baseline loss effects at the primary budget."""
    rows = [
        row for row in comparisons
        if int(row["target_budget"]) == int(primary_budget)
        and all(math.isfinite(float(row[key])) for key in (
            "median_difference", "ci_low", "ci_high"
        ))
    ]
    if not rows:
        return
    tasks = sorted({row["task"] for row in rows})
    columns = min(columns or 2, len(tasks))
    rows_count = int(math.ceil(len(tasks) / columns))
    compact_three_column = columns >= 3
    column_width = 4.6 if compact_three_column else 6.4
    figure, axes = plt.subplots(
        rows_count, columns,
        figsize=(column_width * columns, 3.4 * rows_count), squeeze=False,
    )
    for axis, task in zip(axes.flat, tasks):
        task_rows = sorted(
            (row for row in rows if row["task"] == task),
            key=lambda row: (float(row["median_difference"]), row["family_b"]),
        )
        y = np.arange(len(task_rows))
        centers = np.asarray([float(row["median_difference"]) for row in task_rows])
        lower = centers - np.asarray([float(row["ci_low"]) for row in task_rows])
        upper = np.asarray([float(row["ci_high"]) for row in task_rows]) - centers
        for index, center in enumerate(centers):
            color = "#287a58" if center < 0 else "#b84a3a"
            axis.errorbar(
                center, y[index],
                xerr=np.asarray([[lower[index]], [upper[index]]]),
                fmt="o", color=color, capsize=3, markersize=5,
            )
        axis.axvline(0.0, color="#333333", linewidth=1.0, linestyle="--")
        axis.set_yticks(y, [row["family_b"] for row in task_rows])
        axis.set_title(
            task.replace("_", " "), fontsize=13 if compact_three_column else None
        )
        axis.set_xlabel(
            "CFNN - baseline comparison loss",
            fontsize=11 if compact_three_column else None,
        )
        if compact_three_column:
            axis.tick_params(labelsize=11)
        axis.grid(axis="x", alpha=0.2)
        axis.invert_yaxis()
    for axis in axes.flat[len(tasks):]:
        axis.set_visible(False)
    figure.suptitle(
        f"Primary realized-parameter budget: {int(primary_budget)}; negative favors CFNN",
        fontsize=14 if compact_three_column else None,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def render_task_examples(
    config: dict,
    raw_root: Path,
    output: Path,
    *,
    data_seed: int = 233,
) -> None:
    """Render the four scientific task types and their frozen sparse samples."""
    tasks = (
        "controlled_fano", "microwave_fano",
        "microstrip_resonator", "aluminium_frf",
    )
    titles = {
        "controlled_fano": "Controlled Fano response",
        "microwave_fano": "Measured microwave Fano response",
        "microstrip_resonator": "Measured microstrip S12",
        "aluminium_frf": "Measured aluminium-plate FRF",
    }
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.2))
    for axis, task in zip(axes.flat, tasks):
        bundle = make_task_bundle(
            task, int(data_seed), config, Path(raw_root), include_test=True
        )
        x_parts = (bundle.x_train[:, 0], bundle.x_validation[:, 0], bundle.x_test[:, 0])
        y_parts = (bundle.y_train, bundle.y_validation, bundle.y_test)
        frequency = np.concatenate(x_parts).astype(np.float64)
        response = np.concatenate(y_parts).astype(np.float64)
        magnitude = np.hypot(response[:, 0], response[:, 1])
        train_magnitude = np.hypot(bundle.y_train[:, 0], bundle.y_train[:, 1])
        scale = 1.0
        x_label = "Normalized coordinate"
        if task in {"microwave_fano", "microstrip_resonator"}:
            scale = 1e9
            x_label = "Frequency (GHz)"
        elif task == "aluminium_frf":
            x_label = "Frequency (Hz)"
        order = np.argsort(frequency)
        axis.plot(
            frequency[order] / scale, magnitude[order],
            color="#244860", linewidth=1.2, label="Released/mechanistic response",
        )
        axis.scatter(
            np.asarray(bundle.x_train[:, 0], dtype=np.float64) / scale,
            train_magnitude,
            color="#c4472d", s=14, marker="o", zorder=3, label="Training samples",
        )
        axis.set_title(titles[task])
        axis.set_xlabel(x_label)
        axis.set_ylabel("Complex-response magnitude")
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        if task == "aluminium_frf":
            axis.set_yscale("log")
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "downstream_config.json")
    parser.add_argument("--schema", type=Path, default=ROOT / "downstream_results_schema.json")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "downstream_data" / "raw")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    inventory = json.loads((args.input_root / "candidate_manifest.json").read_text())
    records = load_records(args.input_root)
    domain_records = load_domain_records(args.input_root)
    validate_records(records, args.schema)
    observed_tasks = sorted({record["task"] for record in records})
    observed_families = sorted({record["family"] for record in records})
    observed_budgets = sorted({int(record["target_budget"]) for record in records})
    if not args.allow_incomplete:
        validate_completeness(
            records,
            config,
            inventory,
            tasks=observed_tasks,
            families=observed_families,
            budgets=observed_budgets,
        )
    derived = args.input_root / "derived"
    summaries = summarize_records(records, int(config["training"]["max_steps"]))
    domain_summaries = summarize_domain_records(domain_records)
    comparisons = paired_comparisons(
        records,
        config["primary_family"],
        int(config["primary_budget"]),
    )
    _write_csv(derived / "summary.csv", summaries)
    _write_csv(derived / "domain_reference_summary.csv", domain_summaries)
    _write_csv(derived / "paired_comparisons.csv", comparisons)
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True))
    (derived / "domain_reference_summary.json").write_text(
        json.dumps(domain_summaries, indent=2, sort_keys=True)
    )
    (derived / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2, sort_keys=True))
    (derived / "downstream_results_macros.tex").write_text(
        render_latex_macros(summaries, comparisons)
    )
    _parameter_curve_figure(summaries, derived / "downstream_parameter_curves.pdf")
    render_primary_effects(
        comparisons,
        int(config["primary_budget"]),
        derived / "downstream_primary_effects.pdf",
    )
    render_task_examples(
        config, args.raw_root, derived / "downstream_task_examples.pdf"
    )
    generated = sorted(path for path in derived.iterdir() if path.is_file())
    manifest = {
        "command": sys.argv,
        "record_count": len(records),
        "config_sha256": _sha256(args.config),
        "schema_sha256": _sha256(args.schema),
        "raw_inputs": [
            {"path": str(path.relative_to(args.input_root)), "sha256": _sha256(path)}
            for path in sorted((args.input_root / "evaluation").rglob("*.jsonl"))
        ] + ([{
            "path": "domain_references/records.jsonl",
            "sha256": _sha256(args.input_root / "domain_references" / "records.jsonl"),
        }] if domain_records else []),
        "generated": [
            {"path": path.name, "sha256": _sha256(path)} for path in generated
        ],
    }
    (derived / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({"records": len(records), "summaries": len(summaries), "comparisons": len(comparisons)}))


if __name__ == "__main__":
    main()

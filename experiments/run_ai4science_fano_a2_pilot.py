"""Pilot run for Microwave Fano sparse-observation parameter recovery."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from ai4science_fano import FanoReference, fit_complex_fano_curve
from ai4science_fano_a2 import (
    compare_fano_parameters,
    completed_job_keys,
    complex_nrmse,
    enumerate_a2_jobs,
    job_key,
    make_sparse_curve_split,
    order_a2_jobs,
    peak_window_complex_nrmse,
)
from confirmatory_models import FAMILIES, build_model
from confirmatory_protocol import seeded_build
from downstream_protocol import load_microwave_fano
from downstream_training import fit_task_model, prepare_bundle


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable A2 sparse-observation parameter-recovery pilot jobs."
    )
    parser.add_argument("--raw-root", type=Path, default=ROOT / "downstream_data" / "raw")
    parser.add_argument(
        "--a1-reference-json",
        type=Path,
        default=ROOT / "ai4science_results" / "microwave_fano_a1" / "reference_parameters.json",
    )
    parser.add_argument(
        "--role-manifest",
        type=Path,
        default=ROOT / "peak_sensitive_results" / "provenance" / "microwave_fano_role_manifest.json",
    )
    parser.add_argument(
        "--selection-root",
        type=Path,
        default=ROOT / "peak_sensitive_results" / "selections" / "microwave_fano",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "ai4science_results" / "microwave_fano_a2_pilot",
    )
    parser.add_argument(
        "--curve-ids",
        nargs="+",
        default=None,
        help="Curve ids to evaluate. Defaults to the frozen Microwave Fano evaluation curves.",
    )
    parser.add_argument("--families", nargs="+", default=["CFNN", "MLP"])
    parser.add_argument("--observation-budgets", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    parser.add_argument("--model-budget", type=int, default=512)
    parser.add_argument("--validation-count", type=int, default=128)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[719])
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Backward-compatible single split seed alias. Prefer --split-seeds.",
    )
    parser.add_argument("--init-seeds", nargs="+", type=int, default=[11003])
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--job-order",
        choices=["deterministic", "balanced", "coverage", "science_priority"],
        default="deterministic",
        help="Execution order only; resume identity is unchanged.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--min-steps", type=int, default=200)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--validation-interval", type=int, default=20)
    return parser.parse_args()


def read_jsonl_records(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    records = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL record at {path}:{line_number}") from exc
    return records


def select_pending_jobs(
    jobs: list[dict],
    existing_records: list[dict],
    *,
    max_records: int | None,
) -> list[dict]:
    finished = completed_job_keys(existing_records)
    pending = [job for job in jobs if job_key(job) not in finished]
    if max_records is not None:
        if int(max_records) < 0:
            raise ValueError("max_records must be non-negative")
        pending = pending[: int(max_records)]
    return pending


def _resolve_split_seeds(args: argparse.Namespace) -> list[int]:
    if args.split_seed is not None:
        return [int(args.split_seed)]
    return sorted({int(seed) for seed in args.split_seeds})


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


_FAMILY_BY_SLUG = {_slug(family): family for family in FAMILIES}
_FAMILY_BY_EXACT = {family: family for family in FAMILIES}


def canonical_family_name(value: str) -> str:
    family = str(value)
    if family in _FAMILY_BY_EXACT:
        return _FAMILY_BY_EXACT[family]
    slug = _slug(family)
    if slug in _FAMILY_BY_SLUG:
        return _FAMILY_BY_SLUG[slug]
    raise ValueError(f"unknown family: {value}")


def _load_reference_map(path: Path) -> dict[str, FanoReference]:
    rows = json.loads(Path(path).read_text())
    return {row["curve_id"]: FanoReference(**row) for row in rows}


def _load_evaluation_curve_ids(path: Path) -> list[str]:
    manifest = json.loads(Path(path).read_text())
    return list(manifest["seed_curve_id_map"]["evaluation"].values())


def _load_selection(selection_root: Path, family: str, budget: int) -> dict:
    path = Path(selection_root) / f"{_slug(family)}__{int(budget)}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _csv_safe_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        }
        for row in rows
    ]


def _median_non_null(items: list[dict], field: str) -> float | None:
    values = [item[field] for item in items if item.get(field) is not None]
    return float(np.median(values)) if values else None


def _write_outputs(output_root: Path, rows: list[dict], *, total_job_count: int) -> tuple[Path, Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    raw_path = output_root / "raw_records.jsonl"
    csv_path = output_root / "parameter_errors.csv"
    summary_path = output_root / "summary.json"

    with raw_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    safe_rows = _csv_safe_rows(rows)
    if safe_rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(safe_rows[0]))
            writer.writeheader()
            writer.writerows(safe_rows)
    else:
        csv_path.write_text("")

    grouped = {}
    for row in rows:
        key = (row["family"], int(row["observation_budget"]))
        grouped.setdefault(key, []).append(row)

    summary = {
        "task": "microwave_fano_parameter_recovery_pilot",
        "record_count": len(rows),
        "total_job_count": int(total_job_count),
        "remaining_job_count": max(int(total_job_count) - len(completed_job_keys(rows)), 0),
        "ok_count": sum(1 for row in rows if row["parameter_recovery_status"] == "ok"),
        "families": sorted({row["family"] for row in rows}),
        "curve_ids": sorted({row["curve_id"] for row in rows}),
        "observation_budgets": sorted({int(row["observation_budget"]) for row in rows}),
        "split_seeds": sorted({int(row["split_seed"]) for row in rows}),
        "init_seeds": sorted({int(row["init_seed"]) for row in rows}),
        "by_family_budget": [
            {
                "family": family,
                "observation_budget": budget,
                "record_count": len(items),
                "parameter_recovery_ok_count": sum(
                    1 for item in items if item["parameter_recovery_status"] == "ok"
                ),
                "median_global_complex_nrmse": _median_non_null(items, "global_complex_nrmse"),
                "median_peak_window_complex_nrmse": _median_non_null(
                    items, "peak_window_complex_nrmse"
                ),
                "median_f0_abs_error_hz": _median_non_null(items, "f0_abs_error_hz"),
                "median_quality_factor_relative_error": _median_non_null(
                    items, "quality_factor_relative_error"
                ),
                "median_q_abs_error": _median_non_null(items, "q_abs_error"),
            }
            for (family, budget), items in sorted(grouped.items())
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return raw_path, csv_path, summary_path


def run_one(
    args: argparse.Namespace,
    *,
    curve,
    reference: FanoReference,
    family: str,
    observation_budget: int,
    split_seed: int,
    init_seed: int,
    device: torch.device,
) -> dict:
    selection = _load_selection(args.selection_root, family, args.model_budget)
    candidate = selection["candidate"]
    optimizer = selection["optimizer"]
    bundle = make_sparse_curve_split(
        curve,
        observation_count=int(observation_budget),
        validation_count=int(args.validation_count),
        split_seed=int(split_seed),
    )
    model, model_metadata = seeded_build(
        int(init_seed),
        lambda: build_model(
            family,
            input_dim=1,
            output_dim=2,
            config=candidate["config"],
            seed=int(init_seed),
        ),
    )
    model = model.to(device)
    prepared = prepare_bundle(bundle, "complex_regression", device)
    started = time.perf_counter()
    trained = fit_task_model(
        model,
        prepared,
        "complex_regression",
        learning_rate=float(optimizer["lr"]),
        weight_decay=float(optimizer["weight_decay"]),
        max_steps=int(args.max_steps),
        min_steps=int(args.min_steps),
        patience=int(args.patience),
        checkpoints=[20, 80, 200, 500, int(args.max_steps)],
        gradient_clip=5.0,
        batch_seed=int(split_seed) + 49979687,
        validation_interval=int(args.validation_interval),
    )
    predicted_reference = fit_complex_fano_curve(
        bundle.x_test[:, 0],
        trained.prediction,
        curve_id=curve.curve_id,
        residual_threshold=0.05,
        conditions=curve.conditions,
        metadata={**curve.metadata, "prediction_source": "a2_pilot_model"},
    )
    parameter_errors = compare_fano_parameters(reference, predicted_reference)
    return {
        "task": "microwave_fano_parameter_recovery_pilot",
        "curve_id": curve.curve_id,
        "family": family,
        "model_budget": int(args.model_budget),
        "observation_budget": int(observation_budget),
        "validation_count": int(args.validation_count),
        "split_seed": int(split_seed),
        "init_seed": int(init_seed),
        "training_status": trained.status,
        "best_step": int(trained.best_step),
        "optimizer_steps": int(trained.optimizer_steps),
        "validation_score": float(trained.best_validation_score),
        "global_complex_nrmse": complex_nrmse(bundle.y_test, trained.prediction),
        "peak_window_complex_nrmse": peak_window_complex_nrmse(
            bundle.x_test[:, 0],
            bundle.y_test,
            trained.prediction,
            f0_hz=reference.f0_hz,
            linewidth_hz=reference.linewidth_hz,
        ),
        "training_wall_seconds": float(trained.wall_seconds),
        "total_wall_seconds": float(time.perf_counter() - started),
        "actual_parameters": int(model_metadata["actual_parameters"]),
        "candidate_key": candidate["candidate_key"],
        "optimizer": optimizer,
        "reference_model": reference.reference_model,
        "prediction_reference_status": predicted_reference.status,
        "prediction_reference_relative_magnitude_rmse": predicted_reference.relative_magnitude_rmse,
        **parameter_errors,
    }


def main() -> None:
    args = parse_args()
    references = _load_reference_map(args.a1_reference_json)
    curve_ids = args.curve_ids or _load_evaluation_curve_ids(args.role_manifest)
    families = [canonical_family_name(family) for family in args.families]
    missing_references = sorted(set(curve_ids) - set(references))
    if missing_references:
        raise ValueError(f"curves missing from A1 references: {missing_references}")

    curves = {curve.curve_id: curve for curve in load_microwave_fano(args.raw_root)}
    missing_curves = sorted(set(curve_ids) - set(curves))
    if missing_curves:
        raise ValueError(f"curves missing from raw Microwave Fano data: {missing_curves}")

    split_seeds = _resolve_split_seeds(args)
    jobs = order_a2_jobs(enumerate_a2_jobs(
        curve_ids=curve_ids,
        families=families,
        observation_budgets=args.observation_budgets,
        split_seeds=split_seeds,
        init_seeds=args.init_seeds,
    ), order=args.job_order)
    raw_path = args.output_root / "raw_records.jsonl"
    rows = read_jsonl_records(raw_path) if args.resume else []
    pending = select_pending_jobs(jobs, rows, max_records=args.max_records)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for index, job in enumerate(pending, start=1):
        print(json.dumps({"event": "start_job", "index": index, "total": len(pending), **job}))
        row = run_one(
            args,
            curve=curves[job["curve_id"]],
            reference=references[job["curve_id"]],
            family=job["family"],
            observation_budget=int(job["observation_budget"]),
            split_seed=int(job["split_seed"]),
            init_seed=int(job["init_seed"]),
            device=device,
        )
        rows.append(row)
        _write_outputs(args.output_root, rows, total_job_count=len(jobs))
        print(json.dumps({
            "event": "finish_job",
            "curve_id": row["curve_id"],
            "family": row["family"],
            "observation_budget": row["observation_budget"],
            "split_seed": row["split_seed"],
            "init_seed": row["init_seed"],
            "global_complex_nrmse": row["global_complex_nrmse"],
            "peak_window_complex_nrmse": row["peak_window_complex_nrmse"],
            "f0_abs_error_hz": row["f0_abs_error_hz"],
            "parameter_recovery_status": row["parameter_recovery_status"],
        }, sort_keys=True))

    raw_path, csv_path, summary_path = _write_outputs(args.output_root, rows, total_job_count=len(jobs))
    print(json.dumps({
        "record_count": len(rows),
        "completed_job_count": len(completed_job_keys(rows)),
        "total_job_count": len(jobs),
        "ran_job_count": len(pending),
        "raw_path": str(raw_path),
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "device": str(device),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

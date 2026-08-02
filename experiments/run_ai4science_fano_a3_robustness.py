"""Resumable Microwave Fano A3 noise/mismatch robustness runner."""
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
    complex_nrmse,
    peak_window_complex_nrmse,
)
from ai4science_fano_a3 import (
    _condition_value,
    a3_completed_job_keys,
    a3_job_key,
    enumerate_a3_jobs,
    make_robust_sparse_curve_split,
)
from confirmatory_models import build_model
from confirmatory_protocol import seeded_build
from downstream_protocol import load_microwave_fano
from downstream_training import fit_task_model, prepare_bundle
from run_ai4science_fano_a2_pilot import (
    _csv_safe_rows,
    _load_evaluation_curve_ids,
    _load_reference_map,
    _load_selection,
    _median_non_null,
    canonical_family_name,
    read_jsonl_records,
)


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A3 Microwave Fano robustness jobs under noisy/mismatched observations."
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
        default=ROOT / "ai4science_results" / "microwave_fano_a3_robustness",
    )
    parser.add_argument("--curve-ids", nargs="+", default=None)
    parser.add_argument("--families", nargs="+", default=["CFNN", "MLP", "local_nested_cf_control"])
    parser.add_argument("--observation-budgets", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--model-budget", type=int, default=512)
    parser.add_argument("--validation-count", type=int, default=128)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[719])
    parser.add_argument("--init-seeds", nargs="+", type=int, default=[11003])
    parser.add_argument("--noise-snr-dbs", nargs="+", type=float, default=[40.0, 30.0, 20.0, 10.0])
    parser.add_argument("--frequency-drift-ppms", nargs="+", type=float, default=[0.0])
    parser.add_argument("--missing-fractions", nargs="+", type=float, default=[0.0])
    parser.add_argument("--perturbation-seeds", nargs="+", type=int, default=[3001])
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--min-steps", type=int, default=200)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--validation-interval", type=int, default=20)
    return parser.parse_args()


def select_pending_jobs(
    jobs: list[dict],
    existing_records: list[dict],
    *,
    max_records: int | None,
) -> list[dict]:
    finished = a3_completed_job_keys(existing_records)
    pending = [job for job in jobs if a3_job_key(job) not in finished]
    if max_records is not None:
        if int(max_records) < 0:
            raise ValueError("max_records must be non-negative")
        pending = pending[: int(max_records)]
    return pending


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
        key = (
            row["family"],
            int(row["observation_budget"]),
            _condition_value(row["noise_snr_db"]),
            float(row["frequency_drift_ppm"]),
            float(row["missing_fraction"]),
        )
        grouped.setdefault(key, []).append(row)

    summary = {
        "task": "microwave_fano_a3_noise_mismatch_robustness",
        "record_count": len(rows),
        "total_job_count": int(total_job_count),
        "remaining_job_count": max(int(total_job_count) - len(a3_completed_job_keys(rows)), 0),
        "ok_count": sum(1 for row in rows if row["parameter_recovery_status"] == "ok"),
        "families": sorted({row["family"] for row in rows}),
        "curve_ids": sorted({row["curve_id"] for row in rows}),
        "observation_budgets": sorted({int(row["observation_budget"]) for row in rows}),
        "noise_snr_dbs": sorted({_condition_value(row["noise_snr_db"]) for row in rows}),
        "frequency_drift_ppms": sorted({float(row["frequency_drift_ppm"]) for row in rows}),
        "missing_fractions": sorted({float(row["missing_fraction"]) for row in rows}),
        "split_seeds": sorted({int(row["split_seed"]) for row in rows}),
        "init_seeds": sorted({int(row["init_seed"]) for row in rows}),
        "perturbation_seeds": sorted({int(row["perturbation_seed"]) for row in rows}),
        "by_family_budget_condition": [
            {
                "family": family,
                "observation_budget": budget,
                "noise_snr_db": noise_snr_db,
                "frequency_drift_ppm": frequency_drift_ppm,
                "missing_fraction": missing_fraction,
                "record_count": len(items),
                "parameter_recovery_ok_count": sum(
                    1 for item in items if item["parameter_recovery_status"] == "ok"
                ),
                "parameter_recovery_failure_rate": (
                    1.0
                    - sum(1 for item in items if item["parameter_recovery_status"] == "ok") / len(items)
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
            for (family, budget, noise_snr_db, frequency_drift_ppm, missing_fraction), items
            in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3], item[0][4]))
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
    noise_snr_db: float | None,
    frequency_drift_ppm: float,
    missing_fraction: float,
    perturbation_seed: int,
    device: torch.device,
) -> dict:
    selection = _load_selection(args.selection_root, family, args.model_budget)
    candidate = selection["candidate"]
    optimizer = selection["optimizer"]
    bundle = make_robust_sparse_curve_split(
        curve,
        observation_count=int(observation_budget),
        validation_count=int(args.validation_count),
        split_seed=int(split_seed),
        perturbation_seed=int(perturbation_seed),
        noise_snr_db=noise_snr_db,
        frequency_drift_ppm=float(frequency_drift_ppm),
        missing_fraction=float(missing_fraction),
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
        batch_seed=int(split_seed) + int(perturbation_seed) + 49979687,
        validation_interval=int(args.validation_interval),
    )
    predicted_reference = fit_complex_fano_curve(
        bundle.x_test[:, 0],
        trained.prediction,
        curve_id=curve.curve_id,
        residual_threshold=0.05,
        conditions=curve.conditions,
        metadata={**curve.metadata, "prediction_source": "a3_robustness_model"},
    )
    parameter_errors = compare_fano_parameters(reference, predicted_reference)
    return {
        "task": "microwave_fano_a3_noise_mismatch_robustness",
        "curve_id": curve.curve_id,
        "family": family,
        "model_budget": int(args.model_budget),
        "observation_budget": int(observation_budget),
        "requested_observation_budget": int(bundle.metadata["requested_observation_budget"]),
        "actual_observation_count": int(bundle.metadata["actual_observation_count"]),
        "validation_count": int(args.validation_count),
        "split_seed": int(split_seed),
        "init_seed": int(init_seed),
        "noise_snr_db": _condition_value(noise_snr_db),
        "frequency_drift_ppm": float(frequency_drift_ppm),
        "missing_fraction": float(missing_fraction),
        "perturbation_seed": int(perturbation_seed),
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

    jobs = enumerate_a3_jobs(
        curve_ids=curve_ids,
        families=families,
        observation_budgets=args.observation_budgets,
        split_seeds=args.split_seeds,
        init_seeds=args.init_seeds,
        noise_snr_dbs=args.noise_snr_dbs,
        frequency_drift_ppms=args.frequency_drift_ppms,
        missing_fractions=args.missing_fractions,
        perturbation_seeds=args.perturbation_seeds,
    )
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
            noise_snr_db=_condition_value(job["noise_snr_db"]),
            frequency_drift_ppm=float(job["frequency_drift_ppm"]),
            missing_fraction=float(job["missing_fraction"]),
            perturbation_seed=int(job["perturbation_seed"]),
            device=device,
        )
        rows.append(row)
        _write_outputs(args.output_root, rows, total_job_count=len(jobs))
        print(json.dumps({
            "event": "finish_job",
            "curve_id": row["curve_id"],
            "family": row["family"],
            "observation_budget": row["observation_budget"],
            "noise_snr_db": row["noise_snr_db"],
            "frequency_drift_ppm": row["frequency_drift_ppm"],
            "missing_fraction": row["missing_fraction"],
            "global_complex_nrmse": row["global_complex_nrmse"],
            "peak_window_complex_nrmse": row["peak_window_complex_nrmse"],
            "f0_abs_error_hz": row["f0_abs_error_hz"],
            "parameter_recovery_status": row["parameter_recovery_status"],
        }, sort_keys=True))

    raw_path, csv_path, summary_path = _write_outputs(args.output_root, rows, total_job_count=len(jobs))
    print(json.dumps({
        "record_count": len(rows),
        "completed_job_count": len(a3_completed_job_keys(rows)),
        "total_job_count": len(jobs),
        "ran_job_count": len(pending),
        "raw_path": str(raw_path),
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "device": str(device),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

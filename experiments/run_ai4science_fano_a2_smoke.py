"""Smoke run for Microwave Fano sparse-observation parameter recovery."""
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
    make_sparse_curve_split,
)
from confirmatory_models import build_model
from confirmatory_protocol import seeded_build
from downstream_protocol import load_microwave_fano
from downstream_training import fit_task_model, prepare_bundle


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small A2 sparse-observation parameter-recovery smoke experiment."
    )
    parser.add_argument("--raw-root", type=Path, default=ROOT / "downstream_data" / "raw")
    parser.add_argument(
        "--a1-reference-json",
        type=Path,
        default=ROOT / "ai4science_results" / "microwave_fano_a1" / "reference_parameters.json",
    )
    parser.add_argument(
        "--selection-root",
        type=Path,
        default=ROOT / "peak_sensitive_results" / "selections" / "microwave_fano",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "ai4science_results" / "microwave_fano_a2_smoke",
    )
    parser.add_argument("--curve-id", default="undercoupled:r2:p07")
    parser.add_argument("--families", nargs="+", default=["CFNN", "MLP"])
    parser.add_argument("--observation-budgets", nargs="+", type=int, default=[32, 128])
    parser.add_argument("--model-budget", type=int, default=512)
    parser.add_argument("--validation-count", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=719)
    parser.add_argument("--init-seed", type=int, default=11003)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--min-steps", type=int, default=200)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--validation-interval", type=int, default=20)
    return parser.parse_args()


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _load_reference_map(path: Path) -> dict[str, FanoReference]:
    rows = json.loads(Path(path).read_text())
    references = {}
    for row in rows:
        references[row["curve_id"]] = FanoReference(**row)
    return references


def _load_selection(selection_root: Path, family: str, budget: int) -> dict:
    path = Path(selection_root) / f"{_slug(family)}__{int(budget)}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _csv_safe_rows(rows: list[dict]) -> list[dict]:
    safe = []
    for row in rows:
        safe.append({
            key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        })
    return safe


def _write_outputs(output_root: Path, rows: list[dict]) -> tuple[Path, Path, Path]:
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
        "record_count": len(rows),
        "ok_count": sum(1 for row in rows if row["parameter_recovery_status"] == "ok"),
        "families": sorted({row["family"] for row in rows}),
        "observation_budgets": sorted({int(row["observation_budget"]) for row in rows}),
        "by_family_budget": [
            {
                "family": family,
                "observation_budget": budget,
                "record_count": len(items),
                "median_global_complex_nrmse": float(np.median([
                    item["global_complex_nrmse"] for item in items
                ])),
                "median_f0_abs_error_hz": (
                    float(np.median([
                        item["f0_abs_error_hz"] for item in items
                        if item["f0_abs_error_hz"] is not None
                    ]))
                    if any(item["f0_abs_error_hz"] is not None for item in items)
                    else None
                ),
                "parameter_recovery_ok_count": sum(
                    1 for item in items if item["parameter_recovery_status"] == "ok"
                ),
            }
            for (family, budget), items in sorted(grouped.items())
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return raw_path, csv_path, summary_path


def run_one(args: argparse.Namespace, *, curve, reference: FanoReference,
            family: str, observation_budget: int, device: torch.device) -> dict:
    selection = _load_selection(args.selection_root, family, args.model_budget)
    candidate = selection["candidate"]
    optimizer = selection["optimizer"]
    bundle = make_sparse_curve_split(
        curve,
        observation_count=int(observation_budget),
        validation_count=int(args.validation_count),
        split_seed=int(args.split_seed),
    )
    model, model_metadata = seeded_build(
        int(args.init_seed),
        lambda: build_model(
            family,
            input_dim=1,
            output_dim=2,
            config=candidate["config"],
            seed=int(args.init_seed),
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
        batch_seed=int(args.split_seed) + 49979687,
        validation_interval=int(args.validation_interval),
    )
    predicted_reference = fit_complex_fano_curve(
        bundle.x_test[:, 0],
        trained.prediction,
        curve_id=curve.curve_id,
        residual_threshold=0.05,
        conditions=curve.conditions,
        metadata={**curve.metadata, "prediction_source": "a2_smoke_model"},
    )
    parameter_errors = compare_fano_parameters(reference, predicted_reference)
    return {
        "task": "microwave_fano_parameter_recovery_smoke",
        "curve_id": curve.curve_id,
        "family": family,
        "model_budget": int(args.model_budget),
        "observation_budget": int(observation_budget),
        "validation_count": int(args.validation_count),
        "split_seed": int(args.split_seed),
        "init_seed": int(args.init_seed),
        "training_status": trained.status,
        "best_step": int(trained.best_step),
        "optimizer_steps": int(trained.optimizer_steps),
        "validation_score": float(trained.best_validation_score),
        "global_complex_nrmse": complex_nrmse(bundle.y_test, trained.prediction),
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
    if args.curve_id not in references:
        raise ValueError(f"curve {args.curve_id!r} is missing from A1 references")
    curves = {curve.curve_id: curve for curve in load_microwave_fano(args.raw_root)}
    if args.curve_id not in curves:
        raise ValueError(f"curve {args.curve_id!r} is missing from raw Microwave Fano data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for family in args.families:
        for observation_budget in args.observation_budgets:
            rows.append(run_one(
                args,
                curve=curves[args.curve_id],
                reference=references[args.curve_id],
                family=family,
                observation_budget=int(observation_budget),
                device=device,
            ))
    raw_path, csv_path, summary_path = _write_outputs(args.output_root, rows)
    print(json.dumps({
        "record_count": len(rows),
        "raw_path": str(raw_path),
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "device": str(device),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

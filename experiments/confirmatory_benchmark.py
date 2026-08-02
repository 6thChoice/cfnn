"""Auditable tuning and confirmatory evaluation for the PARN study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

# Must be set before the first CUDA/cuBLAS operation for reproducible GEMM kernels.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

torch.use_deterministic_algorithms(True, warn_only=True)
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from confirmatory_artifacts import build_run_manifest, freeze_budget_grid
from confirmatory_models import build_model, enumerate_candidates
from confirmatory_protocol import (
    DatasetBundle,
    make_analytic_bundle,
    make_energy_bundle,
    make_nmr_bundle,
    rank_finalists_by_validation,
    seeded_build,
)
from confirmatory_training import compute_regression_metrics, train_with_validation


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def _read_json(path: Path):
    return json.loads(path.read_text())


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False))


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(_json_safe(value), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _task_shape(task: str) -> tuple[int, int]:
    return (3, 1) if task == "manifold_bump" else (8, 1) if task == "energy" else (1, 1)


def make_bundle(task: str, data_seed: int, config: dict) -> DatasetBundle:
    dataset = config["dataset"]
    if task.startswith("nmr_"):
        return make_nmr_bundle(
            task.removeprefix("nmr_"), dataset["nmr_observed"], dataset["nmr_noise"],
            data_seed, dataset["nmr_dense"],
        )
    if task == "energy":
        return make_energy_bundle(data_seed)
    return make_analytic_bundle(
        task, data_seed, dataset["analytic_train"], dataset["analytic_validation"],
        dataset["analytic_test"], dataset["analytic_noise"],
    )


def _prepare(bundle: DatasetBundle):
    x_mean = bundle.x_train.mean(axis=0, keepdims=True)
    x_std = bundle.x_train.std(axis=0, keepdims=True) + 1e-6
    y_mean = bundle.y_train.mean(axis=0, keepdims=True)
    y_std = bundle.y_train.std(axis=0, keepdims=True) + 1e-6
    normalized = (
        (bundle.x_train - x_mean) / x_std,
        (bundle.y_train - y_mean) / y_std,
        (bundle.x_validation - x_mean) / x_std,
        (bundle.y_validation - y_mean) / y_std,
        (bundle.x_test - x_mean) / x_std,
    )
    x_train, y_train, x_validation, y_validation, x_test = [
        torch.from_numpy(np.asarray(value, dtype=np.float32)).to(DEVICE)
        for value in normalized
    ]
    return x_train, y_train, x_validation, y_validation, x_test, y_mean, y_std


def train_one(task: str, family: str, candidate: dict, optimizer: dict,
              data_seed: int, init_seed: int, config: dict) -> dict:
    bundle = make_bundle(task, data_seed, config)
    input_dim, output_dim = _task_shape(task)
    started = time.perf_counter()
    model, metadata = seeded_build(
        init_seed,
        lambda: build_model(family, input_dim, output_dim, candidate["config"], seed=init_seed),
    )
    model = model.to(DEVICE)
    x_train, y_train, x_validation, y_validation, x_test, y_mean, y_std = _prepare(bundle)
    y_test_normalized = torch.from_numpy(
        np.asarray((bundle.y_test - y_mean) / y_std, dtype=np.float32)
    ).to(DEVICE)
    settings = config["training"]
    trained = train_with_validation(
        model, x_train, y_train, x_validation, y_validation,
        learning_rate=optimizer["lr"], weight_decay=optimizer["weight_decay"],
        max_steps=settings["max_steps"], min_steps=settings["min_steps"],
        patience=settings["patience"], checkpoints=settings["checkpoints"],
        gradient_clip=settings["gradient_clip"],
        x_test=x_test, y_test=y_test_normalized,
    )
    trained.model.eval()
    with torch.no_grad():
        normalized_prediction = trained.model(x_test).detach().cpu().numpy()
    prediction = normalized_prediction * y_std + y_mean
    metrics = compute_regression_metrics(bundle.y_test, prediction)
    status = trained.status if trained.status != "ok" else metrics["status"]
    return {
        "task": task,
        "family": family,
        "data_seed": int(data_seed),
        "init_seed": int(init_seed),
        "candidate_key": candidate["candidate_key"],
        "config": candidate["config"],
        "actual_parameters": candidate["actual_parameters"],
        "optimizer": optimizer,
        "best_step": trained.best_step,
        "optimizer_steps": trained.optimizer_steps,
        "validation_nrmse": trained.best_validation_nrmse,
        "test_metrics": metrics,
        "status": status,
        "curve": trained.curve,
        "training_wall_seconds": trained.wall_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "peak_memory_bytes": trained.peak_memory_bytes,
        "device": str(DEVICE),
        "dataset": bundle.metadata,
        "model_metadata": metadata,
    }


def build_inventory(config: dict, output_root: Path, *, maximum_candidates: int | None = None,
                    tasks: list[str] | None = None, families: list[str] | None = None,
                    budgets: list[int] | None = None) -> dict:
    tasks = tasks or config["primary_tasks"] + config["supplementary_tasks"]
    families = families or config["families"]
    budgets = budgets or config["budgets"]
    inventory = {
        "protocol_version": config["protocol_version"],
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "tasks": {},
    }
    cache = {}
    maximum = maximum_candidates or config["maximum_candidates_per_budget"]
    for task in tasks:
        input_dim, output_dim = _task_shape(task)
        inventory["tasks"][task] = {}
        for family in families:
            cache_key = (family, input_dim, output_dim)
            if cache_key not in cache:
                cache[cache_key] = enumerate_candidates(family, input_dim, output_dim)
            frozen = freeze_budget_grid(
                cache[cache_key], budgets, tolerance=config["budget_tolerance"],
                minimum_per_budget=config["minimum_candidates_per_budget"],
                maximum_per_budget=maximum,
            )
            inventory["tasks"][task][family] = frozen
    _write_json(output_root / "candidate_manifest.json", inventory)
    return inventory


def _selection_path(output_root: Path, task: str, family: str, budget: int) -> Path:
    return output_root / "selections" / _slug(task) / f"{_slug(family)}__{budget}.json"


def _tuning_path(output_root: Path, task: str, family: str, budget: int) -> Path:
    return output_root / "tuning_final" / _slug(task) / f"{_slug(family)}__{budget}.jsonl"


def _screening_path(output_root: Path, task: str, family: str, budget: int) -> Path:
    return output_root / "tuning_screen" / _slug(task) / f"{_slug(family)}__{budget}.jsonl"


def tune_shard(config: dict, inventory: dict, output_root: Path,
               task: str, family: str, budget: int) -> dict | None:
    candidates = inventory["tasks"][task][family][str(budget)]
    if not candidates:
        _write_json(_selection_path(output_root, task, family, budget), {
            "task": task, "family": family, "target_budget": budget, "status": "missing_budget_match"
        })
        return None
    optimizers = config["optimizer_grids"][family]
    screening_optimizer = optimizers[min(1, len(optimizers) - 1)]
    screening_config = json.loads(json.dumps(config))
    screening_config["training"].update({
        "max_steps": config["screening_steps"],
        "min_steps": min(config["screening_steps"], config["training"]["min_steps"]),
        "patience": config["screening_steps"],
        "checkpoints": sorted(set(
            [step for step in config["training"]["checkpoints"] if step <= config["screening_steps"]]
            + [config["screening_steps"]]
        )),
    })
    screen_path = _screening_path(output_root, task, family, budget)
    screen_records = _load_jsonl(screen_path)
    screened = {(record["candidate_key"], record["tuning_seed"]) for record in screen_records}
    for candidate in candidates:
        for tuning_seed in config["tuning_seeds"]:
            if (candidate["candidate_key"], tuning_seed) in screened:
                continue
            record = train_one(
                task, family, candidate, screening_optimizer,
                data_seed=tuning_seed, init_seed=tuning_seed + 1_000_003,
                config=screening_config,
            )
            record["stage"] = "tuning_screen"
            record["target_budget"] = budget
            record["tuning_seed"] = tuning_seed
            _append_jsonl(screen_path, record)
            print("screen", task, family, budget, candidate["candidate_key"],
                  tuning_seed, record["validation_nrmse"], flush=True)
    screen_records = _load_jsonl(screen_path)
    finalist_keys = set(rank_finalists_by_validation(
        screen_records, limit=min(config["finalists_per_budget"], len(candidates))
    ))
    finalists = [candidate for candidate in candidates if candidate["candidate_key"] in finalist_keys]
    path = _tuning_path(output_root, task, family, budget)
    existing = _load_jsonl(path)
    completed = {
        (record["candidate_key"], record["optimizer"]["key"], record["tuning_seed"])
        for record in existing
    }
    for candidate in finalists:
        for optimizer in optimizers:
            for tuning_seed in config["tuning_seeds"]:
                key = (candidate["candidate_key"], optimizer["key"], tuning_seed)
                if key in completed:
                    continue
                record = train_one(
                    task, family, candidate, optimizer,
                    data_seed=tuning_seed, init_seed=tuning_seed + 1_000_003, config=config,
                )
                record["stage"] = "tuning"
                record["target_budget"] = budget
                record["tuning_seed"] = tuning_seed
                _append_jsonl(path, record)
                print("tuning", task, family, budget, candidate["candidate_key"],
                      optimizer["key"], tuning_seed, record["validation_nrmse"], flush=True)
    records = _load_jsonl(path)
    grouped = {}
    for record in records:
        grouped.setdefault((record["candidate_key"], record["optimizer"]["key"]), []).append(record)
    expected = len(config["tuning_seeds"])
    complete_groups = {key: values for key, values in grouped.items() if len(values) == expected}
    winner_key = min(
        complete_groups,
        key=lambda key: (
            float(np.median([
                value["validation_nrmse"] if value["validation_nrmse"] is not None and math.isfinite(value["validation_nrmse"]) else float("inf")
                for value in complete_groups[key]
            ])),
            key,
        ),
    )
    winner_records = complete_groups[winner_key]
    candidate = next(item for item in candidates if item["candidate_key"] == winner_key[0])
    optimizer = next(item for item in optimizers if item["key"] == winner_key[1])
    selection = {
        "task": task,
        "family": family,
        "target_budget": budget,
        "status": "selected",
        "candidate": candidate,
        "optimizer": optimizer,
        "selected_epoch": int(round(np.median([item["best_step"] for item in winner_records]))),
        "median_validation_nrmse": float(np.median([
            item["validation_nrmse"] if item["validation_nrmse"] is not None else float("inf")
            for item in winner_records
        ])),
        "screened_candidate_count": len(candidates),
        "finalist_count": len(finalists),
        "optimizer_count": len(optimizers),
        "tuning_seed_count": expected,
    }
    _write_json(_selection_path(output_root, task, family, budget), selection)
    return selection


def evaluate_shard(config: dict, output_root: Path, task: str, family: str, budget: int) -> None:
    selection_path = _selection_path(output_root, task, family, budget)
    if not selection_path.exists():
        raise FileNotFoundError(f"missing tuning selection: {selection_path}")
    selection = _read_json(selection_path)
    if selection["status"] != "selected":
        return
    pairs = [
        (data_seed, init_seed)
        for data_seed in config["evaluation_data_seeds"]
        for init_seed in config["evaluation_init_seeds"]
    ]
    if "evaluation_pairs_limit" in config:
        pairs = pairs[:config["evaluation_pairs_limit"]]
    if task in config["supplementary_tasks"]:
        pairs = pairs[: config["auxiliary_pairs"]]
    path = output_root / "evaluation" / _slug(task) / f"{_slug(family)}__{budget}.jsonl"
    existing = _load_jsonl(path)
    completed = {(item["data_seed"], item["init_seed"]) for item in existing}
    for data_seed, init_seed in pairs:
        if (data_seed, init_seed) in completed:
            continue
        record = train_one(
            task, family, selection["candidate"], selection["optimizer"],
            data_seed=data_seed, init_seed=init_seed, config=config,
        )
        record["stage"] = "confirmatory_evaluation"
        record["target_budget"] = budget
        record["selection_path"] = str(selection_path)
        _append_jsonl(path, record)
        print("evaluation", task, family, budget, data_seed, init_seed,
              record["test_metrics"]["nrmse"], record["status"], flush=True)


def _filter(values: list, requested: str | None, cast=str):
    if not requested:
        return values
    wanted = [cast(value.strip()) for value in requested.split(",") if value.strip()]
    unknown = set(wanted) - set(values)
    if unknown:
        raise ValueError(f"unknown requested values: {sorted(unknown)}")
    return wanted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "confirmatory_config.json"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage", choices=("inventory", "tune", "evaluate", "all"), default="all")
    parser.add_argument("--tasks")
    parser.add_argument("--families")
    parser.add_argument("--budgets")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--min-steps", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--maximum-candidates", type=int)
    parser.add_argument("--maximum-optimizers", type=int)
    parser.add_argument("--tuning-seed-count", type=int)
    parser.add_argument("--evaluation-pairs", type=int)
    args = parser.parse_args()
    config = _read_json(Path(args.config))
    if args.smoke:
        config = json.loads(json.dumps(config))
        config["tuning_seeds"] = config["tuning_seeds"][:1]
        config["evaluation_data_seeds"] = config["evaluation_data_seeds"][:1]
        config["evaluation_init_seeds"] = config["evaluation_init_seeds"][:1]
        config["training"].update({"max_steps": 20, "min_steps": 5, "patience": 10, "checkpoints": [5, 20]})
        for family in config["families"]:
            config["optimizer_grids"][family] = config["optimizer_grids"][family][:1]
    for key, value in (("max_steps", args.max_steps), ("min_steps", args.min_steps),
                       ("patience", args.patience)):
        if value is not None:
            config["training"][key] = value
    if args.tuning_seed_count is not None:
        config["tuning_seeds"] = config["tuning_seeds"][:args.tuning_seed_count]
    if args.maximum_optimizers is not None:
        for family in config["families"]:
            config["optimizer_grids"][family] = config["optimizer_grids"][family][:args.maximum_optimizers]
    if args.evaluation_pairs is not None:
        config["evaluation_pairs_limit"] = args.evaluation_pairs
    config["screening_steps"] = min(config["screening_steps"], config["training"]["max_steps"])
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = _filter(config["primary_tasks"] + config["supplementary_tasks"], args.tasks)
    families = _filter(config["families"], args.families)
    budgets = _filter(config["budgets"], args.budgets, int)
    code_paths = [
        Path(__file__), ROOT / "confirmatory_protocol.py", ROOT / "confirmatory_models.py",
        ROOT / "confirmatory_training.py", ROOT / "confirmatory_artifacts.py",
        ROOT / "fair_baselines.py", ROOT / "budget_matching.py", ROOT / "nmr_data.py",
        ROOT / "energy_efficiency.csv",
    ]
    _write_json(output_root / "run_manifest.json", build_run_manifest(config, sys.argv, code_paths))
    inventory_path = output_root / "candidate_manifest.json"
    if args.stage == "inventory" or not inventory_path.exists():
        inventory = build_inventory(
            config, output_root,
            maximum_candidates=(2 if args.smoke else args.maximum_candidates),
            tasks=tasks, families=families, budgets=budgets,
        )
    else:
        inventory = _read_json(inventory_path)
    if args.stage == "inventory":
        return
    for task in tasks:
        for family in families:
            for budget in budgets:
                if args.stage in {"tune", "all"}:
                    tune_shard(config, inventory, output_root, task, family, budget)
                if args.stage in {"evaluate", "all"}:
                    evaluate_shard(config, output_root, task, family, budget)


if __name__ == "__main__":
    main()

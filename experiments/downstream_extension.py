"""Append-only protocol extensions for the CFNN downstream study."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selection_tree_sha256(base_root: Path) -> str:
    """Hash selection paths and bytes so additions and renames are detectable."""
    base_root = Path(base_root)
    selection_root = base_root / "selections"
    paths = sorted(selection_root.rglob("*.json"))
    if not paths:
        raise ValueError(f"selection tree is empty: {selection_root}")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(selection_root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def load_extension(
    path: Path,
    base_config_path: Path,
    base_root: Path,
) -> dict:
    """Load an extension only when every frozen base artifact still matches."""
    path = Path(path)
    base_config_path = Path(base_config_path)
    base_root = Path(base_root)
    extension = json.loads(path.read_text())
    checks = {
        "base_config_sha256": sha256_file(base_config_path),
        "base_candidate_manifest_sha256": sha256_file(
            base_root / "candidate_manifest.json"
        ),
        "base_conventional_selection_sha256": sha256_file(
            base_root / "conventional_selection.json"
        ),
        "base_selection_tree_sha256": selection_tree_sha256(base_root),
    }
    for key, actual in checks.items():
        if extension.get(key) != actual:
            raise ValueError(
                f"{key} mismatch: expected {extension.get(key)!r}, found {actual!r}"
            )
    base_config = json.loads(base_config_path.read_text())
    base_seeds = {int(seed) for seed in base_config["evaluation_init_seeds"]}
    added = [int(seed) for seed in extension["additional_evaluation_init_seeds"]]
    if len(added) != len(set(added)):
        raise ValueError("additional evaluation initialization seeds are not unique")
    overlap = sorted(base_seeds.intersection(added))
    if overlap:
        raise ValueError(f"extension initialization seeds overlap base seeds: {overlap}")
    if len(base_seeds) + len(added) < 10:
        raise ValueError("extension must provide at least ten total initialization seeds")
    return {**extension, "additional_evaluation_init_seeds": added}


def _slug(value: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "_" for character in value
    ).strip("_")


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def evaluate_seed_extension(
    base_config_path: Path,
    extension_path: Path,
    base_root: Path,
    output_root: Path,
    task: str,
    family: str,
    budget: int,
    raw_root: Path,
    *,
    train_fn=None,
) -> Path | None:
    """Evaluate only added seeds while treating the base evidence as read-only."""
    base_config_path = Path(base_config_path)
    extension_path = Path(extension_path)
    base_root = Path(base_root)
    output_root = Path(output_root)
    extension = load_extension(extension_path, base_config_path, base_root)
    config = json.loads(base_config_path.read_text())
    selection_path = (
        base_root / "selections" / _slug(task)
        / f"{_slug(family)}__{int(budget)}.json"
    )
    if not selection_path.exists():
        raise ValueError(f"missing frozen base selection: {selection_path}")
    selection = json.loads(selection_path.read_text())
    if selection.get("status") != "selected":
        return None
    output_path = (
        output_root / "evaluation" / _slug(task)
        / f"{_slug(family)}__{int(budget)}.jsonl"
    )
    existing = [
        json.loads(line) for line in output_path.read_text().splitlines() if line.strip()
    ] if output_path.exists() else []
    completed = {
        (int(record["data_seed"]), int(record["init_seed"])) for record in existing
    }
    if train_fn is None:
        from downstream_benchmark import train_one
        train_fn = train_one
    extension_sha256 = sha256_file(extension_path)
    selection_sha256 = sha256_file(selection_path)
    for data_seed in config["evaluation_data_seeds"]:
        for init_seed in extension["additional_evaluation_init_seeds"]:
            pair = (int(data_seed), int(init_seed))
            if pair in completed:
                continue
            record = train_fn(
                task,
                family,
                selection["candidate"],
                selection["optimizer"],
                data_seed=pair[0],
                init_seed=pair[1],
                config=config,
                raw_root=Path(raw_root),
                evaluate_test=True,
            )
            record.update({
                "stage": "confirmatory_seed_extension",
                "stage_role": "confirmatory_test",
                "target_budget": int(budget),
                "selection_path": str(selection_path),
                "base_selection_sha256": selection_sha256,
                "protocol_extension_sha256": extension_sha256,
            })
            _append_jsonl(output_path, record)
            completed.add(pair)
    return output_path


def _filter(values: list, requested: str | None, converter=str) -> list:
    if not requested:
        return list(values)
    allowed = set(values)
    selected = [converter(item.strip()) for item in requested.split(",") if item.strip()]
    unknown = [item for item in selected if item not in allowed]
    if unknown:
        raise ValueError(f"unknown requested values: {unknown}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--extension-config", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--tasks")
    parser.add_argument("--families")
    parser.add_argument("--budgets")
    args = parser.parse_args()

    extension = load_extension(args.extension_config, args.base_config, args.base_root)
    base_config = json.loads(args.base_config.read_text())
    conventional = json.loads(
        (args.base_root / "conventional_selection.json").read_text()
    )["selected_tasks"]
    tasks = _filter(
        list(base_config["fixed_scientific_tasks"]) + list(conventional), args.tasks
    )
    families = _filter(base_config["families"], args.families)
    budgets = _filter(base_config["budgets"], args.budgets, int)

    from confirmatory_artifacts import build_run_manifest
    from downstream_benchmark import _write_json, ensure_protocol_manifest

    code_paths = [
        Path(__file__), Path(__file__).with_name("downstream_benchmark.py"),
        Path(__file__).with_name("downstream_protocol.py"),
        Path(__file__).with_name("downstream_training.py"),
        Path(__file__).with_name("downstream_metrics.py"),
        Path(__file__).with_name("confirmatory_models.py"),
        args.base_config, args.extension_config,
    ]
    run_config = {
        "protocol": "confirmatory_seed_extension",
        "extension": extension,
    }
    manifest = build_run_manifest(run_config, sys.argv, code_paths)
    args.output_root.mkdir(parents=True, exist_ok=True)
    ensure_protocol_manifest(args.output_root / "run_manifest.json", manifest)
    _write_json(
        args.output_root / "execution_manifests"
        / f"{_slug(socket.gethostname())}_{os.getpid()}.json",
        manifest,
    )
    for task in tasks:
        for family in families:
            for budget in budgets:
                evaluate_seed_extension(
                    args.base_config, args.extension_config, args.base_root,
                    args.output_root, task, family, int(budget), args.raw_root,
                )


if __name__ == "__main__":
    main()

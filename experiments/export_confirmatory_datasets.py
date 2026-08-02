"""Export deterministic confirmatory task instances with file-level checksums."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from confirmatory_protocol import make_analytic_bundle, make_energy_bundle, make_nmr_bundle


ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(task: str, seed: int, dataset: dict):
    if task.startswith("nmr_"):
        return make_nmr_bundle(
            task.removeprefix("nmr_"), dataset["nmr_observed"], dataset["nmr_noise"],
            seed, dataset["nmr_dense"],
        )
    if task == "energy":
        return make_energy_bundle(seed)
    return make_analytic_bundle(
        task, seed, dataset["analytic_train"], dataset["analytic_validation"],
        dataset["analytic_test"], dataset["analytic_noise"],
    )


def export_datasets(config_path: Path, output: Path) -> dict:
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    tasks = config["primary_tasks"] + config["supplementary_tasks"]
    seeds = sorted(set(config["tuning_seeds"] + config["evaluation_data_seeds"]))
    files = []
    instances = []
    arrays = ("x_train", "y_train", "x_validation", "y_validation", "x_test", "y_test")
    for task in tasks:
        for seed in seeds:
            bundle = _bundle(task, int(seed), config["dataset"])
            directory = output / task / f"seed-{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            for name in arrays:
                path = directory / f"{name}.npy"
                np.save(path, np.asarray(getattr(bundle, name)), allow_pickle=False)
                files.append({
                    "path": path.relative_to(output).as_posix(),
                    "sha256": _sha256(path),
                    "shape": list(getattr(bundle, name).shape),
                    "dtype": str(getattr(bundle, name).dtype),
                })
            instances.append({"task": task, "data_seed": int(seed), "metadata": bundle.metadata})
    manifest = {
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "generator_sources": {
            path.name: _sha256(path)
            for path in (
                Path(__file__), ROOT / "confirmatory_protocol.py", ROOT / "nmr_data.py",
                ROOT / "energy_efficiency.csv",
            )
        },
        "task_instance_count": len(instances),
        "instances": instances,
        "files": files,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "confirmatory_config.json"))
    parser.add_argument("--output", default=str(ROOT / "confirmatory_datasets"))
    args = parser.parse_args()
    manifest = export_datasets(Path(args.config), Path(args.output))
    print(f"exported {manifest['task_instance_count']} task instances")


if __name__ == "__main__":
    main()

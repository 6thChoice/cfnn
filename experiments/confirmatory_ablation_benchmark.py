"""Ablation-only entry point with an explicit fixed-offset PARN reference."""
from __future__ import annotations

from pathlib import Path


def register_ablation_families() -> None:
    import confirmatory_models

    confirmatory_models.PARN_VARIANTS["PARN-epsilon-0.1"] = {"epsilon": 0.1}


def ablation_code_paths(code_paths) -> list[Path]:
    return [*code_paths, Path(__file__)]


def main() -> None:
    register_ablation_families()
    import confirmatory_benchmark

    base_manifest_builder = confirmatory_benchmark.build_run_manifest

    def build_run_manifest_with_entrypoint(config, command, code_paths):
        return base_manifest_builder(config, command, ablation_code_paths(code_paths))

    confirmatory_benchmark.build_run_manifest = build_run_manifest_with_entrypoint
    confirmatory_benchmark.main()


if __name__ == "__main__":
    main()

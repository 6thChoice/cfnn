#!/usr/bin/env python3
"""Build an EIS applicability map starter for Package E."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import eis_utils as U
from ai4science_eis_boundary_e import (
    eis_spectral_descriptors,
    grouped_loss_effects,
    layered_loss_effect_rows,
    loss_effect_row,
    summarize_effects,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "ai4science_results" / "eis_e_applicability_map_starter"
F_MIN, F_MAX = 1e-2, 1e5
TWO_TC_BASE = dict(Rs=10.0, R1=300.0, C1=1e-4, R2=800.0, C2=1e-6)
NONRATIONAL_CASES = {
    "randles_warburg": dict(Rs=10.0, Rct=400.0, Cdl=1e-5, sigma=80.0),
    "randles_cpe": dict(Rs=10.0, Rct=400.0, Q=1e-5, alpha=0.75),
}
NEURAL_BASELINES = {"MLP", "MLP-matched", "MLP-large", "KAN"}
PHYSICAL_PRIOR_BASELINES = {"Classical-NLSQ", "Classical-fixed2", "Classical-AIC", "Classical-oracle"}


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "NA"
    number = float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _load_json(path: Path):
    return json.loads(Path(path).read_text())


def _two_tc_descriptors() -> dict:
    frequency = U.freq_grid(F_MIN, F_MAX, 400)
    z = U.two_tc_impedance(frequency, **TWO_TC_BASE)
    return eis_spectral_descriptors(
        "two_tc_hpo",
        frequency,
        z,
        metadata={
            "task_family": "two_tc_hpo",
            "known_pole_count": 2,
            "distributed_relaxation": False,
            "n_points": 80,
            "noise": 0.02,
        },
    )


def _voigt_descriptors(K: int, n_points: int, noise: float, seed: int = 42) -> dict:
    ds = U.make_voigt_dataset(int(K), F_MIN, F_MAX, int(n_points), float(noise), int(seed), "log")
    dense = U.freq_grid(F_MIN, F_MAX, 400)
    z = U.voigt_impedance(dense, ds["params"]["Rs"], ds["params"]["R_list"], ds["params"]["C_list"])
    return eis_spectral_descriptors(
        f"voigt_K{K}_n{n_points}_noise{noise:g}",
        dense,
        z,
        metadata={
            "task_family": "voigt_phaseB",
            "known_pole_count": int(K),
            "distributed_relaxation": False,
            "n_points": int(n_points),
            "noise": float(noise),
        },
    )


def _nonrational_descriptors(circuit: str) -> dict:
    frequency = U.freq_grid(F_MIN, F_MAX, 400)
    z = U.CIRCUITS[circuit]["func"](frequency, **NONRATIONAL_CASES[circuit])
    return eis_spectral_descriptors(
        circuit,
        frequency,
        z,
        metadata={
            "task_family": "nonrational_E5",
            "known_pole_count": 1,
            "distributed_relaxation": True,
            "n_points": 80,
            "noise": 0.02,
        },
    )


def _best_hpo_effect(hpo_summary: list[dict]) -> dict:
    rows = [
        {
            "family": row["family"],
            "model": row["family"],
            "params": row["params"],
            "pole_err_median": row["pole_err_median"],
        }
        for row in hpo_summary
    ]
    effect = loss_effect_row(
        rows=rows,
        metric="pole_err_median",
        cfnn_models={"CFNN"},
        baseline_filter=lambda row: row.get("family") in {"MLP", "KAN"},
    )
    effect.update({"task_id": "two_tc_hpo", "source": "eis_hpo_results/hpo_summary.json"})
    return effect


def _best_hpo_layered_effect(hpo_summary: list[dict]) -> dict:
    rows = [
        {
            "family": row["family"],
            "model": row["family"],
            "params": row["params"],
            "pole_err_median": row["pole_err_median"],
        }
        for row in hpo_summary
    ]
    effects = layered_loss_effect_rows(
        rows=rows,
        metric="pole_err_median",
        cfnn_models={"CFNN"},
        neural_baseline_models={"MLP", "KAN"},
        physical_prior_models=set(),
    )
    for layer, effect in effects.items():
        effect.update({"task_id": "two_tc_hpo", "source": "eis_hpo_results/hpo_summary.json", "effect_layer": layer})
    return effects


def _phaseb_effects(raw_rows: list[dict], *, exp: str) -> list[dict]:
    return grouped_loss_effects(
        [row for row in raw_rows if row.get("exp") == exp],
        group_keys=("K", "n_points", "noise"),
        metric="curve_mse",
        cfnn_models={"CFNN-Hybrid"},
        baseline_filter=lambda row: row.get("model") in {
            "MLP-matched",
            "KAN",
            "Classical-fixed2",
            "Classical-AIC",
            "Classical-oracle",
        },
    )


def _phaseb_layered_effects(raw_rows: list[dict], *, exp: str) -> list[dict]:
    rows = [row for row in raw_rows if row.get("exp") == exp]
    grouped = {}
    for row in rows:
        grouped.setdefault((row.get("K"), row.get("n_points"), row.get("noise")), []).append(row)
    effects = []
    for key_values, items in sorted(grouped.items()):
        layered = layered_loss_effect_rows(
            rows=items,
            metric="curve_mse",
            cfnn_models={"CFNN-Hybrid"},
            neural_baseline_models={"MLP-matched", "KAN"},
            physical_prior_models={"Classical-fixed2", "Classical-AIC", "Classical-oracle"},
        )
        for layer, effect in layered.items():
            effect.update({
                "K": key_values[0],
                "n_points": key_values[1],
                "noise": key_values[2],
                "row_count": len(items),
                "effect_layer": layer,
            })
            effects.append(effect)
    return effects


def _e5_effects(raw_rows: list[dict]) -> list[dict]:
    return grouped_loss_effects(
        [row for row in raw_rows if row.get("exp") == "E5"],
        group_keys=("circuit",),
        metric="curve_mse",
        cfnn_models={"CFNN-Hybrid"},
        baseline_filter=lambda row: row.get("model") in {
            "MLP-matched",
            "MLP-large",
            "KAN",
            "Classical-NLSQ",
        },
    )


def _e5_layered_effects(raw_rows: list[dict]) -> list[dict]:
    rows = [row for row in raw_rows if row.get("exp") == "E5"]
    grouped = {}
    for row in rows:
        grouped.setdefault((row.get("circuit"),), []).append(row)
    effects = []
    for key_values, items in sorted(grouped.items()):
        layered = layered_loss_effect_rows(
            rows=items,
            metric="curve_mse",
            cfnn_models={"CFNN-Hybrid"},
            neural_baseline_models={"MLP-matched", "MLP-large", "KAN"},
            physical_prior_models={"Classical-NLSQ"},
        )
        for layer, effect in layered.items():
            effect.update({
                "circuit": key_values[0],
                "row_count": len(items),
                "effect_layer": layer,
            })
            effects.append(effect)
    return effects


def build_report(summary: dict) -> str:
    lines = [
        "# EIS Package E Applicability Map Starter",
        "",
        "This report converts existing EIS runs into a descriptor/effect table. Positive signed effect means CFNN has lower loss than the strongest non-CFNN baseline in that cell; negative means the strongest baseline is better.",
        "",
        "## Effect Summary",
        "",
        "| source | layer | cells | median signed CFNN effect | positive | negative |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for source, layers in summary["layered_effect_summaries"].items():
        for layer, row in layers.items():
            lines.append(
                f"| {source} | {layer} | {row['effect_count']} | {_fmt(row['median_signed_cfnn_effect'])} | {row['positive_effect_count']} | {row['negative_effect_count']} |"
            )
    lines.extend([
        "",
        "## Layered Interpretation",
        "",
        "`neural_only` compares CFNN only against MLP/KAN-style neural baselines. `physical_prior_only` compares CFNN against explicit EIS circuit priors such as Classical-AIC or oracle fits. This separation prevents the physical-prior result from being misread as a generic neural-network failure.",
        "",
        "## Representative Cells",
        "",
        "| task | descriptor | metric | CFNN | best baseline | signed effect | peaks | poles | distributed |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in summary["effect_descriptor_rows"]:
        descriptor = row.get("descriptor", {})
        lines.append(
            "| "
            + " | ".join([
                f"`{row['task_id']}`",
                f"`{descriptor.get('spectrum_id', 'NA')}`",
                row.get("metric", "NA"),
                _fmt(row.get("cfnn_metric")),
                f"{row.get('best_baseline_model')} / {_fmt(row.get('best_baseline_metric'))}",
                _fmt(row.get("signed_cfnn_effect")),
                str(descriptor.get("relaxation_peak_count")),
                str(descriptor.get("known_pole_count")),
                str(descriptor.get("distributed_relaxation")),
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The HPO-revalidated two-time-constant EIS task does not support a broad CFNN dominance claim: after tuning, the strongest MLP/KAN baseline is competitive or better. Phase B also shows that classical circuit priors often dominate curve-MSE when the generative family is known or selectable by AIC. This is consistent with the planned Package E framing: EIS is a boundary task where rational neural bias is not sufficient by itself, especially when spectra are broad-band, multi-timescale, or distributed-relaxation dominated.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpo-summary", type=Path, default=ROOT / "eis_hpo_results" / "hpo_summary.json")
    parser.add_argument("--phaseb-b1a", type=Path, default=ROOT / "eis_phaseB_results" / "B1a_order_mismatch_raw.json")
    parser.add_argument("--phaseb-b1b", type=Path, default=ROOT / "eis_phaseB_results" / "B1b_illposed_raw.json")
    parser.add_argument("--e5-raw", type=Path, default=ROOT / "eis_pilot_results" / "E5_nonrational_raw.json")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hpo_summary = _load_json(args.hpo_summary)
    b1a_raw = _load_json(args.phaseb_b1a)
    b1b_raw = _load_json(args.phaseb_b1b)
    e5_raw = _load_json(args.e5_raw)
    hpo_effect = _best_hpo_effect(hpo_summary)
    hpo_layered = _best_hpo_layered_effect(hpo_summary)
    b1a_effects = _phaseb_effects(b1a_raw, exp="B1a")
    b1b_effects = _phaseb_effects(b1b_raw, exp="B1b")
    e5_effects = _e5_effects(e5_raw)
    b1a_layered = _phaseb_layered_effects(b1a_raw, exp="B1a")
    b1b_layered = _phaseb_layered_effects(b1b_raw, exp="B1b")
    e5_layered = _e5_layered_effects(e5_raw)

    descriptors = {"two_tc_hpo": _two_tc_descriptors()}
    for effect in b1a_effects + b1b_effects:
        if effect.get("status") != "ok":
            continue
        key = f"voigt_K{effect['K']}_n{effect['n_points']}_noise{float(effect['noise']):g}"
        if key not in descriptors:
            descriptors[key] = _voigt_descriptors(effect["K"], effect["n_points"], effect["noise"])
    for circuit in NONRATIONAL_CASES:
        descriptors[circuit] = _nonrational_descriptors(circuit)

    effect_descriptor_rows = []
    layered_effect_descriptor_rows = []
    hpo_effect["task_id"] = "two_tc_hpo"
    effect_descriptor_rows.append({**hpo_effect, "descriptor": descriptors["two_tc_hpo"]})
    for effect in hpo_layered.values():
        layered_effect_descriptor_rows.append({**effect, "descriptor": descriptors["two_tc_hpo"]})
    for effect in b1a_effects + b1b_effects:
        if effect.get("status") != "ok":
            continue
        task_id = f"voigt_K{effect['K']}_n{effect['n_points']}_noise{float(effect['noise']):g}"
        effect_descriptor_rows.append({**effect, "task_id": task_id, "descriptor": descriptors[task_id]})
    for effect in b1a_layered + b1b_layered:
        if effect.get("status") != "ok":
            continue
        task_id = f"voigt_K{effect['K']}_n{effect['n_points']}_noise{float(effect['noise']):g}"
        layered_effect_descriptor_rows.append({**effect, "task_id": task_id, "descriptor": descriptors[task_id]})
    for effect in e5_effects:
        if effect.get("status") != "ok":
            continue
        task_id = str(effect["circuit"])
        effect_descriptor_rows.append({**effect, "task_id": task_id, "descriptor": descriptors[task_id]})
    for effect in e5_layered:
        if effect.get("status") != "ok":
            continue
        task_id = str(effect["circuit"])
        layered_effect_descriptor_rows.append({**effect, "task_id": task_id, "descriptor": descriptors[task_id]})

    summary = {
        "analysis_type": "eis_e_applicability_map_starter",
        "descriptor_count": len(descriptors),
        "effect_summaries": {
            "hpo_two_tc": summarize_effects([hpo_effect]),
            "phaseB_B1a": summarize_effects(b1a_effects),
            "phaseB_B1b": summarize_effects(b1b_effects),
            "E5_nonrational": summarize_effects(e5_effects),
            "all_cells": summarize_effects(effect_descriptor_rows),
        },
        "layered_effect_summaries": {
            "hpo_two_tc": {
                layer: summarize_effects([effect])
                for layer, effect in hpo_layered.items()
            },
            "phaseB_B1a": {
                layer: summarize_effects([effect for effect in b1a_layered if effect.get("effect_layer") == layer])
                for layer in ("all_baselines", "neural_only", "physical_prior_only")
            },
            "phaseB_B1b": {
                layer: summarize_effects([effect for effect in b1b_layered if effect.get("effect_layer") == layer])
                for layer in ("all_baselines", "neural_only", "physical_prior_only")
            },
            "E5_nonrational": {
                layer: summarize_effects([effect for effect in e5_layered if effect.get("effect_layer") == layer])
                for layer in ("all_baselines", "neural_only", "physical_prior_only")
            },
            "all_cells": {
                layer: summarize_effects([effect for effect in layered_effect_descriptor_rows if effect.get("effect_layer") == layer])
                for layer in ("all_baselines", "neural_only", "physical_prior_only")
            },
        },
        "descriptor_rows": list(descriptors.values()),
        "effect_descriptor_rows": effect_descriptor_rows,
        "layered_effect_descriptor_rows": layered_effect_descriptor_rows,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (args.output_root / "REPORT.md").write_text(build_report(summary) + "\n")
    print(json.dumps({
        "output_root": str(args.output_root),
        "descriptor_count": len(descriptors),
        "effect_row_count": len(effect_descriptor_rows),
        "layered_effect_row_count": len(layered_effect_descriptor_rows),
        "median_signed_effect_all": summary["effect_summaries"]["all_cells"]["median_signed_cfnn_effect"],
        "median_signed_effect_neural_only": summary["layered_effect_summaries"]["all_cells"]["neural_only"]["median_signed_cfnn_effect"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

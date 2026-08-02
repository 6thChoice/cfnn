#!/usr/bin/env python3
"""Build Package E external EIS spectrum boundary audit skeleton."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ai4science_eis_boundary_e import (
    audit_eis_neural_positive_edges,
    eis_descriptor_stratum_summary,
    external_spectrum_boundary_gate,
    hsc_condition_descriptor_rows,
    hsc_neural_metric_rows,
    hsc_same_split_layered_effect_rows as build_hsc_same_split_layered_effect_rows,
    hsc_same_split_voigt_prior_rows,
    hsc_voigt_physical_prior_rows,
)
from peak_sensitive_benchmark import make_peak_sensitive_bundle
from peak_sensitive_protocol import load_hybrid_supercapacitor_eis


ROOT = Path(__file__).resolve().parent
DEFAULT_EIS_SUMMARY = ROOT / "ai4science_results" / "eis_e_applicability_map_starter" / "summary.json"
DEFAULT_OUTPUT_ROOT = ROOT / "ai4science_results" / "eis_external_spectrum_boundary"
DEFAULT_PEAK_SENSITIVE_ROOT = ROOT / "peak_sensitive_results"
DEFAULT_PEAK_SENSITIVE_CONFIG = ROOT / "peak_sensitive_config.json"


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


def _median(values) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def load_hsc_frozen_neural_records(peak_sensitive_root: Path) -> list[dict]:
    rows = []
    evaluation_root = Path(peak_sensitive_root) / "evaluation" / "hybrid_supercapacitor_eis"
    for path in sorted(evaluation_root.glob("*.jsonl")):
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise FileNotFoundError(evaluation_root)
    return rows


def hsc_same_split_bundles_from_records(records: list[dict], config: dict, raw_root: Path) -> list[object]:
    keys = sorted({
        int(record["data_seed"])
        for record in records
        if record.get("task") == "hybrid_supercapacitor_eis" and record.get("data_seed") is not None
    })
    bundles = []
    for data_seed in keys:
        bundle = make_peak_sensitive_bundle(
            "hybrid_supercapacitor_eis",
            data_seed,
            config,
            raw_root,
            scientific_role="evaluation",
            visibility="opened",
        )
        bundles.append(bundle)
    return bundles


def build_report(summary: dict) -> str:
    gate = summary.get("gate", {})
    same_split_audit = summary.get("hsc_same_split_audit", {})
    same_split_status_counts = same_split_audit.get("status_counts", {})
    lines = [
        "# EIS External Spectrum Boundary Audit",
        "",
        "This audit is an executable skeleton for Package E external-spectrum validation. It does not launch new neural training; it checks whether current layered EIS evidence remains a physical-prior boundary before broader spectra are added.",
        "",
        "## Gate",
        "",
        f"- Claim status: `{gate.get('claim_status')}`",
        f"- Neural-positive edge count: `{gate.get('neural_positive_edge_count')}`",
        f"- Physical-prior eliminated edge count: `{gate.get('physical_prior_eliminated_edge_count')}`",
        f"- Physical-prior elimination rate: `{_fmt(gate.get('physical_prior_elimination_rate'))}`",
        f"- All-baseline surviving positive edge count: `{gate.get('all_baseline_surviving_positive_edge_count')}`",
        f"- Median all-baseline signed CFNN effect: `{_fmt(gate.get('median_all_baseline_signed_cfnn_effect'))}`",
        "",
        "## HSC Condition Descriptors",
        "",
        f"- HSC status: `{summary.get('hsc_status')}`",
        f"- HSC descriptor rows: `{summary.get('hsc_descriptor_row_count')}`",
        f"- HSC physical-prior rows: `{summary.get('hsc_physical_prior_row_count')}`",
        f"- HSC AIC median curve MSE: `{_fmt(summary.get('hsc_aic_median_curve_mse'))}`",
        f"- HSC AIC median complex R2: `{_fmt(summary.get('hsc_aic_median_complex_r2'))}`",
        f"- HSC error: `{summary.get('hsc_error') or 'NA'}`",
        "",
        "## HSC Same-Split Physical-Prior Alignment",
        "",
        f"- HSC same-split status: `{summary.get('hsc_same_split_status')}`",
        f"- Frozen neural records: `{summary.get('hsc_frozen_neural_record_count')}`",
        f"- Neural metric rows: `{summary.get('hsc_neural_metric_row_count')}`",
        f"- Same-split physical-prior rows: `{summary.get('hsc_same_split_physical_prior_row_count')}`",
        f"- Same-split layered effect rows: `{summary.get('hsc_same_split_layered_effect_row_count')}`",
        f"- Same-split AIC median global complex NRMSE: `{_fmt(summary.get('hsc_same_split_aic_median_global_complex_nrmse'))}`",
        f"- Same-split neural-positive edges: `{same_split_audit.get('positive_neural_edge_count')}`",
        f"- Same-split physical-prior eliminated edges: `{same_split_status_counts.get('physical_prior_eliminates_neural_edge', 0)}`",
        f"- HSC same-split error: `{summary.get('hsc_same_split_error') or 'NA'}`",
        "",
        "## Descriptor Strata",
        "",
        "| stratum | descriptors | median poles | median relaxation peaks | distributed count |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.get("descriptor_strata", []):
        lines.append(
            "| "
            + " | ".join([
                f"`{row.get('descriptor_stratum')}`",
                str(row.get("descriptor_count")),
                _fmt(row.get("median_known_pole_count")),
                _fmt(row.get("median_relaxation_peak_count")),
                str(row.get("distributed_relaxation_count")),
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Neural-only positive edges are not structural CFNN advantages unless they also survive physical-prior and all-baseline layers inside a predefined descriptor stratum. The current skeleton preserves the EIS boundary claim and defines the gate that broader synthetic spectra and HSC condition curves must pass through.",
        "",
        "For HSC, the same-split physical-prior layer is the comparable bridge: Voigt models are fitted only on the frozen train+validation frequencies and scored on the same held-out frequencies used by neural evaluation. This prevents full-curve physical fits from being mistaken for fair neural baselines.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eis-summary", type=Path, default=DEFAULT_EIS_SUMMARY)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "downstream_data" / "raw")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--peak-sensitive-root", type=Path, default=DEFAULT_PEAK_SENSITIVE_ROOT)
    parser.add_argument("--peak-sensitive-config", type=Path, default=DEFAULT_PEAK_SENSITIVE_CONFIG)
    parser.add_argument("--elimination-threshold", type=float, default=0.80)
    parser.add_argument("--hsc-voigt-max-order", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.eis_summary.read_text())
    descriptor_rows = list(source.get("descriptor_rows", []))
    layered_rows = list(source.get("layered_effect_descriptor_rows", []))
    try:
        hsc_curves = load_hybrid_supercapacitor_eis(args.raw_root)
        hsc_descriptors = hsc_condition_descriptor_rows(hsc_curves)
        hsc_physical_prior_rows = hsc_voigt_physical_prior_rows(
            hsc_curves,
            max_order=int(args.hsc_voigt_max_order),
        )
        hsc_status = "ok"
        hsc_error = None
    except (FileNotFoundError, ValueError) as exc:
        hsc_descriptors = []
        hsc_physical_prior_rows = []
        hsc_status = "unavailable"
        hsc_error = str(exc)
    try:
        hsc_frozen_neural_records = load_hsc_frozen_neural_records(args.peak_sensitive_root)
        hsc_neural_rows = hsc_neural_metric_rows(
            hsc_frozen_neural_records,
            metric="global_complex_nrmse",
        )
        peak_config = json.loads(args.peak_sensitive_config.read_text())
        hsc_same_split_bundles = hsc_same_split_bundles_from_records(
            hsc_frozen_neural_records,
            peak_config,
            args.raw_root,
        )
        hsc_same_split_physical_prior_rows = hsc_same_split_voigt_prior_rows(
            hsc_same_split_bundles,
            max_order=int(args.hsc_voigt_max_order),
        )
        hsc_same_split_layered_effect_rows = build_hsc_same_split_layered_effect_rows(
            neural_rows=hsc_neural_rows,
            physical_rows=hsc_same_split_physical_prior_rows,
            metric="global_complex_nrmse",
        )
        hsc_same_split_audit = audit_eis_neural_positive_edges(
            hsc_same_split_layered_effect_rows
        )
        hsc_same_split_status = "ok"
        hsc_same_split_error = None
    except (FileNotFoundError, ValueError, KeyError) as exc:
        hsc_frozen_neural_records = []
        hsc_neural_rows = []
        hsc_same_split_physical_prior_rows = []
        hsc_same_split_layered_effect_rows = []
        hsc_same_split_audit = {
            "analysis_type": "eis_neural_positive_edge_audit",
            "positive_neural_edge_count": 0,
            "status_counts": {},
            "positive_edge_rows": [],
        }
        hsc_same_split_status = "unavailable"
        hsc_same_split_error = str(exc)
    combined_descriptors = descriptor_rows + hsc_descriptors
    gate = external_spectrum_boundary_gate(
        layered_rows,
        combined_descriptors,
        elimination_threshold=float(args.elimination_threshold),
    )
    summary = {
        "analysis_type": "eis_external_spectrum_boundary",
        "source_eis_summary": str(args.eis_summary),
        "source_hsc_raw_root": str(args.raw_root),
        "input_layered_effect_row_count": int(len(layered_rows)),
        "input_descriptor_row_count": int(len(descriptor_rows)),
        "hsc_status": hsc_status,
        "hsc_error": hsc_error,
        "hsc_descriptor_row_count": int(len(hsc_descriptors)),
        "hsc_descriptor_rows": hsc_descriptors,
        "hsc_descriptor_strata": eis_descriptor_stratum_summary(hsc_descriptors),
        "hsc_physical_prior_row_count": int(len(hsc_physical_prior_rows)),
        "hsc_physical_prior_rows": hsc_physical_prior_rows,
        "hsc_aic_rows": [
            row for row in hsc_physical_prior_rows
            if row.get("model") == "Classical-AIC"
        ],
        "hsc_aic_median_curve_mse": _median(
            row.get("curve_mse") for row in hsc_physical_prior_rows
            if row.get("model") == "Classical-AIC"
        ),
        "hsc_aic_median_complex_r2": _median(
            row.get("complex_r2") for row in hsc_physical_prior_rows
            if row.get("model") == "Classical-AIC"
        ),
        "hsc_same_split_status": hsc_same_split_status,
        "hsc_same_split_error": hsc_same_split_error,
        "hsc_frozen_neural_record_count": int(len(hsc_frozen_neural_records)),
        "hsc_neural_metric_row_count": int(len(hsc_neural_rows)),
        "hsc_neural_metric_rows": hsc_neural_rows,
        "hsc_same_split_physical_prior_row_count": int(len(hsc_same_split_physical_prior_rows)),
        "hsc_same_split_physical_prior_rows": hsc_same_split_physical_prior_rows,
        "hsc_same_split_layered_effect_row_count": int(len(hsc_same_split_layered_effect_rows)),
        "hsc_same_split_layered_effect_rows": hsc_same_split_layered_effect_rows,
        "hsc_same_split_audit": hsc_same_split_audit,
        "hsc_same_split_aic_median_global_complex_nrmse": _median(
            row.get("global_complex_nrmse") for row in hsc_same_split_physical_prior_rows
            if row.get("model") == "Classical-AIC-SameSplit"
        ),
        "gate": gate,
        "descriptor_strata": eis_descriptor_stratum_summary(combined_descriptors),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (args.output_root / "REPORT.md").write_text(build_report(summary) + "\n")
    print(json.dumps({
        "output_root": str(args.output_root),
        "claim_status": gate.get("claim_status"),
        "hsc_descriptor_row_count": len(hsc_descriptors),
        "hsc_physical_prior_row_count": len(hsc_physical_prior_rows),
        "hsc_status": hsc_status,
        "hsc_same_split_status": hsc_same_split_status,
        "hsc_same_split_layered_effect_row_count": len(hsc_same_split_layered_effect_rows),
        "neural_positive_edge_count": gate.get("neural_positive_edge_count"),
        "physical_prior_elimination_rate": gate.get("physical_prior_elimination_rate"),
        "all_baseline_surviving_positive_edge_count": gate.get("all_baseline_surviving_positive_edge_count"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Run AI4Science A1 reference fitting for Microwave Fano curves."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ai4science_fano import (
    FanoReference,
    build_microwave_fano_reference_catalog,
    references_to_csv_rows,
)


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit independent complex Fano reference parameters for Microwave Fano curves."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "downstream_data" / "raw",
        help="Root containing downstream raw data.",
    )
    parser.add_argument(
        "--role-manifest",
        type=Path,
        default=ROOT / "peak_sensitive_results" / "provenance" / "microwave_fano_role_manifest.json",
        help="Frozen Microwave Fano role manifest.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "ai4science_results" / "microwave_fano_a1",
        help="Output directory for A1 reference artifacts.",
    )
    parser.add_argument(
        "--all-curves",
        action="store_true",
        help="Fit every released Microwave Fano curve instead of only frozen selected curves.",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=0.05,
        help="Maximum relative magnitude RMSE for status=ok.",
    )
    return parser.parse_args()


def write_outputs(output_root: Path, references: list[FanoReference]) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "reference_parameters.json"
    csv_path = output_root / "reference_parameters.csv"
    records = [reference.to_dict() for reference in references]
    json_path.write_text(json.dumps(records, indent=2, sort_keys=True, allow_nan=False) + "\n")
    rows = references_to_csv_rows(references)
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("")
    return json_path, csv_path


def main() -> None:
    args = parse_args()
    references = build_microwave_fano_reference_catalog(
        args.raw_root,
        args.role_manifest,
        selected_only=not args.all_curves,
        residual_threshold=args.residual_threshold,
    )
    records = [reference.to_dict() for reference in references]
    json_path, csv_path = write_outputs(args.output_root, references)
    ok_count = sum(1 for record in records if record["status"] == "ok")
    high_residual_count = sum(1 for record in records if record["status"] == "high_residual")
    rejected_count = sum(1 for record in records if record["status"] == "unidentifiable")
    summary = {
        "curve_count": len(records),
        "ok_count": ok_count,
        "high_residual_count": high_residual_count,
        "unidentifiable_count": rejected_count,
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

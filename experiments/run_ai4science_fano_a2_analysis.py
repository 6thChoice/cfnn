"""Build enhanced summaries for Microwave Fano A2 result CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai4science_fano_a2_analysis import write_enhanced_summary


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate A2 parameter_errors.csv with paired metrics.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=ROOT / "ai4science_results" / "microwave_fano_a2_pilot" / "parameter_errors.csv",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--baseline-family", default="CFNN")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_json = args.output_json
    if output_json is None:
        output_json = args.input_csv.parent / "enhanced_summary.json"
    summary = write_enhanced_summary(
        args.input_csv,
        output_json,
        baseline_family=args.baseline_family,
    )
    print(json.dumps({
        "input_csv": str(args.input_csv),
        "output_json": str(output_json),
        "record_count": summary["record_count"],
        "families": summary["families"],
        "observation_budgets": summary["observation_budgets"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

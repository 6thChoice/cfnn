#!/usr/bin/env python3
"""Build Microstrip low-calibration event-transfer consolidation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ai4science_microstrip_event_transfer import microstrip_low_calibration_event_transfer


DEFAULT_INPUT = ROOT / "ai4science_results" / "microstrip_b2_confirmatory_guardrail_strata" / "summary.json"
DEFAULT_OUT = ROOT / "ai4science_results" / "microstrip_low_calibration_event_transfer"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def build_report(summary: dict) -> str:
    gate = summary["claim_gate"]
    return "\n".join([
        "# Microstrip Low-Calibration Event Transfer",
        "",
        f"- claim status: `{gate['claim_status']}`",
        f"- localized transfer passing rows: `{gate['localized_transfer_pass_count']}`",
        f"- guardrail-blocked Q rows: `{gate['guardrail_blocked_q_count']}`",
        f"- digital twin status: `{gate['digital_twin_status']}`",
        "",
        "This artifact reports Q/linewidth transfer as a localized event signal, not a full device digital twin.",
    ]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-summary", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    source = json.loads(args.input_summary.read_text())
    summary = microstrip_low_calibration_event_transfer(source)
    _write_json(args.output_root / "summary.json", summary)
    (args.output_root / "REPORT.md").write_text(build_report(summary))
    print(json.dumps({
        "analysis_type": summary["analysis_type"],
        "claim_status": summary["claim_gate"]["claim_status"],
        "output_root": str(args.output_root),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

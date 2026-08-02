#!/usr/bin/env python3
"""Build Microwave Fano event-focused consolidation from existing artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ai4science_fano_event_focus import fano_event_focus_consolidation


DEFAULT_A2 = ROOT / "ai4science_results" / "microwave_fano_a2_confirmatory_64_128_queue" / "enhanced_summary.json"
DEFAULT_OUT = ROOT / "ai4science_results" / "microwave_fano_event_focused_consolidation"
DEFAULT_HIGHRISK = ["undercoupled:r8:p03"]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def build_report(summary: dict) -> str:
    gate = summary["claim_gate"]
    lines = [
        "# Microwave Fano Event-Focused Consolidation",
        "",
        f"- claim status: `{gate['claim_status']}`",
        f"- observation budget: `{gate['observation_budget']}`",
        f"- full inversion status: `{gate['full_inversion_status']}`",
        f"- high-risk curves: `{', '.join(gate['highrisk_curve_ids'])}`",
        "",
        "This artifact narrows Fano reporting to peak-window and f0 event recovery unless q_abs_error passes separately.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2-summary", type=Path, default=DEFAULT_A2)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--budget", type=int, default=128)
    args = parser.parse_args()

    a2_summary = json.loads(args.a2_summary.read_text())
    summary = fano_event_focus_consolidation(
        a2_summary,
        budget=args.budget,
        highrisk_curve_ids=DEFAULT_HIGHRISK,
    )
    _write_json(args.output_root / "summary.json", summary)
    (args.output_root / "REPORT.md").write_text(build_report(summary))
    print(json.dumps({
        "analysis_type": summary["analysis_type"],
        "claim_status": summary["claim_gate"]["claim_status"],
        "output_root": str(args.output_root),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

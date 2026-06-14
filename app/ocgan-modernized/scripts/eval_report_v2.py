#!/usr/bin/env python3
"""Eval report for v2 (optv2) retrain. Same plots/CSV as eval_report.py
but reading optv2 CSVs and writing to eval_report_v2/."""
from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval_report as er

ROOT = Path(__file__).resolve().parents[1]
er.RUNS_CSV = ROOT / "optv2_multiseed_runs.csv"
er.AGG_CSV = ROOT / "optv2_multiseed_aggregated.csv"
er.OUTPUT_DIR = ROOT / "eval_report_v2"

if __name__ == "__main__":
    er.main()

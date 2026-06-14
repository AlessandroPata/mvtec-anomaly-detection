#!/usr/bin/env python3
"""Parse {cat}_optv2_s{seed}_* runs → runs CSV + aggregated CSV.

Output:
  optv2_multiseed_runs.csv       (same schema as final_per_category_multiseed_runs.csv)
  optv2_multiseed_aggregated.csv (same schema as final_per_category_multiseed_aggregated.csv)
"""
from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = Path("/notebooks/storage_project_outputs_datasets/outputs/ocgan-modernized")

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
]

RUN_RE = re.compile(r"^(?P<cat>[a-z_]+)_optv2_s(?P<seed>\d+)_seed\d+_(?P<ts>\d{8}_\d{6})$")


def latest_runs() -> dict[tuple[str, int], Path]:
    """Return {(cat, seed): latest_run_dir}."""
    found: dict[tuple[str, int], Path] = {}
    ts: dict[tuple[str, int], str] = {}
    for p in OUTPUTS.iterdir():
        if not p.is_dir():
            continue
        m = RUN_RE.match(p.name)
        if not m:
            continue
        cat = m.group("cat")
        if cat not in CATEGORIES:
            continue
        seed = int(m.group("seed"))
        key = (cat, seed)
        if key not in ts or m.group("ts") > ts[key]:
            ts[key] = m.group("ts")
            found[key] = p
    return found


def read_final_metrics(run_dir: Path) -> dict | None:
    f = run_dir / "metrics.jsonl"
    if not f.exists():
        return None
    with open(f) as h:
        lines = h.readlines()
    for line in reversed(lines):
        try:
            m = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if m.get("epoch") == "final_test" or "test_blind_auroc" in m or "test_auroc" in m:
            return m
    return None


def read_val_selection(run_dir: Path) -> dict:
    """Return the val metrics from the best (highest val_selection_score) epoch."""
    f = run_dir / "metrics.jsonl"
    out = {"val_selection_score": "", "val_auroc": "", "val_auprc": "",
           "val_best_f1": "", "val_best_threshold": "", "val_fpr_at_95_tpr": ""}
    if not f.exists():
        return out
    best_score = -1.0
    with open(f) as h:
        for line in h:
            try:
                m = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if m.get("epoch") == "final_test":
                continue
            s = m.get("val_selection_score")
            if s is None:
                continue
            if s > best_score:
                best_score = s
                out = {
                    "val_selection_score": s,
                    "val_auroc": m.get("val_mixed_auroc", m.get("val_auroc", "")),
                    "val_auprc": m.get("val_mixed_auprc", m.get("val_auprc", "")),
                    "val_best_f1": m.get("val_mixed_best_f1", m.get("val_best_f1", "")),
                    "val_best_threshold": m.get("val_mixed_best_threshold", m.get("val_best_threshold", "")),
                    "val_fpr_at_95_tpr": m.get("val_mixed_fpr_at_95_tpr", m.get("val_fpr_at_95_tpr", "")),
                }
    return out


def main() -> None:
    runs = latest_runs()

    runs_rows: list[dict] = []
    per_cat: dict[str, list[dict]] = {c: [] for c in CATEGORIES}

    for (cat, seed), path in sorted(runs.items()):
        final = read_final_metrics(path)
        if not final:
            print(f"[warn] no final metrics in {path.name}")
            continue
        val = read_val_selection(path)
        test_auroc = final.get("test_blind_auroc", final.get("test_auroc"))
        test_auprc = final.get("test_blind_auprc", final.get("test_auprc"))
        test_best_f1 = final.get("test_blind_best_f1", final.get("test_best_f1"))
        test_f1_at_thr = final.get("test_blind_f1_at_given_threshold",
                                    final.get("test_f1_at_val_threshold"))
        test_fpr95 = final.get("test_blind_fpr_at_95_tpr", final.get("test_fpr_at_95_tpr"))

        row = {
            "run_name": f"{cat}_optv2_s{seed}",
            "category": cat,
            "seed": seed,
            "val_selection_score": val["val_selection_score"],
            "val_auroc": val["val_auroc"],
            "val_auprc": val["val_auprc"],
            "val_best_f1": val["val_best_f1"],
            "val_best_threshold": val["val_best_threshold"],
            "val_fpr_at_95_tpr": val["val_fpr_at_95_tpr"],
            "test_auroc": test_auroc,
            "test_auprc": test_auprc,
            "test_best_f1": test_best_f1,
            "test_f1_at_val_threshold": test_f1_at_thr,
            "test_fpr_at_95_tpr": test_fpr95,
        }
        runs_rows.append(row)
        per_cat[cat].append(row)

    # Write runs CSV
    runs_csv = ROOT / "optv2_multiseed_runs.csv"
    fieldnames = ["run_name", "category", "seed", "val_selection_score", "val_auroc",
                  "val_auprc", "val_best_f1", "val_best_threshold", "val_fpr_at_95_tpr",
                  "test_auroc", "test_auprc", "test_best_f1", "test_f1_at_val_threshold",
                  "test_fpr_at_95_tpr"]
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(runs_rows)
    print(f"Wrote {runs_csv} ({len(runs_rows)} runs)")

    # Aggregate per category
    def mean(xs): return statistics.mean(xs) if xs else 0.0
    def std(xs): return statistics.pstdev(xs) if len(xs) > 1 else 0.0  # population std to match v1 style? v1 uses sample
    def sstd(xs): return statistics.stdev(xs) if len(xs) > 1 else 0.0

    agg_rows = []
    for cat in CATEGORIES:
        rows = per_cat[cat]
        if not rows:
            print(f"[warn] no v2 runs for category={cat}")
            continue
        aurocs = [r["test_auroc"] for r in rows]
        auprcs = [r["test_auprc"] for r in rows]
        f1s = [r["test_best_f1"] for r in rows]
        f1thrs = [r["test_f1_at_val_threshold"] for r in rows]
        fpr95s = [r["test_fpr_at_95_tpr"] for r in rows]
        best = max(rows, key=lambda r: r["test_auroc"])
        agg_rows.append({
            "category": cat,
            "num_seeds": len(rows),
            "mean_test_auroc": mean(aurocs),
            "std_test_auroc": sstd(aurocs),
            "mean_test_auprc": mean(auprcs),
            "std_test_auprc": sstd(auprcs),
            "mean_test_best_f1": mean(f1s),
            "std_test_best_f1": sstd(f1s),
            "mean_test_f1_at_val_threshold": mean(f1thrs),
            "std_test_f1_at_val_threshold": sstd(f1thrs),
            "mean_test_fpr_at_95_tpr": mean(fpr95s),
            "std_test_fpr_at_95_tpr": sstd(fpr95s),
            "best_run_name": best["run_name"],
            "best_seed": best["seed"],
            "best_test_auroc": best["test_auroc"],
        })

    agg_csv = ROOT / "optv2_multiseed_aggregated.csv"
    with open(agg_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)
    print(f"Wrote {agg_csv} ({len(agg_rows)} categories)")


if __name__ == "__main__":
    main()

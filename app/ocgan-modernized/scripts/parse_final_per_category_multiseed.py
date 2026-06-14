from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

summary_path = Path("final_per_category_multiseed_summary.log")
runs_out_path = Path("final_per_category_multiseed_runs.csv")
agg_out_path = Path("final_per_category_multiseed_aggregated.csv")

run_re = re.compile(r"^===== RUNNING (.+) =====$")
val_re = re.compile(
    r"^\[Validation\] AUROC=([0-9.]+) AUPRC=([0-9.]+) best_F1=([0-9.]+) "
    r"best_threshold=([-0-9.]+) FPR@95TPR=([0-9.]+) selection_score=([0-9.]+)"
)
test_re = re.compile(
    r"^\[Test\] AUROC=([0-9.]+) AUPRC=([0-9.]+) best_F1=([0-9.]+) "
    r"F1@val_threshold=([0-9.]+) FPR@95TPR=([0-9.]+)"
)

rows = []
current = None
best_val = None

for line in summary_path.read_text().splitlines():
    m = run_re.match(line)
    if m:
        if current is not None and best_val is not None:
            rows.append({**current, **best_val})
        current = {"run_name": m.group(1)}
        best_val = None
        continue

    m = val_re.match(line)
    if m and current is not None:
        cand = {
            "val_auroc": float(m.group(1)),
            "val_auprc": float(m.group(2)),
            "val_best_f1": float(m.group(3)),
            "val_best_threshold": float(m.group(4)),
            "val_fpr_at_95_tpr": float(m.group(5)),
            "val_selection_score": float(m.group(6)),
        }
        if best_val is None or cand["val_selection_score"] > best_val["val_selection_score"]:
            best_val = cand
        continue

    m = test_re.match(line)
    if m and current is not None:
        current.update(
            {
                "test_auroc": float(m.group(1)),
                "test_auprc": float(m.group(2)),
                "test_best_f1": float(m.group(3)),
                "test_f1_at_val_threshold": float(m.group(4)),
                "test_fpr_at_95_tpr": float(m.group(5)),
            }
        )

if current is not None and best_val is not None:
    rows.append({**current, **best_val})

for row in rows:
    parts = row["run_name"].split("_")

    if parts[0] == "metal" and parts[1] == "nut":
        row["category"] = "metal_nut"
        seed_part = parts[-1]
    else:
        row["category"] = parts[0]
        seed_part = parts[-1]

    row["seed"] = int(seed_part.replace("s", ""))

run_fieldnames = [
    "run_name",
    "category",
    "seed",
    "val_selection_score",
    "val_auroc",
    "val_auprc",
    "val_best_f1",
    "val_best_threshold",
    "val_fpr_at_95_tpr",
    "test_auroc",
    "test_auprc",
    "test_best_f1",
    "test_f1_at_val_threshold",
    "test_fpr_at_95_tpr",
]

with runs_out_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=run_fieldnames)
    writer.writeheader()
    writer.writerows(rows)

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")

def std(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

grouped: dict[str, list[dict]] = defaultdict(list)
for row in rows:
    grouped[row["category"]].append(row)

agg_rows = []
for category, items in sorted(grouped.items()):
    test_auroc = [x["test_auroc"] for x in items]
    test_auprc = [x["test_auprc"] for x in items]
    test_best_f1 = [x["test_best_f1"] for x in items]
    test_f1_thr = [x["test_f1_at_val_threshold"] for x in items]
    test_fpr95 = [x["test_fpr_at_95_tpr"] for x in items]

    best_item = max(items, key=lambda x: x["test_auroc"])

    agg_rows.append(
        {
            "category": category,
            "num_seeds": len(items),
            "mean_test_auroc": mean(test_auroc),
            "std_test_auroc": std(test_auroc),
            "mean_test_auprc": mean(test_auprc),
            "std_test_auprc": std(test_auprc),
            "mean_test_best_f1": mean(test_best_f1),
            "std_test_best_f1": std(test_best_f1),
            "mean_test_f1_at_val_threshold": mean(test_f1_thr),
            "std_test_f1_at_val_threshold": std(test_f1_thr),
            "mean_test_fpr_at_95_tpr": mean(test_fpr95),
            "std_test_fpr_at_95_tpr": std(test_fpr95),
            "best_run_name": best_item["run_name"],
            "best_seed": best_item["seed"],
            "best_test_auroc": best_item["test_auroc"],
        }
    )

agg_fieldnames = [
    "category",
    "num_seeds",
    "mean_test_auroc",
    "std_test_auroc",
    "mean_test_auprc",
    "std_test_auprc",
    "mean_test_best_f1",
    "std_test_best_f1",
    "mean_test_f1_at_val_threshold",
    "std_test_f1_at_val_threshold",
    "mean_test_fpr_at_95_tpr",
    "std_test_fpr_at_95_tpr",
    "best_run_name",
    "best_seed",
    "best_test_auroc",
]

with agg_out_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=agg_fieldnames)
    writer.writeheader()
    writer.writerows(agg_rows)

print(f"Saved per-run results to {runs_out_path}")
print(f"Saved aggregated results to {agg_out_path}")

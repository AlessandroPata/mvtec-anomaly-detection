from __future__ import annotations

import csv
import re
from pathlib import Path

summary_path = Path("big_grid_seed_memory_teacher_fusion_maxpatch_summary.log")
out_path = Path("big_grid_seed_memory_teacher_fusion_maxpatch_results.csv")

run_re = re.compile(r"^===== RUNNING (.+) =====$")
val_re = re.compile(
    r"^\[Validation\] AUROC=([0-9.]+) AUPRC=([0-9.]+) best_F1=([0-9.]+) "
    r"best_threshold=([-0-9.]+) FPR@95TPR=([0-9.]+)(?: selection_score=([0-9.]+))?"
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
        if current is not None and best_val is not None and "test_auroc" in current:
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
            "val_selection_score": float(m.group(6)) if m.group(6) is not None else float("nan"),
        }
        if best_val is None:
            best_val = cand
        else:
            old = best_val["val_selection_score"]
            new = cand["val_selection_score"]
            if (new == new and old != old) or (new == new and old == old and new > old):
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

if current is not None and best_val is not None and "test_auroc" in current:
    rows.append({**current, **best_val})

for row in rows:
    parts = row["run_name"].split("_")
    if parts[0] == "metal" and parts[1] == "nut":
        row["category"] = "metal_nut"
        rest = parts[2:]
    else:
        row["category"] = parts[0]
        rest = parts[1:]

    row["t_label"] = rest[0]
    row["m_label"] = rest[1]
    row["lf_label"] = rest[2]
    row["mp_label"] = rest[3]
    row["oc_label"] = rest[4]
    row["seed_label"] = rest[5]

fieldnames = [
    "run_name",
    "category",
    "t_label",
    "m_label",
    "lf_label",
    "mp_label",
    "oc_label",
    "seed_label",
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

with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {out_path}")

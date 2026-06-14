#!/usr/bin/env python3
"""Compare v1 (final_per_category_multiseed) vs v2 (optv2) eval results.

Output:
  eval_report_v2/comparison_v1_vs_v2.csv      per-category side-by-side + delta
  eval_report_v2/comparison_v1_vs_v2.md       markdown table + summary
  eval_report_v2/comparison_auroc_delta.png   bar chart of Δ AUROC per category
  eval_report_v2/comparison_grouped_bars.png  v1 vs v2 AUROC side-by-side
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V1_CSV = ROOT / "final_per_category_multiseed_aggregated.csv"
V2_CSV = ROOT / "optv2_multiseed_aggregated.csv"
OUT = ROOT / "eval_report_v2"

METRIC_KEYS = [
    ("mean_test_auroc", "AUROC"),
    ("mean_test_auprc", "AUPRC"),
    ("mean_test_best_f1", "Best F1"),
    ("mean_test_f1_at_val_threshold", "F1@thr"),
    ("mean_test_fpr_at_95_tpr", "FPR@95"),
]


def load(p: Path) -> dict[str, dict]:
    with open(p) as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict] = {}
    for r in rows:
        cat = r["category"]
        if cat == "category" or cat in out:
            continue  # skip duplicated header row (v1 CSV) / dupes
        out[cat] = r
    return out


def write_csv(v1: dict[str, dict], v2: dict[str, dict], path: Path) -> None:
    cats = sorted(set(v1) | set(v2))
    fieldnames = ["category", "v1_num_seeds", "v2_num_seeds"]
    for k, _ in METRIC_KEYS:
        fieldnames += [f"v1_{k}", f"v2_{k}", f"delta_{k}"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for cat in cats:
            r1 = v1.get(cat, {})
            r2 = v2.get(cat, {})
            row = {"category": cat,
                   "v1_num_seeds": r1.get("num_seeds", ""),
                   "v2_num_seeds": r2.get("num_seeds", "")}
            for k, _ in METRIC_KEYS:
                v1v = float(r1[k]) if r1 else float("nan")
                v2v = float(r2[k]) if r2 else float("nan")
                row[f"v1_{k}"] = f"{v1v:.4f}"
                row[f"v2_{k}"] = f"{v2v:.4f}"
                row[f"delta_{k}"] = f"{v2v - v1v:+.4f}"
            w.writerow(row)
    print(f"Saved: {path}")


def write_md(v1: dict[str, dict], v2: dict[str, dict], path: Path) -> None:
    cats = sorted(set(v1) | set(v2))

    def row_values(cat: str) -> tuple[dict, dict, dict]:
        r1 = v1.get(cat, {})
        r2 = v2.get(cat, {})
        d = {}
        for k, _ in METRIC_KEYS:
            v1v = float(r1[k]) if r1 else float("nan")
            v2v = float(r2[k]) if r2 else float("nan")
            d[k] = v2v - v1v
        return r1, r2, d

    lines = ["# OCGAN2026 — v1 vs v2 Comparison", ""]
    lines.append("**v1** = final_per_category_multiseed (seed 43–47)")
    lines.append("**v2** = optv2 retrain (seed 43–45, skip connections + cosine schedule + memory bank kcenter + learned fusion + EMA + AMP + augmentations)")
    lines.append("")

    # Per-category AUROC
    lines.append("## Per-category AUROC")
    lines.append("")
    lines.append("| Category | seeds v1/v2 | v1 AUROC | v2 AUROC | Δ | v1 AUPRC | v2 AUPRC | Δ |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    # Sort by v2 AUROC desc (fall back to v1)
    def sortkey(c):
        r2 = v2.get(c, {})
        if r2: return -float(r2["mean_test_auroc"])
        r1 = v1.get(c, {})
        return -float(r1["mean_test_auroc"]) if r1 else 0
    for cat in sorted(cats, key=sortkey):
        r1, r2, d = row_values(cat)
        ns = f"{r1.get('num_seeds','-')}/{r2.get('num_seeds','-')}"
        a1 = float(r1["mean_test_auroc"]) if r1 else float("nan")
        a2 = float(r2["mean_test_auroc"]) if r2 else float("nan")
        p1 = float(r1["mean_test_auprc"]) if r1 else float("nan")
        p2 = float(r2["mean_test_auprc"]) if r2 else float("nan")
        dauroc = d["mean_test_auroc"]
        dauprc = d["mean_test_auprc"]
        ind_a = "🟢" if dauroc > 0.005 else ("🔴" if dauroc < -0.005 else "⚪")
        ind_p = "🟢" if dauprc > 0.005 else ("🔴" if dauprc < -0.005 else "⚪")
        lines.append(f"| {cat} | {ns} | {a1:.4f} | {a2:.4f} | {ind_a} {dauroc:+.4f} | {p1:.4f} | {p2:.4f} | {ind_p} {dauprc:+.4f} |")
    lines.append("")

    # Macro
    def macro(d: dict[str, dict], k: str) -> float:
        vals = [float(r[k]) for r in d.values()]
        return sum(vals) / len(vals) if vals else float("nan")

    lines.append("## Macro averages (mean across 15 categories)")
    lines.append("")
    lines.append("| Metric | v1 | v2 | Δ |")
    lines.append("|---|---:|---:|---:|")
    for k, label in METRIC_KEYS:
        m1 = macro(v1, k)
        m2 = macro(v2, k)
        d = m2 - m1
        arrow = "🟢" if (d > 0.005 and "fpr" not in k) or (d < -0.005 and "fpr" in k) else \
                ("🔴" if (d < -0.005 and "fpr" not in k) or (d > 0.005 and "fpr" in k) else "⚪")
        lines.append(f"| {label} | {m1:.4f} | {m2:.4f} | {arrow} {d:+.4f} |")
    lines.append("")

    # Improvements / regressions
    improvements, regressions, neutral = [], [], []
    for cat in cats:
        if cat not in v1 or cat not in v2:
            continue
        d = float(v2[cat]["mean_test_auroc"]) - float(v1[cat]["mean_test_auroc"])
        if d > 0.005: improvements.append((cat, d))
        elif d < -0.005: regressions.append((cat, d))
        else: neutral.append((cat, d))
    improvements.sort(key=lambda x: -x[1])
    regressions.sort(key=lambda x: x[1])

    lines.append("## Improvements (Δ AUROC > +0.005)")
    if improvements:
        for c, d in improvements:
            lines.append(f"- **{c}**: {d:+.4f}")
    else:
        lines.append("_none_")
    lines.append("")
    lines.append("## Regressions (Δ AUROC < -0.005)")
    if regressions:
        for c, d in regressions:
            lines.append(f"- **{c}**: {d:+.4f}")
    else:
        lines.append("_none_")
    lines.append("")
    lines.append("## Stable (|Δ| ≤ 0.005)")
    if neutral:
        lines.append(", ".join(c for c, _ in neutral))
    else:
        lines.append("_none_")
    lines.append("")

    # Tier transitions
    def tier(x: float) -> str:
        if x >= 0.95: return "T1"
        if x >= 0.80: return "T2"
        return "T3"

    lines.append("## Tier transitions")
    lines.append("")
    lines.append("| Category | v1 tier | v2 tier | note |")
    lines.append("|---|:---:|:---:|---|")
    for cat in sorted(cats, key=sortkey):
        if cat not in v1 or cat not in v2:
            continue
        t1 = tier(float(v1[cat]["mean_test_auroc"]))
        t2 = tier(float(v2[cat]["mean_test_auroc"]))
        if t1 == t2:
            continue
        # T1 is best, T3 is worst: upgrade means t2 has lower number than t1
        note = "⬆ upgrade" if t2 < t1 else "⬇ downgrade"
        lines.append(f"| {cat} | {t1} | {t2} | {note} |")
    lines.append("")

    path.write_text("\n".join(lines))
    print(f"Saved: {path}")


def plot_delta_bars(v1: dict[str, dict], v2: dict[str, dict], path: Path) -> None:
    cats_common = sorted(set(v1) & set(v2),
                         key=lambda c: float(v2[c]["mean_test_auroc"]) - float(v1[c]["mean_test_auroc"]))
    deltas = [float(v2[c]["mean_test_auroc"]) - float(v1[c]["mean_test_auroc"]) for c in cats_common]
    colors = ["#10b981" if d > 0.005 else ("#ef4444" if d < -0.005 else "#9ca3af") for d in deltas]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(cats_common, deltas, color=colors, edgecolor="white")
    ax.axvline(0, color="#374151", linewidth=0.8)
    ax.axvline(0.005, color="#d1d5db", linestyle=":", linewidth=0.8)
    ax.axvline(-0.005, color="#d1d5db", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Δ AUROC (v2 − v1)", fontsize=12, fontweight="bold")
    ax.set_title("v2 vs v1 — Per-category AUROC change", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for b, d in zip(bars, deltas):
        ax.text(d + (0.002 if d >= 0 else -0.002), b.get_y() + b.get_height() / 2,
                f"{d:+.3f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_grouped_bars(v1: dict[str, dict], v2: dict[str, dict], path: Path) -> None:
    cats = sorted(set(v1) & set(v2),
                  key=lambda c: -float(v2[c]["mean_test_auroc"]))
    v1_auroc = [float(v1[c]["mean_test_auroc"]) for c in cats]
    v2_auroc = [float(v2[c]["mean_test_auroc"]) for c in cats]
    v1_std = [float(v1[c]["std_test_auroc"]) for c in cats]
    v2_std = [float(v2[c]["std_test_auroc"]) for c in cats]

    x = np.arange(len(cats))
    w = 0.4
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w/2, v1_auroc, w, yerr=v1_std, label="v1 (final)",
           color="#6366f1", edgecolor="white", capsize=3)
    ax.bar(x + w/2, v2_auroc, w, yerr=v2_std, label="v2 (optv2)",
           color="#10b981", edgecolor="white", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_ylabel("Mean Test AUROC", fontsize=12, fontweight="bold")
    ax.set_title("OCGAN2026 — v1 vs v2 AUROC per Category", fontsize=13, fontweight="bold")
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.95, color="#10b981", linestyle="--", alpha=0.3)
    ax.axhline(0.80, color="#f59e0b", linestyle="--", alpha=0.3)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v1 = load(V1_CSV)
    v2 = load(V2_CSV)
    write_csv(v1, v2, OUT / "comparison_v1_vs_v2.csv")
    write_md(v1, v2, OUT / "comparison_v1_vs_v2.md")
    plot_delta_bars(v1, v2, OUT / "comparison_auroc_delta.png")
    plot_grouped_bars(v1, v2, OUT / "comparison_grouped_bars.png")

    # Print summary to console
    def macro(d, k): return sum(float(r[k]) for r in d.values()) / len(d)
    print()
    print("=" * 80)
    print(f"{'Macro summary':^80}")
    print("=" * 80)
    print(f"{'Metric':<12} {'v1':>10} {'v2':>10} {'Δ':>10}")
    print("-" * 80)
    for k, label in METRIC_KEYS:
        m1 = macro(v1, k); m2 = macro(v2, k)
        print(f"{label:<12} {m1:>10.4f} {m2:>10.4f} {m2-m1:>+10.4f}")
    print()


if __name__ == "__main__":
    main()

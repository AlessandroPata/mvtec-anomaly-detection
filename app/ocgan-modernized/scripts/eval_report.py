#!/usr/bin/env python3
"""
Evaluation report for all 15 OCGAN2026 production models.

Reads the multi-seed benchmark results and generates:
1. Console summary table
2. CSV export
3. Matplotlib charts (AUROC bar chart, radar chart, box plot)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS_CSV = ROOT / "final_per_category_multiseed_runs.csv"
AGG_CSV = ROOT / "final_per_category_multiseed_aggregated.csv"
OUTPUT_DIR = ROOT / "eval_report"


def load_runs(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def load_aggregated(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def print_summary_table(agg: list[dict]) -> None:
    # Sort by mean_test_auroc descending
    agg_sorted = sorted(agg, key=lambda r: float(r["mean_test_auroc"]), reverse=True)

    print()
    print("=" * 100)
    print(f"{'OCGAN2026 — Model Performance Report (MVTec AD)':^100}")
    print("=" * 100)
    print()
    print(f"{'Category':<14} {'Seeds':>5} {'AUROC':>10} {'±σ':>8} "
          f"{'AUPRC':>10} {'Best F1':>10} {'F1@thr':>10} "
          f"{'FPR@95':>10} {'Best AUROC':>11}")
    print("-" * 100)

    auroc_values = []
    for r in agg_sorted:
        auroc = float(r["mean_test_auroc"])
        auroc_values.append(auroc)
        print(
            f"{r['category']:<14} {r['num_seeds']:>5} "
            f"{auroc:>10.4f} {float(r['std_test_auroc']):>8.4f} "
            f"{float(r['mean_test_auprc']):>10.4f} "
            f"{float(r['mean_test_best_f1']):>10.4f} "
            f"{float(r['mean_test_f1_at_val_threshold']):>10.4f} "
            f"{float(r['mean_test_fpr_at_95_tpr']):>10.4f} "
            f"{float(r['best_test_auroc']):>11.4f}"
        )

    print("-" * 100)
    mean_auroc = np.mean(auroc_values)
    median_auroc = np.median(auroc_values)
    print(f"{'MEAN':<14} {'':>5} {mean_auroc:>10.4f}")
    print(f"{'MEDIAN':<14} {'':>5} {median_auroc:>10.4f}")
    print()

    # Tier classification
    tier1 = [r["category"] for r in agg_sorted if float(r["mean_test_auroc"]) >= 0.95]
    tier2 = [r["category"] for r in agg_sorted if 0.80 <= float(r["mean_test_auroc"]) < 0.95]
    tier3 = [r["category"] for r in agg_sorted if float(r["mean_test_auroc"]) < 0.80]

    print("Performance Tiers:")
    print(f"  Tier 1 (AUROC >= 0.95): {', '.join(tier1) if tier1 else 'none'}")
    print(f"  Tier 2 (0.80 - 0.95):  {', '.join(tier2) if tier2 else 'none'}")
    print(f"  Tier 3 (< 0.80):       {', '.join(tier3) if tier3 else 'none'}")
    print()


def plot_auroc_bar_chart(agg: list[dict], output_dir: Path) -> None:
    agg_sorted = sorted(agg, key=lambda r: float(r["mean_test_auroc"]), reverse=True)

    categories = [r["category"] for r in agg_sorted]
    means = [float(r["mean_test_auroc"]) for r in agg_sorted]
    stds = [float(r["std_test_auroc"]) for r in agg_sorted]

    colors = []
    for m in means:
        if m >= 0.95:
            colors.append("#10b981")  # emerald
        elif m >= 0.80:
            colors.append("#f59e0b")  # amber
        else:
            colors.append("#ef4444")  # red

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(categories))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="white",
                  linewidth=0.8, error_kw={"linewidth": 1.2, "color": "#6b7280"})

    ax.set_ylabel("Test AUROC", fontsize=12, fontweight="bold")
    ax.set_title("OCGAN2026 — Image-Level AUROC per Category (MVTec AD)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(y=0.95, color="#10b981", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(y=0.80, color="#f59e0b", linestyle="--", alpha=0.4, linewidth=1)
    ax.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Legend
    patches = [
        mpatches.Patch(color="#10b981", label="Tier 1 (>= 0.95)"),
        mpatches.Patch(color="#f59e0b", label="Tier 2 (0.80 - 0.95)"),
        mpatches.Patch(color="#ef4444", label="Tier 3 (< 0.80)"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "auroc_bar_chart.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'auroc_bar_chart.png'}")


def plot_multi_metric(agg: list[dict], output_dir: Path) -> None:
    agg_sorted = sorted(agg, key=lambda r: float(r["mean_test_auroc"]), reverse=True)

    categories = [r["category"] for r in agg_sorted]
    auroc = [float(r["mean_test_auroc"]) for r in agg_sorted]
    auprc = [float(r["mean_test_auprc"]) for r in agg_sorted]
    best_f1 = [float(r["mean_test_best_f1"]) for r in agg_sorted]
    f1_thr = [float(r["mean_test_f1_at_val_threshold"]) for r in agg_sorted]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(categories))
    width = 0.2

    ax.bar(x - 1.5 * width, auroc, width, label="AUROC", color="#6366f1")
    ax.bar(x - 0.5 * width, auprc, width, label="AUPRC", color="#8b5cf6")
    ax.bar(x + 0.5 * width, best_f1, width, label="Best F1", color="#06b6d4")
    ax.bar(x + 1.5 * width, f1_thr, width, label="F1 @ val threshold", color="#f59e0b")

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("OCGAN2026 — Multi-Metric Comparison per Category", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "multi_metric_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'multi_metric_comparison.png'}")


def plot_seed_variability(runs: list[dict], output_dir: Path) -> None:
    from collections import defaultdict

    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        by_cat[r["category"]].append(float(r["test_auroc"]))

    # Sort by median AUROC
    cats_sorted = sorted(by_cat.keys(), key=lambda c: np.median(by_cat[c]), reverse=True)
    data = [by_cat[c] for c in cats_sorted]

    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(data, patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "#ef4444", "markersize": 5},
                    medianprops={"color": "#1f2937", "linewidth": 1.5})

    for patch in bp["boxes"]:
        patch.set_facecolor("#c7d2fe")
        patch.set_edgecolor("#6366f1")

    # Overlay individual points
    for i, (cat, vals) in enumerate(zip(cats_sorted, data)):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter([i + 1 + j for j in jitter], vals, color="#6366f1", alpha=0.6, s=30, zorder=5)

    ax.set_xticklabels(cats_sorted, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Test AUROC", fontsize=12, fontweight="bold")
    ax.set_title("OCGAN2026 — Seed Variability per Category (Box Plot)", fontsize=14, fontweight="bold", pad=15)
    ax.axhline(y=0.95, color="#10b981", linestyle="--", alpha=0.4)
    ax.axhline(y=0.80, color="#f59e0b", linestyle="--", alpha=0.4)
    ax.set_ylim(0.4, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "seed_variability_boxplot.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'seed_variability_boxplot.png'}")


def plot_fpr_vs_auroc(agg: list[dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    for r in agg:
        auroc = float(r["mean_test_auroc"])
        fpr = float(r["mean_test_fpr_at_95_tpr"])
        cat = r["category"]

        if auroc >= 0.95:
            color = "#10b981"
        elif auroc >= 0.80:
            color = "#f59e0b"
        else:
            color = "#ef4444"

        ax.scatter(fpr, auroc, s=80, color=color, edgecolors="white", linewidths=0.8, zorder=5)
        ax.annotate(cat, (fpr, auroc), fontsize=8, ha="left", va="bottom",
                    xytext=(5, 3), textcoords="offset points")

    ax.set_xlabel("FPR @ 95% TPR (lower is better)", fontsize=12, fontweight="bold")
    ax.set_ylabel("AUROC (higher is better)", fontsize=12, fontweight="bold")
    ax.set_title("OCGAN2026 — AUROC vs FPR@95TPR Trade-off", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(0.45, 1.05)
    ax.axvline(x=0.1, color="#d1d5db", linestyle=":", alpha=0.5)
    ax.axhline(y=0.95, color="#d1d5db", linestyle=":", alpha=0.5)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_dir / "fpr_vs_auroc.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'fpr_vs_auroc.png'}")


def export_summary_csv(agg: list[dict], output_dir: Path) -> None:
    agg_sorted = sorted(agg, key=lambda r: float(r["mean_test_auroc"]), reverse=True)

    fieldnames = [
        "rank", "category", "num_seeds",
        "mean_auroc", "std_auroc",
        "mean_auprc", "mean_best_f1",
        "mean_f1_at_val_thr", "mean_fpr_at_95_tpr",
        "best_seed_auroc", "tier",
    ]

    rows = []
    for i, r in enumerate(agg_sorted, 1):
        auroc = float(r["mean_test_auroc"])
        tier = "Tier 1" if auroc >= 0.95 else ("Tier 2" if auroc >= 0.80 else "Tier 3")
        rows.append({
            "rank": i,
            "category": r["category"],
            "num_seeds": r["num_seeds"],
            "mean_auroc": f"{auroc:.4f}",
            "std_auroc": f"{float(r['std_test_auroc']):.4f}",
            "mean_auprc": f"{float(r['mean_test_auprc']):.4f}",
            "mean_best_f1": f"{float(r['mean_test_best_f1']):.4f}",
            "mean_f1_at_val_thr": f"{float(r['mean_test_f1_at_val_threshold']):.4f}",
            "mean_fpr_at_95_tpr": f"{float(r['mean_test_fpr_at_95_tpr']):.4f}",
            "best_seed_auroc": r["best_test_auroc"],
            "tier": tier,
        })

    path = output_dir / "evaluation_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = load_runs(RUNS_CSV)
    agg = load_aggregated(AGG_CSV)

    print_summary_table(agg)

    export_summary_csv(agg, OUTPUT_DIR)
    plot_auroc_bar_chart(agg, OUTPUT_DIR)
    plot_multi_metric(agg, OUTPUT_DIR)
    plot_seed_variability(runs, OUTPUT_DIR)
    plot_fpr_vs_auroc(agg, OUTPUT_DIR)

    print()
    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Re-train all 15 MVTec AD categories with the optimized_v2 config,
then compare results against the original production models.

This script automates:
  1. Training with optimized_v2 config (U-Net, Perlin, top-k, backbone unfreeze, CV fusion)
  2. Multi-seed evaluation (seeds 43, 44, 45)
  3. Results aggregation and comparison with baseline

Usage:
    python scripts/retrain_optimized_v2.py
    python scripts/retrain_optimized_v2.py --categories bottle screw  # subset
    python scripts/retrain_optimized_v2.py --seeds 43 44 45 46 47    # more seeds
    python scripts/retrain_optimized_v2.py --device cuda              # GPU
    python scripts/retrain_optimized_v2.py --dry-run                  # print commands only

Requires:
    - MVTec AD dataset at the path configured in default_mvtec.yaml
    - GPU recommended (CPU training is very slow)
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
BASELINE_CSV = PROJECT_ROOT / "final_per_category_multiseed_aggregated.csv"

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-train with optimized_v2 config")
    p.add_argument("--categories", nargs="+", default=ALL_CATEGORIES)
    p.add_argument("--seeds", type=int, nargs="+", default=[43, 44, 45])
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--output-dir", default=None,
                   help="Override output directory (default: /notebooks/storage_project_outputs_datasets/outputs)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--dataset-root", default=None,
                   help="Override dataset root path")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override number of epochs")
    return p.parse_args()


OPTV2_OVERRIDES = [
    # U-Net skip connections
    "model.reconstruction.use_skip_connections=true",
    # Partial backbone unfreezing
    "model.backbone.unfreeze_from=layer3",
    "model.backbone.unfreeze_lr_factor=0.1",
    # Top-k scoring
    "scoring_topk=100",
    # Perlin anomalies
    "synthetic_anomalies.mode=perlin",
    # Training
    "training.epochs=100",
    # Scheduler
    "scheduler.enabled=true",
    "scheduler.name=cosine",
    "scheduler.warmup_epochs=3",
    "scheduler.warmup_start_factor=0.05",
    "scheduler.min_lr=1e-6",
    # Memory bank
    "memory_bank.enabled=true",
    "memory_bank.max_patches=1024",
    "memory_bank.aggregation=max",
    "++memory_bank.selection_method=kcenter_greedy",
    "++memory_bank.kcenter_init=mean",
    # Learned score fusion with CV
    "score_fusion_learned.enabled=true",
    "score_fusion_learned.C=2.0",
    "score_fusion_learned.cv_folds=5",
    # EMA
    "ema.enabled=true",
    "ema.decay=0.999",
    # AMP (only for CUDA)
    "runtime.amp=false",
    # Augmentations
    "augmentations.train.enabled=true",
    "augmentations.train.rotation_deg=5.0",
    "augmentations.train.color_jitter=true",
    "augmentations.train.brightness=0.08",
    "augmentations.train.contrast=0.08",
    "augmentations.train.gaussian_noise_std=0.008",
    # Early stopping
    "early_stopping.enabled=true",
    "early_stopping.monitor=val_selection_score",
    "early_stopping.mode=max",
    "early_stopping.patience=5",
    "early_stopping.min_delta=0.001",
]


def build_train_command(
    category: str,
    seed: int,
    device: str,
    output_dir: str | None,
    dataset_root: str | None,
    epochs: int | None = None,
) -> list[str]:
    """Build the hydra training command for one category+seed."""
    exp_name = f"{category}_optv2_s{seed}"
    cmd = [
        sys.executable, "-m", "scripts.train",
        "--config-name", "default_mvtec",
        f"dataset.category={category}",
        f"project.seed={seed}",
        f"project.experiment_name={exp_name}",
        f"project.device={device}",
    ]
    # Add all optimized_v2 overrides
    amp_override = f"runtime.amp={'true' if device == 'cuda' else 'false'}"
    for ov in OPTV2_OVERRIDES:
        if ov.startswith("runtime.amp="):
            cmd.append(amp_override)
        else:
            cmd.append(ov)
    if output_dir:
        cmd.append(f"project.output_dir={output_dir}")
    if dataset_root:
        cmd.append(f"dataset.root={dataset_root}")
    if epochs is not None:
        # Override the training.epochs from OPTV2_OVERRIDES
        cmd = [c for c in cmd if not c.startswith("training.epochs=")]
        cmd.append(f"training.epochs={epochs}")
    return cmd


def run_training(args: argparse.Namespace) -> list[dict]:
    """Train all categories × seeds and collect results."""
    results = []
    total = len(args.categories) * len(args.seeds)
    completed = 0

    for category in args.categories:
        for seed in args.seeds:
            completed += 1
            cmd = build_train_command(
                category, seed, args.device,
                args.output_dir, args.dataset_root,
                args.epochs,
            )

            print()
            print("=" * 80)
            print(f"[{completed}/{total}] Training {category} seed={seed}")
            print(f"  Command: {' '.join(cmd)}")
            print("=" * 80)

            if args.dry_run:
                continue

            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if result.returncode != 0:
                print(f"WARNING: Training failed for {category} seed={seed}")
                continue

    return results


def load_baseline() -> dict[str, dict]:
    """Load baseline results from the original multi-seed CSV."""
    baseline = {}
    if not BASELINE_CSV.exists():
        print(f"WARNING: Baseline CSV not found at {BASELINE_CSV}")
        return baseline

    with open(BASELINE_CSV) as f:
        for row in csv.DictReader(f):
            baseline[row["category"]] = {
                "mean_auroc": float(row["mean_test_auroc"]),
                "std_auroc": float(row["std_test_auroc"]),
                "mean_auprc": float(row["mean_test_auprc"]),
                "mean_f1": float(row["mean_test_best_f1"]),
                "mean_fpr95": float(row["mean_test_fpr_at_95_tpr"]),
            }
    return baseline


def find_run_results(output_dir: str, categories: list[str], seeds: list[int]) -> dict[str, list[dict]]:
    """
    Scan output directory for completed runs and extract their test metrics.

    Expected structure: {output_dir}/ocgan-modernized/{category}_optv2_s{seed}_*/metrics.jsonl
    """
    import json

    base = Path(output_dir) / "ocgan-modernized" if output_dir else Path("/notebooks/storage_project_outputs_datasets/outputs/ocgan-modernized")
    results: dict[str, list[dict]] = {}

    for category in categories:
        cat_results = []
        for seed in seeds:
            pattern = f"{category}_optv2_s{seed}_*"
            matches = sorted(base.glob(pattern))
            if not matches:
                continue

            run_dir = matches[-1]  # most recent
            metrics_file = run_dir / "metrics.jsonl"
            if not metrics_file.exists():
                continue

            # Read last line with test metrics
            with open(metrics_file) as f:
                lines = f.readlines()

            for line in reversed(lines):
                try:
                    m = json.loads(line.strip())
                    # Support both naming conventions
                    if "test_auroc" in m or "test_blind_auroc" in m:
                        cat_results.append({
                            "seed": seed,
                            "test_auroc": m.get("test_auroc", m.get("test_blind_auroc", 0)),
                            "test_auprc": m.get("test_auprc", m.get("test_blind_auprc", 0)),
                            "test_best_f1": m.get("test_best_f1", m.get("test_blind_best_f1", 0)),
                            "test_fpr95": m.get("test_fpr_at_95_tpr", m.get("test_blind_fpr_at_95_tpr", 0)),
                        })
                        break
                except json.JSONDecodeError:
                    continue

        if cat_results:
            results[category] = cat_results

    return results


def print_comparison(
    new_results: dict[str, list[dict]],
    baseline: dict[str, dict],
) -> None:
    """Print a side-by-side comparison table."""
    print()
    print("=" * 120)
    print(f"{'COMPARISON: Baseline vs. Optimized v2':^120}")
    print("=" * 120)
    print()
    print(f"{'Category':<14} {'Baseline AUROC':>15} {'OptV2 AUROC':>13} {'Δ AUROC':>10} "
          f"{'Baseline AUPRC':>15} {'OptV2 AUPRC':>13} {'Δ AUPRC':>10} {'Seeds':>6}")
    print("-" * 120)

    improvements = []
    regressions = []

    categories = sorted(set(list(new_results.keys()) + list(baseline.keys())))

    for cat in categories:
        bl = baseline.get(cat, {})
        new = new_results.get(cat, [])

        bl_auroc = bl.get("mean_auroc", float("nan"))
        bl_auprc = bl.get("mean_auprc", float("nan"))

        if new:
            new_aurocs = [r["test_auroc"] for r in new]
            new_auprcs = [r["test_auprc"] for r in new]
            new_auroc = np.mean(new_aurocs)
            new_auprc = np.mean(new_auprcs)
            n_seeds = len(new)
        else:
            new_auroc = float("nan")
            new_auprc = float("nan")
            n_seeds = 0

        delta_auroc = new_auroc - bl_auroc if np.isfinite(new_auroc) and np.isfinite(bl_auroc) else float("nan")
        delta_auprc = new_auprc - bl_auprc if np.isfinite(new_auprc) and np.isfinite(bl_auprc) else float("nan")

        # Color indicator
        if np.isfinite(delta_auroc):
            indicator = "+" if delta_auroc > 0.005 else ("-" if delta_auroc < -0.005 else "=")
            if delta_auroc > 0.005:
                improvements.append(cat)
            elif delta_auroc < -0.005:
                regressions.append(cat)
        else:
            indicator = "?"

        print(
            f"{cat:<14} "
            f"{bl_auroc:>15.4f} {new_auroc:>13.4f} {delta_auroc:>9.4f} {indicator} "
            f"{bl_auprc:>15.4f} {new_auprc:>13.4f} {delta_auprc:>9.4f}  "
            f"{n_seeds:>5}"
        )

    print("-" * 120)

    # Macro averages
    bl_macro = np.mean([bl.get("mean_auroc", 0) for bl in baseline.values()])
    new_cats = [np.mean([r["test_auroc"] for r in v]) for v in new_results.values()]
    new_macro = np.mean(new_cats) if new_cats else float("nan")
    delta_macro = new_macro - bl_macro if np.isfinite(new_macro) else float("nan")

    print(f"{'MACRO AVG':<14} {bl_macro:>15.4f} {new_macro:>13.4f} {delta_macro:>9.4f}")
    print()
    print(f"Improvements (Δ > +0.005): {', '.join(improvements) if improvements else 'none'}")
    print(f"Regressions  (Δ < -0.005): {', '.join(regressions) if regressions else 'none'}")
    print()


def export_results_csv(
    new_results: dict[str, list[dict]],
    output_path: Path,
) -> None:
    """Export new results to CSV for future reference."""
    rows = []
    for cat, runs in sorted(new_results.items()):
        for r in runs:
            rows.append({
                "category": cat,
                "seed": r["seed"],
                "test_auroc": r["test_auroc"],
                "test_auprc": r["test_auprc"],
                "test_best_f1": r["test_best_f1"],
                "test_fpr_at_95_tpr": r["test_fpr95"],
            })

    if not rows:
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results exported to: {output_path}")


def main():
    args = parse_args()

    print("OCGAN2026 — Optimized v2 Re-training Pipeline")
    print(f"  Categories: {args.categories}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Device: {args.device}")
    print(f"  Total runs: {len(args.categories) * len(args.seeds)}")
    print()

    # Phase 1: Training
    run_training(args)

    if args.dry_run:
        print("\n[DRY RUN] No training executed. Commands printed above.")
        return

    # Phase 2: Collect results
    output_dir = args.output_dir or "/notebooks/storage_project_outputs_datasets/outputs"
    new_results = find_run_results(output_dir, args.categories, args.seeds)

    if not new_results:
        print("No completed runs found. Check output directory.")
        return

    # Phase 3: Load baseline and compare
    baseline = load_baseline()
    print_comparison(new_results, baseline)

    # Phase 4: Export
    export_path = PROJECT_ROOT / "optimized_v2_results.csv"
    export_results_csv(new_results, export_path)


if __name__ == "__main__":
    main()

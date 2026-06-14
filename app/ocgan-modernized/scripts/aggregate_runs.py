from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="/notebooks/storage/outputs/ocgan-modernized",
        help="Root directory containing run folders.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Only include runs whose folder name starts with this prefix.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON path.",
    )
    return parser.parse_args()


def load_metrics_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def pick_last_epoch_row(rows: list[dict]) -> dict | None:
    epoch_rows = [r for r in rows if isinstance(r.get("epoch"), int)]
    if not epoch_rows:
        return None
    epoch_rows.sort(key=lambda x: x["epoch"])
    return epoch_rows[-1]


def pick_final_test_row(rows: list[dict]) -> dict | None:
    for r in rows:
        if r.get("epoch") == "final_test":
            return r
    return None


def aggregate_scalar(values: list[float]) -> dict:
    if len(values) == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}

    if len(values) == 1:
        return {
            "count": 1,
            "mean": float(values[0]),
            "std": 0.0,
            "min": float(values[0]),
            "max": float(values[0]),
        }

    return {
        "count": len(values),
        "mean": float(mean(values)),
        "std": float(stdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    if not root.exists():
        raise FileNotFoundError(f"Root non trovata: {root}")

    run_dirs = [p for p in root.iterdir() if p.is_dir()]
    if args.prefix is not None:
        run_dirs = [p for p in run_dirs if p.name.startswith(args.prefix)]

    run_dirs = sorted(run_dirs)

    collected = {
        "val_mixed_auroc": [],
        "val_mixed_auprc": [],
        "val_mixed_best_f1": [],
        "val_selection_score": [],
        "test_blind_auroc": [],
        "test_blind_auprc": [],
        "test_blind_best_f1": [],
        "test_blind_f1_at_given_threshold": [],
    }

    included_runs = []

    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            continue

        rows = load_metrics_jsonl(metrics_path)
        last_epoch = pick_last_epoch_row(rows)
        final_test = pick_final_test_row(rows)

        if last_epoch is None:
            continue

        included_runs.append(run_dir.name)

        for key in ["val_mixed_auroc", "val_mixed_auprc", "val_mixed_best_f1", "val_selection_score"]:
            value = last_epoch.get(key)
            if value is not None:
                collected[key].append(float(value))

        if final_test is not None:
            for key in [
                "test_blind_auroc",
                "test_blind_auprc",
                "test_blind_best_f1",
                "test_blind_f1_at_given_threshold",
            ]:
                value = final_test.get(key)
                if value is not None:
                    collected[key].append(float(value))

    summary = {
        "root": str(root),
        "prefix": args.prefix,
        "num_runs": len(included_runs),
        "runs": included_runs,
        "aggregates": {k: aggregate_scalar(v) for k, v in collected.items()},
    }

    print(json.dumps(summary, indent=2))

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "scripts" / "train.py"


@dataclass
class Spec:
    name: str
    category: str
    seed: int
    overrides: list[str]


TOP_COMBOS = {
    "bottle": [
        ("t1b", "m1b", "lf1a", "oc1"),
        ("t0",  "m1a", "lf1a", "oc1"),
        ("t0",  "m1a", "lf1b", "oc0"),
        ("t1a", "m1b", "lf1b", "oc0"),
        ("t1b", "m1b", "lf1b", "oc0"),
        ("t0",  "m1b", "lf1a", "oc1"),
        ("t0",  "m1b", "lf1b", "oc0"),
        ("t0",  "m1b", "lf1a", "oc0"),
    ],
    "cable": [
        ("t0",  "m1a", "lf1b", "oc0"),
        ("t1b", "m1b", "lf1b", "oc1"),
    ],
    "capsule": [
        ("t0",  "m1b", "lf1b", "oc1"),
        ("t1b", "m1b", "lf1b", "oc0"),
        ("t1a", "m1b", "lf1b", "oc1"),
        ("t1b", "m1b", "lf1a", "oc0"),
        ("t1a", "m1b", "lf1b", "oc0"),
        ("t1b", "m0",  "lf1a", "oc0"),
        ("t1a", "m0",  "lf1a", "oc1"),
        ("t1b", "m1a", "lf1b", "oc0"),
    ],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-path", default="../configs")
    p.add_argument("--config-name", default="default_mvtec")
    p.add_argument("--categories", nargs="+", default=["bottle", "cable", "capsule"])
    p.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45, 46])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--max-parallel", type=int, default=3)
    p.add_argument("--out-dir", default="./grid_runs/final_grid_72")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def teacher_overrides(tag: str) -> list[str]:
    if tag == "t0":
        return [
            "teacher_student.enabled=false",
            "teacher_student.score_weight=0.0",
            "score_fusion.teacher_student_weight=0.0",
            "model.scoring.teacher_student_score=false",
        ]
    if tag == "t1a":
        return [
            "teacher_student.enabled=true",
            "teacher_student.score_weight=0.1",
            "score_fusion.teacher_student_weight=1.0",
            "model.scoring.teacher_student_score=true",
        ]
    if tag == "t1b":
        return [
            "teacher_student.enabled=true",
            "teacher_student.score_weight=0.2",
            "score_fusion.teacher_student_weight=1.0",
            "model.scoring.teacher_student_score=true",
        ]
    raise ValueError(f"unknown teacher tag: {tag}")


def memory_overrides(tag: str) -> list[str]:
    if tag == "m0":
        return [
            "memory_bank.enabled=false",
            "memory_bank.score_weight=0.0",
            "score_fusion.memory_weight=0.0",
            "model.scoring.memory_score=false",
        ]
    if tag == "m1a":
        return [
            "memory_bank.enabled=true",
            "memory_bank.score_weight=0.05",
            "score_fusion.memory_weight=0.05",
            "model.scoring.memory_score=true",
        ]
    if tag == "m1b":
        return [
            "memory_bank.enabled=true",
            "memory_bank.score_weight=0.1",
            "score_fusion.memory_weight=0.1",
            "model.scoring.memory_score=true",
        ]
    raise ValueError(f"unknown memory tag: {tag}")


def learned_fusion_overrides(tag: str) -> list[str]:
    if tag == "lf0":
        return [
            "score_fusion_learned.enabled=false",
        ]
    if tag == "lf1a":
        return [
            "score_fusion_learned.enabled=true",
            "score_fusion_learned.C=0.5",
        ]
    if tag == "lf1b":
        return [
            "score_fusion_learned.enabled=true",
            "score_fusion_learned.C=1.0",
        ]
    raise ValueError(f"unknown learned_fusion tag: {tag}")


def one_class_overrides(tag: str) -> list[str]:
    if tag == "oc0":
        return [
            "one_class.enabled=true",
            "one_class.score_weight=0.0",
            "model.scoring.latent_score=false",
        ]
    if tag == "oc1":
        return [
            "one_class.enabled=true",
            "one_class.score_weight=1.0",
            "model.scoring.latent_score=true",
        ]
    raise ValueError(f"unknown one_class tag: {tag}")


def make_specs(categories: list[str], seeds: list[int], epochs: int) -> list[Spec]:
    specs: list[Spec] = []
    for category in categories:
        combos = TOP_COMBOS[category]
        for teacher, memory, lf, oc in combos:
            for seed in seeds:
                name = f"{category}_{teacher}_{memory}_{lf}_{oc}_s{seed}"
                overrides = [
                    f"project.experiment_name={name}",
                    f"project.seed={seed}",
                    f"dataset.seed={seed}",
                    f"dataset.category={category}",
                    f"training.epochs={epochs}",
                    *teacher_overrides(teacher),
                    *memory_overrides(memory),
                    *learned_fusion_overrides(lf),
                    *one_class_overrides(oc),
                ]
                specs.append(Spec(name=name, category=category, seed=seed, overrides=overrides))
    return specs


def parse_metrics(text: str) -> dict:
    def grab(pattern: str) -> str:
        m = re.search(pattern, text)
        return m.group(1) if m else ""

    test_line = re.search(r"\[Test\].*", text)
    test_txt = test_line.group(0) if test_line else ""

    return {
        "best_val_selection": grab(r"selection_score=([0-9.]+)"),
        "test_auroc": re.search(r"AUROC=([0-9.]+)", test_txt).group(1) if re.search(r"AUROC=([0-9.]+)", test_txt) else "",
        "test_auprc": re.search(r"AUPRC=([0-9.]+)", test_txt).group(1) if re.search(r"AUPRC=([0-9.]+)", test_txt) else "",
        "test_best_f1": re.search(r"best_F1=([0-9.]+)", test_txt).group(1) if re.search(r"best_F1=([0-9.]+)", test_txt) else "",
        "f1_at_val_threshold": re.search(r"F1@val_threshold=([0-9.]+)", test_txt).group(1) if re.search(r"F1@val_threshold=([0-9.]+)", test_txt) else "",
        "fpr_at_95_tpr": re.search(r"FPR@95TPR=([0-9.]+)", test_txt).group(1) if re.search(r"FPR@95TPR=([0-9.]+)", test_txt) else "",
    }


def run_one(spec: Spec, args, logs_dir: Path) -> dict:
    log_path = logs_dir / f"{spec.name}.log"
    cmd = [
        sys.executable,
        str(TRAIN_PY),
        "--config-path",
        args.config_path,
        "--config-name",
        args.config_name,
        *spec.overrides,
    ]

    with log_path.open("w") as f:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    result = {
        "name": spec.name,
        "category": spec.category,
        "seed": spec.seed,
        "status": "done" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "log_path": str(log_path),
    }

    if log_path.exists():
        metrics = parse_metrics(log_path.read_text(errors="ignore"))
        result.update(metrics)

    return result


def write_results_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "name",
        "category",
        "seed",
        "status",
        "returncode",
        "best_val_selection",
        "test_auroc",
        "test_auprc",
        "test_best_f1",
        "f1_at_val_threshold",
        "fpr_at_95_tpr",
        "log_path",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    specs = make_specs(args.categories, args.seeds, args.epochs)

    if args.dry_run:
        print(f"[dry-run] total specs: {len(specs)}")
        for s in specs[:30]:
            print(s.name, s.overrides)
        return

    results = []
    pbar = tqdm(total=len(specs), desc="grid")

    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futs = {ex.submit(run_one, s, args, logs_dir): s for s in specs}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            failed = sum(r["status"] != "done" for r in results)
            pbar.update(1)
            pbar.set_postfix(failed=failed, pending=len(specs) - len(results))
            write_results_csv(out_dir / "results.csv", sorted(results, key=lambda x: x["name"]))

    pbar.close()
    print(f"[done] total={len(specs)} done={sum(r['status']=='done' for r in results)} failed={sum(r['status']!='done' for r in results)}")
    print(f"[results] {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()#!/usr/bin/env python3

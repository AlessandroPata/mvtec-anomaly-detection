#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "scripts" / "train.py"


@dataclass
class Spec:
    name: str
    category: str
    seed: int
    teacher: str
    memory: str
    learned_fusion: str
    one_class_score: str
    overrides: List[str]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-path", default="../configs")
    p.add_argument("--config-name", default="default_mvtec")
    p.add_argument("--categories", nargs="+", default=["bottle", "cable", "capsule"])
    p.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45, 46])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-parallel", type=int, default=5)
    p.add_argument("--out-dir", default="./grid_runs/rich_default_mvtec")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def teacher_states():
    return [
        ("t0", [
            "teacher_student.enabled=false",
            "teacher_student.score_weight=0.0",
            "score_fusion.teacher_student_weight=0.0",
            "model.scoring.teacher_student_score=false",
        ]),
        ("t1a", [
            "teacher_student.enabled=true",
            "teacher_student.score_weight=0.1",
            "score_fusion.teacher_student_weight=1.0",
            "model.scoring.teacher_student_score=true",
        ]),
        ("t1b", [
            "teacher_student.enabled=true",
            "teacher_student.score_weight=0.2",
            "score_fusion.teacher_student_weight=1.0",
            "model.scoring.teacher_student_score=true",
        ]),
    ]


def memory_states():
    return [
        ("m0", [
            "memory_bank.enabled=false",
            "memory_bank.score_weight=0.0",
            "score_fusion.memory_weight=0.0",
            "model.scoring.memory_score=false",
        ]),
        ("m1a", [
            "memory_bank.enabled=true",
            "memory_bank.score_weight=0.05",
            "score_fusion.memory_weight=0.05",
            "model.scoring.memory_score=true",
        ]),
        ("m1b", [
            "memory_bank.enabled=true",
            "memory_bank.score_weight=0.1",
            "score_fusion.memory_weight=0.1",
            "model.scoring.memory_score=true",
        ]),
    ]


def learned_fusion_states():
    return [
        ("lf0", [
            "score_fusion_learned.enabled=false",
        ]),
        ("lf1a", [
            "score_fusion_learned.enabled=true",
            "score_fusion_learned.C=0.5",
        ]),
        ("lf1b", [
            "score_fusion_learned.enabled=true",
            "score_fusion_learned.C=1.0",
        ]),
    ]


def one_class_states():
    return [
        ("oc0", [
            "one_class.enabled=true",
            "one_class.score_weight=0.0",
            "model.scoring.latent_score=false",
        ]),
        ("oc1", [
            "one_class.enabled=true",
            "one_class.score_weight=1.0",
            "model.scoring.latent_score=true",
        ]),
    ]


def build_specs(args) -> List[Spec]:
    specs = []
    for category in args.categories:
        for seed in args.seeds:
            for t_name, t_over in teacher_states():
                for m_name, m_over in memory_states():
                    for lf_name, lf_over in learned_fusion_states():
                        for oc_name, oc_over in one_class_states():
                            name = f"{category}_{t_name}_{m_name}_{lf_name}_{oc_name}_s{seed}"
                            overrides = [
                                f"project.experiment_name={name}",
                                f"project.seed={seed}",
                                f"dataset.seed={seed}",
                                f"dataset.category={category}",
                                f"training.epochs={args.epochs}",

                                # salva il minimo lato training output
                                "logging.save_debug_images=false",
                                "logging.save_score_histograms=false",
                                "logging.save_metrics_jsonl=false",
                                "logging.save_epoch_summary_yaml=false",
                                "logging.save_env_info=false",
                                "logging.save_config=false",

                                # niente checkpoint pesanti
                                "checkpoint.save_last=false",
                                "checkpoint.save_best=false",

                                # meno spam su stdout
                                "logging.log_every_n_steps=100",

                                # opzionale: evita cartelle output verbose di hydra
                                "hydra.output_subdir=null",
                            ] + t_over + m_over + lf_over + oc_over

                            specs.append(
                                Spec(
                                    name=name,
                                    category=category,
                                    seed=seed,
                                    teacher=t_name,
                                    memory=m_name,
                                    learned_fusion=lf_name,
                                    one_class_score=oc_name,
                                    overrides=overrides,
                                )
                            )
    return specs


def extract_test_metrics(text: str) -> Dict[str, str]:
    out = {
        "best_val_selection": "",
        "test_auroc": "",
        "test_auprc": "",
        "test_best_f1": "",
        "f1_at_val_threshold": "",
        "fpr_at_95_tpr": "",
    }

    val_matches = re.findall(r"\[Validation\].*?selection_score=([0-9.]+)", text)
    if val_matches:
        out["best_val_selection"] = val_matches[-1]

    test_matches = re.findall(
        r"\[Test\]\s+AUROC=([0-9.]+)\s+AUPRC=([0-9.]+)\s+best_F1=([0-9.]+)\s+F1@val_threshold=([0-9.]+)\s+FPR@95TPR=([0-9.]+)",
        text,
    )
    if test_matches:
        auroc, auprc, best_f1, f1_thr, fpr95 = test_matches[-1]
        out["test_auroc"] = auroc
        out["test_auprc"] = auprc
        out["test_best_f1"] = best_f1
        out["f1_at_val_threshold"] = f1_thr
        out["fpr_at_95_tpr"] = fpr95

    return out


def write_results_csv(path: Path, rows: List[Dict[str, str]]):
    fieldnames = [
        "name",
        "category",
        "seed",
        "teacher",
        "memory",
        "learned_fusion",
        "one_class_score",
        "status",
        "returncode",
        "best_val_selection",
        "test_auroc",
        "test_auprc",
        "test_best_f1",
        "f1_at_val_threshold",
        "fpr_at_95_tpr",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one(spec: Spec, args, out_dir: Path) -> Dict[str, str]:
    failed_logs_dir = out_dir / "failed_logs"
    failed_logs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(TRAIN_PY),
        "--config-path", args.config_path,
        "--config-name", args.config_name,
    ] + spec.overrides

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )

    txt = (proc.stdout or "") + "\n" + (proc.stderr or "")
    metrics = extract_test_metrics(txt)

    status = "done" if proc.returncode == 0 and "Training finished." in txt else "failed"

    if status == "failed":
        fail_log = failed_logs_dir / f"{spec.name}.log"
        fail_log.write_text(txt, errors="ignore")

    return {
        "name": spec.name,
        "category": spec.category,
        "seed": str(spec.seed),
        "teacher": spec.teacher,
        "memory": spec.memory,
        "learned_fusion": spec.learned_fusion,
        "one_class_score": spec.one_class_score,
        "status": status,
        "returncode": str(proc.returncode),
        **metrics,
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = build_specs(args)

    if args.dry_run:
        print(f"[dry-run] total specs: {len(specs)}")
        for s in specs[:12]:
            print(s.name, s.overrides)
        return

    results: List[Dict[str, str]] = []
    results_csv = out_dir / "results.csv"

    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futures = {ex.submit(run_one, spec, args, out_dir): spec for spec in specs}

        with tqdm(total=len(specs), desc="grid") as pbar:
            for fut in as_completed(futures):
                row = fut.result()
                results.append(row)

                write_results_csv(results_csv, sorted(results, key=lambda x: x["name"]))

                pbar.update(1)

                failed = sum(r["status"] == "failed" for r in results)
                pending = len(specs) - len(results)
                running = min(args.max_parallel, pending) if pending > 0 else 0

                pbar.set_postfix(
                    failed=failed,
                    pending=pending,
                    running=running,
                )

    failed = sum(r["status"] == "failed" for r in results)
    done = sum(r["status"] == "done" for r in results)
    pending = len(specs) - len(results)

    print(f"[done] total={len(specs)} done={done} failed={failed} pending={pending}")
    print(f"[results] {results_csv}")
    print(f"[failed_logs_dir] {out_dir / 'failed_logs'}")


if __name__ == "__main__":
    main()#!/usr/bin/env python3

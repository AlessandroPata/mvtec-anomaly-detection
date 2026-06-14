#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
import sys
import time
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
    tag: str
    overrides: List[str]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-path", default="../configs")
    p.add_argument("--config-name", default="default_mvtec")
    p.add_argument(
        "--categories",
        nargs="+",
        default=[
            "bottle", "cable", "capsule", "carpet", "grid",
            "hazelnut", "leather", "metal_nut", "pill", "screw",
            "tile", "toothbrush", "transistor", "wood", "zipper",
        ],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45, 46])
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--max-parallel", type=int, default=4)
    p.add_argument("--timeout-minutes", type=int, default=90)
    p.add_argument("--out-dir", default="./grid_runs/shortlist_final16")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    return p.parse_args()

def shortlist_configs():
    return [
        ("t1a_m1a_lf1b_oc0", [
            "teacher_student.enabled=true",
            "teacher_student.score_weight=0.10",
            "score_fusion.teacher_student_weight=1.0",
            "model.scoring.teacher_student_score=true",
            "memory_bank.enabled=true",
            "memory_bank.score_weight=0.05",
            "score_fusion.memory_weight=0.05",
            "model.scoring.memory_score=true",
            "score_fusion_learned.enabled=true",
            "score_fusion_learned.C=1.0",
            "one_class.enabled=true",
            "one_class.score_weight=0.0",
            "model.scoring.latent_score=false",
        ]),
        ("t1b_m1d_lf1d_oc0", [
            "teacher_student.enabled=true",
            "teacher_student.score_weight=0.20",
            "score_fusion.teacher_student_weight=1.0",
            "model.scoring.teacher_student_score=true",
            "memory_bank.enabled=true",
            "memory_bank.score_weight=0.20",
            "score_fusion.memory_weight=0.20",
            "model.scoring.memory_score=true",
            "score_fusion_learned.enabled=true",
            "score_fusion_learned.C=4.0",
            "one_class.enabled=true",
            "one_class.score_weight=0.0",
            "model.scoring.latent_score=false",
        ]),
        ("t1b_m1c_lf1b_oc0", [
            "teacher_student.enabled=true",
            "teacher_student.score_weight=0.20",
            "score_fusion.teacher_student_weight=1.0",
            "model.scoring.teacher_student_score=true",
            "memory_bank.enabled=true",
            "memory_bank.score_weight=0.15",
            "score_fusion.memory_weight=0.15",
            "model.scoring.memory_score=true",
            "score_fusion_learned.enabled=true",
            "score_fusion_learned.C=1.0",
            "one_class.enabled=true",
            "one_class.score_weight=0.0",
            "model.scoring.latent_score=false",
        ]),
        ("t1c_m1c_lf1c_oc0", [
            "teacher_student.enabled=true",
            "teacher_student.score_weight=0.30",
            "score_fusion.teacher_student_weight=1.0",
            "model.scoring.teacher_student_score=true",
            "memory_bank.enabled=true",
            "memory_bank.score_weight=0.15",
            "score_fusion.memory_weight=0.15",
            "model.scoring.memory_score=true",
            "score_fusion_learned.enabled=true",
            "score_fusion_learned.C=2.0",
            "one_class.enabled=true",
            "one_class.score_weight=0.0",
            "model.scoring.latent_score=false",
        ]),
        ("t1b_m1b_lf1a_oc0", [
            "teacher_student.enabled=true",
            "teacher_student.score_weight=0.20",
            "score_fusion.teacher_student_weight=1.0",
            "model.scoring.teacher_student_score=true",
            "memory_bank.enabled=true",
            "memory_bank.score_weight=0.10",
            "score_fusion.memory_weight=0.10",
            "model.scoring.memory_score=true",
            "score_fusion_learned.enabled=true",
            "score_fusion_learned.C=0.5",
            "one_class.enabled=true",
            "one_class.score_weight=0.0",
            "model.scoring.latent_score=false",
        ]),
        ("t1d_m1d_lf1c_oc0", [
            "teacher_student.enabled=true",
            "teacher_student.score_weight=0.40",
            "score_fusion.teacher_student_weight=1.0",
            "model.scoring.teacher_student_score=true",
            "memory_bank.enabled=true",
            "memory_bank.score_weight=0.20",
            "score_fusion.memory_weight=0.20",
            "model.scoring.memory_score=true",
            "score_fusion_learned.enabled=true",
            "score_fusion_learned.C=2.0",
            "one_class.enabled=true",
            "one_class.score_weight=0.0",
            "model.scoring.latent_score=false",
        ]),
    ]

def build_specs(args) -> List[Spec]:
    specs = []
    for category in args.categories:
        for seed in args.seeds:
            for tag, extra in shortlist_configs():
                name = f"{category}_{tag}_s{seed}"
                overrides = [
                    f"project.experiment_name={name}",
                    f"project.seed={seed}",
                    f"dataset.seed={seed}",
                    f"dataset.category={category}",
                    f"training.epochs={args.epochs}",
                    "logging.save_debug_images=false",
                    "logging.save_score_histograms=false",
                    "logging.save_metrics_jsonl=false",
                    "logging.save_epoch_summary_yaml=false",
                    "logging.save_env_info=false",
                    "logging.save_config=false",
                    "checkpoint.save_last=false",
                    "checkpoint.save_best=false",
                    "logging.log_every_n_steps=100",
                ] + extra
                specs.append(Spec(name=name, category=category, seed=seed, tag=tag, overrides=overrides))
    return specs

def parse_float(x: str) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return ""

def extract_run_dir(text: str) -> str:
    m = re.findall(r"Run directory:\s*(.+)", text)
    return m[-1].strip() if m else ""

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
        out["best_val_selection"] = parse_float(val_matches[-1])

    test_matches = re.findall(
        r"\[Test\]\s+AUROC=([0-9.]+)\s+AUPRC=([0-9.]+)\s+best_F1=([0-9.]+)\s+F1@val_threshold=([0-9.]+)\s+FPR@95TPR=([0-9.]+)",
        text,
    )
    if test_matches:
        auroc, auprc, best_f1, f1_thr, fpr95 = test_matches[-1]
        out["test_auroc"] = parse_float(auroc)
        out["test_auprc"] = parse_float(auprc)
        out["test_best_f1"] = parse_float(best_f1)
        out["f1_at_val_threshold"] = parse_float(f1_thr)
        out["fpr_at_95_tpr"] = parse_float(fpr95)
    return out

def has_finished_successfully(text: str) -> bool:
    return "Training finished." in text

def load_existing_results(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        return list(csv.DictReader(path.open()))
    except Exception:
        return []

def results_by_name_from_csv(path: Path) -> Dict[str, Dict[str, str]]:
    rows = load_existing_results(path)
    return {r["name"]: r for r in rows if "name" in r}

def result_row_from_log(spec: Spec, log_path: Path, status: str, returncode: str, elapsed_sec: str = "") -> Dict[str, str]:
    txt = log_path.read_text(errors="ignore") if log_path.exists() else ""
    metrics = extract_test_metrics(txt)
    return {
        "name": spec.name,
        "category": spec.category,
        "seed": str(spec.seed),
        "tag": spec.tag,
        "status": status,
        "returncode": returncode,
        "elapsed_sec": elapsed_sec,
        "run_dir": extract_run_dir(txt),
        **metrics,
        "log_path": str(log_path),
    }

def recover_results_from_logs(specs: List[Spec], out_dir: Path) -> Dict[str, Dict[str, str]]:
    logs_dir = out_dir / "logs"
    recovered = {}
    for spec in specs:
        log_path = logs_dir / f"{spec.name}.log"
        if not log_path.exists():
            continue
        txt = log_path.read_text(errors="ignore")
        if has_finished_successfully(txt):
            recovered[spec.name] = result_row_from_log(spec, log_path, "done", "0")
        elif "[TIMEOUT]" in txt:
            recovered[spec.name] = result_row_from_log(spec, log_path, "timeout", "124")
        elif "[EXCEPTION]" in txt or "[UNHANDLED_EXCEPTION]" in txt:
            recovered[spec.name] = result_row_from_log(spec, log_path, "exception", "-1")
    return recovered

def completed_names_for_resume(rows_by_name: Dict[str, Dict[str, str]], retry_failed: bool) -> set:
    completed = set()
    for name, row in rows_by_name.items():
        status = row.get("status", "")
        if status == "done":
            completed.add(name)
        elif not retry_failed and status in {"failed", "timeout", "exception"}:
            completed.add(name)
    return completed

def write_results_csv(path: Path, rows: List[Dict[str, str]]):
    fieldnames = [
        "name", "category", "seed", "tag", "status", "returncode", "elapsed_sec",
        "run_dir", "best_val_selection", "test_auroc", "test_auprc",
        "test_best_f1", "f1_at_val_threshold", "fpr_at_95_tpr", "log_path",
    ]
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)

def run_one(spec: Spec, args, out_dir: Path) -> Dict[str, str]:
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{spec.name}.log"

    if log_path.exists():
        txt = log_path.read_text(errors="ignore")
        if has_finished_successfully(txt):
            return result_row_from_log(spec, log_path, "done", "0")

    cmd = [sys.executable, str(TRAIN_PY), "--config-path", args.config_path, "--config-name", args.config_name] + spec.overrides
    start_ts = time.time()

    try:
        with log_path.open("w") as f:
            f.write(f"[START] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("CMD: " + " ".join(cmd) + "\n\n")
            f.flush()

            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout_minutes * 60,
            )

        elapsed = f"{time.time() - start_ts:.1f}"
        status = "done" if proc.returncode == 0 and has_finished_successfully(log_path.read_text(errors="ignore")) else "failed"
        return result_row_from_log(spec, log_path, status, str(proc.returncode), elapsed)

    except subprocess.TimeoutExpired:
        elapsed = f"{time.time() - start_ts:.1f}"
        with log_path.open("a") as f:
            f.write(f"\n[TIMEOUT] after {elapsed} sec\n")
        return result_row_from_log(spec, log_path, "timeout", "124", elapsed)

    except Exception as e:
        elapsed = f"{time.time() - start_ts:.1f}"
        with log_path.open("a") as f:
            f.write(f"\n[EXCEPTION] {repr(e)}\n")
        return result_row_from_log(spec, log_path, "exception", "-1", elapsed)

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = build_specs(args)
    results_csv = out_dir / "results.csv"
    manifest_path = out_dir / "manifest.json"

    manifest = {
        "root": str(ROOT),
        "train_py": str(TRAIN_PY),
        "config_path": args.config_path,
        "config_name": args.config_name,
        "epochs": args.epochs,
        "max_parallel": args.max_parallel,
        "timeout_minutes": args.timeout_minutes,
        "total_specs": len(specs),
        "specs": [{"name": s.name, "category": s.category, "seed": s.seed, "tag": s.tag, "overrides": s.overrides} for s in specs],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if args.dry_run:
        print(f"[dry-run] total specs: {len(specs)}")
        for s in specs[:12]:
            print(s.name, s.overrides)
        print(f"[manifest] {manifest_path}")
        return

    existing_by_name = results_by_name_from_csv(results_csv)
    recovered_from_logs = recover_results_from_logs(specs, out_dir)
    existing_by_name.update(recovered_from_logs)

    done_names = completed_names_for_resume(existing_by_name, args.retry_failed)
    pending_specs = [s for s in specs if s.name not in done_names]

    print(f"[resume] known={len(existing_by_name)} completed={len(done_names)} pending={len(pending_specs)}")

    results = sorted(existing_by_name.values(), key=lambda x: x["name"])
    write_results_csv(results_csv, results)

    if not pending_specs:
        print(f"[done] total={len(specs)} done={sum(r['status']=='done' for r in results)} "
              f"failed={sum(r['status']=='failed' for r in results)} "
              f"timeout={sum(r['status']=='timeout' for r in results)} "
              f"exception={sum(r['status']=='exception' for r in results)} pending=0")
        print(f"[manifest] {manifest_path}")
        print(f"[results]  {results_csv}")
        return

    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futures = {ex.submit(run_one, spec, args, out_dir): spec for spec in pending_specs}
        with tqdm(total=len(specs), initial=len(done_names), desc="shortlist") as pbar:
            for fut in as_completed(futures):
                spec = futures[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    log_path = out_dir / "logs" / f"{spec.name}.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open("a") as f:
                        f.write(f"\n[UNHANDLED_EXCEPTION] {repr(e)}\n")
                    row = {
                        "name": spec.name,
                        "category": spec.category,
                        "seed": str(spec.seed),
                        "tag": spec.tag,
                        "status": "exception",
                        "returncode": "-1",
                        "elapsed_sec": "",
                        "run_dir": "",
                        "best_val_selection": "",
                        "test_auroc": "",
                        "test_auprc": "",
                        "test_best_f1": "",
                        "f1_at_val_threshold": "",
                        "fpr_at_95_tpr": "",
                        "log_path": str(log_path),
                    }

                existing_by_name[row["name"]] = row
                results = sorted(existing_by_name.values(), key=lambda x: x["name"])
                write_results_csv(results_csv, results)

                pbar.update(1)
                failed = sum(r["status"] == "failed" for r in results)
                timeout = sum(r["status"] == "timeout" for r in results)
                exc = sum(r["status"] == "exception" for r in results)
                pending = len(specs) - len(results)
                pbar.set_postfix(failed=failed, timeout=timeout, exc=exc, pending=pending)

    results = sorted(existing_by_name.values(), key=lambda x: x["name"])
    write_results_csv(results_csv, results)

    failed = sum(r["status"] == "failed" for r in results)
    timeout = sum(r["status"] == "timeout" for r in results)
    exc = sum(r["status"] == "exception" for r in results)
    done = sum(r["status"] == "done" for r in results)
    pending = len(specs) - len(results)

    print(f"[done] total={len(specs)} done={done} failed={failed} timeout={timeout} exception={exc} pending={pending}")
    print(f"[manifest] {manifest_path}")
    print(f"[results]  {results_csv}")

if __name__ == "__main__":
    main()#!/usr/bin/env python3

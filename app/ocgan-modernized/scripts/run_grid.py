from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm


@dataclass
class RunSpec:
    name: str
    category: str
    seed: int
    teacher: int
    memory: int
    learned_fusion: int
    one_class_score: int
    epochs: int
    overrides: list[str]


def build_spec(
    category: str,
    seed: int,
    teacher: int,
    memory: int,
    learned_fusion: int,
    one_class_score: int,
    epochs: int,
) -> RunSpec:
    name = (
        f"{category}_t{teacher}_m{memory}_lf{learned_fusion}_oc{one_class_score}_s{seed}"
    )

    overrides = [
        f"project.experiment_name={name}",
        f"project.seed={seed}",
        f"dataset.category={category}",
        f"dataset.seed={seed}",
        f"training.epochs={epochs}",
    ]

    # teacher
    if teacher:
        overrides += [
            "teacher_student.enabled=true",
            "teacher_student.score_weight=0.1",
            "score_fusion.teacher_student_weight=1.0",
            "model.scoring.teacher_student_score=true",
        ]
    else:
        overrides += [
            "teacher_student.enabled=false",
            "teacher_student.score_weight=0.0",
            "score_fusion.teacher_student_weight=0.0",
            "model.scoring.teacher_student_score=false",
        ]

    # memory
    if memory:
        overrides += [
            "memory_bank.enabled=true",
            "memory_bank.score_weight=0.1",
            "score_fusion.memory_weight=0.1",
            "model.scoring.memory_score=true",
        ]
    else:
        overrides += [
            "memory_bank.enabled=false",
            "memory_bank.score_weight=0.0",
            "score_fusion.memory_weight=0.0",
            "model.scoring.memory_score=false",
        ]

    # learned fusion
    overrides += [
        f"score_fusion_learned.enabled={'true' if learned_fusion else 'false'}",
    ]

    # one-class score
    if one_class_score:
        overrides += [
            "one_class.enabled=true",
            "one_class.score_weight=1.0",
            "model.scoring.latent_score=true",
        ]
    else:
        overrides += [
            "one_class.enabled=true",
            "one_class.score_weight=0.0",
            "model.scoring.latent_score=false",
        ]

    return RunSpec(
        name=name,
        category=category,
        seed=seed,
        teacher=teacher,
        memory=memory,
        learned_fusion=learned_fusion,
        one_class_score=one_class_score,
        epochs=epochs,
        overrides=overrides,
    )


def generate_specs(
    categories: list[str],
    seeds: list[int],
    epochs: int,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for category in categories:
        for seed in seeds:
            for teacher in [0, 1]:
                for memory in [0, 1]:
                    for learned_fusion in [0, 1]:
                        for one_class_score in [0, 1]:
                            specs.append(
                                build_spec(
                                    category=category,
                                    seed=seed,
                                    teacher=teacher,
                                    memory=memory,
                                    learned_fusion=learned_fusion,
                                    one_class_score=one_class_score,
                                    epochs=epochs,
                                )
                            )
    return specs


def load_manifest(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {"runs": {}}


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def ensure_results_csv(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "category",
                "seed",
                "teacher",
                "memory",
                "learned_fusion",
                "one_class_score",
                "status",
                "returncode",
                "run_dir",
                "best_val_selection",
                "test_auroc",
                "test_auprc",
                "test_best_f1",
                "f1_at_val_threshold",
                "fpr_at_95_tpr",
                "log_path",
            ]
        )


def append_result(path: Path, row: list[Any]) -> None:
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def parse_metrics_from_log(log_path: Path) -> dict[str, Any]:
    metrics = {
        "best_val_selection": "",
        "test_auroc": "",
        "test_auprc": "",
        "test_best_f1": "",
        "f1_at_val_threshold": "",
        "fpr_at_95_tpr": "",
        "run_dir": "",
    }

    if not log_path.exists():
        return metrics

    best_val = None

    for line in log_path.read_text(errors="ignore").splitlines():
        if line.startswith("Run directory:"):
            metrics["run_dir"] = line.split("Run directory:", 1)[1].strip()

        if "[Validation]" in line and "selection_score=" in line:
            try:
                best_val = float(line.split("selection_score=")[-1].strip())
            except Exception:
                pass

        if "[Test]" in line:
            parts = line.strip().split()
            for part in parts:
                if part.startswith("AUROC="):
                    metrics["test_auroc"] = part.split("=", 1)[1]
                elif part.startswith("AUPRC="):
                    metrics["test_auprc"] = part.split("=", 1)[1]
                elif part.startswith("best_F1="):
                    metrics["test_best_f1"] = part.split("=", 1)[1]
                elif part.startswith("F1@val_threshold="):
                    metrics["f1_at_val_threshold"] = part.split("=", 1)[1]
                elif part.startswith("FPR@95TPR="):
                    metrics["fpr_at_95_tpr"] = part.split("=", 1)[1]

    if best_val is not None:
        metrics["best_val_selection"] = f"{best_val:.4f}"

    return metrics


def build_cmd(
    python_exec: str,
    train_script: str,
    config_path: str,
    config_name: str,
    spec: RunSpec,
) -> list[str]:
    return [
        python_exec,
        train_script,
        "--config-path",
        config_path,
        "--config-name",
        config_name,
        *spec.overrides,
    ]


def status_counts(manifest: dict[str, Any]) -> dict[str, int]:
    counts = {"pending": 0, "running": 0, "done": 0, "failed": 0}
    for info in manifest["runs"].values():
        status = info.get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default="../configs")
    parser.add_argument("--config-name", default="default_mvtec")
    parser.add_argument("--train-script", default="scripts/train.py")
    parser.add_argument("--python-exec", default=sys.executable)

    parser.add_argument(
        "--categories",
        nargs="+",
        default=["bottle", "cable", "capsule"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[45, 46])
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--launch-delay", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--out-dir", default="./grid_runs/default_mvtec_grid")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-rerun-failed", action="store_true")
    parser.add_argument("--force-rerun-done", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    results_csv_path = out_dir / "results.csv"

    ensure_results_csv(results_csv_path)
    manifest = load_manifest(manifest_path)

    specs = generate_specs(args.categories, args.seeds, args.epochs)
    if args.limit > 0:
        specs = specs[: args.limit]

    spec_map = {spec.name: spec for spec in specs}

    # initialize / refresh manifest entries
    for spec in specs:
        prev = manifest["runs"].get(spec.name)
        if prev is None:
            manifest["runs"][spec.name] = {
                "spec": asdict(spec),
                "status": "pending",
                "retries": 0,
                "returncode": None,
                "pid": None,
                "log_path": str(logs_dir / f"{spec.name}.log"),
                "started_at": None,
                "finished_at": None,
            }
        else:
            prev["spec"] = asdict(spec)
            if args.force_rerun_done and prev.get("status") == "done":
                prev["status"] = "pending"
                prev["returncode"] = None
                prev["pid"] = None
            if args.force_rerun_failed and prev.get("status") == "failed":
                prev["status"] = "pending"
                prev["returncode"] = None
                prev["pid"] = None

    # remove orphan "running" states from past interrupted sessions
    for name, info in manifest["runs"].items():
        if info.get("status") == "running":
            info["status"] = "pending"
            info["pid"] = None

    save_manifest(manifest_path, manifest)

    if args.dry_run:
        print(f"[dry-run] total specs: {len(specs)}")
        for spec in specs[:10]:
            print(spec.name, spec.overrides)
        return 0

    if args.max_parallel < 1:
        print("max-parallel deve essere >= 1")
        return 2

    running: dict[str, subprocess.Popen] = {}

    def launch_one(spec: RunSpec) -> None:
        log_path = logs_dir / f"{spec.name}.log"
        cmd = build_cmd(
            python_exec=args.python_exec,
            train_script=args.train_script,
            config_path=args.config_path,
            config_name=args.config_name,
            spec=spec,
        )

        with log_path.open("a") as lf:
            lf.write("\n" + "=" * 80 + "\n")
            lf.write(f"[launch_time] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            lf.write("[cmd] " + " ".join(shlex.quote(x) for x in cmd) + "\n")
            lf.write("=" * 80 + "\n")

        log_f = log_path.open("a")
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(Path.cwd()),
            text=True,
        )
        running[spec.name] = proc

        info = manifest["runs"][spec.name]
        info["status"] = "running"
        info["pid"] = proc.pid
        info["started_at"] = time.time()
        info["finished_at"] = None
        info["log_path"] = str(log_path)
        save_manifest(manifest_path, manifest)

    def finalize_one(name: str, proc: subprocess.Popen) -> None:
        rc = proc.returncode
        info = manifest["runs"][name]
        info["pid"] = None
        info["returncode"] = rc
        info["finished_at"] = time.time()

        spec = spec_map[name]

        if rc == 0:
            info["status"] = "done"
            metrics = parse_metrics_from_log(Path(info["log_path"]))
            append_result(
                results_csv_path,
                [
                    spec.name,
                    spec.category,
                    spec.seed,
                    spec.teacher,
                    spec.memory,
                    spec.learned_fusion,
                    spec.one_class_score,
                    "done",
                    rc,
                    metrics["run_dir"],
                    metrics["best_val_selection"],
                    metrics["test_auroc"],
                    metrics["test_auprc"],
                    metrics["test_best_f1"],
                    metrics["f1_at_val_threshold"],
                    metrics["fpr_at_95_tpr"],
                    info["log_path"],
                ],
            )
        else:
            retries = int(info.get("retries", 0))
            if retries < args.max_retries:
                info["retries"] = retries + 1
                info["status"] = "pending"
            else:
                info["status"] = "failed"
                append_result(
                    results_csv_path,
                    [
                        spec.name,
                        spec.category,
                        spec.seed,
                        spec.teacher,
                        spec.memory,
                        spec.learned_fusion,
                        spec.one_class_score,
                        "failed",
                        rc,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        info["log_path"],
                    ],
                )
        save_manifest(manifest_path, manifest)

    total = len(specs)
    pbar = tqdm(total=total, desc="grid", dynamic_ncols=True)

    def refresh_bar() -> None:
        counts = status_counts(manifest)
        done_like = counts["done"] + counts["failed"]
        pbar.n = done_like
        pbar.set_postfix(
            pending=counts["pending"],
            running=counts["running"],
            failed=counts["failed"],
        )
        pbar.refresh()

    refresh_bar()

    try:
        while True:
            # poll running
            finished_names = []
            for name, proc in list(running.items()):
                rc = proc.poll()
                if rc is not None:
                    finished_names.append(name)

            for name in finished_names:
                proc = running.pop(name)
                finalize_one(name, proc)

            # decide pending
            pending_specs = []
            for spec in specs:
                status = manifest["runs"][spec.name]["status"]
                if status == "pending":
                    pending_specs.append(spec)

            # launch
            free_slots = args.max_parallel - len(running)
            if free_slots > 0:
                for spec in pending_specs[:free_slots]:
                    launch_one(spec)
                    time.sleep(args.launch_delay)

            refresh_bar()

            counts = status_counts(manifest)
            if counts["done"] + counts["failed"] >= total and len(running) == 0:
                break

            time.sleep(args.poll_seconds)

    except KeyboardInterrupt:
        print("\n[interrupt] termino i processi attivi...")
        for _, proc in running.items():
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                pass
        time.sleep(2)
        for _, proc in running.items():
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
        raise
    finally:
        refresh_bar()
        pbar.close()

    counts = status_counts(manifest)
    print(
        f"[done] total={total} done={counts['done']} failed={counts['failed']} pending={counts['pending']}"
    )
    print(f"[manifest] {manifest_path}")
    print(f"[results]  {results_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--prefix", type=str, default="multirun_debug")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    for seed in args.seeds:
        exp_name = f"{args.prefix}_seed{seed}"
        cmd = [
            sys.executable,
            "-m",
            "scripts.train",
            f"project.seed={seed}",
            f"project.experiment_name={exp_name}",
        ] + args.overrides

        print("=" * 80)
        print("Launching:", " ".join(shlex.quote(x) for x in cmd))
        print("=" * 80)

        result = subprocess.run(cmd, cwd=project_root)
        if result.returncode != 0:
            raise SystemExit(f"Run failed for seed={seed} with code {result.returncode}")


if __name__ == "__main__":
    main()

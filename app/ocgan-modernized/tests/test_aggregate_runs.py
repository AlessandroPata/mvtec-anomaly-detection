from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_aggregate_runs_script() -> None:
    project_root = Path(__file__).resolve().parents[1]

    cmd = [
        sys.executable,
        "-m",
        "scripts.aggregate_runs",
        "--root",
        "/notebooks/storage/outputs/ocgan-modernized",
    ]

    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"aggregate_runs failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    payload = json.loads(result.stdout)
    assert "aggregates" in payload

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_basic_imports() -> None:
    import datasets.build  # noqa: F401
    import trainers.base_trainer  # noqa: F401
    import utils.env  # noqa: F401
    import utils.repro  # noqa: F401
    import utils.training  # noqa: F401

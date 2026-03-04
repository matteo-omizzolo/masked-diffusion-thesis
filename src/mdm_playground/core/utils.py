"""Shared small utilities: git, I/O, seeding."""
from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if available) PyTorch."""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

def get_git_commit_hash() -> str:
    """Return current HEAD hash, or 'UNKNOWN'."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "UNKNOWN"

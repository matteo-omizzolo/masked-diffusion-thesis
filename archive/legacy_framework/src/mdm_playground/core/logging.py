"""Unified logging: rich console logger + JSONL trajectory logger.

Two independent concerns in one module:

RunLogger
    Standard Python logger configured with a RichHandler + file handler.
    Use for human-readable run progress.

TrajectoryLogger
    Appends structured per-step dicts to a JSONL file and optionally
    saves dense NumPy arrays for offline analysis.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rich.logging import RichHandler


# ---------------------------------------------------------------------------
# Run logger (console + file)
# ---------------------------------------------------------------------------

def setup_logger(
    log_dir: Path,
    name: str = "mdm",
    level: str = "INFO",
) -> logging.Logger:
    """Configure a logger that writes to console (rich) and a log file.

    Args:
        log_dir: Directory to write ``run.log`` into.
        name:    Logger name (used to avoid duplicate handlers on re-calls).
        level:   Log level string, e.g. ``"INFO"`` or ``"DEBUG"``.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    logger.addHandler(RichHandler(rich_tracebacks=True))

    fh = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(fh)

    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Trajectory logger (JSONL + optional NumPy)
# ---------------------------------------------------------------------------

class TrajectoryLogger:
    """Persist per-step inference trajectories.

    JSONL format — one record per diffusion step::

        {run_id, method, block, step, tokens, mask_positions,
         confidence, unmask_indices, remask_indices[, generated_text]}

    NumPy format (opt-in via :meth:`save_arrays`)::

        <out_dir>/arrays/<run_id>/tokens.npy      [n_steps, L] int32
        <out_dir>/arrays/<run_id>/confidence.npy  [n_steps, L] float32
        <out_dir>/arrays/<run_id>/mask_frac.npy   [n_steps]    float32
    """

    def __init__(
        self,
        out_dir: str | Path,
        run_id: Optional[str] = None,
        method: str = "unknown",
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or f"run_{int(time.time())}"
        self.method = method
        self._jsonl_path = self.out_dir / "trajectory.jsonl"
        self._buffer: list[dict[str, Any]] = []

    @property
    def jsonl_path(self) -> Path:
        return self._jsonl_path

    def log_result(
        self,
        result: dict[str, Any],
        generated_text: Optional[str] = None,
    ) -> None:
        """Append all step records from a sampler result dict."""
        blocks: list[dict] = result.get("blocks", [])
        all_steps = [(b["block_idx"], s) for b in blocks for s in b["steps"]]

        for idx, (block_idx, step_data) in enumerate(all_steps):
            record: dict[str, Any] = {
                "run_id": self.run_id,
                "method": self.method,
                "block": block_idx,
                **step_data,
            }
            if idx == len(all_steps) - 1 and generated_text is not None:
                record["generated_text"] = generated_text
            self._buffer.append(record)

        self.flush()

    def flush(self) -> None:
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            for record in self._buffer:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._buffer.clear()

    def save_arrays(self, result: dict[str, Any]) -> None:
        arr_dir = self.out_dir / "arrays" / self.run_id
        arr_dir.mkdir(parents=True, exist_ok=True)

        all_steps = [s for b in result.get("blocks", []) for s in b["steps"]]
        if not all_steps:
            return

        np.save(arr_dir / "tokens.npy",
                np.array([s["tokens"] for s in all_steps], dtype=np.int32))
        np.save(arr_dir / "confidence.npy",
                np.array([s["confidence"] for s in all_steps], dtype=np.float32))
        np.save(arr_dir / "mask_frac.npy",
                np.array(
                    [len(s["mask_positions"]) / max(len(s["tokens"]), 1)
                     for s in all_steps],
                    dtype=np.float32,
                ))


# Legacy name (used in remedi_infer shim)
InferenceLogger = TrajectoryLogger


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_arrays(arr_dir: str | Path) -> dict[str, np.ndarray]:
    arr_dir = Path(arr_dir)
    return {
        "tokens": np.load(arr_dir / "tokens.npy"),
        "confidence": np.load(arr_dir / "confidence.npy"),
        "mask_frac": np.load(arr_dir / "mask_frac.npy"),
    }

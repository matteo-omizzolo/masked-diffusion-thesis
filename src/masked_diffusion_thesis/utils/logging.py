from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


def setup_logger(log_dir: Path, name: str = "mdt", level: str = "INFO") -> logging.Logger:
    """
    Create a logger that writes:
      - pretty logs to console
      - plain logs to logs/<run>/run.log
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    # Pretty console output
    logger.addHandler(RichHandler(rich_tracebacks=True))

    # File output
    fh = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(fh)

    logger.propagate = False
    return logger

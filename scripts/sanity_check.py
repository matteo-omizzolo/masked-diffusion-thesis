from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from masked_diffusion_thesis.utils.config import load_yaml
from masked_diffusion_thesis.utils.logging import setup_logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.get("project_name", "run")
    logs_dir = Path(cfg["paths"]["logs_dir"]) / f"{ts}_{run_name}"
    results_dir = Path(cfg["paths"]["results_dir"]) / f"{ts}_{run_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(logs_dir)
    logger.info("Sanity check script running.")
    logger.info(f"Loaded config: {cfg}")
    logger.info(f"Logs dir: {logs_dir}")
    logger.info(f"Results dir: {results_dir}")

    # Create a small artifact to prove results writing works
    (results_dir / "ok.txt").write_text("ok\n", encoding="utf-8")
    logger.info("Wrote results/..../ok.txt")


if __name__ == "__main__":
    main()

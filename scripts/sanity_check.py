from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from masked_diffusion_thesis.utils.config import load_yaml
from masked_diffusion_thesis.utils.git import get_git_commit_hash
from masked_diffusion_thesis.utils.io import save_json
from masked_diffusion_thesis.utils.logging import setup_logger
from masked_diffusion_thesis.utils.reproducibility import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # Reproducibility
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.get("project_name", "run")
    logs_dir = Path(cfg["paths"]["logs_dir"]) / f"{ts}_{run_name}"
    results_dir = Path(cfg["paths"]["results_dir"]) / f"{ts}_{run_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(logs_dir)
    commit = get_git_commit_hash()

    # Save metadata for reproducibility
    save_json(
        results_dir / "meta.json",
        {
            "git_commit": commit,
            "seed": seed,
            "config": cfg,
        },
    )

    logger.info("Sanity check script running.")
    logger.info(f"Git commit: {commit}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Logs dir: {logs_dir}")
    logger.info(f"Results dir: {results_dir}")

    (results_dir / "ok.txt").write_text("ok\n", encoding="utf-8")
    logger.info("Wrote ok.txt and meta.json")


if __name__ == "__main__":
    main()

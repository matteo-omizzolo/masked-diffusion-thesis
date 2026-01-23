from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from masked_diffusion_thesis.integrations.remdm_adapter import ReMDMAdapter, ReMDMRunConfig
from masked_diffusion_thesis.models.base_mdlm import BaseMDLM, MDLMConfig
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
    seed_everything(int(cfg.get("seed", 42)))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.get("run_name", "remdm")
    logs_dir = Path(cfg["paths"]["logs_dir"]) / f"{ts}_{run_name}"
    results_dir = Path(cfg["paths"]["results_dir"]) / f"{ts}_{run_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(logs_dir)
    commit = get_git_commit_hash()

    save_json(results_dir / "meta.json", {"git_commit": commit, "config": cfg})
    logger.info(f"Git commit: {commit}")

    model_cfg = MDLMConfig(**cfg.get("model", {}))
    model = BaseMDLM.from_checkpoint(model_cfg).to(model_cfg.device).eval()

    remdm_cfg = ReMDMRunConfig(**cfg.get("remdm", {}))
    runner = ReMDMAdapter(model=model, cfg=remdm_cfg)

    with torch.no_grad():
        out = runner.sample()

    torch.save(out, results_dir / "samples.pt")
    logger.info(f"Saved samples to {results_dir / 'samples.pt'}")
    logger.info(f"Meta: {out.get('meta')}")


if __name__ == "__main__":
    main()

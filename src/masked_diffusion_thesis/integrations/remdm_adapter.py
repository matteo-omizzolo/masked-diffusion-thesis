from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch

from masked_diffusion_thesis.models.base_mdlm import BaseMDLM


@dataclass
class ReMDMRunConfig:
    toy_mode: bool = True
    steps: int = 32
    strategy: str = "conf"
    sigma: float = 0.0


def _external_remdm_path() -> Path:
    # repo_root/external/remdm from this file:
    # src/masked_diffusion_thesis/integrations/remdm_adapter.py -> repo root is 4 parents up
    return Path(__file__).resolve().parents[3] / "external" / "remdm"


class ReMDMAdapter:
    """
    This class is YOUR stable interface.
    - Today: it can run toy mode end-to-end.
    - Next: we wire it to the real ReMDM sampler in the external/remdm submodule.
    """

    def __init__(self, model: BaseMDLM, cfg: ReMDMRunConfig):
        self.model = model
        self.cfg = cfg

    def sample(self) -> Dict[str, Any]:
        if self.cfg.toy_mode:
            return self._toy_sample()

        # Real mode: check submodule exists first
        remdm_path = _external_remdm_path()
        if not remdm_path.exists():
            raise FileNotFoundError(
                f"ReMDM submodule not found at {remdm_path}. "
                "Did you run `git submodule update --init --recursive`?"
            )

        # TODO (next step): import and call actual ReMDM sampling code here.
        # We'll implement this after we inspect external/remdm's API/entrypoints.
        raise NotImplementedError("Real ReMDM integration not wired yet.")

    def _toy_sample(self) -> Dict[str, Any]:
        device = next(self.model.parameters()).device
        B, L = 1, 16
        vocab = 256

        z_t = torch.randint(0, vocab, (B, L), device=device)
        t = torch.ones(B, device=device)
        logits = self.model.predict_token_distribution(z_t, t)
        tokens = torch.argmax(logits, dim=-1)

        return {
            "tokens": tokens.detach().cpu(),
            "meta": {"toy": True, "steps": self.cfg.steps, "strategy": self.cfg.strategy},
        }

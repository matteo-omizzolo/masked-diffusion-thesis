from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class MDLMConfig:
    checkpoint_path: Optional[str] = None
    device: str = "cpu"


class BaseMDLM(nn.Module):
    def __init__(self, cfg: MDLMConfig):
        super().__init__()
        self.cfg = cfg

    @classmethod
    def from_checkpoint(cls, cfg: MDLMConfig) -> "BaseMDLM":
        # For now: always return a toy model unless checkpoint loading is implemented.
        return ToyMDLM(cfg)

    def predict_token_distribution(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ToyMDLM(BaseMDLM):
    def __init__(self, cfg: MDLMConfig, vocab_size: int = 256):
        super().__init__(cfg)
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, 64)
        self.proj = nn.Linear(64, vocab_size)

    def predict_token_distribution(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.emb(z_t)
        return self.proj(x)

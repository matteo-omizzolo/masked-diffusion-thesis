"""PRISM model adapter (subprocess-based).

Calls ``external/PRISM/main.py`` via ``python -u -m main`` with Hydra
overrides derived from :class:`PRISMConfig`.

Like the ReMDM adapter, this is subprocess-only (no direct Python import),
which keeps it safe on macOS.  The quality/confidence signal from PRISM comes
from its token-critic / quality head, which is reported in the log files from
the upstream runner.

Currently exposes toy_mode and dry_run for local development.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .base import ForwardOutput, ModelAdapter, ModelMeta

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tiny toy model (used in toy_mode)
# ---------------------------------------------------------------------------

class _ToyPRISMModel(nn.Module):
    def __init__(self, vocab_size: int = 50257):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, 64)
        self.proj = nn.Linear(64, vocab_size)
        self.qual = nn.Linear(64, 1)  # quality head stub

    def forward(self, z_t: torch.Tensor) -> tuple:
        h = self.emb(z_t)
        return self.proj(h), torch.sigmoid(self.qual(h).squeeze(-1))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PRISMConfig:
    # Execution mode
    toy_mode: bool = True
    dry_run: bool = True

    # Basic
    mode: str = "sample_eval"
    seed: int = 1

    # Data & architecture
    data: str = "openwebtext-split"
    model_size: str = "small"
    backbone: str = "dit"
    parameterization: str = "subs"
    sequence_length: int = 1024

    # Checkpoint
    upstream_checkpoint_path: Optional[str] = None

    # Sampling
    steps: int = 1024
    strategy: str = "prism"
    num_sample_batches: int = 5000
    batch_size: int = 1
    eval_batch_size: int = 1

    # Output
    output_dir: Optional[str] = None
    wandb_offline: bool = True

    extra_overrides: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class PRISMAdapter(ModelAdapter):
    """Adapter that drives the upstream PRISM repo via subprocess.

    Args:
        cfg:           :class:`PRISMConfig`.
        run_output_dir: Directory to write outputs into.
    """

    _PRISM_ROOT = (
        Path(__file__).resolve().parents[3] / "external" / "PRISM"
    )

    def __init__(self, cfg: PRISMConfig, run_output_dir: Path):
        self.cfg = cfg
        self.run_output_dir = Path(run_output_dir)
        self._toy_model = _ToyPRISMModel()
        self._meta = ModelMeta(
            mask_token_id=50256,
            eos_token_id=50256,
            vocab_size=50257,
            model_id=cfg.upstream_checkpoint_path or "prism",
            device="cpu",
        )

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_id: str = "",
        device: str = "cpu",
        dtype=torch.float32,
        run_output_dir: Optional[Path] = None,
        cfg: Optional[PRISMConfig] = None,
        **kwargs: Any,
    ) -> "PRISMAdapter":
        if cfg is None:
            cfg = PRISMConfig(upstream_checkpoint_path=model_id or None)
        if run_output_dir is None:
            run_output_dir = Path("results/prism")
        return cls(cfg=cfg, run_output_dir=run_output_dir)

    @property
    def meta(self) -> ModelMeta:
        return self._meta

    def forward(self, x: torch.Tensor, **kwargs) -> ForwardOutput:
        """Toy-mode only: forward through a tiny stub."""
        logits, qual = self._toy_model(x)
        x0 = logits.argmax(-1)
        return ForwardOutput(token_logits=logits, confidence=qual, x0=x0)

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    def sample(self) -> Dict[str, Any]:
        if self.cfg.toy_mode:
            return self._toy_sample()
        return self._real_sample()

    def _toy_sample(self) -> Dict[str, Any]:
        z = torch.randint(0, 50257, (1, 16))
        logits, qual = self._toy_model(z)
        return {
            "tokens": logits.argmax(-1).cpu(),
            "confidence": qual.cpu(),
            "meta": {"toy": True, "steps": self.cfg.steps, "strategy": self.cfg.strategy},
        }

    def _real_sample(self) -> Dict[str, Any]:
        if not self.cfg.dry_run and not self._PRISM_ROOT.exists():
            raise FileNotFoundError(
                f"PRISM submodule not found at {self._PRISM_ROOT}. "
                "Run `git submodule update --init --recursive`."
            )
        external_dir = self.run_output_dir / "external_prism"
        external_dir.mkdir(parents=True, exist_ok=True)
        overrides = self._build_hydra_overrides(external_dir)
        cmd = [sys.executable, "-u", "-m", "main"] + overrides
        log.info(f"PRISM cmd (cwd={self._PRISM_ROOT}): {' '.join(cmd)}")

        if self.cfg.dry_run:
            log.warning("DRY RUN — skipping execution.")
            return {
                "dry_run": True,
                "command": cmd,
                "working_directory": str(self._PRISM_ROOT),
                "external_run_dir": str(external_dir),
                "meta": {"dry_run": True, "config": vars(self.cfg)},
            }

        try:
            result = subprocess.run(
                cmd, cwd=self._PRISM_ROOT, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            log.error(f"PRISM failed (rc={e.returncode})\n{e.stderr}")
            raise RuntimeError(f"PRISM subprocess failed: {e}") from e

        return {
            "external_run_dir": str(external_dir),
            "meta": {"strategy": self.cfg.strategy, "steps": self.cfg.steps},
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-1000:],
        }

    def _build_hydra_overrides(self, output_dir: Path) -> List[str]:
        cfg = self.cfg
        ov = [
            f"mode={cfg.mode}",
            f"data={cfg.data}",
            f"model={cfg.model_size}",
            f"parameterization={cfg.parameterization}",
            f"backbone={cfg.backbone}",
            f"model.length={cfg.sequence_length}",
            f"loader.batch_size={cfg.batch_size}",
            f"loader.eval_batch_size={cfg.eval_batch_size}",
            f"sampling.steps={cfg.steps}",
            f"sampling.num_sample_batches={cfg.num_sample_batches}",
            f"hydra.run.dir={output_dir}",
            f"+wandb.offline={str(cfg.wandb_offline).lower()}",
            f"seed={cfg.seed}",
        ]
        if cfg.upstream_checkpoint_path:
            ov.append(f"eval.checkpoint_path={cfg.upstream_checkpoint_path}")
        ov.extend(cfg.extra_overrides)
        return ov

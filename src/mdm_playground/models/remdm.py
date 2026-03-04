"""ReMDM model adapter (subprocess-based).

Calls ``external/remdm/main.py`` via ``python -u -m main`` with Hydra
overrides derived from :class:`ReMDMConfig`.  No Python-level import of
the upstream code is required, making this safe on macOS (CPU-only).

Three execution modes:

toy_mode=True
    Mock sampling via a tiny ToyMDLM — no external call.

dry_run=True
    Print the command that would be executed; skip actual subprocess.

(default)
    Execute upstream ReMDM as a subprocess and collect outputs.
"""
from __future__ import annotations

import json
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

class _ToyModel(nn.Module):
    def __init__(self, vocab_size: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, 64)
        self.proj = nn.Linear(64, vocab_size)

    def forward(self, z_t: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.proj(self.emb(z_t))


# ---------------------------------------------------------------------------
# Configuration dataclass (mirrors configs/remdm.yaml)
# ---------------------------------------------------------------------------

@dataclass
class ReMDMConfig:
    # Execution mode
    toy_mode: bool = True
    dry_run: bool = True

    # Basic
    mode: str = "sample_eval"
    seed: int = 1
    upstream_config: Optional[str] = None

    # Data & model architecture
    data: str = "openwebtext-split"
    model_size: str = "small"
    backbone: str = "dit"
    parameterization: str = "subs"
    sequence_length: int = 1024

    # Checkpoint
    upstream_checkpoint_path: Optional[str] = None

    # Time
    T: int = 0
    time_conditioning: bool = False

    # Sampling
    steps: int = 1024
    strategy: str = "remdm-conf"
    nucleus_p: float = 0.9
    num_sample_batches: int = 5000

    # Batch sizes
    batch_size: int = 1
    eval_batch_size: int = 1
    perplexity_batch_size: int = 1

    # Output
    generated_seqs_path: Optional[str] = None
    output_dir: Optional[str] = None

    # Wandb
    wandb_offline: bool = True

    # Strategy-specific (remdm-loop)
    eta: float = 0.02
    t_on: float = 0.55
    t_off: float = 0.05
    alpha_on: float = 0.9

    extra_overrides: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class ReMDMAdapter(ModelAdapter):
    """Adapter that drives the upstream ReMDM repo via subprocess.

    Args:
        cfg:          :class:`ReMDMConfig` instance.
        run_output_dir: Directory to write outputs into.
    """

    _REMDM_ROOT = (
        Path(__file__).resolve().parents[3] / "external" / "remdm"
    )

    def __init__(self, cfg: ReMDMConfig, run_output_dir: Path):
        self.cfg = cfg
        self.run_output_dir = Path(run_output_dir)
        self._toy_model = _ToyModel()
        self._meta = ModelMeta(
            mask_token_id=50256,
            eos_token_id=50256,
            vocab_size=50257,
            model_id=cfg.upstream_checkpoint_path or "remdm",
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
        cfg: Optional[ReMDMConfig] = None,
        **kwargs: Any,
    ) -> "ReMDMAdapter":
        if cfg is None:
            cfg = ReMDMConfig(upstream_checkpoint_path=model_id or None)
        if run_output_dir is None:
            run_output_dir = Path("results/remdm")
        return cls(cfg=cfg, run_output_dir=run_output_dir)

    @property
    def meta(self) -> ModelMeta:
        return self._meta

    def forward(self, x: torch.Tensor, **kwargs) -> ForwardOutput:
        """Toy-mode only: forward through the tiny stub model."""
        logits = self._toy_model(x)
        x0 = logits.argmax(dim=-1)
        return ForwardOutput(token_logits=logits, confidence=None, x0=x0)

    # ------------------------------------------------------------------
    # Main sample entry-point
    # ------------------------------------------------------------------

    def sample(self) -> Dict[str, Any]:
        """Run sampling (toy, dry_run, or real)."""
        if self.cfg.toy_mode:
            return self._toy_sample()
        return self._real_sample()

    def _toy_sample(self) -> Dict[str, Any]:
        z = torch.randint(0, 256, (1, 16))
        t = torch.ones(1)
        logits = self._toy_model(z, t)
        tokens = logits.argmax(-1)
        return {
            "tokens": tokens.cpu(),
            "meta": {"toy": True, "steps": self.cfg.steps, "strategy": self.cfg.strategy},
        }

    def _real_sample(self) -> Dict[str, Any]:
        if not self.cfg.dry_run and not self._REMDM_ROOT.exists():
            raise FileNotFoundError(
                f"ReMDM submodule not found at {self._REMDM_ROOT}. "
                "Run `git submodule update --init --recursive`."
            )
        external_dir = self.run_output_dir / "external_remdm"
        external_dir.mkdir(parents=True, exist_ok=True)

        overrides = self._build_hydra_overrides(external_dir)
        cmd = [sys.executable, "-u", "-m", "main"] + overrides

        log.info(f"ReMDM cmd (cwd={self._REMDM_ROOT}): {' '.join(cmd)}")

        if self.cfg.dry_run:
            log.warning("DRY RUN — skipping execution.")
            return {
                "dry_run": True,
                "command": cmd,
                "working_directory": str(self._REMDM_ROOT),
                "external_run_dir": str(external_dir),
                "meta": {"dry_run": True, "config": vars(self.cfg)},
            }

        try:
            result = subprocess.run(
                cmd, cwd=self._REMDM_ROOT, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            log.error(f"ReMDM failed (rc={e.returncode})\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
            raise RuntimeError(f"ReMDM subprocess failed: {e}") from e

        return self._collect_outputs(external_dir, result)

    def _build_hydra_overrides(self, output_dir: Path) -> List[str]:
        cfg = self.cfg
        ov = []
        ov.append(f"mode={cfg.mode}")
        ov += [
            f"loader.batch_size={cfg.batch_size}",
            f"loader.eval_batch_size={cfg.eval_batch_size}",
            f"eval.perplexity_batch_size={cfg.perplexity_batch_size}",
            f"data={cfg.data}",
            f"model={cfg.model_size}",
            f"parameterization={cfg.parameterization}",
            f"backbone={cfg.backbone}",
            f"model.length={cfg.sequence_length}",
        ]
        if cfg.upstream_checkpoint_path:
            ov.append(f"eval.checkpoint_path={cfg.upstream_checkpoint_path}")
        ov += [
            f"time_conditioning={str(cfg.time_conditioning).lower()}",
            f"+wandb.offline={str(cfg.wandb_offline).lower()}",
            f"hydra.run.dir={output_dir}",
            f"T={cfg.T}",
            f"sampling.steps={cfg.steps}",
            f"seed={cfg.seed}",
            f"sampling.num_sample_batches={cfg.num_sample_batches}",
            f"sampling.generated_seqs_path={cfg.generated_seqs_path or output_dir / 'generated_sequences.json'}",
            f"sampling.nucleus_p={cfg.nucleus_p}",
            f"sampling.sampler={cfg.strategy}",
        ]
        if cfg.strategy == "remdm-loop":
            ov += [
                f"sampling.eta={cfg.eta}",
                f"sampling.t_on={cfg.t_on}",
                f"sampling.t_off={cfg.t_off}",
                f"sampling.alpha_on={cfg.alpha_on}",
            ]
        ov.extend(cfg.extra_overrides)
        return ov

    def _collect_outputs(self, output_dir: Path, result) -> Dict[str, Any]:
        artifacts: Dict[str, Any] = {}
        seqs_path = output_dir / "generated_sequences.json"
        if seqs_path.exists():
            artifacts["generated_sequences"] = str(seqs_path)
            try:
                seqs = json.loads(seqs_path.read_text())
                if isinstance(seqs, list):
                    artifacts["num_sequences"] = len(seqs)
            except Exception:
                pass
        config_tree = output_dir / "config_tree.txt"
        if config_tree.exists():
            artifacts["config_tree"] = str(config_tree)
        return {
            "external_run_dir": str(output_dir),
            "artifacts": artifacts,
            "meta": {
                "strategy": self.cfg.strategy,
                "steps": self.cfg.steps,
                "upstream_checkpoint_path": self.cfg.upstream_checkpoint_path,
                "upstream_returncode": result.returncode,
            },
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-1000:],
        }

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from masked_diffusion_thesis.models.base_mdlm import BaseMDLM

logger = logging.getLogger(__name__)


@dataclass
class ReMDMRunConfig:
    # Execution mode
    toy_mode: bool = True
    dry_run: bool = True
    
    # Basic config
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
    
    # Time parameterization
    T: int = 0
    time_conditioning: bool = False
    
    # Sampling parameters
    steps: int = 1024
    strategy: str = "remdm-conf"
    nucleus_p: float = 0.9
    num_sample_batches: int = 5000
    
    # Batch sizes
    batch_size: int = 1
    eval_batch_size: int = 1
    perplexity_batch_size: int = 1
    
    # Output paths
    generated_seqs_path: Optional[str] = None
    output_dir: Optional[str] = None
    
    # Wandb
    wandb_offline: bool = True
    
    # Strategy-specific parameters (remdm-loop)
    eta: float = 0.02
    t_on: float = 0.55
    t_off: float = 0.05
    alpha_on: float = 0.9
    
    # Advanced
    extra_overrides: List[str] = field(default_factory=list)


def _external_remdm_path() -> Path:
    # repo_root/external/remdm from this file:
    # src/masked_diffusion_thesis/integrations/remdm_adapter.py -> repo root is 4 parents up
    return Path(__file__).resolve().parents[3] / "external" / "remdm"


class ReMDMAdapter:
    """
    This class is YOUR stable interface.
    - Toy mode: runs end-to-end with mock toy sampling.
    - Real mode: calls external/remdm/main.py via subprocess with Hydra config overrides.
    - Dry run: prints the command without executing (for local macOS development).
    """

    def __init__(self, model: BaseMDLM, cfg: ReMDMRunConfig, run_output_dir: Path):
        self.model = model
        self.cfg = cfg
        self.run_output_dir = Path(run_output_dir)

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

        return self._real_sample(remdm_path)

    def _toy_sample(self) -> Dict[str, Any]:
        """Toy mode: simple mock sampling for testing."""
        device = next(self.model.parameters()).device
        B, L = 1, 16
        vocab = 256

        z_t = torch.randint(0, vocab, (B, L), device=device)
        t = torch.ones(B, device=device)
        logits = self.model.predict_token_distribution(z_t, t)
        tokens = torch.argmax(logits, dim=-1)

        return {
            "tokens": tokens.detach().cpu(),
            "meta": {
                "toy": True,
                "steps": self.cfg.steps,
                "strategy": self.cfg.strategy,
            },
        }

    def _real_sample(self, remdm_path: Path) -> Dict[str, Any]:
        """Real mode: invoke upstream ReMDM via subprocess."""
        # 1. Create external output directory
        external_output_dir = self.run_output_dir / "external_remdm"
        external_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Build Hydra overrides from config
        overrides = self._build_hydra_overrides(external_output_dir)
        
        # 3. Build command (run from upstream repo root for Hydra config discovery)
        # Use python -u -m main (relative) since cwd=external/remdm
        # -u = unbuffered output (matches upstream scripts)
        cmd = [sys.executable, "-u", "-m", "main"] + overrides
        
        logger.info(f"Running upstream ReMDM from: {remdm_path}")
        logger.info(f"External output dir: {external_output_dir}")
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"Working directory: {remdm_path}")
        
        # 4. Dry run mode: only print, don't execute
        if self.cfg.dry_run:
            logger.warning("DRY RUN MODE: Command above would be executed on Linux/CUDA.")
            return {
                "dry_run": True,
                "command": cmd,
                "working_directory": str(remdm_path),
                "external_run_dir": str(external_output_dir),
                "meta": {
                    "dry_run": True,
                    "remdm_path": str(remdm_path),
                    "config": vars(self.cfg),
                },
            }
        
        # 5. Execute subprocess
        try:
            result = subprocess.run(
                cmd,
                cwd=remdm_path,
                check=True,
                capture_output=True,
                text=True,
            )
            
            logger.info("Upstream ReMDM completed successfully.")
            logger.info(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                logger.warning(f"STDERR:\n{result.stderr}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Upstream ReMDM failed with exit code {e.returncode}")
            logger.error(f"STDOUT:\n{e.stdout}")
            logger.error(f"STDERR:\n{e.stderr}")
            raise RuntimeError(f"ReMDM subprocess failed: {e}") from e
        
        # 6. Collect outputs
        return self._collect_outputs(external_output_dir, result)

    def _build_hydra_overrides(self, output_dir: Path) -> List[str]:
        """
        Build Hydra CLI overrides from config.
        
        Order matches upstream scripts (external/remdm/scripts/remdm-conf.sh).
        """
        cfg = self.cfg
        overrides = []
        
        # Mode
        overrides.append(f"mode={cfg.mode}")
        
        # Batch sizes (early, like in upstream scripts)
        overrides.append(f"loader.batch_size={cfg.batch_size}")
        overrides.append(f"loader.eval_batch_size={cfg.eval_batch_size}")
        overrides.append(f"eval.perplexity_batch_size={cfg.perplexity_batch_size}")
        
        # Data & model
        overrides.append(f"data={cfg.data}")
        overrides.append(f"model={cfg.model_size}")
        overrides.append(f"parameterization={cfg.parameterization}")
        overrides.append(f"backbone={cfg.backbone}")
        overrides.append(f"model.length={cfg.sequence_length}")
        
        # Checkpoint
        if cfg.upstream_checkpoint_path:
            overrides.append(f"eval.checkpoint_path={cfg.upstream_checkpoint_path}")
        else:
            logger.warning("No upstream_checkpoint_path provided. Upstream may fail if required.")
        
        # Time parameterization
        overrides.append(f"time_conditioning={str(cfg.time_conditioning).lower()}")
        
        # Wandb
        overrides.append(f"+wandb.offline={str(cfg.wandb_offline).lower()}")
        
        # Hydra run directory
        overrides.append(f"hydra.run.dir={output_dir}")
        
        # T parameter
        overrides.append(f"T={cfg.T}")
        
        # Sampling parameters
        overrides.append(f"sampling.steps={cfg.steps}")
        
        # Seed
        overrides.append(f"seed={cfg.seed}")
        
        # Sampling batches
        overrides.append(f"sampling.num_sample_batches={cfg.num_sample_batches}")
        
        # Generated sequences path
        if cfg.generated_seqs_path:
            overrides.append(f"sampling.generated_seqs_path={cfg.generated_seqs_path}")
        else:
            # Auto-generate under output_dir
            generated_seqs_path = output_dir / "generated_sequences.json"
            overrides.append(f"sampling.generated_seqs_path={generated_seqs_path}")
        
        # Nucleus sampling
        overrides.append(f"sampling.nucleus_p={cfg.nucleus_p}")
        
        # Sampler (strategy)
        overrides.append(f"sampling.sampler={cfg.strategy}")
        
        # Strategy-specific parameters (remdm-loop)
        if cfg.strategy == "remdm-loop":
            overrides.append(f"sampling.eta={cfg.eta}")
            overrides.append(f"sampling.t_on={cfg.t_on}")
            overrides.append(f"sampling.t_off={cfg.t_off}")
            overrides.append(f"sampling.alpha_on={cfg.alpha_on}")
        
        # Add any extra overrides from config
        overrides.extend(cfg.extra_overrides)
        
        return overrides

    def _collect_outputs(self, output_dir: Path, result: subprocess.CompletedProcess) -> Dict[str, Any]:
        """
        Collect outputs from upstream ReMDM run.
        
        TODO: Adjust based on actual upstream output structure.
        Common outputs:
        - generated_sequences.json (if sampling.generated_seqs_path was set)
        - checkpoints/ (if training)
        - logs/ (Hydra logs)
        - config_tree.txt (config dump)
        """
        artifacts = {}
        
        # Look for generated sequences
        generated_seqs_path = output_dir / "generated_sequences.json"
        if generated_seqs_path.exists():
            logger.info(f"Found generated sequences at: {generated_seqs_path}")
            artifacts["generated_sequences"] = str(generated_seqs_path)
            
            # Try to load and get basic stats
            try:
                with open(generated_seqs_path, "r") as f:
                    seqs = json.load(f)
                    if isinstance(seqs, list):
                        artifacts["num_sequences"] = len(seqs)
            except Exception as e:
                logger.warning(f"Could not parse generated sequences: {e}")
        else:
            logger.warning(f"Generated sequences not found at: {generated_seqs_path}")
        
        # Look for config dump
        config_tree = output_dir / "config_tree.txt"
        if config_tree.exists():
            artifacts["config_tree"] = str(config_tree)
        
        # Look for any .pt or .ckpt files
        for ext in ["*.pt", "*.ckpt"]:
            for f in output_dir.rglob(ext):
                artifacts.setdefault("checkpoints", []).append(str(f))
        
        return {
            "external_run_dir": str(output_dir),
            "artifacts": artifacts,
            "meta": {
                "strategy": self.cfg.strategy,
                "steps": self.cfg.steps,
                "upstream_checkpoint_path": self.cfg.upstream_checkpoint_path,
                "upstream_returncode": result.returncode,
            },
            "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,  # Last 2000 chars
            "stderr": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr,  # Last 1000 chars
        }


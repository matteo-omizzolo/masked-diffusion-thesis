"""Real MDLM backend for Protocol A + B experiments.

Loads the MDLM checkpoint via Lightning and runs step-by-step generation,
instrumenting each predictor step and branching for per-step corrector loops.

The corrector at step t is defined as:
  One re-run of the MDLM predictor *at the same noise level t* (same dt=0),
  which resamples masked positions from p_x0(· | x_t).  This is a
  "single Gibbs-style sweep" corrector: it leaves unmasked positions fixed
  and re-draws each masked position independently from the model's x0 posterior
  at that noise level.

Usage (from run_phase1_pilot.py, no --surrogate):
  gen = MDLMGenerator(checkpoint='/home/3316152/mdm/checkpoints/mdlm.ckpt', T=64)
  y_base   = gen.run_base(seed=0)
  y_branch = gen.run_branch(t_corrected=30, seed=0)
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# PyTorch >= 2.6 changed the torch.load default to weights_only=True.
# Old checkpoints contain numpy scalars whose __module__ is 'numpy._core.multiarray'
# but are pickled as 'numpy.core.multiarray.scalar', causing a name mismatch that
# breaks add_safe_globals.  Force weights_only=False for all torch.load calls in this
# process (safe because we only load trusted local checkpoints).
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# ---------------------------------------------------------------------------
# Import from external/remdm (subprocess-free, direct Python import)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[4]  # repo root
_REMDM_DIR = _REPO_ROOT / "external" / "remdm"

if str(_REMDM_DIR) not in sys.path:
    sys.path.insert(0, str(_REMDM_DIR))


def _load_diffusion_model(checkpoint_path: str, steps: int = 64, device: str = "cuda"):
    """Load the MDLM Diffusion model from a Lightning checkpoint.

    WHY we read config from the checkpoint rather than constructing a minimal one:
    diffusion.Diffusion.__init__ accesses config keys unconditionally that are present
    in the full training config but absent from any hand-written minimal dict.  The
    immediate failures are config.eval.gen_ppl_eval_model_name_or_path (line 83) and
    config.model.cond_dim / config.model.scale_by_sigma (DIT backbone init).  All of
    these live in the checkpoint's hyper_parameters['config'].

    We override only inference-specific settings; all architecture/training parameters
    are taken verbatim from the checkpoint.
    """
    import omegaconf
    import diffusion as remdm_diffusion

    # --- 1. Read the full training config from the checkpoint ---
    print(f"  [_load_diffusion_model] Reading checkpoint config from: {checkpoint_path}")
    raw_ckpt = torch.load(checkpoint_path, map_location="cpu")
    saved_cfg = raw_ckpt["hyper_parameters"]["config"]

    cfg_dict = omegaconf.OmegaConf.to_container(
        saved_cfg, resolve=False, throw_on_missing=False
    )

    # --- 2. Keys absent from training config but required by current diffusion.py ---
    cfg_dict.setdefault("T", 0)
    cfg_dict.setdefault("subs_masking", False)

    sampling = cfg_dict.setdefault("sampling", {})
    sampling.setdefault("sampler", "mdlm")
    sampling.setdefault("nucleus_p", 1.0)
    sampling.setdefault("eta", 0.0)
    sampling.setdefault("t_on", 0.0)
    sampling.setdefault("t_off", 0.0)
    sampling.setdefault("alpha_on", 0.0)
    sampling.setdefault("dfm", False)
    sampling.setdefault("semi_ar", False)
    sampling.setdefault("stride_length", 1)
    sampling.setdefault("num_strides", 1)

    # --- 3. Inference settings override ---
    cfg_dict["eval"]["compute_generative_perplexity"] = False
    cfg_dict["eval"]["generate_samples"] = False
    cfg_dict["eval"]["compute_perplexity_on_sanity"] = False
    cfg_dict["eval"]["checkpoint_path"] = checkpoint_path
    cfg_dict["sampling"]["predictor"] = "ddpm_cache"
    cfg_dict["sampling"]["steps"] = steps
    cfg_dict["checkpointing"]["save_dir"] = "/tmp/mdlm_phase1"

    cfg = omegaconf.OmegaConf.create(cfg_dict)

    omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd, replace=True)
    omegaconf.OmegaConf.register_new_resolver(
        "device_count", torch.cuda.device_count, replace=True
    )
    omegaconf.OmegaConf.register_new_resolver("eval", eval, replace=True)
    omegaconf.OmegaConf.register_new_resolver(
        "div_up", lambda x, y: (x + y - 1) // y, replace=True
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = cfg.model.length

    model = remdm_diffusion.Diffusion.load_from_checkpoint(
        checkpoint_path,
        tokenizer=tokenizer,
        config=cfg,
        map_location=device,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer, cfg


# ---------------------------------------------------------------------------
# MDLMGenerator — implements Generator protocol
# ---------------------------------------------------------------------------

class MDLMGenerator:
    """Real MDLM generator for Protocol A + B experiments.

    Loads the MDLM checkpoint and runs instrumented step-by-step generation.
    Implements the same interface as SurrogateGenerator.

    Parameters
    ----------
    checkpoint : str
        Path to the MDLM Lightning checkpoint (e.g. ~/mdm/checkpoints/mdlm.ckpt).
    T : int
        Number of predictor steps. Default 64 for pilot; 128 for full run.
    device : str
        Torch device. Default 'cuda'.
    seq_len : int
        Sequence length. Must match the checkpoint (default 1024).
    ref_model_name : str
        HuggingFace model used to compute neg-NLL reference scores.
        Default 'gpt2' (fast; for higher quality use 'gpt2-large').
    """

    def __init__(
        self,
        checkpoint: str,
        T: int = 64,
        device: str = "cuda",
        seq_len: int = 1024,
        ref_model_name: str = "gpt2",
    ):
        self.T = T
        self.device = device
        self.seq_len = seq_len

        print(f"[MDLMGenerator] Loading checkpoint: {checkpoint}")
        self.model, self.tokenizer, self.cfg = _load_diffusion_model(
            checkpoint, steps=T, device=device
        )
        self.mask_id = self.model.mask_index
        self.eps = 1e-5

        print(f"[MDLMGenerator] Loading reference scorer: {ref_model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer as AutoTok
        self._ref_tok = AutoTok.from_pretrained(ref_model_name)
        self._ref_lm = AutoModelForCausalLM.from_pretrained(ref_model_name).to(device)
        self._ref_lm.eval()
        print("[MDLMGenerator] Ready.")

    # ------------------------------------------------------------------
    # Internal: instrumented sampling loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_loop(
        self,
        seed: int,
        corrector_at_t: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the MDLM predictor loop, optionally applying one corrector loop.

        Parameters
        ----------
        seed : int
            Random seed (for CRN across branches).
        corrector_at_t : int or None
            If set, apply one corrector loop immediately after predictor step
            `corrector_at_t` (0-indexed). All subsequent randomness uses the
            same seed (CRN).
        """
        torch.manual_seed(seed)
        B = 1
        L = self.cfg.model.length

        # Initial state: all masked
        x = self.mask_id * torch.ones(B, L, dtype=torch.long, device=self.device)
        timesteps = torch.linspace(1, self.eps, self.T + 1, device=self.device)
        dt = (1 - self.eps) / self.T

        p_x0_cache = None
        confident_score = (
            -torch.ones(B, L, device=self.device).to(torch.bfloat16) * float("inf")
        )

        per_step_signals: List[Dict] = []

        for step_i in range(self.T):
            t = timesteps[step_i] * torch.ones(B, 1, device=self.device)

            # Predictor step (standard MDLM caching update)
            p_x0_cache, x, confident_score = self.model._ddpm_caching_update(
                x, t, dt, p_x0=p_x0_cache, conf=confident_score
            )

            # Record per-step signals (on the post-predictor state)
            signals = self._extract_signals(x, t)
            per_step_signals.append({"t": step_i, **signals})

            # Corrector: one additional forward pass at the same noise level,
            # resampling masked positions from the posterior p_x0(· | x_t).
            if corrector_at_t is not None and step_i == corrector_at_t:
                x = self._apply_corrector(x, t)
                # Invalidate the cache since x changed
                p_x0_cache = None

        return {
            "tokens": x[0].cpu().numpy().astype(np.int32),
            "neg_nll": float(self._compute_neg_nll(x[0])),
            "per_step_signals": per_step_signals,
            "seed": seed,
        }

    @torch.no_grad()
    def _apply_corrector(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """One corrector loop: resample masked positions from p_x0(· | x_t).

        This is a Gibbs-style sweep: for each masked position independently,
        draw a new token from the model's x0 posterior at the current t.
        Unmasked positions are left unchanged.
        """
        sigma_t, _ = self.model.noise(t)
        log_p_x0 = self.model.forward(x, sigma_t)  # (B, L, V)
        p_x0 = log_p_x0.exp()

        # Sample from the posterior for masked positions
        masked = x == self.mask_id  # (B, L)
        flat_probs = p_x0[masked]   # (n_masked, V)
        sampled = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)  # (n_masked,)

        x_new = x.clone()
        x_new[masked] = sampled
        return x_new

    @torch.no_grad()
    def _extract_signals(self, x: torch.Tensor, t: torch.Tensor) -> Dict[str, float]:
        """Extract per-step aggregate signals from the current state."""
        sigma_t, _ = self.model.noise(t)
        log_p_x0 = self.model.forward(x, sigma_t)  # (B, L, V)
        p_x0 = log_p_x0.exp()[0]  # (L, V)

        revisable = (x[0] != self.mask_id)  # unmasked positions
        n_rev = int(revisable.sum())
        D = x.shape[1]

        if n_rev == 0:
            return {
                "entropy": 0.0,
                "inverse_margin": 0.0,
                "quality_mass_proxy": 0.0,
                "unmasked_fraction": 0.0,
                "n_revisable": 0,
            }

        p_rev = p_x0[revisable].float()  # (n_rev, V)
        # Entropy
        H = -(p_rev * (p_rev + 1e-12).log()).sum(-1).mean().item()
        # Inverse margin
        top2 = p_rev.topk(2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).mean().item()
        # Quality proxy
        p_argmax = p_rev.max(-1).values.mean().item()

        return {
            "entropy": float(H),
            "inverse_margin": float(1.0 - margin),
            "quality_mass_proxy": float(1.0 - p_argmax),
            "unmasked_fraction": float(n_rev / D),
            "n_revisable": n_rev,
        }

    @torch.no_grad()
    def _compute_neg_nll(self, tokens: torch.Tensor) -> float:
        """Compute negative NLL of token sequence under the reference LM.

        Returns neg-NLL per token (higher = better quality).
        Uses the MDLM checkpoint's GPT-2 tokenizer; decodes to text then
        re-encodes for the reference scorer.
        """
        # Decode MDLM tokens → text (strip any mask tokens)
        token_ids = tokens.cpu().tolist()
        # Remove mask tokens
        token_ids = [t for t in token_ids if t != self.mask_id]
        if not token_ids:
            return 0.0
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        if not text.strip():
            return 0.0

        # Re-encode with reference LM tokenizer
        enc = self._ref_tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        input_ids = enc["input_ids"]
        if input_ids.shape[1] < 2:
            return 0.0

        with torch.no_grad():
            outputs = self._ref_lm(input_ids, labels=input_ids)
        nll = outputs.loss.item()  # cross-entropy loss = NLL per token
        return float(-nll)  # negative NLL; higher = better

    # ------------------------------------------------------------------
    # Generator protocol (same as SurrogateGenerator)
    # ------------------------------------------------------------------

    def run_base(self, seed: int = 0) -> Dict[str, Any]:
        """Run base trajectory (no correction)."""
        return self._run_loop(seed=seed, corrector_at_t=None)

    def run_branch(self, t_corrected: int, seed: int = 0) -> Dict[str, Any]:
        """Run branch: one corrector loop at step t_corrected, same seed."""
        result = self._run_loop(seed=seed, corrector_at_t=t_corrected)
        result["t_corrected"] = t_corrected
        return result

    def run_with_schedule(
        self, allocation: Dict[int, int], seed: int = 0
    ) -> Dict[str, Any]:
        """Run with a multi-step allocation (one corrector loop per step in S).

        Note: For Protocol B, we need G(S) = F(y^S) - F(y_base) with the
        *joint* effect of all corrector steps. This runs the full loop,
        applying the corrector at each step t ∈ allocation.
        """
        torch.manual_seed(seed)
        B = 1
        L = self.cfg.model.length

        x = self.mask_id * torch.ones(B, L, dtype=torch.long, device=self.device)
        timesteps = torch.linspace(1, self.eps, self.T + 1, device=self.device)
        dt = (1 - self.eps) / self.T

        p_x0_cache = None
        confident_score = (
            -torch.ones(B, L, device=self.device).to(torch.bfloat16) * float("inf")
        )

        for step_i in range(self.T):
            t = timesteps[step_i] * torch.ones(B, 1, device=self.device)
            p_x0_cache, x, confident_score = self.model._ddpm_caching_update(
                x, t, dt, p_x0=p_x0_cache, conf=confident_score
            )
            if step_i in allocation:
                x = self._apply_corrector(x, t)
                p_x0_cache = None

        return {
            "tokens": x[0].cpu().numpy().astype(np.int32),
            "neg_nll": float(self._compute_neg_nll(x[0])),
            "schedule_steps": sorted(allocation.keys()),
            "seed": seed,
        }

    def signal_trace(self, seed: int = 0) -> List[Dict]:
        """Return per-step signals from a base run."""
        result = self.run_base(seed=seed)
        return result["per_step_signals"]

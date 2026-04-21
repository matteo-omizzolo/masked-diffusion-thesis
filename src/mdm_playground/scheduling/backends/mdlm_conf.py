"""MDLM-conf backend: confidence-guided partial-resample corrector.

Fixes both bugs from the original MDLM corrector (job 478600):

  Bug #1 — signals over UNMASKED positions → always 0
    Fix: _extract_signals uses masked positions (x == mask_id), which are the
    positions the corrector can still change.

  Bug #2 — corrector resamples ALL masked positions → universally harmful
    Fix: _apply_corrector resamples only the K lowest-confidence masked
    positions (those with smallest max p_x0 probability), i.e. the positions
    the model is most uncertain about.

Corrector kernel:
  At step t, after the predictor, identify the n_masked still-masked positions.
  Rank them by confidence = max_{v} p_x0(v | x_t).
  Resample the top_k least-confident positions from the full p_x0 distribution.
  Unmasked positions are left unchanged.  This targets the positions most likely
  to benefit from an alternative draw.

Signal interpretation:
  Entropy and inverse-margin over masked positions now measure the corrector's
  opportunity: high entropy → many uncertain positions → larger expected gain from
  the K-resample.  This directly supports the thesis hypothesis that aggregate
  trajectory signals predict the marginal value of a corrective loop.

Usage:
  gen = MDLMConfGenerator(
      checkpoint='path/to/mdlm.ckpt',
      T=64,
      top_k=20,
  )
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

# PyTorch >= 2.6: force weights_only=False for trusted local checkpoints
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

_REPO_ROOT = Path(__file__).resolve().parents[4]
_REMDM_DIR = _REPO_ROOT / "external" / "remdm"
if str(_REMDM_DIR) not in sys.path:
    sys.path.insert(0, str(_REMDM_DIR))


def _load_diffusion_model(checkpoint_path: str, steps: int = 64, device: str = "cuda"):
    """Load MDLM checkpoint.  Identical to mdlm.py — reads checkpoint config directly."""
    import omegaconf
    import diffusion as remdm_diffusion

    print(f"  [_load_diffusion_model] Reading checkpoint config from: {checkpoint_path}")
    raw_ckpt = torch.load(checkpoint_path, map_location="cpu")
    saved_cfg = raw_ckpt["hyper_parameters"]["config"]

    cfg_dict = omegaconf.OmegaConf.to_container(
        saved_cfg, resolve=False, throw_on_missing=False
    )

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

    cfg_dict["eval"]["compute_generative_perplexity"] = False
    cfg_dict["eval"]["generate_samples"] = False
    cfg_dict["eval"]["compute_perplexity_on_sanity"] = False
    cfg_dict["eval"]["checkpoint_path"] = checkpoint_path
    cfg_dict["sampling"]["predictor"] = "ddpm_cache"
    cfg_dict["sampling"]["steps"] = steps
    cfg_dict["checkpointing"]["save_dir"] = "/tmp/mdlm_conf_phase1"

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


class MDLMConfGenerator:
    """MDLM-conf generator: confidence-guided K-resample corrector on masked positions.

    Parameters
    ----------
    checkpoint : str
        Path to the MDLM Lightning checkpoint.
    T : int
        Number of predictor steps.
    top_k : int
        Number of least-confident masked positions to resample per corrector call.
        Capped at n_masked if fewer positions remain.  Default 20.
    device : str
        Torch device.  Default 'cuda'.
    ref_model_name : str
        HuggingFace model for neg-NLL scoring.  Default 'gpt2'.
    """

    def __init__(
        self,
        checkpoint: str,
        T: int = 64,
        top_k: int = 20,
        device: str = "cuda",
        ref_model_name: str = "gpt2",
    ):
        self.T = T
        self.top_k = top_k
        self.device = device

        print(f"[MDLMConfGenerator] Loading checkpoint: {checkpoint}")
        self.model, self.tokenizer, self.cfg = _load_diffusion_model(
            checkpoint, steps=T, device=device
        )
        self.mask_id = self.model.mask_index
        self.eps = 1e-5
        self.seq_len = self.cfg.model.length

        print(f"[MDLMConfGenerator] Loading reference scorer: {ref_model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer as AutoTok
        self._ref_tok = AutoTok.from_pretrained(ref_model_name)
        self._ref_lm = AutoModelForCausalLM.from_pretrained(ref_model_name).to(device)
        self._ref_lm.eval()
        print(f"[MDLMConfGenerator] Ready.  T={T}, top_k={top_k}")

    def corrector_description(self) -> str:
        return (
            f"MDLMConf corrector: resample top_k={self.top_k} "
            f"least-confident masked positions per step"
        )

    # ------------------------------------------------------------------
    # Core: instrumented sampling loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_loop(
        self,
        seed: int,
        corrector_at_t: Optional[int] = None,
        record_signals: bool = True,
    ) -> Dict[str, Any]:
        """Run the MDLM predictor loop, optionally applying the conf-corrector once.

        p_x0 is computed once per step (either from the ddpm cache when available,
        or via an extra forward pass), and shared between signal extraction and
        the corrector — no duplicate NFEs.
        """
        torch.manual_seed(seed)
        B = 1
        L = self.seq_len

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

            p_x0_cache, x, confident_score = self.model._ddpm_caching_update(
                x, t, dt, p_x0=p_x0_cache, conf=confident_score
            )

            # Obtain p_x0 — reuse cache when available (saves 1 NFE most steps)
            if p_x0_cache is not None:
                p_x0 = p_x0_cache  # (B, L, V), log-probs → need .exp()
                # _ddpm_caching_update stores log-probs in the cache
                p_x0_probs = p_x0.exp()
            else:
                sigma_t, _ = self.model.noise(t)
                log_p = self.model.forward(x, sigma_t)  # (B, L, V) log-probs
                p_x0_probs = log_p.exp()

            if record_signals:
                signals = self._extract_signals(x, p_x0_probs)
                per_step_signals.append({"t": step_i, **signals})

            if corrector_at_t is not None and step_i == corrector_at_t:
                x = self._apply_corrector(x, p_x0_probs)
                p_x0_cache = None  # cache invalid after x changes

        return {
            "tokens": x[0].cpu().numpy().astype(np.int32),
            "neg_nll": float(self._compute_neg_nll(x[0])),
            "per_step_signals": per_step_signals,
            "seed": seed,
        }

    def _extract_signals(
        self, x: torch.Tensor, p_x0_probs: torch.Tensor
    ) -> Dict[str, float]:
        """Signals over the corrector's action set — the top_k lowest-confidence
        masked positions (NOT all masked positions).

        Option (b) fix: proxy ψ(s_t) must summarise the distribution the
        corrector actually samples from, otherwise Theorem A's ε is not the
        relevant residual.  See PROSECO_PREFLIGHT_STATUS.md §3.
        """
        masked_idx = (x[0] == self.mask_id).nonzero(as_tuple=True)[0]  # (n_masked,)
        n_masked = int(masked_idx.numel())
        D = x.shape[1]
        unmasked_fraction = float((D - n_masked) / D)

        if n_masked == 0:
            return {
                "entropy": 0.0,
                "inverse_margin": 0.0,
                "quality_mass_proxy": 0.0,
                "unmasked_fraction": unmasked_fraction,
                "n_masked": 0,
                "n_action": 0,
            }

        # Action set: top_k lowest-confidence masked positions.
        p_masked_all = p_x0_probs[0, masked_idx].float()       # (n_masked, V)
        confidence = p_masked_all.max(-1).values               # (n_masked,)
        k = min(self.top_k, n_masked)
        _, low_conf_local = confidence.topk(k, largest=False)  # (k,)
        p_m = p_masked_all[low_conf_local]                     # (k, V)

        H = -(p_m * (p_m + 1e-12).log()).sum(-1).mean().item()
        top2 = p_m.topk(2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).mean().item()
        p_argmax = p_m.max(-1).values.mean().item()

        return {
            "entropy": float(H),
            "inverse_margin": float(1.0 - margin),
            "quality_mass_proxy": float(1.0 - p_argmax),
            "unmasked_fraction": unmasked_fraction,
            "n_masked": n_masked,
            "n_action": int(k),
        }

    def _apply_corrector(
        self, x: torch.Tensor, p_x0_probs: torch.Tensor
    ) -> torch.Tensor:
        """Resample top_k least-confident masked positions from p_x0.

        p_x0_probs: (B, L, V) probabilities.
        Confidence at position l = max_v p_x0_probs[0, l, v].
        The K positions with lowest confidence are resampled via multinomial.
        Unmasked and high-confidence masked positions are left unchanged.
        """
        masked_idx = (x[0] == self.mask_id).nonzero(as_tuple=True)[0]  # (n_masked,)
        n_masked = masked_idx.shape[0]

        if n_masked == 0:
            return x

        k = min(self.top_k, n_masked)

        p_masked = p_x0_probs[0, masked_idx]     # (n_masked, V)
        confidence = p_masked.max(-1).values      # (n_masked,)

        # Indices into masked_idx of the k least-confident positions
        _, low_conf_local = confidence.topk(k, largest=False)
        target_positions = masked_idx[low_conf_local]  # (k,)

        flat_probs = p_x0_probs[0, target_positions]   # (k, V)
        sampled = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)  # (k,)

        x_new = x.clone()
        x_new[0, target_positions] = sampled
        return x_new

    @torch.no_grad()
    def _compute_neg_nll(self, tokens: torch.Tensor) -> float:
        token_ids = [t for t in tokens.cpu().tolist() if t != self.mask_id]
        if not token_ids:
            return 0.0
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        if not text.strip():
            return 0.0

        enc = self._ref_tok(
            text, return_tensors="pt", truncation=True, max_length=512,
        ).to(self.device)
        input_ids = enc["input_ids"]
        if input_ids.shape[1] < 2:
            return 0.0

        outputs = self._ref_lm(input_ids, labels=input_ids)
        return float(-outputs.loss.item())

    # ------------------------------------------------------------------
    # Generator protocol
    # ------------------------------------------------------------------

    def run_base(self, seed: int = 0) -> Dict[str, Any]:
        return self._run_loop(seed=seed, corrector_at_t=None, record_signals=True)

    def run_branch(self, t_corrected: int, seed: int = 0) -> Dict[str, Any]:
        result = self._run_loop(
            seed=seed, corrector_at_t=t_corrected, record_signals=False
        )
        result["t_corrected"] = t_corrected
        return result

    def run_with_schedule(
        self, allocation: Dict[int, int], seed: int = 0
    ) -> Dict[str, Any]:
        """Run with a multi-step allocation dict {step: n_corrector_loops}."""
        torch.manual_seed(seed)
        B = 1
        L = self.seq_len

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
                if p_x0_cache is not None:
                    p_x0_probs = p_x0_cache.exp()
                else:
                    sigma_t, _ = self.model.noise(t)
                    p_x0_probs = self.model.forward(x, sigma_t).exp()
                for _ in range(allocation[step_i]):
                    x = self._apply_corrector(x, p_x0_probs)
                p_x0_cache = None

        return {
            "tokens": x[0].cpu().numpy().astype(np.int32),
            "neg_nll": float(self._compute_neg_nll(x[0])),
            "schedule_steps": sorted(allocation.keys()),
            "seed": seed,
        }

    def signal_trace(self, seed: int = 0) -> List[Dict]:
        return self.run_base(seed=seed)["per_step_signals"]

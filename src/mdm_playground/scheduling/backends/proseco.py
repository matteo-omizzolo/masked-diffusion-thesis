"""ProSeCo-style corrector backend for Protocol A + B experiments.

Implements ProSeCo's annealed iterative-refinement corrector on top of the
existing MDLM checkpoint, which is already validated on the HPC cluster.

**What makes this different from the MDLM heuristic backend:**

  MDLM heuristic: resample *masked* positions from p(x_0 | x_t) in one Gibbs
    sweep.  Action set = masked positions.  The heuristic proved harmful at
    all steps (all Δ_t ≤ 0) because a full resample with <5% unmasked context
    destroys sequence coherence.

  ProSeCo corrector: take x̂_0 = argmax p_θ(x_0 | x_t), then run
    corrector_steps backbone calls at decreasing noise levels (annealed
    refinement), and apply the refined sequence to *already-unmasked*
    positions only.  Action set = unmasked (committed) positions.  This
    corrects committed tokens rather than blindly resampling uncommitted ones.

**Corrector mechanism (exactly as in ProSeCo's _diffusion_sample):**

  1. After predictor step i: x̂_0 = argmax(p_x0_cache)          [free — cache hit]
  2. corrector_x = x̂_0
  3. For j = 0 … corrector_steps-1:
       tau = linspace(1, eps, corrector_steps+1)[j]              [high→low noise]
       sigma_τ, _ = model.noise(tau)
       log_p = model.forward(corrector_x, sigma_τ)               [1 NFE]
       corrector_x = argmax(log_p)
  4. x_new = where(x != mask_id, corrector_x, x)                 [unmasked only]

**Signals (over UNMASKED positions — corrector's action set):**
  At step i, signals are computed from the predictor's p_x0_cache over
  positions that are currently committed in x (x != mask_id).  These are
  the positions the corrector will actually revise.

  - entropy: mean H(p_θ(x_i | x_t)) over committed positions
  - inverse_margin: 1 − mean(p_1 − p_2) over committed positions
  - quality_mass_proxy: 1 − mean(p_argmax) over committed positions
  - unmasked_fraction: |committed| / L (context scalar)
  - n_revisable: |committed| (raw count)
  - n_masked: |masked| positions remaining

**Note on checkpoint:**
  Uses the MDLM checkpoint (/home/3316152/mdm/checkpoints/mdlm.ckpt) loaded
  via the remdm Diffusion class.  This is the same model as the MDLM backend.
  The key difference is the corrector strategy, not the backbone weights.

  The proseco-owt checkpoint (kuleshov-group/proseco-owt), trained with
  corrector co-training loss, would give stronger correction quality.  Loading
  it requires internet access and HuggingFace download (~500MB).  A separate
  hf_dit loading path is provided at the bottom of this file.

Usage (from run_phase1_proseco.py):
  gen = ProSeCoGenerator(checkpoint='...', T=64, corrector_steps=2)
  y_base   = gen.run_base(seed=0)
  y_branch = gen.run_branch(t_corrected=30, seed=0)
  y_sched  = gen.run_with_schedule({10: 1, 30: 1, 50: 1}, seed=0)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# PyTorch >= 2.6 changed the torch.load default to weights_only=True.
# Old checkpoints contain numpy scalars whose __module__ is 'numpy._core.multiarray'
# but are pickled as 'numpy.core.multiarray.scalar', causing a name mismatch that
# breaks add_safe_globals.  The only reliable fix is the same monkey-patch used in
# external/remdm/main.py: force weights_only=False for all torch.load calls in this
# process (safe because we only load trusted local checkpoints).
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# ---------------------------------------------------------------------------
# Import from external/remdm (same as MDLM backend)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[4]
_REMDM_DIR = _REPO_ROOT / "external" / "remdm"

if str(_REMDM_DIR) not in sys.path:
    sys.path.insert(0, str(_REMDM_DIR))


def _load_diffusion_model(checkpoint_path: str, steps: int = 64, device: str = "cuda"):
    """Load MDLM model from checkpoint using the checkpoint's own training config.

    WHY we read config from the checkpoint rather than constructing a minimal one:
    diffusion.Diffusion.__init__ accesses config keys unconditionally that are present
    in the full training config but absent from any hand-written minimal dict.  The most
    immediate failures are:

      - config.eval.gen_ppl_eval_model_name_or_path   (line 83 of diffusion.py)
      - config.model.cond_dim / config.model.scale_by_sigma  (DIT backbone init)

    These keys live in the checkpoint's hyper_parameters['config'] — the OmegaConf
    DictConfig that was passed to Diffusion.__init__ at training time.  Using it as the
    base guarantees all architecture/training keys are present and correct.

    We then apply a small set of overrides and additions:
      Category A — keys added to diffusion.py AFTER the checkpoint was trained
                   (T, subs_masking, sampling.sampler, sampling.nucleus_p)
      Category B — inference settings that must be disabled/overridden
                   (eval flags, sampling.steps, checkpointing.save_dir)
    """
    import os
    import omegaconf
    import diffusion as remdm_diffusion

    # --- 1. Read the full training config from the checkpoint ---
    print(f"  [_load_diffusion_model] Reading checkpoint config from: {checkpoint_path}")
    raw_ckpt = torch.load(checkpoint_path, map_location="cpu")
    saved_cfg = raw_ckpt["hyper_parameters"]["config"]

    # Convert to a plain Python dict so we can freely add/update keys without
    # worrying about OmegaConf struct-mode restrictions.  resolve=False keeps
    # interpolation strings (${resolver:...}) intact for re-wrapping.
    cfg_dict = omegaconf.OmegaConf.to_container(
        saved_cfg, resolve=False, throw_on_missing=False
    )

    # --- 2. Category A: keys absent from training config but required by current diffusion.py ---
    # T=0  → continuous-time SUBS parameterization (not D3PM discrete time).
    # subs_masking=False → required by _validate_configuration() for subs parameterization.
    # These were added to diffusion.py after the checkpoint was trained, so the checkpoint
    # config doesn't have them.
    cfg_dict.setdefault("T", 0)
    cfg_dict.setdefault("subs_masking", False)

    # sampling.sampler  → controls the branching in _ddpm_caching_update.  'mdlm' is the
    #   standard MDLM predictor; other values (remdm-cap, remdm-conf, …) are not needed.
    # sampling.nucleus_p → probability mass for nucleus truncation.  1.0 = disabled.
    # All remaining keys are safe defaults for sampler branches that are never reached in
    # inference mode with sampler='mdlm'.
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

    # --- 3. Category B: override inference settings ---
    # Disable generative-perplexity evaluation: this would trigger a download of the
    # gpt2-large *model* (700 MB) at validation time; we only need the tokenizer (tiny,
    # always cached) which is loaded unconditionally in __init__.
    cfg_dict["eval"]["compute_generative_perplexity"] = False
    cfg_dict["eval"]["generate_samples"] = False        # no val-time generation
    cfg_dict["eval"]["compute_perplexity_on_sanity"] = False
    cfg_dict["eval"]["checkpoint_path"] = checkpoint_path

    # Set the predictor and step count for our experiment.
    cfg_dict["sampling"]["predictor"] = "ddpm_cache"
    cfg_dict["sampling"]["steps"] = steps

    # Use a writable scratch dir (the training path is on OCI/cloud storage).
    cfg_dict["checkpointing"]["save_dir"] = "/tmp/proseco_phase1"

    # --- 4. Reconstruct as a fresh (non-struct) OmegaConf DictConfig ---
    cfg = omegaconf.OmegaConf.create(cfg_dict)

    # Register custom resolvers used by the checkpoint's Hydra-style interpolations
    # (trainer.devices = "${device_count:}", etc.).  We never access those keys in
    # inference mode, but registering ensures no ResolverNotFound if they are touched.
    omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd, replace=True)
    omegaconf.OmegaConf.register_new_resolver(
        "device_count", torch.cuda.device_count, replace=True)
    omegaconf.OmegaConf.register_new_resolver("eval", eval, replace=True)
    omegaconf.OmegaConf.register_new_resolver(
        "div_up", lambda x, y: (x + y - 1) // y, replace=True)

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
# ProSeCoGenerator — Generator protocol with ProSeCo corrector
# ---------------------------------------------------------------------------

class ProSeCoGenerator:
    """ProSeCo-style corrector generator for Protocol A + B experiments.

    Loads MDLM checkpoint and runs an instrumented predictor loop with
    ProSeCo's annealed iterative-refinement corrector.

    Parameters
    ----------
    checkpoint : str
        MDLM Lightning checkpoint path (e.g. ~/mdm/checkpoints/mdlm.ckpt).
    T : int
        Number of predictor steps.
    device : str
        Torch device.
    corrector_steps : int
        Number of inner refinement steps per corrector loop. Default 2.
        Each corrector step costs 1 backbone NFE.
    ref_model_name : str
        HuggingFace model for neg-NLL scoring.
    """

    def __init__(
        self,
        checkpoint: str,
        T: int = 64,
        device: str = "cuda",
        corrector_steps: int = 2,
        ref_model_name: str = "gpt2",
    ):
        self.T = T
        self.device = device
        self.corrector_steps = corrector_steps
        self.eps = 1e-5

        print(f"[ProSeCoGenerator] Loading checkpoint: {checkpoint}")
        self.model, self.tokenizer, self.cfg = _load_diffusion_model(
            checkpoint, steps=T, device=device
        )
        self.mask_id = self.model.mask_index

        print(f"[ProSeCoGenerator] Loading reference scorer: {ref_model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer as AutoTok
        self._ref_tok = AutoTok.from_pretrained(ref_model_name)
        self._ref_lm = AutoModelForCausalLM.from_pretrained(ref_model_name).to(device)
        self._ref_lm.eval()
        print(f"[ProSeCoGenerator] Ready. T={T}, corrector_steps={corrector_steps}")

    # ------------------------------------------------------------------
    # Internal: extract signals from p_x0_cache (no extra forward pass)
    # ------------------------------------------------------------------

    def _extract_signals(
        self,
        x: torch.Tensor,
        p_x0: torch.Tensor,
    ) -> Dict[str, float]:
        """Extract per-step signals from predictor's p_x0 output.

        Signals are computed over UNMASKED (committed) positions — these are
        the positions that the ProSeCo corrector will revise.

        Parameters
        ----------
        x : (B, L) token ids; mask_id for masked positions.
        p_x0 : (B, L, V) probabilities from predictor.
        """
        p = p_x0[0].float()  # (L, V)
        D = x.shape[1]

        # ProSeCo corrector action set = unmasked (committed) positions
        revisable = (x[0] != self.mask_id)  # (L,)
        n_rev = int(revisable.sum())
        n_masked = D - n_rev

        if n_rev == 0:
            return {
                "entropy": 0.0,
                "inverse_margin": 0.0,
                "quality_mass_proxy": 0.0,
                "unmasked_fraction": 0.0,
                "n_revisable": 0,
                "n_masked": D,
            }

        p_rev = p[revisable].clamp(min=1e-12)  # (n_rev, V)

        # Entropy (nats)
        H = -(p_rev * p_rev.log()).sum(-1).mean().item()

        # Inverse confidence margin
        top2 = p_rev.topk(2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).mean().item()

        # Quality mass proxy
        p_argmax = p_rev.max(-1).values.mean().item()

        return {
            "entropy": float(H),
            "inverse_margin": float(1.0 - margin),
            "quality_mass_proxy": float(1.0 - p_argmax),
            "unmasked_fraction": float(n_rev / D),
            "n_revisable": n_rev,
            "n_masked": n_masked,
        }

    # ------------------------------------------------------------------
    # Internal: ProSeCo corrector loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _apply_corrector(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        p_x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One ProSeCo corrector loop: annealed iterative refinement.

        1. x̂_0 = argmax p_x0 (full predicted clean sequence)
        2. For corrector_steps iterations at decreasing noise levels τ:
             corrector_x = argmax model(corrector_x, σ_τ)
        3. Apply corrector_x to UNMASKED positions in x only.

        This is the direct translation of ProSeCo's _diffusion_sample corrector
        block, using the existing MDLM backbone.

        p_x0 may be None when _ddpm_caching_update invalidated the cache (which
        happens whenever x changed — i.e., on most steps).  In that case one fresh
        forward pass is made at sigma(t) to initialise x̂_0.
        """
        # Step 1: predicted clean x0 from current predictor output.
        # If p_x0 is not cached, recompute it.  This is one extra NFE but
        # unavoidable: _ddpm_caching_update does not expose p_x0 when x changes.
        if p_x0 is None:
            sigma_t, _ = self.model.noise(t)
            p_x0 = self.model.forward(x, sigma_t).exp()
        corrector_x = p_x0.argmax(-1)  # (B, L)

        # Step 2: annealed refinement (τ: 1 → eps)
        corrector_timesteps = torch.linspace(
            1.0, self.eps, self.corrector_steps + 1, device=self.device
        )

        for j in range(self.corrector_steps):
            tau = corrector_timesteps[j] * torch.ones(
                x.shape[0], 1, device=self.device
            )
            sigma_tau, _ = self.model.noise(tau)
            log_p_corr = self.model.forward(corrector_x, sigma_tau)  # (B, L, V)
            corrector_x = log_p_corr.exp().argmax(-1)  # (B, L)

        # Step 3: apply to UNMASKED positions only (ProSeCo rule)
        unmasked = (x != self.mask_id)  # (B, L)
        x_new = x.clone()
        x_new[unmasked] = corrector_x[unmasked]
        return x_new

    # ------------------------------------------------------------------
    # Internal: quality functional
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_neg_nll(self, tokens: torch.Tensor) -> float:
        """Negative NLL under GPT-2 reference (higher = better)."""
        token_ids = tokens.cpu().tolist()
        token_ids = [t for t in token_ids if t != self.mask_id]
        if not token_ids:
            return 0.0
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        if not text.strip():
            return 0.0

        enc = self._ref_tok(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        if enc["input_ids"].shape[1] < 2:
            return 0.0
        with torch.no_grad():
            outputs = self._ref_lm(enc["input_ids"], labels=enc["input_ids"])
        return float(-outputs.loss.item())

    # ------------------------------------------------------------------
    # Internal: main sampling loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_loop(
        self,
        seed: int,
        corrector_at_t: Optional[int] = None,
        record_signals: bool = True,
    ) -> Dict[str, Any]:
        """Run the ProSeCo predictor+corrector loop.

        Parameters
        ----------
        seed : int
            Random seed (CRN across branches).
        corrector_at_t : int or None
            If set, apply one ProSeCo corrector loop after predictor step
            `corrector_at_t` (0-indexed). Subsequent steps use corrected state.
        record_signals : bool
            If True, extract per-step signals (requires no extra forward pass —
            signals reuse p_x0 already computed by _ddpm_caching_update).
            Set False for branch runs to save time.
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

        per_step_signals: List[Dict] = []

        for step_i in range(self.T):
            t = timesteps[step_i] * torch.ones(B, 1, device=self.device)

            # Predictor step.
            # _ddpm_caching_update returns p_x0_cache = None whenever x changes
            # (i.e., on most steps: every time a token gets unmasked).  It only
            # preserves the cache when x is identical to the input — rare in practice.
            p_x0_cache, x, confident_score = self.model._ddpm_caching_update(
                x, t, dt, p_x0=p_x0_cache, conf=confident_score
            )

            # Signal extraction over the post-predictor state.
            # Use cached p_x0 if available; otherwise do one fresh forward pass.
            # The extra NFE is necessary: _ddpm_caching_update doesn't expose p_x0
            # when it invalidates the cache (i.e., when x changed).
            if record_signals:
                if p_x0_cache is not None:
                    p_x0_for_signals = p_x0_cache
                else:
                    sigma_t, _ = self.model.noise(t)
                    p_x0_for_signals = self.model.forward(x, sigma_t).exp()
                signals = self._extract_signals(x, p_x0_for_signals)
                per_step_signals.append({"t": step_i, **signals})

            # ProSeCo corrector at requested step
            if corrector_at_t is not None and step_i == corrector_at_t:
                x = self._apply_corrector(x, t, p_x0=p_x0_cache)
                p_x0_cache = None  # force recomputation from corrected state

        return {
            "tokens": x[0].cpu().numpy().astype(np.int32),
            "neg_nll": float(self._compute_neg_nll(x[0])),
            "per_step_signals": per_step_signals,
            "seed": seed,
        }

    # ------------------------------------------------------------------
    # Multi-step schedule loop (Protocol B)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_schedule_loop(
        self,
        seed: int,
        allocation: Dict[int, int],
    ) -> Dict[str, Any]:
        """Run predictor loop with corrector at each step in allocation.

        Does NOT record per-step signals (Protocol B only needs final quality).
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
                x = self._apply_corrector(x, t, p_x0=p_x0_cache)
                p_x0_cache = None

        return {
            "tokens": x[0].cpu().numpy().astype(np.int32),
            "neg_nll": float(self._compute_neg_nll(x[0])),
            "schedule_steps": sorted(allocation.keys()),
            "seed": seed,
        }

    # ------------------------------------------------------------------
    # Generator protocol (backend-agnostic interface)
    # ------------------------------------------------------------------

    def run_base(self, seed: int = 0) -> Dict[str, Any]:
        """Run base trajectory (no correction). Records per-step signals."""
        return self._run_loop(seed=seed, corrector_at_t=None, record_signals=True)

    def run_branch(self, t_corrected: int, seed: int = 0) -> Dict[str, Any]:
        """Run branch: one ProSeCo corrector loop at step t_corrected.

        Does NOT record signals (only needs final neg_nll for Δ_t).
        Uses same seed as run_base for common-random-number variance reduction.
        """
        result = self._run_loop(
            seed=seed, corrector_at_t=t_corrected, record_signals=False
        )
        result["t_corrected"] = t_corrected
        return result

    def run_with_schedule(
        self, allocation: Dict[int, int], seed: int = 0
    ) -> Dict[str, Any]:
        """Run with corrector at each step in allocation (Protocol B).

        G(S) = F(y^S) - F(y_base) is computed externally by evaluate_schedule.
        """
        return self._run_schedule_loop(seed=seed, allocation=allocation)

    def signal_trace(self, seed: int = 0) -> List[Dict]:
        """Per-step signals from a base run."""
        return self.run_base(seed=seed)["per_step_signals"]

    def corrector_description(self) -> str:
        """Human-readable description of the corrector."""
        return (
            f"ProSeCo annealed-refinement corrector: "
            f"corrector_steps={self.corrector_steps}, "
            f"action_set=unmasked_positions, "
            f"backbone=MDLM(checkpoint)"
        )

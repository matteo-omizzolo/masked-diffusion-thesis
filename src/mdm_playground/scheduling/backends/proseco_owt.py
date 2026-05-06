"""ProSeCo-OWT backend for the thesis mainline.

This backend expects a locally staged HuggingFace snapshot of
``kuleshov-group/proseco-owt`` and runs the thesis Protocol A/B workflow on
that snapshot. The checkpoint directory is the only external dependency for
the active Phase 2b / Phase 3a path.

Corrector action set: UNMASKED (committed) positions.
Signal extraction: entropy / inverse_margin / quality_mass_proxy over the
committed positions — the corrector's action set.

Use ``scripts/stage_proseco_owt.py`` to download and patch the snapshot before
running the thesis scripts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from mdm_playground.scheduling.trace import GenerationTrace

# PyTorch >= 2.6: force weights_only=False for trusted local checkpoints
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


def _validate_snapshot_dir(snapshot_path: Path) -> Path:
    """Ensure the staged ProSeCo-OWT snapshot is present and complete."""
    required = ("config.json", "configuration_proseco.py", "modeling_proseco.py")
    missing = [name for name in required if not (snapshot_path / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing ProSeCo-OWT snapshot files in "
            f"{snapshot_path}: {', '.join(missing)}. "
            "Stage the snapshot with "
            "`python scripts/stage_proseco_owt.py --dest <proseco_owt_snapshot_dir>` "
            "or point --checkpoint to an existing staged directory."
        )
    return snapshot_path


def _load_proseco_owt(snapshot_path: str, device: str = "cuda"):
    """Load proseco-owt from a local HuggingFace snapshot directory.

    Loads modeling_proseco.py directly via importlib to avoid HuggingFace
    trust_remote_code caching (which writes to ~/.cache and can hit disk quota).

    snapshot_path: local directory containing config.json, pytorch_model.bin
    (or model.safetensors), and modeling_proseco.py (patched flash_attn).
    """
    import importlib.util
    import json

    snap = _validate_snapshot_dir(Path(snapshot_path))
    print(f"  [_load_proseco_owt] Loading from: {snap}")

    # Load config
    with open(snap / "config.json") as f:
        cfg_dict = json.load(f)

    # Import modeling_proseco directly from snapshot (no HF caching).
    # modeling_proseco.py uses "from .configuration_proseco import ProsecoConfig"
    # so we must load both files under a shared fake package to satisfy the
    # relative import.
    import sys
    import types

    pkg_name = "_proseco_snapshot_pkg"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(snap)]          # type: ignore[assignment]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg

    def _load_submod(name: str) -> types.ModuleType:
        full = f"{pkg_name}.{name}"
        if full in sys.modules:
            return sys.modules[full]
        s = importlib.util.spec_from_file_location(full, snap / f"{name}.py")
        m = importlib.util.module_from_spec(s)  # type: ignore[arg-type]
        m.__package__ = pkg_name
        sys.modules[full] = m
        s.loader.exec_module(m)             # type: ignore[union-attr]
        return m

    _load_submod("configuration_proseco")   # must come first
    mod = _load_submod("modeling_proseco")

    # Build config object + model
    from transformers import PretrainedConfig
    cfg = PretrainedConfig(**cfg_dict)
    cfg.architectures = ["Proseco"]

    model = mod.Proseco(cfg)

    # Load weights — prefer safetensors, fall back to pytorch_model.bin
    weights_safe = snap / "model.safetensors"
    weights_bin = snap / "pytorch_model.bin"
    if weights_safe.exists():
        from safetensors.torch import load_file as safetensors_load
        state = safetensors_load(str(weights_safe), device=device)
        model.load_state_dict(state, strict=True)
    elif weights_bin.exists():
        state = torch.load(str(weights_bin), map_location=device)
        model.load_state_dict(state, strict=True)
    else:
        raise FileNotFoundError(f"No weight file found in {snap}")

    model = model.to(device)
    model.eval()

    vocab_size = cfg.vocab_size
    mask_index = getattr(cfg, "mask_index", vocab_size - 1)
    print(f"  [_load_proseco_owt] Ready. vocab={vocab_size}, mask_index={mask_index}")
    return model, mask_index, cfg


def _loglinear_sigma(t: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """LogLinear noise schedule: sigma(t) = -log(1 - (1-eps)*t)."""
    return -torch.log1p(-(1.0 - eps) * t.clamp(max=1.0 - eps))


class ProSeCoOWTGenerator:
    """ProSeCo-OWT corrector generator for Protocol A + B.

    Parameters
    ----------
    checkpoint : str
        Path to local proseco-owt HuggingFace snapshot directory.
    T : int
        Number of predictor steps.
    corrector_steps : int
        Inner refinement iterations per corrector call. Default 1.
    device : str
    ref_model_name : str
        HuggingFace model name for neg-NLL scoring.
    """

    def __init__(
        self,
        checkpoint: str,
        T: int = 64,
        corrector_steps: int = 1,
        device: str = "cuda",
        ref_model_name: str = "gpt2",
    ):
        self.T = T
        self.corrector_steps = corrector_steps
        self.device = device
        self.eps = 1e-3  # ProSeCo's sampling_eps
        self.seq_len = 1024  # ProSeCo-OWT default sequence length

        print(f"[ProSeCoOWTGenerator] Loading checkpoint: {checkpoint}")
        self.model, self.mask_id, self.cfg = _load_proseco_owt(checkpoint, device)
        if hasattr(self.cfg, "max_position_embeddings"):
            self.seq_len = self.cfg.max_position_embeddings
        elif hasattr(self.cfg, "seq_len"):
            self.seq_len = self.cfg.seq_len

        print(f"[ProSeCoOWTGenerator] Loading reference scorer: {ref_model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._ref_tok = AutoTokenizer.from_pretrained(ref_model_name)
        self._ref_lm = AutoModelForCausalLM.from_pretrained(ref_model_name).to(device)
        self._ref_lm.eval()
        print(f"[ProSeCoOWTGenerator] Ready. T={T}, corrector_steps={corrector_steps}, "
              f"seq_len={self.seq_len}")

    def corrector_description(self) -> str:
        return (
            f"ProSeCo-OWT annealed-refinement corrector: "
            f"corrector_steps={self.corrector_steps}, "
            f"action_set=unmasked_positions, "
            f"backbone=proseco-owt (co-trained)"
        )

    @property
    def corrector_nfe_per_placement(self) -> int:
        """Extra backbone forward calls per scheduled corrector placement."""
        return self.corrector_steps + 1

    def _rng_fingerprint(self) -> str:
        """Short fingerprint of the active torch RNG state for CRN audits."""
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            state = torch.cuda.get_rng_state(self.device)
        else:
            state = torch.get_rng_state()
        return hashlib.sha1(state.cpu().numpy().tobytes()).hexdigest()[:16]

    @staticmethod
    def _np_tokens(x: torch.Tensor) -> np.ndarray:
        return x[0].detach().cpu().numpy().astype(np.int32).copy()

    def _np_mask(self, x: torch.Tensor) -> np.ndarray:
        return (x[0] == self.mask_id).detach().cpu().numpy().astype(bool).copy()

    # ------------------------------------------------------------------
    # Forward pass wrapper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Run proseco-owt backbone forward pass.

        Parameters
        ----------
        x : (B, L) token ids
        t : (B, 1) normalized time ∈ [0, 1]

        Returns
        -------
        p_x0_probs : (B, L, V) probability distribution over clean tokens
        """
        sigma = _loglinear_sigma(t, self.eps).squeeze(-1)  # (B,) — TimestepEmbedder expects 1-D
        out = self.model(input_ids=x, timesteps=sigma)
        # proseco-owt config has return_dict=false → returns raw tensor; handle both
        logits = out.logits if hasattr(out, "logits") else out  # (B, L, V)
        # Zero out mask token probability (subs parameterization)
        logits[:, :, self.mask_id] = -1e9
        return logits.softmax(dim=-1)

    # ------------------------------------------------------------------
    # MDLM predictor step (same schedule as ProSeCo)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predictor_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """One MDLM predictor step: unmask from t→s using p_θ(x_0 | x_t).

        Masked positions at time t are either revealed (unmasked) or kept
        masked, according to the MDLM posterior.
        """
        p_x0 = self._forward(x, t)  # (B, L, V)

        # Move chance: prob of transitioning from masked→revealed
        # Using ProSeCo's absorbing-state posterior: sample from p(x_s | x_t, x_0)
        # p(masked at s | masked at t) = (1 - move_chance_s) / (1 - move_chance_t)
        # We approximate: reveal each masked token w.p. 1 - exp(-sigma_s) / (1 - exp(-sigma_t))
        # For simplicity, use MDLM's standard predictor: sample from p_θ(x_0 | x_t) at masked
        mask = (x == self.mask_id)  # (B, L)
        if not mask.any():
            return x

        # Sample from p_x0 at masked positions
        probs_masked = p_x0[mask]  # (n_masked, V)
        sampled = torch.multinomial(probs_masked.clamp(min=1e-9), num_samples=1).squeeze(-1)

        # Decide which masked positions to reveal (MDLM absorbing-state schedule)
        sigma_t = _loglinear_sigma(t, self.eps)  # (B, 1)
        sigma_s = _loglinear_sigma(s, self.eps)  # (B, 1)
        # move_chance = 1 - exp(sigma_s - sigma_t) for each masked position
        # Broadcast to (n_masked,) using the batch dimension of the masked positions
        # Simplified: compute per-sequence move chance
        b_idx = mask.nonzero(as_tuple=True)[0]  # batch indices of masked positions
        move_chance_t = 1.0 - (-sigma_t).exp()  # (B, 1)
        move_chance_s = 1.0 - (-sigma_s).exp()  # (B, 1)
        # p(reveal | masked at t, going to s)
        p_reveal = ((move_chance_t[b_idx, 0] - move_chance_s[b_idx, 0])
                    / move_chance_t[b_idx, 0].clamp(min=1e-9))
        reveal = torch.bernoulli(p_reveal.clamp(0.0, 1.0)).bool()

        x_new = x.clone()
        mask_positions = mask.nonzero(as_tuple=True)
        pos_rows = mask_positions[0][reveal]
        pos_cols = mask_positions[1][reveal]
        x_new[pos_rows, pos_cols] = sampled[reveal]
        return x_new

    # ------------------------------------------------------------------
    # ProSeCo corrector
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _apply_corrector(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        p_x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One ProSeCo corrector loop: annealed iterative refinement on UNMASKED positions.

        1. x̂_0 = argmax p_x0 (full predicted clean sequence).
        2. For corrector_steps iterations at decreasing σ: refine x̂_0 via backbone.
        3. Apply refined tokens to UNMASKED positions in x only.
        """
        if p_x0 is None:
            p_x0 = self._forward(x, t)

        corrector_x = p_x0.argmax(-1)  # (B, L)

        # Annealed refinement: τ from 1 → eps across corrector_steps steps
        corrector_times = torch.linspace(1.0, self.eps, self.corrector_steps + 1,
                                         device=self.device)
        for j in range(self.corrector_steps):
            tau = corrector_times[j] * torch.ones(x.shape[0], 1, device=self.device)
            p_corr = self._forward(corrector_x, tau)
            corrector_x = p_corr.argmax(-1)

        # Apply to UNMASKED positions only
        unmasked = (x != self.mask_id)  # (B, L)
        x_new = x.clone()
        x_new[unmasked] = corrector_x[unmasked]
        return x_new

    # ------------------------------------------------------------------
    # Signal extraction (over UNMASKED positions — corrector action set)
    # ------------------------------------------------------------------

    def _extract_signals(
        self,
        x: torch.Tensor,
        p_x0: torch.Tensor,
    ) -> Dict[str, float]:
        """Signals over the corrector's action set (unmasked/committed positions)."""
        p = p_x0[0].float()  # (L, V)
        D = x.shape[1]

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

        H = -(p_rev * p_rev.log()).sum(-1).mean().item()
        top2 = p_rev.topk(2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).mean().item()
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
    # Quality functional
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_neg_nll(self, tokens: torch.Tensor) -> float:
        token_ids = [t for t in tokens.cpu().tolist() if t != self.mask_id]
        if not token_ids:
            return 0.0
        text = self._ref_tok.decode(token_ids, skip_special_tokens=True)
        if not text.strip():
            return 0.0
        enc = self._ref_tok(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        if enc["input_ids"].shape[1] < 2:
            return 0.0
        outputs = self._ref_lm(enc["input_ids"], labels=enc["input_ids"])
        return float(-outputs.loss.item())

    # ------------------------------------------------------------------
    # Main sampling loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_loop(
        self,
        seed: int,
        corrector_at_t: Optional[int] = None,
        record_signals: bool = True,
        allocation: Optional[Dict[int, int]] = None,
        return_trace: bool = False,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        B = 1
        L = self.seq_len

        x = self.mask_id * torch.ones(B, L, dtype=torch.long, device=self.device)
        timesteps = torch.linspace(1.0, self.eps, self.T + 1, device=self.device)

        per_step_signals: List[Dict] = []
        schedule = set(allocation or {})
        if corrector_at_t is not None:
            schedule.add(corrector_at_t)

        tokens_by_step: List[np.ndarray] = []
        masks_by_step: List[np.ndarray] = []
        revisable_sets_by_step: List[np.ndarray] = []
        corrected_positions_by_step: List[np.ndarray] = []
        signal_masks_by_step: List[np.ndarray] = []
        rng_fingerprints: List[str] = []
        signals_by_step: Dict[str, List[float]] = {
            "entropy": [],
            "inverse_margin": [],
            "quality_mass_proxy": [],
            "unmasked_fraction": [],
            "n_revisable": [],
        }

        if return_trace:
            tokens_by_step.append(self._np_tokens(x))
            masks_by_step.append(self._np_mask(x))
            rng_fingerprints.append(f"init:{self._rng_fingerprint()}")

        for step_i in range(self.T):
            t = timesteps[step_i] * torch.ones(B, 1, device=self.device)
            s = timesteps[step_i + 1] * torch.ones(B, 1, device=self.device)

            if return_trace:
                rng_fingerprints.append(f"pre_predictor_{step_i}:{self._rng_fingerprint()}")
            p_x0 = self._forward(x, t)
            x = self._predictor_step(x, t, s)

            revisable_mask = (x[0] != self.mask_id).detach().cpu().numpy().astype(bool)
            revisable_idx = np.flatnonzero(revisable_mask).astype(np.int32)

            if record_signals:
                signals = self._extract_signals(x, p_x0)
                per_step_signals.append({"t": step_i, **signals})
                if return_trace:
                    for key in signals_by_step:
                        signals_by_step[key].append(float(signals.get(key, 0.0)))

            if return_trace:
                revisable_sets_by_step.append(revisable_idx)
                signal_masks_by_step.append(revisable_mask.copy())

            if step_i in schedule:
                if return_trace:
                    corrected_positions_by_step.append(revisable_idx.copy())
                x = self._apply_corrector(x, s, p_x0=None)
                if return_trace:
                    rng_fingerprints.append(
                        f"post_corrector_{step_i}:{self._rng_fingerprint()}"
                    )
            elif return_trace:
                corrected_positions_by_step.append(np.array([], dtype=np.int32))

            if return_trace:
                tokens_by_step.append(self._np_tokens(x))
                masks_by_step.append(self._np_mask(x))
                rng_fingerprints.append(f"post_step_{step_i}:{self._rng_fingerprint()}")

        score = float(self._compute_neg_nll(x[0]))
        final_tokens = self._np_tokens(x)

        result = {
            "tokens": final_tokens,
            "neg_nll": score,
            "per_step_signals": per_step_signals,
            "seed": seed,
        }
        if return_trace:
            result["trace"] = GenerationTrace(
                seed=seed,
                schedule=tuple(sorted(schedule)),
                tokens_by_step=tokens_by_step,
                masks_by_step=masks_by_step,
                revisable_sets_by_step=revisable_sets_by_step,
                corrected_positions_by_step=corrected_positions_by_step,
                signal_masks_by_step=signal_masks_by_step,
                signals_by_step=signals_by_step,
                forward_pass_count=self.T
                + len(schedule) * self.corrector_nfe_per_placement,
                rng_fingerprint_by_step=rng_fingerprints,
                final_tokens=final_tokens,
                score=score,
            )
        return result

    # ------------------------------------------------------------------
    # Generator protocol (backend-agnostic interface)
    # ------------------------------------------------------------------

    def run_base(self, seed: int = 0, return_trace: bool = False) -> Dict[str, Any]:
        return self._run_loop(
            seed=seed,
            corrector_at_t=None,
            record_signals=True,
            allocation={},
            return_trace=return_trace,
        )

    def run_branch(
        self, t_corrected: int, seed: int = 0, return_trace: bool = False
    ) -> Dict[str, Any]:
        result = self._run_loop(
            seed=seed,
            corrector_at_t=t_corrected,
            record_signals=False,
            allocation={t_corrected: 1},
            return_trace=return_trace,
        )
        result["t_corrected"] = t_corrected
        return result

    @torch.no_grad()
    def run_with_schedule(
        self, allocation: Dict[int, int], seed: int = 0, return_trace: bool = False
    ) -> Dict[str, Any]:
        result = self._run_loop(
            seed=seed,
            allocation=allocation,
            record_signals=True,
            return_trace=return_trace,
        )
        result["schedule_steps"] = sorted(allocation.keys())
        return result

    def signal_trace(self, seed: int = 0) -> List[Dict]:
        return self.run_base(seed=seed)["per_step_signals"]

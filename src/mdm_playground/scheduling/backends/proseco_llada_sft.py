"""ProSeCo-LLaDA-SFT backend for bounded cross-backbone replication."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


def _validate_snapshot_dir(snapshot_path: Path) -> Path:
    required = ("config.json", "configuration_llada.py", "modeling_llada.py")
    missing = [name for name in required if not (snapshot_path / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing ProSeCo-LLaDA-SFT snapshot files in "
            f"{snapshot_path}: {', '.join(missing)}. "
            "Stage the snapshot with "
            "`python scripts/legacy/stage_proseco_llada_sft.py --dest <snapshot_dir>` "
            "or point --checkpoint to an existing staged directory."
        )
    has_weights = (snapshot_path / "model.safetensors.index.json").exists() or any(
        snapshot_path.glob("model-*.safetensors")
    )
    if not has_weights:
        raise FileNotFoundError(
            f"No sharded safetensors weights found in {snapshot_path}. "
            "Expected model.safetensors.index.json plus model-*.safetensors files."
        )
    return snapshot_path


def _load_proseco_llada_sft(snapshot_path: str, device: str = "cuda"):
    snap = _validate_snapshot_dir(Path(snapshot_path))
    print(f"  [_load_proseco_llada_sft] Loading from: {snap}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        str(snap),
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(snap),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
    )
    model = model.to(device)
    model.eval()

    cfg = model.config
    mask_id = getattr(cfg, "mask_token_id", None)
    if mask_id is None:
        raise RuntimeError(
            f"Loaded LLADA config from {snap} has no mask_token_id; cannot run "
            "masked trajectory scheduling protocol."
        )
    print(
        "  [_load_proseco_llada_sft] Ready. "
        f"vocab={getattr(cfg, 'vocab_size', 'unknown')}, mask_id={mask_id}"
    )
    return model, tokenizer, int(mask_id), cfg


def _loglinear_sigma(t: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return -torch.log1p(-(1.0 - eps) * t.clamp(max=1.0 - eps))


class ProSeCoLLaDASFTGenerator:
    """ProSeCo-LLaDA-SFT generator with schedule-controlled corrective refinement."""

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
        self.eps = 1e-3
        self.seq_len = 1024

        print(f"[ProSeCoLLaDASFTGenerator] Loading checkpoint: {checkpoint}")
        self.model, self._gen_tok, self.mask_id, self.cfg = _load_proseco_llada_sft(
            checkpoint, device
        )
        self.seq_len = int(
            getattr(
                self.cfg,
                "max_sequence_length",
                getattr(self.cfg, "max_position_embeddings", self.seq_len),
            )
        )

        print(f"[ProSeCoLLaDASFTGenerator] Loading reference scorer: {ref_model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._ref_tok = AutoTokenizer.from_pretrained(ref_model_name)
        self._ref_lm = AutoModelForCausalLM.from_pretrained(ref_model_name).to(device)
        self._ref_lm.eval()
        print(
            "[ProSeCoLLaDASFTGenerator] Ready. "
            f"T={T}, corrector_steps={corrector_steps}, seq_len={self.seq_len}"
        )

    def corrector_description(self) -> str:
        return (
            "ProSeCo-LLaDA-SFT iterative-refinement corrector: "
            f"corrector_steps={self.corrector_steps}, "
            "action_set=unmasked_positions, "
            "backbone=proseco-llada-sft"
        )

    @torch.no_grad()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=x)
        logits = out.logits if hasattr(out, "logits") else out
        if 0 <= self.mask_id < logits.shape[-1]:
            logits[:, :, self.mask_id] = -1e9
        return logits.softmax(dim=-1)

    @torch.no_grad()
    def _predictor_step(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        p_x0 = self._forward(x)
        mask = x == self.mask_id
        if not mask.any():
            return x

        probs_masked = p_x0[mask]
        sampled = torch.multinomial(probs_masked.clamp(min=1e-9), num_samples=1).squeeze(-1)

        sigma_t = _loglinear_sigma(t, self.eps)
        sigma_s = _loglinear_sigma(s, self.eps)
        b_idx = mask.nonzero(as_tuple=True)[0]
        move_chance_t = 1.0 - (-sigma_t).exp()
        move_chance_s = 1.0 - (-sigma_s).exp()
        p_reveal = (
            (move_chance_t[b_idx, 0] - move_chance_s[b_idx, 0])
            / move_chance_t[b_idx, 0].clamp(min=1e-9)
        )
        reveal = torch.bernoulli(p_reveal.clamp(0.0, 1.0)).bool()

        x_new = x.clone()
        mask_pos = mask.nonzero(as_tuple=True)
        pos_rows = mask_pos[0][reveal]
        pos_cols = mask_pos[1][reveal]
        x_new[pos_rows, pos_cols] = sampled[reveal]
        return x_new

    @torch.no_grad()
    def _apply_corrector(self, x: torch.Tensor, p_x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        if p_x0 is None:
            p_x0 = self._forward(x)
        corrector_x = p_x0.argmax(-1)

        for _ in range(self.corrector_steps):
            p_corr = self._forward(corrector_x)
            corrector_x = p_corr.argmax(-1)

        unmasked = x != self.mask_id
        x_new = x.clone()
        x_new[unmasked] = corrector_x[unmasked]
        return x_new

    def _extract_signals(self, x: torch.Tensor, p_x0: torch.Tensor) -> Dict[str, float]:
        p = p_x0[0].float()
        D = x.shape[1]
        revisable = x[0] != self.mask_id
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

        p_rev = p[revisable].clamp(min=1e-12)
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

    @torch.no_grad()
    def _compute_neg_nll(self, tokens: torch.Tensor) -> float:
        token_ids = [int(t) for t in tokens.cpu().tolist() if int(t) != self.mask_id]
        if not token_ids:
            return 0.0

        text = self._gen_tok.decode(token_ids, skip_special_tokens=True)
        if not text.strip():
            return 0.0

        enc = self._ref_tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        if enc["input_ids"].shape[1] < 2:
            return 0.0
        outputs = self._ref_lm(enc["input_ids"], labels=enc["input_ids"])
        return float(-outputs.loss.item())

    @torch.no_grad()
    def _run_loop(
        self,
        seed: int,
        corrector_at_t: Optional[int] = None,
        record_signals: bool = True,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        B = 1
        L = self.seq_len
        x = self.mask_id * torch.ones(B, L, dtype=torch.long, device=self.device)
        timesteps = torch.linspace(1.0, self.eps, self.T + 1, device=self.device)
        per_step_signals: List[Dict[str, Any]] = []

        for step_i in range(self.T):
            t = timesteps[step_i] * torch.ones(B, 1, device=self.device)
            s = timesteps[step_i + 1] * torch.ones(B, 1, device=self.device)

            p_x0 = self._forward(x)
            x = self._predictor_step(x, t, s)

            if record_signals:
                sig = self._extract_signals(x, p_x0)
                per_step_signals.append({"t": step_i, **sig})

            if corrector_at_t is not None and step_i == corrector_at_t:
                x = self._apply_corrector(x, p_x0=None)

        return {
            "tokens": x[0].cpu().numpy().astype(np.int32),
            "neg_nll": float(self._compute_neg_nll(x[0])),
            "per_step_signals": per_step_signals,
            "seed": seed,
        }

    def run_base(self, seed: int = 0) -> Dict[str, Any]:
        return self._run_loop(seed=seed, corrector_at_t=None, record_signals=True)

    def run_branch(self, t_corrected: int, seed: int = 0) -> Dict[str, Any]:
        result = self._run_loop(seed=seed, corrector_at_t=t_corrected, record_signals=False)
        result["t_corrected"] = t_corrected
        return result

    @torch.no_grad()
    def run_with_schedule(self, allocation: Dict[int, int], seed: int = 0) -> Dict[str, Any]:
        torch.manual_seed(seed)
        B = 1
        L = self.seq_len
        x = self.mask_id * torch.ones(B, L, dtype=torch.long, device=self.device)
        timesteps = torch.linspace(1.0, self.eps, self.T + 1, device=self.device)

        for step_i in range(self.T):
            t = timesteps[step_i] * torch.ones(B, 1, device=self.device)
            s = timesteps[step_i + 1] * torch.ones(B, 1, device=self.device)
            p_x0 = self._forward(x)
            x = self._predictor_step(x, t, s)
            if step_i in allocation:
                x = self._apply_corrector(x, p_x0=p_x0)

        return {
            "tokens": x[0].cpu().numpy().astype(np.int32),
            "neg_nll": float(self._compute_neg_nll(x[0])),
            "schedule_steps": sorted(allocation.keys()),
            "seed": seed,
        }

    def signal_trace(self, seed: int = 0) -> List[Dict[str, Any]]:
        return self.run_base(seed=seed)["per_step_signals"]


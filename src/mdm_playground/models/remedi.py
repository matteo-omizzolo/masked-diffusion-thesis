"""RemeDi model adapter (direct HF load + forward pass).

Replaces and extends ``remedi_infer/load_model.py``.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F
import transformers

from .base import ForwardOutput, ModelAdapter, ModelMeta

# Ensure external/remedi/remedi/ is importable.
_REMEDI_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "external", "remedi")
)
if _REMEDI_ROOT not in sys.path:
    sys.path.insert(0, _REMEDI_ROOT)


def _import_remedi():
    from remedi import RemeDiUPMModelLM  # noqa: PLC0415
    return RemeDiUPMModelLM


def _compute_confidence(
    token_logits: torch.Tensor,
    x0: torch.Tensor,
    ups_confidences: Optional[torch.Tensor],
    use_ups: bool,
) -> torch.Tensor:
    """Normalised per-token confidence in ``[0, 1]``.

    Extracted as a standalone function so tests can call it directly.

    Args:
        token_logits:    ``[B, L, V]`` raw TPS logits.
        x0:              ``[B, L]`` argmax predicted token ids.
        ups_confidences: ``[B, L, 1]`` or ``[B, L]`` raw UPS scalar (or None).
        use_ups:         Use sigmoid(UPS) when available; else fallback to
                         softmax probability of the predicted token.
    """
    if use_ups and ups_confidences is not None:
        raw = ups_confidences
        if raw.dim() == 3:
            raw = raw.squeeze(-1)
        return torch.sigmoid(raw.to(torch.float32))
    probs = F.softmax(token_logits.to(torch.float32), dim=-1)
    return probs.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)



class RemeDiAdapter(ModelAdapter):
    """Adapter for RemeDi (maple-research-lab/RemeDi-RL or -Instruct).

    Exposes the UPS confidence head when available, falling back to the
    softmax probability of the predicted token.

    Args:
        model:     Loaded ``RemeDiUPMModelLM`` instance.
        tokenizer: Corresponding HF tokenizer.
        _meta:     Pre-computed :class:`ModelMeta`.
        use_ups:   If ``True``, use the UPS head confidence; otherwise use
                   softmax-probability baseline.
    """

    def __init__(
        self,
        model,
        tokenizer: transformers.PreTrainedTokenizerBase,
        _meta: ModelMeta,
        use_ups: bool = True,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._meta = _meta
        self.use_ups = use_ups

        # Import custom DynamicCache from the remedi package.
        try:
            from remedi.modelling_remedi_bitowel import DynamicCache  # noqa: PLC0415
            self._DynamicCache = DynamicCache
        except ImportError:
            from transformers.cache_utils import DynamicCache  # noqa: PLC0415
            self._DynamicCache = DynamicCache

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_id: str = "maple-research-lab/RemeDi-RL",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_ups: bool = True,
        compile: bool = False,
        **kwargs: Any,
    ) -> "RemeDiAdapter":
        RemeDiUPMModelLM = _import_remedi()
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        model = RemeDiUPMModelLM.from_pretrained(model_id, torch_dtype=dtype)
        model.eval().requires_grad_(False).to(device)
        if compile:
            model = torch.compile(model)

        cfg = model.config
        meta = ModelMeta(
            mask_token_id=getattr(cfg, "mask_token_id", 126336),
            eos_token_id=getattr(cfg, "eos_token_id", 126081),
            vocab_size=getattr(cfg, "vocab_size", 152064),
            model_id=model_id,
            device=device,
        )
        return cls(model, tokenizer, meta, use_ups=use_ups)

    @property
    def meta(self) -> ModelMeta:
        return self._meta

    def forward(self, x: torch.Tensor, **kwargs) -> ForwardOutput:
        """Single forward pass for a block of tokens.

        Args:
            x:             ``[B, L]`` token ids.
            position_ids:  Optional ``[B, L]``.
            kv_cache:      Optional DynamicCache.

        Returns:
            :class:`ForwardOutput` with ``token_logits``, ``confidence``, ``x0``.
        """
        dev = self.device
        use_amp = dev.type == "cuda"

        with torch.autocast(device_type=dev.type, enabled=use_amp, dtype=torch.bfloat16):
            out = self._model(x, **kwargs)

        token_logits = out.logits.to(torch.float32)   # [B, L, V]
        x0 = token_logits.argmax(dim=-1)              # [B, L]

        ups_raw: Optional[torch.Tensor] = getattr(out, "confidences", None)
        if self.use_ups and ups_raw is not None:
            raw = ups_raw.squeeze(-1) if ups_raw.dim() == 3 else ups_raw
            confidence = torch.sigmoid(raw.to(torch.float32))
        else:
            probs = F.softmax(token_logits, dim=-1)
            confidence = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

        return ForwardOutput(
            token_logits=token_logits,
            confidence=confidence,
            x0=x0,
            extra={"raw_out": out},
        )

    # ------------------------------------------------------------------
    # Tokenizer wrappers — kept here for the block-diffusion sampler
    # ------------------------------------------------------------------

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def model(self):
        return self._model

    def new_kv_cache(self):
        return self._DynamicCache()

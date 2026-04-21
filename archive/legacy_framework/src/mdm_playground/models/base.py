"""Abstract ModelAdapter interface.

Every backend (RemeDi, ReMDM, PRISM) implements :class:`ModelAdapter`.
The sampler only knows about this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class ModelMeta:
    """Common metadata returned by :meth:`ModelAdapter.load`."""

    mask_token_id: int
    eos_token_id: int
    vocab_size: int
    model_id: str = ""
    device: str = "cpu"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForwardOutput:
    """Standardised output of one forward pass.

    Attributes
    ----------
    token_logits:
        ``[B, L, V]`` raw logits from the Token Prediction Stream (or
        equivalent for subprocess-based models — may be ``None`` for those).
    confidence:
        ``[B, L]`` normalised confidence in ``[0, 1]``, or ``None`` if the
        model does not produce a confidence signal.
    x0:
        ``[B, L]`` argmax predicted tokens.
    extra:
        Any additional outputs the adapter wants to expose.
    """

    token_logits: Optional[torch.Tensor]
    confidence: Optional[torch.Tensor]
    x0: Optional[torch.Tensor]
    extra: Dict[str, Any] = field(default_factory=dict)


class ModelAdapter(ABC):
    """Uniform interface for all three backends.

    Concrete subclasses live in:
    - :mod:`mdm_playground.models.remedi`  (direct HF forward pass)
    - :mod:`mdm_playground.models.remdm`   (subprocess call)
    - :mod:`mdm_playground.models.prism`   (subprocess call)
    """

    @classmethod
    @abstractmethod
    def load(
        cls,
        model_id: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        **kwargs: Any,
    ) -> "ModelAdapter":
        """Load the model from a checkpoint or HF hub.

        Returns an initialised adapter ready to call :meth:`forward`.
        """
        ...

    @property
    @abstractmethod
    def meta(self) -> ModelMeta:
        """Return :class:`ModelMeta` for this adapter."""
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs: Any) -> ForwardOutput:
        """Run one forward pass.

        Args:
            x: ``[B, L]`` token ids; masked positions hold ``meta.mask_token_id``.

        Returns:
            :class:`ForwardOutput` with at least ``x0`` populated.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def mask_token_id(self) -> int:
        return self.meta.mask_token_id

    @property
    def eos_token_id(self) -> int:
        return self.meta.eos_token_id

    @property
    def device(self) -> torch.device:
        return torch.device(self.meta.device)

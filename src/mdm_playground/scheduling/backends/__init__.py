"""Backends for the thesis scheduling mainline.

The active path uses :mod:`proseco_owt`, with :mod:`mdlm_conf` and
:mod:`mdlm` retained for chronology and regression checks.
"""

from .mdlm import MDLMGenerator
from .mdlm_conf import MDLMConfGenerator
from .proseco import ProSeCoGenerator
from .proseco_llada_sft import ProSeCoLLaDASFTGenerator
from .proseco_owt import ProSeCoOWTGenerator
from .checkpoint_utils import detect_proseco_snapshot_backend

__all__ = [
    "MDLMGenerator",
    "MDLMConfGenerator",
    "ProSeCoGenerator",
    "ProSeCoLLaDASFTGenerator",
    "ProSeCoOWTGenerator",
    "detect_proseco_snapshot_backend",
]

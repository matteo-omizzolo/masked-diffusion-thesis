"""Checkpoint snapshot helpers for ProSeCo-family backends."""

from __future__ import annotations

from pathlib import Path


def detect_proseco_snapshot_backend(checkpoint: str) -> str:
    """Detect backend type from a local HuggingFace snapshot directory.

    Returns one of:
      - ``proseco_owt``
      - ``proseco_llada_sft``
    """
    snap = Path(checkpoint)
    if not snap.exists():
        raise FileNotFoundError(
            f"Checkpoint path does not exist: {snap}. "
            "Provide a staged local snapshot directory."
        )
    if not snap.is_dir():
        raise FileNotFoundError(
            f"Checkpoint path is not a directory: {snap}. "
            "Expected a local HuggingFace snapshot directory."
        )
    if not (snap / "config.json").exists():
        raise FileNotFoundError(
            f"Missing config.json in checkpoint directory: {snap}."
        )

    has_proseco = (snap / "configuration_proseco.py").exists() and (
        snap / "modeling_proseco.py"
    ).exists()
    has_llada = (snap / "configuration_llada.py").exists() and (
        snap / "modeling_llada.py"
    ).exists()

    if has_proseco and not has_llada:
        return "proseco_owt"
    if has_llada and not has_proseco:
        return "proseco_llada_sft"
    if has_proseco and has_llada:
        raise RuntimeError(
            f"Ambiguous checkpoint snapshot at {snap}: both ProSeCo and LLaDA "
            "modeling files are present."
        )
    raise RuntimeError(
        f"Unsupported checkpoint snapshot layout at {snap}. "
        "Expected either {configuration_proseco.py, modeling_proseco.py} or "
        "{configuration_llada.py, modeling_llada.py}."
    )


"""Diffusion step schedules shared across methods."""
from __future__ import annotations

import math

import torch
from torch import Tensor


def transfer_schedule(block_size: int, steps: int, device: torch.device) -> Tensor:
    """Cumulative token-transfer counts per diffusion step.

    Returns a ``[steps]`` int64 tensor where ``schedule[i]`` is the total
    number of tokens that should be committed (unmasked) by the end of step
    ``i``.  The schedule distributes ``block_size`` tokens as evenly as
    possible across ``steps`` steps.

    Example::

        transfer_schedule(8, 4, ...) -> tensor([2, 4, 6, 8])
        transfer_schedule(7, 4, ...) -> tensor([1, 3, 5, 7])
    """
    base = block_size // steps
    sched = torch.full((steps,), base, dtype=torch.long, device=device)
    remainder = block_size % steps
    if remainder:
        sched[-remainder:] += 1
    return sched.cumsum(dim=0)


def cosine_remask_prob(step: int, total: int, max_prob: float = 0.1) -> float:
    """Cosine annealing from ``max_prob`` at step 0 down to 0 at final step."""
    t = step / max(total - 1, 1)
    return max_prob * 0.5 * (1.0 + math.cos(math.pi * t))


def linear_remask_prob(step: int, total: int, max_prob: float = 0.1) -> float:
    """Linear decay from ``max_prob`` at step 0 to 0 at final step."""
    t = step / max(total - 1, 1)
    return max_prob * (1.0 - t)


def noise_schedule_linear(t: Tensor, T: int = 1000) -> Tensor:
    """Simple linear noise level: mask probability grows linearly with t."""
    return t.float() / T


def noise_schedule_cosine(t: Tensor, T: int = 1000, s: float = 0.008) -> Tensor:
    """Cosine noise schedule (from DDPM/MDM literature)."""
    ft = torch.cos(((t.float() / T + s) / (1 + s)) * (math.pi / 2)) ** 2
    f0 = math.cos((s / (1 + s)) * (math.pi / 2)) ** 2
    return 1 - ft / f0

"""Block-diffusion sampler for direct-forward models (RemeDi).

Model-agnostic at the strategy level; model-aware only in that it expects
the adapter to provide a tokenizer and KV-cache-compatible forward call.
This is the generalized replacement for ``remedi_infer/sampler.py``.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from ..core.schedules import transfer_schedule
from ..strategies.base import BaseStrategy, StepState


@torch.no_grad()
def run_block_diffusion(
    adapter,
    messages: list[dict[str, str]],
    strategy: BaseStrategy,
    steps: int = 32,
    max_length: int = 1024,
    block_size: int = 32,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Run block-diffusion inference with a pluggable strategy.

    This works with any adapter that exposes:
    - ``adapter.tokenizer`` — an HF tokenizer
    - ``adapter.model`` — a callable with ``(x, position_ids, kv_cache, ...)``
    - ``adapter.new_kv_cache()`` — returns a fresh KV-cache object
    - ``adapter.meta.mask_token_id`` / ``eos_token_id``
    - ``adapter.device``

    In practice this means :class:`~mdm_playground.models.remedi.RemeDiAdapter`.
    Subprocess-based models (ReMDM, PRISM) use their own ``sample()`` method.

    Args:
        adapter:     A direct-forward model adapter.
        messages:    Chat messages, e.g. ``[{"role": "user", "content": "..."}]``.
        strategy:    A :class:`~mdm_playground.strategies.base.BaseStrategy`.
        steps:       Diffusion steps per block.
        max_length:  Maximum new tokens to generate.
        block_size:  Tokens per block.
        seed:        Random seed for determinism.

    Returns:
        Dict with ``"generated_text"``, ``"prompt_text"``, ``"blocks"``.
        Each block has ``"block_idx"`` and ``"steps"`` (list of per-step dicts).

        Per-step dict fields::

            step, tokens, mask_positions, confidence,
            unmask_indices, remask_indices
    """
    if seed is not None:
        torch.manual_seed(seed)

    tokenizer = adapter.tokenizer
    model = adapter.model
    device = adapter.device
    mask_id = adapter.meta.mask_token_id
    eos_id = adapter.meta.eos_token_id

    # ------------------------------------------------------------------
    # Tokenize
    # ------------------------------------------------------------------
    prompt_str: str = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(prompt_str, return_tensors="pt", padding=True, padding_side="left")
    x_prefix: Tensor = inputs["input_ids"].to(device)
    attn_mask: Tensor = inputs["attention_mask"].to(device)
    prompt_len: int = int(attn_mask.sum().item())

    position_ids_prefix = (
        torch.arange(x_prefix.shape[1], device=device, dtype=torch.long).unsqueeze(0)
        - (1 - attn_mask).sum(dim=-1, keepdim=True)
    )

    # ------------------------------------------------------------------
    # Warm-up KV cache on prefix
    # ------------------------------------------------------------------
    kv_cache = adapter.new_kv_cache()
    use_amp = device.type == "cuda"
    with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
        model(x_prefix, position_ids=position_ids_prefix, kv_cache=kv_cache, update_kv_cache=True)

    # ------------------------------------------------------------------
    # Block diffusion loop
    # ------------------------------------------------------------------
    sched = transfer_schedule(block_size, steps, device)
    all_blocks: list[dict[str, Any]] = []
    generated_blocks: list[Tensor] = []

    cur_block = 0
    total_generated = 0
    stop = False

    while total_generated < max_length and not stop:
        block_start = prompt_len + cur_block * block_size
        x_t = torch.full((1, block_size), fill_value=mask_id, device=device, dtype=torch.long)
        position_ids = torch.arange(
            block_start, block_start + block_size, device=device, dtype=torch.long
        ).unsqueeze(0)
        committed = torch.zeros(1, block_size, dtype=torch.bool, device=device)

        step_logs: list[dict[str, Any]] = []

        for i in range(steps):
            mask_index = (x_t == mask_id)

            # Forward
            fwd = adapter.forward(x_t, position_ids=position_ids, kv_cache=kv_cache)
            token_logits = fwd.token_logits         # [1, L, V]
            x0 = fwd.x0                             # [1, L]
            x0 = torch.where(mask_index, x0, x_t)  # keep committed

            # Confidence
            confidence = fwd.confidence             # [1, L] or None
            if confidence is None:
                import torch.nn.functional as F
                probs = F.softmax(token_logits, dim=-1)
                confidence = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

            # Build state and call strategy
            state = StepState(
                x_t=x_t, x0=x0,
                token_logits=token_logits,
                confidence=confidence,
                mask_index=mask_index,
                committed=committed,
                step=i, total_steps=steps,
                num_to_transfer=int(sched[i].item()),
                mask_token_id=mask_id,
            )
            unmask_pos, remask_pos = strategy.select(state)

            # Apply unmask
            if unmask_pos.shape[-1] > 0:
                x_t[0, unmask_pos[0]] = x0[0, unmask_pos[0]]
                committed[0, unmask_pos[0]] = True

            # Apply remask
            remask_flat: list[int] = []
            if remask_pos is not None and remask_pos.shape[-1] > 0:
                valid = remask_pos[0][remask_pos[0] >= 0]
                if valid.numel() > 0:
                    x_t[0, valid] = mask_id
                    committed[0, valid] = False
                    remask_flat = valid.tolist()

            step_logs.append({
                "step": i,
                "tokens": x_t[0].tolist(),
                "mask_positions": (x_t[0] == mask_id).nonzero(as_tuple=True)[0].tolist(),
                "confidence": confidence[0].tolist(),
                "unmask_indices": unmask_pos[0].tolist() if unmask_pos.shape[-1] > 0 else [],
                "remask_indices": remask_flat,
            })

        if eos_id in x_t[0]:
            stop = True

        generated_blocks.append(x_t.clone())
        all_blocks.append({"block_idx": cur_block, "steps": step_logs})
        total_generated += block_size
        cur_block += 1

        if not stop:
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
                model(x_t, position_ids=position_ids, kv_cache=kv_cache, update_kv_cache=True)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
    full = torch.cat([x_prefix] + generated_blocks, dim=1)[0]
    gen = full[prompt_len:]
    if eos_id in gen:
        eos_idx = (gen == eos_id).nonzero(as_tuple=True)[0][0].item()
        gen = gen[:eos_idx]

    return {
        "prompt_text": tokenizer.decode(full[:prompt_len].tolist(), skip_special_tokens=True),
        "generated_text": tokenizer.decode(gen.tolist(), skip_special_tokens=True),
        "blocks": all_blocks,
    }

"""Unit and integration tests for remedi_infer.

Run with::

    pytest tests/test_infer.py -v

These tests use a tiny mock model so no HuggingFace checkpoint is required.
"""

from __future__ import annotations

import sys
import os
import types
from dataclasses import dataclass
from typing import Optional, Any, NamedTuple

import pytest
import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Make remedi_infer importable without a real 'remedi' package by stubbing it.
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Build a minimal stub for the 'remedi' package so import doesn't fail when
# external/remedi is absent from PYTHONPATH.
if "remedi" not in sys.modules:
    _remedi_stub = types.ModuleType("remedi")

    class _FakeConfig:
        mask_token_id = 99
        eos_token_id = 98

    class _FakeModelOutput(NamedTuple):
        logits: Tensor
        attn_key_values: Any = None
        hidden_states: Any = None
        confidences: Optional[Tensor] = None

    class _FakeRemeDiUPMModelLM(nn.Module):
        """Minimal mock of RemeDiUPMModelLM with correct output shapes."""

        def __init__(self, vocab_size: int = 100, seq_len: int = 16):
            super().__init__()
            self.vocab_size = vocab_size
            self.seq_len = seq_len
            self.config = _FakeConfig()
            self._dummy = nn.Linear(1, 1)  # so .parameters() is non-empty

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def forward(self, input_ids: Tensor, **kwargs) -> _FakeModelOutput:
            B, L = input_ids.shape
            # Deterministic-ish logits: token 0 always has highest logit.
            logits = torch.zeros(B, L, self.vocab_size)
            logits[:, :, 0] = 2.0  # token 0 wins
            confidences = torch.zeros(B, L, 1)  # raw UPS logit -> sigmoid -> 0.5
            return _FakeModelOutput(logits=logits, confidences=confidences)

    _remedi_stub.RemeDiUPMModelLM = _FakeRemeDiUPMModelLM
    sys.modules["remedi"] = _remedi_stub
    sys.modules["remedi.modelling_remedi_bitowel"] = _remedi_stub

# Stub DynamicCache inside the modelling module so sampler can import it.
if not hasattr(sys.modules.get("remedi.modelling_remedi_bitowel", object()), "DynamicCache"):
    class _FakeDynamicCache:
        def __init__(self):
            self.layers = {}

        def update(self, k, v, layer_id):
            if layer_id not in self.layers:
                self.layers[layer_id] = (k, v)
            else:
                prev_k, prev_v = self.layers[layer_id]
                k = torch.cat([prev_k, k], dim=-2)
                v = torch.cat([prev_v, v], dim=-2)
                self.layers[layer_id] = (k, v)
            return self.layers[layer_id]

    sys.modules["remedi.modelling_remedi_bitowel"].DynamicCache = _FakeDynamicCache

from mdm_playground.strategies import (
    StepState,
    BaselineUnmaskStrategy,
    RemediPolicyStrategy,
    ConfidenceThresholdRemaskStrategy,
    TopKLowConfidenceRemaskStrategy,
    ScheduledRemaskStrategy,
)
from mdm_playground.models.remedi import _compute_confidence
from mdm_playground.models.base import ForwardOutput, ModelMeta
from mdm_playground.samplers.block_diffusion import run_block_diffusion
from mdm_playground.core.schedules import transfer_schedule as _transfer_schedule


# ---------------------------------------------------------------------------
# Fake adapter (replaces old RemeDiModelBundle mock)
# ---------------------------------------------------------------------------

EOS_ID = 98


class _StubTokenizer:
    """Tiny tokenizer stub — 4 fixed prefix tokens."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return messages[0]["content"]

    def __call__(self, text, return_tensors="pt", padding=True, padding_side="left"):
        ids = torch.tensor([[1, 2, 3, 4]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(t) for t in ids if t not in (MASK_ID, EOS_ID))


class _FakeForwardAdapter:
    """Minimal RemeDiAdapter-compatible object for sampler tests."""

    def __init__(self, use_ups: bool = True):
        self.device = torch.device("cpu")
        self.use_ups = use_ups
        self.meta = ModelMeta(
            mask_token_id=MASK_ID,
            eos_token_id=EOS_ID,
            vocab_size=VOCAB,
            model_id="fake",
            device=self.device,
        )
        self._tokenizer = _StubTokenizer()
        # Raw model callable — used only for prefix KV-cache warmup; return
        # value is discarded by run_block_diffusion.
        self.model = lambda *a, **kw: None

    @property
    def tokenizer(self):
        return self._tokenizer

    def new_kv_cache(self):
        return sys.modules["remedi.modelling_remedi_bitowel"].DynamicCache()

    def forward(self, x: torch.Tensor, **kwargs) -> ForwardOutput:
        B, L = x.shape
        token_logits = torch.zeros(B, L, VOCAB)
        token_logits[:, :, 0] = 2.0  # token 0 always wins
        x0 = token_logits.argmax(dim=-1)
        ups_raw = torch.zeros(B, L, 1)
        confidence = _compute_confidence(token_logits, x0, ups_raw, use_ups=self.use_ups)
        return ForwardOutput(token_logits=token_logits, confidence=confidence, x0=x0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BLOCK_SIZE = 8
VOCAB = 100
MASK_ID = 99

@pytest.fixture
def base_state() -> StepState:
    B, L = 1, BLOCK_SIZE
    x_t = torch.full((B, L), fill_value=MASK_ID, dtype=torch.long)
    x0 = torch.zeros(B, L, dtype=torch.long)
    token_logits = torch.randn(B, L, VOCAB)
    confidence = torch.rand(B, L)
    mask_index = torch.ones(B, L, dtype=torch.bool)
    committed = torch.zeros(B, L, dtype=torch.bool)
    return StepState(
        x_t=x_t, x0=x0, token_logits=token_logits,
        confidence=confidence, mask_index=mask_index,
        committed=committed, step=0, total_steps=8,
        num_to_transfer=2, mask_token_id=MASK_ID,
    )


@pytest.fixture
def partial_state() -> StepState:
    """State where some positions are already committed."""
    B, L = 1, BLOCK_SIZE
    x_t = torch.full((B, L), fill_value=MASK_ID, dtype=torch.long)
    committed = torch.zeros(B, L, dtype=torch.bool)
    committed[0, :3] = True  # first 3 positions committed
    x_t[0, :3] = 42           # non-mask token
    x0 = torch.zeros(B, L, dtype=torch.long)
    token_logits = torch.randn(B, L, VOCAB)
    confidence = torch.rand(B, L)
    mask_index = ~committed
    return StepState(
        x_t=x_t, x0=x0, token_logits=token_logits,
        confidence=confidence, mask_index=mask_index,
        committed=committed, step=4, total_steps=8,
        num_to_transfer=4, mask_token_id=MASK_ID,
    )


@pytest.fixture
def mock_adapter() -> _FakeForwardAdapter:
    return _FakeForwardAdapter(use_ups=True)



# ---------------------------------------------------------------------------
# Tests: confidence computation
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_ups_sigmoid_shape(self):
        B, L, V = 2, 8, VOCAB
        token_logits = torch.randn(B, L, V)
        x0 = token_logits.argmax(dim=-1)
        ups = torch.randn(B, L, 1)
        conf = _compute_confidence(token_logits, x0, ups, use_ups=True)
        assert conf.shape == (B, L), f"Expected [{B}, {L}], got {conf.shape}"

    def test_ups_values_in_unit_interval(self):
        B, L, V = 1, 8, VOCAB
        token_logits = torch.randn(B, L, V)
        x0 = token_logits.argmax(dim=-1)
        ups = torch.randn(B, L, 1) * 5  # large logits -> should still be in [0,1]
        conf = _compute_confidence(token_logits, x0, ups, use_ups=True)
        assert conf.min().item() >= 0.0
        assert conf.max().item() <= 1.0

    def test_fallback_shape_and_range(self):
        B, L, V = 1, 16, VOCAB
        token_logits = torch.randn(B, L, V)
        x0 = token_logits.argmax(dim=-1)
        conf = _compute_confidence(token_logits, x0, None, use_ups=False)
        assert conf.shape == (B, L)
        assert conf.min().item() >= 0.0
        assert conf.max().item() <= 1.0


# ---------------------------------------------------------------------------
# Tests: transfer schedule
# ---------------------------------------------------------------------------

class TestTransferSchedule:
    def test_cumulative_ends_at_block_size(self):
        for block_size, steps in [(8, 4), (32, 32), (16, 5)]:
            sched = _transfer_schedule(block_size, steps, torch.device("cpu"))
            assert sched[-1].item() == block_size, (
                f"block_size={block_size}, steps={steps}: "
                f"last cumulative value should be {block_size}, got {sched[-1].item()}"
            )

    def test_monotone_nondecreasing(self):
        sched = _transfer_schedule(32, 10, torch.device("cpu"))
        diffs = sched[1:] - sched[:-1]
        assert (diffs >= 0).all()


# ---------------------------------------------------------------------------
# Tests: strategies
# ---------------------------------------------------------------------------

class TestBaselineStrategy:
    def test_no_remask(self, base_state):
        strat = BaselineUnmaskStrategy()
        unmask, remask = strat.select(base_state)
        assert remask is None

    def test_unmask_count(self, base_state):
        strat = BaselineUnmaskStrategy()
        unmask, _ = strat.select(base_state)
        # base_state: already=0, num_to_transfer=2 -> should unmask 2
        assert unmask.shape[-1] == 2

    def test_unmask_within_mask_positions(self, partial_state):
        strat = BaselineUnmaskStrategy()
        unmask, _ = strat.select(partial_state)
        # All unmasked indices must currently be MASKED in the state.
        if unmask.shape[-1] > 0:
            for idx in unmask[0].tolist():
                assert partial_state.mask_index[0, idx].item(), (
                    f"Position {idx} was not masked but was selected for unmask."
                )

    def test_no_out_of_bounds(self, base_state):
        strat = BaselineUnmaskStrategy()
        unmask, _ = strat.select(base_state)
        L = base_state.x_t.shape[1]
        for idx in unmask[0].tolist():
            assert 0 <= idx < L


class TestRemediPolicyStrategy:
    def test_unmask_count(self, base_state):
        strat = RemediPolicyStrategy()
        unmask, remask = strat.select(base_state)
        assert unmask.shape[-1] == base_state.num_to_transfer

    def test_remask_is_complement(self, base_state):
        strat = RemediPolicyStrategy()
        unmask, remask = strat.select(base_state)
        L = base_state.x_t.shape[1]
        unmask_set = set(unmask[0].tolist())
        remask_valid = remask[0][remask[0] >= 0].tolist()
        remask_set = set(remask_valid)
        # unmask and remask sets must be disjoint
        assert unmask_set.isdisjoint(remask_set), "unmask and remask overlap"
        # union must cover the full sequence
        assert unmask_set | remask_set == set(range(L))

    def test_no_out_of_bounds(self, base_state):
        strat = RemediPolicyStrategy()
        unmask, remask = strat.select(base_state)
        L = base_state.x_t.shape[1]
        for idx in unmask[0].tolist():
            assert 0 <= idx < L
        if remask is not None:
            for idx in remask[0][remask[0] >= 0].tolist():
                assert 0 <= idx < L


class TestConfidenceThresholdStrategy:
    def test_remask_only_committed(self, partial_state):
        # Force partial_state confidence to be very low for committed positions.
        partial_state.confidence[0, :3] = 0.01  # far below any reasonable tau
        strat = ConfidenceThresholdRemaskStrategy(tau=0.5)
        unmask, remask = strat.select(partial_state)
        if remask is not None:
            for idx in remask[0][remask[0] >= 0].tolist():
                assert partial_state.committed[0, idx].item(), (
                    f"Position {idx} remasked but was not committed."
                )

    def test_high_tau_triggers_remask(self, partial_state):
        partial_state.confidence[0, :] = 0.1
        strat = ConfidenceThresholdRemaskStrategy(tau=0.9)
        _, remask = strat.select(partial_state)
        assert remask is not None and remask.shape[-1] > 0


class TestTopKLowConfidenceStrategy:
    def test_remask_count(self, partial_state):
        partial_state.confidence[0, :3] = 0.01
        strat = TopKLowConfidenceRemaskStrategy(k_remask=2)
        _, remask = strat.select(partial_state)
        assert remask is not None
        assert remask.shape[-1] == 2

    def test_remask_only_committed(self, partial_state):
        strat = TopKLowConfidenceRemaskStrategy(k_remask=2)
        _, remask = strat.select(partial_state)
        if remask is not None:
            for idx in remask[0].tolist():
                assert partial_state.committed[0, idx].item()


class TestScheduledRemaskStrategy:
    def test_early_steps_may_remask(self, partial_state):
        torch.manual_seed(0)
        strat = ScheduledRemaskStrategy(max_remask_prob=1.0, schedule="linear")
        partial_state.step = 0  # p = max at step 0
        _, remask = strat.select(partial_state)
        # With prob=1.0 and 3 committed tokens, should remask at least 1.
        if remask is not None and remask.shape[-1] > 0:
            valid = remask[0][remask[0] >= 0]
            for idx in valid.tolist():
                assert partial_state.committed[0, idx].item()

    def test_final_step_no_remask(self, partial_state):
        strat = ScheduledRemaskStrategy(max_remask_prob=0.1, schedule="linear")
        partial_state.step = partial_state.total_steps - 1  # p = 0 at final step
        _, remask = strat.select(partial_state)
        # At t=1 (final step), linear schedule gives p=0.
        assert remask is None


# ---------------------------------------------------------------------------
# Integration test: full sampler loop with mock adapter
# ---------------------------------------------------------------------------

class TestSamplerIntegration:
    @pytest.mark.parametrize("strategy_cls", [
        BaselineUnmaskStrategy,
        RemediPolicyStrategy,
    ])
    def test_two_step_run(self, mock_adapter, strategy_cls):
        """Run 2 steps, 1 block; check all output shapes and invariants."""
        strategy = strategy_cls()
        result = run_block_diffusion(
            adapter=mock_adapter,
            messages=[{"role": "user", "content": "hello"}],
            strategy=strategy,
            steps=2,
            max_length=BLOCK_SIZE,
            block_size=BLOCK_SIZE,
            seed=42,
        )

        assert "generated_text" in result
        assert "blocks" in result
        assert len(result["blocks"]) >= 1

        block = result["blocks"][0]
        steps = block["steps"]
        assert len(steps) == 2  # 2 steps

        for step_data in steps:
            L = len(step_data["tokens"])
            assert L == BLOCK_SIZE, f"tokens length {L} != block_size {BLOCK_SIZE}"

            conf = step_data["confidence"]
            assert len(conf) == BLOCK_SIZE, f"confidence length {len(conf)} != {BLOCK_SIZE}"
            assert all(0.0 <= c <= 1.0 for c in conf), "confidence not in [0, 1]"

            unmask = step_data["unmask_indices"]
            remask = step_data["remask_indices"]

            # All indices must be valid
            for idx in unmask + remask:
                assert 0 <= idx < BLOCK_SIZE, f"index {idx} out of bounds [0, {BLOCK_SIZE})"

            # Unmask and remask must be disjoint
            assert set(unmask).isdisjoint(set(remask)), "unmask and remask overlap"

    def test_confidence_shape_per_step(self, mock_adapter):
        mock_adapter.use_ups = False
        result = run_block_diffusion(
            adapter=mock_adapter,
            messages=[{"role": "user", "content": "test"}],
            strategy=BaselineUnmaskStrategy(),
            steps=4,
            max_length=BLOCK_SIZE,
            block_size=BLOCK_SIZE,
            seed=0,
        )
        for block in result["blocks"]:
            for step_data in block["steps"]:
                assert len(step_data["confidence"]) == BLOCK_SIZE

    def test_deterministic_under_seed(self, mock_adapter):
        kwargs = dict(
            adapter=mock_adapter,
            messages=[{"role": "user", "content": "hi"}],
            strategy=ScheduledRemaskStrategy(max_remask_prob=0.3),
            steps=2,
            max_length=BLOCK_SIZE,
            block_size=BLOCK_SIZE,
            seed=7,
        )
        r1 = run_block_diffusion(**kwargs)
        r2 = run_block_diffusion(**kwargs)
        assert r1["generated_text"] == r2["generated_text"]
        # Check first step tokens identical
        t1 = r1["blocks"][0]["steps"][0]["tokens"]
        t2 = r2["blocks"][0]["steps"][0]["tokens"]
        assert t1 == t2


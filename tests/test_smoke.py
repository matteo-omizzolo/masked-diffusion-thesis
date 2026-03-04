"""Lightweight smoke tests for all three model backends.

These tests require **no GPU and no downloaded weights**.  They exercise:
- Toy/dry-run execution paths (no subprocess I/O)
- CLI argument parsing and end-to-end dispatch
- The full RemeDi inference pipeline via a FakeAdapter (already in test_infer,
  but reproduced here with explicit ``@pytest.mark.smoke`` tags for CI filtering)

Heavier tests that need real checkpoints are marked ``@pytest.mark.integration``
and are skipped by default.  Run them explicitly with::

    pytest -m integration

Run only smoke tests::

    pytest tests/test_smoke.py -v
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Stub 'remedi' for imports that reach RemeDiAdapter path
# ---------------------------------------------------------------------------
import types, os

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if "remedi" not in sys.modules:
    _r = types.ModuleType("remedi")
    sys.modules["remedi"] = _r
    sys.modules["remedi.modelling_remedi_bitowel"] = _r

# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------

from mdm_playground.models.remdm import ReMDMAdapter, ReMDMConfig
from mdm_playground.models.prism import PRISMAdapter, PRISMConfig


# ===========================================================================
# C2 — ReMDM smoke tests
# ===========================================================================

class TestReMDMSmoke:
    def test_toy_sample_keys(self):
        """Toy mode should return tokens without hitting any subprocess."""
        cfg = ReMDMConfig(toy_mode=True, dry_run=False)
        adapter = ReMDMAdapter(cfg=cfg, run_output_dir=Path(tempfile.mkdtemp()))
        result = adapter.sample()
        assert "tokens" in result, "toy sample must return 'tokens'"
        assert "meta" in result, "toy sample must return 'meta'"
        assert result["meta"]["toy"] is True

    def test_toy_token_shape(self):
        cfg = ReMDMConfig(toy_mode=True)
        adapter = ReMDMAdapter(cfg=cfg, run_output_dir=Path(tempfile.mkdtemp()))
        result = adapter.sample()
        toks = result["tokens"]
        assert isinstance(toks, torch.Tensor), "tokens must be a Tensor"
        assert toks.ndim >= 1

    def test_dry_run_returns_command(self):
        """Dry-run mode should build a multi-word command without executing it."""
        cfg = ReMDMConfig(toy_mode=False, dry_run=True, steps=64)
        adapter = ReMDMAdapter(cfg=cfg, run_output_dir=Path(tempfile.mkdtemp()))
        result = adapter.sample()
        assert result.get("dry_run") is True, "dry_run key must be True"
        cmd = result.get("command", [])
        assert len(cmd) >= 3, f"Expected non-trivial command, got: {cmd}"
        # Command must invoke main.py from the remdm submodule
        assert "main" in " ".join(cmd), f"'main' not found in command: {cmd}"

    def test_dry_run_includes_steps_override(self):
        cfg = ReMDMConfig(toy_mode=False, dry_run=True, steps=128, strategy="remdm-conf")
        adapter = ReMDMAdapter(cfg=cfg, run_output_dir=Path(tempfile.mkdtemp()))
        result = adapter.sample()
        cmd_str = " ".join(result["command"])
        assert "sampling.steps=128" in cmd_str, f"steps override missing: {cmd_str}"
        assert "sampling.sampler=remdm-conf" in cmd_str, f"strategy override missing: {cmd_str}"

    def test_load_factory(self):
        """ReMDMAdapter.load() should return a working adapter."""
        adapter = ReMDMAdapter.load(
            run_output_dir=Path(tempfile.mkdtemp()),
            cfg=ReMDMConfig(toy_mode=True),
        )
        result = adapter.sample()
        assert "tokens" in result


# ===========================================================================
# C2 — PRISM smoke tests
# ===========================================================================

class TestPRISMSmoke:
    def test_toy_sample_keys(self):
        cfg = PRISMConfig(toy_mode=True, dry_run=False)
        adapter = PRISMAdapter(cfg=cfg, run_output_dir=Path(tempfile.mkdtemp()))
        result = adapter.sample()
        assert "tokens" in result
        assert "confidence" in result
        assert result["meta"]["toy"] is True

    def test_toy_confidence_in_unit_interval(self):
        cfg = PRISMConfig(toy_mode=True)
        adapter = PRISMAdapter(cfg=cfg, run_output_dir=Path(tempfile.mkdtemp()))
        result = adapter.sample()
        conf = result["confidence"]
        assert isinstance(conf, torch.Tensor)
        assert conf.min().item() >= 0.0
        assert conf.max().item() <= 1.0

    def test_dry_run_returns_command(self):
        cfg = PRISMConfig(toy_mode=False, dry_run=True, steps=32)
        adapter = PRISMAdapter(cfg=cfg, run_output_dir=Path(tempfile.mkdtemp()))
        result = adapter.sample()
        assert result.get("dry_run") is True
        cmd = result.get("command", [])
        assert len(cmd) >= 3

    def test_dry_run_includes_steps_override(self):
        cfg = PRISMConfig(toy_mode=False, dry_run=True, steps=256)
        adapter = PRISMAdapter(cfg=cfg, run_output_dir=Path(tempfile.mkdtemp()))
        result = adapter.sample()
        cmd_str = " ".join(result["command"])
        assert "sampling.steps=256" in cmd_str, f"steps override missing: {cmd_str}"


# ===========================================================================
# CLI smoke tests
# ===========================================================================

_PYTHON = sys.executable


class TestCLISmoke:
    def test_help_exits_zero(self):
        """CLI --help must print usage and exit 0."""
        result = subprocess.run(
            [_PYTHON, "-m", "mdm_playground.cli.run", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"--help exited {result.returncode}:\n{result.stderr}"
        assert "remedi" in result.stdout.lower()

    def test_remdm_toy_via_cli(self):
        """End-to-end CLI dispatch for ReMDM toy mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    _PYTHON, "-m", "mdm_playground.cli.run",
                    "--method", "remdm",
                    "--toy_mode",
                    "--steps", "16",
                    "--out_dir", tmpdir,
                ],
                capture_output=True, text=True,
            )
            assert result.returncode == 0, (
                f"CLI remdm toy failed (rc={result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
            out = Path(tmpdir)
            assert (out / "run_meta.json").exists(), "run_meta.json not written"
            assert (out / "summary.json").exists(), "summary.json not written"

    def test_prism_toy_via_cli(self):
        """End-to-end CLI dispatch for PRISM toy mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    _PYTHON, "-m", "mdm_playground.cli.run",
                    "--method", "prism",
                    "--toy_mode",
                    "--steps", "16",
                    "--out_dir", tmpdir,
                ],
                capture_output=True, text=True,
            )
            assert result.returncode == 0, (
                f"CLI prism toy failed (rc={result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
            assert (Path(tmpdir) / "run_meta.json").exists()

    def test_remdm_dry_run_via_cli(self):
        """Dry-run mode should complete without subprocess execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    _PYTHON, "-m", "mdm_playground.cli.run",
                    "--method", "remdm",
                    "--dry_run",
                    "--steps", "256",
                    "--out_dir", tmpdir,
                ],
                capture_output=True, text=True,
            )
            assert result.returncode == 0, (
                f"dry-run failed (rc={result.returncode}):\n{result.stderr}"
            )
            assert "DRY RUN" in result.stdout, "Expected 'DRY RUN' in output"

    def test_prism_dry_run_via_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    _PYTHON, "-m", "mdm_playground.cli.run",
                    "--method", "prism",
                    "--dry_run",
                    "--steps", "256",
                    "--out_dir", tmpdir,
                ],
                capture_output=True, text=True,
            )
            assert result.returncode == 0, (
                f"PRISM dry-run failed (rc={result.returncode}):\n{result.stderr}"
            )
            assert "DRY RUN" in result.stdout


# ===========================================================================
# @integration — real model tests (skipped unless -m integration)
# ===========================================================================

@pytest.mark.integration
class TestRemeDiRealModel:
    """Requires internet access and ~2 GB disk for HF checkpoint download."""

    def test_remedi_baseline_cpu(self):
        """Download RemeDi-RL and run 2 steps on CPU with baseline strategy."""
        from mdm_playground.models.remedi import RemeDiAdapter
        from mdm_playground.strategies import BaselineUnmaskStrategy
        from mdm_playground.samplers import run_block_diffusion

        adapter = RemeDiAdapter.load(
            model_id="maple-research-lab/RemeDi-RL",
            device="cpu",
            use_ups=True,
        )
        result = run_block_diffusion(
            adapter=adapter,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            strategy=BaselineUnmaskStrategy(),
            steps=2,
            max_length=16,
            block_size=16,
            seed=0,
        )
        assert "generated_text" in result
        assert len(result["blocks"]) >= 1
        block = result["blocks"][0]
        assert len(block["steps"]) == 2
        for step in block["steps"]:
            conf = step["confidence"]
            assert len(conf) == 16
            assert all(0.0 <= c <= 1.0 for c in conf)

    def test_remedi_threshold_cpu(self):
        """Same as above but with ConfidenceThresholdRemaskStrategy."""
        from mdm_playground.models.remedi import RemeDiAdapter
        from mdm_playground.strategies import ConfidenceThresholdRemaskStrategy
        from mdm_playground.samplers import run_block_diffusion

        adapter = RemeDiAdapter.load(
            model_id="maple-research-lab/RemeDi-RL",
            device="cpu",
        )
        result = run_block_diffusion(
            adapter=adapter,
            messages=[{"role": "user", "content": "Hello"}],
            strategy=ConfidenceThresholdRemaskStrategy(tau=0.5),
            steps=2,
            max_length=16,
            block_size=16,
        )
        assert "generated_text" in result

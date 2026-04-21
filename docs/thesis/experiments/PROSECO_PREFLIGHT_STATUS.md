# ProSeCo-OWT Preflight Status

*Written: 2026-04-19.*
*Status: COMPLETE — pilot job 479381 submitted.*

---

## 1. Checkpoint

| Item | Value |
|------|-------|
| HuggingFace repo | `kuleshov-group/proseco-owt` |
| Local path (HPC) | `~/mdm/checkpoints/proseco_owt/` |
| Downloaded | 2026-04-18 via `scripts/stage_proseco_owt.py` |
| Size | 648 MB (`model.safetensors`) |
| Weight format | safetensors (no pytorch_model.bin) |

Config values from `config.json`:

| Key | Value | Notes |
|-----|-------|-------|
| `vocab_size` | 50258 | GPT-2 tokenizer |
| `model_length` | 1024 | Sequence length |
| `hidden_dim` | 768 | |
| `n_blocks` | 12 | |
| `n_heads` | 12 | |
| `time_conditioning` | false | Backbone ignores σ; annealing is purely corrector-driven |
| `return_dict` | false | forward() returns raw logits tensor |
| `mask_index` | 50257 | vocab_size − 1 |

---

## 2. Loading approach

`_load_proseco_owt()` uses `importlib.util.spec_from_file_location` to load
`modeling_proseco.py` and `configuration_proseco.py` directly from the snapshot
directory, without `trust_remote_code=True`. This avoids HuggingFace's module
caching (`~/.cache/huggingface/modules`) which fails when home-dir quota is
near-full (~47GB of 50GB).

A fake package namespace `_proseco_snapshot_pkg` is used to satisfy the relative
import `from .configuration_proseco import ProsecoConfig` in `modeling_proseco.py`.

Weights are loaded via `safetensors.torch.load_file` (safetensors v0.7.0).

---

## 3. Fixes applied during preflight (2026-04-19)

| Issue | Root cause | Fix |
|-------|-----------|-----|
| Disk quota exceeded | `trust_remote_code=True` writes to `~/.cache` | Direct importlib loading |
| Relative import fails | `modeling_proseco.py` loaded as standalone module | Fake package namespace |
| 6-value unpack error | `sigma` shape `(B,1)` → `t[:,None]` gives `(B,1,1)` in TimestepEmbedder | `squeeze(-1)` in `_forward` |
| `'Tensor' has no attribute logits` | `config.return_dict=false` → raw tensor returned | `out.logits if hasattr(out,'logits') else out` |

---

## 4. CPU preflight results (2026-04-19)

Device: CPU, T=4, corrector_steps=1, seed=42

```
[PASS] Snapshot files present — all present
[PASS] ProSeCoOWTGenerator instantiates
[PASS] run_base: neg_nll finite — neg_nll=-7.3179
[PASS] run_branch(t=2): neg_nll finite — neg_nll=-7.2760
[PASS] Δ_t(t=2) non-trivially zero — Δ_t=+0.041880
```

Per-step signal trace:

| t | H (entropy) | inv_margin | n_rev |
|---|------------|-----------|-------|
| 0 | 7.576 | 1.000 | 234 |
| 1 | 3.716 | 0.507 | 482 |
| 2 | 2.504 | 0.359 | 757 |
| 3 | 1.903 | 0.285 | 1023 |

**Key findings:**
- Δ_t = +0.042 confirms the corrector modifies committed tokens non-trivially.
  The proseco-owt backbone is co-trained and responds to the corrector.
- `n_rev` grows monotonically: 234 → 1023, confirming MDLM predictor unmasking
  behaves correctly.
- Entropy declines sharply: 7.58 → 1.90. Signal has dynamic range.
- `time_conditioning=false` means the backbone is applied uniformly across σ.
  Annealing in the corrector is purely through repeated backbone calls.

---

## 5. Pilot job

| Item | Value |
|------|-------|
| Job ID | 479382 (479381 failed: missing script; resubmitted) |
| Command | `sbatch hpc/phase1_proseco_owt.sbatch` |
| Config | N=20, T=64, M=15, P=120, B∈{4,8,16}, seed=42 |
| Wall time | 8h (A100 80GB) |
| Expected output | `results/phase1_proseco_owt/summary.json` |
| Submit time | 2026-04-19 |

---

## 6. Next steps

1. Monitor job 479381: `squeue -u 3316152`
2. Pull results when done: check `results/phase1_proseco_owt/summary.json`
3. Evaluate §7.2 success criteria:
   - C1: n_positive_delta_steps > 10 and peak_mean_delta ≥ 0.05
   - C2: |Spearman_mean| ≥ 0.10 for ≥1 signal
   - C6: ≥1 top_B policy beats uniform at B=8 or B=16
4. If C1+C2 pass → submit full run (N=50, T=64, M=30, P=300)
5. Write `PROSECO_ANALYSIS.md` with full results

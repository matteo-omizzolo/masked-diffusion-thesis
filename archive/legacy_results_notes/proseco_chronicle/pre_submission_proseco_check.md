# Pre-Submission ProSeCo Check

**Date:** 2026-04-17  
**Checker:** systematic code audit before HPC pilot submission  
**Verdict: CLEARED — submit pilot as-is**

---

## 1. Checkpoint path

| Check | Result |
|---|---|
| Path in sbatch | `/home/3316152/mdm/checkpoints/mdlm.ckpt` |
| File exists on HPC | ✅ YES (`ls ~/mdm/checkpoints/` confirmed) |
| Same path as MDLM diagnostic run (478600) | ✅ YES — already validated at load time |

---

## 2. Backend corrector mechanism (`backends/proseco.py`)

| Check | Result |
|---|---|
| Corrector starts from x̂_0 = argmax(p_x0) | ✅ `corrector_x = p_x0.argmax(-1)` (line 301) |
| Annealed refinement loop: corrector_steps iterations | ✅ `for j in range(self.corrector_steps)` (line 308) |
| Noise levels: tau from linspace(1.0, eps, corrector_steps+1) | ✅ `corrector_timesteps = torch.linspace(1.0, self.eps, ...)` |
| Applied to UNMASKED positions only | ✅ `unmasked = (x != self.mask_id); x_new[unmasked] = corrector_x[unmasked]` |
| Signals over UNMASKED positions (ProSeCo action set) | ✅ `revisable = (x[0] != self.mask_id)` in `_extract_signals` |
| Signal extraction zero-NFE (reuses p_x0_cache) | ✅ No extra `model.forward()` call in `_extract_signals` |
| p_x0_cache reset after corrector | ✅ `p_x0_cache = None` forces recomputation from corrected state |

**Important note: time_conditioning=False**

The MDLM backbone was trained with `time_conditioning=False`. In `_process_sigma`:
```python
if not self.time_conditioning:
    sigma = torch.zeros_like(sigma)
```
This means the noise level `tau` passed to the corrector is zeroed before the backbone. The corrector is effectively two iterations of `argmax(model.forward(corrector_x, σ=0))` applied to unmasked positions. The "annealing" across tau values is a no-op for this backbone.

**This is not a bug.** ProSeCo paper experiments use checkpoints trained with corrector co-training; on the MDLM backbone (not co-trained), the corrector is deterministic argmax refinement. This is documented in the backend audit. Whether Δ_t > 0 from this deterministic refinement is the central empirical question.

---

## 3. Script output paths (`scripts/run_phase1_proseco.py`)

| Output | Path | Created by |
|---|---|---|
| Protocol A trajectories | `{out_dir}/protocol_a/trajectory_{i}.json` | `out_dir.mkdir(parents=True, exist_ok=True)` |
| Protocol B schedules | `{out_dir}/protocol_b/schedule_{j}.json` | same |
| Protocol B pairs | `{out_dir}/protocol_b/pairs.json` | same |
| Policy comparison | `{out_dir}/policy_comparison/policy_comparison.json` | same |
| Summary | `{out_dir}/summary.json` | `compute_summary()` |
| Run config | `{out_dir}/run_config.json` | at startup |

For the pilot: `out_dir = results/phase1_proseco/` (relative to repo root). The sbatch script `cd`s to `REPO=~/mdm/masked-diffusion-thesis` before running, so the absolute path is `~/mdm/masked-diffusion-thesis/results/phase1_proseco/`.

✅ All output directories created at job start via `mkdir -p`.

---

## 4. HPC sbatch configuration (`hpc/phase1_proseco.sbatch`)

| Check | Result |
|---|---|
| Partition / QOS | `stud / stud` — correct |
| GPUs | `gres=gpu:1` — 1× A100 |
| Memory | `48G` — adequate for MDLM + GPT-2 + 1024-token sequences |
| Time limit | `8:00:00` — pilot est. ~3–5 hours, safe |
| Log output | `out/phase1_proseco_JOBID.out` — `out/` dir EXISTS |
| Log errors | `err/phase1_proseco_JOBID.err` — `err/` dir EXISTS |
| Active block | PILOT block (N=20, T=64, M=15, P=120) — ✅ |
| Sanity block | Commented out — ✅ |
| Full block | Commented out — ✅ |
| Checkpoint path | `/home/3316152/mdm/checkpoints/mdlm.ckpt` — ✅ |
| Python env | `conda activate remdm311` — same env as MDLM run 478600 |
| Package install | `pip install -e . -q --break-system-packages` — flag worked in run 478600 |

---

## 5. Import chain verification

| Module | Import | Status |
|---|---|---|
| `mdm_playground.scheduling` | `allocate_budget`, `estimate_single_step_gain`, `evaluate_schedule` | ✅ in `__init__.py` |
| `mdm_playground.scheduling.backends.proseco` | `ProSeCoGenerator` | ✅ file exists and surrogate validated |
| `mdm_playground.scheduling.allocation` | `ALLOCATION_POLICIES` | ✅ all policy names used in script are defined |
| `external/remdm/diffusion.py` | `Diffusion`, `_ddpm_caching_update`, `forward`, `noise` | ✅ confirmed signatures match usage |

**Policy names used in run_policy_comparison vs ALLOCATION_POLICIES:**

| Policy name in script | Defined in ALLOCATION_POLICIES | Result |
|---|---|---|
| `uniform` | ✅ | `_uniform_indices()` |
| `front` | ✅ | `list(range(B))` |
| `back` | ✅ | `list(range(T-B, T))` |
| `middle` | ✅ | `_middle_indices()` |
| `entropy_top_B` → `"top_B"` | ✅ | `_top_B_indices(descending=True)` |
| `margin_top_B` | ✅ | alias for `top_B` |
| `quality_top_B` | ✅ | alias for `top_B` |
| `entropy_burn_in_gated` → `"burn_in_gated"` | ✅ | excludes steps with entropy ≤ 0.0 |
| `oracle` → `"top_B"` with `mean_delta` | ✅ | top-B by actual Δ_t mean |
| `bottom_B` | ✅ | `_top_B_indices(descending=False)` |

---

## 6. Pilot configuration safety check

| Parameter | Value | Rationale |
|---|---|---|
| T | 64 | Same as MDLM diagnostic run 478600 |
| N | 20 | 20 trajectories → 20 × 64 = 1280 (t, i) data points for calibration |
| M | 15 | 15 random schedules per B → 45 total |
| P | 120 | 120 pairwise pairs for γ estimation |
| B_values | 4, 8, 16 | T/16, T/8, T/4 — meaningful budget fractions |
| corrector_steps | 2 | 2 NFEs per corrector loop (as in ProSeCo paper) |

**NFE budget estimate:**
- Protocol A: N × (T + T×(T+2)) = 20 × (64 + 64×66) = 20 × 4288 = 85,760 forward passes
- Protocol B (schedules): 45 × (T + (avg B + 2)) ≈ 45 × ~74 = ~3,330 forward passes  
- Protocol B (pairs): 120 × (T + (T + 2) + (T + 2) + (T + 4)) ≈ 120 × 268 = ~32,160 forward passes
- Policy comparison: 3B × 10 policies × (T + ~8) ≈ very rough ~7,000 forward passes
- **Total rough estimate: ~130,000 forward passes**

At ~A100 throughput of ~100ms/forward-pass for this model (1024-token, 12-layer ViT), 
this is ~3.6 hours. The 8-hour time limit is safe.

**Key empirical questions this pilot can answer:**
1. ✅ Whether any Δ_t > 0 (n_positive_delta_steps > 0 in summary.json)
2. ✅ Whether signals non-degenerate (entropy, margin, quality_mass ≠ 0 for t > 5)
3. ✅ Whether entropy predicts Δ_t (Spearman + ε in summary.json)
4. ✅ Whether burn-in region exists (T_low identification in summary.json)
5. ✅ Whether Theorem A is non-vacuous (bound_useful in theorem_A_bound_check)

---

## 7. Known limitations (documented, not bugs)

1. **time_conditioning=False**: Corrector noise annealing is a no-op; corrector is deterministic argmax refinement. This is a property of the MDLM checkpoint, not a code error. Documented in `proseco_backend_audit.md`.

2. **Checkpoint not ProSeCo co-trained**: `mdlm.ckpt` was trained without corrector loss; `proseco-owt` (HuggingFace) was co-trained. Using the non-co-trained checkpoint may reduce Δ_t magnitudes. Documented as an upgrade path.

3. **GPT-2 reference scorer**: NLL computed under GPT-2 (not the generative model itself). This is the primary F for relative comparisons; not suitable for absolute quality claims.

4. **N=20 pilot**: Small sample for variance estimation. ε and η_B estimates will have high uncertainty. If the pilot shows interesting patterns, follow up with N=50.

---

## Verdict

**CLEARED. No code changes required. Submit with:**

```bash
cd ~/mdm/masked-diffusion-thesis
sbatch hpc/phase1_proseco.sbatch
```

Expected outputs: `results/phase1_proseco/`, `out/phase1_proseco_JOBID.out`, `err/phase1_proseco_JOBID.err`

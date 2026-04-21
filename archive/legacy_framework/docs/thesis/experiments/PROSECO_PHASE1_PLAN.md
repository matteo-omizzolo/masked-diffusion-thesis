# ProSeCo Phase 1 Experimental Plan

*Written: 2026-04-18.*
*Status: ACTIVE — pilot not yet run.*

---

## 1. Motivation

MDLM-conf signal-aligned pilot (job 479257) showed:
- Δ_t > 0 at 25/64 steps (corrector active)
- But Spearman(signal, Δ_t) ≈ 0.02–0.05 (noise-level)
- margin_top_B beats uniform 4.3× at B=8 (policy works despite weak signal)
- Theorem A bounds remain non-trivial (ratio 10–14×)

Root cause: mdlm.ckpt backbone was not co-trained with the corrector; per-step
gain has high intra-trajectory variance overwhelms any signal correlation.
Scaling to N=50 would confirm weak Spearman with tighter CIs but not change
the conclusion.

**proseco-owt** (co-trained backbone) is the right next step.

---

## 2. Chosen configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Backbone | `kuleshov-group/proseco-owt` | Co-trained with corrector; non-trivial Δ_t expected |
| Corrector | Annealed refinement, argmax mode | Standard ProSeCo corrector; deterministic |
| corrector_steps | 1 | One inner refinement pass per application; cheapest |
| T | 64 | Same as MDLM-conf pilots; comparable results |
| Sequence length | 1024 (proseco-owt default) | Fixed by architecture |
| B values | 4, 8, 16 | Same as prior runs; comparable |
| Signals | entropy, inverse_margin, quality_mass_proxy over UNMASKED positions | Same signal family as proseco.py backend |

Fixing `corrector_steps=1` keeps the NFE budget transparent: Protocol A
measures the value of one corrector application at each step, using exactly
1 extra NFE per branch.

---

## 3. Experiments and expected outputs

### Protocol A (per-step calibration)

For each of N=20 trajectories, run `run_base` (records per-step signals) and
T=64 `run_branch` calls (one corrector application at each step t∈{0,…,63}).

Outputs per trajectory: `protocol_a/trajectory_{i}.json` with fields:
- `per_t[t].delta` = Δ_t = F(y^t) - F(y_base)  where F = neg_nll
- `per_t[t].entropy`, `inverse_margin`, `quality_mass_proxy`
- `per_t[t].unmasked_fraction`, `n_revisable`

Aggregate: `summary.calibration.*` with:
- `eps_rms` = RMS of (Δ_t - ψ̂(s_t)) residuals (Theorem A's ε)
- `spearman_mean` = mean per-trajectory Spearman correlation
- `peak_mean_delta`, `n_positive_delta_steps`

### Protocol B (additivity and budget allocation)

M=15 random schedules × 3 budget levels → η_B estimates.
P=120 pairwise gain pairs → γ estimates.

Outputs: `protocol_b/schedule_*.json`, `protocol_b/pairs.json`
Aggregate: `summary.eta_by_B`, `summary.gamma`, `summary.theorem_A_bound_check`

### Policy comparison

Evaluate 10 policies at each B ∈ {4, 8, 16}:
- uniform, front, back, middle
- entropy_top_B, margin_top_B, quality_top_B
- entropy_burn_in_gated
- oracle (top-B by measured mean Δ_t)
- bottom_B (sanity check)

Output: `summary.policy_comparison`

---

## 4. Pilot vs full decision tree

```
Pilot (N=20, T=64, M=15, P=120)
    │
    ├─ Δ_t > 0 at ≥10/64 steps? NO → diagnose (check backbone loading, corrector)
    │
    ├─ |Spearman| ≥ 0.10 for ≥1 signal? NO → report weak calibration, discuss in thesis
    │
    └─ YES → Full run (N=50, T=64, M=30, P=300)
                 → Update Theorem A with tightened ε
                 → Write ch7
```

Even if Spearman is weak (as with MDLM-conf), if policy_comparison shows
margin_top_B or entropy_top_B beats uniform at B=8 or B=16, that is a
positive empirical result for the thesis.

---

## 5. Compute footprint

| Phase | Wall time (estimated) | Notes |
|-------|----------------------|-------|
| Checkpoint staging | ~10 min | login node, downloads ~500 MB |
| CPU preflight (T=4) | ~5 min | login node |
| Pilot (N=20, T=64) | ~4–6 h | A100 80GB, 64×21 = 1344 forward passes per trajectory |
| Full (N=50, T=64) | ~10–14 h | Scale --time to 14:00:00 |

proseco-owt forward pass is ~2× slower than MDLM on the same hardware (larger
seq_len=1024 vs 128, plus corrector annealing steps).

---

## 6. Files created / modified

| File | Status |
|------|--------|
| `src/mdm_playground/scheduling/backends/proseco_owt.py` | NEW — ProSeCo-OWT backend |
| `scripts/stage_proseco_owt.py` | NEW — checkpoint staging + flash_attn patch |
| `scripts/debug_proseco_owt_load.py` | NEW — CPU preflight |
| `scripts/run_phase1_proseco_owt.py` | NEW — Phase 1 runner (to be written or adapt run_phase1_proseco.py) |
| `hpc/phase1_proseco_owt.sbatch` | NEW — sbatch wrapper |
| `external/proseco/models/hf/modeling_proseco.py` | PATCHED — flash_attn optional |
| `docs/thesis/experiments/PROSECO_EXPERIMENT_OPTIONS.md` | NEW — options audit |
| `docs/thesis/experiments/PROSECO_PHASE1_PLAN.md` | THIS FILE |
| `docs/thesis/experiments/PROSECO_PREFLIGHT_STATUS.md` | UPDATE after staging |

---

## 7. Success criteria (§7.2 — analogous to §7.1 for MDLM-conf)

| Criterion | Threshold | Notes |
|-----------|-----------|-------|
| C1 | n_positive_delta_steps > 10 and peak_mean_delta ≥ 0.05 | Stronger threshold than MDLM-conf: co-trained backbone should produce more positive steps |
| C2 | ≥1 signal with \|Spearman_mean\| ≥ 0.10, spearman_std < 0.6 | Looser than §7.1 (0.15) given prior evidence signal quality is hard |
| C3 | eps_rms ≤ 0.5 for all three signals | Same as §7.1 |
| C4 | eta_95 > 0 for all B and gamma_95 > 0 | Same as §7.1 |
| C5 | Theorem A ratio ≤ 5× for at least one B | Tighter than §7.1 — proseco-owt ε should be smaller |
| C6 | ≥1 signal-based top_B policy beats uniform at B=8 or B=16 | Same as §7.1 |

If C1 fails: diagnose model loading / corrector implementation.
If C1 passes but C2 fails: report weak calibration; still write ch7 around C6.
If C2 and C6 both pass: strong positive result; proceed to full run and theory update.

---

## 8. Why this plan is the right one

1. **ProSeCo-OWT guarantees structural non-degeneracy.** The backbone was
   trained with the corrector loss, ensuring Δ_t ≠ 0 by design.

2. **Fixed corrector_steps=1 is the right first experiment.** It cleanly
   measures the value of ONE corrector application, which is exactly what
   Protocol A is designed to calibrate.

3. **Same Protocol A/B infrastructure.** No code changes to the scheduling
   framework — only a new backend.  Results are directly comparable to the
   MDLM-conf pilot.

4. **Does not pre-commit to larger compute.** Pilot first; full run only if
   pilot is meaningful.  Budget is not wasted on confirmed-degenerate configs.

5. **Directly answers the thesis question.** If the ProSeCo-OWT pilot shows
   non-trivial Spearman or strong policy improvement, that is empirical
   evidence that signal-adaptive corrector scheduling works on a real,
   production-quality masked diffusion model.

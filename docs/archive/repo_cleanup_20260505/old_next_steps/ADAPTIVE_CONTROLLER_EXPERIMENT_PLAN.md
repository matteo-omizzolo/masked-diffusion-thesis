> ARCHIVED — historical only.
> Do not use this file to infer the current thesis status unless explicitly asked.
> Current source of truth: see `START_HERE.md` and `docs/README.md`.

---


> **STATUS:** EXPERIMENT PLAN (binding spec for Protocol C on OWT)
> **LAST VERIFIED:** 2026-04-25
> **SCOPE:** Pre-registered specification of Protocol C re-targeted to OWT, per
> the activation audit verdict in
> `docs/thesis/next_steps/ADAPTIVE_CONTROLLER_ACTIVATION_AUDIT.md`. CPU-only,
> ≤ 1 day, reuses existing artefacts. Pre-registered decision rule produces a
> binary verdict.

---

# Adaptive Controller Experiment Plan — Protocol C on OWT

## 1. Research question

> Given (a) Phase 1 OWT Protocol A trajectories with per-step Δ_t and
> per-step signals (50 seeds × T=64), (b) Phase 2b OWT MC oracle headroom
> Δ_open ≈ +0.45 paired at B ∈ {2, 3, 4}, and (c) Refinement A′'s
> measured σ_ξ at B ∈ {2, 3, 4} bounding the additive surrogate slack,
>
> does the threshold-λ policy on bucketed state z = (s_t, phase(t)) achieve
> ε̃ / ε ≤ 0.7 (state conditioning materially shrinks calibration error)
> AND Δ_close_A / Δ_open ≥ 0.5 (state-conditional ranker recovers a
> measurable fraction of MC-oracle headroom on the additive surrogate, after
> subtracting the σ_ξ · √B / √2 uncertainty band)?

Pre-registered binary verdict per the table in
`ADAPTIVE_BUDGETED_CONTROLLERS.md` §4.2.

## 2. Exact policy class

The policy class is the abstract score-policy class of Theorem A-ad
(`ADAPTIVE_BUDGETED_CONTROLLERS.md` §2.1, formal restatement). Protocol C
runs **two** members of that class:

- **Threshold-λ.** σ_λ(z_{1:T}) := {t : ψ̃(z_t) > λ ∧ b_t > 0}, with λ
  tuned so 𝔼_seed[|σ_λ|] = B. The B-cap (keep only the B largest ψ̃ if
  |σ_λ| > B) prevents budget overrun on individual seeds.
- **Top-B-bucketed.** σ_topB(z_{1:T}) := argmax_{|S|=B} Σ_{t∈S} ψ̃(z_t).
  This is the open-loop top-B-by-bucket-score. It serves as the "in-class
  baseline" — the same scoring function ψ̃, applied with the open-loop
  selection rule.

Comparing σ_λ to σ_topB isolates the *threshold-vs-top-B* difference;
comparing both to Phase 2b's `entropy_top_B_per_trajectory` (which uses
ψ(s_t) directly, not ψ̃(z_t)) isolates the *bucketing-vs-raw-signal*
effect.

## 3. Artefacts reused

- `results/phase1_proseco_owt_full/protocol_a/trajectory_*.json` — 50 OWT
  Phase 1 seeds. Each file: `{seed, T=64, per_t: [{t, delta, tcr,
  f_base, f_branch, n_changed, entropy, inverse_margin,
  quality_mass_proxy, unmasked_fraction, n_revisable, n_masked}, …]}`.
  Confirmed structure on 2026-04-25.
- `results/phase2b_proseco_owt/per_seed/policy_rows_seed*.json` — 30 OWT
  Phase 2b seeds, paired policy G/A/residual.
- `results/phase2b_proseco_owt/per_seed/mc_rows_seed*.json` — 30 OWT
  Phase 2b seeds × 100 MC rows × B ∈ {2, 3, 4} = 9 000 G/A/residual
  schedules.
- `results/phase2b/mc_oracle.json` — paired Δ_open per seed per B.
- `results/phase2b/theorem_a_constants.json` — σ_ξ pooled at B ∈ {2, 3,
  4} = 0.174 / 0.240 / 0.309. ρ pooled at same B = 0.601 / 0.542 / 0.462.
  σ_Δ at same B (TBD from same file; load at run time).
- `results/phase2b/policy_comparison_paired.json` — paired uniform-vs-X
  G differences for cross-validation of the additive surrogate.

## 4. Code to be written

- `src/mdm_playground/analysis/protocol_c.py` — module:
  - `bucket_state(s_value, t, T, signal_thresholds) -> (signal_bucket,
    phase_bucket)`
  - `compute_signal_thresholds(per_step_signals, n_signal_bins=4) ->
    list[float]`
  - `compute_psi_tilde_bucket(trajectories, signal_kind, n_signal_bins=4,
    n_phase_bins=3) -> dict[(int, int), float]`
  - `compute_eps_linear(trajectories, signal_kind) -> float` (least-
    squares ε)
  - `compute_eps_tilde(trajectories, signal_kind, psi_tilde_bucket) ->
    float`
  - `tune_lambda(trajectories, psi_tilde_bucket, B, signal_kind) ->
    float`
  - `apply_threshold_policy(trajectory, psi_tilde_bucket, lam, B,
    signal_kind) -> set[int]`
  - `apply_topB_bucket_policy(trajectory, psi_tilde_bucket, B,
    signal_kind) -> set[int]`
  - `compute_additive_surrogate(trajectory, schedule) -> float` (Σ Δ_t
    over schedule)
  - `compute_uniform_schedule(T, B) -> set[int]`
  - `compute_hamming(schedule_a, schedule_b, T) -> int`
  - `protocol_c_pipeline(phase1_dir, phase2b_dir, B_values,
    signal_kinds, output_dir) -> dict` (the full pipeline, returning
    the JSON output structure)

- `scripts/run_protocol_c_owt.py` — entry script that calls the
  pipeline with paths and writes JSON outputs.

- `tests/test_protocol_c.py` — unit tests for the bucketing, threshold
  tuning, additive surrogate, and pipeline-level reproducibility.

## 5. GPU/HPC required?

**No.** The full pipeline is ≤ 100 MB of input JSON, all computations are
NumPy / Python on existing data, and the runtime is dominated by JSON
parsing. CPU laptop, ≤ 30 seconds wall time.

## 6. Success criterion

A single JSON output at `results/protocol_c_owt/protocol_c_summary.json`
with the following structure:

```json
{
  "meta": {
    "protocol": "C",
    "backbone": "ProSeCo-OWT",
    "n_phase1_seeds": 50,
    "n_phase2b_seeds_for_sigma_xi": 30,
    "T": 64,
    "B_values": [2, 3, 4],
    "signal_kinds": ["entropy", "inverse_margin", "quality_mass_proxy"],
    "n_signal_bins": 4,
    "n_phase_bins": 3
  },
  "data": {
    "psi_tilde_bucket": { "<signal>": { "<bucket_key>": <float> } },
    "eps": { "<signal>": <float> },
    "eps_tilde": { "<signal>": <float> },
    "eps_ratio": { "<signal>": <float> },
    "delta_open_per_B": { "2": 0.45, "3": 0.45, "4": 0.45 },
    "sigma_xi_per_B": { "2": 0.174, "3": 0.240, "4": 0.309 },
    "uncertainty_band_per_B": { "2": <float>, "3": <float>, "4": <float> },
    "lambda_per_signal_per_B": { "<signal>": { "<B>": <float> } },
    "delta_close_threshold_per_signal_per_B": { ... },
    "delta_close_topB_per_signal_per_B": { ... },
    "delta_close_ratio_threshold_per_signal_per_B": { ... },
    "delta_close_ratio_topB_per_signal_per_B": { ... },
    "hamming_diagnostics_per_signal_per_B": { ... }
  },
  "verdict": {
    "best_signal": "<entropy|inverse_margin|quality_mass_proxy>",
    "best_B": 2,
    "eps_ratio_at_best": <float>,
    "delta_close_ratio_at_best_after_uncertainty": <float>,
    "outcome_class": "preliminary_positive | honest_negative | inconclusive"
  }
}
```

The `outcome_class` field fires the pre-registered decision rule.

## 7. What a null result teaches

A null result (outcome_class = "honest_negative") teaches:

1. **The ranker-class corollary tightens.** The current statement bounds
   any ψ that is a separable per-step score; a null Protocol C result
   shows the bound also holds for ψ̃(z) with z = (s_t, phase(t)) bucketing
   on OWT. This is informative — it strengthens the corollary and shows
   that the recoverable structure does not factor through *any* per-step
   score (signal-only or state-conditional).

2. **The thesis chapter ordering tightens.** A null result removes any
   ambiguity about whether to position adaptive control as Future Work; it
   becomes "Future Work, conditional on a richer state abstraction"
   instead of "Future Work, conditional on running Protocol C."

3. **A-ad becomes existence-only in the appendix.** The conditional
   theorem (under stated assumptions) is still valid; what is absent is a
   non-vacuity demonstration on this dataset.

A null result is **scientifically valuable** and explicitly does not
require any change to the OWT mainline framing.

## 8. What would count as overreach

The following are **not** authorised by this plan even on a positive
outcome:

- Running Protocol C on a different backbone (cross-backbone is
  out-of-scope per the activation audit).
- Expanding state z beyond (s_t, phase(t)).
- Introducing a function approximator for ψ̃.
- Implementing F3 (Particle Gibbs / cSMC).
- Promoting A-ad from Appendix F to main body.
- Running new GPU/HPC jobs on any backbone.
- Reopening LLaDA-SFT Phase 3a.
- Modifying the OWT Phase 2b / Phase 3a mainline framing.

Any of these would violate the "bounded pilot" framing and revert the
audit verdict to "future work only."

## 9. Sequencing within this experiment plan

1. **Implement.** Write `src/mdm_playground/analysis/protocol_c.py` with
   the module spec in §4. Write `scripts/run_protocol_c_owt.py`. Write
   `tests/test_protocol_c.py`.
2. **Verify.** Run tests; confirm pipeline reproducibility on a fixed seed.
3. **Run.** Execute `scripts/run_protocol_c_owt.py` with default args;
   write `results/protocol_c_owt/protocol_c_summary.json`.
4. **Inspect.** Read the verdict JSON field `outcome_class`. Apply the
   pre-registered decision rule.
5. **Decide.** Write `POST_ADAPTIVE_CONTROLLER_DECISION.md` with the
   binary verdict and the next-step shape consistent with the outcome.
6. **Update.** Conservative narrow updates to `THEORY_STATUS.md`,
   `ADAPTIVE_BUDGETED_CONTROLLERS.md`, and (if positive) `CURRENT_INDEX.md`.

## 10. Total budget

≤ 2 days end-to-end (audit + theory tightening already complete; remaining
is implementation + run + decide + doc updates).

---

*End of experiment plan. Pre-registered, decision-binary, CPU-only,
non-overreach. Implementation starts immediately.*

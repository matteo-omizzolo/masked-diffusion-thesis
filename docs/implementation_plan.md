# Implementation Plan: Corrector Scheduling Experiments

**Updated:** April 2026 (restructured after GPT Pro v2 assessment)

> Restructured to center Phase 1 on the **one-loop marginal-gain experiment**
> that empirically estimates ε (calibration) and η_B (additivity slack)
> from Theorem A. The earlier schedule-comparison sweeps become Phase 2
> once Theorem A's inputs are measured.

For the experiment design in detail see
`docs/experiments/entropy_proxy_experiment.md`. For the current status of the
scheduling package see `docs/experiments/implementation_status.md`. For the
theorem this implementation plan serves see
`research/candidate_theorems.md` (Theorem A).

---

## Phase 1 — One-Loop Marginal Gain Experiment (Theorem A Calibration)

**Goal.** Empirically estimate ε (proxy calibration) and η_B (approximate
additivity slack) on MDLM-OWT with the ReMDM conf-refinement corrector.
These are the quantities that appear in Theorem A's 2Bε + 2η_B bound.

### Protocol A — per-step one-loop gain (ε)

For each step t ∈ {1, …, T} on a fixed trajectory:
1. Run predictor from Z_0 to Z_T without any correction → record y_base.
2. Branch at step t: apply exactly one corrector loop at step t, continue
   predictor to the end → record y_t^{+1}.
3. Compute Δ_t = F(y_t^{+1}) − F(y_base) under F = negative LM-NLL under a
   reference LM (primary) and pool-level MAUVE (secondary).
4. Record s_t (entropy, inverse margin, quality mass), TCR_t, and the random
   seed (common random numbers across branches for variance control).

**Output.** Tables of (t, Δ_t, s_t, TCR_t) over N ≥ 50 trajectories at
T = 128, with bootstrap CIs.

**Calibration analysis.**
- Spearman and Pearson correlation between ψ(s_t) and Δ_t for each signal.
- Extract ε per signal: ε = max_t |Δ_t − ψ(s_t)| and expectation version.
- Identify candidate T_low where Δ_t ≤ δ.
- Report TCR_t vs Δ_t separately — do not conflate.

### Protocol B — joint gain vs additive surrogate (η_B, γ)

For sampled schedules S with |S| = B at various B ∈ {T/16, T/8, T/4}:
1. Compute G(S) = F(y^S) − F(y_base) directly by running with the
   full S-schedule.
2. Compare with A(S) = ∑_{t ∈ S} Δ_t from Protocol A.
3. Report residual |G(S) − A(S)|; estimate η_B as the 95th-percentile
   magnitude across sampled schedules at a given B.

**Pairwise diagnostic.** For a subsampled set of pairs (t, t'),
measure ξ_{t, t'} = G({t, t'}) − Δ_t − Δ_{t'}; estimate γ = max |ξ|.
Plug into Proposition C's bound η_B ≤ γ B (B − 1) / 2.

**Budget sweep.** Repeat at multiple B to check the scaling of η_B with B.

### Infrastructure

- `external/remdm/` with MDLM-OWT checkpoint on HPC.
- `src/mdm_playground/scheduling/` package for signal computation, single-step
  gain estimation, budget allocation policies, and schedule evaluation.
  (See `docs/experiments/implementation_status.md`.)

### Estimated compute

- ~3 A100h per Protocol A trajectory × 50 trajectories × 3 seeds ≈ moderate.
- Protocol B sampled schedules: ~5 A100h for a budget sweep.
- Total Phase 1: **~8–12 A100h** on Bocconi HPC.

---

## Phase 2 — Schedule Comparison on MDLM + ReMDM

**Goal.** Given ε, η_B from Phase 1, compare signal-guided schedules to
uniform on full generation runs. Verify that top-B-by-ψ matches Theorem A's
predicted regret.

1. Keep predictor schedule fixed (MDLM default).
2. Define corrector budget B (total extra NFE for correction).
3. Compare allocations under fixed B:
   - Uniform
   - Front-loaded, back-loaded, middle-loaded
   - Entropy-proportional
   - Top-B-by-entropy
   - Top-B-by-inverse-margin
   - Burn-in-gated (exclude T_low identified in Phase 1)
4. Evaluate at B ∈ {T/8, T/4, T/2}, T ∈ {128, 256}.
5. Report: MAUVE, gen-PPL, G(S) vs Theorem A prediction.

**Diagnostic.** For each run, also report the *predicted* G from Theorem A
given the schedule's top-B-by-ψ profile and Phase 1's ε, η_B — does the
theorem's bound track empirical gaps?

**Estimated compute.** ~6 A100h.

---

## Phase 3 — Cross-Model Replication (ProSeCo)

**Goal.** Check whether ε, η_B and the best schedule transfer to a model
with native correction. Resolves Q4 (ProSeCo novelty) by reading the paper
and comparing frameworks.

1. Clone ProSeCo + download checkpoint.
2. Run Protocol A + B on ProSeCo-OWT.
3. Compare the calibration (ε) and additivity (η_B) across MDLM and ProSeCo.
4. Run Phase 2's schedule comparison on ProSeCo.

**Only proceed** if Phase 1 + Phase 2 give a positive signal on MDLM/ReMDM.

**Estimated compute.** ~6 A100h.

---

## Phase 4 — Stronger Signals (PRISM Quality Mass)

**Goal.** Test whether quality-based signals give smaller ε than entropy or
margin.

1. Add PRISM quality head or a lighter proxy quality signal.
2. Re-run Protocol A with ψ = quality mass; extract ε.
3. Compare ε across (entropy, margin, quality).
4. Re-run Phase 2 schedule comparison with the best signal.

**Infrastructure.** PRISM (`external/PRISM/`) or a lightweight proxy.
**Estimated compute.** ~4 A100h.

---

## Phase 5 — Optional Cross-Architecture Validation

**Goal.** Check whether gains are architecture-specific.

1. Port the strongest scheduler to a second model family (LLaDA inference-only,
   or SEDD with CTMC corrector).
2. Validate whether relative ordering of schedules holds.

**Only pursue** if Phases 1–4 are complete.

---

## Logging and Metrics Standard

All experiments must log:

| Metric | Description |
|--------|-------------|
| `total_predictor_nfe` | Total predictor (unmasking) function evaluations |
| `total_corrector_nfe` | Total corrector function evaluations |
| `schedule_allocation` | Vector of corrector steps per noise level: {k_t} |
| `per_step_entropy` | Aggregate entropy over revisable tokens at each step |
| `per_step_confidence_margin` | Mean (max_logit − second_max_logit) at each step |
| `per_step_quality_mass` | If available: mean quality score at each step |
| `per_step_token_change_frac` (TCR_t) | Fraction of tokens changed by each correction loop — **distinct from Δ_t** |
| `per_step_one_loop_delta` (Δ_t) | F(y_t^{+1}) − F(y_base); Protocol A output |
| `joint_gain` (G(S)) | F(y^S) − F(y_base) for schedule S; Protocol B output |
| `additive_surrogate` (A(S)) | ∑_{t ∈ S} Δ_t |
| `mauve` | MAUVE score against OWT reference |
| `gen_ppl` | Generative perplexity |
| `wall_time` | Wall-clock time for generation |

### Experiment Table Format

Each run produces a row:

```
run_id | phase | T | B | schedule | signal | F | Δ_t_table | G | A | η_est | ε_est | mauve | gen_ppl | corrector_nfe | wall_time
```

### Variance control

- Use **common random numbers** across branches (Protocol A: shared seed for
  y_base and y_t^{+1} except at step t). This dramatically reduces the
  variance of the Δ_t estimator.
- Report bootstrap 95% CIs on all ε, η_B, γ estimates.

---

## Experimental Questions Tracked

1. Under fixed extra NFE budget, does any non-uniform corrector schedule beat uniform?
2. Does the best schedule differ by total budget B?
3. Are early high-entropy regions misleading because context is incomplete?
4. Does a burn-in gate (low-gain-region exclusion) improve entropy-based scheduling?
5. Does confidence margin outperform entropy (smaller ε)?
6. Does a quality signal outperform both entropy and confidence?
7. Can the marginal value of a correction loop (Δ_t) be predicted from trajectory diagnostics?
8. **[Central]** How large is η_B across realistic B? Does Proposition C's
   η_B ≤ γ B(B−1)/2 bound hold?
9. Does a distinct low-gain region T_low exist?
10. Are TCR_t and Δ_t materially different in practice? (Q8 in `open_questions.md`.)

See `docs/experiments/entropy_proxy_experiment.md` for the full experiment
design corresponding to Phase 1.

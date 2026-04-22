> **STATUS:** ARCHIVED
> **ARCHIVED ON:** 2026-04-22
> **SUPERSEDED BY:** `docs/thesis/CANONICAL_EXPERIMENT_OVERVIEW.md`
> **REASON:** Pre-Phase-2b experiment spec variant; canonical Protocol A/B now lives in the canonical experiment overview.

---

# Entropy-Proxy Experiment — Full Design

**Created:** April 2026
**Status:** Design locked; implementation in progress.
**Provenance:** Design adapted from GPT Pro v2 assessment (items A7, A8, A9)
with novel concrete instantiations. See `docs/gpt_pro_assessment_response.md`
for item-level audit.

This document defines the Phase 1 experiment from
`docs/implementation_plan.md`. Its purpose is to empirically estimate the
quantities that appear in Theorem A's 2Bε + 2η_B regret bound (see
`research/candidate_theorems.md`).

---

## Background in one paragraph

Theorem A (main result) states that, under binary placement, approximate
additivity |G(S) − ∑ Δ_t| ≤ η_B, and calibrated proxy
|Δ_t − ψ(s_t)| ≤ ε, the top-B-by-proxy schedule Ŝ_B satisfies
G(S_B*) − G(Ŝ_B) ≤ 2 B ε + 2 η_B. The bound is useful exactly when ε and
η_B are small relative to G(Ŝ_B). This experiment measures all three
quantities on MDLM-OWT with the ReMDM conf-refinement corrector, for
signals ψ ∈ {entropy, inverse margin, quality mass}.

---

## Definitions

- **Trajectory length T.** Number of predictor steps. Primary: T = 128.
  Sensitivity: T ∈ {128, 256}.
- **Quality functional F.** Primary: negative LM-NLL under a reference LM
  (GPT-2-medium); secondary: pool-level MAUVE at the end of the trajectory.
- **Base trajectory y_base.** Generation with predictor only, no correction,
  with fixed random seed σ.
- **Branched trajectory y_t^{+1}.** Same as y_base, except at step t the
  corrector applies exactly one loop. All other randomness identical to
  y_base (common random numbers).
- **One-loop marginal gain Δ_t.** Δ_t := F(y_t^{+1}) − F(y_base).
- **Token-change rate TCR_t.** Fraction of positions changed between
  y_t^{+1} and y_base. *Not* equal to Δ_t.
- **Joint gain G(S).** For a schedule S ⊆ {1, …, T}, G(S) := F(y^S) − F(y_base)
  where y^S applies one corrector loop at each t ∈ S.
- **Additive surrogate A(S).** A(S) := ∑_{t ∈ S} Δ_t.
- **Aggregate signals s_t.**
  - Entropy: H_t := mean_i H(x_i | x_{-i}, Z_t) over revisable positions i.
  - Inverse margin: M̃_t := 1 − mean_i (p_1(i) − p_2(i)).
  - Quality mass: Q_t := mean_i (1 − q_φ(x_i | Z_t)) with q_φ from PRISM or
    a lightweight proxy (Phase 4).
- **Proxy score ψ(s_t).** Primary: identity ψ(s) = s. Calibration variants:
  rank-normalized ψ(s) = rank(s) / T, and z-scored variants.

---

## Protocol A — Per-Step One-Loop Gain

**Goal.** Estimate ε = sup_t |Δ_t − ψ(s_t)| (and its expectation version)
and identify T_low = {t : Δ_t ≤ δ} for Proposition B.

### Procedure

For each of N trajectories (primary N = 50, sensitivity N = 100):

1. Fix random seed σ. Generate y_base with predictor only.
2. For each step t ∈ {1, …, T}:
   - Reset to seed σ, run predictor up to step t.
   - Apply one corrector loop at step t.
   - Continue with predictor until Z_T, with the *same* randomness as y_base
     for all remaining steps (common random numbers).
   - Record y_t^{+1}.
   - Compute Δ_t = F(y_t^{+1}) − F(y_base).
   - Compute TCR_t = Hamming(y_t^{+1}, y_base) / D.
   - Record s_t (three signals) and any logit-level diagnostics at step t.
3. Record the full trajectory-level vector {Δ_t, TCR_t, s_t} for each t.

### Variance control

**Common random numbers (CRN) are critical.** Without CRN, the variance of
Δ_t is dominated by seed-level generation variance, and N must be
impractically large. With CRN, only the step-t corrector noise varies
between y_base and y_t^{+1}.

**Implementation note.** The predictor and corrector likely both sample
from categorical distributions. Use `torch.Generator` with a fixed seed per
trajectory and advance it deterministically; fork only at the corrector
step.

### Analysis

- **Calibration scatter.** For each signal, scatter plot ψ(s_t) vs Δ_t
  over all (trajectory, t). Report Spearman and Pearson correlation,
  bootstrap 95% CIs.
- **ε estimate.** ε_sup := max_{t, trajectories} |Δ_t − ψ(s_t)|; ε_mean :=
  E[|Δ_t − ψ(s_t)|]; ε_rms := sqrt(E[(Δ_t − ψ(s_t))²]). Report all three.
- **Low-gain region.** Plot mean_trajectories Δ_t as a function of t; identify
  T_low ≈ {t : mean Δ_t ≤ δ}. Sweep δ across plausible values.
- **TCR vs Δ.** Scatter and correlation between TCR_t and Δ_t. If correlation
  is weak, document the distinction as material.

### Outputs

- `results/protocol_a/trajectory_{i}.json` — per-trajectory full tables.
- `results/protocol_a/summary.json` — aggregate ε, T_low, correlation.
- `figures/protocol_a/` — calibration scatter, Δ_t vs t, correlation matrix.

---

## Protocol B — Joint Gain vs Additive Surrogate

**Goal.** Estimate η_B for realistic B; measure pairwise interaction γ.

### Procedure — η_B estimation

For each B ∈ {T/16, T/8, T/4} and each of M sampled schedules S of size B:

1. Draw S uniformly at random from subsets of {1, …, T} of size B.
   (Sensitivity: also test ψ-top-B and bottom-B schedules.)
2. Reset to seed σ, run the full schedule (predictor + correctors at each
   t ∈ S, with one loop each).
3. Record y^S. Compute G(S) = F(y^S) − F(y_base).
4. Look up A(S) = ∑_{t ∈ S} Δ_t from Protocol A.
5. Record residual r(S) := G(S) − A(S).

**Sample size.** Primary M = 30 per B; sensitivity M = 100.

### Analysis

- **η_B estimate.** η_B^{emp} := 95th-percentile(|r(S)|) across sampled S at
  fixed B. Report also the mean and the full distribution.
- **Scaling.** Plot η_B^{emp} as a function of B. Compare to Proposition C's
  γ B (B − 1) / 2 bound.
- **Sign structure.** Is r(S) systematically positive (super-additive) or
  negative (sub-additive)? A systematic sign suggests a specific interaction
  mechanism to investigate.

### Procedure — pairwise interaction γ

For a random subset of P ≤ T × (T − 1) / 2 pairs (t, t'):

1. Run schedules {t}, {t'}, {t, t'}. Compute Δ_t, Δ_{t'}, G({t, t'}).
2. Define ξ_{t, t'} := G({t, t'}) − Δ_t − Δ_{t'}.
3. Record |ξ_{t, t'}|.

**Sample size.** Primary P = 300 pairs across trajectories.

**Analysis.**
- Histogram of |ξ|.
- γ^{emp} := 95th-percentile (|ξ|).
- Check Proposition C prediction η_B ≤ γ B (B − 1) / 2.
- Investigate whether ξ depends on |t − t'| (temporal structure of
  interactions).

### Outputs

- `results/protocol_b/schedule_{j}.json` — per-schedule G, A, residual.
- `results/protocol_b/pairs.json` — pairwise ξ estimates.
- `results/protocol_b/summary.json` — aggregate η_B, γ by B.
- `figures/protocol_b/` — η_B vs B curve, residual sign distribution,
  |ξ| histogram.

---

## Decision Rules After Phase 1

After Protocol A + Protocol B complete, decide whether Theorem A is
empirically supported:

1. **If 2 B ε + 2 η_B < G(Ŝ_B) for B ∈ {T/8, T/4}:** Theorem A is
   non-vacuous. Proceed to Phase 2 schedule-comparison experiments.
2. **If 2 B ε dominates:** the proxy is poorly calibrated. Test alternative
   signals (margin, quality mass in Phase 4). If none are better, report as
   a negative result.
3. **If 2 η_B dominates:** strong interactions. Switch to small-B regime
   where the bound is tight, or pursue a sequential formulation (DP) as
   follow-up work.
4. **If both are small but G(Ŝ_B) ≈ 0:** corrector scheduling has no room
   to outperform uniform on this model / budget. Report as a well-defined
   negative result.

---

## Compute Budget

Bocconi HPC (`stud` partition, 4× A100 80GB):

| Item | Compute |
|------|---------|
| Protocol A: 50 trajectories × 128 branches × 3 seeds | ~6 A100h |
| Protocol B: 30 schedules × 3 B values × 3 seeds | ~3 A100h |
| Pairwise diagnostic: 300 pairs | ~3 A100h |
| Analysis and retries | ~2 A100h |
| **Total Phase 1** | **~14 A100h** |

Should fit within a few days on the `stud` partition.

---

## Reproducibility Checklist

- [ ] Seed σ logged per trajectory.
- [ ] CRN implementation verified (branch trajectories differ only at step t).
- [ ] Predictor schedule locked (MDLM default).
- [ ] Corrector kernel locked (ReMDM conf-refinement, one loop per application).
- [ ] Quality functional F documented (LM-NLL with specific reference LM).
- [ ] All Δ_t, TCR_t, s_t tables saved in structured format.
- [ ] Configs version-controlled.

---

## Relationship to Other Docs

- **Theory.** `research/candidate_theorems.md` (Theorem A, Lemma A1, A2,
  Prop B, Prop C).
- **Provenance.** `research/proof_ledger.md` for tags on each measured quantity.
- **Open questions addressed.** `research/open_questions.md` Q2, Q5, Q8, Q9.
- **Scheduling package.** `src/mdm_playground/scheduling/` (see
  `docs/experiments/implementation_status.md`).
- **HPC workflow.** `hpc/push.sh`, `hpc/submit.sh`.
- **Assessment response.** `docs/gpt_pro_assessment_response.md` items A7, A8, A9.

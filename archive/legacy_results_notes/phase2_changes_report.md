# Phase 2 Changes Report

**Date:** April 2026
**Scope:** GPT Pro v2 integration + experimental infrastructure
**Author:** Claude (Cowork session), integrating GPT Pro v2 assessment

This document summarises all changes made in the current session, spanning
two major workstreams: (A) theory restructure driven by the GPT Pro v2 audit,
and (B) experimental infrastructure built from scratch.

---

## A. Theory Restructure

### A.1 Theorem stack — from contraction to proxy-regret

**Previous state (Phase 1 session):**
- Candidate 1: binary allocation (standard, low value on its own)
- Candidate 2: factorization-error contraction **[main theorem]**
- Candidate 3: burn-in gating via MI monotonicity in u_t
- Candidate 4: confidence margin (empirical only)

**Current state (Phase 2):**
- **Theorem A** (main): proxy-regret bound G(S*_B) − G(Ŝ_B) ≤ 2Bε + 2η_B
- **Lemma A1**: oracle top-B optimality under exact additivity
- **Lemma A2**: proxy calibration → 2Bε regret in the additive regime
- **Proposition B**: low-gain-region exclusion (replaces MI-monotonicity burn-in)
- **Proposition C**: η_B ≤ γ B(B−1)/2 under pairwise interaction model
- **Stretch Appendix C2**: Candidate 2 (contraction) demoted, preserved
- **Stretch Appendix C3**: Candidate 4 (confidence margin) reframed as empirical

**Why.** GPT Pro v2 identified two hard problems with the contraction approach:
(a) Ascolani et al. 2024 treats random-scan (not systematic-scan) Gibbs, and
log-concavity does not transfer to masked text; (b) there is no mechanism linking
ρ(t) to any trajectory signal. The proxy-regret framework is strictly weaker
(requires only approximate additivity and calibration) but is empirically
testable and still provides a meaningful bound.

### A.2 Corrections recorded

| Error | Correct version |
|-------|-----------------|
| Ascolani 2024: "systematic-scan Gibbs under log-concavity" | Random-scan Gibbs; hypotheses are not classical log-concavity |
| Candidate 3: MI(x_i; x_{−i} \| Z_t) monotone in u_t | Not uniformly true; conditional distribution depends on both mask set and realized values |

Both corrections are recorded in `research/proof_ledger.md` under "Incorrect as Stated."

### A.3 New definitions (all in proof_ledger.md)

Δ_t (one-loop marginal gain), TCR_t (token-change rate, *distinct* from Δ_t),
ψ(s_t) (proxy score), ε (calibration error), η_B (additivity slack),
γ (pairwise interaction bound), T_low (low-gain region), and seven others.

### A.4 Open questions updated

- Q5 (additivity) promoted to **Central**
- Q2 (signal as proxy) promoted to **Central**
- Q6 (corrector definition) elevated to **Central** (depends on corrector kernel used)
- Q1 (geometric contraction) downgraded to **Medium** (only relevant to Stretch C2)
- Q4 (ProSeCo novelty) narrowed: thesis claims **theoretical proxy-regret framework**,
  not merely "nobody studied corrector scheduling"
- Q8, Q9 added: TCR vs Δ_t distinction; choice of quality functional F

---

## B. Experimental Infrastructure

### B.1 Scheduling package — `src/mdm_playground/scheduling/`

Four new modules implementing the Protocol A/B measurement abstractions:

| Module | Function |
|--------|----------|
| `signals.py` | `compute_signals(state, logits, meta)` — entropy, margin, quality mass |
| `gain.py` | `estimate_single_step_gain(y_base, y_branch, F)` — Δ_t and TCR_t, kept separate |
| `allocation.py` | `allocate_budget(signal_trace, budget, policy_name)` — 10 policies |
| `evaluate.py` | `evaluate_schedule(allocation, delta_trace, generator, F)` — G(S), A(S), residual |
| `surrogate.py` | `SurrogateGenerator` — analytic MDLM-mimicking surrogate for pipeline tests |

All policies available: `top_B`, `uniform`, `front`, `back`, `middle`,
`random`, `burn_in_gated`, `entropy_prop`, `margin_top_B`, `quality_top_B`.

### B.2 Experiment scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_phase1_pilot.py` | Protocol A + B driver; `--surrogate` flag for local runs |
| `scripts/analyze_phase1.py` | Reads JSON results, produces 6 publication-quality figures |

### B.3 HPC infrastructure

- `hpc/phase1_pilot.sbatch` — Slurm job script for the real MDLM Phase 1 run
- `hpc/submit.sh` updated: `bash hpc/submit.sh phase1-pilot` to submit from laptop

### B.4 Surrogate pilot run (local, no GPU)

Run completed: T=64, N=40, M=25 schedules/B, P=200 pairs, seed=42.

**Key surrogate results:**

| Quantity | Value | Implication |
|----------|-------|-------------|
| Spearman(H_t, Δ_t) | −0.22 ± 0.05 | Entropy inversely tracks gain in surrogate |
| ε_rms (entropy) | 0.0167 | Calibration poor when proxy misaligned |
| T_low | steps 0–9 | Low-gain region exists; Proposition B applicable |
| η_B (B=8, 95th pct) | 0.079 | Non-negligible interactions |
| γ (95th pct) | 0.016 | Proposition C bound: η_B ≤ 0.016×B(B−1)/2 |
| Theorem A vacuous? | Yes at all B | G(Ŝ_B) << 2Bε + 2η_B in surrogate |

The bound is vacuous in the surrogate because entropy is a misaligned proxy
(by design). This cleanly illustrates the pathology Theorem A captures: when
the signal is wrong, the proxy schedule picks the wrong steps, G(Ŝ_B) is
small, and the bound is large. The real MDLM run will determine whether
any signal produces a non-vacuous bound.

**Figures produced** (in `figures/phase1_pilot/`):
- `calibration_scatter.png` — weak scatter for all 3 signals vs Δ_t
- `delta_vs_t.png` — bell-shaped Δ_t, high-entropy early = low gain (Q8 visible)
- `eta_vs_B.png` — η_B grows with B; Proposition C bound overlaid
- `pairwise_xi_hist.png` — light-tailed |ξ| distribution; γ̂ = 0.016
- `theorem_A_budget.png` — bound vs G(Ŝ_B) bar chart; all vacuous in surrogate
- `tcr_vs_delta.png` — TCR ≠ Δ_t scatter; Q8 confirmed (Pearson ≈ 0.18)

### B.5 Documentation files

| File | Purpose |
|------|---------|
| `docs/experiments/entropy_proxy_experiment.md` | Full Protocol A + B design spec |
| `docs/experiments/implementation_status.md` | Package status and HPC workflow |
| `docs/experiments/phase1_interpretation.md` | This session's interpretation doc |
| `docs/gpt_pro_assessment_response.md` | Item-level audit of GPT Pro v2 output |

---

## C. Documentation Updates

| File | Change summary |
|------|---------------|
| `research/candidate_theorems.md` | Full rewrite: Theorem A + Lemmas A1/A2 + Props B/C + stretch C2/C3; deprecation trail |
| `research/proof_ledger.md` | Expanded tag system; Ascolani + MI monotonicity corrections; new definitions table |
| `research/open_questions.md` | Q5/Q2/Q6 promoted to Central; Q1 downgraded; Q4 narrowed; Q8/Q9 added |
| `research/proof_worklog.md` | Entry 6 added: Theorem A proof sketch, correction history, new next steps |
| `docs/thesis_direction.md` | Theory Target section replaced with Theorem A + supporting results; experimental questions 8–10 added |
| `docs/implementation_plan.md` | Restructured around Phase 1 one-loop experiment as primary; Phases 2–5 reordered |

---

## D. To Do Before Supervisor Meeting

1. **Run Phase 1 on HPC.** From your laptop:
   ```bash
   bash hpc/push.sh && bash hpc/submit.sh phase1-pilot
   ```
   Replace `[SURROGATE]` values in `docs/experiments/phase1_interpretation.md`
   with real numbers.

2. **Read ProSeCo.** Resolve Q4 (novelty claim). Update `research/open_questions.md`.

3. **Read Ascolani et al. 2024.** Confirm random-scan scan type; assess whether
   any contraction framework applies to masked diffusion correctors. Update
   Stretch C2 status.

4. **Formalize Theorem A proof.** Entry 6 in `proof_worklog.md` has a first pass;
   write it cleanly with explicit exchange-argument accounting and publish to
   `thesis/ch4_theory.tex` (once ch3 is drafted).

5. **Draft ch3 (Discrete Diffusion).** Required before ch4 can be written. See
   `docs/md/research_plan.md` for chapter dependencies.

---

## E. File Index (new files this session)

```
src/mdm_playground/scheduling/
  __init__.py          — package exports
  signals.py           — compute_signals()
  gain.py              — estimate_single_step_gain()
  allocation.py        — allocate_budget(), ALLOCATION_POLICIES
  evaluate.py          — evaluate_schedule()
  surrogate.py         — SurrogateGenerator

scripts/
  run_phase1_pilot.py  — Protocol A + B experiment driver
  analyze_phase1.py    — Figure generation (6 plots)

hpc/
  phase1_pilot.sbatch  — Slurm job for real MDLM Phase 1

docs/experiments/
  entropy_proxy_experiment.md   — Protocol A + B design spec
  implementation_status.md      — Package status
  phase1_interpretation.md      — This session's interpretation

docs/
  gpt_pro_assessment_response.md — Item-level GPT Pro v2 audit

research/  (updated, not new)
  candidate_theorems.md
  proof_ledger.md
  open_questions.md
  proof_worklog.md

docs/ (updated, not new)
  thesis_direction.md
  implementation_plan.md
  phase2_changes_report.md (this file)

figures/phase1_pilot/
  calibration_scatter.png
  delta_vs_t.png
  eta_vs_B.png
  pairwise_xi_hist.png
  theorem_A_budget.png
  tcr_vs_delta.png

results/phase1_pilot/
  protocol_a/trajectory_{0..39}.json
  protocol_b/schedule_{0..74}.json
  protocol_b/pairs.json
  summary.json
```

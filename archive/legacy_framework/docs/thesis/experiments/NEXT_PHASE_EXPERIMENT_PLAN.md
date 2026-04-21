# Next-Phase Experiment Plan — Phase 2 (Post-Audit)

**Author:** Claude Code (co-advisor review, Workstream C)
**Date:** 2026-04-19
**Status:** **DRAFT — requires user approval before any HPC submission.**
**Informed by:**
`docs/thesis/experiments/EXPERIMENT_CRITICAL_AUDIT.md` (Workstream A)
`docs/thesis/theory/THEORY_STRESS_TEST.md` (Workstream B)

---

## 0. Guiding principle

Phase 1 produced a Δ_t calibration and an additivity slack estimate on
ProSeCo-OWT, but the policy-comparison evidence we reported is statistically
unsafe (single-seed, per §2 of the audit) and methodologically misleading (A(S)
vs G(S) confusion, per §1). Before doing anything new — new backbone, new
corrector, new F — we must fix Phase 1 on its own terms.

Phase 2 has **three gated stages**, each producing evidence that decides
whether the next stage runs. No stage fires without user approval and a clear
budget estimate.

**Budget discipline.** The user has a Bocconi HPC allocation and wants
statistically careful evidence over ambitious noisy claims. Every stage below
is costed in GPU-hours with a cap. If a stage exceeds its cap, the experiment
stops and we re-plan rather than expand.

---

## 1. Guiding questions Phase 2 must answer

Derived from audit §§1–7 and theory stress-test §§3–8:

| Q# | Question | Phase 2 stage |
|----|----------|---------------|
| P1 | Does A(S) rank schedules in the same order as G(S) at B∈{4,8,16}? | 2a (offline) |
| P2 | What is the per-seed SE of each (policy, B) true G? | 2b (HPC) |
| P3 | Does any per-trajectory (not mean-profile) signal-adaptive policy beat uniform on true G? | 2b (HPC) |
| P4 | Does a Monte-Carlo "true" oracle beat the mean-field oracle? And is uniform near-optimal? | 2b (HPC) |
| P5 | Is the entropy-direction inversion (ρ ≈ −0.19) a ProSeCo-OWT artifact or a robust structural finding? | 2b (requires per-trajectory Δ ranking) |
| P6 | Does rank-based ε_R behave differently from ε_rms? | 2a offline, tied to P1 |
| P7 | Is there a small-B regime (B=2 or B=3) where Theorem A is non-vacuous? | 2b, include small B |
| P8 | Would a less noisy F (MAUVE) change the story? | 2c (HPC stretch) |

Questions explicitly **out of scope** for Phase 2 (park to later phase):

- Cross-backbone replication (MDLM-conf, ReMDM-conf): only touch if Phase 2a/b
  give a positive signal worth replicating.
- Stretch C2 contraction route: independent theory task, not a Phase 2 experiment.

---

## 2. Stage 2a — offline re-analysis (no HPC)

**Goal.** Squeeze every remaining bit of evidence out of the existing
`results/phase1_proseco_owt_full/` data before spending HPC.

**Compute.** Local laptop, minutes. No new predictor runs.

### 2a.1 A↔G rank correlation on existing 30 schedules

- We have 30 schedules × 3 B-values = 90 (S, G(S), A(S)) records (under
  `results/phase1_proseco_owt_full/protocol_b/` schedules). Need to verify
  they survived; if the sbatch didn't persist them, re-run with surrogate to
  confirm the code path.
- Compute Spearman ρ(A_j, G_j) within each B, with 95% bootstrap CI (1000
  resamples).
- Emit: `results/phase2a/A_vs_G_rank_correlation.json`.
- **Decision gate:** If ρ(A, G) at B=8 is ≥ 0.5, the additive surrogate is
  usable for top-B selection; keep using it and investigate why mean-profile
  policies underperformed. If ρ < 0.5 (expected), the surrogate cannot be the
  thesis's workhorse object and Phase 2b must evaluate policies end-to-end on
  true G.

### 2a.2 Rank-based calibration ε_R

- For each trajectory (50 of them), compute within-trajectory Spearman(ψ, Δ)
  for each signal. Already partially available as `spearman.{signal}` mean ± std;
  extend to the full per-trajectory distribution with `quantile` at 25/50/75.
- Compute ε_R := (1 − |ρ|) · σ_Δ per trajectory; aggregate.
- Emit: `results/phase2a/eps_R.json`.

### 2a.3 Per-trajectory sign count

- For each trajectory, record sign(Spearman(entropy_t, Δ_t)): how many of 50 are
  negative, positive, ≈0?
- This tests whether "inverted direction" is universal or population-averaged.
- Emit into `results/phase2a/eps_R.json`.

### 2a.4 Honest-oracle fallback check

- For each trajectory, among the 30 sampled schedules at B=8, find the one
  with max G(S). Report the distribution of (max_schedule_G − uniform_G) per
  trajectory.
- This is a **lower bound** on G(S_B*), computed from existing data at zero HPC
  cost.
- If even this max-over-30 lower bound fails to beat uniform per trajectory,
  the thesis has a strong negative result right now.
- Emit: `results/phase2a/mc_oracle_lowerbound.json`.

### 2a.5 Corrected policy_comparison table

- Recompute `A(S)` and `G(S)` columns from the existing policy_comparison
  records and produce a single honest table, with the A(S) column explicitly
  labeled and the G(S) column flagged as single-seed.
- Emit: `docs/thesis/experiments/results/phase2a_corrected_policy_table.md`
  (a markdown table, not a claim).

### Gate for Stage 2a

Stage 2a produces decisions only; no thesis claims. After 2a the user and I
should reconvene to decide Stage 2b parameters (K seeds, which B values, which
per-trajectory policies to evaluate).

---

## 3. Stage 2b — paired-seed policy evaluation on ProSeCo-OWT (HPC)

**Goal.** Give every (policy, B) a G-with-SE measurement and a paired comparison
to uniform. Answer P2, P3, P4, P5, P7.

**Compute budget cap:** 2 GPU-days (≈ 48 A100-hours). If the stage exceeds
this cap, abort and re-plan.

### 3.1 Experimental design

- **Base seeds K = 30** (paired across all policies; same 30 base trajectories
  evaluated for every policy). K=30 gives SE ≈ σ/√30 ≈ σ/5.5; with observed
  σ(F) ~ 1, SE per policy ≈ 0.18. A policy surplus of 0.2 would be ≈ 1σ — too
  noisy. At K=30 we can robustly detect surpluses ≥ 0.4.
- **B values = {2, 3, 4, 8, 16}** (adds B=2 and B=3 to Phase 1's {4,8,16},
  per P7).
- **Policies:**
  - `uniform` (baseline)
  - `front`, `back`, `middle` (positional controls)
  - **Per-trajectory** `entropy_top_B_pt`, `entropy_bot_B_pt` (new — top-B /
    bottom-B of the *per-trajectory* entropy profile, not the mean)
  - **Per-trajectory** `margin_top_B_pt`, `margin_bot_B_pt`
  - **Per-trajectory** `quality_top_B_pt`, `quality_bot_B_pt`
  - **Monte-Carlo oracle** `mc_oracle_B` (for B ∈ {2,3,4} only): per
    trajectory, sample 300 random schedules of size B, take argmax G(S).
    This gives an empirical lower bound on G(S_B*).
  - `mean_profile_oracle` (top-B of mean Δ̄_t) — kept as Phase 1 comparator.

  Note: old `entropy_top_B`, `entropy_burn_in_gated`, `entropy_bot_B` (mean-profile
  variants) collapse to positional policies per audit §3; skip them as they
  duplicate `front`/`back`.

- **Total evaluations:** K_base × policies × B_values
  - Policies with no per-B oracle overhead: 10 policies × K=30 × 5 B = 1500
    `evaluate_schedule` calls
  - MC oracle: K=30 × 300 schedules × 3 B (2,3,4) = 27,000 additional calls
  - Grand total ≈ 28,500 evaluate_schedule calls.
- **Estimated time per evaluate_schedule call:** ~1.5 s on A100 (2 full
  predictor passes + B corrector loops; B ≤ 16 so ≤ 2.5 s upper).
- **Total GPU-hours:** 28,500 × 2 s / 3600 ≈ 16 A100-hours. Well within the
  2-day cap with room for re-runs.

### 3.2 Pipeline changes required

All changes are additive; no replacement of existing code.

1. `src/mdm_playground/scheduling/allocation.py`
   - Add `top_B_per_trajectory`: takes `per_t_signals: List[float]` (the
     trajectory-specific signal trace) and returns top-B indices. This is
     equivalent to calling `top_B(signal, B)` with the per-trajectory
     signal instead of the mean — just needs a hook so
     `run_policy_comparison` can feed per-trajectory traces.
   - Add `bottom_B_per_trajectory`: analogous.

2. `src/mdm_playground/scheduling/evaluate.py`
   - No change; `evaluate_schedule` already computes true G(S).

3. `scripts/run_phase2b_proseco_owt.py` (new)
   - Similar structure to `run_phase1_proseco_owt.py::run_policy_comparison`,
     but with K base seeds per (policy, B) and per-trajectory traces.
   - Emit paired G_list[k] for each (policy, B) so downstream can compute
     paired t-test vs uniform.
   - Emit MC-oracle schedule and G per trajectory.

4. `scripts/analyze_phase2b.py` (new)
   - Read paired G_list, compute G_mean ± SE, paired bootstrap 95% CI for
     (policy − uniform) differences.
   - Rank A(S) vs G(S) correlation on all 30 random schedules.
   - Emit `results/phase2b/policy_comparison_paired.json` and
     `figures/phase2b/policy_gains.png`.

### 3.3 HPC sbatch file

`hpc/phase2b_proseco_owt.sbatch` (new):

```bash
#SBATCH --job-name=phase2b_proseco_owt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00   # cap below the 48h cluster limit
# body: same conda activation as phase1_proseco_owt_full.sbatch
srun python -u scripts/run_phase2b_proseco_owt.py \
    --T 64 \
    --K 30 \
    --B_values 2,3,4,8,16 \
    --mc_oracle_samples 300 \
    --mc_oracle_B 2,3,4 \
    --seed 42 \
    --checkpoint ~/mdm/checkpoints/proseco_owt \
    --out_dir results/phase2b_proseco_owt
```

### 3.4 Deliverables

- `results/phase2b_proseco_owt/policy_comparison_paired.json` — G_mean ± SE
  for each (policy, B), paired with uniform.
- `results/phase2b_proseco_owt/mc_oracle.json` — per-trajectory argmax G over
  300 random schedules at B ∈ {2,3,4}.
- `results/phase2b_proseco_owt/rank_A_vs_G.json` — Spearman ρ(A, G) over
  random schedules.
- `figures/phase2b/policy_gains.png` — box-whisker of G per policy per B with
  paired uniform comparison.

### 3.5 Stage 2b decision gates

After Stage 2b completes, the Phase 2 decision matrix is:

| Observation | Decision |
|-------------|----------|
| Any per-trajectory signal policy beats uniform with ≥ 2σ at B=8 | Keep signal. Proceed to 2c (F swap). |
| All per-trajectory signal policies within 1σ of uniform, but MC oracle is ≥ 0.3 above uniform | Headroom exists but our signals don't exploit it — proceed to 2c with focus on F or a new signal. |
| MC oracle within 0.1 of uniform at B=8 | Uniform is (empirically) near-optimal on ProSeCo-OWT + GPT-2 NLL — this is a **thesis-level negative result**. Skip 2c, write up. |
| Small-B regime (B=2, B=3) shows any signal policy beats uniform at ≥ 2σ | Re-focus thesis on small-B corrector scheduling and write up non-vacuous Theorem A bound there. |

---

## 4. Stage 2c — F substitution (MAUVE vs owt-reference) [stretch, gated]

**Goal.** Test whether the F=GPT-2 NLL quality functional is the bottleneck.
Swap to MAUVE vs an OWT reference and rerun the same paired evaluation.

**Gate.** Fire only if Stage 2b produces a signal worth replicating (row 1 or
row 2 of 3.5). If 2b says "uniform is optimal" (row 3), 2c is a waste of
compute.

**Compute budget cap:** 1 GPU-day.

### 4.1 Infrastructure

- MAUVE reference: `/home/3316152/mdm/data/owt_reference_1000.json` (already
  staged per CLAUDE.md §OpenWebText).
- `mauve-text` Python package is in requirements.txt.
- `src/mdm_playground/scheduling/evaluate.py::evaluate_schedule` currently
  supports `F="neg_nll"`. Needs additional `F="mauve_owt"` branch that
  computes MAUVE between the generated batch and the reference.

### 4.2 Design

- Replay the K=30 base seeds × policies × B from Stage 2b **with MAUVE** as F.
- **No new predictor trajectories** — reuse Stage 2b trajectories if we cached
  them (requires 2b to save y_base and y^S arrays). Otherwise re-run.

### 4.3 Deliverable

- `results/phase2c_proseco_owt_mauve/policy_comparison_paired.json`
- `figures/phase2c/policy_gains_mauve.png`
- Side-by-side comparison: same policy ranking under NLL vs MAUVE?

---

## 5. Stage 2d — second backbone/corrector pair [deferred, not in Phase 2 budget]

**Not part of this plan.** Moves to Phase 3 after Phase 2 is written up.

Candidates (pre-agreed in `CANONICAL_RESEARCH_DIRECTION.md`):

- MDLM backbone + MDLM-conf partial-resample corrector (`backends/mdlm_conf.py`,
  existing).
- ReMDM-conf (pending checkpoint availability).

---

## 6. Total Phase 2 HPC budget (conservative)

| Stage | GPU-hours | Wall clock | Job type |
|-------|-----------|------------|----------|
| 2a    | 0         | hours      | Local    |
| 2b    | ≤ 24      | 1 day      | 1× A100  |
| 2c    | ≤ 12      | half day   | 1× A100  |
| **Total** | **≤ 36** | 1.5 days | 2 jobs |

Well under a week of Bocconi HPC wall clock; comfortable within the stud QoS
cadence.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| MC-oracle sampling (300 schedules) misses the true S_B* by a lot at B=4 | Report MC-oracle as "empirical lower bound on G(S_B*)"; for B=4, also run exhaustive C(64,4)=635k enumeration on 3 trajectories as a spot check |
| K=30 base seeds is underpowered for small surplus detection | Pre-register α=0.05 with Bonferroni across {2,3,4,8,16}; smaller effects we treat as "not resolved" rather than "null" |
| `evaluate_schedule` with per-trajectory signals requires pipeline changes that introduce bugs | Implement on surrogate (CPU) first; regression-test that mean-profile policies give the same G as Phase 1 for the matching seed; only then run on GPU |
| MAUVE reference staging breaks on HPC (already happened once per CLAUDE.md §11) | Verify reference file exists before the 2c submission via a dry-run command in the sbatch |
| ProSeCo-OWT checkpoint is not loaded identically to Phase 1 | Pin checkpoint by file hash in `run_phase2b_proseco_owt.py` startup assertion |

---

## 8. Pre-flight checklist (must pass before 2b submission)

- [ ] Stage 2a complete; A↔G rank correlation and MC-oracle lower bound results
      reviewed with user.
- [ ] `src/mdm_playground/scheduling/allocation.py` has `top_B_per_trajectory`
      and `bottom_B_per_trajectory` tested on surrogate.
- [ ] `scripts/run_phase2b_proseco_owt.py` passes `--surrogate` smoke test
      locally (K=5, B=4, 1 policy).
- [ ] `hpc/phase2b_proseco_owt.sbatch` dry-run (`-t 00:10:00`) passes
      checkpoint verification and imports.
- [ ] User has approved the policy list in §3.1 (in particular, whether to
      include `mc_oracle_B=4` — adds ~10 GPU-hours on its own).
- [ ] User has approved the 30-seed K. If 50 seeds are preferred the budget
      scales ×1.67.

---

## 9. What I am **not** doing in Phase 2

Explicit non-goals, to keep scope honest:

- **No new backbone.** One backbone (ProSeCo-OWT) throughout Phase 2; cross-backbone
  is Phase 3.
- **No new corrector kernel.** ProSeCo annealed-refinement throughout.
- **No new theorem.** The variance-form η bound (§10.1 of theory stress-test)
  is a parallel theory task; Phase 2 measures data, not proofs.
- **No narrative rewrite.** `CURRENT_INDEX.md` and `RESULTS_STATUS.md` stay
  un-edited until Phase 2 is complete (Workstream H). Until then the existing
  documents remain as-is, with a prominent `[UNDER AUDIT, see
  EXPERIMENT_CRITICAL_AUDIT.md]` banner at the top.

---

## 10. Open questions for the user (required before I submit anything)

1. **K=30 base seeds vs K=50.** 30 is the minimum for CLT; 50 doubles cost.
   Which?
2. **Include B ∈ {2, 3} at MC-oracle?** Adds 50%-ish to total cost but
   directly tests the Theorem A "small-B non-vacuous" regime.
3. **MAUVE stage 2c — do or skip?** Only do if 2b shows positive signal, or
   do it anyway as a check on the F=NLL assumption?
4. **Freeze existing canonical docs or edit them now?** I propose
   freeze-with-banner until Phase 2 is done.
5. **Approval for local Stage 2a?** No HPC cost; just runs analysis scripts
   and writes JSON/markdown outputs. I propose starting Stage 2a immediately
   after approval since it has no blast radius.

Without answers to these, **no HPC submission**. Stage 2a I will start as soon
as you say "go 2a"; Stages 2b and 2c require explicit gates to fire.

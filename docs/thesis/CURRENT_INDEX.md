# CURRENT_INDEX — Thesis Navigation

**Last updated:** 2026-04-20
**Phase:** **Phase 3a COMPLETE (search-class positive on ProSeCo-OWT) → Phase 3b ACTIVE (theory finalisation).** Phase 3a job 479941 completed
2026-04-20 (verdict in `RESULTS_STATUS.md` §14: CD-G + BS-AG both PASS at
every B ∈ {2,3,4,8}; CD-G recovers 74–84 %, BS-AG 49–64 % of the Phase 2b
+0.45 MC-oracle headroom; both still PASS at B = 8 where the ranker
envelope is NULL). Thesis story tightens to **fixed-budget corrector
allocation is a combinatorial trajectory-control problem; cheap greedy
rankers are the wrong solution class for it**. Negative-Result Corollary
**rescoped to the greedy/separable-ranker class only** — search procedures
explicitly exceed the ranker envelope. PRISM rejection upheld with refined
reason (PRISM is a member of the bounded ranker class). Phase 2c F-swap and
cross-backbone replication remain closed/parked.

**Source-of-truth pointers for the Phase 3 direction:**
`docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md`,
`docs/thesis/experiments/PHASE3_DIRECTION_AUDIT.md` (with post-Phase-3a addendum),
`docs/thesis/experiments/PHASE3_ALTERNATIVE_PLAN.md`,
`docs/thesis/experiments/RESULTS_STATUS.md` §§12–14,
`docs/thesis/theory/THEORY_STATUS.md` (banner + Honesty Ledger 2026-04-20).

This is the single authoritative entry point into the thesis repo. Everything
active is reachable from here. Archived material is not listed; see
`archive/ARCHIVE_MANIFEST.md`.

---

## 1. Thesis question

For a fixed predictor schedule and fixed corrector NFE budget in masked
diffusion language models, can aggregate trajectory signals — entropy,
confidence margin, or quality mass — predict the marginal value of a
corrective refinement loop well enough to outperform uniform corrector
placement?

See `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md` for the full framing.

---

## 2. Canonical thesis documents (read in this order)

| # | Document | Purpose |
|---|---|---|
| 1 | `docs/thesis/CURRENT_INDEX.md` | This file |
| 2 | `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md` | Authoritative research direction, scope, theorem targets (rewritten 2026-04-19 post-audit) |
| 3 | `docs/thesis/CANONICAL_EXPERIMENT_OVERVIEW.md` | End-to-end explanation of the experiment (Protocol A/B) |
| 4 | `docs/thesis/experiments/EXPERIMENT_CRITICAL_AUDIT.md` | **Audit** of Phase 1 policy claims — A(S) vs G(S), single-seed inadmissibility, mean-field vs trajectory-aware policies |
| 5 | `docs/thesis/theory/THEORY_STRESS_TEST.md` | **Theory stress test** — vacuity verdict, ε-conflation diagnosis, Refinement A′/A″ candidates |
| 6 | `docs/thesis/experiments/ANALYSIS_SPEC.md` | Tier T1–T4 evidence rules, paired estimator definition, BCa CI |
| 7 | `docs/thesis/experiments/NEXT_PHASE_EXPERIMENT_PLAN.md` | Phase 2 stage spec (2a offline / 2b HPC / 2c MAUVE-swap) |
| 8 | `docs/thesis/experiments/RESULTS_STATUS.md` | Summary of all runs + Phase 2 in-flight status |
| 9 | `docs/thesis/experiments/PROSECO_ANALYSIS.md` | Phase 1 trajectory-level analysis (numerical inputs to the stress test) |
| 10 | `docs/thesis/theory/THEORY_STATUS.md` | Theorem A, Propositions B/C, Refinements A′/A″, honesty ledger |
| 11 | `docs/thesis/theory/NEXT_THEORY_STEPS.md` | Next 3 theory tasks |
| 12 | `docs/thesis/experiments/PHASE3_DIRECTION_AUDIT.md` | Audit that rejected the PRISM pivot (with post-Phase-3a addendum) |
| 13 | `docs/thesis/experiments/PHASE3_ALTERNATIVE_PLAN.md` | Plan for Phase 3a + 3b (with Phase 3a result block) |
| 14 | `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md` | **Phase 3a result report** — CD-G + BS-AG search-class positive |

---

## 3. Theory (chronological record — keep consulting these for provenance)

| Document | Purpose |
|---|---|
| `research/candidate_theorems.md` | Theorem A, Lemmas A1/A2, Propositions B/C, Stretch C2/C3 — with provenance tags |
| `research/proof_worklog.md` | Entry-by-entry derivation log (Entry 7 = latest) |
| `research/proof_ledger.md` | Provenance of every proof ingredient |
| `research/open_questions.md` | Q5 (additivity) and Q2 (calibration) are central |

---

## 4. Experiment specification and protocol references

| Document | Purpose |
|---|---|
| `docs/experiments/proseco_experiment_definition.md` | Mathematical spec of Protocol A/B on ProSeCo |
| `docs/experiments/proseco_protocol_mapping.md` | Code-level Protocol A/B mapping |
| `docs/experiments/proseco_backend_audit.md` | Backend design audit |
| `docs/experiments/proseco_import_audit.md` | Import / dependency audit |
| `docs/experiments/proseco_validation_checklist.md` | Structural validation checklist |
| `docs/experiments/entropy_proxy_experiment.md` | Protocol A/B general specification (kernel-agnostic) |
| `docs/experiments/implementation_status.md` | Scheduling package API (signals/gain/allocation/evaluate) |
| `docs/experiments/phase1_interpretation.md` | Interpretation of the MDLM diagnostic negative result |
| `docs/experiments/results/result_inventory.md` | Canonical run inventory (MDLM 478600 + ProSeCo 478929) |
| `docs/experiments/results/proseco_backend_failure_audit.md` | Root-cause audit of jobs 478826/827/828 |

---

## 5. Active scripts

| Script | Purpose |
|---|---|
| `scripts/run_phase1_mdlm_conf.py` | **PRIMARY** — MDLM-conf corrector Phase 1 runner |
| `scripts/run_phase1_proseco.py` | Alternate — ProSeCo runner (currently no-op on mdlm.ckpt) |
| `scripts/run_phase1_pilot.py` | MDLM heuristic runner (diagnostic baseline only) |
| `scripts/analyze_phase1.py` | Figure generation |
| `scripts/debug_mdlm_conf_load.py` | CPU preflight for MDLM-conf backend |
| `scripts/debug_proseco_load.py` | CPU preflight for ProSeCo backend |
| `scripts/stage_owt_reference.py` | Stream OWT samples to HPC for MAUVE reference |

---

## 6. Active HPC workflow

| File | Purpose |
|---|---|
| `hpc/phase3a_combinatorial.sbatch` | **MOST RECENT** — Phase 3a CD-G + BS-AG combinatorial baselines (job 479941, complete 2026-04-20) |
| `hpc/phase2b_proseco_owt.sbatch` | Phase 2b paired K=30 sweep (job 479581, complete 2026-04-19/20) |
| `hpc/phase1_proseco_owt_full.sbatch` | Phase 1 ProSeCo-OWT FULL RUN (job 479537, complete 2026-04-19) |
| `hpc/phase1_proseco_owt.sbatch` | Phase 1 ProSeCo-OWT pilot (job 479382, complete) |
| `hpc/phase1_mdlm_conf.sbatch` | Phase 1 MDLM-conf signal-aligned (job 479257, complete) |
| `hpc/phase1_pilot.sbatch` | DEPRECATED — MDLM heuristic (Δ_t ≤ 0 everywhere) |
| `hpc/push.sh` | rsync repo to HPC |
| `hpc/pull.sh` | rsync results back (broken on macOS; use ssh instead) |
| `hpc/setup_env.sh` | One-time conda env bootstrap |
| `hpc/submit.sh` | Legacy multi-strategy submitter (pre-Phase-1) |

HPC details:
- Host: `slogin.hpc.unibocconi.it`, user `3316152`
- Repo on HPC: `~/mdm/masked-diffusion-thesis`
- Checkpoint: `~/mdm/checkpoints/mdlm.ckpt` (~2.5 GB)
- Env: `conda activate remdm311`
- Known HPC issues: see root `CLAUDE.md`

---

## 7. Active backend code

| File | Purpose |
|---|---|
| `src/mdm_playground/scheduling/backends/proseco_owt.py` | **PRIMARY** — ProSeCo-OWT co-trained backbone, importlib loader |
| `src/mdm_playground/scheduling/backends/mdlm_conf.py` | MDLM-conf top-K resample (Spearman noise-level, do not rerun) |
| `src/mdm_playground/scheduling/backends/proseco.py` | ProSeCo on mdlm.ckpt (structural no-op, deprecated) |
| `src/mdm_playground/scheduling/backends/mdlm.py` | MDLM heuristic (all-masked resample; diagnostic only) |
| `src/mdm_playground/scheduling/signals.py` | Signal extraction (entropy, margin, quality mass) |
| `src/mdm_playground/scheduling/allocation.py` | Policies: uniform, top_B, burn_in_gated, front/back/middle |
| `src/mdm_playground/scheduling/gain.py` | `estimate_single_step_gain` |
| `src/mdm_playground/scheduling/evaluate.py` | `evaluate_schedule` |
| `src/mdm_playground/scheduling/surrogate.py` | Surrogate backend for CPU pipeline validation |

---

## 8. Main results files

| Path | Interpretation |
|---|---|
| `results/phase3a_proseco_owt/{cd,bs}_paired.json` | **Phase 3a — combinatorial baselines** (K=30 paired, T=64, job 479941, COMPLETE 2026-04-20). CD-G PASS at all B with Δ ∈ [+0.32, +0.38]; BS-AG PASS at all B with Δ ∈ [+0.15, +0.29]. Full report: `PHASE3A_COMBINATORIAL_RESULTS.md`. |
| `results/phase3a_proseco_owt/oracle_gap_closure.json` | **Phase 3a oracle-gap closure** — CD-G 78.9 / 74.1 / 84.3 % at B ∈ {2,3,4}; BS-AG 64.1 / 57.1 / 48.8 %. |
| `results/phase2b_proseco_owt/{policy_raw,mc_raw}.json` | **Phase 2b — paired K=30 sweep** (T=64, job 479581, COMPLETE 2026-04-19/20). 5 PASS / 30 FAIL / 9 BORDERLINE; MC oracle +0.45 vs uniform at B ∈ {2,3,4}; pooled ρ(A,G) decays 0.66 → 0.39. Aggregator: `scripts/analyze_phase2b.py`. |
| `results/phase2b/policy_comparison_paired.json` | **Phase 2b paired CIs** (BCa per policy/B). |
| `results/phase2b/mc_oracle.json` | **Phase 2b MC-oracle bound** (best-of-100 random schedules per seed). |
| `results/phase1_proseco_owt_full/summary.json` | **Phase 1 — ProSeCo-OWT FULL RUN** (N=50, T=64, job 479537, COMPLETE 2026-04-19 07:04). Trajectory-level (Tier T1) results valid: 61/64 positive Δ_t, peak=0.157, Spearman=−0.191. Policy-ranking rows (Tier T4) are inadmissible per audit; superseded by Phase 2b. Full analysis: `PROSECO_ANALYSIS.md`. |
| `results/phase1_proseco_owt/summary.json` | **Phase 1 — ProSeCo-OWT pilot** (N=20, T=64, job 479382, COMPLETE). 59/64 positive Δ_t, peak=0.178. |
| `results/phase1_mdlm_conf_signal_aligned/summary.json` | **MDLM-conf signal-aligned pilot** (N=20, T=64, job 479257). M2 fix applied. Verdict: signal noise-level; switched to proseco-owt. |
| `results/phase1_mdlm_conf/summary.json` | **MDLM-conf unaligned pilot** (N=20, T=64, job 478962). Superseded — do not cite. |
| `results/phase1_proseco/summary.json` | **ProSeCo pilot on mdlm.ckpt** (N=20, T=64, job 478929). All Δ_t = 0 — structural no-op. |
| `results/phase1_pilot/summary.json` | **MDLM heuristic pilot** (N=20, T=64, job 478600). All Δ_t ≤ 0 — diagnostic negative. |
| `results/phase1_{*}_surrogate_sanity/summary.json` | Surrogate sanity for the corresponding backend. |
| `results/phase1_smoke/` | Smoke-test logs (pre-Phase-1). |
| `results/full_eval/` / `results/sweep/` / `results/t1000_eval/` / `results/remdm_smoke/` | Pre-Phase-1 step-sweep results (archive candidate). |

HPC stdout/stderr for each job live at `out/{phase}_{*}_{jobid}.out` /
`err/{phase}_{*}_{jobid}.err`. Reference jobs: `479941` (Phase 3a),
`479581` (Phase 2b), `479537` (Phase 1 ProSeCo-OWT full),
`479257` (MDLM-conf), `478600` (MDLM heuristic), `478929` (ProSeCo no-op).

---

## 9. Thesis writing

| Path | Status |
|---|---|
| `thesis/main.tex` | Entry point |
| `thesis/chapters/abstract.tex` | TODO |
| `thesis/chapters/ch1_introduction.tex` | TODO |
| `thesis/chapters/ch2_background_diffusion.tex` | First draft done |
| `thesis/chapters/ch3_discrete_diffusion.tex` | TODO |
| `thesis/chapters/ch4_masked_diffusion.tex` | TODO |
| `thesis/chapters/ch5_informed_correctors.tex` | TODO |
| `thesis/chapters/ch6_contribution.tex` | TODO — Theorem A / Propositions B/C write-up |
| `thesis/chapters/ch7_experiments.tex` | TODO — Phase 1 results |
| `thesis/chapters/ch8_conclusion.tex` | TODO |

---

## 10. Archived material

Everything legacy is under `archive/`. See `archive/ARCHIVE_MANIFEST.md`
for a complete move-log. At a glance:

- `archive/legacy/` — pre-April 2026 study docs and PDFs
- `archive/legacy_docs/` — April 2026 cleanup: old CURRENT_INDEX, docs/md, docs/pdf
- `archive/legacy_prompts/` — GPT Pro v1/v2 prompts and responses
- `archive/legacy_results_notes/` — ProSeCo chronicle, phase2 change report, LaTeX stray
- `archive/legacy_results_notes/proseco_chronicle/` — chronological audit fragments (superseded by RESULTS_STATUS.md)
- `archive/old_directions/` / `archive/old_notes/` — March 2026 direction / notes

---

## 11. What to read first (for a new reader or yourself in 3 weeks)

1. `docs/thesis/CURRENT_INDEX.md` (this file)
2. `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md`
3. `docs/thesis/experiments/EXPERIMENT_CRITICAL_AUDIT.md` and
   `docs/thesis/theory/THEORY_STRESS_TEST.md` (the post-audit framing)
4. `docs/thesis/experiments/ANALYSIS_SPEC.md`
5. `docs/thesis/experiments/NEXT_PHASE_EXPERIMENT_PLAN.md`
6. `docs/thesis/experiments/RESULTS_STATUS.md`
7. `docs/thesis/theory/THEORY_STATUS.md` (Honesty Ledger + Refinements A′/A″)

---

## 12. What to run next

**Phase 3a complete (2026-04-20, job 479941).** No HPC runs are in flight or
queued. The empirical contract for Chapter 7 is closed: Phase 1 (trajectory
diagnostics) + Phase 2b (paired ranker sweep + MC oracle) + Phase 3a
(combinatorial search-class positive). See `RESULTS_STATUS.md` §§11–14 and
`PHASE3A_COMBINATORIAL_RESULTS.md`.

**Active work — Phase 3b theory finalisation (notebook, no HPC):**

1. Formal proof of **Refinement A′** (variance-form additivity slack,
   `𝔼|G − A| ≤ σ_ξ · √B / √2`); σ_ξ already measured from 9 000 Phase 2b MC
   residuals.
2. Formal proof of **Refinement A″** (rank-based ε_R = (1−|ρ|)·σ_Δ); ρ-decay
   already measured (0.66 → 0.39 across B ∈ {2..16}).
3. Statement and proof of the **Negative-Result Corollary scoped to the
   greedy/separable-ranker class**: any policy `Ŝ_B = top-B(ψ)` is upper-
   bounded by the `mean_delta_oracle` envelope (NULL by B = 8 on
   ProSeCo-OWT). Phase 3a CD-G + BS-AG explicitly exceed this envelope, so
   the corollary cannot generalise to "informed scheduling in general".
4. Update `research/candidate_theorems.md` (promote A′/A″ from candidate to
   stated-and-proven informal), `research/proof_worklog.md` (proof entries),
   `THEORY_STATUS.md` (already partially anchored).
5. Draft `thesis/chapters/ch6_theory.tex` skeleton with refined theorems.
6. Optional: two-page Phase 3 PDF status report at
   `docs/pdf/status/phase3_status_report_2026-04-XX.pdf`.

**Thesis writing — chapters now unblocked.** Chapter 7 (experiments) can
cite Phase 1 trajectory-level findings (Protocol A, Tier T1) + Phase 2b
paired ranker evidence (Tier T2) + Phase 3a search-class positive
(Tier T2). Chapter 6 (theory) can ship Theorem A + Refinements A′/A″ +
ranker-class Negative-Result Corollary.

**Closed / parked workstreams.**

- Phase 2c MAUVE F-swap — closed without execution (Cohen's d up to ±2
  leaves cell ordering robust; Phase 3a's positive eliminates need to
  defend small-B ranker win).
- PRISM pivot — rejected, refined reason post-Phase-3a.
- Cross-backbone (MDLM Phase 2b/3a replication) — parked, out of scope.
- Formal lower bounds on CD-G / BS-AG closure ratios — out of thesis scope.

See `RESULTS_STATUS.md` §§13–14 and `CANONICAL_RESEARCH_DIRECTION.md`
("Immediate next milestone — Phase 3b") for the binding spec.

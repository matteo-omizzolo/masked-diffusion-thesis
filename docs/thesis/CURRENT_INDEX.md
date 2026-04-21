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
| 4 | `docs/thesis/experiments/EXPERIMENT_CRITICAL_AUDIT.md` | **Historical audit** of the Phase 1 policy claims — retained for provenance; superseded by the Phase 2b / Phase 3a mainline |
| 5 | `docs/thesis/theory/THEORY_STRESS_TEST.md` | **Theory stress test** — vacuity verdict, ε-conflation diagnosis, Refinement A′/A″ candidates |
| 6 | `docs/thesis/experiments/ANALYSIS_SPEC.md` | Tier T1–T4 evidence rules, paired estimator definition, BCa CI |
| 7 | `docs/thesis/experiments/NEXT_PHASE_EXPERIMENT_PLAN.md` | Historical Phase 2 stage spec (2a offline / 2b HPC / 2c MAUVE-swap) |
| 8 | `docs/thesis/experiments/RESULTS_STATUS.md` | Summary of Phase 1–3 results and evidence ledger |
| 9 | `docs/thesis/experiments/PROSECO_ANALYSIS.md` | Phase 1 trajectory-level analysis (historical numerical inputs to the stress test) |
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
| `scripts/run_phase2b_proseco_owt.py` | **PRIMARY** — Phase 2b paired K-seed evaluation and MC oracle |
| `scripts/analyze_phase2b.py` | Phase 2b aggregation and ranking analysis |
| `scripts/run_phase3a_combinatorial.py` | **PRIMARY** — Phase 3a combinatorial scheduling baselines |
| `scripts/analyze_phase3a.py` | Phase 3a paired comparison and oracle-gap closure analysis |
| `scripts/stage_proseco_owt.py` | Reproducibility helper: stage the ProSeCo-OWT snapshot |
| `scripts/debug_proseco_owt_load.py` | CPU preflight for the staged ProSeCo-OWT backend |

Legacy Phase 1 scripts live in `archive/legacy_scripts/`.

---

## 6. Active HPC workflow

| File | Purpose |
|---|---|
| `hpc/phase2b_proseco_owt.sbatch` | Phase 2b paired K=30 sweep |
| `hpc/phase3a_combinatorial.sbatch` | Phase 3a CD-G + BS-AG combinatorial baselines |
| `hpc/push.sh` | rsync repo to HPC |
| `hpc/pull.sh` | rsync results back (broken on macOS; use ssh instead) |
| `hpc/setup_env.sh` | One-time conda env bootstrap |

Legacy Phase 1 sbatch files and the old `submit.sh` wrapper live in
`archive/legacy_scripts/` and are not the thesis mainline.

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
| `src/mdm_playground/scheduling/backends/proseco_owt.py` | **PRIMARY** — ProSeCo-OWT staged snapshot loader |
| `src/mdm_playground/scheduling/backends/mdlm_conf.py` | Supporting backend for Phase 1 chronology and regression checks |
| `src/mdm_playground/scheduling/backends/proseco.py` | **LEGACY** — ProSeCo on `mdlm.ckpt` structural no-op |
| `src/mdm_playground/scheduling/backends/mdlm.py` | **LEGACY** — MDLM heuristic baseline |
| `src/mdm_playground/scheduling/signals.py` | Signal extraction (entropy, margin, quality mass) |
| `src/mdm_playground/scheduling/allocation.py` | Policies: uniform, top_B, burn_in_gated, front/back/middle |
| `src/mdm_playground/scheduling/gain.py` | `estimate_single_step_gain` |
| `src/mdm_playground/scheduling/evaluate.py` | `evaluate_schedule` |
| `src/mdm_playground/scheduling/surrogate.py` | Surrogate backend for CPU pipeline validation |

---

## 8. Main results files

| Path | Interpretation |
|---|---|
| `results/phase3a_proseco_owt/{cd,bs}_paired.json` | **Phase 3a** paired outputs (K=30, T=64). CD-G and BS-AG both PASS at every budget tested. |
| `results/phase3a_proseco_owt/oracle_gap_closure.json` | **Phase 3a** oracle-gap closure ratios and verdicts. |
| `results/phase2b_proseco_owt/{policy_raw,mc_raw}.json` | **Phase 2b** raw paired policy rows and MC oracle samples. |
| `results/phase2b/policy_comparison_paired.json` | **Phase 2b** paired BCa confidence intervals. |
| `results/phase2b/mc_oracle.json` | **Phase 2b** MC-oracle headroom summary. |
| `results/phase1_proseco_owt_full/summary.json` | **Prerequisite chronology artifact** for Phase 2b; keep as a read-only supporting result. |

Legacy Phase 1, Phase 2a, and old smoke/eval outputs are archived outside the main path.

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

Everything legacy is under `archive/`. See `archive/ARCHIVE_MANIFEST.md` for
the move-log. The main legacy buckets are `archive/legacy_docs/`,
`archive/legacy_prompts/`, `archive/legacy_results_notes/`, `archive/old_*`,
and `archive/legacy_scripts/`.

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
queued. The empirical contract for the thesis is now: Phase 2b paired
evaluation + Phase 3a combinatorial search-class positive. See
`RESULTS_STATUS.md` §§11–14 and `PHASE3A_COMBINATORIAL_RESULTS.md`.

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

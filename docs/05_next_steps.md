# Next Steps — Action Plan

> **Current source of truth.** Updated 2026-05-05.
> Phase: theory-first reassessment and Phase 0 reproducibility planning.
> No full-scale HPC experiments until the theory scaffold is stable, PF1–PF8
> pass, and the K=3 smoke matches qualitatively.

---

## Sequential research gates (do in order)

### Gate 1 — Opus theory pass ✅ (2026-05; tightened 2026-05-06)

Theorem stack formalized in `research/candidate_theorems.md` §0–§7:

- **Theorem A** (uniform marginal proxy regret 2Bε + 2η_B) — proved; baseline.
- **Diagnostics A′ (additivity scale), A″ (rankability)** — demoted from
  prior proof status to **empirical diagnostics**; do not control
  selected-schedule regret without finite-pool conversion.
- **Theorem A as B′(Q := A)** (§2.7) — safe finite-pool regret form.
- **Empirical Ranker-Class Limitation** — replaces "Negative-Result Corollary";
  formal part for time-only / seed-averaged separable ψ; empirical part on
  tested separable rankers.
- **Theorem B / B′** — central rigorous interaction framework.
  B exact (§2.1), B estimated (§2.2; constant 2α_B), B′ finite-candidate-pool /
  high-probability with κ_B and **no-leakage data-dependence caveat** (§2.3).
- **Levels 1 / 2 / 3** (§2.4) and **level-specific metrics** (§2.6)
  P_B^seed / P_B^pop / P_B^feat, C_B^pop / C_B^feat.
- **Diagnostic Framework C** — regime classification with disciplined notation
  U_B^{MC,N} vs U_B^{pool} vs U_B^* (the last unobservable, never reported).
- **Theorem D** — proof sketch; optional / appendix; first to cut.
- **Lemma E** — conditional / clipped F_C only; optional side lemma.

Theory-to-experiment map: `research/candidate_theorems.md` §7. Backbone is
**A → B / B′ → Diagnostic Framework C**.

---

### Gate 2 — Phase 0 reproducibility audit (prerequisite for any new HPC)

Before launching any new experiment, reproduce the existing ProSeCo-OWT baseline.

**Step 2a — code path audit.**
- [ ] Local package import check: `python -c "import mdm_playground"` passes.
- [ ] Stage ProSeCo-OWT checkpoint: `python scripts/stage_proseco_owt.py`.

**Step 2b — pre-flight assertions (BLOCKING; no smoke until these pass).**
The following invariants must be implemented as tests under `tests/` (or
manually verified) before any HPC submission, including K=3 smoke:

- [ ] **PF1 deterministic base** — same seed + config ⇒ same base tokens and F.
- [ ] **PF2 empty schedule = base** — `run_with_schedule({}) == run_base`.
- [ ] **PF3 single-correction = Protocol A branch** — `run_with_schedule({t})`
      equals the Protocol A single-corrector branch at step t.
- [ ] **PF4 budget accounting** — |S| = B; total extra forward passes = c_corr · B;
      schedules with same |S| have same compute cost.
- [ ] **PF5 CRN consistency** — base and any branch share random numbers
      everywhere except the corrector path.
- [ ] **PF6 F-scoring consistency** — same token sequence ⇒ same F score.
- [ ] **PF7 corrector action set** — ProSeCo corrects only positions in R_t.
- [ ] **PF8 signal/action-set consistency** — H_t, M_t^{-1}, QM_t are computed
      over the same R_t the corrector acts on (avoid historical Bug #1).

Current local entry point: `pytest tests/test_phase0_preflight.py -q`.
The real ProSeCo backend exposes `return_trace=True` hooks for branch/schedule
comparison, RNG fingerprints, and acted-on R_t positions. Checkpoint-backed
checks may remain skipped only when the staged snapshot is unavailable; run
them with `PROSECO_OWT_CHECKPOINT=/path/to/proseco_owt pytest tests/test_phase0_preflight.py -q`.

**Step 2c — K=3 smoke (only after Step 2b passes).**
- [ ] K=3 seeds, T=64, B ∈ {2,4}; uniform + mean_delta_oracle + CD-G + BS-AG.
- [ ] Compare against existing `results/phase2b/` and
      `results/phase3a_proseco_owt/oracle_gap_closure.json` keys.
- [ ] Write only to `results/phase0_smoke_<git-short-sha>/` with config,
      preflight status, per-seed rows, aggregate summaries, and exact commands.

**Step 2d — K=30 critical replication (only after Step 2c matches qualitatively).**

Do not launch Phase 1 until Step 2c passes. Do not launch K=30 until Step 2c
matches qualitatively.

---

### Gate 3 — Interaction diagnostics + schedule-level validation (only after Gate 2 passes)

Two-step gate (pair-level alone is **not** sufficient to validate Theorem B):

**Gate 3a — Sparse pair diagnostics.** Sample a small stratified set of step
pairs (t, t'), measure ξ_{t,t'} = G({t,t'}) − Δ_t − Δ_{t'} with paired CRN,
assess whether the interaction term is negligible or structured by phase /
distance / signal.

**Gate 3b — Schedule-level validation (REQUIRED before Phase 2).** For
B ∈ {2, 3, 4}, fix a no-leakage candidate schedule pool C_B; evaluate G(S)
for sampled schedules S ∈ C_B; compute A(S); compute Q(S) using estimated
ξ; and compare η_{B,C} = sup/robust-percentile |G(S) − A(S)|,
ζ_{B,C} = sup/robust-percentile |G(S) − Q(S)|, R_B = ρ(A(S), G(S)), and
P_B = ρ(Q(S), G(S)) with nested-bootstrap CIs. Pair heatmaps and phase-pair
summaries are diagnostics; they do **not** by themselves validate Theorem B.
Q(S) must be computed from measured ξ for every pair inside S, or from a
pre-registered ξ estimator fitted only on training pairs/seeds. Missing ξ
values must not be filled using held-out G; uncovered schedules are ineligible
for the Q(S) validation pool.

Decision gate: proceed to Gate 4 only if Gate 3b validates Theorem B at the
schedule level, i.e. ζ_{B,C} < η_{B,C} and/or P_B > R_B with uncertainty
accounted for. Otherwise classify provisionally toward Regime IV and
emphasize Diagnostic Framework C.

Do not run dense all-pair maps until sparse diagnostics justify it.

## Restart experiment contract

Every future experiment phase must produce: config JSON/YAML with git commit
hash, per-seed raw rows, aggregate summary JSON, analysis script, paired
bootstrap CI where applicable, one concise interpretation markdown or summary
section, canonical plots when meaningful, and the exact command to regenerate
plots from raw results.

Each experiment must state the research question, theorem/assumption tested,
expected outcome, observed outcome, interpretation, opened/closed gate, and
the first plot/table to inspect. Phase 0 uses only a sanity table comparing old
vs rerun keys. Phase 1 defaults to ξ heatmap, P(ξ>0) heatmap, phase-pair
summary, Q-vs-G versus A-vs-G scatter, ζ_{B,C} versus η_{B,C}, and CI for
P_B − R_B. Phase 2 defaults to gain over uniform, closure ratio versus
MC-oracle or pool oracle, held-out Q_hat-vs-G scatter, and true-G calls versus
performance.

---

### Gate 4 — Pairwise surrogate scheduler (only after Gate 3 shows structure)

If sparse interaction diagnostics show structured interactions:

- Implement the pairwise surrogate scheduler (see plan §3).
- Evaluate against CD-G and BS-AG on ProSeCo-OWT.
- Compare recovery of MC-oracle headroom.

If interactions are negligible → skip this gate, strengthen the negative result claim.

---

### Gate 5 — Regime map and secondary backbone (only if Gate 4 succeeds and time permits)

After the primary pipeline is trustworthy:

- Map interaction structure across different budget levels B.
- Optional: one secondary backbone probe if supervisor approves.

This is a stretch goal, not on the critical path. September deadline takes priority.

---

## Writing tasks (parallel with Gate 1–2, sequential after Gate 3+)

Once the theory scaffold (Gate 1) is stable, writing can proceed in parallel:

1. **ch3 — Discrete Diffusion background** (~15–20 pages)
   - MDLM forward process, predictor, training objective.
   - ReMDM corrector kernel (Barker/MPF). Brief.
   - ProSeCo annealed refinement.

2. **ch4 — Correctors background** (~10 pages)
   - Corrector scheduling problem formalization.
   - L&Z E_fact/E_learn decomposition.
   - Existing approaches and what they do NOT address.

3. **ch5 — Experiments** (~15–20 pages)
   - ProSeCo-OWT backbone description.
   - Phase 0 reproducibility audit result.
   - Protocol A (signal calibration), Phase 2b (policy comparison + MC-oracle),
     Phase 3a (combinatorial search baselines).
   - Results tables and figures. Cross-backbone probe as Section 5.X.
   - Protocol C (adaptive controller) as Section 5.Y or Appendix F.

4. **ch7 — Discussion / Limitations** (~5–8 pages)
   - Single-backbone scope caveat.
   - CD-G as structural existence result.
   - Theorem A's uniform L∞ bound is empirically vacuous; the operative
     selected-schedule form is the finite-pool corollary (Theorem A as
     B′(Q := A)). A′ and A″ are reported as diagnostics, not regret
     refinements.
   - Future: interaction-aware scheduling, multi-backbone replication.

5. **Abstract + Introduction** — write last.

6. **Conclusion** (~3–4 pages) — what was answered, what is still open.

### Theory clean-up tasks (run in parallel with writing)

7. Clean LaTeX proof of Theorem A combining step in `thesis/chapters/ch6_contribution.tex`
   (uniform form), then state the finite-pool corollary as the operative form.
8. State Theorem B / B′ in ch6 with the no-leakage candidate-pool caveat.
9. Report A′ and A″ diagnostics in ch5 / ch7 (not as theorems).
10. Add the Empirical Ranker-Class Limitation (formal time-only part + empirical
    part on tested rankers) in ch6.

---

## What is NOT on the critical path

- Full-scale HPC runs before Phase 0 passes.
- LLaDA-SFT Phase 3a (pre-registered no-go).
- Online controller experiments until Gate 3–4 are complete.
- PRISM pivot — not pursued as a thesis pillar (separable PRISM falls in
  ranker class; non-separable use is optional / future).
- Stretch C2 (Gibbs contraction) — not applicable.
- Third-backbone replication (out of thesis scope unless supervisor approves).

---

## Fallback if Level-3 scheduling fails

If Level-3 feature-conditioned scheduling fails, the thesis does not collapse.
The contribution becomes: Theorem A baseline and empirical falsification on
ProSeCo-OWT; Theorem B/B′ as a rigorous diagnostic framework; evidence that
ProSeCo-OWT is interaction-driven or higher-order/chaotic; CD-G/BS-AG as
search-based existence results using true-G feedback; and Diagnostic Framework C
as a protocol for future model/corrector triples.

## Supervisor check-in

Schedule a Zanella meeting to present:
- Phase 3a positive result (CD-G/BS-AG recover MC-oracle headroom).
- Empirical Ranker-Class Limitation: tested separable rankers do not recover
  MC-oracle headroom; formal time/mean-profile envelope; non-separable PRISM
  uses are not ruled out.
- Theory-first programme (Theorem B/B′, Diagnostic Framework C, appendix D).
- Phase 0 reproducibility gate.
- Writing plan for ch3–ch7.

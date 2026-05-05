# Next Steps — Action Plan

> **Current source of truth.** Updated 2026-05-05.
> Phase: theory-first reassessment and Phase 0 reproducibility planning.
> No full-scale HPC experiments until theory scaffold + Phase 0 audit are complete.

---

## Sequential research gates (do in order)

### Gate 1 — Opus theory pass

Formalize the theorem stack for the theory-first programme:

1. **Theorem B — pairwise surrogate regret**: characterize when corrector timing is
   interaction-driven (the pairwise surrogate ψ₂ approach).
2. **Proposition C — separable ranker failure construction**: show explicitly that any
   separable per-step ranker fails when interactions exceed a threshold.
3. **Theorem D — online budgeted controller abstraction**: define the budgeted online
   decision problem and bound the gap to the offline oracle.

See `docs/06_theory_first_research_plan.md` §2–5 for the full formal programme.

For each theorem, add a theorem-to-experiment mapping: what result would support it,
what would falsify it, what experiment measures it.

---

### Gate 2 — Phase 0 reproducibility audit (prerequisite for any new HPC)

Before launching any new experiment, reproduce the existing ProSeCo-OWT baseline:

**Phase 0 checklist:**
- [ ] Local script import checks: `python -c "import src.mdm_playground"` passes.
- [ ] Stage ProSeCo-OWT checkpoint: `python scripts/stage_proseco_owt.py`.
- [ ] K=3 smoke on ProSeCo-OWT: verify G(S), A(S), Δ_t, F all consistent.
- [ ] Compare against existing Phase 2b + Phase 3a result keys in `results/`.
- [ ] Only then plan full K=30 replication.

Do not launch K=30 until the K=3 smoke matches existing results qualitatively.

---

### Gate 3 — Sparse interaction diagnostics (only after Gate 2 passes)

Run sparse stratified pair sampling to test whether corrector placements interact:

- Sample a small set of step pairs (t, t') ≠ same step.
- Measure ξ_{t,t'} = G({t,t'}) − G({t}) − G({t'}).
- Assess: is the interaction term negligible (σ_ξ small) or structured?
- Decision gate: if interactions are structured → proceed to pairwise scheduler;
  if interactions are negligible → ranker failure is additivity-breaking only.

Do not run dense all-pair maps until sparse diagnostics justify it.

---

### Gate 4 — Pairwise surrogate scheduler (only after Gate 3 shows structure)

If sparse interaction diagnostics show structured interactions:

- Implement the pairwise surrogate scheduler (see plan §3).
- Evaluate against CD-G and BS-AG on ProSeCo-OWT.
- Compare recovery of oracle headroom.

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
   - Protocol A (signal calibration), Phase 2b (policy comparison + MC oracle),
     Phase 3a (combinatorial search baselines).
   - Results tables and figures. Cross-backbone probe as Section 5.X.
   - Protocol C (adaptive controller) as Section 5.Y or Appendix F.

4. **ch7 — Discussion / Limitations** (~5–8 pages)
   - Single-backbone scope caveat.
   - CD-G as structural existence result.
   - Empirically vacuous L∞ bound; operative Refinement A″.
   - Future: interaction-aware scheduling, multi-backbone replication.

5. **Abstract + Introduction** — write last.

6. **Conclusion** (~3–4 pages) — what was answered, what is still open.

### Theory clean-up tasks (run in parallel with writing)

7. Clean LaTeX proof of Theorem A combining step in `thesis/chapters/ch6_contribution.tex`.
8. Write Refinement A′ order-statistics derivation in ch6.
9. Write Refinement A″ rank-based derivation in ch6.
10. Add Negative-Result Corollary formal environment in ch6.

---

## What is NOT on the critical path

- Full-scale HPC runs before Phase 0 passes.
- LLaDA-SFT Phase 3a (pre-registered no-go).
- Online controller experiments until Gate 3–4 are complete.
- PRISM pivot — rejected.
- Stretch C2 (Gibbs contraction) — not applicable.
- Third-backbone replication (out of thesis scope unless supervisor approves).

---

## Supervisor check-in

Schedule a Zanella meeting to present:
- Phase 3a positive result (CD-G/BS-AG recover oracle headroom).
- Negative-Result Corollary (ranker class + PRISM rejection).
- Theory-first programme (Theorem B, D, regime map plan).
- Phase 0 reproducibility gate.
- Writing plan for ch3–ch7.

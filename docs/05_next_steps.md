# Next Steps — Action Plan

> **Current source of truth.** Updated 2026-05-05.
> Synthesized from `docs/thesis/next_steps/POST_REASSESSMENT_DECISION.md`
> (terminal decision as of 2026-04-26).

---

## Critical path: LaTeX writing only

**No new experiments or HPC runs are authorized.** The thesis critical path is
exclusively writing and theory clean-up.

### Immediate writing tasks (ordered)

1. **ch3 — Discrete Diffusion background** (~15–20 pages)
   - MDLM forward process, predictor, training objective.
   - ReMDM corrector kernel (informed Barker/MPF). Brief.
   - ProSeCo annealed refinement. Enough to set up the experiment backbone.

2. **ch4 — Correctors background** (~10 pages)
   - Corrector scheduling problem formalization.
   - L&Z E_fact/E_learn decomposition (why correctors help).
   - Existing approaches: token-selection, remasking, informed kernels — and
     what they do NOT address (trajectory-level budget allocation).

3. **ch5 — Experiments** (~15–20 pages)
   - ProSeCo-OWT backbone description.
   - Protocol A (signal calibration), Protocol B (additivity), Phase 2b (policy
     comparison + MC oracle), Phase 3a (combinatorial search baselines).
   - Results tables and figures. Cross-backbone probe as Section 5.X.
   - Protocol C (adaptive controller) as Section 5.Y or Appendix F.

4. **ch7 — Discussion / Limitations** (~5–8 pages)
   - Single-backbone scope caveat (primary risk).
   - CD-G as structural existence result vs practical scheduler.
   - Empirically vacuous L∞ bound; operative Refinement A″ form.
   - Future: function-approximator policies, multi-backbone replication.

5. **Abstract + Introduction** (~4–6 pages total)
   - Write last (or nearly last) once ch3–ch7 are drafted.

6. **Conclusion** (~3–4 pages)
   - What was answered, what was learned, what is still open.

### Theory clean-up tasks (parallel with writing)

7. **Theorem A combining-step proof** — write the clean LaTeX argument in ch6:
   G(S_B*) − G(Ŝ_B) ≤ [A(S_B*) − A(Ŝ_B)] + 2η_B ≤ 2Bε + 2η_B.
   Section bodies in `thesis/chapters/ch6_contribution.tex` are marked TODO.

8. **Refinement A′ write-up** — order-statistics proof for σ_ξ · √B/√2 form.
   Reference the mixing/cancellation hypothesis explicitly as an assumption.

9. **Refinement A″ write-up** — order-statistics derivation under Gaussian-A.
   Make the Gaussian-A hypothesis explicit. Note it is heuristic.

10. **Negative-Result Corollary in ch6** — formal corollary environment.
    Cite Phase 2b `mean_delta_oracle` envelope entering NULL band by B = 8.

---

## What is NOT on the critical path

- New HPC experiments or runs.
- LLaDA-SFT Phase 3a (pre-registered no-go).
- Adaptive controller (Protocol C) re-runs or extensions.
- PRISM pivot or new backbone studies.
- Stretch C2 (Gibbs contraction theorem) — not applicable.
- Third-backbone replication — out of thesis scope.
- Any further extension of the adaptive-controller direction.

---

## Supervisor check-in

A Zanella meeting should be scheduled to present:
- Phase 3a positive result (combinatorial search recovers oracle headroom).
- Negative-Result Corollary (ranker class + PRISM rejection).
- Theorem A + Refinements A′/A″ formal status.
- Protocol C honest negative → Appendix F.
- Writing plan for ch3–ch7.

---

## Repo cleanup follow-ups (this branch)

After the current cleanup:
- Merge `repo-cleanup-compact-current-state` → main.
- Delete stale tracking files and intermediate docs (already archived).
- Verify `CLAUDE.md` entry points are current.

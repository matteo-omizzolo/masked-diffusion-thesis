# Next Steps — Action Plan

> **Current source of truth.** Updated 2026-05-05.

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

## Second phase: contribution reassessment

After the LaTeX writing is underway, decide among these directions with Zanella.
No new experiments needed for A or B. C/D/E may require new work.

| Option | Gain | Cost | New experiments? | Thesis risk |
|---|---|---|---|---|
| **A. Continue current story** — "rankers fail, search works" as a ProSeCo-OWT case study | Clean, focused | Scope limited to one backbone | No | Single-backbone caveat |
| **B. Narrow scope** — focus on negative result for separable rankers + empirical search evidence | Honest, compact | Positive result (CD-G/BS-AG) may feel undersold | No | May feel insufficient |
| **C. Strengthen practical algorithm** — turn BS-AG into a deployable scheduler via surrogate for G | Practical value | Requires function-approximator work | Yes (new GPU) | Significant scope expansion |
| **D. Second backbone validation** — ask Zanella for one additional backbone experiment | Stronger external validity | Requires HPC + Zanella approval | Yes (HPC) | Adds delay |
| **E. Pivot theory** — reframe as interaction-driven budgeted trajectory control, de-emphasize signals | Novel framing | Theory rewrite needed | Possibly | Requires new analysis |

**Recommended default:** Option A. The current story is coherent and the single-backbone
caveat is manageable as a clearly stated limitation in ch7.

Reassessment should happen at the next Zanella meeting (see §Supervisor check-in above).

> **STATUS:** RECOMMENDATION (closes the 6-phase adaptive-controller research study)
> **LAST VERIFIED:** 2026-04-22
> **SCOPE:** Final recommendation on whether the thesis should pivot to adaptive,
> budget-aware, state-conditional corrector scheduling, or stay with the open-loop
> Theorem-A framing. Reads as the terminal node of the study launched in
> `ADAPTIVE_CONTROLLER_DIRECTION_AUDIT.md`.

---

# Adaptive Controllers — Final Recommendation

## TL;DR

**Do not pivot the thesis to adaptive control.** Position the adaptive extension
as a **Future-Work appendix** (Appendix F) or a single **Future-Work chapter
section**. Commit to **Protocol C only** — a bounded, no-new-GPU re-use of
Phase-2b artefacts — as the smallest scientifically honest step that either
(a) supplies preliminary evidence for the appendix or (b) closes the
adaptive direction with a negative-result statement. Under no circumstances
should work on adaptive controllers delay the remaining main-thesis deliverables
(ch3 discrete diffusion, the Zanella-meeting write-up, Phase-2b close-out on
LLaDA-SFT).

## 1. The three decision questions, answered

**Q-6A.** *Should the thesis pivot to adaptive control now?*

No. The evidence supporting a pivot is inconclusive:

- The Phase-2b +0.45 MC-oracle headroom at B ∈ {2, 3, 4} is a bound on
  Δ_open + noise, not on Δ_open itself. Without a clean Δ_open > 0
  measurement, an adaptive theorem bounds a quantity that might be
  vanishingly small.
- 62 % within-seed variance at B = 4 is **consistent** with state
  conditioning shrinking ε but **does not prove** it.
- The Phase-3a ranker-class negative result is compatible with both
  "adaptivity helps" and "adaptivity doesn't help" (the ranker class
  excludes search procedures, which are also non-adaptive but already
  recover 49–84 % of headroom).
- The cost of writing a formal adaptive theorem without first estimating
  ε̃, η̃_B is: a theorem with unmeasured constants, potentially inert.
  This would weaken the thesis, not strengthen it.

**Q-6B.** *Should the thesis frame the adaptive extension as next-stage work?*

Yes. Framing choices:

- **Preferred:** Appendix F — "Adaptive Extensions to Budgeted Corrector
  Scheduling". Formal statement of the FH-CMDP problem (§2.1 of the
  theory doc) + Theorem A-ad shape + one-paragraph Particle-Gibbs / cSMC
  algorithmic complement + one-paragraph adaptive-submodularity foil +
  Protocol C proposal. ~4 pages.
- **Acceptable alternative:** final chapter section §"Future Work:
  Adaptive State-Conditional Controllers" with the same content compressed
  to ~2 pages. This is preferred if the committee prefers compactness.
- **Not recommended:** promoting A-ad to a main-body theorem without
  Protocol C evidence. This would require the thesis to claim a
  conditional theorem with unestimated constants, which violates the
  direction doc's honesty requirement.

**Q-6C.** *What is the smallest justified next step?*

**Protocol C (pilot, bounded).** Specified in
`docs/thesis/theory/ADAPTIVE_BUDGETED_CONTROLLERS.md` §4.2:

- Re-use Phase-2b's per-seed JSON outputs (LLaDA-SFT, 8 seeds, already
  on disk once Jobs 481264/481265 complete).
- Bucket by z_t = (s_t, b_t, phase(t)); signals s_t ∈ {H_t, M_t^{-1}, Q_t};
  phase(t) ∈ {early, mid, late}.
- Estimate ε̃, η̃_B, Δ_close(π̂_λ,N=1) / Δ_open.
- Single scalar report: fraction of adaptive oracle headroom recovered by
  the deterministic λ-threshold policy.
- Decision rule:
  - If Δ_close / Δ_open > 0.5: write Appendix F with A-ad F1 as a
    formal conditional theorem and Protocol C as preliminary evidence.
  - If Δ_close / Δ_open ≤ 0.5: write Appendix F with A-ad F1 as a
    sketched conditional theorem *and* report the Protocol-C negative
    result. Either way, do not pivot the main thesis.

**Total cost.** ~1 day of analysis work, zero new GPU time, zero new
Phase-2b-style sbatch submissions. Can be done on a laptop from the
existing JSON outputs.

## 2. Why not adopt a larger adaptive programme?

The Phase-1 audit (`ADAPTIVE_CONTROLLER_DIRECTION_AUDIT.md` §4) enumerates
five dangers of a full pivot. Revisiting each with Phase-3 framework
knowledge:

| Danger | Mitigation under Protocol C only |
|---|---|
| Premature framework escalation | Protocol C stays within F1's threshold policy; no learned controller. |
| Theorem loss | A-ad reduces to Theorem A when z_t = s_t; main-body theorem is preserved. |
| Scope creep into token-selection | Action a_t ∈ {0,1} is the only policy output; scope-guard held. |
| Sample-complexity blowup | Bucketed state z_t keeps |Z| ≲ 81; no function approximator. |
| Opportunity cost on Phase-2b | Protocol C re-uses Phase-2b artefacts — zero GPU cost. |
| Loss of portability | Bucketed state is tokenizer-robust if the signal is normalized (M_t, Q_t); entropy H_t may need a tokenizer-aware recast. |

All five dangers are eliminated by the "bounded, no-new-GPU" framing.
Escalation to F3 (Particle Gibbs / cSMC) is **not authorized** by this
recommendation; PG-DLM (arXiv:2507.08390) already covers that empirically.

## 3. Explicit things the thesis does NOT commit to

- Training a learned adaptive controller. Out of scope.
- Running Particle-Gibbs / cSMC on LLaDA-SFT or any backbone. Out of
  scope; analytical citation only.
- Expanding the state beyond (s_t, b_t, phase(t)). Out of scope; any
  richer state requires a function approximator and separate paper.
- Making adaptive optimality a main-body claim. Out of scope; strictly
  Future Work.
- Delaying Phase-2b close-out. The 2h26m RUNNING clock on Jobs
  481264/481265 must remain the top priority.

## 4. Sequencing

Recommended order after Phase-2b close-out:

1. **Phase-2b finish and aggregation** — aggregate `policy_raw.shard*-of-8.json`
   and `mc_raw.shard*-of-8.json` once Jobs 481264/481265 complete; run
   `analyze_phase2b.py`; write `POST_CROSS_BACKBONE_DECISION.md`. This is
   the main-thesis critical path and is **prior to all adaptive work**.
2. **ch3 / Zanella meeting / main thesis writing** — per canonical direction,
   these remain higher priority than any adaptive-extension work.
3. **Protocol C (pilot)** — run when there is a 1-day slot free, re-using
   Phase-2b artefacts. Decision rule per §1.Q-6C above.
4. **Appendix F draft** — written immediately after Protocol C completes
   (whether positive or negative).
5. **Optional**: if the committee signals interest post-defense, explore
   F3 as follow-on work in a paper, not in the thesis.

## 5. Explicit non-escalation clause

Even if Protocol C returns Δ_close/Δ_open > 0.5, the recommendation is to
**keep Appendix F preliminary** — not to expand to:

- a full Particle-Gibbs implementation,
- a learned-controller study,
- a search over state abstractions richer than (s_t, b_t, phase(t)),
- a cross-backbone Protocol C replication.

Each of those is a separate paper. The thesis's contribution on this
extension is: (i) the formal problem statement, (ii) the conditional
theorem, (iii) a pilot empirical measurement. That is enough to earn
Appendix-F status without opening any new scope.

## 6. If the pilot fails (Δ_close/Δ_open ≤ 0.5): the honest negative

A failure to recover oracle headroom with the deterministic threshold
policy on bucketed state does **not** close the adaptive direction — it
closes the *minimal bucketed* adaptive direction and leaves richer-state
adaptivity open. The appendix should say this explicitly:

> *"On LLaDA-SFT at K = 8 with bucketed state z_t = (s_t, b_t, phase(t)),
> the deterministic λ-threshold policy recovers only Δ_close/Δ_open = X <
> 0.5 of the adaptive oracle headroom. This is consistent with two
> hypotheses: (a) Δ_open is small and the ranker-class negative result of
> Phase 3a already captures most of the structure; (b) richer state is
> needed. Distinguishing (a) from (b) is out of thesis scope."*

This preserves the thesis's honesty requirement and gives future work a
clear open question rather than an unqualified negative.

## 7. Links

- Phase-1 audit: `docs/thesis/next_steps/ADAPTIVE_CONTROLLER_DIRECTION_AUDIT.md`.
- Phases-3+4 derivations: `research/adaptive_controller_research_notes.md`.
- Phase-5 theory doc: `docs/thesis/theory/ADAPTIVE_BUDGETED_CONTROLLERS.md`.
- Open-loop canonical theory: `docs/thesis/theory/THEORY_STATUS.md`.
- Open questions (Q-adapt-*): `research/open_questions.md`.
- Provenance (A-ad tags): `research/proof_ledger.md`.
- Thesis direction: `docs/thesis_direction.md`.
- Phase-2b resume plan (for sequencing): `docs/thesis/next_steps/PHASE2B_RESUME_PLAN.md`.

---

*End of 6-phase adaptive-controller research study. The next artefact on
the main-thesis critical path is `POST_CROSS_BACKBONE_DECISION.md`, unblocked
once Phase 2b completes on LLaDA-SFT.*

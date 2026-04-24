# Phase 1 — Independent Audit of Research Direction + Deep-Search Memos

*Date: 2026-04-21. Auditor: Claude (fresh read). Scope: skeptical,
evidence-first re-read of the Phase-3a-aware canonical direction plus the two
deep-research memos in `docs/future ideas/`. Every claim is assigned one of:*

- **Supported**        — file evidence in the repo backs the claim.
- **Plausible**        — consistent with evidence but not directly proven.
- **Speculative**      — forward-looking extrapolation, not evidence-backed.
- **Contradicted**     — repo evidence weakens or rejects the claim.
- **Not assessable**   — outside the evidence the repo provides.

---

## 1. Direction documents under audit

| Document | Core claim |
|---|---|
| `docs/thesis/CURRENT_INDEX.md` (2026-04-20) | Phase 3a complete; thesis story is "combinatorial, not signal-greedy"; cross-backbone is **parked, out of scope**. |
| `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md` (2026-04-20) | Same as above, with explicit scope decisions. |
| `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md` | Phase 3a: CD-G 74–84 % closure, BS-AG 49–64 % closure at B∈{2,3,4}; both PASS at B=8 too. |
| `docs/thesis/theory/THEORY_STATUS.md` + `research/candidate_theorems.md` | Theorem A + refinements A′/A″; Negative-Result Corollary scoped to separable-ranker class. |

## 2. Deep-search memos under audit

| Memo | Core recommendation |
|---|---|
| `docs/future ideas/Deep Research Audit of Budgeted Informed Corrector Scheduling.md` | Cross-backbone replication on LLaDA-SFT is **priority #1**; ProSeCo-OWT is too narrow; push MDP / SMC / BP frameworks. |
| `docs/future ideas/Theoretical Frameworks for Budgeted Informed-Corrector Scheduling in Masked Diffusion Language Models.md` | Two-layer formulation: finite-horizon budgeted MDP as outer layer + control-as-inference / Feynman-Kac / SMC as importable inner approximations. |

---

## 3. Claim-by-claim status

### 3.1 Direction claims

| # | Claim (paraphrased) | Status | Evidence / rationale |
|---|---|---|---|
| D1 | Phase 3a CD-G closes 74–84 % of MC-oracle headroom at B∈{2,3,4}. | **Supported** | `PHASE3A_COMBINATORIAL_RESULTS.md` §3.1 with file-backed numbers in `cd_paired.json`/`oracle_gap_closure.json`. |
| D2 | BS-AG closes 49–64 % at B∈{2,3,4}. | **Supported** | Same file. |
| D3 | Both methods PASS paired BCa vs uniform at B∈{2,3,4,8}. | **Supported** | §3.1 tabulates each CI strictly above 0. |
| D4 | The +0.45 MC-oracle headroom at B∈{2,3,4} is real, not a sampling artefact. | **Supported** | Phase 2b MC ran 9 000 schedules (30 seeds × 3 B × 100 P); paired against uniform per seed. |
| D5 | The Negative-Result Corollary is scoped to the greedy/separable-ranker class, not to informed scheduling in general. | **Supported** | `PHASE3A_COMBINATORIAL_RESULTS.md` §4 + §6 explicitly rewrites the corollary; `research/candidate_theorems.md` matches. |
| D6 | PRISM remains rejected as a member of the ranker class. | **Plausible** | Rejection rests on the argument that PRISM is a per-step learned score, hence separable. The memo's counter-argument (PRISM could be reframed as a non-separable critic in a Feynman-Kac scheme) is not tested in the repo, but nothing in the repo lets PRISM escape the ranker envelope either. |
| D7 | Fixed-budget corrector allocation is a combinatorial trajectory-control problem. | **Plausible** | True in the weak sense that schedules are combinatorial objects; the stronger claim that the *recoverable* structure is irreducibly combinatorial (no separable ranker can match it) is **supported** empirically on ProSeCo-OWT but not proven in general. |
| D8 | Cross-backbone replication is parked / out of scope for the thesis core. | **Supported** | `CURRENT_INDEX.md` and `CANONICAL_RESEARCH_DIRECTION.md` both say so explicitly. |
| D9 | Within-seed variance ≈62 % at B=4 is exploitable. | **Supported** | `combinatorial_diagnostics.json` confirms 61.9 % at B=4; the gap between MC-oracle-G and uniform-G paired at 0.45 is the exploitation. |
| D10 | Phase 2c F-swap was closed without execution. | **Supported** | `RESULTS_STATUS.md` (previous sessions) documents the close-without-execution decision. Not re-verified. |
| D11 | The 2026-04-20 session's reframing overturns no empirical fact from Phase 2b. | **Supported** | Phase 2b raw numbers are reused unchanged by Phase 3a's paired analysis. |
| D12 | Theorem A's proxy-regret bound G(S*) − G(Ŝ) ≤ 2Bε + 2η is proven for the greedy ranker class. | **Plausible** | A proof sketch exists in `research/candidate_theorems.md`; a formal proof with clean assumptions is flagged as open (A′ and A″ refinements explicitly "to formalize"). Thesis defensibility does not require A′/A″ to ship. |

### 3.2 Memo 1 claims (Deep Research Audit)

| # | Claim | Status | Evidence / rationale |
|---|---|---|---|
| M1.1 | ProSeCo-OWT-only evidence is too narrow to support a thesis-level claim. | **Plausible** | The thesis has one backbone + one corrector + one F. External validity is genuinely a gap. But "too narrow for a thesis" is a judgment call: thesis-tier contributions are routinely published on single settings if the scope is declared honestly, which `CURRENT_INDEX.md` now does. |
| M1.2 | ProSeCo-LLaDA-SFT cross-backbone is the highest-value next experiment. | **Contradicted (partially)** by `PRINCIPLED_NEXT_STEPS_PLAN.md` | The principled plan ranks Options 2 (diagnostics) and 5 (pairwise interaction) above it, on cost/evidence grounds. The memo's ranking comes from external-validity priors, not from repo evidence. |
| M1.3 | Pushing to finite-horizon budgeted MDP is the correct framing. | **Plausible but premature** | MDP framing is defensible; the memo's "correct" is strong. Repo evidence supports *descriptive* combinatorial framing; MDP/SMC are importable but not validated on ProSeCo-OWT. |
| M1.4 | Control-as-inference / Feynman-Kac is a promising practical inner layer. | **Speculative** | No repo artefact tests any SMC/Feynman-Kac variant; the memo relies on analogy to image diffusion work. |
| M1.5 | PRISM could be revived inside a non-separable critic (Feynman-Kac twist / BP correction). | **Speculative** | Would require a new experiment; no code or data points in this direction. |
| M1.6 | Cross-backbone is "#1 priority". | **Contradicted in-repo** by `PRINCIPLED_NEXT_STEPS_PLAN.md` | The principled plan *postpones* cross-backbone; the newer `CROSS_BACKBONE_REPLICATION_PLAN.md` gives a bounded GO. These two next-steps documents are **internally inconsistent**. This is an open governance question, not a matter of external validity. |

### 3.3 Memo 2 claims (Theoretical Frameworks)

| # | Claim | Status | Evidence / rationale |
|---|---|---|---|
| M2.1 | The scheduling problem is naturally a finite-horizon budgeted MDP / CMDP / BMDP. | **Plausible** | Formally correct as a modelling choice; choice-of-formalism, not a tested claim. |
| M2.2 | SMC / Particle Gibbs / Feynman-Kac give tractable approximations. | **Plausible but untested** | No repo validation. These are known-good techniques in other domains; porting them to this setting is a research programme, not a settled fact. |
| M2.3 | Adaptive submodularity applies as a falsification lens. | **Plausible** | Phase 2b's ranker failure is consistent with non-submodular (diminishing-returns-violating) structure, but the repo has no formal submodular-gap measurement. |
| M2.4 | Two-layer "outer MDP + inner SMC" is the *right* framework. | **Speculative** | Memo's strongest recommendation; the repo does not test it. Adopting it as the thesis frame would be a **premature framework escalation** — exactly what `ASSESSMENT_OF_MEMOS_AND_ANALYSIS.md` already flagged. |

---

## 4. Internal inconsistency findings

Two direction-track documents give **contradictory** recommendations:

1. **`PRINCIPLED_NEXT_STEPS_PLAN.md`** (earlier, cost/evidence-driven): cross-backbone
   is **postponed**, combinatorial diagnostics + pairwise interaction come first.
2. **`CROSS_BACKBONE_REPLICATION_PLAN.md`** (later, conditional GO): bounded
   cross-backbone replication gets GO for Stage 0+1+2+3 at K=8, B∈{2,4}.

The canonical direction (`CURRENT_INDEX.md`, `CANONICAL_RESEARCH_DIRECTION.md`) says
cross-backbone is **parked, out of scope**. So the binding canonical document
and the latest plan **disagree**.

Resolution pathway (conservative): the *canonical* document is the source of
truth for what ships in the thesis. Cross-backbone can still run as an
*optional appendix* if the bounded run is cheap and clean, but the core thesis
story must remain defensible without it.

---

## 5. Summary verdict

- The **empirical core** of the Phase-3a-aware direction is **supported** by
  in-repo evidence (claims D1–D5, D9, D11). The rewrite of the Negative-Result
  Corollary is consistent and not overreached.
- The **scope** claim (D8: cross-backbone parked) is supported by the
  canonical document but **contradicted inside the next-steps folder** by
  `CROSS_BACKBONE_REPLICATION_PLAN.md`. This inconsistency is a
  documentation-hygiene issue, not an evidence issue.
- **Memo 1's "cross-backbone is #1" framing**: not supported by the in-repo
  principled next-steps plan; defensible only as an external-validity opinion.
- **Memo 2's two-layer MDP + SMC framework**: theoretically sound but
  **speculative** for thesis adoption; no repo experiment validates it.
- **Directional verdict**: proceed with the Phase-3a-aware canonical story
  (combinatorial, scoped to ProSeCo-OWT, ranker-class negative + search-class
  positive). Treat cross-backbone as a **bounded appendix experiment**, not as
  a thesis-core requirement. Treat the MDP/SMC framework memo as a
  future-work section, not an active research direction for this thesis.

---

## 6. Open risks flagged for downstream phases

- **R1**: The 1.5× claim for top-k-vs-oracle Jaccard in some direction drafts
  may not be supported by `combinatorial_diagnostics.json` (observed ≈1.2×).
  Hand-off to Phase 2 audit for verification.
- **R2**: Cross-backbone replication plan's internal consistency with the
  canonical direction needs GO/NO-GO triage in Phase 3.
- **R3**: Absence of any submodular-gap measurement or formal search-class
  lower bound means the "search-class positive" claim is empirical-only.
  Acceptable for the thesis but must be declared honestly.
- **R4**: Any memo-driven framework adoption (MDP, SMC, BP) must not be taken
  into the thesis without matching repo-side evidence.

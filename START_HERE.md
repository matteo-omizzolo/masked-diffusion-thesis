# START HERE — Thesis Orientation (2-minute read)

> **Current source of truth.** Updated 2026-05-05.
> All detailed docs are reachable via `docs/README.md`.
> Files under `docs/archive/` are historical and must not be used to infer
> current status unless explicitly requested.

---

## Thesis in one sentence

Corrector budget allocation in masked diffusion LMs is a **combinatorial
schedule-search problem**, not a greedy signal-ranking problem. Cheap per-step
rankers fail; search procedures (CD-G, BS-AG) recover 49–84 % of oracle
headroom on ProSeCo-OWT.

---

## Current thesis question

> For a fixed predictor schedule and fixed corrector NFE budget in masked
> diffusion language models, can aggregate trajectory signals — entropy,
> confidence margin, or quality mass — predict the marginal value of a
> corrective refinement loop well enough to outperform uniform corrector
> placement?

---

## What is established (reliable)

- **Phase 1 OWT.** 50 trajectories × T = 64; per-step Δ_t and signals measured
  on ProSeCo-OWT. MC-oracle headroom over uniform = **+0.45 paired G**
  at B ∈ {2, 3, 4} (95 % CI excludes 0). Raw: `results/phase1_proseco_owt_full/`.
- **Phase 2b OWT.** All greedy single-step rankers (including cheating
  oracle-score ranker) fail to beat uniform by B = 8. Additivity residual
  σ_ξ and Spearman ρ(A,G) measured. Raw: `results/phase2b_proseco_owt/`,
  `results/phase2b/`.
- **Phase 3a OWT.** CD-G recovers 74–84 %, BS-AG 49–64 % of +0.45 headroom
  at B ∈ {2, 3, 4}; both still pass at B = 8. This is the primary positive
  result. Raw: `results/phase3a_proseco_owt/`.
- **Cross-backbone (LLaDA-SFT bounded).** Uniform-not-beaten transfers; MC-oracle
  headroom does NOT transfer at tested resolution. Phase 3a not authorized on
  this backbone. Raw: `results/cross_backbone/`.
- **Protocol C (adaptive controller OWT, CPU).** Bucketed-state conditioning
  shrinks ε by < 1.7 %. Honest negative. Theorem A-ad lives in Appendix F.
  Raw: `results/protocol_c_owt/`.
- **Theory.** Theorem A, Refinements A′/A″, Negative-Result Corollary all
  formally proved under explicit assumptions. ch6 skeleton drafted.

---

## What is NOT done (open)

- **LaTeX chapters 3, 4, 5, 7.** Background (discrete diffusion, correctors,
  experiments) and discussion. ch6 skeleton exists; bodies are TODO.
- **Abstract + Introduction + Conclusion.** Not started.
- **Clean Theorem A proof in LaTeX.** Skeleton in ch6; full narrative not written.

---

## What is deprecated / closed

- **LLaDA-SFT Phase 3a:** Pre-registered no-go.
- **Adaptive controller (Protocol C):** Closed with honest negative. Appendix F only.
- **Greedy ranker as primary method:** Negative result. Search procedures are
  the right policy class.
- **PRISM pivot:** Rejected. PRISM is in the ranker class bounded by the
  Negative-Result Corollary.

---

## Current phase

We are no longer in "writing only" mode. The current phase is **theory-first reassessment**:

1. **Opus theory pass** — formalize Theorem B (pairwise surrogate regret), Proposition C
   (separable ranker failure construction), and Theorem D (online budgeted controller abstraction).
2. **Phase 0 reproducibility audit** — reproduce the ProSeCo-OWT baseline locally before
   any new HPC experiments.
3. **Interaction diagnostics** — only if Phase 0 passes; run sparse pairwise Δ_t maps to
   test whether corrector placements interact.
4. **Pairwise scheduler** — only if interaction diagnostics show structure.
5. **Regime map** — only after the primary pipeline is trustworthy.
6. **LaTeX writing** — ch3, ch4, ch5, ch7, Abstract, Introduction, Conclusion.

> No full-scale new HPC experiments until the theory scaffold and Phase 0 audit are complete.
> ProSeCo-OWT remains the baseline. See `docs/05_next_steps.md` for the sequential plan.
> Full theory-first plan: `docs/06_theory_first_research_plan.md`.

---

## Key files by purpose

| Purpose | File |
|---|---|
| Research direction + scope | `docs/01_research_direction.md` |
| Experiment summary | `docs/02_experiments.md` |
| Theory summary | `docs/03_theory.md` |
| Raw results index | `docs/04_results_index.md` |
| Detailed theorems (worklog) | `research/candidate_theorems.md` |
| Proof derivations | `research/proof_worklog.md` |
| ch6 LaTeX skeleton | `thesis/chapters/ch6_contribution.tex` |
| HPC / environment | `CLAUDE.md` §HPC |

> For agents: see `docs/README.md` for the complete doc structure.
> Do not use files under `docs/archive/`, `archive/`, or `docs/thesis/`
> to infer current thesis status — those are historical.

# START HERE — Thesis Orientation (2-minute read)

> **Current source of truth.** Updated 2026-05-05 as part of repo cleanup.
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

- **HPC runs:** No new runs authorized. Phase 2b / 3a / Protocol C complete.
- **LLaDA-SFT Phase 3a:** Pre-registered no-go.
- **Adaptive controller (Protocol C):** Closed with honest negative. Appendix F only.
- **Greedy ranker as primary method:** Negative result. Search procedures are
  the right policy class.
- **PRISM pivot:** Rejected. PRISM is in the ranker class bounded by the
  Negative-Result Corollary.

---

## Immediate next steps

1. Write ch3 (Discrete Diffusion background) — ~15–20 pages.
2. Write ch4 (Correctors background) — ~10 pages.
3. Write ch5 (Experiments: protocol, results) — ~15–20 pages.
4. Write ch7 (Discussion / Limitations) — ~5–8 pages.
5. Write Abstract + Introduction + Conclusion.
6. Clean Theorem A LaTeX proof narrative in ch6.

See `docs/05_next_steps.md` for detailed action plan.

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

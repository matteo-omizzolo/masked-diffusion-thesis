# START HERE — Thesis Orientation (2-minute read)

> **Current source of truth.** Updated 2026-05-08.
> All detailed docs are reachable via `docs/README.md`.
> Files under `docs/archive/` are historical and must not be used to infer
> current status unless explicitly requested.

---

## Thesis in one sentence

Corrector budget allocation in masked diffusion LMs is a **regime question**:
when is timing reducible to marginal ranking, and when does it need interaction
or search? On ProSeCo-OWT, tested separable rankers do not recover
MC-oracle headroom; search procedures (CD-G, BS-AG) recover 49–84 % of that
headroom.

---

## Current thesis question

> For a fixed predictor schedule and fixed corrector-placement budget B in
> masked diffusion language models, when is informed correction timing
> reducible to marginal signal ranking, and when does it require
> interaction-aware or search-based scheduling?

---

## What is established (reliable)

- **Phase 1 OWT.** 50 trajectories × T = 64; per-step Δ_t and signals measured
  on ProSeCo-OWT. MC-oracle headroom over uniform = **+0.45 paired G**
  at B ∈ {2, 3, 4} (95 % CI excludes 0). Raw: `results/phase1_proseco_owt_full/`.
- **Phase 2b OWT.** Separable per-step rankers (including the cheating
  paired-Δ̂_t oracle-score ranker) do not recover the available MC-oracle
  headroom and saturate as B grows; the mean-Δ̂_t oracle envelope enters the
  no-detectable-gain band by B = 8. Additivity residual σ_ξ and ρ(A,G)
  measured. Raw: `results/phase2b_proseco_owt/`, `results/phase2b/`.
- **Phase 3a OWT.** CD-G recovers 74–84 %, BS-AG 49–64 % of +0.45 headroom
  at B ∈ {2, 3, 4}; both still pass at B = 8. This is the primary positive
  result. Raw: `results/phase3a_proseco_owt/`.
- **Cross-backbone (LLaDA-SFT bounded).** Uniform-not-beaten transfers; MC-oracle
  headroom does NOT transfer at tested resolution. Phase 3a not authorized on
  this backbone. Raw: `results/cross_backbone/`.
- **Protocol C (adaptive controller OWT, CPU).** Bucketed-state conditioning
  shrinks ε by < 1.7 %. Honest negative. Theorem A-ad lives in Appendix F.
  Raw: `results/protocol_c_owt/`.
- **Theory.** Theorem A (uniform marginal proxy regret) proved under explicit
  assumptions; A′ and A″ are empirical diagnostics (additivity scale and
  rankability), not unconditional regret refinements. The safe selected-schedule
  consequence of A is its finite-pool form (Theorem A as B′(Q := A)). The
  former "Negative-Result Corollary" is reframed as the **Empirical
  Ranker-Class Limitation** with a formal part for time-only / seed-averaged
  separable rankers and an empirical part on tested rankers. ch6 full
  mathematical framework drafted.

---

## What is NOT done (open)

- **LaTeX ch1–ch6.** Introduction, background, informed-corrector bridge, and
  ch6 mathematical framework now have full drafts. They still need supervisor
  review and final polish. Empirical claims in ch7 may now be written.
- **LaTeX ch7 / experiments, abstract, conclusion.** Not written.
- **Interaction diagnostics (Phase 1).** Gate 3 (sparse pair diagnostics + schedule-level
  validation) not yet run. This is the current blocking gate for further theory claims.

---

## What is deprecated / closed

- **LLaDA-SFT Phase 3a:** Pre-registered no-go.
- **Adaptive controller (Protocol C):** Closed with honest negative. Appendix F only.
- **Greedy ranker as primary method:** Negative result for the *separable
  per-step ranker class*. Schedule-search and pairwise-aware procedures are
  the right policy class for the observed headroom on ProSeCo-OWT.
- **PRISM pivot:** Not pursued as a thesis pillar. PRISM-style quality
  signals, **when used as separable per-step scores**, fall within the
  ranker class limited by the Empirical Ranker-Class Limitation (`research/candidate_theorems.md` §1.5); a non-separable
  use of PRISM is not ruled out and remains optional / future work.

---

## Current phase

K=30 critical replication is **complete**. The current phase is **Phase 1 interaction diagnostics**:

1. **Opus theory pass ✅ (2026-05).** Theorem stack formalized in
   `research/candidate_theorems.md` §0–§7: Theorem A (marginal baseline),
   Theorem B exact / estimated / **B′ finite-pool high-probability**,
   **Diagnostic Framework C** (regime classification; renamed from Proposition C),
   Theorem D (optional/appendix), Lemma E (optional/conditional).
2. **Phase 0 reproducibility audit ✅ (2026-05-07).** PF1–PF8: 11/11 passed on
   HPC login node (CPU, slnode01). K=3 smoke on A100 (gnode01, job 489457)
   qualitatively matches prior results.
3. **K=30 critical replication ✅ (2026-05-08).**
   - Phase 2b K=30 replication: job 490106, gnode01, finished 2026-05-07.
     Output: `results/phase2b_k30_rep_cf89e00/`. Headroom ~+0.37 (canonical +0.45);
     separable rankers fail (−38 % to −72 %); qualitative conclusions unchanged.
   - Phase 3a K=30: job 479941, gnode02, all 30 seeds complete (2026-04-20).
     Output: `results/phase3a_proseco_owt/`. CD-G 74–84 %, BS-AG 49–64 % of
     +0.45 headroom (canonical baseline; p < 0.001 at B = 2, 3, 4). Gate closed.
4. **Phase 1 interaction diagnostics** — ✅ **NOW OPEN.** Sparse pair diagnostics
   (Gate 3a) then schedule-level validation (Gate 3b). See `docs/05_next_steps.md`.
5. **Phase 2 pairwise scheduler** — blocked until Gate 3b passes.
6. **Phase 3 regime map** — optional secondary backbone.
7. **LaTeX writing** — ch1–ch6 drafted; ch7 / experiments, Abstract, and
   Conclusion may now be written (K=30 gate closed).

> K=30 replication complete (Phase 2b job 490106 ✅, Phase 3a job 479941 ✅).
> Phase 1 interaction diagnostics gate is **open**. Submit Phase 1 sbatch
> after Codex pre-submit review. ProSeCo-OWT remains the only active backbone.
> See `docs/05_next_steps.md` for the full sequential gate plan.

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
| ch6 mathematical framework | `thesis/chapters/ch6_contribution.tex` |
| HPC / environment | `CLAUDE.md` §HPC |

> For agents: see `docs/README.md` for the complete doc structure.
> Do not use files under `docs/archive/`, `archive/`, or `docs/thesis/`
> to infer current thesis status — those are historical.

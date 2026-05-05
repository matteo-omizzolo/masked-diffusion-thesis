> ARCHIVED — historical only.
> Do not use this file to infer the current thesis status unless explicitly asked.
> Current source of truth: see `START_HERE.md` and `docs/README.md`.

---


> **STATUS:** DECISION (closes the bounded cross-backbone replication on LLaDA-SFT)
> **LAST VERIFIED:** 2026-04-23
> **SCOPE:** Final go/no-go on Phase 3a (CD-G, BS-AG, B ∈ {2, 3, 4}) on the LLaDA-SFT backbone, conditioned on the bounded K=8 Phase 2b evidence recorded in `docs/thesis/experiments/CROSS_BACKBONE_REPLICATION_RESULTS.md` §10–§11. Reads as the terminal node of the replication launched under `docs/thesis/next_steps/LARGE_MODEL_CONTINUATION_DECISION.md` (GO for Phase 2b only) and `docs/thesis/next_steps/PHASE2B_RESUME_PLAN.md`.

---

# Post-Cross-Backbone Decision — LLaDA-SFT Bounded Replication

## TL;DR

**Do not run Phase 3a on LLaDA-SFT.** The bounded K=8 Phase 2b on this backbone, conducted as an external-validity probe for the OWT K=30 mainline, shows **zero MC-oracle headroom over uniform at B=4** (paired mean = 0.00, bootstrap 95% CI = [0, 0]) and **negative MC-oracle headroom over uniform at B=2** (paired mean = −2.64, CI = [−4.07, −1.07]). Phase 3a's research question — "does a ranker-class search recover most of the MC-oracle headroom?" — is not answerable when the headroom observed at this bounded setup is zero. The replication is **closed with a partial-transfer verdict** (the uniform-not-beaten observation is corroborated at T3 tier under the tested bounded setup; positive MC-oracle headroom did not transfer at the tested budgets (T=64, B ∈ {2, 4}, GPT-2 reference)). Phase 3a on LLaDA-SFT is not authorized. The OWT Phase 2b / Phase 3a mainline is not displaced by this probe.

## 1. The go/no-go questions, answered

**Q-xb-1.** *Does Phase 2b on LLaDA-SFT reproduce the OWT Phase 2b positive on MC-oracle headroom?*

No, not at the tested budgets. On OWT K=30 the MC-oracle-minus-uniform paired mean was +0.45 NLL units at B ∈ {2, 3, 4} with bootstrap 95% CIs excluding zero. On LLaDA-SFT K=8 with B_values = {2, 4}, T=64, GPT-2 reference, and ProSeCo-style corrector, the paired mean is 0.00 at B=4 (exact — 0/8 seeds see any MC-oracle schedule exceed uniform) and −2.64 at B=2 (5/8 seeds have negative diff, 3/8 have zero). **Positive MC-oracle headroom over uniform did not transfer** on LLaDA-SFT at the tested bounded setup. No claim is made about other (T, B, reference) points on this backbone.

Source: `results/cross_backbone/proseco_llada_sft_bounded/phase2b/mc_oracle.json` → `data.mc_oracle_minus_uniform`.

**Q-xb-2.** *Given the answer to Q-xb-1, should Phase 3a (CD-G, BS-AG) be run on LLaDA-SFT?*

No. Phase 3a's entire research question assumes that MC-oracle headroom exists — it tests whether a ranker-class search procedure can recover most of that headroom in paired G. When headroom is zero, there is nothing for a search procedure to recover. Running Phase 3a on this backbone would produce either (a) a null result that restates §10's Phase 2b verdict at additional cost, or (b) a spurious positive from K=8 variance — neither of which improves the thesis. Additionally, Phase 3a was explicitly pre-gated in `LARGE_MODEL_CONTINUATION_DECISION.md` on a positive Phase 2b; that precondition failed.

**Q-xb-3.** *Should the replication be extended to a larger K or a different (T, B, reference) regime on LLaDA-SFT?*

Not on the thesis critical path. The three hypotheses that explain the non-transfer (corrector-dominance H1, protocol-sparseness H2, reference-mismatch H3 — see `CROSS_BACKBONE_REPLICATION_RESULTS.md` §11.1) are not distinguishable at K=8 and require a **multi-dimensional sweep** to disambiguate: at minimum (K ≥ 30) × (B-sweep reaching B/T ≥ 10%) × (within-backbone reference) — which is outside the remaining main-thesis budget. Any such sweep should be proposed as a separate paper or committee-signalled follow-on, not inserted into the current thesis.

**Q-xb-4.** *What does the thesis say about cross-backbone generality?*

The thesis may report, with Tier-3 ("n=8 < 30; exploratory only") annotations and always scoped to the tested bounded setup:
- The **uniform-not-beaten** observation is corroborated on a second backbone (LLaDA-SFT + ProSeCo corrector) under bounded resolution. This is T3 evidence for the OWT finding's external validity; it does **not** broaden to a cross-backbone claim of uniform-optimality.
- The positive **MC-oracle headroom** over uniform observed on OWT is not observed on LLaDA-SFT under this bounded protocol — a **partial, not full, transfer**. Not a claim about other (T, B, reference) points.
- The **per-trajectory > mean-profile** result transfers with a very large effect size (cohens_d ≈ 7.8 at B=4 on LLaDA-SFT).

The thesis must **not** say: "MC-oracle-headroom is a generic feature of masked-diffusion corrector scheduling" (the LLaDA-SFT probe does not provide evidence for that reading at this bounded setup); "the ranker-class negative-result corollary generalises cross-backbone" (its premise, positive MC-oracle headroom, was not observed on LLaDA-SFT at this bounded setup — the corollary is neither confirmed nor refuted there); or "uniform is optimal across masked-diffusion corrector scheduling" (the LLaDA-SFT probe is T3 external-validity evidence scoped to one bounded setup — it is not a breadth statement).

## 2. Why not extend the run?

| Extension | Rationale for NOT doing it now |
|---|---|
| K: 8 → 30 on LLaDA-SFT | Phase 2b seed wall-time ≈ 4.5 h × 8 MIG slices ≈ 36 GPU-hours for K=8. K=30 scales to ~135 GPU-hours and does not disambiguate H1/H2/H3. |
| Add B ∈ {3, 6, 8} | Requires a re-run of Phase 2b with modified `--B_values`; does not fit the bounded protocol; the present null-at-B=4 result would still need explaining. |
| Switch reference from GPT-2 small to LLaDA-SFT own-likelihood | Requires a new scorer plumbing on the ProSeCo-SFT backend and a repeat of Protocol A + Phase 2b. Not prototyped, outside current scope. |
| Run Phase 3a anyway as "exploratory" | Would consume ~10 GPU-hours on a backbone where the MC oracle cannot beat uniform. Strictly lower-value than the no-go decision and would add publication risk if reported at face value. |
| Run on a third backbone (MDLM, ReMDM, dLLM) | Prior evidence on MDLM (Job 478600) and ReMDM-conf (Job 478929) showed structural failure modes (all-negative and all-zero Δ_t respectively); these backbones are not informative for this question. |

## 3. What the bounded K=8 replication *did* deliver to the thesis

Three contributions that stand without Phase 3a on LLaDA-SFT:

1. **A second-backbone corroboration of the uniform-not-beaten observation under bounded resolution.** At Tier-3 confidence, the code-named `uniform_is_optimal` gate passes on LLaDA-SFT K=8 just as it did on OWT K=30. The thesis can cite this as external-validity corroboration for the main-body ranker-class negative result, provided the tier annotation is preserved and the statement is scoped to the tested bounded setup (T=64, B ∈ {2, 4}, GPT-2 reference). This does not broaden to "uniform is generically optimal across masked diffusion corrector scheduling".

2. **An empirical demarcation of where corrector scheduling has non-trivial structure.** The transfer matrix in `CROSS_BACKBONE_REPLICATION_RESULTS.md` §11 is now a live data point: backbones × (T, B, reference) configurations partition into "headroom > 0" (OWT + T=32) and "headroom ≈ 0" (LLaDA-SFT + T=64 + GPT-2 ref). This is a genuine open question for future work and belongs in the discussion, not as a main-body claim.

3. **A large-effect transfer of the per-trajectory-signal > mean-profile-schedule result** at cohens_d ≈ 7.8 (B=4, LLaDA-SFT). This is the single positive finding in the LLaDA-SFT Phase 2b matrix. It is not enough to overturn the overall partial-transfer reading but is worth citing when the thesis discusses why `mean_delta_oracle` is a misleading baseline.

## 4. Explicit non-escalation clause

The decision to not run Phase 3a on LLaDA-SFT is **not** contingent on any of the following conditions becoming true later:

- **K=8 → K=16 or K=30 on LLaDA-SFT.** Out of scope. If the committee explicitly signals interest, it becomes a follow-on paper, not an addendum to the thesis.
- **Phase 3a with B ∈ {3, 6} on LLaDA-SFT.** Out of scope; would require a fresh Phase 2b first to establish whether headroom exists at those Bs.
- **A different scorer (e.g. LLaDA-SFT own-likelihood).** Out of scope; not a drop-in change.
- **A third backbone (e.g. dLLM).** Out of scope; no backbone in the current pipeline has a ProSeCo-style corrector + a K=30-viable compute budget that would not displace main-thesis writing.

None of these extensions is authorized by this decision. Each is a separate follow-on question.

## 5. Sequencing after this decision

Recommended order for the remaining main-thesis critical path:

1. **Main-thesis writing (ch3 Discrete Diffusion, ch5/6 empirical, ch7 experiments).** Resume immediately. The `CROSS_BACKBONE_REPLICATION_RESULTS.md` file is the empirical anchor for the external-validity probe; `PHASE3A_COMBINATORIAL_RESULTS.md` is the anchor for the OWT mainline; `POST_ADAPTIVE_CONTROLLER_RECOMMENDATION.md` (this directory, already closed) and this file are the two terminal decision nodes for the post-Phase-2b state.
2. **Zanella meeting writeup** — drafted at `docs/thesis/next_steps/ZANELLA_MEETING_WRITEUP.md`. Presents the partial-transfer verdict and the three-hypothesis enumeration (H1/H2/H3) as a genuine open question for a disambiguation sweep, not as a thesis-body claim. Read alongside this file at the meeting.
3. **Protocol C pilot** (per `POST_ADAPTIVE_CONTROLLER_RECOMMENDATION.md`). Independently scheduled, re-uses Phase 2b artefacts, no new GPU time — can be done once main-thesis writing is unblocked. Remains Future-Work classification; a Protocol C positive would motivate, not validate, the adaptive-controller extension.
4. **Metric-layer strengthening planning (optional, non-HPC).** If reference-mismatch (H3) is the leading concern raised at the meeting, prepare a within-backbone scoring plumbing plan for LLaDA-SFT own-likelihood as a **follow-on planning artefact**, not as a run authorization. Does not reopen Phase 3a on LLaDA-SFT absent the §6 precondition.
5. **Adaptive-budgeted-controller theory refinement (future work only).** Theorem A-ad is a strict generalisation of Theorem A with a documented plan at `docs/thesis/theory/ADAPTIVE_BUDGETED_CONTROLLERS.md`. The bounded LLaDA-SFT probe does not validate, test, or motivate this extension; keep main-thesis framing to a single future-work paragraph in ch5 / ch6.
6. **Appendix / future-work framing.** Position the LLaDA-SFT non-transfer alongside the adaptive-extension Future Work section; both are honest open questions. No broadening of the thesis object.

## 6. What would re-open Phase 3a on LLaDA-SFT?

**Only one precondition:** a Phase 2b rerun on LLaDA-SFT (with either K ≥ 30, an expanded B ∈ {2, 3, 4, 6}, or a within-backbone reference) that produces `mc_oracle_minus_uniform.bootstrap_95_ci_lo > 0.05` at any B. This is the same pass/fail threshold used in `analyze_phase2b.py` → `headroom_exists_realistic` gate. Absent that signal, Phase 3a on this backbone is strictly dominated by either writing the main thesis or running Protocol C.

## 7. Honesty ledger

Each claim and its K=8 Tier-3 caveat:

| Claim | Evidence | Tier |
|---|---|---|
| MC-oracle does not beat uniform at B=4 on LLaDA-SFT, under the tested bounded setup | `mc_oracle_minus_uniform.4` diff = 0.0 on 8/8 seeds; bootstrap CI degenerate | T3 (K=8 wide CIs) |
| MC-oracle is strictly worse than uniform at B=2 on LLaDA-SFT, under the tested bounded setup | paired mean = −2.64, CI = [−4.07, −1.07] over 5/8 seeds | T3 |
| Phase 3a precondition fails on this backbone at this bounded setup | two independent gates (`small_B_non_vacuous` = FAIL, `headroom_exists_realistic` = FAIL) | T3 on each, both the same direction |
| Uniform-not-beaten observation is corroborated at T3 under bounded resolution | code-named `uniform_is_optimal` gate = PASS on both OWT K=30 and LLaDA-SFT K=8 | T1 on OWT, T3 on LLaDA-SFT |
| `back`, `middle`, `entropy_bot_B_pt`, `mean_delta_oracle` lose to uniform with very large effect on LLaDA-SFT | cohens_d ∈ [−8.5, −5.1]; paired p < 10⁻⁵ uncorrected | Effect size large enough that K=8 is adequate for direction-of-effect |
| Per-seed A-vs-G rho is bimodal on LLaDA-SFT | per-seed ρ spans [−0.83, +0.97] at B=2 on K=8 seeds | T3 — qualitative observation only |
| The non-transfer on LLaDA-SFT could be driven by H1, H2, or H3 (see CROSS_BACKBONE §11.1) | Three hypotheses not discriminable at K=8 | Open question, not a thesis claim |
| LLaDA-SFT bounded probe does not displace OWT mainline | this file §3 + CROSS_BACKBONE §11.2; thesis object (fixed predictor, fixed informed corrector, fixed corrector budget) unchanged | By construction (asymmetry preserved) |

## 8. Links

- Phase 2b results (this backbone): `docs/thesis/experiments/CROSS_BACKBONE_REPLICATION_RESULTS.md` §10–§12.
- Phase 2b analysis outputs: `results/cross_backbone/proseco_llada_sft_bounded/phase2b/{policy_comparison_paired,mc_oracle,rank_A_vs_G_phase2b}.json`.
- Phase 2b raw artefacts: `results/cross_backbone/proseco_llada_sft_bounded/phase2b_raw/`.
- OWT Phase 2b (for comparison): `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md` §3 and associated `results/phase2b/*.json`.
- Pre-run pre-gating: `docs/thesis/next_steps/LARGE_MODEL_CONTINUATION_DECISION.md` (GO for Phase 2b only; Phase 3a blocked on this file).
- Resume plan: `docs/thesis/next_steps/PHASE2B_RESUME_PLAN.md`.
- Adaptive-controller terminal decision: `docs/thesis/next_steps/POST_ADAPTIVE_CONTROLLER_RECOMMENDATION.md`.

---

*End of bounded cross-backbone replication. The next artefact on the main-thesis critical path is `thesis/ch3/*.tex` (Discrete Diffusion) and the Zanella meeting writeup. Phase 3a on LLaDA-SFT is **not** authorized by this decision and may only be re-opened under the precondition stated in §6.*

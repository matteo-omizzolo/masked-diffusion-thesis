> **STATUS:** COMMUNICATION BRIEF (meeting draft; not a canonical artefact)
> **LAST VERIFIED:** 2026-04-24
> **SCOPE:** Single-document brief for the next Zanella supervision meeting, consolidating (i) the OWT K=30 Phase 2b + Phase 3a main discovery body and (ii) the bounded LLaDA-SFT K=8 Phase 2b external-validity probe, with the partial-transfer verdict and the non-escalation clause. Reads downstream from `CROSS_BACKBONE_REPLICATION_RESULTS.md` and `POST_CROSS_BACKBONE_DECISION.md`; does not re-adjudicate their contents. Contains no new numerical claims beyond what those two files already record.

---

# Zanella Meeting Writeup — Post-Cross-Backbone State

## 0. One-paragraph precis

The thesis' main discovery backbone is **ProSeCo-OWT K=30** (T=64), which produced at Tier-1/Tier-2:
(i) a ranker-class negative result (no separable per-step ranker beats uniform at B=8) and
(ii) a search-class positive (CD-G and BS-AG recover 49–84% of the MC-oracle gap).
The bounded external-validity probe on **ProSeCo-LLaDA-SFT 8B K=8** (T=64, B ∈ {2, 4}, GPT-2 reference) **closed** on 2026-04-23 with a **partial transfer** verdict, all scoped to the tested bounded setup: the uniform-not-beaten observation is corroborated at T3 tier; positive MC-oracle headroom over uniform did **not** transfer at the tested budgets. **Phase 3a on LLaDA-SFT is not authorized** under the present decision (single reopening precondition: MC-oracle−uniform bootstrap 95% CI lower bound > 0.05 on a re-run that addresses reference or B/T sparseness). The OWT mainline is **not** displaced by this probe. The three non-discriminable hypotheses for the non-transfer (corrector dominance, protocol sparseness, reference mismatch) are open questions for discussion, **not** thesis-body claims.

The thesis object is held fixed: fixed predictor schedule, fixed informed corrector kernel (ProSeCo-style), fixed extra corrector NFE budget B. Adaptive-budgeted controllers (state-conditional a_t = π(z_t)) are a **Future-Work extension**; the bounded LLaDA-SFT probe does not validate, test, or motivate that extension.

---

## 1. Core OWT result (main discovery backbone)

**Setup.** ProSeCo-OWT, MDLM-OWT 130M backbone, T=64, K=30 seeds, paired-G protocol, BCa bootstrap 95% CIs.

**Phase 2b (K=30, paired):**
- **Uniform is not beaten** by any separable per-step ranker in the tested signal set (entropy, confidence margin, quality mass, top-B per-trajectory vs mean-profile, front/back/middle). Bootstrap CIs for `policy−uniform` include 0 for every policy in the positive direction.
- **MC-oracle headroom over uniform: +0.45 NLL at B=8** (paired mean, 95% CI excludes 0 at Tier-2 eligibility). This is the empirical anchor for the claim that headroom exists for a search procedure to recover.
- Three smoking guns for the ranker class:
  1. `mean_delta_oracle` saturates: its envelope enters the NULL band by B=8.
  2. Per-trajectory signal schedules dominate the mean-profile schedule (same direction on both backbones).
  3. Spearman ρ(A, G) across policies is positive but not large enough to close the ranker-vs-oracle gap.

**Phase 3a (K=30, combinatorial search):**
- **CD-G closure: 74–84%** of the MC-oracle gap (Tier-1 eligible at B ∈ {3, 4}).
- **BS-AG closure: 49–64%** (Tier-2).
- Both strictly exceed the mean_delta_oracle envelope. The ranker-class negative result is a **scope claim**: it upper-bounds the ranker class via `mean_delta_oracle`, and says nothing about the search class — which non-trivially beats uniform under the same informed-kernel + same-budget constraints.

**Theory anchoring.** Theorem A gives the open-loop proxy-regret bound `G(S_B*) − G(Ŝ_B) ≤ 2Bε + 2η_B` under (1) binary placement, (2) approximate additivity with slack η_B, (3) proxy calibration with slack ε. Refinement A′ replaces the L∞ η_B with a variance-form using σ_ξ from Monte-Carlo residuals; Refinement A″ replaces the L∞ ε with a rank-based ε_R via Spearman ρ(A, G). The Negative-Result Corollary states: any separable per-step ranker is upper-bounded by the mean_delta_oracle envelope, which enters the NULL band by B=8 on OWT; Phase 3a's search procedures exceed that envelope. **The corollary characterises the ranker class only** and does not generalise to "informed scheduling" as a whole.

**What the OWT result licenses.** The thesis may present Theorem A + Refinements + Negative-Result Corollary, empirically anchored on OWT K=30 at Tier-1/Tier-2, with Phase 3a's search-class positive cited as the *upper* side of the gap (ranker < search < MC oracle).

---

## 2. Bounded LLaDA-SFT result (external-validity probe)

**Setup.** ProSeCo-LLaDA-SFT 8B backbone, T=64, K=8 seeds (Tier-3: n=8 < 30, exploratory only), B ∈ {2, 4}, GPT-2 reference scorer (reference-mismatch caveat), paired-G protocol identical to OWT Phase 2b. HPC jobs 481264 + 481265 both completed 0:0.

**Observations, scope-limited to the tested bounded setup:**

| Finding | Direction | Tier |
|---|---|---|
| Uniform is not beaten by any informed policy at B ∈ {2, 4} | Corroborates OWT ranker-class observation | T3 |
| MC-oracle minus uniform paired mean at B=4 | **0.00** (bootstrap 95% CI [0, 0], n_nonzero/n = 0/8) | T3 |
| MC-oracle minus uniform paired mean at B=2 | **−2.64 NLL** (CI [−4.07, −1.07], n_nonzero/n = 5/8) | T3 |
| MC-oracle minus mean_profile_oracle paired mean at B=4 | **+4.29 NLL** (CI [+3.94, +4.66], cohens_d ≈ 7.8) | T3, large effect |
| Pooled Spearman ρ(A, G) at B ∈ {2, 4} | Near zero; per-seed ρ bimodal (−0.83 to +0.97) | T3, open |

**Read:** Under this bounded protocol on this backbone with this reference, positive MC-oracle headroom over uniform **was not observed**. The negative mean at B=2 is "clearest signal" of placement hurting at that (T, B, reference) — not a claim that placement is intrinsically harmful. Per-trajectory > mean-profile ordering transfers with a very large effect size (single clean positive in the matrix).

**Scope assertions that will be made in the meeting:**
- This is **not** a revalidation of OWT theorem ingredients. The bounded probe measured the driving quantities on a second triple at Tier-3 only.
- This is **not** a contradiction of OWT. Theorem A's bound applies on any (backbone, corrector, F) triple where assumptions (1)–(3) hold; the probe did not test the ranker-vs-MC-oracle gap structure at adequate K.
- The OWT K=30 mainline is **unchanged**.

---

## 3. What transferred vs what did not transfer

All statements scoped to the tested bounded LLaDA-SFT setup (T=64, B ∈ {2, 4}, GPT-2 reference) vs OWT K=30.

| OWT finding | LLaDA-SFT bounded status | Verdict |
|---|---|---|
| Uniform is not beaten by any separable per-step ranker | Uniform was not beaten under the tested setup | **Corroborated at T3 tier** |
| Positive MC-oracle headroom over uniform at B ∈ {3, 4, 8} | CI [0, 0] at B=4; CI [−4.07, −1.07] at B=2 | **Did not transfer at the tested budgets** |
| CD-G + BS-AG recover 49–84% of MC-oracle gap | Not tested (Phase 3a not authorized) | **Not applicable at this bounded setup** |
| Per-trajectory signal > mean-profile schedule | MC − mean_profile oracle = +4.29 at B=4 (cohens_d ≈ 7.8) | **Transferred under bounded resolution** |
| η_B grows with B (additivity slack structure) | Pooled ρ(A, G) near zero; per-seed bimodal | **Direction consistent; CI-separable conclusion out of reach at K=8** |

**Key framing (non-negotiable at the meeting):**
- "Partial transfer" means *some* of OWT's observations corroborate at T3 and *some* do not transfer at the tested budgets. It does **not** mean "uniform is optimal across masked diffusion" or "MC-oracle headroom is backbone-generic".
- The probe is a **second-triple data point**, not a validation or refutation of the OWT mainline.
- **No broadening of the thesis object.** Any adaptive-controller direction remains future work.

---

## 4. No-Phase-3a rationale

**Decision** (per `POST_CROSS_BACKBONE_DECISION.md`): Phase 3a (CD-G, BS-AG, B ∈ {2, 3, 4}) on LLaDA-SFT is **not authorized**.

**Why:** Phase 3a's research question assumes MC-oracle headroom > 0 — it tests whether a ranker-class search procedure recovers most of that headroom. On LLaDA-SFT at the tested bounded setup, the MC-oracle over uniform paired CI is [0, 0] at B=4 and [−4.07, −1.07] at B=2. A search procedure cannot recover headroom where the bounded probe observed none. Running Phase 3a here would yield either (a) a null result restating Phase 2b or (b) a spurious positive from K=8 variance. Pre-run gating (`LARGE_MODEL_CONTINUATION_DECISION.md`) explicitly pre-conditioned Phase 3a on a positive Phase 2b; that precondition failed.

**Explicit non-escalation clause.** The decision is **not** contingent on any of the following becoming true later:
- K=8 → K=16 or K=30 on LLaDA-SFT (would become a follow-on paper, not a thesis addendum);
- Phase 3a with B ∈ {3, 6} on LLaDA-SFT (would need a fresh Phase 2b first);
- A different scorer (e.g. LLaDA-SFT own-likelihood) — not a drop-in change;
- A third backbone (MDLM and ReMDM-conf showed structural failure modes in prior diagnostics).

**Single reopening precondition (from §6 of the decision doc):** a Phase 2b rerun on LLaDA-SFT — with K ≥ 30, an expanded B ∈ {2, 3, 4, 6}, or a within-backbone reference — that produces `mc_oracle_minus_uniform.bootstrap_95_ci_lo > 0.05` at any B. Absent that signal, Phase 3a on this backbone is strictly dominated by main-thesis writing or Protocol C.

---

## 5. Top unresolved explanations (H1 / H2 / H3)

Three hypotheses are consistent with the non-transfer of positive MC-oracle headroom on LLaDA-SFT at this bounded setup. **They are not discriminable at K=8.**

- **H1 — Corrector-dominance.** The LLaDA-SFT + ProSeCo-corrector combination is sufficiently good at T=64 that per-step Δ_t is close to white noise around zero in G-value terms. Protocol A on LLaDA-SFT shows both signs of Δ_t in every seed and 0/512 exact zeros — *consistent* with H1, not proof of it.
- **H2 — Protocol sparseness.** B/T = 3.1%–6.3% on the LLaDA-SFT bounded run vs 6.3%–12.5% on OWT Phase 2b (same B, smaller T). A sparser budget may be below the minimum at which placement is informative on this backbone.
- **H3 — Reference mismatch.** GPT-2 small scoring LLaDA-SFT outputs is an off-task reference. The degenerate G values for `uniform`, `front`, and `entropy_top_B_mp` (identical G_per_seed at both B) are consistent with GPT-2 being insensitive to timestep choice when the final decoded sequence is coherent.

**H1/H2/H3 are not orthogonal.** Disambiguation requires either K ≥ 30 on LLaDA-SFT, a within-backbone reference (e.g. LLaDA-SFT own-likelihood), or a B-sweep reaching B/T ≥ 10%. None is on the main-thesis critical path.

**How to present these at the meeting.** As **open questions for follow-on work**, not as thesis-body claims. The thesis may cite H1/H2/H3 as "three non-discriminable hypotheses for the non-transfer" in a discussion / future-work section; it may not adjudicate them.

---

## 6. Recommended next step

In priority order, for the post-meeting critical path:

1. **Resume main-thesis writing.** ch3 (Discrete Diffusion) → ch5 (Informed Correctors) → ch6 (Theoretical Contribution: Theorem A + Refinements A′/A″ + Negative-Result Corollary) → ch7 (Experiments: OWT K=30 main body + bounded LLaDA-SFT Tier-3 external-validity probe). Empirical anchor: `CROSS_BACKBONE_REPLICATION_RESULTS.md` and `PHASE3A_COMBINATORIAL_RESULTS.md`. Communication anchor: this file.

2. **Present the partial-transfer verdict to Zanella** — as an external-validity probe with a non-escalation clause, not as a secondary main-body study. Flag the three-hypothesis decomposition (H1/H2/H3) as a genuine open question to be framed in discussion / future work, not to be adjudicated.

3. **Protocol C pilot** (non-HPC, re-uses Phase 2b artefacts; see `POST_ADAPTIVE_CONTROLLER_RECOMMENDATION.md`). Independently scheduled. Decide post-meeting whether to start now or defer behind thesis writing — it does not displace writing and does not need new GPU time.

4. **Metric-layer strengthening** (optional, non-HPC). If reference-mismatch (H3) is the leading concern, prepare a small within-backbone scoring plumbing plan for LLaDA-SFT own-likelihood — as a **future-work planning** artefact, not as a run authorization.

5. **Adaptive-budgeted-controller theory refinement** (future work only). Theorem A-ad (state-conditional a_t = π(z_t)) strictly generalises Theorem A and reduces to it under trivial z_t. It is **not** validated, tested, or motivated by the bounded LLaDA-SFT probe. If discussed at all at the meeting, keep it to a one-line "future-work direction with a documented plan in `ADAPTIVE_BUDGETED_CONTROLLERS.md`" framing.

**What not to do next:**
- Do **not** re-run Phase 3a on LLaDA-SFT at the current bounded setup.
- Do **not** launch a third backbone without the disambiguation design.
- Do **not** re-scope the thesis object around state-conditional controllers.

---

## 7. Thesis-body vs future-work partition

Clean partition for the meeting discussion:

**Thesis-body (canonical at Tier-1/Tier-2, ranker- and search-class scoped):**
- Theorem A + Refinements A′/A″ + Negative-Result Corollary (ranker-class scope).
- OWT K=30 Phase 2b smoking guns (1)–(3) + Phase 3a CD-G + BS-AG search-class positive.
- Bounded LLaDA-SFT K=8 Phase 2b as a clearly-labelled external-validity subsection (Tier-3), scoped to the tested bounded setup, not a co-equal second empirical study.

**Thesis-discussion (open questions, not adjudicated):**
- Three non-discriminable hypotheses H1/H2/H3 for the LLaDA-SFT non-transfer at the tested bounded setup.
- What other (T, B, reference) points on LLaDA-SFT would look like (framed as question, not answer).
- The backbone-generality frontier: `{headroom > 0}` (OWT Phase 2b @ T=32) vs `{headroom ≈ 0}` (LLaDA-SFT @ T=64 + GPT-2 ref) as an empirical demarcation, not a breadth claim.

**Future-work (not thesis-core):**
- Adaptive-budgeted controllers (Theorem A-ad; state-conditional a_t = π(z_t)) — pending a non-HPC Protocol C re-use of Phase 2b artefacts as a proof-of-concept. Canonical plan: `docs/thesis/theory/ADAPTIVE_BUDGETED_CONTROLLERS.md`; terminal recommendation: `docs/thesis/next_steps/POST_ADAPTIVE_CONTROLLER_RECOMMENDATION.md`.
- Disambiguation sweep across (K, B-sweep, within-backbone reference) on LLaDA-SFT — out of thesis scope; framed as follow-on paper if committee signals interest.
- Third-backbone replication (dLLM or other MDM-native backbones with usable ProSeCo-style correctors).

**What is explicitly neither thesis-body nor future-work:**
- Archived Phase 1-era step-sweeps (MDLM/ReMDM-conf/ReMDM-loop at T ∈ {128, 256, 512, 1000}). Infrastructure only; not cited as evidence.
- Prior RemeDi-RL direction (permanently skipped).

---

## Links

- Main empirical anchor (OWT): `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md`.
- External-validity probe (LLaDA-SFT): `docs/thesis/experiments/CROSS_BACKBONE_REPLICATION_RESULTS.md` §10–§12.
- Terminal decision (Phase 3a no-go on LLaDA-SFT): `docs/thesis/next_steps/POST_CROSS_BACKBONE_DECISION.md`.
- Theorem A canonical record: `docs/thesis/theory/THEORY_STATUS.md`.
- Canonical direction: `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md`.
- Future-work theory (adaptive-budgeted controllers): `docs/thesis/theory/ADAPTIVE_BUDGETED_CONTROLLERS.md` + `docs/thesis/next_steps/POST_ADAPTIVE_CONTROLLER_RECOMMENDATION.md`.

---

*End of meeting brief. This document is a communication artefact; canonical claims remain in the files linked above. No numerical claim in this brief supersedes any upstream artefact.*

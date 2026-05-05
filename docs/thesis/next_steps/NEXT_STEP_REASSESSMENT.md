> **STATUS:** REASSESSMENT (skeptical re-audit of post-Protocol-C state;
> supersedes the HPC question only)
> **LAST VERIFIED:** 2026-04-26
> **SCOPE:** Independent reassessment of (i) the current empirical/theoretical
> state, (ii) whether the adaptive-controller pilot's `honest_negative` is
> robust to a stronger skeptical reading, and (iii) whether any new HPC
> experiment is currently justified. Decisive HPC verdict in §C.

---

# Next-Step Reassessment

## A. Current established state — independent classification

### A.1 OWT mainline

| Item | Status | Anchor |
|---|---|---|
| Phase 1 OWT Protocol A — 50 trajectories × T = 64 with per-step Δ_t and per-step signals | **Supported** | `results/phase1_proseco_owt_full/protocol_a/trajectory_*.json` |
| Phase 2b OWT — K = 30 paired, 10 policies × B ∈ {2, 3, 4, 8, 16}, MC oracle at B ∈ {2, 3, 4} | **Supported** | `results/phase2b_proseco_owt/per_seed/{policy,mc}_rows_seed*.json`, `results/phase2b/{policy_comparison_paired,mc_oracle,combinatorial_diagnostics,theorem_a_constants}.json` |
| Phase 3a OWT — CD-G + BS-AG closure 49–84 % at B ∈ {2, 3, 4}; both PASS at B = 8 | **Supported** | `results/phase3a_proseco_owt/oracle_gap_closure.json`: CD-G 0.79 / 0.74 / 0.84, BS-AG 0.64 / 0.57 / 0.49 at B = 2 / 3 / 4 |
| MC-oracle headroom over uniform = +0.45 paired (B = 2, 3, 4); CIs exclude 0 | **Supported** | `oracle_gap_closure.json`: oracle_ci [0.383, 0.528] / [0.366, 0.519] / [0.386, 0.520] at B = 2 / 3 / 4 |
| Theorem A (open-loop, L∞ form) | **Proved under (1)–(3); empirically vacuous on OWT at every B ∈ {4, 8, 16}** | `THEORY_STATUS.md` Honesty Ledger; `research/candidate_theorems.md` |
| Refinement A′ (variance-form η_B via σ_ξ) | **Empirically anchored; formal proof pending Phase 3b** — σ_ξ_pooled = 0.174 / 0.240 / 0.309 at B = 2 / 3 / 4; the √B / √2 mass-form needs a clean order-statistics derivation | `results/phase2b/theorem_a_constants.json` |
| Refinement A″ (rank-based ε_R via Spearman ρ(A, G)) | **Empirically anchored; formal proof pending Phase 3b** — ρ_pooled = 0.601 / 0.542 / 0.462 at B = 2 / 3 / 4 | same |
| Negative-Result Corollary, ranker-class scope | **Empirically supported on Phase 2b + Phase 3a; formal statement pending Phase 3b** | `THEORY_STATUS.md` |

### A.2 LLaDA-SFT bounded probe

| Item | Status | Anchor |
|---|---|---|
| Uniform-not-beaten transfers at T3 under tested setup | **Supported under bounded resolution (T = 64, B ∈ {2, 4}, GPT-2 ref)** | `CROSS_BACKBONE_REPLICATION_RESULTS.md` §10 |
| Positive MC-oracle headroom does not transfer at tested setup | **Supported under bounded resolution** | `mc_oracle.json`: paired CI [0, 0] at B = 4; [−4.07, −1.07] at B = 2 |
| Phase 3a NOT authorized; reopening precondition pre-registered | **Decision** | `POST_CROSS_BACKBONE_DECISION.md` §6 |
| Three-hypothesis decomposition (H1 corrector dominance / H2 protocol sparseness / H3 reference mismatch) is non-discriminable at K = 8 | **Supported as open question** | same |

### A.3 Adaptive-controller direction

| Item | Status | Anchor |
|---|---|---|
| Theorem A-ad (formal abstract-policy-class form) | **Proved under (1)–(4)** as conditional theorem; bound shape is 2 B ε̃ + 2 η̃_B + 𝒪(√(B / N_cal)) | `ADAPTIVE_BUDGETED_CONTROLLERS.md` §2.1 (formal restatement) |
| Strict-generalisation reduction Theorem A-ad → Theorem A | **Resolved 2026-04-25**: the reduction goes through for the abstract policy class (top-B and threshold-λ are two members), not for a specific policy | same |
| Δ_open > 0 at B ∈ {2, 3, 4} on OWT | **Empirically anchored** at +0.45 paired | `mc_oracle.json` |
| Bucketed state z = (s_t, phase(t)) shrinks ε meaningfully | **Refuted on OWT** — observed ε̃ / ε ∈ [0.983, 0.986] (< 1.7 % shrinkage) | `results/protocol_c_owt/protocol_c_summary.json` |
| Bucketed-state ranker class on OWT recovers Δ_open after σ_ξ uncertainty | **Refuted on OWT** — best after-uncertainty close ratio = +0.015 (entropy, B = 2); negative at every B ≥ 3 | same |
| Hamming distance between threshold schedule and best MC schedule ≈ 2 B | **Supported empirically on OWT** — mean 3.93 / 5.87 / 7.47 at B = 2 / 3 / 4 (max possible 2 B); 0 / 30 exact matches across all signals | same |

### A.4 Plausible-but-unproven

- Refinement A′ formal order-statistics proof (variance-form η_B). Empirically the
  measured 𝔼[\|G − A\|] tracks σ_ξ · √B / √2 to within ~30 % at B = 2, 3, 4 on Phase
  2b but the proof under a mixing/cancellation hypothesis on (ξ_{t,t'}) is not
  written.
- Refinement A″ formal proof. The rank-based ε_R = (1 − \|ρ\|) · σ_Δ definition is
  motivated heuristically; the formal claim 𝔼[A(S_A*) − A(Ŝ_B)] ≤ B · ε_R needs an
  order-statistics derivation under a Gaussian-A hypothesis.
- Negative-Result Corollary as a formal proposition in `candidate_theorems.md`.
  The empirical anchors are clean; the formal corollary statement is not yet
  written.
- Theorem A's combining-step LaTeX writeup. The 2 η_B vs 3 η_B accounting is
  resolved but not in LaTeX form.

### A.5 Speculative / open

- Whether the search-class positive (Phase 3a CD-G / BS-AG) generalises to
  any other backbone. **Out of thesis scope** per
  `POST_CROSS_BACKBONE_DECISION.md`.
- Whether richer state abstractions (e.g. predictor hidden representation,
  triple-position interactions) would shrink ε̃ enough on OWT for an
  adaptive policy to clear the σ_ξ band. Out of thesis scope (function
  approximator, separate paper).
- H1 / H2 / H3 discrimination on LLaDA-SFT.

### A.6 Contradicted

- The "strict generalisation" reduction of Theorem A-ad as originally
  written (resolved 2026-04-25 via abstract-policy-class restatement).
- "Run Protocol C on LLaDA-SFT" (Δ_open ≈ 0 → uninterpretable; refuted by
  the activation audit on 2026-04-25).
- "Bucketed (s_t, phase(t)) is sufficient state for adaptive control on
  OWT" (refuted by Protocol C 2026-04-26).

### A.7 Not assessable

- The σ_ξ scaling beyond B ∈ {2, 3, 4} on OWT. Phase 2b ran MC at B ∈ {2, 3, 4};
  σ_ξ at B = 8 / 16 is currently extrapolated, not measured. Materially
  matters only if a thesis claim depends on the variance-form η_B at B = 8 / 16.

---

## B. Adaptive-controller reassessment — skeptical second look

### B.1 Is the current adaptive-controller study genuinely closed?

**Yes**, but the closure rests on **two independent failure modes**, not one. The
activation audit's `honest_negative` verdict fires on the eps_ratio leg
(eps_ratio > 0.9, i.e., bucketing does not shrink calibration RMS by ≥ 30 %).
The second leg — Δ_close_A / Δ_open after σ_ξ uncertainty < 0.3 — also fires
independently at B ≥ 3 for every signal. A skeptical reader could attempt to
explain away the eps_ratio leg by appealing to "the bucket is too coarse"; but
the σ_ξ-band leg is a structural barrier that does not depend on the bucket
abstraction. Together, the two legs make the closure robust.

A third independent line corroborates the closure: **Hamming distance between
the threshold-λ schedule and the best MC schedule is ≈ 2 B at every (signal,
B) tested on OWT, with 0 / 30 exact matches.** The threshold policy lives in
a disjoint region of schedule space from the high-G MC schedules. This is
qualitatively the same structural finding as Phase 2b's "top-K MC ∩ oracle
Jaccard ≈ 1.2 × random" — the recoverable structure does not factor through
any per-step score.

### B.2 Was the pilot too weak, or the right negative test?

**Right negative test for the bucketed-state ranker class.** Three reasons:

1. **The bucketed-state policy is empirically near-equivalent to existing
   Phase 2b OWT policies.** The threshold-λ policy with bucket-mean ψ̃ on
   entropy on (s_t, phase(t)) preferentially selects low-entropy / late-phase
   steps (since Δ̄ is highest there). This is the same *family* as
   `entropy_bot_B_pt`, which Phase 2b ran at K = 30 paired with TRUE G:

   | Policy | B = 2 | B = 4 | B = 8 |
   |---|---|---|---|
   | uniform G (paired baseline) | 0.113 | 0.282 | 0.476 |
   | entropy_bot_B_pt G | 0.218 | 0.314 | 0.425 |
   | **paired diff** | **+0.105** | **+0.032** | **−0.051** |

   Phase 2b shows entropy_bot_B_pt PASSes only at B = 2 (small positive, +0.105),
   is near-zero at B = 4, and is **negative** at B = 8. This is the SAME
   pattern Protocol C documents under the additive surrogate
   (Δ_close_A = 0.18 / 0.31 at B = 2 / 4 — both lie in the σ_ξ band of the
   above paired diffs). **Phase 2b paired data is the TRUE-G measurement
   that Protocol C was designed to approximate.** No HPC experiment is
   needed to verify Protocol C's verdict at TRUE G — Phase 2b paired data
   already confirms the additive-surrogate verdict at TRUE G under Refinement
   A′'s σ_ξ band.

2. **The pre-registration is honest.** Decision rule was recorded before the
   pilot ran; outcome_class fired on the pre-registered thresholds; verdict
   is binary.

3. **Two independent failure modes** (eps_ratio + after-uncertainty close
   ratio), plus a structural Hamming finding. Robust.

### B.3 Is a stronger but still bounded adaptive experiment scientifically meaningful?

Three candidates were considered:

| Candidate | CPU/GPU | Verdict |
|---|---|---|
| (a) Richer offline state abstraction (linear probe over additional Phase 1 signals: u_t, n_revisable, signal interaction terms) — measure ε̃ / ε on richer z | CPU only | **Available; would only shift bucket-vs-richer-state framing slightly. Not load-bearing.** |
| (b) Threshold-λ policy with TRUE G (run ProSeCo corrector on threshold schedules at K = 30, B = 2, 3, 4, 8) | GPU, ~ 30–50 GPU hours | **Already done by Phase 2b's `entropy_bot_B_pt` cell at K = 30 paired.** A direct run would re-measure the same family of policies on the same backbone. Redundant. |
| (c) Cross-backbone Protocol C on LLaDA-SFT or third backbone | GPU | Refuted by the activation audit (Δ_open ≈ 0 on LLaDA-SFT at tested setup; out-of-scope per `POST_CROSS_BACKBONE_DECISION.md`) |

**No stronger-bounded adaptive experiment passes the bar.** The Phase 2b
paired data on OWT at K = 30 is empirically equivalent to (b); the remaining
candidates are either CPU-cheap diagnostics that don't change the verdict or
out-of-scope cross-backbone work.

### B.4 Would a stronger adaptive experiment require HPC?

(b) would require GPU. But (b) is empirically redundant with Phase 2b's
existing data, so the GPU is not actually needed. **No HPC adaptive
experiment is justified.**

### B.5 Would the adaptive direction strengthen the thesis or just create side work?

**Side work.** Theorem A-ad already lives in Appendix F as a formal
conditional theorem with a documented inert-bound diagnostic. Protocol C
is closed. Any extension beyond bucketed (s_t, phase(t)) would require a
function approximator and is explicitly a separate-paper direction. The
appendix-active classification is the right level.

---

## C. HPC reassessment

### C.1 Candidates considered

| Candidate | Question it answers | Already answered? | Cost (A100-h) | Verdict |
|---|---|---|---|---|
| 1. **No more HPC runs** | — | — | 0 | **Default** |
| 2. **Bounded adaptive HPC: TRUE-G threshold-λ at K = 30** | Does Protocol C's additive surrogate accurately predict TRUE G? | **Yes** — Phase 2b's entropy_bot_B_pt at K = 30 is empirically the same family; paired diffs +0.105 / +0.032 / −0.051 at B = 2 / 4 / 8 | ~ 30–50 | **NO-GO** (redundant) |
| 3. **B = 8 MC oracle on OWT** | What fraction of B = 8 oracle headroom does CD-G recover? Currently CD-G at B = 8 has Δ = +0.322 with no oracle anchor | Phase 2b ran MC at B ∈ {2, 3, 4} only; B = 8 oracle is missing | ~ 30–60 (30 seeds × 50 schedules × T = 64 corrector NFEs) | **NO-GO** (qualitative finding "CD-G beats ranker at B = 8" already established without the closure ratio; the oracle ratio would be informational but not load-bearing) |
| 4. **Pairwise γ at B = 8 / 16 on OWT** | Does the pairwise-interaction bound γ B(B − 1) / 2 hold at B ≥ 8? | σ_ξ at B = 8 / 16 is currently extrapolated from B = {2, 3, 4} | ~ 5–15 (sample 200 random pairs of B = 8 schedules) | **NO-GO** (Refinement A′ replaced Proposition C as the load-bearing additivity bound; γ at B = 8 / 16 is not on the critical path) |
| 5. **Cross-backbone replication on a third backbone (MDLM, dLLM)** | Does the search-vs-ranker dichotomy generalise? | Out of scope per `POST_CROSS_BACKBONE_DECISION.md`; prior MDLM and ReMDM-conf attempts showed structural failure modes | ~ 50–150 | **NO-GO** (out of thesis scope) |
| 6. **Reopen LLaDA-SFT Phase 3a under different (T, B, reference)** | Does Phase 3a transfer with H2 or H3 addressed? | Single reopening precondition pre-registered (`POST_CROSS_BACKBONE_DECISION.md` §6); precondition is `mc_oracle_minus_uniform.bootstrap_95_ci_lo > 0.05` on a re-run that addresses H2 or H3 — **not currently satisfied** | ~ 50–150 | **NO-GO** (precondition not satisfied; would be a follow-on paper if pursued) |

### C.2 What remains genuinely unanswered

Three honest gaps remain in the empirical picture:

- B = 8 / B = 16 MC oracle headroom over uniform on OWT (the existing
  +0.45 figure is at B ∈ {2, 3, 4}). Phase 3a's CD-G at B = 8 has no
  oracle to compare against, so the closure ratio at B = 8 is undefined.
- Search-class generality across (backbone, corrector, F) triples.
- Whether richer-state adaptive control could clear the σ_ξ band.

None of these is on the thesis critical path. The first is a small
informational gap, but Phase 3a's qualitative result "search class beats
ranker class at B = 8" is independent of the closure ratio. The second
and third are explicitly future-work / separate-paper.

### C.3 Decisive verdict

**NO-GO on HPC.** The thesis is best served by Phase 3b theory
finalisation (formal proofs of Refinements A′ and A″, Negative-Result
Corollary formal statement) and LaTeX chapter writeup (ch5 informed
correctors, ch6 Theorem A + refinements + corollary, ch7 OWT main
discovery body + bounded LLaDA-SFT probe). No new HPC experiment
dominates these in scientific value.

The non-HPC critical path is laid out in §D below.

---

## D. Non-HPC critical path — what to do instead

In priority order, all CPU-only and on-thesis-scope:

1. **Phase 3b theory finalisation.** Formal proof of Refinement A′
   (variance-form η_B with order-statistics derivation under a mixing
   hypothesis on ξ_{t,t'}; σ_ξ already measured); formal proof of
   Refinement A″ (rank-based ε_R via Spearman-driven order statistics on
   A(S) under Gaussian-A hypothesis); formal statement of the
   Negative-Result Corollary scoped to the ranker class (and bucketed-state
   ranker class on OWT per Protocol C). Update
   `research/candidate_theorems.md` and `research/proof_worklog.md`. ETA:
   1–2 days.

2. **ch6 LaTeX skeleton.** Formal Theorem A statement + proof sketch
   (Lemma A1 + Lemma A2 + combining step with explicit η bookkeeping);
   formal Refinement A′ + A″ statements with proof sketches; formal
   Negative-Result Corollary; one-paragraph Theorem A-ad future-work
   notice. ETA: 2–3 days.

3. **ch7 LaTeX skeleton.** Phase 1 chronology; Phase 2b paired with
   smoking guns (1)–(3); Phase 3a CD-G + BS-AG closure; bounded
   LLaDA-SFT external-validity probe (subordinate Tier-3 subsection); 
   discussion of three threads (ranker negative, search positive, what the
   bounded probe does and does not establish). ETA: 2–3 days.

4. **ch5 LaTeX skeleton.** Informed-corrector literature review (Zhao
   et al.; PRISM); position the thesis at trajectory-level allocation
   under fixed kernel; one-paragraph Theorem A-ad pointer to Appendix F.
   ETA: 1–2 days.

5. **ch3 / ch4 LaTeX skeleton.** Discrete diffusion + masked-diffusion
   background. ETA: 2–3 days each.

6. **ch1 / ch8 / abstract.** Final pass after ch5–ch7 are drafted.

The total non-HPC budget for ch3 → ch8 + Phase 3b is ~ 2–3 weeks of
focused writing. This is the binding milestone.

---

## E. Honesty ledger

| Claim | Tag |
|---|---|
| Phase 2b's `entropy_bot_B_pt` is empirically the TRUE-G version of Protocol C's bucketed threshold policy | `[Plausible-supported]` — the policies are not literally identical (entropy_bot_B_pt picks bottom-B by *raw* entropy per trajectory; Protocol C threshold-λ picks top-B by *bucket-mean* Δ̄); but the implicit selection is empirically the same family (low-entropy / late-phase steps) and Phase 2b paired diffs match the σ_ξ-band predictions of Protocol C |
| `honest_negative` verdict on Protocol C is robust to skeptical re-reading | `[Supported]` — three independent corroborating lines (eps_ratio, after-uncertainty close ratio, Hamming-2B disjointness) |
| No HPC experiment is currently justified | `[Decision]` — every candidate (§C.1) fails at least one of: redundant with existing data, out of scope, expected gain not load-bearing |
| Phase 3b theory finalisation is the binding next milestone | `[Decision]` — already labelled "ACTIVE" in `CANONICAL_RESEARCH_DIRECTION.md` but no formal proofs / LaTeX yet completed |
| ch5 / ch6 / ch7 LaTeX is currently placeholder-only with structural comments | `[Empirically anchored]` — `wc -l thesis/chapters/*.tex` confirms ch6 = 94 lines mostly comments; ch5 = 100 lines mostly comments; ch7 = 221 lines mostly comments |

---

## F. Links

- Activation audit: `docs/thesis/next_steps/ADAPTIVE_CONTROLLER_ACTIVATION_AUDIT.md`
- Experiment plan: `docs/thesis/next_steps/ADAPTIVE_CONTROLLER_EXPERIMENT_PLAN.md`
- Post-Protocol-C decision: `docs/thesis/next_steps/POST_ADAPTIVE_CONTROLLER_DECISION.md`
- Theory: `docs/thesis/theory/ADAPTIVE_BUDGETED_CONTROLLERS.md`,
  `docs/thesis/theory/THEORY_STATUS.md`
- Empirical anchors: `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md`,
  `docs/thesis/experiments/CROSS_BACKBONE_REPLICATION_RESULTS.md`
- Open questions: `research/open_questions.md`
- Candidate theorems: `research/candidate_theorems.md`

---

*End of reassessment. NO-GO on HPC. Phase 3b theory finalisation +
LaTeX writeup is the binding non-HPC critical path. Adaptive-controller
direction stays at appendix-active. The thesis object is unchanged.*

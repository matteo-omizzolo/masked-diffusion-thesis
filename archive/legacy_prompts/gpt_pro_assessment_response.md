# Response to GPT Pro Assessment (April 2026)

**Source:** `docs/instructions/v2/gpt_pro_assessment.md`,
`docs/instructions/v2/gpt_pro_theory_plan.md`,
`docs/instructions/v2/gpt_pro_experiment_design.md`

**Purpose:** Critically evaluate GPT Pro's findings and record which parts are accepted,
rejected, or deferred. This file is the audit record of the integration.

---

## Bottom-Line Assessment

GPT Pro's critique is largely sound and well-researched. Its central recommendation —
shift the main theorem from geometric contraction (Candidate 2) to a proxy-based
regret bound under approximate additivity (new Theorem A) — is a material improvement
over the earlier plan. The experiment design is detailed, practical, and aligned with
that theorem route.

The response below is organized accept / reject / defer, with explicit reasoning for
each item.

---

## Accepted Without Modification

### A1. Demote Candidate 2 (geometric contraction) to stretch goal

**GPT Pro claim:** Candidate 2's contraction model for masked-diffusion correctors
is too ambitious because:
- Ascolani et al. 2024 prove contraction for **random-scan Gibbs** under **strong
  log-concavity** (not systematic-scan as earlier notes stated);
- text distributions are not log-concave;
- the target distribution changes with predictor step.

**Assessment:** Correct. The earlier worklog overstated what is borrowable from
Ascolani et al. The mismatch (random-scan, log-concavity, stationarity) is large.

**Action:** Move Candidate 2 to stretch appendix status. Correct the Ascolani ref
in `proof_ledger.md`.

### A2. Correct Ascolani reference

**GPT Pro claim:** The proof ledger incorrectly states that Ascolani et al. prove
contraction for **systematic-scan** Gibbs.

**Assessment:** GPT Pro is right to flag this. The ledger should say **random-scan
Gibbs** (with extensions to Metropolis-within-Gibbs and Hit-and-Run). Tagged with
`[Incorrect as stated — to correct]`.

**Action:** Correct in `proof_ledger.md` immediately.

### A3. Replace Candidate 3 with a low-gain-region proposition

**GPT Pro claim:** The current Candidate 3 (burn-in gating via MI monotonicity) is
harder to prove than needed. A cleaner version is: "if early steps form a uniformly
low-gain region, schedules that spend budget there are weakly dominated."

**Assessment:** Correct. The MI monotonicity approach was attractive conceptually
but required proving a universal theorem about masked diffusion. The low-gain-region
formulation avoids the universality claim.

**Action:** Rewrite Candidate 3 as Proposition B.

### A4. Upgrade Q5 (additivity) from "follow-up limitation" to main modeling choice

**GPT Pro claim:** Additivity should not be a caveat; it should be the central
modeling approximation, surfaced via interaction bounds.

**Assessment:** Correct. Approximate additivity is cleaner and experimentally
testable.

**Action:** Rewrite Q5 in `open_questions.md` with upgraded framing.

### A5. Q1 (contraction) is not a blocker

**GPT Pro claim:** Q1's fate should not determine thesis viability. It only matters
if Candidate 2 remains the main theorem.

**Assessment:** Correct. Once the main theorem becomes proxy-regret, contraction
becomes a diagnostic/appendix, not a blocker.

**Action:** Downgrade Q1 in `open_questions.md`.

### A6. Q4 (ProSeCo) narrows but does not kill novelty

**GPT Pro claim:** ProSeCo exposes correction-loop hyperparameters and studies
budget allocation empirically, but does not implement a signal-adaptive trajectory
scheduler. The defensible novelty is narrower than "nobody studied when to correct"
— it is "signal-adaptive fixed-budget corrector allocation with a regret formulation."

**Assessment:** Correct. The reframing is sharper and more defensible.

**Action:** Update `open_questions.md` Q4 and `proof_ledger.md` novelty claim.

### A7. Experiment design (Protocol A + Protocol B)

**GPT Pro claim:** The main experiment should use:
- Protocol A (reference-available masked-reconstruction) for clean local correctness
  and low-variance marginal-gain measurement;
- Protocol B (unconditional generation) for final quality.

**Assessment:** Accepted. This is stronger than the earlier plan (which was mostly
Protocol B style).

**Action:** Adopt as the design in `docs/experiments/entropy_proxy_experiment.md`.

### A8. Variance control via common random numbers

**GPT Pro claim:** Use same continuation seed after branching, or deterministic
(argmax) continuation.

**Assessment:** Accepted. This materially reduces estimator variance.

**Action:** Specify in the experiment design.

### A9. Separate token-change rate from marginal gain

**GPT Pro claim:** TCR_t and Δ_t must not be collapsed into one scalar. High TCR
with low Δ means "loop is active but poorly targeted."

**Assessment:** Correct and important. This distinction is a key analysis point.

**Action:** Reflected in the experiment and scheduling code.

### A10. Interaction diagnostic for additivity

**GPT Pro claim:** Measure pairwise gains G({t1}), G({t2}), G({t1, t2}) for a small
set of pairs; report I(t1, t2) = G({t1, t2}) − Δ_t1 − Δ_t2.

**Assessment:** Accepted. This directly answers Q5 empirically.

**Action:** Include in Stage 4 of the experiment.

---

## Accepted With Modification

### M1. Main theorem shape — regret bound under approximate additivity

**GPT Pro's Theorem A:** Top-B by ψ(s_t) has regret ≤ 2Bε + 2η_B where ε = sup-norm
proxy error and η_B = additivity gap at size B.

**Assessment:** Accept the theorem shape. Modify by:
- Keeping Lemma A1 (oracle top-B under exact additivity) as a named lemma rather
  than absorbing it silently into Theorem A.
- Making the calibration map ψ explicit in the statement, with the option to soften
  sup-norm error to average-case or inversion-count error.
- Being explicit that this is a worst-case bound; the practical regret may be smaller.

**Action:** State Theorem A with Lemma A1, Lemma A2, and Proposition B/C as separate
named results. Write in `candidate_theorems.md`.

### M2. Primary platform: MDLM/ReMDM for Phase 1 (not ProSeCo)

**GPT Pro recommendation:** ProSeCo-OWT as primary platform.

**Assessment:** Mostly agree, but practical reality is that ProSeCo is not yet
cloned, while MDLM+ReMDM is already on HPC and patched. Phase 1 should start with
MDLM+ReMDM (where correction loops can be approximated by extending the standard
MDLM denoising step with an additional resample at the same noise level), with
ProSeCo becoming the primary platform once cloned.

**Action:** Reorder Phase 1 to use MDLM+ReMDM while ProSeCo setup proceeds in
parallel. Document this in `docs/experiments/implementation_status.md`.

**Reasoning:** GPT Pro is right that ProSeCo is the cleanest platform in principle,
but the thesis cannot block on cloning and validating a new codebase. MDLM's basic
masked-diffusion forward process supports a simple "one extra resample step" that
approximates a corrector loop well enough for the one-loop gain study. Results on
MDLM will transfer to ProSeCo naturally.

### M3. Experiment "quality mass" requires PRISM

**GPT Pro:** Quality mass is a Phase 2 / optional signal.

**Assessment:** Agree. PRISM integration is non-trivial (the quality head must
either be fine-tuned or downloaded, and PRISM's checkpoint format may differ).
Treat quality mass as Stage 3 work.

**Action:** Document in experiment design as "optional / Stage 3".

---

## Deferred / Defer for Zanella Meeting

### D1. Final theorem statement — exact vs approximate additivity version

**GPT Pro's recommendation:** Prove exact-additive Lemma A1/A2 first, then extend.

**Assessment:** Agree in principle. The order of proof is a good suggestion. But
the final thesis statement depends on how large η_B turns out to be empirically.
If η_B is small, the approximate version subsumes the exact version cleanly; if
η_B is large, the theorem must be stated more carefully.

**Action:** Include both versions in `candidate_theorems.md`. Defer final choice
until interaction diagnostic produces η_B estimates.

### D2. Strong-log-concavity stretch appendix

**GPT Pro:** Keep contraction as stretch only.

**Assessment:** Agree. But the exact form (stylized toy model? diminishing-returns
fit? geometric-only regime test?) can be deferred until the main theorem is done.

**Action:** Add a one-paragraph stub in `candidate_theorems.md` under "stretch
material."

---

## Not Accepted (or partially rejected)

### R1. Dropping Candidate 2 entirely

**GPT Pro's implication:** Candidate 2 is dead.

**Position:** I disagree with the strongest version. Candidate 2 should be
**demoted to stretch appendix**, not dropped. Even a stylized contraction result in
a simplified model (e.g., under a mean-field approximation, or for a toy absorbing
state) could be a valuable appendix contribution. It also provides vocabulary for
discussing "how much can one corrector loop achieve" in the limit.

**Action:** Keep Candidate 2 in `candidate_theorems.md` with explicit "stretch
appendix" status and a note that it depends on either:
- a toy model (e.g., two-state masked diffusion), or
- a diagnostic fit to empirical diminishing-returns data.

### R2. Treating the interaction error η_B as uniformly bounded

**GPT Pro's Theorem A:** |G(S) − Σ Δ_t| ≤ η_B for all |S| = B.

**Position:** This is cleaner to state but may be too strong in practice. Different
sets S of the same size may have very different interaction errors. A more realistic
version is average-case η_B over the random top-B selection. We should state Theorem
A with a sup-norm η_B (as GPT Pro suggests), but the proof should also note that an
expectation-version is likely easier to satisfy empirically.

**Action:** State the main theorem with sup-norm η_B. Add a remark about the
expectation version as a practically-weaker alternative.

---

## Summary Table

| Item | GPT Pro Recommendation | Decision | Action |
|------|-------------------------|----------|--------|
| Main theorem | Proxy-regret (Theorem A) | **Accepted** | Rewrite candidate_theorems.md |
| Candidate 2 (contraction) | Drop to stretch | **Accepted with modification** | Keep as stretch appendix, not dropped |
| Candidate 3 (burn-in) | Rewrite as low-gain prop | **Accepted** | Replace with Proposition B |
| Ascolani ref (random-scan) | Correct the citation | **Accepted** | Fix proof_ledger.md |
| Q1 (contraction blocker) | Downgrade | **Accepted** | Rewrite in open_questions.md |
| Q4 (ProSeCo novelty) | Narrow the claim | **Accepted** | Rewrite positioning |
| Q5 (additivity) | Upgrade to central | **Accepted** | Rewrite as main assumption |
| Primary experiment platform | ProSeCo-OWT | **Modified** | Start with MDLM+ReMDM; ProSeCo parallel |
| Protocol A + B | Both | **Accepted** | Full experiment design |
| Interaction diagnostic | Include | **Accepted** | Stage 4 of experiment |
| Sup-norm η_B | Clean form | **Accepted with remark** | State + add expectation alternative |
| Quality mass (PRISM) | Optional / Stage 3 | **Accepted** | Document as Stage 3 |

---

## Provenance Note

GPT Pro's contribution is substantial and will be tagged consistently throughout the
repo as `[Adapted from GPT Pro assessment v2]` wherever its framing or specific
claims are incorporated. This is not a paper-level citation — GPT Pro is an
assistant, not a source — but the traceability matters: the move from a contraction
theorem to a regret theorem is GPT Pro's framing, not my own, and the theory chapter
should acknowledge this chronology internally.

The final theorem statements, proofs, and experimental results remain my
responsibility as the author; GPT Pro's role was advisory and critical.

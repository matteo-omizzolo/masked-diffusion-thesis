# Open Questions

**Updated:** April 2026 (repriotized after GPT Pro v2 assessment)

> Unresolved technical points, fragile assumptions, and places where empirical
> evidence is needed to guide theory. The April 2026 restructure moved the
> main theorem from contraction to proxy-regret (Theorem A), which shifts
> priorities: the central uncertainty is now **calibration of the signal-to-gain
> relationship (Q5, Q2)**, not whether a Gibbs-style contraction bound holds.

---

## Q5 — Is the additive / approximate-additive gains assumption realistic? **[CENTRAL]**

**Promoted to central question after GPT Pro v2.**

**Context.** Theorem A assumes approximate additivity:

    sup_{|S| ≤ B} |G(S) − ∑_{t ∈ S} Δ_t| ≤ η_B.

The regret bound scales as 2Bε + 2η_B. If η_B is large relative to typical
G(Ŝ_B), Theorem A is vacuous.

**Failure modes.**
- Correcting at step t changes Z_t', which propagates through subsequent
  predictor steps. The gain from correcting at step t may depend on whether
  step t' was also corrected.
- Heavy-tailed interactions — a single pair (t, t') may dominate η_B.

**Action.**
1. **Protocol B primary measurement.** For a grid of B ∈ {T/16, T/8, T/4},
   measure ∑_{t ∈ S} Δ_t vs G(S) for randomly sampled |S| = B schedules.
2. **Pairwise diagnostic.** Estimate γ = sup |ξ_{t, t'}| from a sampled set of
   pairs; feed into Proposition C's bound η_B ≤ γ B² / 2.
3. **Decision.** If η_B is empirically small, Theorem A is non-vacuous and
   this thesis's main theorem stands. If η_B is large, consider:
   (a) reducing B to a small regime where the bound is tight,
   (b) reformulating as a sequential optimization (dynamic programming) with
       locally additive rewards plus state dependence,
   (c) using expectation-version bound with variance structure.

**Severity.** Critical — determines whether Theorem A's bound is useful.

**Dependencies.** Requires Phase 1 experimental infrastructure (MDLM + ReMDM
one-loop harness). See `docs/experiments/entropy_proxy_experiment.md`,
Protocol B.

---

## Q2 — Is entropy a good proxy for marginal gain? **[CENTRAL]**

**Context.** Theorem A's calibration hypothesis is
|Δ_t − ψ(s_t)| ≤ ε. The regret scales as 2Bε. The choice ψ = entropy-based
proxy is the baseline candidate; confidence margin and quality mass are alternatives.

**Sharpened from GPT Pro v2.** The earlier draft conflated the token-change
rate (TCR_t) with Δ_t. These must be measured and calibrated separately:
- TCR_t correlates with entropy by construction (high entropy → many possible
  resamples → more positions changed in expectation) but may be only weakly
  correlated with *quality* gain Δ_t.
- The proxy design question is: which ψ minimizes ε relative to Δ_t, not
  relative to TCR_t.

**Failure modes.**
1. **Early trajectory.** High entropy, low context; corrector is ineffective;
   entropy overpredicts Δ_t. (Handled by Proposition B's low-gain-region
   exclusion.)
2. **Late trajectory.** Low entropy, small residual uncertainty; but a few
   uncertain positions may carry large quality gain.
3. **Miscalibration.** Model entropy may not reflect true conditional
   uncertainty, especially on out-of-distribution trajectories.

**Action.**
- **Protocol A primary measurement.** For each t, measure one-loop Δ_t under
  a chosen F. Correlate with s_t across three choices:
  (a) mean conditional entropy H_t,
  (b) inverse confidence margin (1 − M_t),
  (c) quality mass Q_t (PRISM-style).
- Report Pearson, Spearman, and a calibration curve; extract ε per choice.

**Severity.** Critical — the whole thesis turns on whether *some* trajectory
signal gives ε small enough to beat uniform.

**Dependencies.** Protocol A of the entropy-proxy experiment. PRISM
availability gates quality-mass estimation.

---

## Q6 — What is the right formal definition of "corrector loop"? **[RESOLVED for Phase 1]**

**Context.** Different papers define correctors differently:
- Zhao et al.: Barker / MPF Metropolis-Hastings steps
- ProSeCo: annealed iterative refinement of committed tokens
- ReMDM: remasking + re-prediction (debatably a corrector vs. a predictor-
  schedule extension)
- Gibbs sampling: resample one or more positions from the conditional

**Resolution (April 2026 — ProSeCo adoption).** The Phase 1 corrector is
definitively the ProSeCo annealed refinement kernel:

> **A corrector loop at step t** = take x̂_0 = argmax(p_θ(x_0|x_t)); run
> `corrector_steps=2` backbone calls at decreasing noise levels τ ∈ {1.0, 0.5};
> apply the result to unmasked positions R_t only. One loop costs 2 NFEs.

This definition was chosen over the MDLM Gibbs-resample corrector because the
MDLM heuristic was harmful at all steps (all Δ_t ≤ 0) — an unrecoverable
failure mode that makes Theorem A vacuous by construction. The ProSeCo kernel
refines committed tokens rather than resampling masked tokens, providing a
principled mechanism with non-trivially positive expected Δ_t.

**Theorem A remains corrector-kernel agnostic in its statement.** The ProSeCo
kernel is the empirical instantiation used to measure ε, η_B, γ.

**What remains open.** Whether a second kernel (e.g., Zhao et al.'s Barker
kernel or conf-guided partial resample) gives smaller ε or η_B — left as a
thesis extension if Phase 1 results are strong.

**Severity.** Resolved for Phase 1. Kernel comparison remains open as extension.

---

## Q4 — Does ProSeCo already subsume the thesis direction? **[RESOLVED — novelty confirmed]**

**Context.** ProSeCo (arXiv:2602.11590) is the closest existing
corrector-scheduling paper. It exposes scheduling knobs and empirically studies
corrector-budget allocation.

**Resolution (April 2026 — ProSeCo audit complete).** ProSeCo has been read and
audited in detail (`docs/experiments/proseco_backend_audit.md`). Confirmed:

1. ProSeCo uses a fixed periodic corrector schedule (every `corrector_every_n_steps`
   steps, starting from `corrector_start_iter`). It does **not** use trajectory
   signals to decide *when* to apply the corrector.
2. ProSeCo does not define or measure Δ_t (per-step marginal gain).
3. ProSeCo does not provide any proxy-regret theorem, ε bound, or η_B estimate.
4. ProSeCo's schedule comparison is empirical (fixed periodic vs. no corrector);
   it does not search over signal-adaptive schedules.

**The thesis uses ProSeCo as its corrector backend, not as a competitor.**
This is strictly cleaner than the earlier framing. The thesis says:

> Given the ProSeCo corrector kernel, we provide a theoretical proxy-regret
> framework (Theorem A) and empirically estimate ε, η_B, γ that ProSeCo's
> own analysis does not provide.

**What would still invalidate the thesis.** If a concurrent preprint provides
Theorem A's proxy-regret bound with empirical ε/η_B estimates for masked diffusion
corrector scheduling. As of April 2026, no such preprint is known.

**Severity.** Resolved. No further action needed unless a new preprint appears.

---

## Q1 — Does geometric contraction hold for masked diffusion correctors? **[DOWNGRADED]**

**Downgraded from critical to medium after GPT Pro v2.**

**Context.** Stretch Appendix C2 (formerly Candidate Theorem 2) models
corrector contraction as E_fact(t) · ρ(t)^{k_t}. The question is whether this
geometric model holds for real masked-diffusion correctors.

**Why downgraded.** Theorem A (proxy-regret) is now the main theorem and does
not require geometric contraction. The contraction route is preserved as a
stretch result for an eventual appendix.

**Why it might hold.** A discrete-MCMC contraction framework (not the Ascolani
et al. log-concave-systematic-scan version, which does not apply; but a more
general coupling / Dobrushin-style bound) might yield per-loop contraction
for the corrector kernel.

**Why it might fail.** Text distributions are discrete, high-dimensional, and
not log-concave. The corrector targets the conditional posterior at a
particular noise level, which changes across steps.

**Action.**
- Read Ascolani et al. 2024 carefully (correct scan type: random-scan).
- Read the Denoising Entropy Bounds paper.
- Empirically test: run k ∈ {0, 1, 2, 4, 8} loops at fixed t and check for
  approximate exponential decay in error. (This is Q7 below.)
- Pursue the contraction theorem only if both the literature and the empirics
  support it.

**Severity.** Medium — affects only Stretch Appendix C2.

---

## Q3 — How sensitive is the optimal schedule to the total budget B?

**Context.** Under Theorem A, the regret 2Bε + 2η_B scales with B. If η_B
grows faster than linearly in B, the useful range of B is bounded.

**Implication.** Test multiple budget levels (B ∈ {T/16, T/8, T/4, T/2}) and
report Theorem A's bound at each B. Expect a sweet spot where 2Bε + 2η_B is
small relative to G(Ŝ_B).

**Severity.** Medium — an empirical Pareto question rather than a theoretical
obstacle.

---

## Q7 — Can the contraction factor ρ(t) be estimated empirically?

**Context.** Only relevant if Stretch Appendix C2 is pursued. The
empirical diagnostic is the same as the "approximate exponential decay"
check in Q1.

**Action.** Design a diagnostic experiment: for each step t in a sampled
subset, run k ∈ {0, 1, 2, 4, 8} corrector loops and measure per-loop error
reduction. Check whether log(error) decays linearly in k. Fit ρ(t) and see
whether it correlates with any aggregate signal.

**Severity.** Low — useful diagnostic regardless of the theorem; only
critical if the contraction route becomes the main theorem.

---

## Q8 — Is the TCR ≠ Δ_t distinction empirically significant? **[NEW]**

**Raised by GPT Pro v2.**

**Context.** The token-change rate (TCR_t — fraction of positions that a single
corrector loop actually changes) and the quality gain (Δ_t — change in a
trajectory quality functional F) are distinct quantities. The previous draft
sometimes conflated them. Confusing the two would systematically mis-calibrate
the proxy.

**Action.**
- Protocol A measures both TCR_t and Δ_t at each t.
- Report correlation between TCR_t and Δ_t; if weak, the distinction is
  material and Δ_t should be used exclusively for calibration.
- If TCR_t and Δ_t are strongly correlated, TCR_t may be usable as a
  cheap surrogate for Δ_t (though this weakens the novelty contribution).

**Severity.** High — affects the validity of Protocol A's ε estimate.

---

## Q9 — What F should define Δ_t? **[NEW]**

**Raised by GPT Pro v2.**

**Context.** Δ_t := F(y_t^{+1}) − F(y_base) requires choosing F. Candidates:
- Negative LM-NLL under a reference language model (cheap, per-trajectory)
- MAUVE (computed over a pool; not per-trajectory)
- PRISM quality score (requires PRISM checkpoint)
- Perplexity under a reference scorer

**Trade-off.** Per-trajectory quantities are ideal for Δ_t but may be noisy;
pool-level quantities (MAUVE) are stable but cannot isolate per-step effects.

**Proposed.** Primary F = negative LM-NLL; sanity-check with pool-level MAUVE
at the end of the trajectory.

**Severity.** Medium — affects numerical magnitudes of ε, η_B but not the
structure of the theorem. **Phase 2c may swap to MAUVE/F-pool** if Phase 2b
shows uniform-vs-signal margins are within F-noise band.

---

## Q10 — Does √B-scaling variance bound on η_B fit Phase 1 better than Prop C's B² bound? **[NEW — post-audit]**

**Raised by 2026-04-19 stress test (`THEORY_STRESS_TEST.md` §6, §10.1).**

**Context.** Proposition C predicts η_B ≤ γ·B(B−1)/2. On Phase 1 ProSeCo-OWT this
gives η(B=8) ≤ 7.4 vs measured η_95(B=8) = 0.68 — a ≈11× looseness. The variance-
form Refinement A′ predicts 𝔼|G−A| ≈ σ_ξ·B/√2, which is consistent with the
linear-in-B scaling of the measurements.

**Action.**
1. Compute σ_ξ (std of pairwise interactions) on Phase 1 data; check whether
   σ_ξ·B/√2 fits the (η_95(B=4), η_95(B=8), η_95(B=16)) sequence to within 30%.
2. If yes, write up Refinement A′ as a thesis chapter result; replace Prop C in
   the main statement of Theorem A.
3. If no, investigate higher-order interactions (triple diagnostic).

**Severity.** High — determines whether Theorem A can be made non-vacuous via a
sharper additivity bound rather than a smaller (backbone, corrector, F).

**Dependencies.** None new — uses existing Phase 1 protocol_b data.

---

## Q11 — Is rank-based ε_R the right calibration quantity for Top-B selection? **[NEW — post-audit]**

**Raised by 2026-04-19 stress test (`THEORY_STRESS_TEST.md` §4, §10.2).**

**Context.** On Phase 1 ProSeCo-OWT, ε_rms ≈ 0.134 looks small only because the
proxy is nearly flat (Spearman ≈ −0.19). Theorem A's uniform-ε bound cannot
distinguish "informative-low-ε" from "uninformative-low-ε". Refinement A″
proposes ε_R := (1 − |ρ|)·σ_Δ as a rank-sensitive substitute.

**Action.**
1. Recompute ε_R for each signal on Phase 1 data; report alongside ε_rms.
2. Phase 2a (offline reanalysis) should report A↔G rank correlation per B,
   which directly tests whether rank-ordering of A approximates rank-ordering
   of G — the operational claim Refinement A″ rests on.
3. If ε_R is materially different from ε_rms across signals, propose to swap
   the calibration object in Theorem A's Assumption 3 to ε_R.

**Severity.** High — determines whether Theorem A's calibration assumption is
the right one for Top-B selection.

**Dependencies.** None new — uses existing Phase 1 protocol_a data plus Phase 2a
A↔G analysis.

---

## Q12 — What is the true G(S_B*)? **[NEW — post-audit]**

**Raised by 2026-04-19 stress test (`THEORY_STRESS_TEST.md` §8).**

**Context.** Theorem A references S_B* := argmax_{|S|=B} G(S), but the experiment
never computes this. `policy_comparison.oracle` is mean-field top-B of Δ̄_t, which
is S_A*, not S_B*. We cannot honestly report "proxy regret vs oracle" because we
don't know what the oracle is.

**Action.**
- For B = 4 at T = 64, all C(64, 4) ≈ 635k subsets are enumerable on a single
  trajectory — feasible on ≈10 trajectories.
- For B = 8, Monte-Carlo 300–1000 random schedules per trajectory and report
  the maximum G observed (a lower bound on G(S_B*)).
- Phase 2b includes mc_B_values = [2, 3, 4] for exactly this purpose.

**Severity.** Critical — without it, no proxy-regret statement can honestly
appear in the thesis.

**Dependencies.** Phase 2b paired sweep.

---

## Priority Ordering (April 2026, post-audit)

1. **Q5** (approximate additivity / η_B) — **Central**; Phase 2b plus offline
   reanalysis.
2. **Q2** (signal as proxy / ε) — **Central**; Phase 2a (ε_R) + Phase 2b paired.
3. **Q12** (true S_B*) — **Critical**; Phase 2b Monte-Carlo upper-envelope.
4. **Q6** (corrector definition) — **Central** framing; settled for Phase 1;
   may revisit if Phase 2b says swap is warranted.
5. **Q11** (ε_R as right calibration) — High; Phase 2a deliverable.
6. **Q10** (variance vs Prop C bound) — High; offline check then write-up.
7. **Q8** (TCR vs Δ_t) — High; Protocol A.
8. **Q4** (ProSeCo novelty) — Resolved; ProSeCo is the backend, not a competitor.
9. **Q9** (choice of F) — Medium; Phase 2c MAUVE-swap pending Phase 2b.
10. **Q3** (budget sensitivity) — Medium; Phase 2b sweep covers B ∈ {2,3,4,8,16}.
11. **Q1** (contraction) — Medium; only for Stretch C2.
12. **Q7** (empirical ρ) — Low; diagnostic only.

All questions cross-reference `docs/experiments/entropy_proxy_experiment.md`
and `docs/gpt_pro_assessment_response.md`. Q10–Q12 originate from
`docs/thesis/theory/THEORY_STRESS_TEST.md` and
`docs/thesis/experiments/EXPERIMENT_CRITICAL_AUDIT.md`.

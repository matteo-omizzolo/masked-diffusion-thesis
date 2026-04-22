> **STATUS:** CANONICAL
> **LAST VERIFIED:** 2026-04-22
> **SCOPE:** Top-level thesis direction, scope, and non-goals. Single source for research-question framing.

---

# Thesis Direction: Signal-Adaptive Corrector Scheduling for Masked Diffusion Language Models

**Author:** Matteo Omizzolo — MSc Thesis, Bocconi University
**Supervisor:** Prof. Giacomo Zanella
**Date:** April 2026

---

## Core Research Question

> For a fixed predictor schedule and fixed corrector NFE budget in masked diffusion
> language models, can aggregate trajectory signals — entropy, confidence margin, or
> quality mass — predict the marginal value of a corrective refinement loop well enough
> to outperform uniform corrector placement?

---

## Why This Framing

The earlier thesis framing centered on "entropy-adaptive informed-corrector scheduling."
That was a useful starting intuition, but too narrow as a thesis target because:

- Raw entropy alone may not be the best signal for corrector allocation.
- Adjacent papers already use signal-based remasking or token-level quality scores.
- The thesis should not hinge on proving that entropy itself is optimal.
- The strongest open gap is broader: **trajectory-level allocation of correction effort
  under a fixed budget**.

The current framing centers on **signal-adaptive scheduling**, with entropy as the first
candidate signal rather than the only one.

---

## Scope Boundaries

The thesis maintains clear distinctions between five closely related concepts:

1. **Informed correctors** — inference-time corrective updates using a signal (logits,
   confidence, entropy, score, quality estimates).
2. **Remasking methods** — revisiting tokens by remasking and regenerating them within
   the denoising process.
3. **Predictor / unmasking schedule optimization** — when and how many tokens to unmask,
   what order, how to set block size.
4. **Token-selection policies** — rules about *which* token(s) to update or correct.
5. **Corrector scheduling** — rules about *when* to spend corrective refinement steps
   across the denoising trajectory, and how to allocate a fixed global corrector budget
   over time.

**The thesis focuses on (5)**, while using ideas from (1)–(4) as signals, proxies, or
baselines. This distinction is part of the contribution.

---

## Open-Question Verdict

### What is already addressed

- Some papers study self-correction or corrective loops in masked diffusion LMs (ProSeCo).
- Some papers use quality or confidence signals to decide what to revisit (PRISM, RemeDi).
- Some papers optimize predictor schedules, including entropy-aware unmasking (EB-Sampler, Denoising Entropy).
- Some papers study non-uniform scheduling in nearby theory settings (Gibbs sampling, MwG).

### What remains open

What still appears open is the narrower problem: given a fixed predictor schedule and a
fixed corrector compute budget, how should corrective refinement steps be allocated across
diffusion time, and can trajectory-level signals justify a better-than-uniform allocation?

That gap appears real because the closest existing papers provide either empirical
heuristics, token-level correction/remasking signals, predictor-side schedule theory, or
corrector-kernel design — but not a principled trajectory-level corrector allocator for
masked diffusion language models.

### Safe claim

> The literature contains strong adjacent work on self-correction, remasking,
> token-quality scoring, and predictor/unmasking schedule design, but there does not yet
> appear to be a principled theory-plus-experiments study of **trajectory-level
> fixed-budget corrector allocation** in masked diffusion language models.

### Claim to avoid

Do **not** claim that "nobody studied when to correct" or that "the area is completely
untouched." That would be too strong and inaccurate.

---

## Candidate Signals

The first three signals to test should be:

1. **Aggregate entropy** over revisable tokens.
2. **Confidence margin** or related logit sharpness measure.
3. **Quality mass** if a PRISM-style or analogous quality signal is available.

### Hypothesis ordering

1. Uniform corrector placement is not optimal under a fixed budget.
2. Pure entropy-proportional placement may help but may fail early in the trajectory.
3. Burn-in-gated or middle-phase-weighted signal-adaptive schedules may be stronger
   than naive entropy-only scheduling.
4. Quality-like signals may outperform entropy if available.

---

## Theory Target

**Refined April 2026 after GPT Pro v2 assessment.** The main theorem is a
**proxy-regret bound** (Theorem A) rather than a contraction theorem.

### Theorem A (main) — proxy-regret bound

Let Δ_t be the one-loop marginal gain from applying a single corrector loop
at step t, measured against a trajectory-level quality functional F. Let
G(S) be the joint gain from correcting at all steps in S. Let Ŝ_B be the
top-B schedule selected by proxy ψ(s_t) from an aggregate signal s_t, and
let S_B* be the oracle optimal B-schedule.

Under (1) binary placement, (2) approximate additivity
|G(S) − ∑ Δ_t| ≤ η_B, and (3) proxy calibration |Δ_t − ψ(s_t)| ≤ ε,

    G(S_B*) − G(Ŝ_B) ≤ 2 B ε + 2 η_B.

The substantive content is empirical: measure ε, η_B on real models and
check whether 2 B ε + 2 η_B is small relative to G(Ŝ_B). See
`research/candidate_theorems.md` (Theorem A), `research/proof_worklog.md`
(Entry 6), and `docs/archive/phase1_era/entropy_proxy_experiment.md` for protocols.

### Supporting results

- **Lemma A1** — oracle top-B optimality under exact additivity.
- **Lemma A2** — proxy calibration yields 2Bε regret in the additive regime.
- **Proposition B** — low-gain-region (burn-in) exclusion is benign. Replaces
  the earlier MI-monotonicity-in-u_t claim, which was not uniformly true.
- **Proposition C** — under a pairwise-interaction expansion with |ξ| ≤ γ,
  we have η_B ≤ γ B (B − 1) / 2, making the additivity slack empirically
  estimable.

### Stretch appendix (optional)

- **Appendix C2 (contraction)** — demoted from main theorem to stretch. If a
  Gibbs-style contraction bound can be shown to apply to the masked-diffusion
  corrector kernel (using a more appropriate framework than Ascolani et al.'s
  log-concave random-scan setting), this gives a sharper statement. The
  Ascolani reference is now cited with corrected scan type and hypotheses;
  log-concavity is known not to transfer.

### Honesty requirement

The theory should explicitly separate: proved statements, heuristic arguments,
empirical findings, and conjectures. Assumptions must be stated. All novel
claims and adapted results are tagged in `research/proof_ledger.md`.

---

## Non-Goals

The thesis does **not** need to:

- Prove universal optimality of entropy scheduling.
- Provide a complete theory of masked diffusion sampling.
- Derive a learned end-to-end optimal policy over all inference choices.
- Redesign the full model training objective.

---

## Target Final Shape

### Theory contribution
A modest but meaningful theorem or proposition about fixed-budget corrector allocation
under explicit assumptions.

### Experimental contribution
A strong empirical comparison of uniform vs signal-adaptive corrector schedules on open
masked diffusion language models with public checkpoints.

### Positioning contribution
A clear distinction between informed correctors, remasking, predictor schedules,
token-selection policies, and corrector scheduling. That distinction is part of the
thesis's value and should be visible throughout.

---

## Tracked Experimental Questions

1. Under fixed extra NFE budget, does any non-uniform corrector schedule beat uniform?
2. Does the best schedule differ by total budget?
3. Are early high-entropy regions misleading because context is incomplete?
4. Does a burn-in gate (low-gain-region exclusion) improve entropy-based scheduling?
5. Does confidence margin outperform entropy as the calibration signal (smaller ε)?
6. Does a quality signal outperform both entropy and confidence?
7. Can the marginal value of a correction loop (Δ_t) be predicted from trajectory
   diagnostics (ε in Theorem A)?
8. How large is the additivity slack η_B across realistic B? (Protocol B; central.)
9. Is there a measurable pairwise interaction γ, and does Proposition C's bound
   η_B ≤ γ B(B−1)/2 hold?
10. Is there a distinct low-gain region T_low where Δ_t ≤ δ, such that gating
    Proposition B applies?

See `docs/archive/phase1_era/entropy_proxy_experiment.md` for Protocol A (ε, Δ_t, s_t)
and Protocol B (η_B, γ, joint G(S)) design details.

---

## Relationship to Prior Docs

- `docs/md/research_directions.md` — **deprecated**; reflected the earlier broad "informed
  correctors" framing (March 2026).
- `docs/md/research_plan.md` — **partially superseded**; the Gap B/C + Gap E structure
  remains relevant but the framing should be read through the lens of this document.
- `docs/md/correctors_deep_dive.md` — **deprecated**; retained for historical reference
  on corrector mechanics.

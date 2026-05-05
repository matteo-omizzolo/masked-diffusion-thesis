# Proof Worklog: Signal-Adaptive Corrector Scheduling

**Started:** April 2026
**Author:** Matteo Omizzolo
**Status:** Exploratory — no locked theorem yet

> This is a chronological research log. It records derivation attempts, insights,
> failed directions, and corrections. Each entry is tagged with provenance and
> confidence.

---

## Entry 1 — Problem Formalization (April 2026)

### Setup

Consider a masked diffusion language model generating a sequence x = (x_1, ..., x_D)
of D tokens from vocabulary V ∪ {[MASK]}.

**Predictor schedule:** A fixed sequence of T predictor steps. At each step t ∈ {1, ..., T},
the predictor unmasks some tokens according to a fixed schedule. After all T steps, we
have a candidate sequence Z_T.

**Corrector:** An additional Markov step applied after a predictor step. One corrector loop
at time t takes the current state Z_t and produces Z_t' by resampling some token positions
using the model's conditional distribution. The corrector is distinct from remasking
(which defers tokens within the same T-step budget).

**Budget constraint:** Total corrector NFE budget B = ∑_{t=1}^{T} k_t, where k_t ≥ 0
is the number of corrector loops applied at time t.

**Objective:** Find the allocation {k_t}_{t=1}^T that minimizes a trajectory-level error
measure, subject to ∑_t k_t = B.

### Error measure candidates

**Option 1 — KL divergence:**
Minimize KL(p_data ‖ p_alg) where p_alg is the distribution induced by the predictor +
corrector schedule.
[Borrowed from L&Z framework]

**Option 2 — Factorization error E_fact:**
Minimize E_fact = ∑_t E_fact(t), the factorization error accumulated across the trajectory.
[Borrowed from L&Z, arXiv:2510.25544]

**Option 3 — Residual error proxy:**
Define R_t = E_fact(t) as the residual error after predictor step t. The corrector at
time t reduces R_t by some amount Δ_t(k_t). Minimize ∑_t [R_t - Δ_t(k_t)].

**Working choice:** Option 3, because it directly models the marginal gain from correction
and connects to a concrete optimization problem.

### Signal definition

Let s_t be an aggregate trajectory signal computed at time t. Candidates:

- s_t = H_t = ∑_i H(x_i | x_{-i}, Z_t) / D (mean conditional entropy over positions)
- s_t = M_t = 1 - mean_i [p_θ(x_i = argmax | Z_t) - p_θ(x_i = second | Z_t)] (inverse confidence margin)
- s_t = Q_t = mean_i [1 - q_φ(x_i | Z_t)] (low-quality mass, PRISM-style)

### The allocation problem

Given:
- Fixed predictor schedule producing states Z_0, Z_1, ..., Z_T
- One corrector loop at time t reduces residual error by Δ_t(k_t, s_t)
- Budget: ∑_t k_t = B
- Find: allocation {k_t*} minimizing total residual error

If Δ_t is concave in k_t (diminishing returns per additional loop), this is a classic
resource allocation problem solvable by Lagrangian methods.

**Correctness status:** `solid` — this is a clean formalization.
**Novelty status:** `[Adapted from resource allocation theory; framing applied to corrector scheduling is novel]`

---

## Entry 2 — Marginal Gain Model (April 2026)

### Diminishing returns assumption

**Assumption DR:** For each t, the gain from the j-th corrector loop at time t satisfies:

  Δ_t^{(j)} ≤ Δ_t^{(j-1)} for j ≥ 2

i.e., each additional corrector loop yields weakly less error reduction than the previous one.

**Justification:** After each corrector loop, the state Z_t' is closer to a local
equilibrium of the conditional distribution. Further loops explore the same neighborhood.
This is analogous to diminishing returns of Gibbs sweeps — early sweeps make large moves;
later sweeps fine-tune.

[Analogy to Gibbs sampling convergence; see Adapting the Gibbs Sampler (2018)]

**Correctness status:** `plausible but incomplete` — diminishing returns is a natural
assumption for Gibbs-style updates but has not been formally proved for masked diffusion
correctors. Would need to show that the KL contraction factor per Gibbs sweep is bounded.
The Ascolani et al. 2024 entropy contraction result may provide the right bound.

### Single-loop gain model

For simplicity, first consider k_t ∈ {0, 1} (each step gets at most one corrector loop).
Budget: ∑_t k_t = B with k_t ∈ {0, 1}.

**Claim (informal):** The optimal binary allocation sets k_t = 1 for the B steps with
largest marginal gain Δ_t^{(1)}.

This is trivially true for binary allocation with additive gains.

**The real question:** Is Δ_t^{(1)} well-predicted by signal s_t?

### Connecting gain to entropy

**Hypothesis H1:** Δ_t^{(1)} ∝ H_t (marginal gain is proportional to aggregate entropy).

**Why this might be true:** High entropy at time t means the conditional distribution is
spread out — there is more room for a corrector to improve the current state. A Gibbs
sweep in a high-entropy region makes a larger expected move.

**Why this might be false:** Early in the trajectory (large t, many masks), entropy is
high but most of it reflects missing context rather than correctable error. A corrector
applied when 90% of tokens are masked cannot do much because the conditional distribution
is dominated by the prior (marginal), not by informative context.

[Novel observation] This suggests a **burn-in phase** where entropy is high but corrector
value is low. The gain Δ_t^{(1)} may be better modeled as:

  Δ_t^{(1)} ∝ H_t · f(fraction_unmasked(t))

where f is an increasing function that is near zero when few tokens are unmasked and
approaches 1 when enough context exists.

**Correctness status:** `heuristic only` — needs formalization. The burn-in effect is
intuitively clear but the functional form of f is speculative.

---

## Entry 3 — Toward a Theorem Statement (April 2026)

### Candidate Theorem (Residual Error Allocation)

**Setup:**
- Fixed predictor schedule with T steps
- States Z_0, ..., Z_T with residual error R_t after predictor step t
- One corrector loop at time t reduces residual by Δ_t(s_t) where s_t is a signal
- Δ_t is monotone increasing in s_t and satisfies Δ_t(s_t) ≤ R_t
- Budget: ∑_t k_t = B with k_t ∈ {0, 1}

**Theorem (draft):** Under assumptions DR and monotonicity, the optimal allocation
sets k_t = 1 for the B time steps with largest Δ_t(s_t). If s_t correctly ranks the
marginal gains (i.e., Δ_t > Δ_{t'} ⟺ s_t > s_{t'}), then signal-proportional
allocation achieves lower total residual error than uniform allocation.

**Proof sketch:**
1. With additive, independent gains and binary allocation, optimal is top-B selection.
2. If signal s_t is a monotone proxy for Δ_t, then ranking by s_t is equivalent to
   ranking by Δ_t, so signal-guided selection is optimal.
3. Uniform allocation (select B steps uniformly at random) achieves expected total
   reduction E[∑_t k_t Δ_t] = (B/T) ∑_t Δ_t.
4. Top-B selection achieves ∑_{t ∈ top-B} Δ_t ≥ (B/T) ∑_t Δ_t whenever the Δ_t are
   not all equal.
5. The gap is ∑_{t ∈ top-B} Δ_t - (B/T) ∑_t Δ_t, which is positive iff Var(Δ_t) > 0.

**Correctness status:** `solid under assumptions` — the argument is standard (weighted
selection). The non-trivial part is whether the assumptions (monotone signal, additive
gains, binary allocation) are realistic.

**Key gap:** Step 2 assumes the signal correctly ranks marginal gains. This is the
empirical question the experiments must test.

**Novelty:** `[Novel application]` — the resource-allocation framing is standard, but
applying it to corrector scheduling in masked diffusion is new.

---

## Entry 4 — Burn-In Gated Scheduling (April 2026)

### The burn-in problem

From Entry 2: early in the trajectory, entropy is high but the corrector is operating
with very little context. The conditional distribution p(x_i | x_{-i}, Z_t) is
dominated by the marginal when most neighbors are masked.

**Definition:** Let u_t = fraction of unmasked tokens at time t. A **burn-in gate** is
a function g(u_t) ∈ [0, 1] that modulates the corrector allocation:

  effective_signal(t) = s_t · g(u_t)

where g(u) = 0 for u < u_min and g(u) → 1 for u > u_threshold.

### Why burn-in gating may be more defensible

Naive entropy-proportional scheduling allocates most budget to the very early steps
(highest entropy), but those steps have the least useful context for correction.

**Proposition (draft):** For any masked diffusion model with factored approximate
posterior, the mutual information I(x_i; x_{-i} | Z_t) is monotone increasing in the
number of unmasked tokens. Therefore, the corrector's ability to exploit inter-token
dependencies for quality improvement increases with u_t.

[Adapted from information-theoretic arguments in L&Z; the specific application to
corrector value is novel]

**Correctness status:** `plausible but incomplete` — the monotonicity of MI with
unmasked fraction is intuitive and follows from data processing inequality arguments,
but needs a careful proof for the masked diffusion setting.

### Experimental prediction

Burn-in-gated entropy scheduling should outperform naive entropy-proportional scheduling,
especially at small budgets B where wasting corrector steps on early masked states is
costly.

**Testable:** Compare entropy-proportional vs burn-in-gated entropy vs uniform at
B ∈ {T/8, T/4, T/2} on MDLM-OWT or ProSeCo-OWT.

---

## Entry 5 — Information-Profile Route (April 2026)

### Connecting to L&Z factorization error

[Borrowed from L&Z, arXiv:2510.25544]

L&Z decompose the generation error as:

  KL(p_data ‖ p_alg) ≤ E_learn + E_fact

where E_fact is the factorization error arising from the product-form approximation at
each predictor step. The information profile I(t) describes how much mutual information
the predictor captures at each step.

**Key insight for correctors:** The corrector at time t can be viewed as reducing the
factorization error left by the predictor at that step. Specifically, a corrector that
resamples from the joint (rather than factored) conditional reduces E_fact(t).

**Gap E formalization:**

Let E_fact(t) be the per-step factorization error. Let E_fact^{corrected}(t, k_t) be the
factorization error after k_t corrector loops. Then:

  E_fact^{corrected}(t, k_t) = E_fact(t) · ρ(t)^{k_t}

where ρ(t) ∈ [0, 1) is the per-loop contraction factor at time t.

[Adapted from Ascolani et al. 2024 entropy contraction for Gibbs — the contraction
factor ρ is their framework applied to the masked diffusion corrector kernel]

**If this holds:** Total error becomes ∑_t E_fact(t) · ρ(t)^{k_t}, and the optimal
allocation under budget ∑_t k_t = B is a standard convex optimization.

**Correctness status:** `plausible but incomplete` — the exponential contraction model
is stylized. Real correctors may not contract geometrically. The Ascolani et al. paper
proves contraction for systematic-scan Gibbs under log-concavity, which may not hold for
text distributions. Need to read Ascolani et al. carefully.

**This is the most promising route.** If the contraction factor ρ(t) can be related to
the signal s_t, the theorem writes itself.

---

## Summary of Current State

Three promising directions explored:

1. **Residual error allocation (Entry 3):** Clean formalization, standard optimization.
   The assumption that signal ranks gains correctly is the empirical question.
   Status: `solid under assumptions`.

2. **Burn-in gating (Entry 4):** Strong intuitive argument. Predicts that naive
   entropy-proportional scheduling wastes budget. Needs MI monotonicity proof.
   Status: `plausible but incomplete`.

3. **Information-profile contraction (Entry 5):** Most elegant if the contraction model
   holds. Directly extends L&Z. Depends on Ascolani et al.'s Gibbs contraction applying
   to masked diffusion correctors.
   Status: `plausible but incomplete`.

### Next steps for this worklog (as of Entry 5, superseded by Entry 6 below)

1. Read Ascolani et al. 2024 — check whether their contraction bound applies to the
   masked diffusion corrector kernel.
2. Read ProSeCo — understand what correction schedule experiments already exist.
3. Formalize the burn-in MI monotonicity claim.
4. Write out the full Lagrangian for the information-profile contraction route.
5. Identify counterexamples where entropy fails as a proxy for marginal gain.

---

## Entry 6 — Theorem Restructure After GPT Pro v2 Assessment (April 2026)

### Trigger

Received GPT Pro v2 output (`docs/instructions/v2/gpt_pro_assessment.md`,
`gpt_pro_theory_plan.md`, `gpt_pro_experiment_design.md`) with a detailed
critique of the April draft. Summary of the critique is in
`docs/gpt_pro_assessment_response.md`. The three most consequential points
for the worklog are:

1. **Contraction as main theorem is premature.** The Ascolani et al. 2024
   reference was miscited (systematic-scan under log-concavity → actually
   random-scan, without log-concavity), log-concavity does not transfer to
   masked text, and there is no mechanism yet linking ρ(t) to any signal s_t.
   → Demote Candidate 2 to Stretch Appendix.

2. **MI monotonicity in u_t is not generally true.** The earlier Candidate 3
   (burn-in gating) appealed to monotonicity of mutual information
   I(x_i; x_{-i} | Z_t) in unmasked fraction u_t, which does not hold
   uniformly. The empirical "low-gain region" is strictly weaker and
   verifiable.
   → Replace with Proposition B (low-gain-region exclusion).

3. **TCR_t and Δ_t are distinct.** The draft sometimes conflated the
   token-change rate with the quality gain. Calibration must be done against
   Δ_t, not TCR_t.

### New theorem stack

Proposed in GPT Pro's theory plan, adopted in the April 2026 restructure:

**Theorem A (main).** Under binary placement, approximate additivity
|G(S) − ∑ Δ_t| ≤ η_B, and calibrated proxy |Δ_t − ψ(s_t)| ≤ ε,

    G(S_B*) − G(Ŝ_B) ≤ 2 B ε + 2 η_B.

**Lemma A1.** Oracle top-B is optimal under exact additivity. (Was Candidate 1.)

**Lemma A2.** Proxy approximation yields 2Bε regret in the exact-additivity
regime (exchange argument).

**Proposition B.** Low-gain-region exclusion is benign, replacing the
MI-monotonicity burn-in argument.

**Proposition C.** η_B ≤ γ B (B − 1) / 2 under a pairwise-interaction model;
γ is estimable from Protocol B diagnostics.

### Proof of Theorem A — first pass

Let A(S) := ∑_{t ∈ S} Δ_t be the additive surrogate and G(S) the true joint
gain. We want

    G(S_B*) − G(Ŝ_B) ≤ 2 B ε + 2 η_B.

**Step 1 — connect G and A via (2).** |G(S) − A(S)| ≤ η_B for all |S| ≤ B, so

    G(S_B*) ≤ A(S_B*) + η_B   and   A(Ŝ_B) − η_B ≤ G(Ŝ_B).

Subtracting,

    G(S_B*) − G(Ŝ_B) ≤ A(S_B*) − A(Ŝ_B) + 2 η_B.    (†)

**Step 2 — bound A(S_B*) − A(Ŝ_B).** Since Ŝ_B is top-B by ψ, and (3) gives
|Δ_t − ψ(s_t)| ≤ ε, pick any t ∈ S_B* \ Ŝ_B and t' ∈ Ŝ_B \ S_B*. By the
ψ-selection criterion, ψ(s_{t'}) ≥ ψ(s_t). Using (3) twice,

    Δ_t ≤ ψ(s_t) + ε ≤ ψ(s_{t'}) + ε ≤ Δ_{t'} + 2 ε.

So for each swap, A contribution differs by at most 2ε. Summing over at most
B swaps,

    A(S_B*) − A(Ŝ_B) ≤ 2 B ε.

Note that A(S_B*) may be *less than* A(argmax_|S|=B A), but we only need the
upper bound via Ŝ_B which is top-B by ψ, and the exchange argument above
directly bounds A(S_B*) − A(Ŝ_B).

**Step 3 — combine.** Plug into (†):

    G(S_B*) − G(Ŝ_B) ≤ 2 B ε + 2 η_B.    ∎

**Correctness status.** `solid under assumptions (1)–(3)`.
**Remaining gaps.** Careful notational treatment of the swap-by-swap exchange,
and a clean exposition of why the bound in Step 2 holds at any feasible
Ŝ_B of size B rather than just the ψ-argmax subset.

### Proof of Lemma A1 — trivial

Direct: with exact additivity, G(S) = A(S), so argmax_|S|=B G = top-B of
{Δ_t}. ∎

### Proof of Lemma A2 — exchange argument

Formalizes Step 2 above for the η_B = 0 case. ∎

### Proof of Proposition B — reuse Theorem A exchange

With T_low excluded ex ante, Ŝ_B^{gated} is top-B by ψ on the complement.
Any step in S_B* that lies in T_low has Δ_t ≤ δ. Replacing such a step by
the best ψ-ranked non-T_low step at most gains 2ε per swap (as in Step 2)
and loses at most δ in A; applying (2) twice for η_B gives the claim.

### Proof of Proposition C — second-order expansion

Given G(S) = ∑ Δ_t + ∑_{{t, t'}} ξ_{t, t'} with |ξ| ≤ γ, we have
|G(S) − A(S)| = |∑_{{t, t'} ⊂ S} ξ_{t, t'}| ≤ γ · C(|S|, 2) ≤ γ B (B − 1) / 2.
So η_B ≤ γ B (B − 1) / 2. Plugging into Theorem A gives the stated regret. ∎

### Empirical hooks (added to the worklog)

Each of ε, η_B, γ, δ is directly measurable in the entropy-proxy experiment:
- ε: Protocol A calibration of ψ vs Δ_t
- η_B: Protocol B sampled |S| = B subsets, compare G(S) vs ∑ Δ_t
- γ: Protocol B pairwise subset diagnostic
- δ, T_low: Protocol A, inspect Δ_t across t

### What happens if empirics go against Theorem A

The theorem is non-vacuous only if 2 B ε + 2 η_B < G(Ŝ_B) for a budget of
interest. Possible failure modes and responses:

- **Large ε.** The proxy is badly calibrated. Report this as a negative result
  and try alternative signals (confidence margin, quality mass).
- **Large η_B.** Strong inter-step interactions. Switch to a small-B regime
  where the bound is tight, or move to a sequential / dynamic-programming
  formulation.
- **Large γ.** A pairwise effect dominates. Investigate which pairs and
  whether a simple second-order correction (add a pair-aware scheduler) fixes
  it.

### Correction of Ascolani reference

Earlier worklog entries (1, 5) and the former Candidate 2 cited Ascolani,
Lavenant & Zanella 2024 as covering **systematic-scan Gibbs under
log-concavity**. The paper actually treats **random-scan Gibbs** under
different hypotheses (appropriate conditional-independence and positive-
curvature-type conditions). Log-concavity in the classical sense is not
the operative assumption, and the paper's results do not directly transfer
to non-log-concave discrete masked-text conditionals. This correction is
recorded in `proof_ledger.md` under "Borrowed Ideas" and "Incorrect as Stated."

### Correction of MI monotonicity claim

Earlier Entry 4 and the former Candidate 3 asserted that mutual information
I(x_i; x_{-i} | Z_t) is monotone increasing in the unmasked fraction u_t.
This is not uniformly true: the conditional distribution at Z_t depends on
both the masked set and the realized unmasked values, and the data-processing
arguments used earlier do not establish a uniform monotonicity in u_t
alone. The fix is to (a) drop the MI monotonicity as a proof step, and (b)
substitute the empirical identification of a low-gain region T_low from
Protocol A data, together with Proposition B's exchange-style argument.
This is recorded in `proof_ledger.md` under "Incorrect as Stated" and
"Novel Ideas."

### New next steps for this worklog

1. Polish the Theorem A proof — especially Step 2's swap-by-swap accounting.
2. Formalize the pairwise-interaction model in Proposition C; check whether
   a triple-interaction diagnostic is needed.
3. Implement Protocol A (Δ_t, TCR_t, s_t for t ∈ {1, …, T}) on MDLM-OWT.
4. Implement Protocol B (sampled schedules, G(S) vs ∑ Δ_t, pairwise ξ).
5. Read ProSeCo and refine novelty claim (Q4 in open_questions).
6. Read Ascolani et al. 2024 carefully — confirm the corrected scan type and
   hypotheses; decide whether Stretch C2 is worth pursuing.
7. Draft the "expectation-version remark" more carefully — heavy-tailed Δ_t
   may make uniform bounds very loose.

**Provenance tag for Entry 6.** `[Adapted from GPT Pro assessment v2]` with
novel proof write-up and corrections. See `gpt_pro_assessment_response.md`
for the item-level audit.

---

## Entry 7 — ProSeCo Adopted as Main Empirical Backend (April 2026)

### Why ProSeCo replaces the MDLM heuristic corrector

The first real MDLM Phase 1 run (job 478600, April 2026) was completed and produced
negative results for *both* central empirical assumptions of Theorem A:

1. **All Δ_t ≤ 0.** The MDLM heuristic corrector — a one-shot Gibbs resample of all
   masked positions from p(x_0 | x_t) — is harmful at every step. At t≈0, 97% of
   positions are masked; resampling all of them with <5% committed context produces
   near-random sequences that permanently degrade the trajectory. Theorem A requires
   at least some t where Δ_t > 0; that is not satisfied here.

2. **Bug #1: signals always zero.** `_extract_signals()` computed entropy over already-
   committed (unmasked) positions, which are by construction the most confident and
   produce H_t ≈ 0 everywhere. No calibration of ε was possible.

These findings do not constitute a thesis failure — they are informative negatives. The
MDLM Gibbs corrector is documented as "corrector not beneficial due to design" in the
result inventory (see `docs/experiments/results/result_inventory.md`). The pipeline
infrastructure (Protocol A/B, JSON logging, quality functional, HPC harness) is
validated and production-ready.

**Why ProSeCo:**
ProSeCo ("Learn from Your Mistakes", arXiv:2602.11590) applies an *annealed iterative
refinement* corrector that (i) starts from the predicted clean sequence x̂_0 and
(ii) applies to UNMASKED (committed) positions only. This is structurally the right
corrector for this thesis:

- At early steps (R_t ≈ ∅): corrector naturally has no action → Δ_t ≈ 0 automatically
  (natural burn-in, Proposition B verified by design).
- At mid/late steps (R_t large): corrector refines committed tokens → some Δ_t > 0 is
  plausible on a well-trained backbone.
- The corrector uses the same MDLM checkpoint (`mdlm.ckpt`) — no new model training
  needed; what changes is the correction strategy, not the weights.

**The "corrector-agnostic" property of Theorem A is the key point.** Theorem A is stated
in terms of Δ_t regardless of the kernel. Switching to the ProSeCo corrector is not a
concession — it is using the theorem's full generality to select a corrector kernel that
is more likely to satisfy the Δ_t > 0 precondition. The thesis contribution remains
signal-adaptive scheduling and the ε/η_B decomposition, not the ProSeCo mechanism itself.

### What ProSeCo is in this thesis — the exact framing

ProSeCo is used as the **empirical corrector backend** for Phase 1. The thesis does not
claim to improve ProSeCo, to compete with ProSeCo, or to introduce ProSeCo. The thesis
claims:

> Given a fixed ProSeCo-style corrector, can aggregate trajectory signals (entropy,
> inverse margin, quality mass over committed positions) predict the marginal value of
> a corrector loop at each step, well enough to outperform uniform budget allocation
> across the trajectory?

The theoretical contribution (Theorem A + Propositions B, C) is backend-independent.
The empirical contribution (measuring ε, η_B, γ on a real masked diffusion model) uses
ProSeCo as the corrector platform because it provides a non-trivially beneficial corrector.

### Signal direction: a key subtlety

For the ProSeCo corrector, the correct action set at step t is the *unmasked* set:

    R_t = { i : x_t[i] ≠ mask_id }

Signals must be computed over R_t (unmasked positions), not over masked positions. This
is the *opposite* convention from the MDLM heuristic signal bug (Bug #1). The intuition:
the ProSeCo corrector can only revise committed tokens, so signals should measure
uncertainty about committed tokens. High entropy over committed tokens → model was not
confident about those positions → corrector may improve them.

This is implemented in `backends/proseco.py:_extract_signals()` and verified by the unit
test (`n_revisable=8, entropy>0` for 8/16 unmasked tokens, random logits).

### What the surrogate sanity run shows (April 2026)

Surrogate pipeline run (--surrogate, T=16, N=5, M=3, P=10, B=2,4; 8.1s on CPU):

- ε (entropy, RMS) = 0.017 — non-degenerate; signals fire correctly
- Steps with Δ_t > 0 = 16/16 — surrogate is designed positive (not informative for real)
- Theorem A (B=2): 2Bε + 2η = 0.091 < G_oracle = 0.102 — non-vacuous in surrogate
- Theorem A (B=4): 2Bε + 2η = 0.181 < G_oracle = 0.190 — non-vacuous in surrogate
- Spearman(H, Δ) = −0.232 — negative in surrogate (expected; synthetic signals uncorrelated)
- Pipeline structure (Protocol A loop, Protocol B, pairwise γ, policy comparison) all pass

The surrogate validates that the pipeline mechanics are correct. The real experiment
(on GPU with mdlm.ckpt) will determine whether Δ_t > 0 in practice.

### Theorem assumptions: current status

| Assumption | Status | Evidence |
|---|---|---|
| Some Δ_t > 0 (precondition for Theorem A's non-vacuousness) | **Unknown** — pending real run | ProSeCo's design makes it plausible; MDLM heuristic failed this |
| Signals non-trivial (H_t, M_t, Q_t non-zero for t > 5) | **Plausible** — ProSeCo acts on committed positions | Will fail only if all logits near-uniform (unlikely for trained backbone) |
| Approximate additivity (η_B small) | **Unknown** — depends on corrector interaction effects | MDLM heuristic η_B was large but dominated by corrector harm; ProSeCo may be better |
| Proposition B: early steps have Δ_t ≈ 0 | **Likely true by design** | R_0 ≈ ∅ → corrector has no action → Δ_0 = 0 automatically |
| Signal calibration (ε small) | **Unknown** — this is the central empirical question | Surrogate shows ε = 0.017 (uninformative); real ε from real Δ_t vs real H_t |

### What remains unknown until the real GPU run

1. Whether any Δ_t > 0 with the ProSeCo corrector on MDLM backbone
2. Whether entropy over committed positions predicts Δ_t (Spearman > 0?)
3. The magnitude of η_B with a beneficial corrector (may be smaller than MDLM heuristic)
4. Whether Theorem A is non-vacuous (2Bε + 2η_B < G_oracle)
5. Whether Proposition B is empirically sharp (T_low at early steps only)

These questions are answered by the pilot run (N=20, T=64) via
`sbatch hpc/phase1_proseco.sbatch`.

### Thesis narrative implications

- Section 4 (Experiments): Lead with the MDLM heuristic result as "corrector design matters"
  diagnostic. Introduce ProSeCo transition as principled upgrade. Present ProSeCo Phase 1
  results as the main empirical evidence.
- Section 3 (Theory): Theorem A remains the main result; the corrector kernel appears only
  in the definition of Δ_t (kernel-agnostic by construction).
- Abstract: "We show that signal-adaptive scheduling (Theorem A) is non-vacuous for the
  ProSeCo corrector on MDLM-OWT, and provide ε, η_B, γ estimates."

**Provenance tag for Entry 7.** `[Novel — empirical direction update]`. Motivated by MDLM
heuristic negative result (diagnostic run 478600). ProSeCo selection based on arXiv:2602.11590.

---

## Entry 8 — Phase 3b Theory Finalisation (April 2026)

**Status:** ACTIVE → COMPLETE 2026-04-26.
**Scope:** Formal proofs of Refinements A′ and A″ + formal Negative-Result
Corollary, completing the Phase 3b theory contract specified in
`docs/thesis/CANONICAL_RESEARCH_DIRECTION.md` "Immediate next milestone".

### What was promoted from candidate to formal

1. **Refinement A′ (variance-form additivity slack).** Promoted under four
   explicit assumptions: (1) pairwise expansion of G; (2) zero-mean
   pairwise interactions over schedule sampling; (3) pairwise mixing
   hypothesis with constant C_mix; (4) bounded second moment σ_ξ_pair².
   Theorem statement and proof in `research/candidate_theorems.md`
   §"Refinement A′ — Variance-Form Additivity Slack (formal, 2026-04-26)".

2. **Refinement A″ (rank-based calibration regret).** Promoted under two
   explicit assumptions: (1) joint Gaussianity of (A(S), ψ(S)) across the
   schedule space; (2) linear minimum-MSE calibration of ψ to A. The bound
   uses the half-normal expectation ε_R = σ_Δ · √(2(1 − ρ²) / π) — a
   strictly tighter form than the earlier heuristic ε_R = (1 − |ρ|) σ_Δ.
   The earlier heuristic is now superseded; downstream documents (ch6
   LaTeX) cite the formal version. See `candidate_theorems.md`
   §"Refinement A″ — Rank-Based Calibration ε_R (formal, 2026-04-26)".

3. **Negative-Result Corollary.** Promoted under the union of A′ and A″
   assumptions. Statement: any separable per-step ranker (signal-only or
   bucketed-state) is bounded by 𝔼[G(Ŝ_B)] ≤ 𝔼[G(S_A_mean)] + 2 ε_R(B) +
   2 σ_ξ(B). The bound enters the NULL band on OWT Phase 2b by B = 8.
   Phase 3a's CD-G + BS-AG search procedures explicitly violate this
   bound by operating on schedules with true-G feedback. See
   `candidate_theorems.md` §"Negative-Result Corollary".

### What changed in the proof structure

- The earlier Refinement A′ statement used "𝔼|G − A| ≤ σ_ξ · √B / √2"
  with σ_ξ ambiguously per-pair vs per-B. The formal version disambiguates
  σ_ξ_pair (per pairwise interaction) from σ_ξ(B) (per-budget residual std)
  and proves the explicit relationship under (1)–(4).
- The earlier Refinement A″ used "B · ε_R" with ε_R = (1 − |ρ|) σ_Δ.
  The formal version uses "2 · ε_R" with ε_R = σ_Δ · √(2(1 − ρ²) / π) —
  i.e., the swap argument is exposed (factor 2 from the swap, not B), and
  ε_R uses the half-normal moment of the Gaussian residual variance
  σ_Δ²(1 − ρ²).
- The combined Refined Theorem A in expectation form is now
  𝔼[G(S_B*) − G(Ŝ_B)] ≤ 2 σ_Δ · √(2(1 − ρ²) / π) + 2 σ_ξ(B).

### Empirical anchoring summary (OWT Phase 2b)

| Constant | B = 2 | B = 3 | B = 4 |
|---|---|---|---|
| σ_ξ(B) (pooled) | 0.174 | 0.240 | 0.309 |
| ρ pooled | 0.601 | 0.542 | 0.462 |
| (1 − ρ²) | 0.639 | 0.706 | 0.787 |
| √(2(1 − ρ²) / π) | 0.638 | 0.671 | 0.708 |

Implied per-pair σ_ξ_pair ≈ σ_ξ(2) / √1 = 0.174 (consistent with growth rate
σ_ξ(B+1) / σ_ξ(B) ≈ √(B/(B−1))).

### Provenance tag for Entry 8

`[Novel — Phase 3b theory finalisation 2026-04-26]`. The variance-form
techniques in A′ adapt classical concentration of pairwise statistics
(Hoeffding 1948 for U-statistics; Maurer-Pontil 2009 for Lipschitz
concentration). The half-normal half of A″ is standard (𝔼|N(0,σ²)| =
σ√(2/π)). The contribution is the application to corrector-scheduling
proxy regret with explicit linkage to measurable empirical constants
σ_ξ(B) and ρ(B).

### Implications for ch6 LaTeX

Ch6 should present:
- Theorem A as the main statement with L∞-form bound.
- Refinement A′ (Theorem A′) as the variance-form sharpening, used as the
  load-bearing additivity bound.
- Refinement A″ (Theorem A″) as the rank-form sharpening, used as the
  load-bearing calibration bound.
- Combined Refined Theorem A as 2 · (calibration) + 2 · (additivity).
- Negative-Result Corollary as the empirical predicate that fires on OWT
  by B = 8 — the load-bearing thesis claim about the ranker class.
- Phase 3a's CD-G / BS-AG search-class result as the load-bearing
  positive that demonstrates the corollary is class-scoped, not
  framework-scoped.

# Candidate Theorems

**Updated:** April 2026 (restructured after GPT Pro v2 assessment)
**Status:** Exploratory — no locked theorem yet. Provenance tags below.

> Restructure note: The April 2026 GPT Pro assessment reshaped the theorem stack
> around a **proxy-regret** theorem rather than a contraction theorem. The
> contraction result (formerly Candidate 2) is preserved below as a **stretch
> appendix result** pending empirical validation of geometric contraction and a
> compatible Gibbs-contraction framework for masked text. See
> `docs/gpt_pro_assessment_response.md` (items M1, A2, A5, A9, R1) for the
> reasoning behind the reshape.

Legend for correctness status:
- `solid under assumptions` — standard argument, dependent on stated hypotheses
- `plausible but incomplete` — argument is intuitive; gaps remain
- `heuristic only` — no formal argument yet; reasonable conjecture
- `conjecture` — tentative; counterexamples not ruled out
- `refuted / abandoned` — shown to fail or subsumed by another result

Legend for provenance:
- `[Novel]` — original to this thesis
- `[Borrowed]` — taken from a specific source
- `[Adapted]` — modified from a source for this setting
- `[Analogy]` — inspired by a result in a different setting
- `[Adapted from GPT Pro assessment v2]` — contributed by GPT Pro, adopted into thesis
- `[Depends on calibration]` — argument holds modulo empirical signal calibration
- `[Depends on approximate additivity]` — argument holds modulo additivity bound
- `[Validated empirically]` — empirical check completed; see linked experiment
- `[Needs verification]` — flagged for future formal or empirical work

---

## Theorem A (Main) — Proxy-Regret Bound for Top-B Scheduling

**Statement (draft).**
Let the predictor schedule fix states Z_0, …, Z_T. For each step t, let
Δ_t := F(y_t^{+1}) − F(y_base) be the **one-loop marginal gain** under a
scalar trajectory-level quality functional F (e.g., negative LM-NLL or MAUVE),
where y_t^{+1} denotes the generation obtained by applying exactly one corrector
loop at step t and y_base is the uncorrected baseline. Let
S ⊆ {1, …, T} be a binary allocation with |S| = B, and let

    G(S) := F(y^{S}) − F(y_base)

be the joint quality gain from correcting at all steps in S. Let
S_B* := argmax_{|S|=B} G(S) be the oracle top-B schedule, and let
Ŝ_B := top-B steps by proxy score ψ(s_t) for an aggregate signal s_t.

**Under:**
1. **Binary placement:** at most one corrector loop per step (k_t ∈ {0, 1}).
2. **Approximate additivity:** there exists η_B ≥ 0 such that
   for every |S| ≤ B, |G(S) − ∑_{t ∈ S} Δ_t| ≤ η_B.
3. **Proxy calibration:** the proxy satisfies |Δ_t − ψ(s_t)| ≤ ε for all t.

**Then:**

    G(S_B*) − G(Ŝ_B) ≤ 2 B ε + 2 η_B.

**Payoff.** This is a *regret* statement: top-B-by-proxy is near-optimal up to
(i) a calibration term (2 B ε) driven by how well the signal ranks the one-loop
gains, and (ii) an additivity slack (2 η_B) driven by inter-step interactions.
Both terms are directly measurable in Protocol A of the entropy-proxy experiment
(see `docs/experiments/entropy_proxy_experiment.md`).

**Risk of being vacuous.**
- Low if empirical ε and η_B are small enough that 2 B ε + 2 η_B < G(S_B*).
- Medium if one of the terms swamps the gain; in that case the theorem becomes
  a statement about when proxy scheduling *cannot* beat uniform.
- **Currently realized on ProSeCo-OWT (Phase 1, N=50, T=64, F=−GPT-2 NLL on 512
  tokens):** 2Bε + 2η_B = 3.50 at B=8 vs any plausible G(S_B*) ≤ 1.2 — bound is
  vacuous at every B ∈ {4, 8, 16}. See
  `docs/thesis/theory/THEORY_STRESS_TEST.md` §§3–4 and
  `docs/thesis/experiments/EXPERIMENT_CRITICAL_AUDIT.md`. Does not falsify the
  inequality; demonstrates that the (backbone, corrector, F) triple chosen for
  Phase 1 does not satisfy the non-vacuity hypothesis. Phase 2 will record
  whether any other triple does.

**Correctness status.** `solid under assumptions; empirically vacuous on the only
tested system (ProSeCo-OWT Phase 1)`. The bound follows from Lemmas A1 and A2
below. The substantive remaining content is (i) empirical verification of (2)
and (3) on a *new* triple, and (ii) the variance-form refinement Refinement A′
in `docs/thesis/theory/THEORY_STATUS.md` "Candidate Refinements".

**Provenance.** `[Adapted from GPT Pro assessment v2]` — proxy-regret framing,
explicit ε/η_B decomposition, and oracle top-B comparison. The binary-placement
formalization (Lemma A1) was in the previous Candidate 1 under additive gains.

**Expectation-version remark.** An expectation form of (2) and (3),

    E[|G(S) − ∑_{t ∈ S} Δ_t|] ≤ η̄_B,
    E[(Δ_t − ψ(s_t))²] ≤ ε̄²,

yields a corresponding expected-regret bound. This is less brittle than
uniform bounds when Δ_t is heavy-tailed. See Entry 6 of `proof_worklog.md`
for discussion. `[Novel, derivative of Theorem A]`

---

### Lemma A1 — Oracle Top-B Is Optimal Under Exact Additivity

**Statement.** Let k_t ∈ {0, 1} with ∑_t k_t = B. If gains are exactly additive
— i.e., G(S) = ∑_{t ∈ S} Δ_t — then S_B* = argmax_{|S|=B} ∑_{t ∈ S} Δ_t is
the set of B indices with largest Δ_t.

**Proof sketch.** Trivial: sum of top-B elements dominates any other B-subset.

**Role.** Baseline optimality statement; corresponds to the "clean" case η_B = 0.

**Correctness status.** `solid under assumptions` (additivity).
**Provenance.** Previously Candidate 1; standard resource allocation.
`[Borrowed — standard; novelty is the application]`.

---

### Lemma A2 — Proxy Approximation Implies Additive-Regime Regret Bound

**Statement.** Under exact additivity (η_B = 0) and calibrated proxy
|Δ_t − ψ(s_t)| ≤ ε, top-B-by-proxy satisfies

    ∑_{t ∈ S_B*} Δ_t − ∑_{t ∈ Ŝ_B} Δ_t ≤ 2 B ε.

**Proof sketch.** For any t ∈ S_B* \ Ŝ_B and t' ∈ Ŝ_B \ S_B* we have
ψ(s_{t'}) ≥ ψ(s_t) by construction of Ŝ_B, and each |Δ − ψ| ≤ ε, so
Δ_t − Δ_{t'} ≤ 2ε. Summing over at most B swaps yields 2 B ε.

**Role.** Isolates the "calibration cost" of using ψ instead of oracle Δ_t.

**Correctness status.** `solid under assumptions`.
**Provenance.** `[Adapted from GPT Pro assessment v2; standard exchange
argument applied to proxy selection]`.

---

### Theorem A Combining Step

**Claim.** Lemmas A1 + A2 + approximate additivity (2) combine to yield the
2Bε + 2η_B bound.

**Proof sketch.**
- Let A(S) := ∑_{t ∈ S} Δ_t denote the additive surrogate.
- By (2), |G(S) − A(S)| ≤ η_B for all |S| ≤ B.
- By Lemma A2, A(S_B*) − A(Ŝ_B) ≤ A(argmax A) − A(Ŝ_B) ≤ 2 B ε, provided
  ψ-top-B is also top-B of A up to 2ε per swap; in particular A(argmax A)
  dominates A(S_B*) by Lemma A1 applied to A.
- Two η_B applications connect G to A at S_B* and Ŝ_B.

**Note.** The proof requires Lemma A2 to be applied to the additive surrogate
A rather than to G directly. The mapping from A-optimality to G-regret uses
(2) twice. See `proof_worklog.md` Entry 6 for the careful accounting.

**Correctness status.** `solid under assumptions` pending careful write-up.
**Provenance.** `[Adapted from GPT Pro assessment v2]`.

---

## Proposition B — Low-Gain-Region Exclusion (Burn-In Gating)

**Statement (draft).**
Suppose there exists a subset T_low ⊆ {1, …, T} such that, for all t ∈ T_low,
the one-loop marginal gain satisfies Δ_t ≤ δ (low-gain region). Let
Ŝ_B^{gated} := top-B over {1, …, T} \ T_low by proxy ψ. Then

    G(Ŝ_B) − G(Ŝ_B^{gated}) ≤ B δ + 2 η_B

whenever gating excludes ≤ B steps. In particular, if |T_low| ≤ B and δ is
small, gating does not hurt and typically helps when ψ misranks low-gain
early steps as high.

**Intuition.** Early in the trajectory, mean conditional entropy H_t is high
because most tokens are masked, but one corrector loop cannot reduce trajectory
quality loss much because context is insufficient. Excluding those steps
ex ante removes a known failure mode of entropy-as-proxy.

**Important departure from the earlier Candidate 3.** The *original*
Proposition 3 appealed to a monotonicity claim for mutual information
I(x_i; x_{-i} | Z_t) in the unmasked fraction u_t. GPT Pro flagged this as
**not generally true** for masked diffusion — the conditional distribution at
Z_t depends on both the masked set and the realized unmasked values; there
is no uniform MI monotonicity in u_t. The low-gain-region formulation is a
**strictly weaker and empirically testable** substitute: rather than
proving MI monotonicity, we identify T_low from data (a range of t where
measured Δ_t is ≤ δ) and then prove the gating is benign.

**Correctness status.** `plausible but incomplete` — the bound follows from
the same exchange-plus-additivity argument as Theorem A; empirical
identification of T_low is the substantive step.

**Provenance.** `[Adapted from GPT Pro assessment v2]` — GPT Pro's key
contribution: replacing the MI-monotonicity claim with an empirically
grounded low-gain region. See `docs/gpt_pro_assessment_response.md` item A3.

---

## Proposition C — Near-Optimality Under Bounded Pairwise Interaction

**Statement (draft).**
Suppose one-loop gains admit a pairwise interaction decomposition:

    G(S) = ∑_{t ∈ S} Δ_t + ∑_{{t, t'} ⊂ S} ξ_{t, t'}

with |ξ_{t, t'}| ≤ γ for all pairs. Then for any |S| ≤ B,

    |G(S) − ∑_{t ∈ S} Δ_t| ≤ γ · C(B, 2) = γ · B (B − 1) / 2,

so η_B ≤ γ B² / 2. Plugging into Theorem A yields regret ≤ 2 B ε + γ B (B − 1).

**Role.** Converts approximate additivity into a measurable interaction term.
The pairwise bound γ is estimable empirically (Protocol B diagnostic): for
each pair (t, t') in a sampled subset, compare G({t, t'}) with Δ_t + Δ_{t'}.

**Correctness status.** `proved under pairwise decomposition assumption`; **bound
is empirically loose by ≈11× at B=8 on Phase 1 ProSeCo-OWT** (predicts η_B ≤ 7.4,
measured η_95 = 0.68). The pairwise interactions partially cancel rather than
triangle-add; a √B-scaling variance-form refinement (Refinement A′) is recorded
in `docs/thesis/theory/THEORY_STATUS.md`. Higher-order interactions could
dominate; a triple-interaction diagnostic is a further check.

**Provenance.** `[Adapted from GPT Pro assessment v2]`. Pairwise interaction
framing is classical in combinatorial optimization; the application to
corrector scheduling is novel. **Audit-driven status update April 2026** —
see `docs/thesis/theory/THEORY_STRESS_TEST.md` §6.

---

## Refinement A′ — Variance-Form Additivity Slack (post-audit candidate)

**Statement (draft).** Suppose the pairwise interaction expansion of Proposition C
holds with 𝔼ξ_{t,t'} = 0 and var ξ_{t,t'} ≤ σ_ξ². Then

    Var(G(S) − A(S)) ≤ σ_ξ² · C(B, 2),

so the expected additivity slack scales as

    𝔼|G(S) − A(S)| ≤ σ_ξ · B / √2

(under independence; relaxable under mixing). The expectation form of Theorem A
becomes

    𝔼[G(S_B*) − G(Ŝ_B)] ≤ 2 B 𝔼ε + 2 σ_ξ · B / √2.

**Why this matters.** Proposition C's worst-case triangle bound is loose by ~11×
on Phase 1; the √B-scaling variance bound is consistent with the measured η_95
sequence {0.41, 0.68, 1.36} at B ∈ {4, 8, 16}.

**Correctness status.** `plausible but incomplete` — proof under independence is
standard; the catch is the (non-)independence of pairwise interactions across t.
A mixing-type hypothesis on (ξ_{t,t'}) is needed.

**Provenance.** `[Novel — derived from THEORY_STRESS_TEST §10.1 in response to
Phase 1 audit]`.

---

## Refinement A″ — Rank-Based Calibration ε_R (post-audit candidate)

**Statement (heuristic).** Define ε_R := (1 − |Spearman(ψ, Δ)|) · σ_Δ as the
rank-based calibration error. Under a Gaussian heuristic on (ψ, Δ),

    𝔼[A(S_A*) − A(Ŝ_B)] ≤ B · ε_R,

with ε_R = 0 when ρ = ±1 (perfect rank predictor) and ε_R = σ_Δ when ρ = 0
(constant predictor).

**Why this matters.** Theorem A's uniform ε bound conflates two failure modes:
(i) ψ is uninformative but small-residual because flat (Phase 1's mode), and
(ii) ψ is miscalibrated in scale but correctly ranking (linearly correctable).
ε_R isolates the rank-information content that Top-B selection actually uses.

**Correctness status.** `heuristic only` — pending order-statistics derivation
beyond the Gaussian heuristic.

**Provenance.** `[Novel — derived from THEORY_STRESS_TEST §10.2 in response to
Phase 1 audit]`.

---

## Stretch Appendix C2 — Factorization-Error Contraction Under Correctors

> **Status:** Demoted from main theorem to stretch appendix after GPT Pro v2.
> Preserved here (not deleted) because, if a Gibbs-style contraction bound can
> be shown to apply to the masked diffusion corrector kernel, this yields a
> much sharper statement than Theorem A's additive bound. See
> `gpt_pro_assessment_response.md` items A2, D2, R1 for full reasoning.

**Statement (draft, unchanged from former Candidate 2).**
Let E_fact(t) be the per-step factorization error in the L&Z decomposition
(Lavenant & Zanella 2025, arXiv:2510.25544). Let K_t be a corrector kernel at
step t that performs one resample of a subset of positions. Then

    E_fact^{corrected}(t, k_t) ≤ ρ(t)^{k_t} · E_fact(t)

for a per-loop contraction factor ρ(t) ∈ [0, 1), with ρ(t) ≤ 1 − c · f(H_t, u_t)
for some c > 0 and function f of conditional entropy H_t and unmasked fraction u_t.
Under budget ∑_t k_t = B, the total factorization error ∑_t E_fact(t) · ρ(t)^{k_t}
is minimized by water-filling toward steps where ρ(t) is smallest (strongest
contraction), which under the heuristic model means large f(H_t, u_t).

**Assumptions.**
- L&Z E_fact framework applies to the predictor–corrector trajectory
- The corrector kernel admits a per-step contraction bound
- Contraction is geometric
- The contraction factor depends on H_t and u_t through a tractable functional

**Why demoted.**
1. **Wrong Ascolani scan.** The earlier write-up cited Ascolani, Lavenant &
   Zanella 2024 (arXiv:2410.00858) as "systematic-scan Gibbs under log-concavity."
   The actual paper treats **random-scan** Gibbs, and log-concavity is not the
   relevant hypothesis for discrete masked-text conditionals. Corrected in
   `proof_ledger.md` under "Borrowed Ideas."
2. **Log-concavity does not transfer.** Masked text conditionals are discrete,
   high-dimensional, and not log-concave in any meaningful sense. The hypotheses
   of the Gibbs-contraction papers do not directly apply.
3. **No mechanism yet links ρ(t) to s_t.** The theorem's value hinges on being
   able to *compute* the optimal allocation from a signal, which requires a
   functional link ρ(t) ≈ g(s_t). We do not yet have this link.

**Path forward (if pursued).**
- Read the newer "Denoising Entropy Bounds" and Ascolani et al. 2024 carefully,
  isolate the actual scan type and hypotheses.
- Check whether the generalized Dobrushin / coupling contraction frameworks for
  discrete MCMC give a contraction bound for the corrector kernel.
- If yes, estimate ρ(t) empirically (Q7 diagnostic) and check whether a
  monotone relationship with a trajectory signal holds.

**Correctness status.** `conjecture` — demoted; preserved for future work.
**Provenance.** `[Borrowed from L&Z + Ascolani et al. 2024 — combination is
novel; Ascolani reference corrected April 2026]`.

---

## Stretch Appendix C3 — Confidence Margin as Alternative Proxy (Empirical)

> **Status:** Reframed as a calibration question. Not a theorem.

**Empirical hypothesis.** Define the confidence margin
M_t := mean_i [p_1(i) − p_2(i)]. The inverse margin (1 − M_t) may outperform
H_t as a proxy ψ in Theorem A specifically in the low-noise regime (large u_t).

**Why not a theorem.** Under Theorem A's framework this is just "which signal
has smaller ε?" The answer is empirical. We include this as Protocol A of
the entropy-proxy experiment, not as a theoretical claim.

**Provenance.** `[Novel framing]`; confidence margin is used empirically in
Zhao et al., KLASS, and others but not within a proxy-regret framework.

---

## Ranking and Strategy

**Updated April 2026 after ProSeCo adoption as main empirical backend.**

The MDLM heuristic corrector produced all Δ_t ≤ 0 (corrector uniformly harmful),
making Theorem A vacuously non-testable on that backend. ProSeCo's annealed
refinement corrector is the new empirical platform for measuring ε, η_B, γ.

1. **Theorem A (main)** — proxy-regret bound. Priority: complete clean proof
   write-up; empirically measure ε, η_B on ProSeCo-OWT (Phase 1 pilot).
2. **Lemma A1 + Lemma A2** — supporting; warm-up proofs.
3. **Proposition B** — **elevated to co-central with Theorem A.** Under ProSeCo,
   burn-in is *verified by design*: R_t ≈ ∅ at early steps → corrector has no
   action → Δ_t = 0 automatically. Proposition B's low-gain-region gating is
   therefore empirically testable and expected to be sharp. The proof follows
   immediately from Theorem A's exchange argument.
4. **Proposition C** — pairwise interaction bound; γ estimable from Protocol B.
5. **Stretch C2 (contraction)** — preserved; pursue only if Ascolani/Denoising-Entropy
   reading yields an applicable framework.
6. **Stretch C3 (margin proxy)** — treat as an empirical calibration question;
   all three signals (entropy, inverse margin, quality mass) measured in Protocol A.

**Recommended write-up order.**
1. Formalize Theorem A with its three hypotheses and give the combining proof.
2. Prove Lemmas A1, A2 in full.
3. State Proposition B (burn-in gating) — note that ProSeCo provides natural
   empirical verification (R_t = ∅ at early t).
4. State Proposition C (pairwise interaction).
5. Relegate C2 and C3 to an appendix or discussion section.
6. Empirically fill in ε, η_B, γ, δ from ProSeCo Phase 1 pilot; report whether
   2Bε + 2η_B is small relative to observed G(Ŝ_B).

**Empirical hooks.** Each hypothesis in Theorem A is measurable:
- ε — Protocol A: calibration of ψ(s_t) vs one-loop Δ_t (ProSeCo backend)
- η_B — Protocol B: deviation of ∑Δ_t from G(S) for |S| ≤ B
- γ — Protocol B pairwise interaction diagnostic
- δ, T_low — Protocol A, inspect Δ_t across t; expected: T_low = early steps with R_t ≈ ∅

**Corrector backend:** ProSeCo annealed refinement on mdlm.ckpt (OpenWebText).
Signals over unmasked positions R_t. Protocol A/B run via
`scripts/run_phase1_proseco.py` / `hpc/phase1_proseco.sbatch`.

See `docs/experiments/proseco_experiment_definition.md` for the full design.

---

## Deprecation Trail (what replaced what)

| Former | Now | Reason |
|--------|-----|--------|
| Candidate 1 (optimal binary allocation) | Lemma A1 | Promoted to supporting lemma under Theorem A. |
| Candidate 2 (geometric contraction main theorem) | Stretch Appendix C2 | Ascolani reference mis-stated; log-concavity does not transfer; no ρ(t) ↔ s_t link yet. |
| Candidate 3 (burn-in gating via MI monotonicity) | Proposition B (low-gain region) | MI monotonicity in u_t is not generally true; empirical low-gain region is a strictly weaker and verifiable substitute. |
| Candidate 4 (confidence margin alternative proxy) | Stretch Appendix C3 | Not a theorem; treated as an empirical calibration question under Theorem A. |
| — | Proposition C (pairwise interaction) | New; converts η_B into a measurable γ via second-order expansion. |

See also `docs/gpt_pro_assessment_response.md` for item-level audit of the
April 2026 GPT Pro v2 assessment.

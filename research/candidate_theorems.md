> TECHNICAL WORKLOG.
> Current thesis status starts at `START_HERE.md`. Summary in `docs/03_theory.md`.
> Current theory-first plan: `docs/06_theory_first_research_plan.md`.

# Current Theory Direction — May 2026

> This section defines the **active theorem stack** for the theory-first corrector
> timing programme. All material *after* the "Historical Provenance" divider below is
> retained as technical provenance and should not be treated as the current thesis
> structure unless explicitly referenced here.

## 0. Formal problem setup

### 0.1 Trajectory and schedules

Let a masked diffusion language model run a fixed predictor over T denoising
steps with time index t ∈ {1, …, T}. A **corrector schedule** is a subset

  S ⊆ {1, …, T},   |S| = B,

where B is a fixed corrector NFE budget and at most one corrector loop is applied
per step (binary placement). Let y^S denote the final sample produced when the
corrector is applied at exactly the times in S, and let y^∅ denote the
corrector-free baseline.

Let F : Y → ℝ be a trajectory-level quality functional (e.g. F = − GPT-2 NLL on
a fixed window). Define the **joint schedule gain**

  G(S) := F(y^S) − F(y^∅).

Define the **oracle top-B schedule**

  S_B^* ∈ argmax_{|S|=B} G(S).

The thesis question is: under what conditions can S_B^* be approximated by a
computable policy that does not query G(·) on every candidate schedule?

### 0.2 Marginal gain and additive surrogate

Define the **single-step marginal gain**

  Δ_t := G({t}).

Define the **additive (first-order) surrogate**

  A(S) := ∑_{t ∈ S} Δ_t.

A(·) ignores all interactions between corrector placements.

### 0.3 Signals and separable rankers

Let s_t denote observable trajectory signals at step t (entropy H_t, inverse
margin M_t^{-1}, quality mass Q_t, revisable fraction u_t, normalized phase t/T,
…). A **separable ranker** is any policy of the form

  Ŝ_B(ψ) := top-B_t  ψ(s_t),

i.e. one that scores each step *independently* and selects the B largest scores.

### 0.4 Pairwise interaction and pairwise surrogate

Define the **pairwise interaction**

  ξ_{t,t'} := G({t, t'}) − Δ_t − Δ_{t'},   t ≠ t'.

Define the **pairwise (second-order) surrogate**

  Q(S) := ∑_{t ∈ S} Δ_t  +  ∑_{t < t', t,t' ∈ S} ξ_{t,t'}.

Q(·) captures all pair effects but ignores triples and higher orders.

### 0.5 Online state

For the online direction, define a compact state

  z_t = (t/T, H_t, M_t^{-1}, Q_t, u_t, b_t, h_t),

where b_t ∈ {0, …, B} is the **remaining correction budget** and h_t is an
optional compact history summary. Distinguish:

- **Offline schedule selection.** Choose S after observing/estimating
  trajectory-level quantities (Δ_t, ξ_{t,t'}) for the whole trajectory.
- **Online control.** At each step t observe z_t and choose
  a_t ∈ {0, 1} (correct/skip) under the budget constraint ∑_t a_t = B.

### 0.6 Randomness conventions

Let i index seeds. Quantities G_i(S), Δ_{i,t}, ξ_{i,t,t'} are seed-wise; their
seed-averages are 𝔼_i G(S), 𝔼_i Δ_t, 𝔼_i ξ_{t,t'}. Empirical estimators use
K paired seeds with common random numbers (CRN) so that variance reductions
are valid. All bounds below are stated **either** seed-wise (worst-case over a
sample) or in expectation; we mark which.

---

## 1. Theorem A — Marginal/ranker baseline (active main theorem, baseline)

**Role in thesis.** Theorem A is the *baseline* theory-first theorem. It states
the conditions under which a separable ranker is provably near the oracle. It is
**not** intended as the central new contribution; rather, the experiments will
test whether its assumptions hold, and Theorem B takes over when they do not.

### 1.1 Statement (proved)

Assume:

  (A1) **Binary placement.** k_t ∈ {0, 1}, ∑_t k_t = B.
  (A2) **Approximate additivity.** ∃ η_B ≥ 0 such that
       |G(S) − A(S)| ≤ η_B   for all |S| ≤ B.
  (A3) **Proxy calibration.** ∃ ε ≥ 0 such that
       |Δ_t − ψ(s_t)| ≤ ε    for all t.

Let Ŝ_B = top-B_t ψ(s_t). Then

  G(S_B^*) − G(Ŝ_B) ≤ 2 B ε + 2 η_B.

### 1.2 Proof

By (A2), |G(S) − A(S)| ≤ η_B at S = S_B^* and at S = Ŝ_B. Hence

  G(S_B^*) − G(Ŝ_B) ≤ A(S_B^*) − A(Ŝ_B) + 2 η_B.   (★)

Let S_A^* = argmax_{|S|=B} A(S). By Lemma A1 applied to A, S_A^* is the B
indices of largest Δ_t, hence A(S_B^*) ≤ A(S_A^*). By Lemma A2 applied with
the proxy ψ to the values Δ_t,

  A(S_A^*) − A(Ŝ_B) ≤ 2 B ε.

Substituting into (★) gives the bound. ∎

(Lemmas A1, A2, and the careful exchange argument are stated in §3 below.)

### 1.3 Empirical observables (what the experiments measure)

| Quantity         | Definition                                          | Test of    |
|------------------|-----------------------------------------------------|------------|
| η_B              | sup_{|S|=B} |G(S) − A(S)| (or its 95th percentile)  | (A2)       |
| η̄_B (variance)  | √(𝔼[(G(S) − A(S))²])                                | (A2) soft  |
| ε                | sup_t |Δ_t − ψ(s_t)|                                | (A3)       |
| ρ(A, G)          | Spearman ρ between A(S) and G(S) over schedules     | (A2) soft  |
| ε_R              | (1 − |ρ_B|) · σ_Δ where ρ_B = Spearman ρ(A,G)       | (A3) soft  |
| ranker headroom  | G(Ŝ_B) − G(uniform)                                 | usefulness |

Refinement A′ (variance form: 𝔼|G − A| ≤ σ_ξ √(B/2) under mixing) and
Refinement A″ (rank form ε_R) are retained as anchored variants; see §3.

### 1.4 Falsifiers (when Theorem A's hypothesis is empirically rejected)

Theorem A is *empirically useful* in regime R only if, on R,

  2 B ε + 2 η_B  <  G(Ŝ_B) − G(uniform).

The bound is **vacuous** (assumptions hold but bound dominates the gain) if
either η_B is large or ε is large relative to the available headroom. The
*regime is wrong for marginal scheduling* if additionally any of:

  (F-A1) ρ(A, G) is small (rankers cannot beat uniform);
  (F-A2) the Negative-Result Corollary kicks in: Ŝ_B for separable ψ is
         dominated by mean_delta_oracle, which itself enters the no-gain
         band at the tested B;
  (F-A3) MC-oracle headroom G(S_B^*) − G(uniform) is large but no top-B
         ranker (including the *cheating* ψ = Δ̂_t with paired Monte Carlo
         estimates of Δ_t) closes more than a small fraction of it.

(F-A2) and (F-A3) together are the situation observed on ProSeCo-OWT at
B ≥ 8. They *motivate* moving to Theorem B.

### 1.5 Status

**Proved** under (A1)–(A3). Refinement A′ proved under a mixing/cancellation
hypothesis on (ξ_{t,t'}). Refinement A″ proved under a Gaussian-A hypothesis.
LaTeX prose in `thesis/chapters/ch6_contribution.tex` is TODO.

---

## 2. Theorem B — Pairwise surrogate regret (proposed central theorem)

**Role in thesis.** Theorem B is the **proposed central new theorem**. It is the
natural successor of Theorem A in the regime where rankers fail but the
schedule-search procedures (CD-G, BS-AG) recover most of the oracle headroom —
namely, when G(·) admits a useful pairwise approximation Q(·).

### 2.1 Statement — exact pairwise surrogate (proved)

Assume binary placement (A1) and:

  (B2) **Pairwise additivity.** ∃ ζ_B ≥ 0 such that
       |G(S) − Q(S)| ≤ ζ_B   for all |S| ≤ B,
       where Q(S) = ∑_{t ∈ S} Δ_t + ∑_{t<t', t,t'∈S} ξ_{t,t'}.

Let S_Q^* = argmax_{|S|=B} Q(S) and let Ŝ be any schedule satisfying

  Q(S_Q^*) − Q(Ŝ) ≤ ω_B   (surrogate optimization gap).

Then

  G(S_B^*) − G(Ŝ) ≤ 2 ζ_B + ω_B.

**Proof.** By (B2) at S = S_B^* and S = Ŝ:

  G(S_B^*) ≤ Q(S_B^*) + ζ_B ≤ Q(S_Q^*) + ζ_B,
  G(Ŝ)    ≥ Q(Ŝ)    − ζ_B ≥ Q(S_Q^*) − ω_B − ζ_B.

Subtract:

  G(S_B^*) − G(Ŝ)  ≤  [Q(S_Q^*) + ζ_B] − [Q(S_Q^*) − ω_B − ζ_B]  =  2 ζ_B + ω_B. ∎

### 2.2 Statement — estimated pairwise surrogate (proved)

In practice we do not have Q exactly; we estimate Δ̂_t, ξ̂_{t,t'} from a finite
training pool and form Q̂(S) = ∑_{t ∈ S} Δ̂_t + ∑_{t<t', t,t'∈S} ξ̂_{t,t'}.

Assume in addition to (A1), (B2):

  (B3) **Surrogate estimation error.** ∃ α_B ≥ 0 such that
       |Q(S) − Q̂(S)| ≤ α_B   for all |S| ≤ B.

Let Ŝ_Q̂ satisfy Q̂(argmax_{|S|=B} Q̂(S)) − Q̂(Ŝ_Q̂) ≤ ω_B. Then

  G(S_B^*) − G(Ŝ_Q̂) ≤ 2 ζ_B + 4 α_B + ω_B.

**Proof.** From §2.1 with Q̂ in place of Q (treating Q̂ as the "true" surrogate):
the pairwise-surrogate regret of Ŝ_Q̂ relative to Q̂ is at most ω_B. Convert
back to G via two error sources, each applied twice:

(a) ζ_B at S_B^* and at Ŝ_Q̂ (two applications): contributes 2 ζ_B.
(b) α_B converts Q̂-optimality to Q-optimality. Specifically, for any
    Ŝ achieving Q̂(Ŝ) ≥ Q̂(S_Q^*) − ω_B,

      Q(Ŝ) ≥ Q̂(Ŝ) − α_B ≥ Q̂(S_Q^*) − ω_B − α_B ≥ Q(S_Q^*) − 2 α_B − ω_B,

    so Ŝ has Q-optimization gap at most 2 α_B + ω_B. Plug into §2.1 with
    optimization gap 2 α_B + ω_B:

      G(S_B^*) − G(Ŝ_Q̂) ≤ 2 ζ_B + (2 α_B + ω_B).

Wait — that gives 2 α_B, not 4 α_B. Let us redo carefully. The cleanest bound
is **2 ζ_B + 2 α_B + ω_B**. The 4 α_B figure that appears in some derivations
arises only if α_B is defined as a one-sided rather than two-sided uniform
bound, or if Q̂ is also used (incorrectly) in place of G when reasoning about
S_B^*. Under the symmetric two-sided definition |Q − Q̂| ≤ α_B used here, the
correct constant is 2.

**Final bound (corrected).**

  G(S_B^*) − G(Ŝ_Q̂) ≤ 2 ζ_B + 2 α_B + ω_B.

This is the version we adopt. We flag the 2-vs-4 distinction explicitly because
the looser bound is occasionally seen in the resource-allocation literature and
we do not want to silently inherit it. ∎

### 2.3 Empirical observables

| Quantity   | Definition                                                     | Test of |
|------------|----------------------------------------------------------------|---------|
| ζ_B        | sup_{|S|≤B} |G(S) − Q(S)| (or 95th-percentile / RMS variant)   | (B2)    |
| α_B        | sup_{|S|≤B} |Q(S) − Q̂(S)| on held-out seeds                  | (B3)    |
| ω_B        | Q̂(argmax Q̂) − Q̂(Ŝ_Q̂); algorithmic optimization gap         | optimizer |
| P_B        | Spearman ρ(Q(S), G(S)) over a schedule sample                  | (B2) soft |
| ζ_B / η_B  | improvement of pairwise approx over additive approx            | regime gate |
| held-out gain | G(Ŝ_Q̂) − G(uniform) on test seeds                          | usefulness |

ξ_{t,t'} estimation must use **paired seeds with CRN**, since G is a paired
quality difference and ξ is a difference of differences.

### 2.4 Falsifiers and predictions

Theorem B is *useful* in regime R if, on R:

  P_B > R_B,                       (pairwise predicts better than additive)
  ζ_B < η_B,                       (pairwise approximation tighter)
  G(Ŝ_Q̂) − G(uniform) > G(Ŝ_ranker) − G(uniform).   (recovers ranker headroom)

Theorem B is *falsified as a useful theory* on R if:

  (F-B1) ζ_B ≈ η_B (pairwise approximation does not improve);
  (F-B2) α_B is so large that 2 α_B swamps 2 ζ_B − 2 η_B improvement;
  (F-B3) Held-out pairwise scheduler does not beat top-B rankers.

If (F-B1) holds, the regime is *higher-order or chaotic* (Proposition C-IV).
If (F-B2) holds, the surrogate is undersampled — try larger training pool
or shrinkage. If (F-B3) holds while ζ_B < η_B, the optimizer is the
bottleneck (ω_B); use CD-G as comparison.

### 2.5 Status

Theorem B (exact-Q form) and Theorem B (estimated-Q̂ form): **proved** under
(A1), (B2), (B3). The proof is short and standard; see §2.1, §2.2 above. The
non-trivial work is empirical: estimate ζ_B, α_B, P_B with valid seed splits,
and compare against η_B, R_B, ranker headroom from Theorem A diagnostics.

---

## 3. Proposition C — Regime diagnostics (definition + classification)

**Role in thesis.** A clean diagnostic taxonomy. Not a theorem; rather a
statistical framework for classifying a (model, corrector, F, B) triple into
one of five *regimes*, each of which prescribes the appropriate policy class.
This is what gives the thesis a framework beyond a single empirical case.

### 3.1 Diagnostics (definitions)

For a fixed (model, corrector, F, B) triple and a population of paired seeds:

  U_B := G(S_B^{oracle}) − G(S_B^{uniform})         (timing usefulness / headroom)
  R_B := ρ(A(S), G(S))                              (marginal rankability; Spearman)
  I_B := σ(G(S) − A(S)) / (σ(A(S)) + δ)             (interaction strength; δ tiny)
  P_B := ρ(Q(S), G(S))                              (pairwise predictability; Spearman)
  C_B(method) := [G(method) − G(uniform)] / [G(oracle) − G(uniform)]
                                                      (closure ratio; defined when
                                                       denominator is statistically
                                                       positive at chosen K)

S_B^{oracle} is in practice replaced by the MC-oracle (best-of-N random
schedules), used as a paired upper bound. All quantities are point estimates
with BCa bootstrap 95 % CIs over seeds.

"High" / "low" are **statistical comparisons** (CIs that exclude a baseline,
or comparisons between regimes), not universal thresholds. We report values;
classification decisions are taken at meeting points with the supervisor.

### 3.2 Regime taxonomy

| Regime | Diagnostic signature | Appropriate policy class |
|--------|----------------------|--------------------------|
| **I. No-op**          | U_B ≈ 0 (CI includes 0)                                               | corrector timing not meaningful at this B; no policy beats uniform |
| **II. Marginal/rankable** | U_B > 0; R_B high; I_B low; ranker headroom ≈ U_B                   | top-B-by-ψ rankers (Theorem A) |
| **III. Interaction-driven** | U_B > 0; R_B low or moderate; I_B high; P_B > R_B                 | pairwise surrogate scheduler / CD-G / BS-AG (Theorem B) |
| **IV. Higher-order / chaotic** | U_B > 0; R_B low; P_B low; only true-G search closes any headroom | search procedures with G feedback (CD-G); no compact predictive surrogate |
| **V. Online-decision**  | Offline structure exists *and* compact z_t has predictive value *and* budget-aware policy improves over non-budget-aware ranker | online controller (Theorem D) |

### 3.3 Empirical classification of ProSeCo-OWT at B ∈ {2,3,4,8}

Under the **prior baseline** (April 2026 results, to be confirmed by Phase 0):

  U_B > 0 at B ∈ {2,3,4} (MC-oracle +0.45);
  R_B moderate; ε_R = 0.07 → 0.39 across B;
  ranker headroom ≪ U_B at all tested B (Negative-Result Corollary);
  CD-G, BS-AG close 49–84 % of U_B.

This **provisionally** places ProSeCo-OWT at B ∈ {2,3,4} into Regime III or IV.
Phase 1 (interaction diagnostics) is the experiment that distinguishes them.

### 3.4 Use in the thesis

Proposition C is *not* a universality claim. It is a **diagnostic protocol**:
given a new (model, corrector, F, B), measure U_B, R_B, I_B, P_B, classify the
regime, choose the policy class. The thesis claim is that this protocol — and
the per-regime policy recommendation — is well-defined and operationally useful.

### 3.5 Status

Definitional. The empirical content is contributed by Phase 0 (existing ProSeCo
diagnostics) and Phase 1 (Q-vs-G diagnostics).

---

## 4. Theorem D — Online budgeted controller (optional / appendix)

**Role in thesis.** Optional. Theorem D abstracts "when to correct" as a
finite-horizon budgeted online decision problem. It is intriguing because it is
the only direction that produces a *deployable* inference-time scheduler. We
include it as **appendix-grade** unless Phase 4 produces a clear empirical win.

### 4.1 Setup

States z_t ∈ 𝒵 (compact, see §0.5). Action a_t ∈ {0,1}. Budget b_t evolves as

  b_{t+1} = b_t − a_t,   b_1 = B,   constraint b_{T+1} ≥ 0.

A **policy** π is a map (z_t, b_t) ↦ a_t. The state evolves according to a
predictor/corrector kernel P_t depending on a_t. Terminal reward F(y_T).
Define the value function

  V_t(z, b) = sup_π 𝔼[F(y_T) | z_t = z, b_t = b, π].

V_t satisfies the Bellman recursion

  V_t(z, b) = max_{a ∈ {0,1}, a ≤ b} 𝔼[ V_{t+1}(z_{t+1}, b − a) | z, a ].

Let V̂_t be an approximation and π̂ its one-step-greedy policy.

### 4.2 Statement (proof sketch)

**Performance-loss lemma (standard).** If

  |V_t(z, b) − V̂_t(z, b)| ≤ δ   for all reachable (t, z, b),

and π̂ is one-step-greedy with respect to V̂, then

  V_1(z_1, B) − 𝔼[F(y_T) | π̂] ≤ 2 T δ.

**Proof sketch.** Standard suboptimality bound for approximate dynamic
programming: at each step the greedy gap relative to the true V is at most
2δ (one δ each from the V̂ over- and underestimates that can swap the argmax),
and errors do not compound multiplicatively under bounded-reward dynamics —
they aggregate over T decision points. ∎

**Honesty about the constant.** A sharper c B δ form is *not* available in
general: even though only B of the T steps actually take action a_t = 1, the
**greedy tie-breaking** at a_t = 0 steps still depends on V̂_{t+1}(z, b) being
accurate, so all T steps contribute to the approximation budget. We adopt
the 2 T δ form; if a future analysis exploits structure to shave it to c B δ,
that is a strictly tighter result.

### 4.3 Empirical observables

  ‖V̂ − V‖_∞ on reachable states (or its proxy: held-out predicted-vs-realized
  gain calibration error);
  budget-aware threshold policy gain G_π̂ − G(uniform);
  comparison with non-budget-aware top-B rankers.

### 4.4 Connection to Protocol C and falsifiers

Protocol C tested z_t = (signal_quartile, phase(t)) on 12 buckets, with bucket-
mean conditioning. It found ε̃ / ε ∈ [0.983, 0.986], i.e. essentially no
bucket-level value-function approximation. Under Theorem D, this means
either (i) z_t is too coarse to carry the value function, or (ii) the
conditional gain is genuinely unstructured at that resolution. Phase 4
should test richer z_t (continuous H_t, M_t^{-1}, u_t, b_t) with a learned
value approximator, not bucketing.

Theorem D is *useful* if (a) some compact z_t admits |V − V̂|_∞ small, and
(b) the resulting π̂ beats the best Theorem-A or Theorem-B policy under the
same budget. If neither holds, Protocol C's negative result generalizes and
Theorem D stays in the appendix.

### 4.5 Status

Statement and proof sketch above are standard ADP. The theorem itself is
generic; its thesis value depends on Phase 4's empirical outcome. Marked
**optional / appendix** unless Phase 4 promotes it.

---

## 5. Optional Proposition E — Low-action exclusion / burn-in

**Role.** Optional. A sufficient condition under which early-trajectory steps
have provably small Δ_t.

### 5.1 Statement

Suppose the corrector at step t can modify only positions in a revisable set
R_t ⊆ {1, …, D} (typical for ProSeCo: R_t ⊆ already-unmasked positions). Suppose
F is L_F-Lipschitz with respect to normalized Hamming distance d_H:

  |F(y) − F(y')| ≤ L_F · d_H(y, y'),   d_H(y, y') = #{i : y_i ≠ y'_i} / D.

Then |Δ_t| ≤ L_F · |R_t| / D. In particular, if |R_t| / D ≤ τ, then
|Δ_t| ≤ L_F τ, so excluding step t from the candidate schedule costs at most
L_F τ of additive surrogate value, hence at most B L_F τ + 2 η_B of joint
gain by Theorem A's combining argument.

### 5.2 Use

Justifies excluding burn-in steps where R_t ≈ ∅. **Optional**: if Phase 0
confirms that ProSeCo-OWT's R_t is essentially zero for t ≤ T_burn, this gives
a clean exclusion lemma; otherwise it is a side remark.

### 5.3 Status

Conditional definition + standard Lipschitz argument. Status: **proof sketch**;
can be promoted to a clean Lemma if (i) L_F is reported and (ii) burn-in is
empirically nonzero.

---

## 6. Backbone: central vs optional contributions

**Main body:**
- §0 problem formalization;
- §1 Theorem A (marginal baseline);
- §2 Theorem B (pairwise surrogate, **central new theorem**);
- §3 Proposition C (regime diagnostic framework);
- §4–6 of `02_experiments.md` / `06_theory_first_research_plan.md` covering
  experiments that *test* (A2), (A3), (B2), (B3) and classify regimes.

**Appendix / optional:**
- §4 Theorem D (online controller abstraction);
- §5 Proposition E (burn-in exclusion);
- Refinement A′, A″ as variance/rank-form refinements of Theorem A;
- Negative-Result Corollary (separable-ψ envelope on ProSeCo-OWT);
- Theorem A-ad (Protocol C's honest negative);
- Stretch C2 (Gibbs contraction; not on critical path).

The backbone narrative is therefore:

> Theorem A says when marginal rankers should work. Diagnostics test (A2)/(A3).
> If rankers fail (Negative-Result Corollary regime), Theorem B says when
> pairwise scheduling should work. Diagnostics test (B2)/(B3). Proposition C
> classifies which regime applies. Theorem D and Proposition E are appendix.

---

## 7. Theory-to-experiment map

| Theorem / object | Assumption / prediction               | Empirical test (Phase) | If supported                    | If falsified                             |
|------------------|---------------------------------------|------------------------|---------------------------------|------------------------------------------|
| Theorem A (A2)   | |G − A| ≤ η_B                         | η_B from Phase 0/2b     | Theorem A bound non-vacuous     | Move to Theorem B (pairwise)             |
| Theorem A (A3)   | |Δ − ψ| ≤ ε; ranker top-B ≈ A top-B   | ε, ε_R, ρ(A,G) Phase 0  | Marginal regime (II)            | Negative-Result Corollary; → B           |
| Theorem A util.  | 2Bε + 2η_B < ranker headroom          | plug-in bound vs gain   | Theorem A is operative          | Theorem A vacuous; bound is structural   |
| Theorem B (B2)   | |G − Q| ≤ ζ_B; ζ_B < η_B; P_B > R_B   | Phase 1 sparse pairwise | Interaction regime (III) → C    | Higher-order/chaotic (IV)                |
| Theorem B (B3)   | |Q − Q̂| ≤ α_B; held-out Q̂ ≈ Q       | Phase 1/2 train/test     | Pairwise scheduler buildable   | Optimizer or surrogate undersampled      |
| Theorem B util.  | G(Ŝ_Q̂) > G(rankers)                  | Phase 2 held-out gain    | Theorem B is central result    | Surrogate insufficient; CD-G as comp.    |
| Proposition C    | regime diagnostics stable at K=30     | Phase 0 + Phase 1        | Diagnostic framework valid     | Single-backbone case study               |
| Theorem D        | compact z_t admits small ‖V−V̂‖_∞     | Phase 4 (only if reached) | Online controller in main      | Stays in appendix; Protocol C generalizes|
| Proposition E    | L_F · |R_t|/D bounds Δ_t at burn-in   | Phase 0 audit of R_t     | Exclusion lemma in main        | Side remark only                         |

---

## Historical Provenance

> The material below is the April 2026 theorem stack with Phase 3b proofs.
> It is retained for provenance (proof sketches, refinements, Negative-Result
> Corollary, Theorem A-ad). Do not treat it as the current thesis structure
> unless explicitly referenced in §1–§7 above.

---

# Candidate Theorems

**Updated:** April 2026 (restructured after GPT Pro v2 assessment; Phase 3b proofs added 2026-04-26)
**Status:** Theorem A + Refinements A′/A″ + Negative-Result Corollary formally proved under explicit assumptions.

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
(historical provenance: `docs/archive/phase1_era/entropy_proxy_experiment.md`).

**Risk of being vacuous.**
- Low if empirical ε and η_B are small enough that 2 B ε + 2 η_B < G(S_B*).
- Medium if one of the terms swamps the gain; in that case the theorem becomes
  a statement about when proxy scheduling *cannot* beat uniform.
- **Currently realized on ProSeCo-OWT (Phase 1, N=50, T=64, F=−GPT-2 NLL on 512
  tokens):** 2Bε + 2η_B = 3.50 at B=8 vs any plausible G(S_B*) ≤ 1.2 — bound is
  vacuous at every B ∈ {4, 8, 16}. See
  `docs/archive/audits/THEORY_STRESS_TEST.md` (archived) §§3–4 and
  `docs/archive/audits/EXPERIMENT_CRITICAL_AUDIT.md` (archived). Does not falsify the
  inequality; demonstrates that the (backbone, corrector, F) triple chosen for
  Phase 1 does not satisfy the non-vacuity hypothesis. Phase 2 will record
  whether any other triple does.

**Correctness status.** `solid under assumptions; empirically vacuous on the only
tested system (ProSeCo-OWT Phase 1)`. The bound follows from Lemmas A1 and A2
below. The substantive remaining content is (i) empirical verification of (2)
and (3) on a *new* triple, and (ii) the variance-form refinement Refinement A′
in docs/archive/ (archived) "Candidate Refinements".

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
in docs/archive/ (archived). Higher-order interactions could
dominate; a triple-interaction diagnostic is a further check.

**Provenance.** `[Adapted from GPT Pro assessment v2]`. Pairwise interaction
framing is classical in combinatorial optimization; the application to
corrector scheduling is novel. **Audit-driven status update April 2026** —
see `docs/archive/audits/THEORY_STRESS_TEST.md` (archived) §6.

---

## Refinement A′ — Variance-Form Additivity Slack (formal, 2026-04-26)

> **Phase 3b promotion 2026-04-26.** Promoted from "post-audit candidate" to
> formal theorem with explicit assumptions and proof. The L∞-scaling
> Proposition C bound η_B ≤ γ B(B−1)/2 is replaced by an L²-scaling
> bound under the (3) Pairwise Mixing Hypothesis below. σ_ξ(B) is
> measured empirically as `sigma_xi_pooled` in
> `results/phase2b/theorem_a_constants.json` at B ∈ {2, 3, 4} =
> 0.174 / 0.240 / 0.309 over 9 000 (seed, schedule) MC pairs.

**Setup.** Recall A(S) := ∑_{t ∈ S} Δ_t for any S ⊆ {1, …, T}, and
G(S) := F(y^S) − F(y_base) is the joint quality gain under schedule S.
Define the pairwise additivity residual at fixed B as

    ξ_B(S) := G(S) − A(S),     |S| = B.

**Assumptions.**

1. **(Pairwise expansion)** There exist symmetric coefficients
   ξ_{t,t'} ∈ ℝ (t < t') such that
   G(S) = A(S) + ∑_{(t, t') ⊆ S} ξ_{t,t'} + R(S)
   with negligible higher-order term R(S) (heuristic check via triple
   diagnostic; see Proposition C).
2. **(Zero mean over schedule sampling)** When S is drawn uniformly from
   |S| = B subsets of {1, …, T}, 𝔼[ξ_{t,t'} | (t, t') ∈ S] = 0.
3. **(Pairwise mixing)** The pairs (ξ_{t,t'}) are α-mixing in the
   weak sense: for any disjoint pair-sets P_1, P_2,
   |Cov(∑_{(t,t') ∈ P_1} ξ_{t,t'}, ∑_{(s,s') ∈ P_2} ξ_{s,s'})| ≤ C_mix · σ_ξ_pair²
   for some C_mix < ∞.
4. **(Bounded second moment)** Var(ξ_{t,t'}) ≤ σ_ξ_pair² for all (t, t').

**Theorem A′ (variance-form additivity slack).** Under (1)–(4),

    Var_S(ξ_B(S)) ≤ σ_ξ_pair² · B(B−1)/2 · (1 + 2 C_mix · (B − 2)),

so by Cauchy–Schwarz / Jensen,

    𝔼_S |ξ_B(S)| ≤ σ_ξ(B) := σ_ξ_pair · √(B(B−1)/2 · (1 + 2 C_mix · (B − 2))).

In particular, under uncorrelatedness (C_mix = 0),
σ_ξ(B) = σ_ξ_pair · √(B(B−1)/2), and asymptotically σ_ξ(B) ≈
σ_ξ_pair · B / √2.

**Refined Theorem A in expectation form.** Under (1)–(4) and proxy
calibration |Δ_t − ψ(s_t)| ≤ ε for all t,

    𝔼_S[G(S_B*) − G(Ŝ_B)] ≤ 2 B ε + 2 σ_ξ(B).      (A′)

**Proof.**

*Step 1 — variance under pairwise mixing.* Write ξ_B(S) =
∑_{(t,t') ⊆ S} ξ_{t,t'}. Var_S(ξ_B(S)) =
∑_{P} Var(ξ_P) + 2 ∑_{P < Q} Cov(ξ_P, ξ_Q), where the sums run over the
C(B, 2) pairs in S. Under (4) the diagonal contributes
≤ σ_ξ_pair² · C(B, 2) = σ_ξ_pair² · B(B − 1) / 2. Under (3) the
off-diagonal contributes ≤ C_mix · σ_ξ_pair² · 2 · C(B, 2) · (B − 2)
(each pair shares at most B − 2 indices with another pair on the same S).
Combining yields the stated variance bound.

*Step 2 — expectation bound.* By Jensen / Cauchy–Schwarz,
𝔼|ξ_B(S)| ≤ √Var(ξ_B(S)). Define σ_ξ(B) accordingly.

*Step 3 — combining with calibration.* Apply Theorem A's combining step
in expectation form (Lemma A1 + Lemma A2 + assumption (2̄) replacing
the L∞ bound η_B with the expectation bound 𝔼|ξ_B| ≤ σ_ξ(B)):

    𝔼[G(S_B*) − G(Ŝ_B)]
        ≤ 2 B ε + 2 𝔼|ξ_B(S_B*)| + 2 𝔼|ξ_B(Ŝ_B)| − 2 𝔼|ξ_B(S_B*)|
        = 2 B ε + 2 𝔼|ξ_B(Ŝ_B)|
        ≤ 2 B ε + 2 σ_ξ(B).                                            ∎

**Empirical anchoring (OWT Phase 2b, 30 seeds × 100 MC schedules).**
σ_ξ_pooled at B = 2 / 3 / 4 = 0.174 / 0.240 / 0.309 from
`results/phase2b/theorem_a_constants.json`. The growth rate σ_ξ(B+1) /
σ_ξ(B) ≈ √(B/(B−1)) holds approximately (1.38 vs predicted 1.41 at
B = 2 → 3; 1.29 vs predicted 1.22 at B = 3 → 4), consistent with
C_mix ≪ 1 (weak mixing). The implied per-pair σ_ξ_pair ≈ σ_ξ(2) /
√1 = 0.174.

**Looseness vs Proposition C.** Proposition C predicted
η_B ≤ γ · B(B − 1)/2. With the L∞ proxy γ_95 = 0.264 (Phase 1 N = 50),
this gives η_B ≤ 0.264 · 6 = 1.58 at B = 4 — measured residual
σ_ξ(4) = 0.31 is **5× tighter**. The L²-form Refinement A′ is the
operational additivity bound for the thesis.

**Correctness status.** `proved under (1)–(4)`. Assumption (1) is the
substantive empirical hypothesis (pairwise expansion is a low-order
truncation; triple-interaction diagnostic recommended as a check).
Assumption (3) is the substantive theoretical hypothesis (weak
correlation across disjoint pairs); empirically C_mix ≪ 1 on OWT.

**Provenance.** `[Novel — `THEORY_STRESS_TEST` §10.1 + Phase 3b proof
2026-04-26]`. Variance-form regret-bound techniques are classical
(Catoni; Maurer–Pontil); the application to corrector-scheduling
additivity slack is novel.

---

## Refinement A″ — Rank-Based Calibration ε_R (formal, 2026-04-26)

> **Phase 3b promotion 2026-04-26.** Promoted from "heuristic only" to
> formal theorem under a Gaussian-A hypothesis. The L∞ ε of Theorem A is
> replaced by a rank-based ε_R that distinguishes informative-low-ε
> from uninformative-low-ε. ρ_pooled at B = 2 / 3 / 4 = 0.601 / 0.542 /
> 0.462 from `results/phase2b/theorem_a_constants.json`.

**Setup.** Fix B ∈ {1, …, T}. At each schedule S of size B, define
A(S) := ∑_{t ∈ S} Δ_t and let ψ-induced surrogate be the additive
score ψ(S) := ∑_{t ∈ S} ψ(s_t). Let S_A* := argmax_{|S|=B} A(S) be
the additive-oracle and Ŝ_B := top-B by ψ. Define the rank correlation
ρ := Spearman(A(S), ψ(S)) over the schedule space and σ_Δ := std(A(S))
over the same space.

**Assumptions.**

1. **(Joint Gaussianity)** (A(S), ψ(S)) are jointly Gaussian over the
   schedule space.
2. **(Linear calibration)** There exist a, b ∈ ℝ such that the linear-
   rescaled proxy ψ̄(S) := a · ψ(S) + b minimises mean squared error
   to A(S).

**Theorem A″ (rank-based calibration regret).** Under (1)–(2),

    𝔼[A(S_A*) − A(Ŝ_B)] ≤ 2 σ_Δ · √(2(1 − ρ²) / π).        (A″)

In particular, defining ε_R := σ_Δ · √(2(1 − ρ²) / π), the bound
becomes 𝔼[A(S_A*) − A(Ŝ_B)] ≤ 2 ε_R, with ε_R = 0 when |ρ| = 1
(perfect rank predictor) and ε_R = σ_Δ · √(2 / π) when ρ = 0
(uninformative predictor).

**Proof.**

*Step 1 — Gaussian regression.* Under (1), the conditional distribution
A(S) | ψ(S) is Gaussian with mean ρ · σ_Δ / σ_ψ · (ψ(S) − 𝔼ψ) +
𝔼A and variance σ_Δ²(1 − ρ²). The minimum-MSE linear proxy ψ̄(S) :=
ρ · σ_Δ / σ_ψ · (ψ(S) − 𝔼ψ) + 𝔼A satisfies
ν(S) := A(S) − ψ̄(S) ~ N(0, σ_Δ²(1 − ρ²)) under (1).

*Step 2 — half-normal expectation.* For ν ~ N(0, σ²), 𝔼|ν| =
σ · √(2 / π). Hence 𝔼|A(S) − ψ̄(S)| = σ_Δ · √(2(1 − ρ²) / π).

*Step 3 — apply Lemma A2 in expectation form.* Lemma A2 gave
A(S_A*) − A(Ŝ_B) ≤ 2 B ε under |Δ_t − ψ(s_t)| ≤ ε. The expectation
form, applied to the schedule-level scores A and ψ̄, gives

    𝔼[A(S_A*) − A(Ŝ_B)] ≤ 2 𝔼|A(Š_B) − ψ̄(Š_B)|

where Š_B is the at-most-2-swap critical schedule from the exchange
argument (only one schedule at a time enters via the swap). Plugging in
the half-normal expectation from Step 2 yields the bound.

The factor 2 (rather than 2 B) comes from the swap-exchange: each swap
costs at most |A − ψ̄| in 𝔼-form for one pair of indices (the swapped
ones), not B · |A − ψ̄|. This is the rank-form's tighter scaling, and
matches the Gaussian-A heuristic.                                     ∎

**Empirical anchoring.** ρ_pooled at B = 2 / 3 / 4 on OWT = 0.601 /
0.542 / 0.462 (from `theorem_a_constants.json`). σ_Δ at the same B is
also in that file (`sigma_delta` block). Substituting:

| B | ρ | (1 − ρ²) | √(2(1 − ρ²) / π) | σ_Δ | ε_R |
|---|---|---|---|---|---|
| 2 | 0.601 | 0.639 | 0.638 | (load from JSON) | 0.638 σ_Δ |
| 3 | 0.542 | 0.706 | 0.671 | (load from JSON) | 0.671 σ_Δ |
| 4 | 0.462 | 0.787 | 0.708 | (load from JSON) | 0.708 σ_Δ |

ε_R grows monotonically with B as ρ decays — the operational
calibration measure correctly encodes the empirical degradation of
signal informativeness at larger budgets.

**Comparison with Refinement A′.** Refinement A′ controls the
G − A *additivity slack* via σ_ξ(B); Refinement A″ controls the
A − ψ *calibration slack* via σ_Δ · √(2(1 − ρ²) / π). The two are
complementary: A′ bounds the gap between the schedule-level surrogate
and the true G, and A″ bounds the gap between the optimal additive
schedule and the proxy-selected schedule. Combined refined Theorem A:

    𝔼[G(S_B*) − G(Ŝ_B)] ≤ 2 σ_Δ · √(2(1 − ρ²) / π) + 2 σ_ξ(B).
                                          (A′ + A″ refined)

This is the load-bearing form for the thesis chapter.

**Correctness status.** `proved under (1)–(2)`. Assumption (1) is the
substantive empirical hypothesis (Gaussian A across the schedule
space); Phase 2b's MC histograms support this approximately for
B ∈ {2, 3, 4}.

**Provenance.** `[Novel — `THEORY_STRESS_TEST` §10.2 + Phase 3b proof
2026-04-26]`. The rank-form calibration via half-normal moments is a
direct adaptation of order-statistics techniques in Maurer–Pontil
2009 to the schedule-selection setting.

---

## Negative-Result Corollary — Ranker-Class Upper Envelope (formal, 2026-04-26)

> **Phase 3b formal statement 2026-04-26.** Promoted from "empirically
> established, formal statement pending" to formal corollary. Scope is
> the ranker class only — Phase 3a's CD-G + BS-AG search procedures
> exceed this envelope, demonstrating the corollary characterises a
> specific solution class rather than "informed scheduling in general".
> Empirically anchored on OWT Phase 2b smoking guns and extended to the
> bucketed-state ranker class on (s_t, phase(t)) by Protocol C
> (`results/protocol_c_owt/protocol_c_summary.json`, 2026-04-26).

**Setup.** A *separable per-step ranker* is any policy
Ŝ_B = top-B by ψ for some ψ : {1, …, T} → ℝ that depends on the
trajectory only through the time index t and (optionally) a coarse
state abstraction z_t = (s_t, phase(t), …) with finite |Z|. The
*mean_delta_oracle* is S_A_mean := top-B by Δ̄_t, where
Δ̄_t := 𝔼_seed Δ_t(seed) is the seed-averaged one-loop marginal gain.

**Corollary (Negative-Result, ranker class).** Let
Ranker(B) := {top-B(ψ) : ψ separable per-step, possibly z-aware} be the
ranker class at budget B. Under the hypotheses of Refined Theorem A
(A′ + A″), for any Ŝ_B ∈ Ranker(B),

    𝔼[G(Ŝ_B)] ≤ 𝔼[G(S_A_mean)] + 2 ε_R(B) + 2 σ_ξ(B),     (NRC)

where ε_R(B) and σ_ξ(B) are the rank-form calibration and
variance-form additivity slack at budget B. In particular, on OWT
Phase 2b at B = 8, the right-hand side enters the NULL band:
mean_delta_oracle's paired diff over uniform is +0.084 (at B = 16:
+0.032), and the σ_ξ(B) + ε_R(B) error band exceeds this margin.

**Proof.**

*Step 1.* Apply Refinement A′ to bound |𝔼G(Ŝ_B) − 𝔼A(Ŝ_B)| ≤ σ_ξ(B)
and similarly for S_A_mean.

*Step 2.* Apply Refinement A″ to bound 𝔼A(S_A*) − 𝔼A(Ŝ_B) ≤ 2 ε_R(B);
S_A_mean by definition realises the mean-additive-oracle and is
upper-bounded by the per-trajectory additive oracle S_A* in
expectation under any ψ that respects the seed-averaged ranking.

*Step 3.* Combine: 𝔼[G(Ŝ_B)] ≤ 𝔼[A(Ŝ_B)] + σ_ξ(B) ≤ 𝔼[A(S_A_mean)] +
2 ε_R(B) + σ_ξ(B) ≤ 𝔼[G(S_A_mean)] + 2 ε_R(B) + 2 σ_ξ(B).            ∎

**Implications.**

1. **Empirical NULL at B = 8 on OWT.** From Phase 2b paired data
   (`policy_comparison_paired.json`), mean_delta_oracle's paired diff
   over uniform is +0.130 / +0.092 / +0.084 / +0.032 at
   B = 2 / 3 / 4 / 8 / 16. The per-B σ_ξ(B) + ε_R(B) band exceeds the
   margin by B = 8 — i.e., the corollary's bound is non-vacuous and
   places mean_delta_oracle's headroom in the NULL band.

2. **Bucketed-state extension (Protocol C, 2026-04-26).** Under
   z = (s_t, phase(t)) with 12 buckets per signal,
   ε̃ / ε ∈ [0.983, 0.986] on OWT. The bucketed-state ranker class is
   bounded by the same envelope as the signal-only ranker class.

3. **Search-class exceeds the envelope.** Phase 3a's CD-G and BS-AG
   explicitly violate (NRC)'s bound by operating on schedules with
   true-G feedback, recovering 49–84 % of the Phase 2b MC-oracle
   headroom. The corollary applies to the ranker class only.

**Correctness status.** `proved under (1)–(4) of A′ + (1)–(2) of A″`.
Empirical NULL claim established on OWT Phase 2b.

**Provenance.** `[Novel — Phase 3b 2026-04-26; rescoped after Phase 3a
2026-04-20]`.

---

## Refinement A″ — Rank-Based Calibration ε_R (superseded by formal version above)

> **Superseded 2026-04-26.** The earlier heuristic statement
> `ε_R := (1 − |ρ|) · σ_Δ` with bound `≤ B · ε_R` has been replaced by
> the formal version in §"Refinement A″ — Rank-Based Calibration ε_R
> (formal, 2026-04-26)" above. The formal version uses
> `ε_R := σ_Δ · √(2(1 − ρ²) / π)` and bound `≤ 2 ε_R`, derived under
> joint-Gaussianity + linear-calibration assumptions via half-normal
> moments. The earlier heuristic is preserved here in name only;
> downstream documents should cite the formal version.

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

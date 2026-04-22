> **STATUS:** ARCHIVED
> **ARCHIVED ON:** 2026-04-22
> **SUPERSEDED BY:** `docs/thesis/theory/THEORY_STATUS.md` §"Candidate Refinements"
> **REASON:** Stress-test conclusions encoded upstream in Refinements A′/A″ and the rescoped Negative-Result Corollary. Preserved for provenance.

---

# Theory Stress Test — Theorem A and Supporting Propositions

**Author:** Claude Code (co-advisor review)
**Date:** 2026-04-19
**Scope:** Adversarial audit of Theorem A, Lemmas A1/A2, Proposition B (low-gain
gating), and Proposition C (pairwise interaction), against the Phase 1 ProSeCo-OWT
full run (`results/phase1_proseco_owt_full/summary.json`, N=50, T=64).
**Companion:** `EXPERIMENT_CRITICAL_AUDIT.md` (the experimental side of the same
argument).

---

## 0. Summary

**Where the theorem stands after Phase 1 evidence:**

| Item | Status before audit | Status after audit |
|------|---------------------|--------------------|
| Theorem A statement | `solid under assumptions` | Still solid *in the sense that the inequality holds*, but the bound is **empirically vacuous** at all B∈{4,8,16} on ProSeCo-OWT. |
| Assumption 1 (binary placement) | Unchallenged | OK for current Protocol-A/B, but the thesis question is *stronger* than this assumption permits — see §2. |
| Assumption 2 (approximate additivity η_B) | Empirically estimable | Measured η₉₅(B=8)=0.680 is ≈**40× larger** than mean |Δ_t|≈0.08 would predict under additivity. Assumption holds *numerically* with a specific η, but η is **not small** relative to G. |
| Assumption 3 (proxy calibration ε) | Empirically estimable | Measured ε_rms≈0.134 is misleadingly small: the calibration slope for entropy is −0.027 (≈ flat), so ψ carries almost no rank information about Δ. The *uniform* ε bound is satisfied because ψ is nearly constant, not because ψ is informative. |
| Proposition B (burn-in gating) | `plausible but incomplete` | Numerically consistent (T_low={0..9} at δ=50% of peak Δ). But gating this out does not help in pipeline G — see §5. |
| Proposition C (pairwise → η_B ≤ γB(B-1)/2) | `plausible but incomplete` | Measured γ₉₅=0.264; Prop C predicts η_B ≤ 7.4 at B=8. Measured η_95=0.68. Prop C's bound is **extremely loose** (≈11×). Pairwise is not the dominant interaction structure; the bound is technically correct but not usefully tight. |
| Lemma A1 | `solid` | OK but the oracle it references is **unmeasured** in our evidence (see §6). |
| Lemma A2 | `solid` | OK, but only binds A(S), not G(S). See §6. |
| Combining step (Theorem A proof) | `solid pending write-up` | Write-up has a subtle issue: the "optimum under A" is not the same as S_B*. See §4. |

**Headline stress-test findings (ordered by priority):**

1. **(Fatal for current claim, not for the theorem.)** The bound 2Bε+2η_B at B=8
   equals 3.50 but G(S_B*) is at most ≈1.2 on this data, so the theorem is
   currently saying "proxy could be up to 3.5 units worse than oracle, which may
   itself be only 1.2 units" — no non-trivial guarantee.

2. **(Structural, affects interpretation.)** The proxy calibration quantity ε_rms
   is small not because the proxy is good but because the proxy is uninformative.
   This is a classical "flat predictor has small residual" failure mode. The
   theorem does not currently distinguish *informative-low-ε* from
   *uninformative-low-ε* — it should.

3. **(Structural.)** The theorem treats the additive surrogate A(S) as the pivot,
   but the *measurement* of Δ_t we use (single-loop, base-trajectory) does not
   equal the *ingredients* of G(S) (composed loops, chained trajectory). The
   proof sketch implicitly assumes the Δ_t's are the same object in A and G;
   that is only true under additivity (Assumption 2). The two η_B applications
   paper over this, but it means the theorem is really a bound on *whichever
   base-trajectory quantity Δ_t we measured*, not on a canonical object.

4. **(Loose bound.)** Proposition C's γB(B−1)/2 predictor of η_B is 11× looser
   than the measured η_95 on this data. Either higher-order interactions cancel
   (so the pairwise sum overcounts), or pair selection (95th percentile of a
   sample of 300 pairs) undersamples the tail. Either way, Prop C is not a
   useful tool for predicting η_B and should be flagged.

5. **(Missing object.)** The theorem's S_B* is never operationally defined in the
   pipeline. `policy_comparison.oracle` is mean-field top-B of Δ̄_t, which is
   *not* S_B*. The "regret" we report is w.r.t. a heuristic, not an oracle.

Each of these is expanded below.

---

## 1. Theorem A as stated

(Reproduced for self-containment; see `research/candidate_theorems.md` for the
canonical version.)

Under (1) binary placement k_t ∈ {0,1}, (2) |G(S) − Σ_{t∈S} Δ_t| ≤ η_B for all
|S|≤B, (3) |Δ_t − ψ(s_t)| ≤ ε for all t:

        G(S_B*) − G(Ŝ_B) ≤ 2 B ε + 2 η_B.

Supporting: Lemma A1 (Δ-top-B is A-optimal under exact additivity), Lemma A2
(proxy-top-B vs Δ-top-B costs 2Bε under exact additivity), combining step.

---

## 2. Assumption 1 — binary placement

**Claim.** At most one corrector loop per step.

**Audit.**

- Matches the current ProSeCo-OWT pipeline: `evaluate_schedule` runs exactly one
  corrector loop on each t ∈ S.
- Is *narrower* than the thesis question. The thesis asks about allocation of
  B corrector **NFEs**, not B distinct steps. A more natural formalization is
  k_t ∈ {0,1,2,…} with Σ k_t = B. The binary restriction avoids combinatorial
  difficulty but makes the comparison to uniform artificially constrained:
  uniform with k=1 everywhere trivially uses the full budget when B = T, but
  below B < T uniform "interpolates" by spacing.
- **Risk the assumption creates:** If in practice the best use of B NFEs is
  to pile 2 loops on the single highest-Δ step rather than spread them, the
  theorem cannot see this regime.
- **Recommendation:** Either (a) retain the binary restriction but also run
  a control experiment where k_t ∈ {0,1,2} and measure whether the additional
  loops-per-step dominates; or (b) extend the theorem to Σ k_t = B with
  monotone-decreasing per-loop returns at a fixed t (which is a separate
  empirical claim that would need its own check).

**Verdict.** The assumption is clean for the current Protocol-A/B, but an honest
thesis should at minimum acknowledge the restriction and motivate why binary
placement is the right object. At present the thesis calls this an "assumption"
without motivation. Add a short paragraph in the theorem environment.

---

## 3. Assumption 2 — approximate additivity η_B

**Claim.** ∃η_B such that |G(S) − Σ_{t∈S} Δ_t| ≤ η_B for all |S| ≤ B.

**Measurement** (full run):

| B  | η̄_B (mean) | η₉₅ (95th pct) | schedules sampled |
|----|------------|----------------|--------------------|
| 4  | 0.175      | 0.413          | 30 |
| 8  | 0.344      | **0.680**      | 30 |
| 16 | 0.771      | 1.357          | 30 |

**Scale context** (measured in same units as G):

- Peak mean Δ ≈ 0.157 (at t=23)
- Uniform G at B=8 (true): 0.758
- Σ of top-8 mean Δ ≈ ~1.2 (additive-surrogate "G_oracle_estimate")
- Claimed "entropy_bot_B surplus" over uniform under A: +0.199

**Verdict on the assumption itself.** The statement "∃η_B" is trivially true for
any finite data; the question is *how large* η_B has to be. We measure
η₉₅(B=8) ≈ 0.68. That is:

- ≈ 4× the mean Δ at the peak step
- ≈ 0.9× the true uniform G
- ≈ 3.4× the claimed entropy_bot_B surplus

So the "approximate additivity" is **quantitatively weak** — the error when
going from A(S) to G(S) is of the same order as the entire quality gain.

**Why this matters for the theorem.** The theorem's RHS 2η_B is 1.36 at B=8.
This single term is larger than any observed G in the policy_comparison table
(uniform 0.76, middle 0.65, oracle 0.64, all others ≪0.6). So the theorem is
saying "proxy-regret is at most 2Bε + 2η_B = 3.5" at a budget where the entire
observed gain from uniform itself is only 0.76. The bound cannot rule out
*any* policy ranking.

**Why the assumption might still be salvageable.**

- At small B (B=2 or B=3), Proposition C predicts η_B ~ γ·B(B−1)/2, so
  η_2 ~ 0.26 and η_3 ~ 0.79. Using the direct measurement of η_B at B=2 and
  B=3 (not currently computed) the theorem might be non-vacuous in the very
  small budget regime.
- The *expected* version of the assumption, 𝔼|G(S) − A(S)| ≤ η̄_B, uses
  mean not 95th percentile. η̄(B=8) = 0.344, which is half the uniform G.
  Still not small, but closer. Expectation-version of Theorem A might be
  the right first delivery.
- If F were swapped for a less noisy quality functional (§ EXPERIMENT_CRITICAL_AUDIT
  §5), the underlying variance of Δ and therefore η_B may shrink.

**Recommendation.** Record η_B as "large relative to target gain" in the honesty
ledger. State a corollary corresponding to the expectation form, and note that
the uniform-bound form of Theorem A is vacuous on this data.

---

## 4. Assumption 3 — proxy calibration ε

**Claim.** |Δ_t − ψ(s_t)| ≤ ε for all t.

**Measurement** (full run, entropy signal):

- ε_rms = 0.134
- ε_max = 0.785
- Spearman(ψ, Δ) = −0.191 ± 0.252
- Linear calibration fit: ψ̂(s) = −0.027·s + 0.108  (slope ≈ 0, intercept ≈ mean Δ)

The same pattern holds for inverse-margin (slope −0.203, Spearman −0.200) and
quality-mass (slope −0.209, Spearman −0.185).

**What this means.**

- **ε_rms is small because ψ is flat.** If we set ψ(s) = E[Δ] constant, the
  residual |Δ − ψ| has RMS equal to σ(Δ). Empirically σ(Δ) ≈ 0.13 (mean Δ
  varies between −0.03 and 0.16 across t, and within-t variance across
  trajectories is also ~0.1). So ε_rms ≈ 0.134 is essentially what we get
  from a *constant* proxy. The signal adds almost nothing.
- **ε_max = 0.785** is 6× ε_rms, indicating heavy tails. For the uniform-bound
  form of Theorem A we should use ε_max, not ε_rms. Plugging ε_max = 0.785
  into the theorem: 2·8·0.785 + 2·0.680 = 13.9. Even more vacuous.
- **Spearman ≈ −0.19** means the proxy is anti-correlated with Δ. The theorem
  does not care about correlation — it uses absolute residual — but the
  Ŝ_B construction (top-B by ψ) depends on rank. A rank correlation of 0.19
  gives essentially random top-B selection, so Ŝ_B ≈ random-B in the limit
  of weak correlation. We could formalize this: if Spearman(ψ, Δ) = ρ, then
  E[A(Ŝ_B)] ≈ ρ · A(S_B*) + (1−ρ) · A(uniform) under Gaussian assumptions.
  With ρ ≈ 0.19 (or −0.19, with inverted direction), Ŝ_B retains very little
  of the oracle signal.

**Structural issue the theorem misses.** The ε bound conflates two failure modes:

1. **Low information in ψ.** Even a perfect calibration fit is flat because ψ
   doesn't predict Δ — small ε_rms, small Spearman.
2. **Miscalibrated direction.** ψ predicts Δ with the wrong sign or slope —
   small Spearman, but linearly corrigible by rescaling ψ.

Theorem A's uniform ε bound treats (1) and (2) as equivalent. In practice we
should report the *informativeness* of ψ separately, e.g. by the fraction of
variance of Δ explained by ψ, or by the effective Spearman over the selection
set. At ρ=0.19 the proxy is nearly useless for top-B selection and the theorem
should flag this rather than hide it inside a small-looking ε.

**Recommendation.** Split ε into two components:

- ε_R (rank error): how well ψ ranks Δ_t. Estimable as
  (1 − |Spearman|) × max |Δ − median(Δ)|, or by a direct
  bottom-B-by-ψ recall of top-B-by-Δ.
- ε_S (scale error): how well a linear rescale of ψ matches Δ. Estimable as
  RMS residual after ψ̂ = aψ + b.

The relevant quantity for Top-B selection is ε_R; ε_S is only relevant for
calibration of Δ̂ magnitudes. Reformulate Theorem A in terms of ε_R, or prove
an auxiliary result that relates the two.

---

## 5. Proposition B — low-gain gating

**Claim.** If T_low ⊆ {t : Δ_t ≤ δ} and gating excludes ≤ B steps, then
G(Ŝ_B) − G(Ŝ_B^{gated}) ≤ Bδ + 2η_B.

**Measurement.**

- t_first_positive_delta = 3 (first t where mean Δ > 0)
- T_low at 50% peak = {0..9}, at 30% peak = {0..8}
- Peak Δ ≈ 0.157 at t=23

**Audit.**

The proposition is structurally sound and the measurement identifies a clean
T_low. Two stress-test concerns:

1. **The "benign gating" bound already includes 2η_B.** At B=8 with gating of
   10 steps, the bound is 8·(0.5·0.157) + 2·0.680 = 2.0. Again the η_B term
   dominates. So the proposition doesn't add headroom over Theorem A; its
   proof strategy is the same exchange-plus-additivity.

2. **Empirically, none of the gated policies beat uniform.** In the full run
   `entropy_burn_in_gated` (which is entropy_top_B restricted to t ≥ 10) has
   the same pipeline G as `entropy_top_B` at B=4, 8, 16. At B=8 they both
   give G=0.066 (vs uniform 0.758). Gating doesn't rescue top-B-by-entropy
   because the underlying proxy is weak (§4).

**Verdict.** Prop B is technically correct and the T_low range is meaningful,
but on its own it cannot rescue a weak proxy. It is *conditional* on having a
useful ψ outside T_low, which we do not have empirically.

**Recommendation.** Keep Prop B but note that it is inert when the proxy outside
T_low is uninformative. Combine with §4's ε_R splitting.

---

## 6. Proposition C — pairwise interaction

**Claim.** If |ξ_{t,t'}| ≤ γ for all pairs, then η_B ≤ γ·B(B−1)/2.

**Measurement.**

- γ₉₅ = 0.264 (95th percentile of |ξ| over 300 pairs)
- γ_mean = 0.083
- γ_max = 0.603

**Prop C predictions vs measured η:**

| B  | Prop C predicts η_B ≤ | Measured η_95 |
|----|-----------------------|---------------|
| 4  | γ₉₅·6 = 1.58          | 0.41          |
| 8  | γ₉₅·28 = 7.39         | 0.68          |
| 16 | γ₉₅·120 = 31.6        | 1.36          |

**Audit.**

- The bound is technically correct (triangle inequality); it is **very loose**.
  Measured η is 4×–23× smaller than Prop C's prediction.
- This means the pairwise interactions at different (t,t') pairs
  **partially cancel** when summed over S — i.e., the interaction field has
  mean ≈ 0 and variance that averages out over O(B²) terms. Prop C's worst-case
  triangle bound ignores this cancellation.
- **Sign of concern:** Prop C as stated is not useful as an empirical
  η-estimator. Using it in a formal bound produces numbers that are 10× too
  pessimistic.
- However, an **expectation** version with √-cancellation would fit:
  𝔼 η_B ≤ γ·B(B−1)/2 under worst-case, but a √B-scaling is more consistent
  with the measurements. Specifically: η_B ≈ γ·B (linear in B) fits the
  measured 0.41, 0.68, 1.36 at B=4,8,16 reasonably, not B(B−1)/2.

**Recommendation.** Either

- Keep Prop C as a *worst-case* upper bound, explicitly note it is loose by
  ~10× on this data, and do not use it to replace direct η_B measurement.
- Or introduce a sharper Prop C′: under pairwise interactions with
  𝔼ξ_{t,t'} = 0 and std σ_ξ, we have σ(A(S) − G(S)) ≈ σ_ξ·√(B(B−1)/2) ≈
  σ_ξ·B/√2. This would explain the observed scaling and is falsifiable.

This is a thesis-level deliverable: the Prop C refinement could be a publishable
original result.

---

## 7. Combining-step issue (proof-sketch-level)

The current proof sketch (research/candidate_theorems.md Theorem A Combining Step)
chains:

1. |G(S_B*) − A(S_B*)| ≤ η_B          (Assumption 2)
2. A(S_B*) ≤ A(S_A*) where S_A* := argmax A           (Lemma A1 applied to A)
3. A(S_A*) − A(Ŝ_B) ≤ 2Bε             (Lemma A2 applied to A)
4. |A(Ŝ_B) − G(Ŝ_B)| ≤ η_B            (Assumption 2)

giving G(S_B*) − G(Ŝ_B) ≤ 2Bε + 2η_B.

**Subtle issue.** Step 2 uses A1 on A, not on G. That is fine *if* the goal is
to bound A-regret, but we want G-regret. The chain proceeds through A and
returns to G via two η_B applications. Because of this, the theorem is
effectively a statement about

        G(Ŝ_B) ≥ G(S_A*) − 2Bε − 2η_B

where S_A* = argmax_{|S|=B} A(S). It is **not** a statement about S_B* itself.
The inequality G(S_B*) ≥ G(S_A*) is what we need, and this only holds up to
2η_B: G(S_B*) ≤ G(S_A*) + 2η_B and vice versa.

Writing this out:

- G(S_B*) ≤ A(S_B*) + η_B ≤ A(S_A*) + η_B (by def. of S_A*) ≤ G(S_A*) + 2η_B.

So the full chain is:

        G(S_B*) − G(Ŝ_B) ≤ (G(S_B*) − A(S_A*)) + (A(S_A*) − A(Ŝ_B)) + (A(Ŝ_B) − G(Ŝ_B))
                         ≤ (2η_B)            + (2Bε)                + (η_B)
                         = 2Bε + 3η_B.

The currently stated bound 2Bε + 2η_B **undercounts η by one application**.
The correct bound is 2Bε + 3η_B (using uniform bounds). At B=8 on our data
this moves 3.50 → 4.18. Still vacuous, but the discrepancy should be fixed in
the write-up.

Alternatively, if we use a single η_B on |G(S_B*) − G(Ŝ_B) through A|, the
composition gives 2Bε + 2η_B only under the assumption that the same η_B
simultaneously bounds all three differences. That is allowed if η_B is defined
as sup over all |S|≤B of |G−A|, which is how we state it — then the cleanest
version is 2Bε + 2η_B by packing the two A1 applications into a single use.
Under this convention Lemma A1 applied to A gives the optimality S_A* that
minimizes both |G-A|'s at the same η. So the original bound is defensible but
the proof should state this packing explicitly.

**Recommendation.** Rewrite the combining step with explicit constants and
choose whether to state 2η or 3η. Either is fine; both should match the η
defined in Assumption 2.

---

## 8. The missing S_B* — what exactly is the "oracle"?

**Issue.** The theorem references S_B* = argmax_{|S|=B} G(S), but the experiment
never computes this. Instead, `policy_comparison.oracle` is top-B of the mean
Δ̄_t profile, i.e. S_A* evaluated on trajectory-averaged signals.

**Consequence.** We cannot report "proxy-regret vs oracle" because we don't know
what the oracle is. The 0.637 value we call "oracle G" at B=8 is a heuristic
upper-envelope; the true S_B* could plausibly reach higher.

**How to recover.** Per-trajectory, for B=4 we can enumerate all C(T, 4)≈635k
subsets at T=64 — expensive but not impossible for a single trajectory, so
doable on ~10 trajectories. For B=8 we cannot enumerate; instead Monte-Carlo
300–1000 random schedules per trajectory and report the max G observed. This
is a lower bound on G(S_B*) but a honest one.

**Recommendation.** Add a "true oracle" measurement to Phase 2 protocol. Without
it, no statement about proxy-regret can honestly appear in the thesis.

---

## 9. What remains defensible as thesis theory after this audit

The following is *defensible* after the stress test, in order of safety:

1. **Lemma A1** (Δ-top-B is A-optimal under exact additivity) — standard and
   correct. Keep.
2. **Lemma A2** (proxy-top-B vs Δ-top-B regret is 2Bε under exact additivity)
   — standard exchange argument, correct. Keep.
3. **Proposition B** (benign gating) — correct under its assumptions; note
   empirical inertness when ψ is weak (§5). Keep as structural result.
4. **Theorem A** (2Bε + 2η_B regret) — correct as an inequality, but
   *empirically vacuous* on ProSeCo-OWT. Keep, **but** downgrade status to
   "correct under assumptions; currently vacuous on the only tested backend"
   and add an explicit empirical hypothesis for when it would become useful
   (η_B ≤ 0.1 × G, ε_rms ≤ 0.05 × G_peak, B ≤ 4).
5. **Proposition C** (pairwise interaction → η bound) — correct but very loose
   (11× at B=8). Either keep and flag the looseness, or replace with a sharper
   variance-based version.
6. **Combining-step proof** — needs rewrite to clarify 2η vs 3η and the S_A*
   packaging. No substantive issue, just careful accounting.
7. **Stretch Appendix C2** (contraction route) — untouched by this audit;
   remains conjectural.

---

## 10. What *new* theorems the current data *could* support

Three positive directions suggested by the stress test:

### 10.1 Variance-form proxy-regret (novel, small proof)

**Claim.** Under pairwise interactions ξ_{t,t'} with 𝔼ξ = 0 and var ξ ≤ σ_ξ²,

        Var(G(S) − A(S)) ≤ σ_ξ² · C(B,2) ≈ σ_ξ² B²/2

and therefore

        𝔼|G(S) − A(S)| ≤ σ_ξ · B / √2.

This is √B-tighter than Prop C's triangle bound and matches the observed
η scaling. Plugging into an expected-regret form of Theorem A gives

        𝔼[G(S_B*) − G(Ŝ_B)] ≤ 2B𝔼ε + 2σ_ξ·B/√2.

At B=8 with σ_ξ ≈ 0.1 this is 2·8·0.134 + 2·0.1·5.66 = 2.14 + 1.13 = 3.27 —
still larger than G but closer. At B=4 the corresponding number is 1.07 + 0.57
= 1.64, also larger than G (0.54) but with smaller multiplicative slack.

**Proof sketch.** Standard; the variance of a sum over C(B,2) independent terms
with variance σ_ξ². The (non-)independence is the catch; will need a
mixing-type assumption. Start with IID pairs, then relax.

### 10.2 Rank-based calibration (novel, small proof)

**Claim.** If Spearman(ψ, Δ) = ρ and Δ has std σ_Δ, then
        𝔼[A(S_A*) − A(Ŝ_B)] ≤ (1 − ρ) · B · σ_Δ  (heuristic, Gaussian assumption)

so ε_R := (1−ρ)σ_Δ is the relevant calibration quantity for top-B selection.

On current data: (1 − |−0.191|)·σ_Δ ≈ 0.81·0.13 ≈ 0.11. Similar to ε_rms but
with a clearer interpretation: *a perfectly informative proxy (ρ=1) has
ε_R = 0; a constant proxy has ε_R = σ_Δ*.

**Proof sketch.** Gaussianize, compute order statistics. Not fully rigorous but
a thesis-scale contribution.

### 10.3 Negative result (honest)

**Claim.** On (ProSeCo-OWT, T=64, F = −GPT-2 NLL on 512 tokens), the measured
ε, η_B, γ imply that Theorem A's bound is vacuous for all B ∈ {1,…,T}, and
pipeline G(S) measurements show no signal-adaptive policy beats uniform at
B=4 or B=8.

This is a real thesis contribution if accompanied by:

- a characterization of which (backbone, corrector, F) regimes have
  η_B/G ≪ 1 (where the bound would become useful);
- a proof sketch that reduces to "when pairwise corrector interactions are
  small, the additive surrogate is accurate";
- at least one positive experiment on a different (backbone, corrector, F)
  triple (e.g., MDLM + simple resample + MAUVE) that exhibits the
  non-vacuous regime.

---

## 11. Honesty-ledger updates required

Changes to `docs/thesis/theory/THEORY_STATUS.md` after this audit:

- Flip "Theorem A — proved under assumptions" status summary to include
  "empirically vacuous on the only tested system (ProSeCo-OWT, Phase 1)".
- Add "Proposition C loose by ≈11× at B=8" as a logged limitation.
- Add the combining-step 2η vs 3η bookkeeping note as an open write-up item.
- Log the variance-form and rank-based refinements (§10.1, §10.2) as
  candidate novel theorems.

Changes to `research/candidate_theorems.md`:

- Under Theorem A "Risk of being vacuous": move from "medium" to **"currently
  realized"** with reference to `phase1_proseco_owt_full`.
- Under Proposition C "Correctness status": note empirical looseness.

Changes to `research/open_questions.md`:

- Add Q-new: "Does a √B-scaling variance bound on η_B match Phase 1 data better
  than Prop C's B² bound?" (link to §10.1)
- Add Q-new: "Is a rank-based ε (ε_R) the right calibration quantity for
  top-B selection?" (link to §10.2)

---

## 12. Questions that cannot be answered until Phase 2

This audit can identify issues with the theorem-to-measurement mapping, but
the following cannot be closed without a better Phase 2 experiment:

- Is η_B this large on a different (F, backbone, corrector) triple? (→ Phase 2
  must vary at least one of these.)
- Is ε_R small for any signal on *any* tested system? (→ Phase 2 must measure
  ε_R directly, not ε_rms.)
- What is the true G(S_B*)? (→ Phase 2 must run per-trajectory Monte-Carlo
  oracle.)
- Is the entropy-direction "inversion" (negative ρ on ProSeCo-OWT) a
  backbone-specific phenomenon? (→ Phase 2 must include ≥ one other backbone.)

All four are explicitly wired into `NEXT_PHASE_EXPERIMENT_PLAN.md` (upcoming
Workstream C).

---

## 13. One-paragraph summary for supervisor

> Theorem A's inequality is correct under its stated assumptions, but on the
> only experimental system we have calibrated (ProSeCo-OWT, T=64, F=−GPT-2 NLL
> on 512 tokens), the bound 2Bε + 2η_B is ≈3.5 at B=8 while any plausible
> G(S_B*) is ≤1.2 — so the bound is vacuous. The proxy calibration ε looks
> small (0.134) only because the proxy is nearly uninformative (Spearman ≈
> −0.19); a rank-sensitive calibration ε_R exposes this. The additivity slack
> η_B is large relative to G because single-loop Δ measured on the base
> trajectory does not compose with other loops on the corrected trajectory —
> this is structural, not a measurement error. Proposition C's worst-case
> pairwise bound is 10× too loose to predict η_B, suggesting a variance-form
> refinement is the right next theorem. Lemmas A1, A2, and Proposition B are
> correct and retain value as structural results. No current evidence is
> inconsistent with Theorem A; all current evidence is inconsistent with
> using Theorem A as a non-vacuous bound.

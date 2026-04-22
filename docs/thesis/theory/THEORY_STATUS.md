> **STATUS:** CANONICAL
> **LAST VERIFIED:** 2026-04-22
> **SCOPE:** Theorem A and refinements A′/A″; Negative-Result Corollary; active theory items.

---

# Theory Status — Canonical Summary

> **[POST–PHASE-3A — 2026-04-20]** Phase 3a (job 479941, K=30 paired,
> combinatorial baselines on ProSeCo-OWT) closes the Phase 3 empirical
> contract with a **search-class positive** that refines the Phase 2b
> verdict. Both **CD-G** (coordinate descent, true-G feedback) and **BS-AG**
> (beam search, cheap-A ranking + true-G rollouts) PASS at every B ∈ {2, 3,
> 4, 8}; CD-G recovers 74–84 % and BS-AG 49–64 % of the Phase 2b MC-oracle
> +0.45 paired headroom, and *both still PASS at B = 8* — the budget at
> which `mean_delta_oracle` itself enters the NULL band. The thesis story
> tightens to:
>
> > **Fixed-budget corrector allocation is a combinatorial trajectory-control
> > problem; cheap greedy rankers are the wrong solution class for it. Search
> > procedures over schedules — even ones that use the cheap-A surrogate to
> > prune candidates — recover most of the oracle headroom.**
>
> Two of the three candidate refinements remain empirically anchored as in the
> Phase-2b banner; the third (Negative-Result Corollary) is **rescoped to the
> greedy/separable-ranker policy class** (search-class is excluded by Phase 3a):
>
> - **Refinement A′ (variance-form η_B):** σ_ξ measured directly from MC
>   residuals G − A on 9000 (seed, schedule) pairs; replaces Prop C's
>   γB(B−1)/2 with the √B-tighter `𝔼|G−A| ≤ σ_ξ·√B/√2`.
> - **Refinement A″ (rank-based ε_R):** pooled Spearman ρ(A, G) decays
>   monotonically 0.66 (B=2) → 0.39 (B=16); ε_R(B) := (1−|ρ_B|)·σ_Δ is the
>   *operational* calibration measure that survives the data and replaces
>   the L∞ ε used in Theorem A.
> - **Negative-Result Corollary (rescoped to ranker class):** any policy of
>   the form `Ŝ_B = top-B(ψ)` for separable per-step `ψ` is upper-bounded by
>   the `mean_delta_oracle` envelope, which on ProSeCo-OWT enters the NULL
>   band by B = 8. The corollary characterises the **ranker class only**;
>   Phase 3a's CD-G and BS-AG explicitly exceed this envelope by operating on
>   schedules, so the corollary does not generalise to "informed scheduling".
>
> Phase 3 audit (`docs/thesis/experiments/PHASE3_DIRECTION_AUDIT.md`) **rejects
> a PRISM pivot** with refined reason: PRISM is a learned per-token quality
> signal, hence a member of the ranker class bounded by the corollary; the
> recoverable structure does not factor through any per-step score. Theorem A
> remains formally correct; its uniform L∞ form is empirically inert, but A″'s
> rank-based ε_R is the version that lands and survives Phase 3a's
> search-class positive (the bound applies to ranker policies, not search).

**Updated:** April 2026. Source of truth for thesis chapter 6. Chronological record lives in `research/proof_worklog.md`, `research/candidate_theorems.md`, `research/proof_ledger.md`, `research/open_questions.md`.

## Main Theorem Target

**Theorem A (Proxy-Regret Bound for Top-B Scheduling).** Let the predictor schedule fix states Z_0, ..., Z_T. For each step t define the one-loop marginal gain Δ_t := F(y_t^{+1}) − F(y_base), where F is a scalar trajectory-level quality functional (e.g., negative LM-NLL or MAUVE), y_t^{+1} is the generation obtained by applying exactly one corrector loop at step t, and y_base is the uncorrected baseline. For S ⊆ {1, ..., T} with |S| = B let G(S) := F(y^S) − F(y_base). Let S_B* := argmax_{|S|=B} G(S) and Ŝ_B := top-B by proxy score ψ(s_t).

Under the three assumptions
(1) **binary placement:** k_t ∈ {0, 1};
(2) **approximate additivity:** there exists η_B ≥ 0 with |G(S) − ∑_{t ∈ S} Δ_t| ≤ η_B for every |S| ≤ B;
(3) **proxy calibration:** |Δ_t − ψ(s_t)| ≤ ε for all t;

we have

    G(S_B*) − G(Ŝ_B) ≤ 2 B ε + 2 η_B.

**Provenance.** `[Adapted from GPT Pro assessment v2]`. Proxy-regret framing, explicit ε/η_B decomposition, and oracle top-B comparison are GPT Pro v2 contributions; binary-placement formalization is inherited from the earlier Candidate 1.

**Status.** `solid under assumptions (1)–(3), pending clean write-up`. Bound is non-vacuous iff 2Bε + 2η_B < G(Ŝ_B) for the budget of interest. Empirical verification of (2) and (3) is the central experimental question.

## Supporting Lemmas

**Lemma A1 (Oracle top-B under exact additivity).** If G(S) = ∑_{t ∈ S} Δ_t exactly, then S_B* is the set of B indices with largest Δ_t. Trivial. Status `solid under assumptions`. Provenance `[Borrowed — standard resource allocation; Novel only in application]`.

**Lemma A2 (Calibration regret in additive regime).** Under η_B = 0 and |Δ_t − ψ(s_t)| ≤ ε, ∑_{t ∈ S_B*} Δ_t − ∑_{t ∈ Ŝ_B} Δ_t ≤ 2Bε. Proof by swap-by-swap exchange: for any t ∈ S_B* \ Ŝ_B and t' ∈ Ŝ_B \ S_B*, ψ(s_{t'}) ≥ ψ(s_t) and |Δ − ψ| ≤ ε give Δ_t − Δ_{t'} ≤ 2ε; there are at most B swaps. Status `solid under assumptions`. Provenance `[Adapted from GPT Pro assessment v2]`.

## Supporting Propositions

**Proposition B (Low-gain-region exclusion / burn-in gating).** Suppose there is T_low ⊆ {1, ..., T} with Δ_t ≤ δ for all t ∈ T_low, and |T_low| ≤ B. Let Ŝ_B^{gated} := top-B by ψ on {1, ..., T} \ T_low. Then G(Ŝ_B) − G(Ŝ_B^{gated}) ≤ Bδ + 2η_B. Status `plausible but incomplete` — follows from the same exchange-plus-additivity argument as Theorem A; empirical identification of T_low is the substantive step. Replaces the earlier MI-monotonicity claim, which is not generally true in u_t. Provenance `[Adapted from GPT Pro assessment v2]`.

**Proposition C (Pairwise interaction bound).** If G(S) admits a pairwise decomposition G(S) = ∑_{t ∈ S} Δ_t + ∑_{{t,t'} ⊂ S} ξ_{t,t'} with |ξ_{t,t'}| ≤ γ, then |G(S) − ∑_{t ∈ S} Δ_t| ≤ γ · B(B−1)/2, so η_B ≤ γ B(B−1)/2 and Theorem A yields regret ≤ 2Bε + γB(B−1). Status `plausible but incomplete` — the pairwise model is a second-order approximation; higher-order interactions could dominate and need a triple diagnostic. Provenance `[Adapted from GPT Pro assessment v2]`.

## Assumptions (recap)

Three, and only three, for Theorem A: (1) binary placement; (2) approximate additivity with slack η_B; (3) proxy calibration with slack ε. Proposition B adds a low-gain region parameter (δ, T_low). Proposition C replaces assumption (2) with an explicit pairwise decomposition and bound γ.

## Empirically Calibrated Quantities

| Quantity | Defined as | Measured via |
|---|---|---|
| ε | sup_t \|Δ_t − ψ(s_t)\| (or expected version) | Protocol A |
| η_B | sup_{\|S\| ≤ B} \|G(S) − ∑ Δ_t\| | Protocol B (sampled schedules) |
| γ | sup_{t,t'} \|ξ_{t,t'}\| | Protocol B pairwise diagnostic |
| δ, T_low | Δ_t ≤ δ on T_low | Protocol A, inspect Δ_t across t |

All four measured on ProSeCo annealed-refinement corrector with MDLM backbone (`mdlm.ckpt` on OpenWebText). Signals (entropy H_t, inverse margin 1−M_t, quality mass Q_t) computed over unmasked positions R_t.

## Biggest Remaining Proof Gap

The combining argument (Theorem A from Lemma A1 + Lemma A2 + approximate additivity) is sketched in `proof_worklog.md` Entry 6 but not yet a clean LaTeX write-up. Two steps still need careful notational treatment: (i) the 2ε-per-swap exchange from Lemma A2 applied to the additive surrogate A(S) := ∑_{t ∈ S} Δ_t rather than to G directly; (ii) the two applications of approximate additivity (2) connecting A(S_B*) and A(Ŝ_B) back to G(S_B*) and G(Ŝ_B). Empirical verification of additivity (η_B small enough that the bound is non-vacuous) is orthogonal and gated on Phase 1 ProSeCo results.

## No Longer Mainline

**Stretch Appendix C2 (E_fact contraction under correctors).** E_fact^{corrected}(t, k_t) ≤ ρ(t)^{k_t} · E_fact(t). Demoted because (i) the Ascolani, Lavenant & Zanella 2024 reference was miscited as systematic-scan Gibbs under log-concavity — it actually treats random-scan Gibbs under conditional-independence / positive-curvature hypotheses, not classical log-concavity; (ii) masked-text conditionals are discrete and not log-concave, so the transfer is broken; (iii) no mechanism yet links ρ(t) to any trajectory signal s_t. Preserved as stretch material only; pursue only if Ascolani et al. 2024 + Denoising Entropy Bounds yield a discrete non-log-concave contraction framework. Status `conjecture`.

**Stretch Appendix C3 (Confidence margin as alternative proxy).** Not a theorem. Under Theorem A this is purely the empirical question "which signal ψ minimizes ε?" Treated as Protocol A calibration, not as a theoretical claim. Status `empirical only`.

## Candidate Refinements (post-stress-test, 2026-04-19)

The April 2026 stress test (`THEORY_STRESS_TEST.md` §10) identified that Theorem A's uniform bound is empirically vacuous on the only system tested (ProSeCo-OWT) and proposed two thesis-scale refinements that the data could already support. Both are listed as candidate theorems pending Phase 2 evidence.

**Refinement A′ (variance-form additivity slack).** Replace Proposition C's worst-case triangle bound η_B ≤ γ·B(B−1)/2 with a variance bound. Under pairwise interactions ξ_{t,t'} with 𝔼ξ = 0 and var ξ ≤ σ_ξ², the additivity error has Var(G(S) − A(S)) ≤ σ_ξ²·B(B−1)/2 ≈ σ_ξ²·B²/2, hence 𝔼|G(S) − A(S)| ≤ σ_ξ·B/√2. Plugging into an expectation form of Theorem A,

    𝔼[G(S_B*) − G(Ŝ_B)] ≤ 2 B 𝔼ε + 2 σ_ξ · B / √2.

This is √B-tighter than Prop C and matches the observed scaling η_95 ∈ {0.41, 0.68, 1.36} at B ∈ {4, 8, 16} (Phase 1 ProSeCo-OWT). Status `plausible but incomplete` — requires a mixing/cancellation hypothesis on (ξ_{t,t'}). Provenance `[Novel — derived from THEORY_STRESS_TEST §10.1]`.

**Refinement A″ (rank-based calibration ε_R).** The empirical observation is that ε_rms ≈ 0.134 is small not because ψ is informative but because ψ is nearly flat (Spearman ≈ −0.19). Theorem A as stated cannot distinguish "informative-low-ε" from "uninformative-low-ε". Define

    ε_R := (1 − |Spearman(ψ, Δ)|) · σ_Δ

as the rank-based calibration error. Under a Gaussian heuristic, the proxy-regret on the additive surrogate satisfies 𝔼[A(S_A*) − A(Ŝ_B)] ≤ B · ε_R, with ε_R = 0 when ρ = ±1 (perfect rank predictor) and ε_R = σ_Δ when ρ = 0 (constant predictor). Status `heuristic only` — pending a clean order-statistics derivation. Provenance `[Novel — derived from THEORY_STRESS_TEST §10.2]`.

**Negative-result corollary (rescoped to ranker class, 2026-04-20).** On (ProSeCo-OWT, T = 64, F = −GPT-2 NLL on 512 tokens), the measured (ε, η_B, γ) imply 2Bε + 2η_B > observed G(S_B*) for all B ∈ {4, 8, 16}. The theorem holds; the bound is currently inert. **The corollary is now scoped to the greedy/separable-ranker class only.** Phase 2b empirically saturates the ranker envelope (`mean_delta_oracle` enters the NULL band at B = 8, top-K MC ∩ oracle Jaccard ≈ 1.5× random, top-K MC internal Jaccard ≈ bottom-K). Phase 3a (job 479941) shows that *non-ranker* procedures — coordinate descent CD-G and beam search BS-AG over schedules — exceed this envelope, recovering 49–84 % of the MC-oracle headroom and PASSing at B = 8. The corollary therefore characterises the *wrong solution class* (cheap separable rankers) rather than "informed scheduling in general"; thesis framing tightens to *fixed-budget corrector allocation is a combinatorial trajectory-control problem; cheap greedy rankers are the wrong solution class for it*. Provenance `[Novel framing — THEORY_STRESS_TEST §10.3, rescoped after Phase 3a]`.

## Combining-step bookkeeping (write-up note)

THEORY_STRESS_TEST §7 flagged a small accounting issue in the original proof sketch: the chain G → A → A → G uses two η_B applications and yields 2Bε + 2η_B *if* η_B is defined as sup_{|S|≤B} |G(S) − A(S)|; under a strict pair-of-G-applications version the bound is 2Bε + 3η_B. The two are reconcilable by stating the η definition explicitly. The clean write-up (LaTeX) must either (a) define η as the uniform sup and state 2η, or (b) decouple the two G applications and state 3η. Either is fine; the choice should be consistent with the symbol used in Assumption 2. Resolved as a write-up choice, not a substantive issue.

## Honesty Ledger

| Object | Proved / Heuristic / Empirical / Conjecture |
|---|---|
| Theorem A bound | `proved under assumptions (1)–(3)`; **empirically vacuous on the only tested system (ProSeCo-OWT, Phase 1, all B ∈ {4, 8, 16})**; non-vacuity hypothesis recorded below |
| Lemma A1 | `proved` (trivial) |
| Lemma A2 | `proved` (exchange) |
| Proposition B | `proved under assumptions` once exchange/additivity from Theorem A is written; T_low existence is `empirical`; **inert when ψ outside T_low is uninformative (Phase 1 ρ ≈ −0.19)** |
| Proposition C | `proved under pairwise decomposition assumption`; **bound is loose by ≈11× at B=8 vs measured η_95 on Phase 1 — pairwise interactions partially cancel rather than triangle-add** |
| Approximate additivity (2) | `empirical` — measured as η_B via Protocol B; **on Phase 1 ProSeCo-OWT, η_95(B=8) = 0.68 ≈ 0.9× true uniform G** |
| Proxy calibration (3) | `empirical` — measured as ε via Protocol A; **ε_rms ≈ 0.134 on Phase 1 is small only because ψ is nearly flat, not because ψ is informative; rank-form ε_R proposed below** |
| Some Δ_t > 0 (non-vacuity precondition) | `empirical, established on ProSeCo-OWT`: 61/64 steps with Δ̄_t > 0, peak Δ̄ ≈ 0.157 (Phase 1 N=50) |
| Expectation-version remark | `heuristic` — stated, not formally written up |
| Refinement A′ (variance-form η_B) | `empirically anchored, formal proof pending Phase 3b` — σ_ξ measured directly from Phase 2b MC residuals (9000 G−A pairs across B ∈ {2,3,4}); √B scaling consistent with η_95 trace η_95(B=4)=0.41 → η_95(B=8)=0.68 → η_95(B=16)=1.36; remaining work is order-statistics derivation under the mixing/cancellation hypothesis on ξ_{t,t'} |
| Refinement A″ (rank-based ε_R) | `empirically anchored, formal proof pending Phase 3b` — pooled Spearman ρ(A,G) decays monotonically across B ∈ {2..16}: 0.66 → 0.62 → 0.50 → 0.44 → 0.45 (Phase 2b §12.3 of `RESULTS_STATUS.md`); replaces L∞ ε of Theorem A as the operational calibration measure; **Phase 3a does not contradict** — A″ bounds the ranker policy class, search procedures are out of scope |
| Negative-Result Corollary (ranker-class scope) | `empirically established on Phase 2b + Phase 3a, formal statement pending Phase 3b` — **scope rescoped 2026-04-20 after Phase 3a**: any policy of the form `Ŝ_B = top-B(ψ)` for separable per-step `ψ` is upper-bounded by `mean_delta_oracle`, which on ProSeCo-OWT enters the NULL band by B=8; smoking guns from Phase 2b: mean_delta_oracle saturates by B=8, top-10 MC ∩ oracle Jaccard ≈ 1.5× random, top-10 MC internal Jaccard ≈ bottom-10; old broad framing ("no greedy ranking beats uniform by > 2σ_F") is correct **for the ranker class only**, *not* for "informed scheduling in general" — Phase 3a's CD-G + BS-AG search procedures explicitly exceed this envelope at every B tested |
| Phase 3a search-class positive | `empirically established on ProSeCo-OWT 2026-04-20` (job 479941, K=30 paired) — CD-G PASS at all B with Δ ∈ [+0.32, +0.38] (closure 74–84 % of MC oracle at B ∈ {2,3,4}); BS-AG PASS at all B with Δ ∈ [+0.15, +0.29] (closure 49–64 %); both still PASS at B=8 where ranker envelope is NULL; closure ratios are empirical descriptors with no theoretical guarantee; CD-G is structural/existence (true-G feedback ≈ 65 × generation cost per cell, not deployable); BS-AG closer to practical (cheap-A ranking + O(B) true-G rollouts) but still requires a true-G evaluator |
| PRISM pivot | `rejected 2026-04-20, refined reason post-Phase-3a` — PRISM is a learned per-token quality signal, i.e. a member of the ranker class bounded by the rescoped Negative-Result Corollary; even a perfect Δ-predictor cannot exceed `mean_delta_oracle`'s envelope, which is NULL by B=8; rejection is **not** "no recoverable structure exists" (Phase 3a refutes that), but "the recoverable structure does not factor through separable per-step scores"; full rationale in `PHASE3_DIRECTION_AUDIT.md` |
| Phase 2c MAUVE F-swap | `closed without execution 2026-04-20` — Cohen's d up to ±2 in FAIL cells leaves cell ordering robust to F-metric choice; reopen only if Phase 3a forces defence of small-B win |
| Combining-step (2η vs 3η) | `proved either way`; choice is a write-up convention pinned to η's definition |
| Stretch C2 contraction | `conjecture` — applicability of Gibbs contraction not established |
| Stretch C3 margin-as-proxy | `empirical` — calibration question, not a theorem |
| MI monotonicity in u_t (old Candidate 3) | `refuted / retired` — replaced by Proposition B |
| Systematic-scan Ascolani citation (old) | `incorrect as stated` — corrected to random-scan |

## Non-Vacuity Hypothesis (resolved on ProSeCo-OWT, 2026-04-20; updated post-Phase-3a)

**Hypothesis NV (original form).** On at least one (backbone, corrector, F) triple distinct from (ProSeCo-OWT, annealed-refinement, −GPT-2 NLL), there exists B ∈ {2, 3, 4} for which the measured (ε_R, η̄_B) satisfy 2Bε_R + 2η̄_B < G(S_B*) under the expectation form of Theorem A.

**NV verdict on ProSeCo-OWT (post-Phase-2b).** Even with the rank-form refinement and the variance-form η, the L∞-style Theorem A bound stays inert at B ≥ 8 because `mean_delta_oracle` itself — the upper envelope of any single-step ranker — falls into the NULL band by B = 8. The thesis contribution that lands is therefore the **rank-based ε_R + variance-form η_B refinements + Negative-Result Corollary (ranker-class scope)**, not a non-vacuous L∞ Theorem A on this triple.

**Post-Phase-3a addendum (2026-04-20).** Phase 3a does not change the NV verdict for Theorem A as stated (Theorem A is a bound on a *ranker* `Ŝ_B = top-B(ψ)`, which is exactly the policy class the rescoped corollary bounds). What Phase 3a establishes separately is that the gap between the ranker envelope and `G(S_B*)` is largely *closable by procedures outside the ranker class* (CD-G recovers 74–84 %, BS-AG 49–64 % of the MC-oracle headroom at B ∈ {2,3,4}, and both PASS at B = 8). A non-vacuous bound on the search class would be a different theorem (no formal lower-bound currently proven for CD-G or BS-AG). Cross-backbone validation of NV on (MDLM, …) remains **parked** as Tier-2; Phase 3a's positive on ProSeCo-OWT does not by itself motivate cross-backbone replication — the gating question is whether the search-vs-ranker dichotomy is universal, which is out of thesis scope.

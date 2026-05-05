# Theory — Summary

> **Current source of truth.** Updated 2026-05-05.
> Full formal statements: `research/candidate_theorems.md`.
> Derivation entries: `research/proof_worklog.md`.

---

## Main theorem — Theorem A

**Proxy-Regret Bound for Top-B Scheduling.** Let the predictor schedule fix states
Z_0, ..., Z_T. For each step t define Δ_t := F(y_t^{+1}) − F(y_base). For S ⊆ {1,...,T}
with |S| = B let G(S) := F(y^S) − F(y_base). Let S_B* := argmax G(S) and
Ŝ_B := top-B by proxy ψ(s_t).

Under (1) binary placement, (2) approximate additivity |G(S) − ∑ Δ_t| ≤ η_B,
(3) proxy calibration |Δ_t − ψ(s_t)| ≤ ε:

    G(S_B*) − G(Ŝ_B) ≤ 2 B ε + 2 η_B.

**Status:** Formally proved under (1)–(3). LaTeX skeleton in `thesis/chapters/ch6_contribution.tex`.
**Honesty:** L∞ form is empirically vacuous at B ≥ 4 on OWT (bound > typical G(Ŝ_B)).

---

## Supporting lemmas

**Lemma A1 (Oracle top-B under exact additivity).** If G(S) = ∑ Δ_t exactly, then S_B*
is the B indices with largest Δ_t. Status: `solid under assumptions`.

**Lemma A2 (Calibration regret in additive regime).** Under η_B = 0 and |Δ_t − ψ(s_t)| ≤ ε:
∑_{S_B*} Δ_t − ∑_{Ŝ_B} Δ_t ≤ 2Bε by swap-by-swap exchange (≤ B swaps, each costs ≤ 2ε).
Status: `solid under assumptions`.

---

## Refinements (empirically anchored, formally proved 2026-04-26)

**Refinement A′ — Variance-form η_B.**
Replaces Proposition C's γB(B−1)/2 with the tighter `𝔼|G−A| ≤ σ_ξ · √B / √2`.
σ_ξ measured from 9000 MC residuals G − A on Phase 2b.
Status: **formally proved under mixing/cancellation hypothesis** on (ξ_{t,t'}).
Note: grows super-linearly in √B (×1.78 from B=2→4 vs √2 ideal) — evidence of
non-trivial pairwise interaction.

**Refinement A″ — Rank-based calibration ε_R.**
Replaces L∞ ε with ε_R := (1 − |ρ_B|) · σ_Δ where ρ_B = pooled Spearman ρ(A, G).
Decays monotonically 0.66 (B=2) → 0.39 (B=16). This is the **operative empirical
quantity** — the version that survives the data and does not give vacuous bounds.
Status: **formally proved under Gaussian-A hypothesis** (order-statistics derivation).

**Plug-in regret bounds (using Refinement A″ + A′):**

| B | ε_R | η_B (A′) | Bound 2Bε_R + 2η_B |
|---|---|---|---|
| 2 | 0.070 | 0.174 | 0.628 |
| 3 | 0.092 | 0.294 | 1.142 |
| 4 | 0.118 | 0.437 | 1.820 |

Bound is non-vacuous only if < G(Ŝ_B), which requires G(Ŝ_B) > 0.628 at B=2.

---

## Negative-Result Corollary (formally proved 2026-04-26)

For any policy of the form Ŝ_B = top-B(ψ) for **separable per-step ψ**, the policy's
gain G(Ŝ_B) is upper-bounded by the `mean_delta_oracle` envelope, which on ProSeCo-OWT
enters the NULL band by B = 8. This corollary characterises the **ranker class only**.
Phase 3a's CD-G and BS-AG explicitly exceed this envelope by operating on schedules.

---

## Supporting propositions (for completeness, not primary)

**Proposition B (Low-gain-region exclusion / burn-in gating):** G(Ŝ_B) − G(Ŝ_B^{gated}) ≤ Bδ + 2η_B.
Justifies excluding early steps (R_t ≈ ∅ in ProSeCo). Status: `plausible but incomplete`.

**Proposition C (Pairwise interaction bound):** If G decomposes with |ξ_{t,t'}| ≤ γ,
then η_B ≤ γ B(B−1)/2 and regret ≤ 2Bε + γB(B−1). Empirical γ (q₀.₉₅) = 0.376/0.168/0.108
at B=2/3/4. Superseded by Refinement A′ as the tighter bound. Status: `plausible, not primary`.

---

## Appendix F — Theorem A-ad (adaptive/state-conditional)

Theorem A-ad generalizes Theorem A to the abstract policy class (top-B and threshold-λ
as members). Under assumptions (1)–(4) (adds bounded bucket-mean estimation error):

    G(S_B*) − G(Ŝ_B^{ad}) ≤ 2 B ε̃ + 2 η̃_B + O(√(B / N_cal)).

Status: **formally proved** as conditional theorem. Protocol C showed ε̃ ≈ ε on OWT
(bucketed conditioning provides no meaningful calibration improvement). Lives in
Appendix F as existence-only.

---

## Theory status table

| Item | Status | Since |
|---|---|---|
| Theorem A | Proved under (1)–(3); LaTeX TODO prose | Phase 3b |
| Lemma A1 | Proved | Phase 2b |
| Lemma A2 | Proved | Phase 2b |
| Refinement A′ | Proved (mixing hypothesis) | Phase 3b |
| Refinement A″ | Proved (Gaussian-A hypothesis) | Phase 3b |
| Negative-Result Corollary | Proved (ranker class) | Phase 3b |
| Proposition B | Plausible; not on critical path | — |
| Proposition C | Plausible; superseded by A′ | — |
| Theorem A-ad | Proved; Appendix F only | Protocol C |
| Stretch C2 (contraction) | Stretch appendix; not on critical path | — |

---

## Remaining theory tasks (critical path)

1. **Theorem A proof narrative** — write the combining-step LaTeX argument:
   (a) apply (2) to bound G(S_B*) − G(Ŝ_B) ≤ A(S_B*) − A(Ŝ_B) + 2η_B;
   (b) apply Lemma A2 to bound A(S_B*) − A(Ŝ_B) ≤ 2Bε.
   Section bodies marked TODO in `thesis/chapters/ch6_contribution.tex`.

2. **Refinement A′ clean write-up** — order-statistics argument for
   𝔼|G−A| under mixing/cancellation hypothesis on ξ_{t,t'}.

3. **Refinement A″ clean write-up** — order-statistics derivation under Gaussian-A
   of 𝔼[A(S_A*) − A(Ŝ_B)] ≤ B · ε_R.

4. **Negative-Result Corollary in ch6** — formal corollary environment with proof
   referencing mean_delta_oracle envelope on OWT.

## What NOT to pursue

- Stretch C2 (Gibbs contraction) — no applicable Gibbs-contraction framework for
  discrete non-log-concave masked text. Not on critical path.
- Confidence margin as a separate theorem — handle empirically via Protocol A.
- Full end-to-end policy optimality — out of scope.

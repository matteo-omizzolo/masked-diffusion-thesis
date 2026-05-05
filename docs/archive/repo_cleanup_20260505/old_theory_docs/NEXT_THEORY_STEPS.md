> ARCHIVED — historical only.
> Do not use this file to infer the current thesis status unless explicitly asked.
> Current source of truth: see `START_HERE.md` and `docs/README.md`.

---


# Next Theory Steps — Canonical

**Updated:** April 2026. Companion to `THEORY_STATUS.md`. Concrete, short-horizon.

## Next Three Precise Theory Tasks

1. **Clean write-up of Theorem A.** Combine Lemma A1 + Lemma A2 + two applications of approximate-additivity (2) into a single LaTeX proof in `thesis/chapters/ch6_contribution.tex` (new section). Explicitly route the exchange argument through the additive surrogate A(S) := ∑_{t ∈ S} Δ_t rather than through G directly: (a) use (2) to bound G(S_B*) − G(Ŝ_B) ≤ A(S_B*) − A(Ŝ_B) + 2η_B; (b) apply Lemma A2's swap argument to bound A(S_B*) − A(Ŝ_B) ≤ 2Bε for the ψ-top-B subset Ŝ_B. Include the expectation-form remark as a corollary: E[|G(S) − ∑ Δ_t|] ≤ η̄_B and E[(Δ_t − ψ(s_t))²] ≤ ε̄² yield an expected-regret bound; note this is less brittle under heavy-tailed Δ_t.

2. **Formalize Proposition B.** State the low-gain-region exclusion bound G(Ŝ_B) − G(Ŝ_B^{gated}) ≤ Bδ + 2η_B and prove it as a direct consequence of the Theorem A exchange argument: each swap out of T_low costs at most δ in A (rather than 2ε), and two applications of (2) connect A back to G. Explicitly note that T_low existence is an empirical identification, not a theorem.

3. **Prove Proposition C.** From the pairwise decomposition G(S) = ∑_{t ∈ S} Δ_t + ∑_{{t,t'} ⊂ S} ξ_{t,t'} with |ξ_{t,t'}| ≤ γ, derive |G(S) − ∑_{t ∈ S} Δ_t| ≤ γ · C(|S|, 2) ≤ γ B(B−1)/2, hence η_B ≤ γ B(B−1)/2, hence regret ≤ 2Bε + γB(B−1). Note triple-interaction diagnostic is a further check if pairwise fit is loose.

## What NOT to Spend Time On Right Now

- **Stretch Appendix C2 (contraction theorem).** No applicable Gibbs-contraction framework is yet established for discrete non-log-concave masked-text conditionals. Pursue only if Ascolani et al. 2024 + Denoising Entropy Bounds reading yields a usable bound. Not in the critical path for the thesis.
- **Confidence margin as a separate theorem (Stretch C3).** Under Theorem A this reduces to "which ψ minimizes ε?" Handle empirically via Protocol A. Do not write a theorem for it.
- **Full end-to-end policy optimality.** Out of scope. Theorem A is a regret bound against the best fixed top-B schedule, not against an optimal adaptive policy.

## What Depends on New Experiment Results

- **Non-vacuity of Theorem A** (2Bε + 2η_B < G(Ŝ_B)): requires ε, η_B measurements from a backend where Δ_t > 0 at some steps. MDLM heuristic failed this; ProSeCo Phase 1 pilot is the test.
- **δ, T_low identification** (Proposition B): requires an experiment where the corrector is non-trivial; ProSeCo's R_t ≈ ∅ at early steps gives a natural-by-design T_low.
- **Tightness of Proposition C's pairwise bound**: requires γ from Protocol B pairwise diagnostic.

## What Can Be Done Immediately (Without New Experiments)

- Clean LaTeX proof of Theorem A in `thesis/chapters/ch6_contribution.tex` (new section).
- Full proofs of Lemma A1 (trivial) and Lemma A2 (exchange argument).
- Proof of Proposition C from the pairwise decomposition.
- Write the expectation-version corollary to Theorem A as a remark.
- Confirm the Ascolani reference in `research/proof_ledger.md` reads "random-scan Gibbs" (already corrected April 2026; verify if not already landed).

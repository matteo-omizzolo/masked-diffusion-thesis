# Theory — Summary

> **Current source of truth.** Updated 2026-05.
> Full formal statements: `research/candidate_theorems.md` §0–§7.
> Derivation entries: `research/proof_worklog.md`.

---

## Active theorem stack (theory-first programme — 2026-05)

The stack has been reframed for the theory-first phase. Detailed statements,
proofs, and theory-to-experiment map are in `research/candidate_theorems.md`
§0–§7; this file is a one-page summary.

The framework is **model- and corrector-agnostic**: fix a predictor, informed
corrector, quality functional F, and placement budget B, then define G(S),
Δ_t, A(S), ξ_{t,t′}, Q(S), and the regime diagnostics. ProSeCo-OWT is the
primary empirical case study, not a theory assumption.

| § | Object | Status | Role |
|---|---|---|---|
| §1.1–§1.2 | **Theorem A** — marginal proxy regret 2Bε + 2η_B (uniform form) | Proved | Baseline; tested by (A2)/(A3) on candidate pool C_B |
| §1.3 | **Diagnostics A′ (additivity scale), A″ (rankability)** | **Empirical diagnostics** (not unconditional regret refinements) | Demoted from previous "proved" status; do not control selected-schedule regret without finite-pool conversion |
| §1.4 | Theorem A as a special case of B′ (safe finite-pool ranking corollary) | Corollary of B′ | The rigorous selected-schedule consequence of A |
| §1.5 | **Empirical Ranker-Class Limitation** (replaces "Negative-Result Corollary") | Formal part for time-only separable ψ; empirical part on tested rankers | Tested separable rankers do not recover MC-oracle headroom; does not bound feature-conditioned, pairwise, or online policies |
| §2.1 | Theorem B exact: G(S_B*) − G(Ŝ) ≤ 2ζ_B + ω_B | Proved | Generic surrogate-regret inequality |
| §2.2 | Theorem B estimated: ≤ 2ζ_B + 2α_B + ω_B | Proved (constant 2) | Operational uniform form |
| §2.3 | **Theorem B′** finite-pool / high-probability + κ_B; data-dependence caveat | Proved | **Experimentally usable form** |
| §2.4 | Levels 1 / 2 / 3 hierarchy | Definition | Diagnostic / population / feature-conditioned |
| §2.6 | Level-specific metrics (P_B^seed, P_B^pop, P_B^feat, C_B^pop, C_B^feat) | Definitions | Match metric to level of claim |
| §2.7 | Theorem A as B′(Q := A) | Corollary | Safe finite-pool form of A |
| §3 | **Diagnostic Framework C** — regimes with U_B^{MC,N}, R_B, I_B, P_B, C_B^{MC,N} | Definition + protocol | Renamed from Proposition C |
| §4 | Theorem D — online controller 2Tδ ADP loss | Proof sketch | Optional / appendix; first to cut |
| §5 | Lemma E — burn-in exclusion under clipped F_C only | Conditional sketch | Optional; F = −GPT-2 NLL is **not** Lipschitz without clipping |

**Backbone:** Theorem A baseline → Theorem B / B′ central → Diagnostic Framework C.
A′, A″ are diagnostics. Empirical Ranker-Class Limitation scopes the negative
result to the tested class. D and E are appendix.

Plan: `docs/06_theory_first_research_plan.md`.

---

## Active theorem stack — full statements

The full statements, proofs, and theory-to-experiment map live in
`research/candidate_theorems.md` §0–§7. To avoid drift this summary file does
not duplicate them. Quick pointers:

- §0 — Formal setup (trajectory, schedules, Δ_t, A(S), ξ_{t,t'}, Q(S), levels).
- §1.1–§1.2 — Theorem A statement and proof.
- §1.3 — Diagnostics A′ (additivity scale), A″ (rankability) — **demoted**
  from prior proof status to empirical diagnostics. The legacy scaling formulas
  are not used as theorem constants in active claims.
- §1.4 — Theorem A as a special case of Theorem B′ (the safe finite-pool
  selected-schedule form; replaces what A′/A″ tried to provide).
- §1.5 — Empirical Ranker-Class Limitation (replaces "Negative-Result
  Corollary"): formal part for time-only / seed-averaged separable ψ;
  empirical part on tested rankers.
- §2.1–§2.7 — Theorem B exact / estimated (constant 2α_B) / B′ finite-pool
  high-probability with κ_B + no-leakage caveat; Levels 1/2/3 hierarchy and
  level-specific metrics; Theorem A as B′(Q := A).
- §3 — Diagnostic Framework C with disciplined U_B^{MC,N} / U_B^{pool} / U_B^*
  notation.
- §4 — Theorem D online controller (appendix-grade).
- §5 — Lemma E burn-in (clipped F_C only; F = −GPT-2 NLL is not Lipschitz
  without clipping; appendix-grade).

## What NOT to pursue

- Any unconditional regret-refinement claim from A′ or A″ (they are
  diagnostics, not unconditional bounds).
- Any universal ranker-failure statement (the formal scope of the
  ranker-class limitation is time-only / seed-averaged separable ψ; the
  empirical scope is the tested ranker class).
- Reporting the unobservable U_B^* (exhaustive oracle headroom) — only the
  MC-oracle / pool-oracle headrooms are observable.
- Stretch C2 (Gibbs contraction). Not on critical path.

## Remaining theory tasks (LaTeX prose)

LaTeX prose in `thesis/chapters/ch6_contribution.tex` is TODO for:
1. Theorem A combining-step argument (write directly in the uniform form,
   then state the finite-pool corollary §2.7 as the operative form).
2. Theorem B / B′ statements and proofs.
3. Empirical Ranker-Class Limitation (formal + empirical parts).
4. Diagnostic Framework C definitions and regime taxonomy.

Defer ch6 LaTeX prose until after Phase 0 + Phase 1 results are in, so the
empirical anchors in §1.5, §2 can be cited.

> **STATUS:** EMPIRICAL ANCHOR (Phase 4 output of the 2026-04-24 audit)
> **LAST VERIFIED:** 2026-04-24
> **SCOPE:** Measured Theorem-A constants on ProSeCo-OWT Phase 2b artefacts,
> with plug-in bounds under Refinement A′, Refinement A″, and Proposition C.
> Feeds directly into ch6 §"Empirical anchoring".

---

# Theorem A Constants — OWT Phase 2b Measurements

## 1. Source

All constants are computed from `results/phase2b_proseco_owt/mc_raw.json`
(9000 MC rows, 30 seeds, B ∈ {2, 3, 4}, T = 64, ProSeCo-OWT backbone) by

```
python scripts/compute_theorem_a_constants.py \
    --phase2b_raw_dir results/phase2b_proseco_owt \
    --out results/phase2b/theorem_a_constants.json
```

Estimator implementation: `src/mdm_playground/analysis/theorem_a_constants.py`.
Test suite: `tests/test_theorem_a_constants.py` (12 tests, 100 % pass).

## 2. Measured constants

All values measured on ProSeCo-OWT, pooled across 30 seeds. σ_ξ and γ quoted
at the pooled / 95 th-quantile level respectively.

| B | σ_ξ (pooled) | ρ (pooled; 95 % CI)       | σ_Δ  | γ (q₀.₉₅) | ε_R   |
|---|--------------|----------------------------|------|-----------|-------|
| 2 | 0.174        | 0.601 [0.571, 0.628]       | 0.176| 0.376     | 0.070 |
| 3 | 0.240        | 0.542 [0.516, 0.572]       | 0.202| 0.168     | 0.092 |
| 4 | 0.309        | 0.462 [0.430, 0.492]       | 0.220| 0.108     | 0.118 |

- **σ_ξ** — pooled std of ξ = G − A, the additivity residual. Grows
  super-linearly in √B (×1.78 from B = 2 to B = 4, vs √2 = 1.41 ideal),
  evidence of non-trivial pairwise interaction.
- **ρ** — pooled Spearman rank correlation between the additive proxy A and
  the true gain G. Modestly positive and CI excludes zero at all B, but
  degrades from 0.60 → 0.46 as B grows — ranker fidelity weakens as selection
  tightens.
- **σ_Δ** — pooled std of G, the scale used in Refinement A″.
- **γ** — 95 th-quantile upper bound on the per-schedule implied pairwise
  interaction 2 |residual| / (B(B − 1)).
- **ε_R = (1 − |ρ|) · σ_Δ** — Refinement A″ plug-in.

## 3. Plug-in bounds

Two variants of η_B, both plugged into G(S_B*) − G(Ŝ_B) ≤ 2Bε_R + 2η_B:

| B | η_B (A′: σ_ξ √B / √2) | η_B (Prop C: γ·B(B−1)/2) | Bound (A″ + A′) | Bound (A″ + Prop C) |
|---|------------------------|---------------------------|------------------|----------------------|
| 2 | 0.174                  | 0.376                     | **0.628**        | 1.032                |
| 3 | 0.294                  | 0.504                     | **1.142**        | 1.562                |
| 4 | 0.437                  | 0.650                     | **1.820**        | 2.247                |

The **A′ variance form is tighter than the Proposition-C pairwise form at
every tested B.** The tighter of the two per-B (always A′ here) is the
reported thesis bound.

## 4. Is the bound non-vacuous?

The relevant empirical upper bound on the gap `G(S_B*) − G(Ŝ_B)` is the
MC-oracle headroom (mean paired diff of MC-oracle over uniform), measured at
+0.45 NLL units on OWT (`results/phase2b/mc_oracle.json` + phase2b memo).
The plug-in bounds should exceed this headroom but not by a large factor in
the "non-vacuous" regime.

| B | Bound (A″ + A′) | Observed headroom | Bound ÷ headroom |
|---|-----------------|--------------------|-------------------|
| 2 | 0.628            | 0.45               | 1.40 ✓ tight      |
| 3 | 1.142            | 0.45               | 2.54 ✓ non-vacuous |
| 4 | 1.820            | 0.45               | 4.04 ≈ vacuous    |

**Result:** Theorem A with Refinements A′ + A″ is **tight at B = 2, non-
vacuous at B = 3, and approaches vacuous at B = 4** on ProSeCo-OWT. This is a
meaningful improvement over the Phase-1 plug-in at B = 8 (bound ≈ 3.5 vs G
≤ 1.2) and justifies stating Theorem A with measured constants rather than
symbolically.

## 5. Proposition B empirical anchor

The **low-gain-share** estimator measures, per seed, the ratio of the max G
observed among the top-10 schedules ranked by the additive proxy A to the
max G observed across **all** MC schedules at that seed:

| B | low_gain_share mean | SE    | min  | max |
|---|----------------------|-------|------|-----|
| 2 | **0.886**            | 0.029 | 0.44 | 1.00 |
| 3 | **0.885**            | 0.025 | 0.56 | 1.00 |
| 4 | **0.876**            | 0.028 | 0.50 | 1.00 |

**Interpretation:** The top-10 schedules ranked by the additive proxy
already capture 87–89 % of the oracle-achievable G per seed. This is the
**Proposition B empirical anchor** — it formalises the intuition that the
ranker class is close to the oracle *in the low-gain-region sense* while
simultaneously leaving a ~12 % gap that Phase 3a's search class (CD-G,
BS-AG) partially closes.

## 6. ρ-degradation with B (an open question anchor)

ρ drops monotonically with B: 0.60 → 0.54 → 0.46 (B = 2 → 3 → 4). This is
consistent with Q5 (additivity at low B breaks down at high B): as more
sites are co-selected, pairwise interactions (bounded by γ) drag down the
proxy's rank fidelity. The effect is modest over B ∈ {2, 3, 4} but would
extrapolate to near-zero ρ by B = 8, consistent with the observed
mean_delta_oracle saturation at B = 8.

## 7. What ch6 should report

| Table row | Value | Source |
|-----------|-------|--------|
| Theorem A plug-in bound at B = 2 | 0.628 | §3 above |
| Theorem A plug-in bound at B = 3 | 1.142 | §3 above |
| Theorem A plug-in bound at B = 4 | 1.820 | §3 above |
| ε_R at B = 2 | 0.070 | §2 above |
| η_B(A′) at B = 2 | 0.174 | §3 above |
| Prop B share at B = 2 | 0.886 | §5 above |
| Prop C γ upper at B = 2 | 0.376 | §2 above |
| ρ(A, G) at B = 2 | 0.60 [0.57, 0.63] | §2 above |
| σ_ξ at B = 2 | 0.174 | §2 above |

Use the same table structure for B = 3 and B = 4.

## 8. What this measurement does NOT claim

- It does **not** claim tightness on LLaDA-SFT at the K = 8 bounded probe.
  At that triple, MC-oracle headroom is ≈ 0 (paired diff −2.64 at B = 2,
  exactly 0.000 at B = 4), so the regret interpretation of Theorem A is
  moot on that backbone.
- It does **not** claim tightness at B = 8 on OWT. At B = 8, A′ plug-in
  would give η_B ≈ σ_ξ · 2 = ~0.60, and ε_R could be larger as ρ
  degrades. The combined bound at B = 8 is expected to cross the vacuous
  threshold.
- It does **not** establish that Refinement A′ is always tighter than
  Prop C. Only that it is tighter on this OWT triple at B ∈ {2, 3, 4}.
  Prop C's advantage may appear on triples with fatter-tailed residuals.

## 9. Reproducibility

```bash
# From repo root
python -m pytest tests/test_theorem_a_constants.py -q
python scripts/compute_theorem_a_constants.py \
    --phase2b_raw_dir results/phase2b_proseco_owt \
    --out results/phase2b/theorem_a_constants.json
```

No GPU required. Runtime ≈ 2 s on a laptop (dominated by bootstrap).

## 10. Links

- Theorem A text: `../../../research/candidate_theorems.md`
- Refinements A′ + A″: same file, §"Refinements"
- Prop B + Prop C: same file, §"Propositions"
- Open-loop theorem status: `THEORY_STATUS.md`
- Decision source: `../next_steps/NEXT_RESEARCH_DIRECTION_DECISION.md`
- Landscape positioning: `MDM_THEORY_LANDSCAPE_POSITIONING.md`
- Estimator module: `../../../src/mdm_playground/analysis/theorem_a_constants.py`
- CLI: `../../../scripts/compute_theorem_a_constants.py`
- Output JSON: `../../../results/phase2b/theorem_a_constants.json`

---

*End of Phase 4 deliverable. Phase 5 (Protocol C on OWT) is sequenced next.*

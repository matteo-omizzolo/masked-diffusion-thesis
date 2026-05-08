> **STATUS:** ARCHIVED
> **ARCHIVED ON:** 2026-04-22
> **SUPERSEDED BY:** `docs/thesis/CANONICAL_EXPERIMENT_OVERVIEW.md`
> **REASON:** Pre-Phase-2b ProSeCo design; canonical Protocol A/B spec lives upstream.

---

# ProSeCo Experiment Definition

**Date:** 2026-04-17  
**Status:** Defined — ready for HPC execution

This document defines the ProSeCo-based Phase 1 experiment mathematically, with
connections to the theorem objects. Read alongside `proseco_protocol_mapping.md`
(implementation) and `proseco_backend_audit.md` (backend details).

---

## Research question (restated for ProSeCo)

> For fixed predictor schedule (MDLM, T steps) and fixed corrector NFE budget B,
> can aggregate trajectory signals — entropy, confidence margin, or quality mass
> over currently unmasked positions — predict the marginal value of a ProSeCo
> corrector loop well enough to outperform uniform corrector placement?

**Why ProSeCo over MDLM heuristic:**
The MDLM Gibbs-style resample proved harmful at all steps (Δ_t ≤ 0 everywhere) because
a full one-shot resample of ~97% masked positions with <5% context destroys coherence.
ProSeCo's annealed refinement of committed tokens provides a principled, trainable
corrector where some Δ_t > 0 is plausible — a necessary condition for Theorem A to be
non-vacuous.

---

## Notation

| Symbol | Meaning |
|---|---|
| x_t | Token sequence at predictor step t (some masked, some committed) |
| p_θ(x_0 \| x_t) | Backbone prediction (predictor logits at step t) |
| x̂_0^t | Predictor's argmax estimate of clean x_0 at step t |
| C_t(x) | ProSeCo corrector applied at step t: argmax-based annealed refinement, applied to unmasked positions |
| y_base | Trajectory without any corrector: x_0, x_1, …, x_T |
| y_t^{+1} | Trajectory with corrector at step t only: x_0, …, C_t(x_t), …, x_T |
| y^S | Trajectory with corrector at each step t ∈ S |
| F(y) | Quality functional: neg-NLL under GPT-2 reference scorer |
| Δ_t | F(y_t^{+1}) − F(y_base): marginal gain from one corrector loop at t |
| G(S) | F(y^S) − F(y_base): joint gain from schedule S |
| s_t | Aggregate signal at step t (entropy / margin / quality mass over R_t) |
| ψ(s_t) | Signal-based proxy for Δ_t |
| ε | sup_t |Δ_t − ψ(s_t)|: calibration error |
| η_B | sup_{|S|=B} |G(S) − ∑_{t∈S} Δ_t|: additivity slack |
| γ | sup_{t,t'} |ξ_{t,t'}|: pairwise interaction bound |

---

## Protocol A on ProSeCo: marginal gain profile

### Setup

- N trajectories (N = 20 pilot, N = 50 full)
- For each trajectory i and each step t ∈ {0, …, T-1}:

**Base:** run T predictor steps with seed_i. No correctors.

**Branch at t:** run T predictor steps with seed_i. At step t, after the predictor
step yields state x_t with committed tokens R_t = {i : x_t[i] ≠ mask_id}:

1. Get p_x0_cache_t (predictor's probability predictions, already computed)
2. x̂_0 = argmax(p_x0_cache_t) — full predicted sequence
3. corrector_x = x̂_0
4. For j = 0, 1 (corrector_steps=2):
   - tau_j = linspace(1.0, 1e-5, 3)[j]
   - corrector_x = argmax(model.forward(corrector_x, sigma(tau_j)))
5. x_t' = where(x_t[i] ≠ mask_id, corrector_x[i], x_t[i]) — apply to R_t only
6. Continue predictor from x_t' for steps t+1, …, T-1

### Outputs

For each (i, t):
- **Δ_t(i) = F(y_t^{+1}(i)) − F(y_base(i)):** one-loop marginal gain
- **TCR_t(i) = Hamming(y_t^{+1}, y_base) / L:** token-change rate (kept separate)
- **Signals from y_base step t:** H_t(i), M_t^{-1}(i), Q_t(i), u_t(i)

### Aggregate quantities

```
Δ̄_t = (1/N) ∑_i Δ_t(i)       — mean marginal gain profile
H̄_t  = (1/N) ∑_i H_t(i)       — mean entropy profile
```

### Calibration estimation (ε)

Linear calibration of signal → Δ:
```
ψ(s_t) = a · s_t + b    (least-squares fit over all (t, i) pairs)
ε_rms = sqrt( (1/(NT)) ∑_{t,i} (Δ_t(i) − ψ(s_t(i)))^2 )
```

Compute separately for each signal candidate.

---

## Protocol B on ProSeCo: schedule-level gain and additivity

### Setup

For each budget B ∈ {4, 8, 16} (= {T/16, T/8, T/4}):

**η_B estimation:** sample M random schedules S with |S| = B; compute:
- G(S) = F(y^S) − F(y_base) — joint gain
- A(S) = ∑_{t∈S} Δ̄_t — additive surrogate
- r(S) = G(S) − A(S) — residual

```
η_B ≈ 95th-pct_{M schedules} |r(S)|
```

**Pairwise γ estimation:** sample P random pairs (t, t'); compute:
```
ξ_{t,t'} = G({t, t'}) − (Δ̄_t + Δ̄_{t'})
γ ≈ 95th-pct_P |ξ_{t,t'}|
```

### Proposition C connection

Proposition C says: η_B ≤ γ · B(B−1)/2.
We estimate γ from pairwise measurements and check whether the implied η_B
bound is tighter or looser than the empirical η_B from Protocol B.

---

## Signals: full specification

### Revisable set

```
R_t = { i ∈ {0,…,L-1} : x_t[i] ≠ mask_id }
```

This is the ProSeCo corrector's action set at step t.
At early steps: R_t is small. At late steps: R_t is nearly all of {0,…,L-1}.

### Signal formulas

Let p_i = p_θ(x_0^i | x_t) ∈ Δ(V) be the model's predictive distribution over the
vocabulary at position i, evaluated at noise level σ_t.

**Entropy:**
```
H_t = (1/|R_t|) ∑_{i ∈ R_t} H(p_i)
    = (1/|R_t|) ∑_{i ∈ R_t} (−∑_v p_i(v) log p_i(v))     [nats]
```
High H_t → model is uncertain about committed tokens → corrector may improve them.

**Inverse confidence margin:**
```
M_t^{-1} = 1 − (1/|R_t|) ∑_{i ∈ R_t} (p_i^{(1)} − p_i^{(2)})
```
where p_i^{(1)} ≥ p_i^{(2)} are the top two probabilities.
High M_t^{-1} → committed tokens have low confidence margin → corrector may revise them.

**Quality mass proxy:**
```
Q_t = 1 − (1/|R_t|) ∑_{i ∈ R_t} p_i^{(1)}
```
High Q_t → low mean maximum probability → poor quality prediction.

**Unmasked fraction (context scalar):**
```
u_t = |R_t| / L
```
Monotone increasing from ~0 at t=0 to ~1 at t=T-1. Used as a deconfounding variable.

### Notes on signal design

1. **No quality head:** MDLM does not have a PRISM-style quality head. All signals are
   derived from the predictor's logits. Quality mass proxy is a logit-based stand-in.

2. **Signal at t=0:** R_0 ≈ ∅ (all tokens masked) → all signals = 0 → no corrector
   placement at step 0 (burn-in is automatic for ProSeCo because the corrector has
   nothing to refine when R_0 is empty).

3. **Entropy vs u_t:** entropy over R_t grows as more tokens commit (lower-confidence
   tokens commit later). The thesis question is whether entropy over u_t RESIDUAL
   (after regressing out u_t) predicts Δ_t.

---

## Quality functional F

**Primary:** negative per-token NLL under GPT-2 reference scorer.
```
F(y) = −(1/|y|) ∑_i log p_{GPT-2}(y_i | y_{<i})
```
Higher = better quality (lower perplexity).

**Limitations noted:**
- Truncation to 512 tokens (from L=1024) avoids OOM
- GPT-2 is a weak reference scorer; PPL differences are meaningful for relative comparison
- MAUVE not used (requires many samples, cannot measure per-step Δ_t)

---

## Theorem connections

### Theorem A (main)

> G(S_B*) − G(Ŝ_B) ≤ 2Bε + 2η_B

Where:
- S_B* = oracle (best) budget-B schedule: argmax_S G(S)
- Ŝ_B = signal-guided schedule: top-B steps by ψ(s_t) = a·H_t + b
- ε = calibration error (from Protocol A)
- η_B = additivity slack (from Protocol B)

**Empirical test:** if 2Bε + 2η_B < G(Ŝ_B), Theorem A's bound is non-vacuous.

### Proposition B (burn-in gating)

> ∃ threshold τ such that T_low = {t : H_t ≤ τ} satisfies Δ_t ≈ 0 for t ∈ T_low,
> and excluding T_low from allocation strictly improves regret.

**Empirical test:** identify T_low from Δ̄_t profile. Check if Δ̄_t > 0 only outside T_low.
For ProSeCo: T_low should be the very early steps where R_t ≈ ∅ (natural burn-in).

### Proposition C (pairwise interaction)

> η_B ≤ γ · B(B−1)/2

**Empirical test:** compare empirical η_B (Protocol B) vs γ · B(B−1)/2 (from pairwise).

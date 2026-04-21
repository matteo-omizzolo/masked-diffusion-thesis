# ProSeCo Protocol Mapping

**Date:** 2026-04-17  
**Status:** Implemented — `backends/proseco.py`, `scripts/run_phase1_proseco.py`

This document explains exactly how Protocol A and Protocol B are instantiated
on the ProSeCo backend. It complements `proseco_experiment_definition.md`
(which defines the mathematical objects) and `proseco_backend_audit.md`
(which describes the code-level mechanism).

---

## Model and setup

**Backbone:** MDLM DiT, loaded from `/home/3316152/mdm/checkpoints/mdlm.ckpt`  
**Tokenizer:** GPT-2 (50,257 tokens + 1 mask = 50,258)  
**Sequence length:** L = 1024 tokens  
**Predictor steps:** T = 64 (pilot) or T = 128 (full run)  
**Predictor sampler:** MDLM absorbing-state DDPM (loglinear noise schedule)  
**Corrector:** ProSeCo annealed refinement, `corrector_steps = 2`  
**Quality functional F:** negative NLL per token under GPT-2 reference scorer (higher = better)

---

## Protocol A on ProSeCo

### Definition

For each trajectory i and each step t ∈ {0, …, T-1}:

**Base trajectory** `y_base(i)`:
- Seed: `seed_i = seed_base + i`
- Run T predictor steps (MDLM absorbing-state DDPM) with `torch.manual_seed(seed_i)`
- No corrector loops anywhere
- Output: `tokens_base`, `neg_nll_base = F(y_base)`
- Also output: `per_step_signals[t]` for t = 0, …, T-1 (see Signals section)

**Branch trajectory** `y_branch(i, t)`:
- Same `seed_i` as base (CRN for variance reduction)
- Run T predictor steps
- At step t: after the predictor step produces `x_t` (with some tokens unmasked),
  apply ONE ProSeCo corrector loop:
  1. `x̂_0 = argmax(p_x0_cache_t)` (predicted clean sequence from step t)
  2. `corrector_x = x̂_0`
  3. For j in {0, 1} (corrector_steps=2):
     `tau_j = linspace(1, eps, 3)[j]`  → tau_0 = 1.0, tau_1 = 0.5
     `corrector_x = argmax(model.forward(corrector_x, sigma(tau_j)))`
  4. `x_t_corrected = where(x_t != mask_id, corrector_x, x_t)` — unmasked only
- Continue predictor from `x_t_corrected` for steps t+1 … T-1
- Output: `tokens_branch`, `neg_nll_branch = F(y_branch)`

**Marginal gain:**
```
Δ_t(i) = F(y_branch(i, t)) − F(y_base(i))
        = neg_nll_branch − neg_nll_base
```

**Token-change rate (kept separate from Δ_t):**
```
TCR_t(i) = Hamming(tokens_branch, tokens_base) / L
```

**Mean estimates from N trajectories:**
```
Δ̄_t = (1/N) ∑_i Δ_t(i)
```

### Implementation

```python
y_base = gen.run_base(seed=seed_i)           # 1 run, T predictor steps + signal extraction
for t in range(T):
    y_branch = gen.run_branch(t_corrected=t, seed=seed_i)  # 1 run, T + corrector_steps steps
    gain = estimate_single_step_gain(y_base, y_branch, F="neg_nll")
    # gain["delta"] = Δ_t, gain["tcr"] = TCR_t
```

**Signals from `run_base` only** (signals not recorded in `run_branch` — cost saving):
```python
sigs = y_base["per_step_signals"][t]  # signals at step t from base trajectory
```

### NFE cost per trajectory pair

| Component | NFE count |
|---|---|
| `run_base`: T predictor calls | T |
| `run_branch(t)`: T predictor calls | T |
| `run_branch(t)`: corrector calls | corrector_steps = 2 |
| Total per (base, branch) pair | 2T + 2 |
| Total for one trajectory (N=1, all T branches) | T + T*(T + 2) = T(T+3) |
| T=64 total | 64 × 67 = 4,288 forward passes |

Note: `run_base` caches p_x0 (only new when cache misses), but we count conservatively.

---

## Protocol B on ProSeCo

### Definition

**Schedule S** of size |S| = B:
- A set of T steps S ⊂ {0, …, T-1} where corrector loops are applied

**Scheduled trajectory** `y_S`:
- Run T predictor steps
- At each t ∈ S: apply one ProSeCo corrector loop (same mechanism as Protocol A)
- Output: `tokens_S`, `neg_nll_S = F(y_S)`

**Joint schedule gain:**
```
G(S) = F(y_S) − F(y_base) = neg_nll_S − neg_nll_base
```

**Additive surrogate:**
```
A(S) = ∑_{t ∈ S} Δ̄_t    (using per-step means from Protocol A)
```

**Residual (additivity slack for one schedule S):**
```
r(S) = G(S) − A(S)
```

**Budget-level slack η_B:**
```
η_B ≈ 95th percentile of |r(S)| over M randomly sampled S with |S|=B
```

**Pairwise interaction ξ_{t,t'}:**
```
ξ_{t,t'} = G({t,t'}) − (Δ̄_t + Δ̄_{t'})
```

**Pairwise bound γ:**
```
γ ≈ 95th percentile of |ξ_{t,t'}| over P randomly sampled pairs
```

### Implementation

```python
# η_B estimation
for B in B_values:
    for m in range(M):
        allocation = random_sample(T, B)
        result = evaluate_schedule(allocation, delta_trace, gen, F="neg_nll")
        # result["G"] = G(S), result["A"] = A(S), result["residual"] = r(S)

# Pairwise γ estimation
for pair in P:
    t, tp = random pair
    y_base = gen.run_base(seed=...)
    y_t   = gen.run_branch(t_corrected=t, seed=...)
    y_tp  = gen.run_branch(t_corrected=tp, seed=...)
    y_pair = gen.run_with_schedule({t: 1, tp: 1}, seed=...)
    xi = (y_pair["neg_nll"] - y_base["neg_nll"]) - (delta_t + delta_tp)
```

---

## Signals at step t (Protocol A extraction)

Signals are extracted from `run_base` at each step t. They are computed from `p_x0_cache`
(already available after each predictor step — no extra NFE).

**Revisable set** (ProSeCo corrector action set):
```
R_t = {i : x_t[i] ≠ mask_id}    # currently UNMASKED positions
```

This is the set of positions that the ProSeCo corrector will revise.

**Why UNMASKED for ProSeCo:** ProSeCo's corrector applies its refined output to unmasked
positions only (`x_new[unmasked] = corrector_x[unmasked]`). The corrector refines committed
tokens. Signals over R_t measure uncertainty about already-committed tokens.

**Signal formulas:**

| Signal | Formula |
|---|---|
| Entropy | `H_t = mean_{i ∈ R_t} H(p_θ(x_i \| x_t))` (nats) |
| Inverse margin | `M_t^{-1} = 1 − mean_{i ∈ R_t} (p_1(i) − p_2(i))` |
| Quality mass proxy | `Q_t = 1 − mean_{i ∈ R_t} p_{argmax}(i)` |
| Unmasked fraction | `u_t = \|R_t\| / L` |
| n_revisable | `\|R_t\|` |
| n_masked | `L − \|R_t\|` |

**Signal trajectory expected behavior:**

- Steps 0–5 (u_t ≈ 0): R_t is empty → all signals = 0 → corrector has no action
- Steps 10–40 (u_t ≈ 0.15–0.6): R_t grows → signals rise as lower-confidence tokens commit
- Steps 55–63 (u_t ≈ 0.9–1.0): most tokens committed → signals reflect full-sequence uncertainty

**Thesis proxy hypothesis:** if Δ_t is positively correlated with H_t (or M_t^{-1} or Q_t),
then signal-adaptive scheduling can outperform uniform placement.

---

## Quality functional F

Primary: **negative per-token NLL under GPT-2 reference scorer** (higher = better).

```python
# Decoded tokens → text → re-encode with GPT-2 → compute NLL
enc = ref_tok(text, return_tensors="pt", truncation=True, max_length=512)
nll = ref_lm(enc["input_ids"], labels=enc["input_ids"]).loss.item()
F(y) = -nll   # negated so higher = better quality
```

**Limitations:**
- GPT-2 NLL is fast but not a perfect quality measure
- Truncated to 512 tokens from 1024 generated tokens
- Suitable for relative comparisons (Δ_t, G(S)) within the same experiment

**Alternative considered:** MAUVE score (matches generation distribution). Not used because
MAUVE requires many samples (not per-trajectory) and cannot measure per-step Δ_t.

---

## Baseline / reference

The existing MDLM heuristic pipeline (`backends/mdlm.py`, `run_phase1_pilot.py`) produces
the Phase 1 diagnostic results. It remains available as a comparison baseline:

- MDLM run: corrector harmful at all steps (Δ_t ≤ 0 everywhere), signals degenerate
- ProSeCo run: expects at least some Δ_t > 0 (corrector refines committed tokens)

The comparison between the two is informative for the thesis narrative (Section 4 of thesis).

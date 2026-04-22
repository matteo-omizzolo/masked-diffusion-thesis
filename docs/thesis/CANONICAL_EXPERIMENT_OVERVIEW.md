> **STATUS:** CANONICAL
> **LAST VERIFIED:** 2026-04-22
> **SCOPE:** Overview of experiment phases, protocols, and results (Phase 1 → Phase 3a).

---

# Canonical Experiment Overview

**Last updated:** 2026-04-18

This file is the canonical specification of the ProSeCo measurement pipeline
that underpins the thesis mainline. It documents the original Phase 1
Protocol A/B setup, which now serves as historical context and a prerequisite
for the Phase 2b / Phase 3a scheduling experiments. Sibling canonical file:
`CANONICAL_RESEARCH_DIRECTION.md`.

Source specs this synthesizes:
`docs/experiments/proseco_experiment_definition.md` (math),
`docs/experiments/proseco_protocol_mapping.md` (implementation),
`docs/experiments/proseco_backend_audit.md` (backend),
`docs/experiments/results/result_inventory.md` (runs),
`docs/experiments/entropy_proxy_experiment.md` (Protocol A/B design),
`docs/experiments/results/proseco_backend_failure_audit.md` (backend fixes).

---

## 1. Model and backend

**Backbone.** MDLM DiT, loaded from `~/mdm/checkpoints/mdlm.ckpt` via
`external/remdm/diffusion.py`'s `Diffusion.load_from_checkpoint`. Trained on
OpenWebText. DiT architecture: hidden 768, 12 heads, 12 blocks.
Tokenizer: GPT-2 (50,257 + 1 mask = 50,258). Sequence length L = 1024.
Noise schedule: loglinear. ~170M parameters total.

**Predictor.** MDLM absorbing-state DDPM, driven by `_ddpm_caching_update`.
Takes T predictor steps from noise level ~1 (fully masked) to noise level
~0 (fully unmasked), committing tokens in order of the model's confidence.
Per step it returns `p_x0_cache`, the model's posterior over clean tokens.
T = 64 for pilots, T = 128 for full runs.

**Corrector (current backend).** ProSeCo annealed iterative refinement,
implemented in `src/mdm_playground/scheduling/backends/proseco.py`
(`ProSeCoGenerator`), layered on top of the MDLM backbone. The corrector is
invoked only at explicitly specified step indices (via `corrector_at_t` for
Protocol A or an `allocation` dict for Protocol B) — the ProSeCo config
knobs `corrector_every_n_steps`, `corrector_start_iter`, `corrector_top_k`
are not used because Protocols A/B require per-step placement control.

At a step t where the corrector is applied, after the predictor step:

1. `x̂_0 = argmax(p_x0_cache_t)` — predicted clean sequence (all positions).
2. `corrector_x = x̂_0`.
3. For j in `range(corrector_steps)` (default `corrector_steps = 2`):
   - `tau_j = linspace(1.0, eps, corrector_steps + 1)[j]` → τ ∈ {1.0, 0.5}.
   - `log_p_corr = model.forward(corrector_x, sigma(tau_j))`.
   - `corrector_x = log_p_corr.argmax(-1)`.
4. `x_t_corrected[i] = corrector_x[i]` where `x_t[i] != mask_id`; masked
   positions are left untouched.

So the ProSeCo corrector acts on **unmasked** positions R_t only, using
argmax (deterministic) refinement from the predictor's argmax x̂_0 at
decreasing τ. One corrector loop = `corrector_steps` extra backbone NFEs
(2 by default). Mechanistic contrast with the MDLM Gibbs corrector:

| Aspect | MDLM heuristic | ProSeCo |
|---|---|---|
| Action set | masked positions | unmasked positions |
| Start | p(x_0 \| x_t) full posterior | x̂_0 = argmax(predictor) |
| Mechanism | one-shot Gibbs resample | annealed argmax refinement |
| NFE/loop | 1 | `corrector_steps` (=2) |
| Early-step effect | destructive (resamples ~97 % with <5 % context) | conservative (few unmasked → small effect) |

## 2. Quality functional F

Primary: negative per-token NLL under GPT-2 as an external reference scorer.

    F(y) = −(1/|y|) ∑_i log p_{GPT-2}(y_i | y_{<i})

Higher = better. Implementation decodes the token sequence, re-encodes with
GPT-2, and truncates to 512 tokens (from L = 1024 generated) to avoid OOM.
Valid for relative comparisons (Δ_t, G(S)) within a run; not used as an
absolute quality claim. MAUVE is not used because it needs many samples and
cannot score per-step Δ_t. PRISM-style quality head is not available on the
MDLM backbone.

## 3. Protocol A — per-step one-loop marginal gain

For each of N trajectories (pilot N = 20, full N = 50) and each step
t ∈ {0, …, T−1}:

**Base** `y_base(i)`. Seed `seed_i = seed_base + i`. Run T predictor steps
with `torch.manual_seed(seed_i)`. No corrector. Record `tokens_base`,
`neg_nll_base = F(y_base)`, and the per-step signals from the base
trajectory.

**Branch at t** `y_t^{+1}(i)`. Same seed (Common Random Numbers for
variance reduction). Run T predictor steps; at step t apply exactly one
ProSeCo corrector loop to x_t before continuing. Record `tokens_branch`,
`neg_nll_branch = F(y_t^{+1})`.

**Marginal gain:**

    Δ_t(i) = F(y_t^{+1}(i)) − F(y_base(i)) = neg_nll_branch − neg_nll_base

**Token-change rate** (deliberately kept separate from Δ_t):

    TCR_t(i) = Hamming(tokens_branch, tokens_base) / L

Δ_t measures impact on quality functional; TCR_t measures how many tokens
the corrector moved. High TCR with small Δ is "corrector active but
poorly targeted" (the distinction is flagged in the GPT Pro assessment,
item A9).

Aggregates across trajectories: Δ̄_t = (1/N) ∑_i Δ_t(i), similarly H̄_t, etc.

**NFE cost.** Per trajectory, Protocol A costs T predictor NFEs for base
plus T · (T + `corrector_steps`) for all T branches, i.e., T(T+3) for
`corrector_steps = 2`. At T = 64 that is 4,288 forward passes per trajectory.

**Calibration ε.** Fit ψ(s_t) = a·s_t + b by least squares over all
(trajectory, t) pairs, compute the RMS residual:

    ε_rms = sqrt( (1/(NT)) ∑_{t,i} (Δ_t(i) − ψ(s_t(i)))² )

Computed separately for each signal candidate.

Implementation: `gen.run_base(seed=...)` plus a loop of
`gen.run_branch(t_corrected=t, seed=...)` in
`archive/legacy_framework/scripts/run_phase1_proseco.py`.

## 4. Protocol B — schedule-level additivity and interaction

For each budget B ∈ {4, 8, 16} (at T = 64; equivalently T/16, T/8, T/4):

**η_B estimation.** Sample M random schedules S with |S| = B (primary M = 15,
sensitivity M = 30–100). For each S:

- `y^S` = full predictor run applying one corrector loop at every t ∈ S.
- G(S) = F(y^S) − F(y_base) (joint gain for schedule S).
- A(S) = ∑_{t∈S} Δ̄_t (additive surrogate using per-step means from
  Protocol A).
- residual r(S) = G(S) − A(S).

Estimator: η_B ≈ 95th-percentile_M |r(S)|.

**Pairwise γ estimation.** Sample P random pairs (t, t') (primary P = 120).
For each pair compute

    ξ_{t,t'} = G({t, t'}) − (Δ̄_t + Δ̄_{t'})

Estimator: γ ≈ 95th-percentile_P |ξ_{t,t'}|.

**G(S) definition (joint gain across schedule).** G(S) is measured
directly by running `gen.run_with_schedule(allocation, seed=...)` — no
surrogate is used for this quantity; only for A(S).

**Proposition C check.** Compare η_B^{emp} to γ · B(B−1)/2. If the
inequality holds with a gap, the pairwise approximation explains the
additivity slack; if empirical η_B exceeds the pairwise bound, higher-order
interactions are non-negligible.

## 5. Signals logged

All signals are computed from `p_x0` at step t of the base trajectory — no
extra NFEs. Let p_i = p_θ(x_0^i | x_t) ∈ Δ(V).

**Revisable set** — the ProSeCo corrector's action set:

    R_t = { i : x_t[i] ≠ mask_id }     (UNMASKED positions)

This is *not* the MDLM-heuristic action set (which would be masked
positions). For ProSeCo, the corrector refines committed tokens, so the
signals describing "where could correction help" are defined over the
already-committed positions R_t.

| Signal | Formula | Intuition |
|---|---|---|
| Entropy H_t | (1/|R_t|) ∑_{i∈R_t} H(p_i) [nats] | high → uncertainty about committed tokens |
| Inverse margin M_t^{-1} | 1 − (1/|R_t|) ∑_{i∈R_t} (p_i^{(1)} − p_i^{(2)}) | high → top two logits close |
| Quality mass proxy Q_t | 1 − (1/|R_t|) ∑_{i∈R_t} p_i^{(1)} | high → low top-1 probability |
| Unmasked fraction u_t | |R_t| / L | deconfounder for signal vs trajectory phase |
| n_revisable | |R_t| | size of corrector action set |
| n_masked | L − |R_t| | remaining mask budget |

At t ≈ 0, R_t ≈ ∅ → all signals = 0 → natural burn-in: ProSeCo's corrector
has no action, which makes Proposition B's "low-gain-region gating" hold by
construction for early t.

Thesis proxy hypothesis: if Δ_t correlates positively (after regressing out
u_t) with H_t, M_t^{-1}, or Q_t, then signal-adaptive scheduling can beat
uniform. The sign/strength of that correlation is what ε measures.

## 6. Meaning of ε, η_B, γ experimentally

- **ε** = RMS residual of a least-squares calibration of ψ(s_t) to Δ_t
  across all (trajectory, t). Small ε → proxy tracks Δ_t well → Theorem A's
  2Bε term is small.
- **η_B** = 95th-percentile |G(S) − ∑Δ_t| across M random size-B schedules.
  Small η_B → gains are nearly additive over step-t choices → Theorem A's
  2η_B term is small.
- **γ** = 95th-percentile |ξ_{t,t'}| across P random pairs. Links η_B
  to a pairwise expansion via Proposition C (η_B ≤ γ B(B−1)/2).

Theorem A is "useful" on a run iff 2Bε + 2η_B < G(Ŝ_B) at the B of interest.

## 7. Previous result — MDLM diagnostic (negative)

**Job 478600** (April 2026, T = 64, N = 20, M = 15, P = 120, 70 min on A100).
MDLM Gibbs-style heuristic corrector (`backends/mdlm.py`):

- Δ̄_t ≤ 0 for all t; corrector uniformly harmful.
- Signals (entropy, inverse margin, quality mass) identically zero due to
  Bug #1 (computed over already-committed, thus near-certain, positions
  rather than masked positions).
- Bug #2: corrector resamples ALL masked positions in one shot; at t ≈ 0
  that is ~97 % of the sequence from a model with ~1.4 % context.

Documented as a structural negative result: even with the signal bug
fixed, the corrector is a resample-everything kernel that cannot be
beneficial. The MDLM heuristic is preserved as a diagnostic baseline.
Details: `docs/experiments/results/result_inventory.md` and
`docs/experiments/phase1_interpretation.md`.

## 8. Previous result — ProSeCo run (structural no-op)

**Job 478929** (2026-04-17, T = 64, N = 20, corrector_steps = 2, 59 min
on A100). After three failed attempts (weights_only bug, then
ConfigAttributeError from a hand-written minimal OmegaConf — root cause
documented in `docs/experiments/results/proseco_backend_failure_audit.md`),
the backend was fixed to hydrate its config from the checkpoint's
`hyper_parameters["config"]` and to recompute `p_x0` for signals when the
cache is None. The pilot then ran cleanly to completion.

KEY RESULTS (from `archive/legacy_framework/results/phase1_proseco/summary.json`):

| Metric | Value |
|---|---|
| Steps with Δ_t > 0 | 0 / 64 |
| Peak mean Δ_t | 1.0 (only at t = 63 — trivial boundary) |
| ε (entropy, RMS) | 0.00000 |
| ε (margin, RMS) | 0.00000 |
| Spearman(H, Δ) | 0.000 ± 0.000 |
| γ (95th pct) | 0.00000 |
| η_B (all B) | 0.00000 |

**Scientific interpretation.** The ProSeCo annealed-refinement corrector
overwrites unmasked positions with argmax p_x0 from a re-run at a lower
noise level. On the MDLM backbone, which was not co-trained with a
corrector loss, this re-run returns the same argmax tokens as the
predictor → corrector is a no-op → Δ_t = 0 for all t. Runtime
infrastructure is confirmed correct; the result reflects a fundamental
backbone/corrector mismatch. Theorem A cannot be calibrated on this pair.

## 9. Next experiment goal

Restore a non-degenerate Δ_t profile — at minimum a few t with Δ_t > 0 —
so Theorem A's ε, η_B, γ can be estimated on real data. Candidate
backbone/corrector pairs:

1. **`kuleshov-group/proseco-owt` checkpoint** (HuggingFace,
   ~500 MB). Co-trained with a corrector loss → annealed refinement should
   produce Δ_t > 0 somewhere. Existing `backends/proseco_owt.py` code loads
   the staged snapshot directly.
2. **MDLM-conf partial resample** on `mdlm.ckpt`: corrector resamples only
   K lowest-confidence masked positions (not all masked positions).
   Expected to give non-zero Δ_t, likely mixed sign. An initial
   `backends/mdlm_conf.py` exists.
3. **ReMDM-conf**, which ships with a confidence-guided corrector trained
   jointly with the predictor. Requires cloning and validating the
   ReMDM-conf checkpoint.

None of these changes the protocol specification in §§3–6 above. Once a
pair yields Δ_t ≠ 0, the existing Protocol A/B pipeline, signal extraction,
and summary computation run unchanged. Full next-step breakdown:
`docs/implementation_plan.md` (Phase 1) and the status block at the end
of `docs/experiments/results/result_inventory.md`.

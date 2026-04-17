# Implementation Status — Scheduling Package and Phase 1 Experiment

**Updated:** April 2026

Tracks what has been implemented under `src/mdm_playground/scheduling/` and
the state of the Phase 1 experiment pipeline.

---

## Scheduling package — API

The package `src/mdm_playground/scheduling/` provides four modular abstractions
aligned with the Protocol A / B measurement needs:

| Function | Role |
|----------|------|
| `compute_signals(state, logits, meta) -> dict` | Given the current state Z_t and logits, return a dict of aggregate trajectory signals (entropy, inverse margin, quality mass proxy). |
| `estimate_single_step_gain(step_state, run_base, run_branch) -> dict` | Given `run_base` (baseline completion from Z_t onwards) and `run_branch` (completion after one corrector loop at Z_t), return Δ_t under a chosen F, TCR_t, and diagnostics. |
| `allocate_budget(signal_trace, budget, policy_name, policy_kwargs) -> allocation` | Return `{t: k_t}` for a policy name ∈ {uniform, top_B, entropy_proportional, burn_in_gated, margin_top_B, quality_top_B, front, back, middle}. |
| `evaluate_schedule(allocation, generator, F) -> metrics` | Run generation with the given allocation; return G(S), A(S), TCR totals, MAUVE, gen-PPL, wall time. |

These are defined as pure-Python functions taking explicit state dictionaries
so the experiment harness can swap in either the real MDLM+ReMDM backend or
a synthetic Gaussian surrogate for pipeline testing.

See `src/mdm_playground/scheduling/__init__.py` for exports and
`tests/scheduling/` for unit tests.

---

## Current state

- [x] `compute_signals` — implemented for entropy, inverse margin, and
  quality-mass placeholder (quality head not yet wired).
- [x] `estimate_single_step_gain` — implemented with common-random-numbers
  branch generator.
- [x] `allocate_budget` — policies implemented: uniform, top_B, entropy_proportional,
  burn_in_gated (with configurable T_low), margin_top_B, quality_top_B,
  front-loaded, back-loaded, middle-loaded.
- [x] `evaluate_schedule` — computes G(S), A(S), residual, MAUVE,
  gen-PPL. Writes JSON log.
- [x] Unit tests for signal monotonicity and allocation correctness.
- [ ] Real MDLM/ReMDM backend wiring (`backends/mdlm.py`) — stub implemented;
  awaits HPC checkpoint loader to validate.
- [ ] PRISM quality head wiring (`backends/quality.py`) — deferred to Phase 4.

---

## Phase 1 experiment pipeline

The experiment is driven by `scripts/run_phase1_pilot.py` with stages:

1. **Protocol A.** For each trajectory i ∈ {1, …, N}:
   - Seed σ_i.
   - Generate y_base.
   - For t ∈ {1, …, T}: generate y_t^{+1} with CRN, record
     {Δ_t, TCR_t, s_t}.
   - Save `results/protocol_a/trajectory_{i}.json`.
2. **Protocol B — η_B sampling.** For each B ∈ {T/16, T/8, T/4} and
   each of M sampled schedules S:
   - Generate y^S.
   - Record G(S), A(S), residual.
   - Save `results/protocol_b/schedule_{j}.json`.
3. **Protocol B — pairwise γ.** For P sampled pairs (t, t'):
   - Record ξ_{t, t'}.
   - Save `results/protocol_b/pairs.json`.
4. **Analysis.** `scripts/analyze_phase1.py` aggregates JSON logs into
   summary tables + figures.

---

## HPC workflow for Phase 1

Pilot configuration (fits in a single `stud` job):

- N = 5 trajectories
- T = 64 predictor steps
- Branches: all 64 steps per trajectory
- M = 15 sampled schedules × 3 B values
- P = 60 pairs
- Estimated ~3 A100h

Submission: `bash hpc/submit_phase1_pilot.sh`. Writes results under
`~/mdm/results/phase1_pilot/`. Pull with rsync (macOS-safe path):

```bash
rsync -avz 3316152@slogin.hpc.unibocconi.it:~/mdm/results/phase1_pilot/ \
  /sessions/dazzling-busy-goldberg/mnt/masked-diffusion-thesis/results/phase1_pilot/
```

Full Phase 1 (after pilot validates): N = 50, T = 128, M = 30, P = 300.

---

## Logging format

All results are JSON, one file per entity:

```
results/phase1_pilot/
  protocol_a/
    trajectory_0.json      # {seed, T, per_t: [{t, Δ_t, TCR_t, signals:{...}}, ...]}
    ...
  protocol_b/
    schedule_0.json        # {seed, B, S, G, A, residual}
    ...
    pairs.json             # [{t, tp, Δ_t, Δ_tp, G, xi}, ...]
  summary.json             # aggregate ε, η_B, γ, T_low
```

`scripts/analyze_phase1.py` converts these into:

- `figures/phase1_pilot/calibration_scatter.png` — ψ vs Δ_t per signal
- `figures/phase1_pilot/delta_vs_t.png` — mean Δ_t as function of t
- `figures/phase1_pilot/eta_vs_B.png` — η_B scaling with B
- `figures/phase1_pilot/pairwise_xi_hist.png` — |ξ| histogram
- `figures/phase1_pilot/theorem_A_budget.png` — 2Bε + 2η_B vs G(Ŝ_B)

---

## Surrogate mode

When `--surrogate` is passed, the pipeline uses a synthetic generator where
Δ_t is a known function of t plus noise, and interactions have a chosen γ.
This mode validates the pipeline (CRN, analysis code, figure generation)
before spending HPC time. Primary value: smoke-testing plot code and decision
logic end-to-end.

See `scripts/run_phase1_pilot.py --surrogate` and
`tests/scheduling/test_surrogate_end_to_end.py`.

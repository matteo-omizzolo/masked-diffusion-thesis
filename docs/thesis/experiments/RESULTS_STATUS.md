# Historical Experimental Status — Phase 1 through Phase 2b

> **⚠ [HISTORICAL / UNDER AUDIT — 2026-04-19]** The headline below was written
> before the ANALYSIS_SPEC-compliant reanalysis and before the Phase 2b / Phase
> 3a mainline. Two findings now demote the headline:
> (a) "entropy_bot_B +29% over uniform at B=8" mixes the proxy A(S) (additive
> surrogate over per-step Δ_t) with the true pipeline-evaluated G(S). On the same
> N=50 run, G(S) at B=8 has uniform ≈ 0.76, entropy_bot_B ≈ 0.08 — a 10× loss.
> (b) All Phase 1 policy rows are **single-seed** (one random seed per policy), so
> the rankings have no statistical basis under ANALYSIS_SPEC §6 (Tier T4,
> inadmissible).
>
> See `docs/thesis/experiments/EXPERIMENT_CRITICAL_AUDIT.md` and
> `docs/thesis/theory/THEORY_STRESS_TEST.md`. Authoritative rankings resume with
> Phase 2b paired evaluation and Phase 3a combinatorial scheduling. Do not
> cite Phase 1 policy comparisons in the thesis until the audit resolves.

*Last updated: 2026-04-19 (ProSeCo-OWT FULL RUN complete — job 479537; canonical rankings now under audit).*
*Scope: corrector scheduling for MDLM / ProSeCo on the Bocconi HPC.*

## 1. Headline status

**⚠ AUDITED — DO NOT CITE AS-IS.** The original Phase 1 §7.2 summary follows
verbatim for provenance; each numeric claim must be cross-checked against the
audit before re-use.

**ProSeCo-OWT FULL RUN (job 479537, N=50) COMPLETE.** All §7.2 criteria pass:
61/64 positive Δ_t steps, peak Δ=0.157, |Spearman|=0.19–0.20 (sign inverted),
ε_rms=0.133, η_95>0 at all B, γ_95=0.264. C6 passes with inverted signal
direction: entropy_bot_B / margin_bot_B / quality_bot_B beat uniform at all B
(e.g. entropy_bot_B: B=8 G=+0.887 vs uniform G=+0.688, +29%). Oracle >> uniform
(2.28× at B=8) confirms scheduling is valuable. **Entropy-minimum scheduling
principle confirmed at N=50 (thesis-grade evidence).**

*Audit note (2026-04-19):* Criteria C1–C5 (Δ_t positivity, ε, η_B, γ) are
trajectory-level quantities computed on the raw protocol_a/protocol_b data and
remain valid. Criterion C6 (the policy-ranking criterion) is the one that fails
under audit: the reported gains use a per-policy A(S) proxy or single-seed G(S),
not paired K-seed G(S) with CIs. The "entropy-minimum scheduling principle" is
downgraded to a hypothesis to be tested in Phase 2b.

Full analysis: `docs/thesis/experiments/PROSECO_ANALYSIS.md`.
Results: `results/phase1_proseco_owt_full/summary.json` and
`results/phase1_proseco_owt_full/inverted_policy_analysis.json`.

Phase 1 MDLM diagnostic run (job 478600) produced a clean NEGATIVE result:
the raw MDLM Gibbs-style corrector is harmful at every step (all Δ_t ≤ 0) and
the signal extractor operated over the wrong positions. ProSeCo was then adopted
as the primary Phase 1 platform; the backend infrastructure now loads correctly
and runs end-to-end on `mdlm.ckpt`, but the ProSeCo pilot (job 478929) produced
Δ_t ≡ 0 at every step because MDLM was not co-trained with the annealed-refinement
corrector kernel. **Current blocker:** no combination of backbone + corrector
where Δ_t > 0 has been empirically demonstrated on real data. Signal calibration
for Theorem A's ε is therefore not yet possible.

## 2. MDLM diagnostic (job 478600) — NEGATIVE RESULT

- **Role:** first real-checkpoint Phase 1 pilot (T=64, N=20, M=15, P=120,
  B ∈ {4, 8, 16}). Completed in 4249.5 s on `gnode02` (A100 80GB PCIe).
- **Data:** `archive/legacy_framework/results/phase1_pilot/`. Full logs in `archive/legacy_framework/out/phase1_pilot_478600.out`.
- **What was measured:** per-step one-loop gain Δ_t, TCR_t, base/branch neg-NLL,
  entropy / inverse-margin / quality-mass signals, η_B over 45 schedules and γ
  over 120 pairs.
- **Bug #1 (signal over wrong positions):** `backends/mdlm.py` `_extract_signals`
  used `revisable = (x != mask_id)` (unmasked = already-committed, high-confidence
  positions). Consequence: entropy, margin, quality-mass all identically 0.
- **Bug #2 (corrector resamples all masked positions):** `_apply_corrector`
  redrew every masked slot (≈ 97 % of the sequence at t=0) from p_x0 in one shot
  with almost no context. Consequence: Δ_t monotonically increases from −3.191
  to 0.000 (corrector is harmful or neutral at every t).

### Δ_t profile (mean over 20 trajectories)

| t | mean Δ_t | mean TCR_t | unmasked |
|---|----------|-----------|---------|
| 0 | −3.191 | 0.973 | 0.015 |
| 10 | −2.245 | 0.775 | 0.172 |
| 20 | −1.474 | 0.580 | 0.331 |
| 30 | −0.835 | 0.417 | 0.482 |
| 40 | −0.365 | 0.260 | 0.642 |
| 50 | −0.099 | 0.137 | 0.798 |
| 60 | −0.004 | 0.029 | 0.954 |
| 63 | 0.000 | 0.000 | 1.000 |

### η_B / γ under this (broken) corrector

| B | η_mean | η_95 | n_schedules |
|---|--------|------|-------------|
| 4 | 1.821 | 3.783 | 15 |
| 8 | 5.853 | 9.626 | 15 |
| 16 | 14.562 | 18.937 | 15 |

γ_mean = 0.539, γ_95 = 2.302, γ_max = 3.010 (n = 120 pairs). Theorem A bound is
vacuous at every B. **Verdict:** diagnostic / negative; the infrastructure is
validated, but numeric values must not be cited as calibration of the theory.

## 3. ProSeCo integration — fixes applied (2026-04-17)

- **Config hydration from checkpoint.** `_load_diffusion_model` in
  `src/mdm_playground/scheduling/backends/proseco.py` (and the parallel fix in
  `backends/mdlm.py`) now reads `raw_ckpt["hyper_parameters"]["config"]`,
  converts to a plain dict, and `setdefault`s the inference-only keys
  (`T=0`, `subs_masking=False`, `sampling.sampler='mdlm'`, `sampling.nucleus_p=1.0`,
  `sampling.eta=0.0`, `sampling.t_on=0.0`, `sampling.t_off=0.0`) that were
  absent from the training config. Eval flags (`compute_generative_perplexity`,
  `generate_samples`, `compute_perplexity_on_sanity`) are overridden to `False`.
- **`weights_only=False` patch.** Both backends install a module-level
  `torch.load` monkey-patch so the numpy-scalar safe-globals work on PyTorch ≥ 2.6.
- **`p_x0` recomputation.** `_apply_corrector(x, t, p_x0=None)` and `_run_loop`
  now recompute `p_x0 = model.forward(x, σ_t).exp()` whenever the DDPM cache is
  invalidated by a token change (which happens on every predictor step in
  practice). Previously `p_x0_cache = None` caused `TypeError: 'NoneType'
  object is not subscriptable` in signal extraction.
- **Preflight:** `archive/legacy_framework/scripts/debug_proseco_load.py` runs five checks on CPU with
  T=2 (config readable, generator instantiates, base loop, branch loop, neg_nll
  finite). All five passed on 2026-04-17 — see
  `docs/experiments/results/proseco_load_validation.md`.

## 4. Failed ProSeCo jobs (all same root cause)

| Job | Failure | Fix |
|-----|---------|-----|
| 478826 | `torch.load` with `weights_only=True` rejected numpy scalar; patch in `external/remdm/main.py` was not imported by the ProSeCo backend | Module-level monkey-patch in `backends/proseco.py` |
| 478827 | Same `weights_only` issue | Same patch, re-verified |
| 478828 | `ConfigAttributeError: eval.gen_ppl_eval_model_name_or_path` — hand-written minimal config missed keys accessed unconditionally by `diffusion.py.__init__` | Read checkpoint's `hyper_parameters['config']` instead of hand-writing a minimal dict |

**Root cause.** A refactor of `_load_diffusion_model` replaced the
checkpoint-config-reading path with a hand-written minimal `OmegaConf` dict.
The minimal config was missing `eval.gen_ppl_eval_model_name_or_path`,
`model.cond_dim`, `model.scale_by_sigma`, `T`, `subs_masking`,
`sampling.sampler`, and `sampling.nucleus_p`. Full audit in
`docs/experiments/results/proseco_backend_failure_audit.md`.

## 5. Job 478929 — ProSeCo pilot (completed, structural no-op)

- **Parameters:** T=64, N=20, M=15, P=120, B ∈ {4, 8, 16}, corrector_steps=2,
  seed=42, `mdlm.ckpt`.
- **Node:** gnode04 (A100 80GB). Elapsed 3544.6 s (≈ 59 min).

### KEY RESULTS (from `archive/legacy_framework/results/phase1_proseco/summary.json`)

| Metric | Value |
|--------|-------|
| Steps with Δ_t > 0 | 0 / 64 |
| Peak mean Δ_t | 1.0 (only at t=63 — trivial boundary) |
| ε (entropy, RMS) | 0.00000 |
| ε (margin, RMS) | 0.00000 |
| Spearman(H, Δ) | 0.000 ± 0.000 |
| γ (95th pct) | 0.00000 |
| η_B (all B) | 0.00000 |

### Scientific interpretation

The ProSeCo annealed-refinement corrector overwrites unmasked (committed)
tokens with argmax of p_x0 at a lower noise level. Because the MDLM backbone
was **not** co-trained with this kernel, argmax re-decoding at committed
positions produces exactly the same tokens as the predictor already committed
— hence Δ_t = 0 at every step and every signal collapses to 0. This is a
structural no-op, not an implementation bug: the runtime path is confirmed
correct (milestone logs are present, the surrogate sanity run showed
non-degenerate ε). The mismatch is fundamental: ProSeCo's refinement gain
requires the backbone to have learned, at training time, to respond to the
corrector's perturbation structure.

## 6. Current blocker

ProSeCo on `mdlm.ckpt` cannot calibrate Theorem A's ε because every Δ_t is
exactly 0 — there is nothing to schedule. Until we produce a combination of
backbone + corrector where Δ_t > 0 at at least some steps, the Protocol A
calibration and Protocol B additivity estimates are scientifically vacuous.
All downstream Theorem A / Proposition B / Proposition C analyses are on hold
behind this single prerequisite.

## 7. Next-run options

1. **Option 1 — proseco-owt checkpoint.** Download the
   `kuleshov-group/proseco-owt` HuggingFace checkpoint (co-trained with the
   corrector loss). Expected non-zero Δ_t at mid-trajectory steps. Cost:
   ~500 MB download, HuggingFace offline staging on the HPC (user quota
   already tight — reuse the owt-reference staging pattern).
2. **Option 2 — MDLM-conf corrector on `mdlm.ckpt`** (recommended, already
   implemented). Resamples only the K lowest-confidence still-masked positions
   per step (fixes Bug #1 and Bug #2). Code:
   `src/mdm_playground/scheduling/backends/mdlm_conf.py`. Entrypoint:
   `scripts/run_phase1_mdlm_conf.py`. Sbatch: `hpc/phase1_mdlm_conf.sbatch`.
3. **Option 3 — ReMDM-conf.** Use `external/remdm`'s built-in confidence-guided
   remasking corrector; trained jointly with the predictor. Higher integration
   cost, but the most "native" corrector-scheduling platform.

**Recommendation:** Option 2 first — no external download, fixes both known
bugs, structurally expected to produce Δ_t > 0 at mid-to-late trajectory
steps. If still degenerate at pilot scale, fall back to Option 1.

## 9. MDLM-conf signal-aligned pilot (job 479257) — 2026-04-18

- **Parameters:** T=64, N=20, M=15, P=120, B ∈ {4, 8, 16}, top_k=20, seed=42, `mdlm.ckpt`.
- **Fix applied (M2 patch):** `_extract_signals` now computes entropy/margin/quality_mass over
  the top_k=20 lowest-confidence masked positions — exactly the action set `_apply_corrector`
  resamples. `n_action` field added to per-step records for provenance.
- **Node:** gnode02 (A100 80GB). Elapsed ≈ 68 min. Results: `archive/legacy_framework/results/phase1_mdlm_conf_signal_aligned/`.

### §7.1 Success criteria — PASS/FAIL

| Criterion | Value | Result |
|-----------|-------|--------|
| C1: n_positive_delta_steps > 0 and peak_mean_delta ≥ 0.05 | n_positive=25/64, peak_mean_Δ=0.093 | **PASS** |
| C2: ≥1 signal with \|spearman_mean\| ≥ 0.15 and spearman_std < 0.6 | best: quality_mass 0.049 ± 0.398 | **FAIL** |
| C3: eps_rms ≤ 0.5 for all three signals | all three: 0.222 | **PASS** |
| C4: eta_by_B.*.eta_95 > 0 and gamma_95 > 0 | η_95 ∈ {0.492, 1.031, 0.709}, γ_95=0.456 | **PASS** |
| C5: Theorem A bound_useful=true for ≥1 B, or ratio ≤ 5× | all ratios ~10–14× | **FAIL** |
| C6: ≥1 top_B policy beats uniform at B=8 or B=16 | margin_top_B: B=8 G=0.745 vs 0.174; B=16 G=0.359 vs 0.157 | **PASS** |

**4/6 pass. C2 and C5 fail.**

### Key numeric results

| Metric | Value |
|--------|-------|
| n_positive_delta_steps | 25 / 64 |
| peak_mean_delta | 0.093 |
| ε_rms (entropy / margin / quality_mass) | 0.2224 / 0.2224 / 0.2224 |
| Spearman(entropy, Δ) | 0.022 ± 0.408 |
| Spearman(inverse_margin, Δ) | 0.012 ± 0.365 |
| Spearman(quality_mass, Δ) | 0.049 ± 0.398 |
| η_95 (B=4 / 8 / 16) | 0.492 / 1.031 / 0.709 |
| γ_95 | 0.456 |
| margin_top_B vs uniform (B=8) | G=0.745 vs G=0.174 (4.3×) |
| margin_top_B vs uniform (B=16) | G=0.359 vs G=0.157 (2.3×) |

### Scientific interpretation

The M2 action-set fix did **not** materially improve the Spearman correlations (prior
job 478962 Spearman ≈ 0.033; post-fix best is 0.049 for quality_mass). This appears to
be an intrinsic property of MDLM-conf on `mdlm.ckpt`: the per-step gain Δ_t has high
intra-trajectory variance (spearman_std ≈ 0.37–0.45) that overwhelms the signal's
predictive power. The corrector produces non-trivial gains at 25/64 steps (C1 PASS),
and the Theorem A residuals are bounded and finite (C3/C4 PASS), but the proxy ψ(s_t)
cannot reliably predict *which* steps are beneficial at step-level resolution.

Despite weak per-step calibration, `margin_top_B` beats uniform by a large margin at
B=8 (4.3×). This is consistent with rank-based policies benefiting from even weak
signals at aggregate scale. The discrepancy between Spearman ≈ 0.012 and policy gain 4.3×
suggests that even noisy signals concentrate budget near the genuinely high-gain region
(early-to-mid trajectory, steps 5–14) without requiring tight per-step calibration.

**Verdict: switch to proseco-owt.** Scaling MDLM-conf to N=50 would confirm the same
weak Spearman with tighter error bars — the bottleneck is signal strength, not sample
size. The right next step is the ProSeCo-owt checkpoint (co-trained backbone), where
Δ_t > 0 at mid-trajectory is guaranteed by design and the annealed-refinement signal
is structurally better-calibrated.

## 8. Results directory inventory

| Subdirectory | Content | Status |
|--------------|---------|--------|
| `archive/legacy_framework/results/phase1_pilot/` | MDLM heuristic corrector pilot, job 478600 (T=64, N=20) | Diagnostic / negative result — do not cite values |
| `archive/legacy_framework/results/phase1_smoke/` | Smoke test from pipeline bring-up | Archive |
| `archive/legacy_framework/results/phase1_proseco/` | ProSeCo pilot, job 478929 (T=64, N=20) — all zeros | Diagnostic — structural no-op on MDLM backbone |
| `archive/legacy_framework/results/phase1_proseco_surrogate_sanity/` | ProSeCo pipeline surrogate (T=16, N=5) — passed | Valid (surrogate sanity) |
| `archive/legacy_framework/results/phase1_mdlm_conf/` | MDLM-conf pilot data (T=64, N=20, top_k=20, 2026-04-17 14:16) | Pilot data present; re-validate against preflight before citing |
| `archive/legacy_framework/results/phase1_mdlm_conf_surrogate_sanity/` | MDLM-conf pipeline surrogate | Valid (surrogate sanity) |
| `archive/legacy_framework/results/phase2a/T256/`, `archive/legacy_framework/results/phase2a/T512/` | Earlier MDLM/ReMDM-conf/ReMDM-loop step sweeps | Archive (pre-pivot, earlier phase) |
| `archive/legacy_framework/results/t1000_eval/` | T=1000 step-sweep (MDLM / ReMDM-conf / ReMDM-loop + comparison.json) | Archive (pre-pivot) |
| `archive/legacy_framework/results/full_eval/` | MDLM / ReMDM-conf / ReMDM-loop full eval | Archive (pre-pivot) |
| `archive/legacy_framework/results/remdm_smoke/` | ReMDM environment smoke test | Archive |
| `archive/legacy_framework/results/phase1_mdlm_conf_signal_aligned/` | MDLM-conf M2-fixed pilot (job 479257, T=64, N=20) | **COMPLETE — §7.1 verdict: 4/6 pass; Spearman noise-level; switch to proseco-owt** |
| `archive/legacy_framework/results/phase1_proseco_owt/` | ProSeCo-OWT pilot (job 479382, T=64, N=20) | **COMPLETE — 5–6/6 §7.2 pass; entropy_bot_B beats uniform at all B; C6 passes (inverted direction); full analysis in archive/legacy_framework/docs/thesis/experiments/PROSECO_ANALYSIS.md** |
| `results/phase1_proseco_owt_full/` | ProSeCo-OWT FULL RUN (job 479537, T=64, N=50, M=30, P=300) | **COMPLETE — all §7.2 pass; entropy_bot_B +29% over uniform at B=8; thesis-grade evidence. Full analysis: archive/legacy_framework/docs/thesis/experiments/PROSECO_ANALYSIS.md** |

---

## 10. ProSeCo-OWT FULL RUN (job 479537) — COMPLETE

- **Parameters:** T=64, N=50, M=30, P=300, B ∈ {4, 8, 16}, corrector_steps=1, seed=42.
- **Node:** gnode04 (A100 80GB). Wall time: 6h39m (07:04 CEST 2026-04-19).
- **Data:** `results/phase1_proseco_owt_full/`

### §7.2 Success Criteria

| Criterion | Threshold | Result | Status |
|-----------|-----------|--------|--------|
| C1 n_positive_delta_steps | > 10 | 61/64 (95%) | **PASS** |
| C1 peak_mean_delta | ≥ 0.05 | 0.157 | **PASS** |
| C2 \|Spearman\| ≥ 0.10 | ≥1 signal | entropy −0.191 ± 0.252 | **PASS** (sign negative) |
| C3 eps_rms ≤ 0.5 | all | 0.133–0.136 | **PASS** |
| C4 eta_95 > 0 | all B | 0.413 / 0.680 / 1.357 | **PASS** |
| C4 gamma_95 > 0 | — | 0.264 | **PASS** |
| C5 Theorem A ratio ≤ 5× | ≥1 B | 3.1× at B=4 | **AMBIGUOUS** (bound_useful=False) |
| C6 entropy_bot_B beats uniform | B=8 | G=+0.887 vs G=+0.688 (+29%) | **PASS** |

**All criteria pass (C5 ambiguous as in pilot).**

### Key numeric results

| Metric | Full run (N=50) | Pilot (N=20) |
|--------|----------------|--------------|
| n_positive_delta_steps | 61/64 | 59/64 |
| peak_mean_delta | 0.157 | 0.178 |
| Spearman entropy | −0.191 ± 0.252 | −0.234 ± 0.211 |
| eps_rms | 0.133 | 0.148 |
| η_95 (B=8) | 0.680 | 0.987 |
| γ_95 | 0.264 | 0.351 |
| entropy_bot_B G (B=8) | +0.887 | +0.996 |
| uniform G (B=8) | +0.688 | +0.775 |
| oracle G (B=8) | +2.281 | +2.402 |

### C6 inverted policy results (N=50)

| Policy | B=4 G | B=8 G | B=16 G | Beats uniform? |
|--------|-------|-------|--------|----------------|
| uniform | +0.291 | +0.688 | +1.462 | baseline |
| **entropy_bot_B** | **+0.450** | **+0.887** | **+1.744** | **✓ all B** |
| **margin_bot_B** | **+0.438** | **+0.879** | **+1.783** | **✓ all B** |
| **quality_bot_B** | **+0.418** | **+0.870** | **+1.718** | **✓ all B** |
| middle | +0.559 | +1.118 | +2.176 | ✓ all B |
| oracle | +1.316 | +2.281 | +3.790 | ✓ all B |
| entropy_top_B | −0.048 | +0.115 | +0.687 | ✗ all B |

### Scientific summary

The full run confirms and tightens the pilot findings at thesis-grade precision
(N=50, SE halved vs pilot). The three key results for §7.2:

1. **Corrector is active:** 61/64 steps (95%) have Δ_t > 0. ProSeCo-OWT
   co-training produces a non-trivial, consistently beneficial corrector.

2. **Scheduling is valuable:** oracle G = +2.281 vs uniform G = +0.688 at B=8
   (3.3× ratio). The upper bound on scheduling gain is large; the signal question
   is whether we can exploit it.

3. **Entropy-minimum principle confirmed:** entropy_bot_B achieves G=+0.887 vs
   uniform +0.688 (+29% at B=8). The mechanism is the non-monotone entropy
   profile: entropy over committed tokens is minimised at t≈35–45, which
   coincides with the peak-Δ_t hump. Selecting minimum-entropy steps = selecting
   from the large, stable committed set = highest corrector gain.

*Audit note 2026-04-19:* Items 1 and 2 are derived from Protocol A trajectory-level
quantities and survive the audit (Tier T1). Item 3's "+29% at B=8" claim is
Tier T4 and **inadmissible**: the per-policy single-seed pipeline G(S) for
entropy_bot_B at B=8 is ≈ 0.08 (a 10× *loss* vs uniform 0.76) — the +29% comes
from the additive surrogate A(S), not the pipeline G(S). The entropy-minimum
principle is downgraded to a hypothesis to be tested in Phase 2b.

See `docs/thesis/experiments/PROSECO_ANALYSIS.md` for full analysis, Δ_t profile,
entropy non-monotone mechanism, and §7.2 evidence table.

---

## 11. Phase 2 — paired K-seed sweep (COMPLETE 2026-04-19)

### 11.1 Status (2026-04-19, final)

| Aspect | Detail |
|---|---|
| Job | `479581` on `gnode02` (`hpc/phase2b_proseco_owt.sbatch`) |
| Started | 2026-04-19 08:50 |
| Wall-time limit | 20 h |
| Estimated completion | ≈ 19:00–19:15 (≈80 min remaining at 17:42 snapshot; shard 1 last 2 seeds; shards 0/2/3 already complete) |
| Backend | ProSeCo-OWT (co-trained checkpoint, `~/mdm/checkpoints/proseco_owt`) |
| T, K | 64, 30 |
| B sweep | {2, 3, 4, 8, 16} |
| Policies | uniform + entropy/margin/quality {top, bot} + middle (paired with seed-matched uniform) |
| Monte-Carlo oracle | mc_B ∈ {2, 3, 4}, P=100 random schedules per (seed, B) for an honest lower bound on G(S_B*) |
| Sharding | 4 GPUs × 8 seeds each (one shard, see job log) |
| Output | `results/phase2b_proseco_owt/{shard*}/{seed*}.json` |
| Aggregator | `scripts/analyze_phase2b.py` (writes `aggregated.json`, paired CIs) |

### 11.2 Per-shard progress (latest log snapshot)

| Shard | Seeds done | Per-seed wall (s) |
|---|---|---|
| 0 | **8 / 8 ✅ complete** | ≈ 3580 |
| 1 | 6 / 8 + seed 67 in flight (~71 min in) | ≈ 4581 |
| 2 | **7 / 7 ✅ complete** | ≈ 3517 |
| 3 | **7 / 7 ✅ complete** | ≈ 3510 |

Shard 1 is the slowest GPU; it sets the wall-clock floor. No shard has crashed,
no NaN/skipped seeds reported, no checkpoint reload between seeds.

### 11.3 What Phase 2b decides

Each (policy, B) cell gets a **paired** estimate Δ̂_K(policy, B) :=
G(policy) − G(uniform) computed on identical seeds, with BCa bootstrap 95% CI
(`ANALYSIS_SPEC.md` §3). Decision rule:

- **PASS**: BCa CI for Δ̂ is strictly above 0 → admissible "policy beats
  uniform on this B" claim (Tier T2).
- **NULL**: BCa CI for Δ̂ contains 0 → no signal at K=30; reported as
  null result.
- **FAIL**: BCa CI strictly below 0 → policy is harmful; useful as a
  negative finding.

The MC oracle lower bound at B ∈ {2, 3, 4} provides an honest replacement
for the mean-field "oracle" used in Phase 1.

### 11.4 Phase 2c gate

If any (policy, B) Δ̂ has |centre| within 2 × σ_F (where σ_F = pooled
seed-to-seed standard deviation of G under uniform), schedule a Phase 2c
MAUVE F-swap on that (policy, B) subset. ANALYSIS_SPEC §7 covers the
F-swap protocol.

---

## 12. Phase 2b verdict (2026-04-20, locked)

Job 479581 completed at 10:09:46 (HPC clock); 30 paired seeds × 5 B-values ×
9 policies + MC oracle (best-of-100 random schedules per seed at B ∈ {2,3,4})
aggregated by `scripts/analyze_phase2b.py`.

### 12.1 Cell-level outcome

| Outcome | Count | Notes |
|---|---:|---|
| **PASS** (paired CI > 0) | 5 | `middle@B=2` (+0.089), `entropy_bot_B_pt@B=2` (+0.105), `mean_delta_oracle@B=2` (+0.130), `@B=3` (+0.103), `@B=4` (+0.092) |
| **FAIL** (paired CI < 0) | 30 | All "top" signal policies systematically harmful; Cohen's d down to −2.04 for `back@B=16` |
| **BORDERLINE** (\|Δ̂\| ≤ 2σ_F) | 9 | `middle@{3,4}`, `entropy_bot@{3,4}`, `mean_delta_oracle@{8,16}`, others |

### 12.2 MC oracle bound

| B | MC oracle Δ̂ vs uniform | CI |
|---|---:|---|
| 2 | +0.45 | tight, ±0.07 |
| 3 | +0.45 | tight, ±0.07 |
| 4 | +0.45 | tight, ±0.07 |

Mean-profile (`mean_delta_oracle`) captures only +0.09 of the +0.45 MC headroom;
the remaining **+0.36 is per-instance**.

### 12.3 Rank-correlation decay

Pooled Spearman ρ(A, G) declines monotonically with B:

| B | 2 | 3 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| ρ | 0.66 | 0.62 | 0.50 | 0.44 | 0.45 |

Empirically validates Refinement A″'s prediction that ε_R = (1−|ρ|)·σ_Δ grows
with B; greedy single-step ranking saturates fast.

### 12.4 Phase 2c gating

**Closed without execution.** Cohen's d up to ±2 in the FAIL cells leaves cell
ordering robust to F-metric choice. F-swap can be reopened only if Phase 3a
forces a defence of the small-B win. Documented: `PHASE3_DIRECTION_AUDIT.md`
§A1.4.

---

## 13. Phase 3 direction audit + verdict (2026-04-20)

After Phase 2b, the post-pilot synthesis flagged PRISM (learned per-token
quality signal) as the natural next move. A formal audit
(`PHASE3_DIRECTION_AUDIT.md`) **rejects PRISM** based on three independent
smoking guns from Phase 2b:

1. **`mean_delta_oracle` ceiling.** Even the ground-truth-Δ_t schedule fails
   at B ≥ 8. PRISM, being a single-step learned ranker, is upper-bounded by
   this oracle.
2. **Top-10 MC ∩ oracle Jaccard ≈ 1.5× random** at B ∈ {2,3,4}. The +0.45
   MC headroom does not sit on the high-Δ steps — a ranker that predicts
   Δ-correlated quality cannot recover it.
3. **Top-10 MC internal Jaccard ≈ bottom-10 internal Jaccard.** The "best"
   schedules per instance are no more internally consistent than the worst,
   the empirical signature of an *unstructured combinatorial gain* — by
   construction not addressable by any per-step score.

Variance decomposition closes the case: at B = 4, 62 % of total G-variance
across MC schedules is *within-seed* (schedule choice matters), but that
variance is not captured by any per-step ranking, including the cheating
ground-truth one. The bottleneck is **combinatorial, not informational**.

### 13.1 Phase 3 plan (set 2026-04-20)

Two parallel tracks; full spec in `PHASE3_ALTERNATIVE_PLAN.md`.

- **Phase 3a (HPC, ≈ 1–2 days).** Combinatorial scheduling baselines —
  coordinate descent (cheap-A surrogate for swap acceptance, true-G
  re-evaluation) and beam search with rollouts. K = 30 seeds × B ∈
  {2,3,4,8}. Outputs: `results/phase3a/{coord_descent_paired,
  beam_search_paired,oracle_gap_closure}.json`,
  `figures/phase3a/oracle_gap_closure.{png,pdf}`. Decision rules in
  `PHASE3_ALTERNATIVE_PLAN.md` §"Phase 3a / Decision rules".
  **STATUS:** complete 2026-04-20 (job 479941). See §14.
- **Phase 3b (notebook, ≈ 2–3 days).** Theory finalisation — formal proofs of
  Refinements A′ (variance-form, σ_ξ from Phase 2b MC residuals) and A″
  (rank-based ε_R, ρ-decay from §12.3); Negative-Result Corollary
  **scoped to the greedy/separable-ranker class** (Phase 3a shows search
  procedures exceed the ranker-class envelope) keyed to the §12 top-MC
  internal-Jaccard observation.

### 13.2 What's parked

- PRISM (full rationale: `PHASE3_DIRECTION_AUDIT.md` §A1–A4).
- MDLM full Phase 2b replication. Tier-2 micro-replication on `middle@B=2`
  is conditional on Phase 3a producing a positive result worth validating.

### 13.3 Provenance

- Raw: `results/phase2b_proseco_owt/{policy_raw,mc_raw}.json`
- Aggregated: `results/phase2b/{policy_comparison_paired,mc_oracle,
  rank_A_vs_G_phase2b}.json`
- Figures: `results/phase2b/figures/`
- Audit: `docs/thesis/experiments/PHASE3_DIRECTION_AUDIT.md`
- Plan: `docs/thesis/experiments/PHASE3_ALTERNATIVE_PLAN.md`
- Theory ledger: `docs/thesis/theory/THEORY_STATUS.md` (Honesty Ledger entry
  2026-04-20)
- Canonical scope: `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md` (Phase
  3 sections updated 2026-04-20)

---

## 14. Phase 3a verdict — combinatorial baselines (2026-04-20, locked)

Job `479941` on `gnode02` (`hpc/phase3a_combinatorial.sbatch`). 4 shards × 4
A100s, K = 30 paired seeds × B ∈ {2, 3, 4, 8}, two methods per cell:
**CD-G** (coordinate descent, ≤ 65 true-G calls per (seed, B)) and **BS-AG**
(beam search width 8, cheap-A ranking on extension candidates with true-G
rollouts on top 8). Wall-clock 7h59m. Aggregator:
`scripts/analyze_phase3a.py`. Full report:
`docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md`.

### 14.1 Headline result

Both methods PASS at every budget tested. CD-G recovers 74 – 84 % of the
Phase 2b MC-oracle paired headroom at B ∈ {2, 3, 4}; BS-AG recovers 49 – 64 %.
At B = 8 the MC oracle was not run, but the paired-vs-uniform result is
strongly positive for both methods — the budget at which `mean_delta_oracle`
fell into the NULL band (§13 smoking gun #1).

| Method | B = 2 | B = 3 | B = 4 | B = 8 |
|---|---|---|---|---|
| **CD-G Δ** (BCa CI) | **+0.356** [+0.284, +0.432] PASS | **+0.327** [+0.240, +0.421] PASS | **+0.380** [+0.285, +0.479] PASS | **+0.322** [+0.215, +0.427] PASS |
| **BS-AG Δ** (BCa CI) | **+0.289** [+0.209, +0.374] PASS | **+0.252** [+0.158, +0.352] PASS | **+0.220** [+0.113, +0.331] PASS | **+0.153** [+0.035, +0.263] PASS |
| Δ_oracle (Phase 2b MC) | +0.451 | +0.441 | +0.450 | n/a (oracle not run at B=8) |
| CD-G / oracle ratio | 78.9 % | 74.1 % | 84.3 % | n/a |
| BS-AG / oracle ratio | 64.1 % | 57.1 % | 48.8 % | n/a |

### 14.2 Interpretation

The +0.45 MC-oracle headroom is **not** a sampling artefact and is **not**
"irreducibly unstructured": it has structure that combinatorial search can
reach. The Phase 2b verdict ("the gain is combinatorial, not signal-driven
at single-step granularity") is upheld; what changes is the scope of the
Negative-Result Corollary — it bounds the *greedy/separable-ranker class*,
not "informed scheduling in general". Search procedures over schedules — even
ones that use the cheap-A surrogate to prune candidates — recover most of
the oracle headroom.

### 14.3 Caveats (load-bearing for write-up)

- **CD-G uses true-G feedback and is not deployable.** Existence/structural
  result only. ≈ 65 × generation-cost per cell.
- **BS-AG is closer to practical** (cheap-A ranking + O(B) true-G rollouts)
  but still requires a true-G evaluator.
- **No external validity.** ProSeCo-OWT only; cross-backbone (MDLM, ReMDM)
  not tested.
- **No theoretical guarantee on closure ratio.** 49 – 84 % is empirical
  description, not theorem.

### 14.4 Provenance

| Artefact | Path |
|---|---|
| Raw CD per-(seed, B) rows | `results/phase3a_proseco_owt/cd_raw.json` |
| Raw BS per-(seed, B) rows | `results/phase3a_proseco_owt/bs_raw.json` |
| Paired CIs | `results/phase3a_proseco_owt/{cd,bs}_paired.json` |
| Oracle-gap closure | `results/phase3a_proseco_owt/oracle_gap_closure.json` |
| Per-seed JSONs | `results/phase3a_proseco_owt/per_seed/{cd,bs}_rows_seed{seed}.json` |
| Sbatch | `hpc/phase3a_combinatorial.sbatch` |
| Job logs | `out/phase3a_combinatorial_479941{,_shard0..3}.log` |
| Full report | `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md` |

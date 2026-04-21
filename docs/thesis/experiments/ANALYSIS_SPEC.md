# Analysis Output Spec — Phase 2

**Author:** Claude Code (co-advisor review, Workstream D)
**Date:** 2026-04-19
**Status:** Draft — spec against which the Stage 2a and 2b analysis scripts
will be implemented.
**Companion:** `NEXT_PHASE_EXPERIMENT_PLAN.md` (what to run; this file
specifies what the analysis produces).

---

## 0. Purpose

Phase 1's analysis outputs were ad-hoc: every script wrote its own JSON layout,
and the thesis text read numbers directly out of them without consistent
uncertainty quantification. Phase 2 will not make this mistake.

This spec defines:

1. The **canonical schema** every analysis JSON must follow.
2. The **statistical methods** used (paired bootstrap, Spearman CI, effect
   size).
3. The **minimum set of plots/tables** required.
4. The **confidence-reporting rules** — when a Phase 2 number is thesis-quotable
   vs preliminary.

Every analysis script written after this spec's approval must conform, and
every thesis claim that quotes a Phase 2 number must cite the schema field
(`file.json#path`) it came from.

---

## 1. JSON schema rules (all Phase 2 outputs)

### 1.1 Top-level structure

Every Phase 2 `results/**/*.json` must include:

```json
{
  "meta": {
    "phase": "2a" | "2b" | "2c",
    "stage_name": "string",
    "generated_at_utc": "ISO-8601",
    "generated_by": "script path + git hash",
    "source_results_paths": ["list of input result dirs"],
    "config": {
      "T": 64,
      "K": 30,
      "B_values": [2, 3, 4, 8, 16],
      "seed_base": 42,
      "checkpoint_hash": "sha256 first-12-hex"
    }
  },
  "data": { ... stage-specific ... },
  "decisions": [
    {
      "gate_id": "string matching NEXT_PHASE_EXPERIMENT_PLAN.md gate",
      "result": "pass" | "fail" | "inconclusive",
      "evidence": "short sentence pointing at a data field"
    }
  ]
}
```

Rationale: `meta` makes every output self-describing; `decisions` makes gate
outcomes machine-readable rather than buried in a narrative.

### 1.2 Units and sign conventions

- **G** and **A** are always `F(y) − F(y_base)`, sign-positive means quality
  increased. F is always specified in `meta.config.F` ∈ {"neg_nll", "mauve_owt"}.
- **ε** (calibration error), **η_B** (additivity slack), **γ** (pairwise
  interaction bound) are always non-negative scalars.
- **ρ** (Spearman) is in [−1, 1] with sign preserved. An "inverted" signal is
  reported as negative ρ, not absolute.
- **SE** on a mean is reported as `{mean: float, se: float, n: int}`. Never
  report a mean without n.

### 1.3 Missing data and NaN

- Never emit `NaN` or `null` for a measured quantity — emit the full
  `{mean, se, n, note: "why missing"}` instead.
- If an evaluation failed (OOM, checkpoint mismatch, etc.), record it under
  `data.failed_evaluations[]` with the seed and reason.

---

## 2. Stage 2a outputs

Four JSON files under `archive/legacy_framework/results/phase2a/`, all conforming to §1.1.

### 2a.1 `rank_A_vs_G.json`

Question: does A(S) rank schedules in the same order as G(S)?

Schema (`data`):

```json
{
  "data": {
    "spearman_by_B": {
      "4":  {"mean": 0.42, "se": 0.11, "n_schedules": 30, "bootstrap_95_ci": [0.21, 0.60]},
      "8":  {"mean": 0.38, "se": 0.10, "n_schedules": 30, "bootstrap_95_ci": [0.17, 0.56]},
      "16": {"mean": 0.29, "se": 0.09, "n_schedules": 30, "bootstrap_95_ci": [0.10, 0.48]}
    },
    "pearson_by_B": { ... same structure ... },
    "scatter_points_by_B": {
      "4": [{"S_id": int, "A": float, "G": float}, ...]
    }
  }
}
```

Statistical method: Spearman rank correlation with 1000-sample bootstrap CI
(resample the 30 schedule records with replacement, recompute Spearman, take
2.5/97.5 quantiles).

Decision gate (mirrors NEXT_PHASE_EXPERIMENT_PLAN §2a.1):
- `gate_id: "A_ranks_G_at_B8"`, pass iff CI lower bound ≥ 0.5.

### 2a.2 `eps_R.json`

Question: does a rank-based ε behave differently from RMS ε?

Schema:

```json
{
  "data": {
    "per_trajectory_spearman": {
      "entropy":            [[i, ρ_i], ... × 50],
      "inverse_margin":     [[...], ...],
      "quality_mass_proxy": [[...], ...],
      "unmasked_fraction":  [[...], ...]
    },
    "spearman_quantiles": {
      "entropy": {"q25": -0.38, "q50": -0.21, "q75": -0.02, "pos_count": 11, "neg_count": 39},
      ...
    },
    "epsilon_R_by_signal": {
      "entropy":        {"mean": 0.11, "se": 0.014, "n": 50, "definition": "(1-|ρ|)·σ_Δ per trajectory"},
      ...
    },
    "epsilon_rms_by_signal": {
      "entropy":        {"mean": 0.134, "se": 0.006, "n": 50, "definition": "RMS residual of Δ-vs-ψ̂"},
      ...
    }
  }
}
```

Decision gate: `gate_id: "eps_R_differentiates_signals"`, pass iff
(ε_R across signals) shows ≥ 1.5× spread while ε_rms stays within 1.1×. (I.e.,
ε_R is a sharper discriminator.)

### 2a.3 `mc_oracle_lowerbound.json`

Question: over the 30 existing sampled schedules per B, how often does any of
them beat uniform, and by how much?

Schema:

```json
{
  "data": {
    "max_over_30_minus_uniform": {
      "4":  {"mean": 0.08, "se": 0.04, "n_trajectories": 50, "paired_t_stat": 2.1, "p_value": 0.04},
      "8":  {"mean": -0.11, "se": 0.06, "n_trajectories": 50, "paired_t_stat": -1.8, "p_value": 0.07},
      "16": {...}
    },
    "fraction_beating_uniform": {
      "4": 0.66, "8": 0.42, "16": 0.30
    }
  }
}
```

Method: for each trajectory, from the 30 random schedules at that B, take
max_j G(S_j). Subtract this trajectory's uniform-G (already in
`summary.json.policy_comparison.B{B}.uniform.G`, but recompute per-trajectory
if available; otherwise use the single-seed value and flag the limitation).
Paired bootstrap 1000×.

Decision gate: `gate_id: "headroom_exists_B8"`, pass iff CI lower bound ≥ 0.05
at B=8.

### 2a.4 `phase2a_corrected_policy_table.md`

Not JSON — a markdown table with three columns per policy at each B:
`A(S)`, `G(S) single-seed`, `G(S) min-of-30-schedules-containing-matching-steps`
(the last column: when a random schedule happens to match the policy's selection,
its G).

Purpose: produce a correct version of the table currently shown in
`RESULTS_STATUS.md`, with explicit columns for A and G, for inclusion as a
replacement table in Workstream H.

---

## 3. Stage 2b outputs

Three JSON files under `results/phase2b_proseco_owt/`.

### 3b.1 `policy_comparison_paired.json`

This is the core deliverable of Phase 2.

Schema:

```json
{
  "data": {
    "policies": ["uniform", "front", "back", "middle", "entropy_top_B_pt", ...],
    "B_values": [2, 3, 4, 8, 16],
    "per_policy_per_B": {
      "uniform": {
        "2":  {"G_per_seed": [g_1, ..., g_30], "mean": ..., "se": ..., "n": 30},
        ...
      },
      "entropy_top_B_pt": {
        "2":  {
          "G_per_seed": [...], "mean": ..., "se": ..., "n": 30,
          "paired_vs_uniform": {
            "diff_per_seed": [d_1, ..., d_30],
            "mean": ..., "se": ..., "paired_t_stat": ..., "p_value": ...,
            "bootstrap_95_ci": [lo, hi],
            "cohens_d": ...
          }
        },
        ...
      }
    },
    "ranking_at_each_B": {
      "8": [
        {"policy": "uniform", "G_mean": ..., "rank": 1},
        {"policy": "mc_oracle_B", "G_mean": ..., "rank": 2, "note": "only for B in {2,3,4}"},
        ...
      ]
    }
  }
}
```

Statistical tests:
- **Paired t-test:** policy G − uniform G at each seed, under the null of
  zero difference.
- **Paired bootstrap CI:** 1000 resamples of 30 seed-paired differences.
- **Cohen's d:** mean(diff) / std(diff) — effect size, independent of n.
- **Bonferroni correction** across `(policies, B_values)` pairs: α=0.05 →
  α_corrected = 0.05 / (n_policies × n_B) ≈ 0.005.

Decision gates:
- `gate_id: "signal_beats_uniform_B8"`: pass iff any per-trajectory signal
  policy has `paired_vs_uniform.bootstrap_95_ci[0] > 0` at B=8.
- `gate_id: "uniform_is_optimal"`: pass iff no policy's bootstrap CI lower
  bound exceeds uniform's SE at any B, and MC oracle at B=4 has
  `(MC-oracle − uniform)` CI overlapping zero.
- `gate_id: "small_B_non_vacuous"`: pass iff at B=2 or B=3 any signal policy
  beats uniform with Cohen's d ≥ 0.5.

### 3b.2 `mc_oracle.json`

Schema:

```json
{
  "data": {
    "per_trajectory": [
      {
        "seed": 42000,
        "B_to_argmax_G": {
          "2": {"G": ..., "schedule": [t1, t2], "n_samples": 300, "A_of_argmax": ...},
          "3": {...},
          "4": {...}
        }
      }, ... × 30
    ],
    "mc_oracle_minus_uniform": {
      "2":  {"mean": ..., "se": ..., "n": 30, "paired_t_stat": ..., "bootstrap_95_ci": [lo, hi]},
      "3":  {...},
      "4":  {...}
    },
    "mc_oracle_minus_mean_profile_oracle": {
      "2": ..., "3": ..., "4": ...
    },
    "enumeration_spot_checks": [
      {"seed": 42000, "B": 4, "mc_max_G": ..., "exact_max_G": ..., "ratio": ...},
      ... × 3 trajectories
    ]
  }
}
```

Purpose: give an empirical lower bound on G(S_B*) and a headroom estimate
vs uniform. The enumeration spot checks validate that 300 MC samples is
close enough to exact oracle on a handful of trajectories at B=4.

Decision gate: `gate_id: "headroom_exists_realistic"`, pass iff
`mc_oracle_minus_uniform.bootstrap_95_ci[0] > 0.05` at B=4.

### 3b.3 `rank_A_vs_G_phase2b.json`

Reproduces 2a.1 on the new K=30-seed data (not on the 30 random schedules,
but on the per-policy records — one A(S) and one G(S) per policy per seed per
B). This tests whether the policy A-rankings match the policy G-rankings
across seeds.

Same schema as 2a.1.

Decision gate: `gate_id: "A_ranks_G_cross_policy_B8"`, pass iff mean Spearman
across seeds ≥ 0.6.

---

## 4. Stage 2c outputs

If 2c fires (F swap to MAUVE), produce:

### 4c.1 `policy_comparison_paired_mauve.json`

Same schema as 3b.1 but with `meta.config.F = "mauve_owt"` and the data
keyed the same way. Direct structural copy so that 2b and 2c numbers are
side-by-side comparable.

### 4c.2 `f_sensitivity.json`

Cross-tabulation: for each (policy, B), report `(G_NLL, G_mauve)` pair and
the correlation across policies within each B. Tests whether the policy
ranking is F-dependent.

Schema:

```json
{
  "data": {
    "per_policy_per_B": {
      "uniform": {
        "4":  {"G_nll": ..., "G_mauve": ..., "diff": ...},
        ...
      }, ...
    },
    "cross_F_ranking_correlation_per_B": {
      "4":  {"spearman_rho": 0.85, "bootstrap_95_ci": [0.72, 0.94]},
      ...
    }
  }
}
```

Decision gate: `gate_id: "F_robust_ranking"`, pass iff all B have
`cross_F_ranking_correlation_per_B[B].bootstrap_95_ci[0] > 0.5`.

---

## 5. Plots (`figures/phase2{a,b,c}/`)

All plots must:

- Be generated by `scripts/analyze_phase2{a,b,c}.py` as part of the analysis
  step (no hand-editing).
- Save as both `.png` (≥ 300 DPI) and `.pdf` for thesis inclusion.
- Have a separate JSON sidecar (`.meta.json`) recording the data fields they
  plot, so the thesis text can cite the plot + the numbers in one pass.

### 5.1 Required plots per stage

Stage 2a:
- `A_vs_G_scatter_B{4,8,16}.{png,pdf}` — one dot per schedule, x=A(S),
  y=G(S), with identity line and Spearman/Pearson CI annotated.
- `per_trajectory_spearman_histogram.{png,pdf}` — histograms of
  per-trajectory ρ for each signal, with the sign-count annotated.

Stage 2b:
- `policy_gains_by_B.{png,pdf}` — a 5-panel box-whisker plot (one panel per B)
  with paired bootstrap 95% CIs overlaid on each policy. Uniform highlighted
  as a reference bar.
- `paired_diff_uniform_B{2,3,4,8,16}.{png,pdf}` — per-seed paired difference
  plots for the four best-performing non-uniform policies.
- `mc_oracle_headroom.{png,pdf}` — at B ∈ {2,3,4}, trajectory-by-trajectory
  MC-oracle G vs uniform G, connected by lines.

Stage 2c (if fires):
- `f_nll_vs_f_mauve_ranking.{png,pdf}` — scatter of (rank_NLL, rank_MAUVE)
  across policies at each B.

### 5.2 What plots must not do

- Must not collapse per-seed variance into a single bar with no error bar.
- Must not use a bar chart for G comparisons; use box-whisker or violin to
  show the full distribution across seeds.
- Must not mix A(S) and G(S) on the same axis (learned from Phase 1 — the
  previous "policy table" blended both units).

---

## 6. Confidence-reporting rules

Every number that appears in a thesis claim must be classified into one of
four tiers, defined as follows:

| Tier | Criterion | Thesis usage |
|------|-----------|--------------|
| **T1 — confirmed** | n ≥ 30, paired bootstrap CI excludes zero under Bonferroni-corrected α=0.005 | Thesis-grade. Quotable in abstract and main results. |
| **T2 — suggestive** | n ≥ 30, CI excludes zero at nominal α=0.05 but not after Bonferroni | Appendix or discussion. Qualified as "preliminary evidence". |
| **T3 — exploratory** | Spearman or sign trend only; no paired test | Appendix only. Explicitly called "hypothesis-generating". |
| **T4 — inadmissible** | Single seed, or ε_rms-style uninformative metric flagged per THEORY_STRESS_TEST §4 | **Must not appear** in the thesis. |

**Every Phase 2 JSON must label each headline number with its tier.** Add
`"tier": "T1"` etc. as a sibling field to `mean`, `se`.

Audit rule: if a claim in `CANONICAL_RESEARCH_DIRECTION.md` or
`RESULTS_STATUS.md` references a number with no tier label, or a T3/T4 tier,
the claim must be softened or removed.

---

## 7. Reproducibility manifest

Every results directory must contain `manifest.json` with:

```json
{
  "git_hash": "...",
  "git_dirty": false,
  "python_version": "3.11.x",
  "torch_version": "2.x",
  "cuda_version": "...",
  "checkpoint_sha256": "...",
  "invocation_command": "python scripts/run_phase2b_proseco_owt.py ...",
  "hpc_job_id": "...",
  "hostname": "...",
  "walltime_seconds": ...,
  "max_gpu_memory_mb": ...
}
```

This is emitted by `scripts/run_phase2b_proseco_owt.py` at startup and
finalized at completion.

---

## 8. Analysis script contract

Every analysis script (`scripts/analyze_phase2{a,b,c}.py`) must:

1. Accept `--results_dir` and `--out_dir`.
2. Emit every JSON matching §§2–4 schemas, every plot matching §5.1.
3. At exit, print a line matching `^ANALYSIS_COMPLETE: gates=N pass=M fail=K
   inconclusive=L$` so orchestration can parse outcomes.
4. Exit 0 iff all JSON writes succeeded; non-zero if any decision gate could
   not be evaluated (e.g., missing input).

The pattern `ANALYSIS_COMPLETE: ...` is the machine-readable summary; it must
include the per-gate pass/fail summary that any downstream CI or thesis
regeneration uses.

---

## 9. Connection to Workstream H (narrative update)

Every paragraph in `docs/thesis/CURRENT_INDEX.md` and
`docs/thesis/experiments/RESULTS_STATUS.md` that cites a Phase 2 number must:

1. End with a citation like `[phase2b_proseco_owt/policy_comparison_paired.json
   #data.per_policy_per_B.uniform.8.mean]`.
2. Include the tier label in parentheses: "(T1)" or "(T2)" etc.
3. Report `mean ± se` or `mean (95% CI: [lo, hi])`, never `mean` alone.

Example (correct):

> At B=8, per-trajectory entropy_bot_B achieves a paired surplus of
> 0.12 ± 0.04 (T1) over uniform on true G(S) [phase2b_proseco_owt/
> policy_comparison_paired.json#data.per_policy_per_B.entropy_bot_B_pt.8.paired_vs_uniform.mean].

Example (incorrect):

> At B=8, entropy_bot_B beats uniform by +29%.

The former is auditable; the latter is the Phase 1 failure mode this spec
exists to prevent.

---

## 10. Implementation checklist

- [ ] `archive/legacy_framework/scripts/analyze_phase2a.py` — archived analysis
  helper; consumes `results/phase1_proseco_owt_full/*` and emits all files in §2.
- [ ] `scripts/analyze_phase2b.py` — new; consumes
  `results/phase2b_proseco_owt/*` and emits all files in §3.
- [ ] `scripts/analyze_phase2c.py` — new; emits §4 (only if 2c fires).
- [ ] `src/mdm_playground/analysis/stats.py` — new module with:
  - `paired_bootstrap_ci(x, y, n_resamples=1000, alpha=0.05)`
  - `spearman_bootstrap_ci(x, y, n_resamples=1000, alpha=0.05)`
  - `cohen_d(diff)` (uses stddev of paired differences)
  - `classify_tier(mean, se, n, alpha_bonf)` returning "T1|T2|T3|T4"
- [ ] `src/mdm_playground/analysis/plots.py` — new module with the plot
  functions referenced in §5.1.
- [ ] Tests (pytest): bootstrap CI coverage on synthetic data;
  paired-t vs scipy reference; tier classification edge cases.

Estimated LoC: ~800 lines across the four new files, ~300 lines of tests.

---

## 11. What this spec does not cover

- **Theorem-specific diagnostics** (e.g., `2Bε + 2η_B vs G(S_B*)` bound
  check). Those belong in a separate `theory_diagnostics.json` emitted by a
  theory-side script, not in the Phase 2 policy analysis. Will spec separately
  after Phase 2 data lands.
- **Phase 3 (cross-backbone)**: out of scope — addressed in its own plan after
  Phase 2.

---

## 12. Pre-approval for this spec

Before implementing §10's checklist I need user confirmation on:

1. Are the four tiers (T1/T2/T3/T4) acceptable? In particular, is the
   Bonferroni-corrected α=0.005 too strict for T1?
2. Is paired bootstrap the preferred uncertainty method, or do you prefer
   Bayesian (posterior on policy difference)?
3. Is a single `manifest.json` per results dir sufficient for reproducibility,
   or do we also want per-JSON provenance (checksum of inputs)?
4. Do we actually need MAUVE (stage 2c) as a gated stage, or would a held-out
   LM-judge be preferred?

Without answers, I will default to: tiers as specified; paired bootstrap +
paired t cross-validation; single manifest; MAUVE as the stretch F (since the
infrastructure already exists per CLAUDE.md §11).

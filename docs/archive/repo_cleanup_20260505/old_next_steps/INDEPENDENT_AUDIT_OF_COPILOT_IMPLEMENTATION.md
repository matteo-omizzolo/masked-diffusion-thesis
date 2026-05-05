> ARCHIVED — historical only.
> Do not use this file to infer the current thesis status unless explicitly asked.
> Current source of truth: see `START_HERE.md` and `docs/README.md`.

---


# Phase 2 — Independent Audit of Copilot's Combinatorial Diagnostics

*Date: 2026-04-22. Auditor: Claude (fresh read). Scope: correctness,
reproducibility, and evidence quality of the combinatorial-diagnostics
module written by Copilot and the resulting output JSON.*

---

## 1. Files under audit

| Role | Path |
|---|---|
| Diagnostics module | `src/mdm_playground/analysis/combinatorial_diagnostics.py` |
| Analysis CLI | `scripts/analyze_combinatorial_diagnostics.py` |
| Unit tests | `tests/test_combinatorial_diagnostics.py` |
| Module re-export | `src/mdm_playground/analysis/__init__.py` |
| Output artifact | `results/phase2b/combinatorial_diagnostics.json` |

---

## 2. Correctness review

### 2.1 `schedule_jaccard`
- Defines Jaccard on two sets of integer time-indices. Returns 1.0 on two
  empty sets (sensible edge case). **Correct**.

### 2.2 `mean_pairwise_jaccard`
- Uses `itertools.combinations` → one pair per unordered (i,j); divides by
  `np.mean` → matches standard "mean pairwise". **Correct**.
- Returns 0.0 on a single schedule. Edge-case handling OK.

### 2.3 `random_jaccard_baseline`
- Monte Carlo sampling of two size-B subsets of [0, T) without replacement;
  computes sample mean Jaccard. Default samples = 5 000, which is enough for
  a stable 3-digit estimate (SE ≈ σ/√5000 ≈ 0.004 for mean Jaccard around
  0.03). **Correct**.
- **Minor concern**: the deterministic seed is `seed + B` inside
  `overlap_diagnostics`. This is fine for reproducibility; it also means the
  random baselines at different B values use different seeds, so comparisons
  across B are statistically independent (good).

### 2.4 `overlap_diagnostics`
- Groups MC rows by `(seed, B)`, takes top-k and bottom-k by G, computes
  per-seed mean Jaccard vs that seed's `mean_delta_oracle` schedule (from
  `policy_raw.json`), then averages across seeds. **Correct**.
- `T` is inferred as `max(step) + 1` across all MC rows at that B. With T=64
  and 9 000 MC rows that max will reliably be 63, so this is correct in
  practice but **fragile**: if a future run happened to never sample the last
  step, T would be under-estimated. **Low risk; recommend accepting the
  existing value for now**, document the limitation.
- `ratio_topk_vs_random = mean_topk_vs_oracle_jaccard / random_baseline`.
  This is the headline number. Interpretation: "how much do top-k schedules
  structurally agree with the mean-delta oracle relative to chance?"
- The analogous bottom-k ratio is NOT reported, only `mean_bottomk_internal_jaccard`.
  Reporting a bottom-k-vs-oracle ratio would be cheap and instructive; flag
  as a minor gap, not a blocker.

### 2.5 `variance_decomposition`
- Computes per-B:
    - `var_total_G` over all (seed, MC) points, ddof=1,
    - `var_between_seed_means` over seed means, ddof=1,
    - `var_within_seed` as the **mean** of per-seed sample variances.
- Reports shares `share_within = within_var / total_var` and analogous `share_between`.
- **Caveat** (not an error): because `total_var = within + between + cross_terms`
  is only exact for the *law-of-total-variance* decomposition
  (`var_total = E[var_within] + var(E[within])`), and the code uses sample
  ddof=1 variances across different-sized samples, `share_within + share_between`
  will not in general equal 1.0 — and indeed the JSON shows the two shares
  sum to ≈1.02–1.03 at each B. **The shares are informative to 2 digits** but
  should not be read as a strict ANOVA decomposition. Recommendation: add a
  note to the downstream doc that these are descriptive shares, not an ANOVA.

### 2.6 `build_combinatorial_diagnostics`
- Just composes 2.4 + 2.5. **Correct**.

### 2.7 Analysis CLI
- Reads `mc_raw.json` + `policy_raw.json`; writes the output JSON. Exit
  codes, input validation (list + non-empty), and deterministic seeding are
  all correct. **Correct**.

### 2.8 Unit tests
- `schedule_jaccard` basic cases: trivial, disjoint, 1/3 overlap. **Correct**.
- `mean_pairwise_jaccard`: 3-schedule case hand-checked to 5/9. **Correct**.
- `variance_decomposition`: asserts nonzero components but does not assert a
  specific number. **Weak**; acceptable as smoke.
- `build_combinatorial_diagnostics` smoke: runs end-to-end with 6 MC rows +
  2 oracle rows; asserts shape of output. **Correct**.
- **Coverage gap**: no test asserts the specific `ratio_topk_vs_random`
  computation on a hand-computed example. Not a blocker, but a small hand
  case would increase confidence.

---

## 3. Output audit: `combinatorial_diagnostics.json`

### 3.1 Headline numbers (from the file)

| B | mean_topk_vs_oracle_jaccard | random_baseline | ratio_topk_vs_random | within_share | between_share |
|---|---|---|---|---|---|
| 2 | 0.0267 | 0.0222 | **1.20** | 0.691 | 0.327 |
| 3 | 0.0320 | 0.0271 | **1.18** | 0.645 | 0.374 |
| 4 | 0.0470 | 0.0361 | **1.30** | 0.619 | 0.400 |

### 3.2 Discrepancy with earlier canonical-direction text (R1 from Phase 1)

Some direction draft text refers to a **≈1.5×** Jaccard signature for top-k
schedules vs oracle. The measured artefact on the real Phase 2b raw data
shows **1.20, 1.18, 1.30** — weaker.

**Interpretation**:
- Direction: **top-k schedules are not close copies of the mean-delta oracle**.
  They are only ≈20–30 % more overlapping with the oracle than chance would
  predict. The best schedules do not factor through the mean-delta signal.
- This is **fully consistent with the Phase 3a finding** that the recoverable
  structure is not captured by a separable per-step score (the oracle IS a
  separable per-step score). So the 1.2× observation strengthens, not
  weakens, the ranker-class negative result.
- **However, any thesis claim of "≈1.5×"** is not supported by this file
  and should be corrected. Hand-off to Phase 7 doc-update triage.

### 3.3 Variance-decomposition numbers

Within-share sits at 0.69 / 0.64 / 0.62 for B = 2 / 3 / 4. The widely-cited
"62 % within-seed at B=4" figure matches to 3 digits (0.619 → 62 %).
**Supported**. The shares sum to 1.02–1.03 — technical caveat noted above;
does not change any conclusion.

### 3.4 Reproducibility
- Output JSON includes the source directory, top_k, sample count, seed, and
  row counts. **Sufficient provenance**.
- Determinism: `random_seed` is passed into the random baseline; reruns with
  the same seed yield identical numbers. **OK**.

---

## 4. Claim-by-claim status

| # | Claim | Status | Evidence |
|---|---|---|---|
| C1 | Diagnostics module computes valid Jaccard statistics. | **Supported** | Hand-verified identities in §2.1–2.2. |
| C2 | Variance decomposition recovers 62 % within-seed at B=4. | **Supported** | `combinatorial_diagnostics.json`: 0.619. |
| C3 | Top-k vs oracle Jaccard ratio is ≈1.5× across B. | **Contradicted** | Observed ratios 1.20 / 1.18 / 1.30. |
| C4 | Top-k schedules do not closely mirror the mean-delta oracle. | **Supported** | Weak ratios (≤1.30×) confirm this directly. |
| C5 | Code is reproducible from raw Phase 2b files. | **Supported** | CLI reads `mc_raw.json` + `policy_raw.json`, writes JSON with meta, deterministic random baseline. |
| C6 | Unit tests adequately cover the module. | **Plausible** | Core math covered; hand-computed ratio test would strengthen. |
| C7 | within_share + between_share equals 1.0. | **Contradicted** | Sums are ≈1.02–1.03 by construction; descriptive shares, not ANOVA. |

---

## 5. Required narrow fixes

1. **FIX-1 (docs)**: anywhere a "≈1.5×" top-k-vs-oracle Jaccard is cited,
   replace with the file-backed range "1.18–1.30× across B ∈ {2, 3, 4}" and
   adjust downstream sentences. Scope: thesis docs; not code.
2. **FIX-2 (docs)**: when quoting `within_share`, note that shares do not
   sum exactly to 1.0 because `total_var` is a direct sample variance, not
   the ANOVA reconstruction. The within-share number is reliable to 2
   decimals as a descriptor.
3. **Optional (code, low priority)**: add a hand-computed ratio test case
   and/or bottom-k-vs-oracle ratio to the diagnostics output. Not a blocker.

No code correctness bug was found.

---

## 6. Verdict

- **Code correctness**: GREEN.
- **Output correctness**: GREEN.
- **Evidence quality**: GREEN for the within-seed variance fact (62 %);
  AMBER for any "≈1.5× Jaccard" narrative — must be corrected to 1.2–1.3×.
- **Reproducibility**: GREEN.

The module is safe to use. The canonical narrative needs a small calibration
fix (FIX-1) before any doc that cites the diagnostics is considered final.

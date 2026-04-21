# Signal-Adaptive Corrector Scheduling for Masked Diffusion Language Models

MSc thesis by Matteo Omizzolo, supervised by Prof. Giacomo Zanella (Bocconi University). The thesis now has a clear empirical verdict: fixed-budget corrector allocation is a **combinatorial trajectory-control problem**, and greedy per-step rankers are the wrong solution class. Phase 3a closes the empirical contract on ProSeCo-OWT: coordinate descent and beam-search scheduling both beat uniform, recover most of the Phase 2b oracle headroom, and show that the recoverable structure lives at the schedule level, not in a separable per-step score.

## What to read first

1. `docs/thesis/CURRENT_INDEX.md`
2. `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md`
3. `docs/thesis/CANONICAL_EXPERIMENT_OVERVIEW.md`
4. `docs/thesis/experiments/RESULTS_STATUS.md`
5. `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md`
6. `docs/thesis/theory/THEORY_STATUS.md`

## Repository map

| Area | Purpose |
|---|---|
| `src/mdm_playground/scheduling/` | Current thesis code: signals, budget allocation, schedule evaluation, and backend loaders |
| `src/mdm_playground/analysis/` | Paired comparisons, bootstrap CIs, and plotting helpers for Phase 2b / Phase 3a |
| `scripts/run_phase2b_proseco_owt.py` | Phase 2b paired evaluation and MC-oracle sweep |
| `scripts/run_phase3a_combinatorial.py` | Phase 3a combinatorial scheduling baselines |
| `scripts/analyze_phase2b.py` / `scripts/analyze_phase3a.py` | Aggregation and result summarisation for the mainline runs |
| `results/phase2b_proseco_owt/` | Raw Phase 2b paired outputs |
| `results/phase3a_proseco_owt/` | Raw Phase 3a combinatorial outputs |
| `results/phase2b/`, `results/phase3a/` | Aggregated summaries used in the thesis write-up |
| `figures/phase3a/` | Main Phase 3a figures |
| `thesis/`, `research/`, `docs/thesis/` | Thesis source, proof ledger, and canonical documentation |
| `archive/` | Legacy framework code, stale experiments, and historical notes |

## Current mainline

- **Phase 2b:** paired K-seed evaluation on ProSeCo-OWT, comparing signal-based policies against uniform and MC-oracle baselines.
- **Phase 3a:** combinatorial schedulers (CD-G and BS-AG) over schedules, not step-wise rankers.
- **Phase 3b:** theory finalisation only; no new HPC runs.

The active thesis path is:

`scripts/run_phase2b_proseco_owt.py` → `scripts/run_phase3a_combinatorial.py` → `src/mdm_playground/scheduling/*`

## Main result artifacts

- `results/phase2b_proseco_owt/policy_raw.json`
- `results/phase2b_proseco_owt/mc_raw.json`
- `results/phase2b/policy_comparison_paired.json`
- `results/phase2b/mc_oracle.json`
- `results/phase3a_proseco_owt/cd_raw.json`
- `results/phase3a_proseco_owt/bs_raw.json`
- `results/phase3a_proseco_owt/cd_paired.json`
- `results/phase3a_proseco_owt/bs_paired.json`
- `results/phase3a_proseco_owt/oracle_gap_closure.json`
- `results/phase3a/oracle_gap_closure.json`

## Reproducibility

The active backend depends on a locally staged **ProSeCo-OWT** HuggingFace snapshot, not on a hidden local source tree. Use `python scripts/stage_proseco_owt.py --dest ~/mdm/checkpoints/proseco_owt` once, then point the thesis scripts at that directory. The backend loader checks for the snapshot files and fails with a clear setup message if they are missing.

See `REPRODUCIBILITY.md` for the exact setup and run commands.

## Scope

This repository is intentionally focused on the thesis experiments and proofs. It is **not** a generic masked-diffusion benchmarking framework; the older adapter/sampler code and stale exploratory runs are archived or labeled as legacy.

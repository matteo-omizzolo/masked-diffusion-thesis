# Scripts

> Current phase: theory-first reassessment and Phase 0 reproducibility planning.
> No full-scale new HPC experiments until PF1–PF8 pass and the K=3 smoke matches qualitatively.
> See `docs/05_next_steps.md` for the Phase 0 checklist.
> See `docs/04_results_index.md` for results map. See `CLAUDE.md` for HPC environment.

## Phase 0 entry point

```bash
pytest tests/test_phase0_preflight.py -q
```

This is the blocking local pre-flight suite. Checkpoint-backed ProSeCo
equivalence checks are marked as explicit integration skips until the backend
exposes the needed comparison hooks. After PF1–PF8 are implemented or manually
verified, run the K=3 smoke, then the K=30 critical replication only if the
smoke matches qualitatively.

## Reproducibility workflow

```bash
# 1. Stage checkpoint (once, on HPC)
python scripts/stage_proseco_owt.py --dest ~/mdm/checkpoints/proseco_owt

# 2. Phase 2b reproduction (after Phase 0 gates)
python scripts/run_phase2b_proseco_owt.py --checkpoint ~/mdm/checkpoints/proseco_owt ...
python scripts/analyze_phase2b.py ...

# 3. Phase 3a reproduction (after Phase 2b replication gate)
python scripts/run_phase3a_combinatorial.py --checkpoint ~/mdm/checkpoints/proseco_owt ...
python scripts/analyze_phase3a.py ...

# 4. Protocol C (CPU, no HPC needed)
python scripts/run_protocol_c_owt.py ...
```

The backend loader expects a staged directory with `config.json`, `configuration_proseco.py`,
`modeling_proseco.py`, and model weights. If any file is missing, the loader raises a setup
error pointing to `stage_proseco_owt.py`.

## Script inventory

| File | Classification | Use now |
|---|---|---|
| `tests/test_phase0_preflight.py` | CURRENT_PHASE0 | Blocking local PF1–PF8 pre-flight suite. |
| `stage_proseco_owt.py` | CURRENT_PHASE0 / CURRENT_REPRODUCE_COMPLETED | Stage ProSeCo-OWT checkpoint before smoke or reproduction. |
| `run_phase2b_proseco_owt.py` | CURRENT_REPRODUCE_COMPLETED | Reproduce Phase 2b after Phase 0 gates. |
| `analyze_phase2b.py` | CURRENT_REPRODUCE_COMPLETED | Analyze Phase 2b raw rows into summaries. |
| `compute_theorem_a_constants.py` | CURRENT_REPRODUCE_COMPLETED | Recompute Theorem A constants from Phase 2b outputs. |
| `analyze_combinatorial_diagnostics.py` | CURRENT_REPRODUCE_COMPLETED | Recompute MC/pool overlap and Jaccard diagnostics. |
| `run_phase3a_combinatorial.py` | CURRENT_REPRODUCE_COMPLETED | Reproduce CD-G / BS-AG Phase 3a after replication gate. |
| `analyze_phase3a.py` | CURRENT_REPRODUCE_COMPLETED | Analyze Phase 3a closure ratios. |
| `run_protocol_c_owt.py` | LEGACY_PROVENANCE | Reproduce appendix-grade online-controller pilot only. |
| `legacy/run_protocol_a_proseco_snapshot.py` | LEGACY_PROVENANCE | Historical Phase 1 snapshot; do not use for current status. |
| `legacy/debug_proseco_owt_load.py` | LEGACY_PROVENANCE | Loader debugging only. |
| `legacy/debug_proseco_llada_sft_load.py` | LEGACY_PROVENANCE | Closed LLaDA-SFT debugging only. |
| `legacy/stage_proseco_llada_sft.py` | LEGACY_PROVENANCE | Closed cross-backbone staging only. |

There is no dedicated FUTURE_PHASE1 or FUTURE_PHASE2 script yet. Add those only
after Phase 0 passes and the Phase 1 schedule-level validation design is fixed.
Do not create new active theory/planning docs for that work; update the active
docs instead.

## Restart experiment contract

Every future experiment phase must produce:

- config JSON/YAML with git commit hash;
- per-seed raw rows;
- aggregate summary JSON;
- analysis script;
- paired bootstrap CI where applicable;
- one concise interpretation markdown or summary section;
- canonical plots when meaningful;
- exact command to regenerate plots from raw results.

Each experiment must answer: research question, theorem/assumption tested,
expected outcome, observed outcome, interpretation, opened/closed gate, and the
first plot/table to inspect.

Default plots: Phase 0 uses only a sanity table comparing old vs rerun keys.
Phase 1 uses ξ heatmap, sign-probability heatmap P(ξ>0), phase-pair summary,
Q-vs-G versus A-vs-G scatter, ζ_{B,C} versus η_{B,C}, and CI for P_B − R_B.
Phase 2 uses gain over uniform by policy, closure ratio versus MC-oracle or pool
oracle, held-out Q_hat-vs-G scatter, and true-G calls versus performance.

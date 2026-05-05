# Scripts

> **No new experiments authorized.** Scripts are preserved for reproducibility only.
> See `docs/04_results_index.md` for results map. See `CLAUDE.md` for HPC environment.

## Reproducibility workflow

```bash
# 1. Stage checkpoint (once, on HPC)
python scripts/stage_proseco_owt.py --dest ~/mdm/checkpoints/proseco_owt

# 2. Phase 2b
python scripts/run_phase2b_proseco_owt.py --checkpoint ~/mdm/checkpoints/proseco_owt ...
python scripts/analyze_phase2b.py ...

# 3. Phase 3a
python scripts/run_phase3a_combinatorial.py --checkpoint ~/mdm/checkpoints/proseco_owt ...
python scripts/analyze_phase3a.py ...

# 4. Protocol C (CPU, no HPC needed)
python scripts/run_protocol_c_owt.py ...
```

The backend loader expects a staged directory with `config.json`, `configuration_proseco.py`,
`modeling_proseco.py`, and model weights. If any file is missing, the loader raises a setup
error pointing to `stage_proseco_owt.py`.

## Current (runnable for reproducibility)

| Script | Phase | Output |
|---|---|---|
| `stage_proseco_owt.py` | Repro | Stages ProSeCo-OWT checkpoint |
| `run_phase2b_proseco_owt.py` | Phase 2b | `results/phase2b_proseco_owt/` |
| `analyze_phase2b.py` | Phase 2b | `results/phase2b/` |
| `compute_theorem_a_constants.py` | Phase 2b | `results/phase2b/theorem_a_constants.json` |
| `analyze_combinatorial_diagnostics.py` | Phase 2b | Jaccard diagnostics |
| `run_phase3a_combinatorial.py` | Phase 3a | `results/phase3a_proseco_owt/` |
| `analyze_phase3a.py` | Phase 3a | `results/phase3a_proseco_owt/oracle_gap_closure.json` |
| `run_protocol_c_owt.py` | Protocol C | `results/protocol_c_owt/` |

## Legacy (superseded)

| Script | Why legacy |
|---|---|
| `legacy/run_protocol_a_proseco_snapshot.py` | Phase 1 one-off snapshot; done |
| `legacy/debug_proseco_owt_load.py` | Debug; OWT load issue fixed |
| `legacy/debug_proseco_llada_sft_load.py` | Debug; LLaDA-SFT direction closed |
| `legacy/stage_proseco_llada_sft.py` | LLaDA-SFT staging; cross-backbone probe closed |

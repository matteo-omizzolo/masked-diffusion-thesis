# Scripts

## Current (runnable for reproducibility)

| Script | Phase | Purpose |
|---|---|---|
| `run_phase2b_proseco_owt.py` | Phase 2b | K=30 paired policy + MC oracle evaluation |
| `analyze_phase2b.py` | Phase 2b | Aggregate and rank analysis |
| `run_phase3a_combinatorial.py` | Phase 3a | CD-G + BS-AG combinatorial search |
| `analyze_phase3a.py` | Phase 3a | Paired comparison + oracle-gap closure |
| `run_protocol_c_owt.py` | Protocol C | Bounded adaptive-controller pilot (CPU) |
| `stage_proseco_owt.py` | Repro | Stage ProSeCo-OWT checkpoint on HPC |
| `compute_theorem_a_constants.py` | Phase 2b | Measure σ_ξ, ρ, γ, ε_R from MC residuals |
| `analyze_combinatorial_diagnostics.py` | Phase 2b | Jaccard diagnostics on MC + oracle schedules |

## Legacy (one-off or debug, superseded)

| Script | Why legacy |
|---|---|
| `legacy/run_protocol_a_proseco_snapshot.py` | Phase 1 one-off snapshot; done |
| `legacy/debug_proseco_owt_load.py` | Debug script; OWT load issue fixed |
| `legacy/debug_proseco_llada_sft_load.py` | Debug script; LLaDA-SFT direction closed |
| `legacy/stage_proseco_llada_sft.py` | LLaDA-SFT staging; cross-backbone probe closed |

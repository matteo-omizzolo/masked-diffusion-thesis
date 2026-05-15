# Signal-Adaptive Corrector Scheduling

MSc thesis by Matteo Omizzolo, supervised by Prof. Giacomo Zanella.

**Open this first:** [`START_HERE.md`](START_HERE.md).

## Current State

ProSeCo-OWT is a completed case study. The active frontier is
informed-correctors/Text8 validation on an external GPU, because the Bocconi
JAX/CUDA/MIG path has been exhausted. Do not infer current status from
`docs/archive/`, `archive/`, or old thesis drafts.

## Compact Repo Map

| Path | Meaning |
|---|---|
| `START_HERE.md` | Human dashboard: status, next action, guardrails |
| `docs/ACTIVE_RESEARCH_STATE.md` | One-page research state |
| `docs/10_external_gpu_text8_fallback.md` | Active Azure/external-GPU runbook |
| `src/mdm_playground/` | Installable package |
| `scripts/proseco/` | Closed ProSeCo analysis/reproduction scripts |
| `scripts/backend_validation/` | informed-correctors/Text8 smoke utilities |
| `hpc/` | Bocconi scripts retained as provenance/runbooks |
| `results/` | Result indexes plus preserved local raw outputs |
| `external/` | Tracked upstream submodules for legacy/provenance backends |
| `external_repos/` | Ignored clone-on-demand workspace; not part of the repo |
| `checkpoints/` | Local-only model weights; not part of git |

## Canonical Evidence

The raw ProSeCo folders stay at their historical paths for reproducibility, but
they are not the entry point. Use:

- `docs/04_results_index.md`
- `results/EXPERIMENT_INDEX.json`
- `results/BACKEND_FEASIBILITY_INDEX.json`

## Reproduction Pointers

To reproduce Phase 2b or Phase 3a on the HPC, stage the ProSeCo-OWT checkpoint once:

```bash
python scripts/proseco/reproduction/stage_proseco_owt.py --dest ~/mdm/checkpoints/proseco_owt
```

Then use `hpc/push.sh` to sync and `hpc/proseco/reproduction/phase2b_proseco_owt.sbatch`
or `hpc/proseco/reproduction/phase3a_combinatorial.sbatch` to submit. See
`scripts/README.md` for the full script index and `CLAUDE.md` for HPC
environment details.

For the current Text8 backend path, use `docs/10_external_gpu_text8_fallback.md`.
No full Text8 training is approved until Stage 1 passes on the external GPU.

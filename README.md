# Signal-Adaptive Corrector Scheduling for Masked Diffusion Language Models

MSc thesis by Matteo Omizzolo, supervised by Prof. Giacomo Zanella (Bocconi University).

## Start here

→ **[`START_HERE.md`](START_HERE.md)** — 2-minute orientation.

## Current-status rule

> `START_HERE.md` and `docs/README.md` are the only entry points.
> Do not use `docs/archive/`, `archive/`, or `docs/thesis/` to infer current status —
> those are historical and may contradict current docs.

## Repo map

| Area | Purpose |
|---|---|
| `START_HERE.md` | Orientation — thesis status, results, open items |
| `docs/` | Active compact docs (`docs/README.md` for index) |
| `docs/archive/` | Archived historical docs — not current |
| `thesis/` | LaTeX chapters (`thesis/main.tex`) |
| `research/` | Theorem worklog, proof ledger, open questions |
| `src/mdm_playground/` | Main Python package (`pip install -e .`) |
| `scripts/` | Analysis scripts; `scripts/legacy/` for superseded |
| `results/` | Raw experiment outputs — never deleted |
| `hpc/` | Bocconi HPC sbatch scripts and push/pull helpers |
| `external/` | Upstream repos (ProSeCo, MDLM, ReMDM, PRISM, sedd) |
| `archive/` | Legacy code and notes from before April 2026 |

## Raw results

| Folder | Phase |
|---|---|
| `results/phase1_proseco_owt_full/` | Phase 1 Protocol A — signal calibration |
| `results/phase2b_proseco_owt/` | Phase 2b — policy comparison + MC-oracle |
| `results/phase2b/` | Phase 2b — aggregated analysis outputs |
| `results/phase3a_proseco_owt/` | Phase 3a — CD-G + BS-AG combinatorial search |
| `results/cross_backbone/` | LLaDA-SFT bounded probe |
| `results/protocol_c_owt/` | Protocol C — adaptive controller pilot |

See `docs/04_results_index.md` for the full results map.

## Reproducibility

To reproduce Phase 2b or Phase 3a on the HPC, stage the ProSeCo-OWT checkpoint once:

```bash
python scripts/stage_proseco_owt.py --dest ~/mdm/checkpoints/proseco_owt
```

Then use `hpc/push.sh` to sync and `hpc/phase2b_proseco_owt.sbatch` or
`hpc/phase3a_combinatorial.sbatch` to submit. See `scripts/README.md` for
the full script index and `CLAUDE.md` for HPC environment details.

> **Note:** No full-scale new HPC experiments until the theory scaffold
> (Theorem A baseline, Theorem B/B′, Diagnostic Framework C) and Phase 0
> reproducibility audit are complete. See `docs/05_next_steps.md`.

# HPC workflow

Phase 0 complete (PF1–PF8 ✅, K=3 smoke ✅, 2026-05-06). K=30 replication is
now open: submit `phase2b_proseco_owt.sbatch` then `phase3a_combinatorial.sbatch`.
Phase 1 interaction diagnostics remain blocked until K=30 passes.

| File | Classification | Use |
|---|---|---|
| `phase0_smoke_k3.sbatch` | CLOSED_PROVENANCE | K=3 smoke provenance record (job 489457, 2026-05-06); gate passed. Do not resubmit. |
| `phase2b_proseco_owt.sbatch` | CURRENT_K30 | K=30 Phase 2b replication — submit now. |
| `phase3a_combinatorial.sbatch` | CURRENT_K30 | K=30 Phase 3a replication — submit after Phase 2b. |
| `push.sh` | CURRENT_UTILITY | Sync code to HPC. |
| `pull.sh` | CURRENT_UTILITY | Pull generated summaries from HPC when needed. |
| `setup_env.sh` | CURRENT_UTILITY | Prepare the HPC environment. |
| `legacy/cross_backbone_proseco_llada_sft_bounded.sbatch` | LEGACY_PROVENANCE | Closed LLaDA-SFT bounded probe; do not use for current status. |
| `legacy/cross_backbone_proseco_llada_sft_resume_phase2b.sbatch` | LEGACY_PROVENANCE | Closed LLaDA-SFT resume job; do not use for current status. |

The older `phase1_*` and `submit.sh` flows are legacy and live in `archive/`.

## Quick start

```bash
bash hpc/push.sh
ssh 3316152@slogin.hpc.unibocconi.it "cd ~/mdm/masked-diffusion-thesis && bash hpc/setup_env.sh"
ssh 3316152@slogin.hpc.unibocconi.it "cd ~/mdm/masked-diffusion-thesis && sbatch hpc/phase2b_proseco_owt.sbatch"
# later
ssh 3316152@slogin.hpc.unibocconi.it "cd ~/mdm/masked-diffusion-thesis && sbatch hpc/phase3a_combinatorial.sbatch"
```

## Cluster details

| Item | Value |
|---|---|
| Host | `slogin.hpc.unibocconi.it` |
| User | `3316152` |
| Partition / QOS | `stud` / `stud` |
| GPU limit | 4 × A100 per job |
| Repo path | `~/mdm/masked-diffusion-thesis` |
| ProSeCo-OWT snapshot | `~/mdm/checkpoints/proseco_owt` |

## Reproducibility note

The active backend expects a staged ProSeCo-OWT snapshot at
`~/mdm/checkpoints/proseco_owt/`. Run `python scripts/stage_proseco_owt.py`
once to stage it. See `scripts/README.md` for the full reproducibility workflow.

> Current phase: K=30 replication. Phase 0 complete (PF1–PF8 ✅, K=3 smoke ✅).
> Submit phase2b then phase3a sbatches. Phase 1 interaction diagnostics blocked
> until K=30 passes. See `docs/05_next_steps.md` for the full sequential gate plan.
> Existing scripts are preserved for reproducibility of ProSeCo-OWT results.

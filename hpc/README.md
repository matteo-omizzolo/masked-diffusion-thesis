# HPC workflow

No full-scale HPC job should be submitted until PF1–PF8 pass and the K=3 smoke
matches qualitatively. The current known entry points are:

| File | Classification | Use |
|---|---|---|
| `phase0_smoke_k3.sbatch` | CURRENT_PHASE0 | K=3 smoke (Phase 0 gate); writes to `results/phase0_smoke_<sha>/`. |
| `phase2b_proseco_owt.sbatch` | CURRENT_REPRODUCE_COMPLETED | Phase 2b reproduction after Phase 0 gates. |
| `phase3a_combinatorial.sbatch` | CURRENT_REPRODUCE_COMPLETED | Phase 3a reproduction after Phase 2b replication gate. |
| `push.sh` | CURRENT_PHASE0 | Sync code to HPC. |
| `pull.sh` | CURRENT_PHASE0 | Pull generated summaries from HPC when needed. |
| `setup_env.sh` | CURRENT_PHASE0 | Prepare the HPC environment. |
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

> Current phase: theory-first reassessment. No full-scale HPC runs until the theory
> scaffold (Theorem A/B/B′ + Diagnostic Framework C; D/E optional) is stable,
> PF1–PF8 pass, and the K=3 smoke matches qualitatively.
> Phase 0 checklist is in `docs/05_next_steps.md`.
> Existing scripts are preserved for reproducibility of ProSeCo-OWT results.

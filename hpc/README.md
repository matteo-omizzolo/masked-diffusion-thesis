# HPC workflow

This directory only documents the thesis mainline:

- `hpc/phase2b_proseco_owt.sbatch`
- `hpc/phase3a_combinatorial.sbatch`
- `hpc/push.sh`
- `hpc/pull.sh`
- `hpc/setup_env.sh`

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

> **No new HPC runs currently authorized.** Experiments are complete.
> This directory is kept for reproducibility only.

> **STATUS:** ARCHIVED
> **ARCHIVED ON:** 2026-04-22
> **SUPERSEDED BY:** n/a (operational snapshot)
> **REASON:** HPC sync log from 2026-04-18; superseded by current state.

---

# HPC Sync Report
**Date:** 2026-04-18  
**Status: COMPLETE**

---

## What Was Successfully Synced

| Directory | Files Transferred | Notes |
|-----------|-------------------|-------|
| `docs/experiments/` | 19 files | Entirely new subtree |
| `docs/CURRENT_INDEX.md` | 1 file | New index |
| `docs/instructions/v2/` | 3 files | New GPT-Pro assessment docs |
| `scripts/` | 4 new files | debug_*.py + run_phase1_*.py |
| `src/mdm_playground/scheduling/backends/` | 4 files | New backend adapters |
| `hpc/` | 3 new files | phase1_*.sbatch + README.md |
| `research/` | 4 files | Updated candidate_theorems, proof_ledger, proof_worklog, open_questions |
| `figures/phase1_pilot/` | 6 PNGs | Phase 1 analysis plots |
| `results/phase1_mdlm_conf/` | ~70 JSON files | 20 traj + 45 schedules + summary |
| `results/phase1_mdlm_conf_surrogate_sanity/` | ~14 JSON files | Sanity check run |
| `results/phase1_proseco/` | ~70 JSON files | 20 traj + 45 schedules + summary |
| `results/phase1_proseco_surrogate_sanity/` | ~14 JSON files | Sanity check run |
| `out/` | 13 new `.out` files | phase1_pilot + phase1_mdlm_conf + phase1_proseco SLURM logs |
| `err/` | 13 new `.err` files | Corresponding stderr logs |
| `.gitignore` | 1 file | Updated exclusion rules |
| `CLAUDE.md` | 1 file | Updated project instructions |

---

## What Was Skipped

| Item | Reason |
|------|--------|
| `third_party/proseco/` | Working tree is empty on HPC (cloned but no files checked out). Clone locally if needed: `git clone https://github.com/kuleshov-group/proseco third_party/proseco` |
| `external/` | Already present locally; not modified on HPC |
| `docs/pdf/` | PDFs only; not modified on HPC |
| `*.pyc` / `__pycache__/` | Compiled bytecode |
| `results/**/generated_sequences.json` | Excluded per gitignore (large) |
| `*.pt`, `*.ckpt` etc. | Large model files |
| `checkpoints/` | Lives outside the repo at `~/mdm/checkpoints/` on HPC |

---

## Is Local Now In Sync With HPC?

**Yes** — for all lightweight code and data files. The local repo now contains:
- All Phase 1 experiment results (mdlm_conf, proseco, and sanity checks)
- All new source code in `src/mdm_playground/scheduling/backends/`
- All experiment planning documents in `docs/experiments/`
- All new HPC sbatch scripts
- All SLURM logs from Phase 1 runs

---

## Files That Still Require Manual Action

### 1. `third_party/proseco/` — clone locally

```bash
git clone https://github.com/kuleshov-group/proseco third_party/proseco
```

The HPC has an empty checkout of ProSeCo (the git objects are there but the working tree is empty). Local doesn't have it at all. Clone fresh locally.

### 2. Large model checkpoints (if needed)

```bash
scp 3316152@slogin.hpc.unibocconi.it:~/mdm/checkpoints/mdlm.ckpt ./checkpoints/
```

These live outside the HPC repo dir and were never synced. Only needed if running inference locally.

---

## Verification Command

To confirm local is up to date with HPC on the key directories:

```bash
# Dry-run rsync of the most important dirs — should show 0 new files
rsync -av --dry-run \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.egg-info/' \
  3316152@slogin.hpc.unibocconi.it:~/mdm/masked-diffusion-thesis/src/ ./src/

rsync -av --dry-run \
  3316152@slogin.hpc.unibocconi.it:~/mdm/masked-diffusion-thesis/docs/ ./docs/

rsync -av --dry-run \
  3316152@slogin.hpc.unibocconi.it:~/mdm/masked-diffusion-thesis/results/ ./results/
```

All three should report `total size is X  speedup is Y` with no file transfers listed.

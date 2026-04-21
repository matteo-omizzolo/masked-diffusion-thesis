# Reproducibility

## Thesis mainline

The current thesis workflow is:

1. `python scripts/stage_proseco_owt.py --dest ~/mdm/checkpoints/proseco_owt`
2. `python scripts/run_phase2b_proseco_owt.py --checkpoint ~/mdm/checkpoints/proseco_owt ...`
3. `python scripts/analyze_phase2b.py ...`
4. `python scripts/run_phase3a_combinatorial.py --checkpoint ~/mdm/checkpoints/proseco_owt ...`
5. `python scripts/analyze_phase3a.py ...`

The backend loader expects a staged directory containing:

- `config.json`
- `configuration_proseco.py`
- `modeling_proseco.py`
- model weights (`model.safetensors` or `pytorch_model.bin`)

If any file is missing, the loader raises a setup error that points back to
`scripts/stage_proseco_owt.py`.

## ProSeCo-OWT dependency

- **What it is:** the co-trained ProSeCo checkpoint used for the thesis mainline.
- **Where it lives:** `~/mdm/checkpoints/proseco_owt/` on the HPC, or any local
  path you pass to `--checkpoint`.
- **How to obtain it:** run `scripts/stage_proseco_owt.py` once on a machine
  with HuggingFace access.
- **What it replaces:** the older MDLM-checkpoint ProSeCo backend, which is kept
  only as a legacy comparison baseline.

## HPC entry points

- `hpc/phase2b_proseco_owt.sbatch`
- `hpc/phase3a_combinatorial.sbatch`
- `hpc/push.sh`
- `hpc/pull.sh`
- `hpc/setup_env.sh`

## Canonical outputs

- Phase 2b: `results/phase2b_proseco_owt/`, `results/phase2b/`
- Phase 3a: `results/phase3a_proseco_owt/`, `results/phase3a/`
- Figures: `figures/phase3a/`

Legacy exploratory runs, smoke logs, and older framework code are archived
under `archive/`.

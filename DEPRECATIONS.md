# DEPRECATIONS

This file documents removed scripts and their replacement commands.

---

## Removed: `scripts/run_remdm.py`

Old usage:
```bash
python scripts/run_remdm.py --config configs/remdm.yaml
```

Replacement:
```bash
python -m mdm_playground.cli.run --method remdm --toy_mode
python -m mdm_playground.cli.run --method remdm --dry_run --steps 256
python -m mdm_playground.cli.run --method remdm --model_id /path/to/ckpt.pt
```

---

## Removed: `scripts/run_infer.py`

Old usage:
```bash
python scripts/run_infer.py \
    --model maple-research-lab/RemeDi-RL \
    --prompt "..." --strategy baseline --steps 32
```

Replacement:
```bash
python -m mdm_playground.cli.run \
    --method remedi \
    --model_id maple-research-lab/RemeDi-RL \
    --prompt "..." --strategy baseline --steps 32
```

---

## Removed: `scripts/sanity_check.py`

One-off integration sanity script.  
No direct replacement — use `pytest tests/` for automated checks.

---

## Removed: `scripts/inspect_remdm_interface.py`

One-off upstream code introspection script.  
No direct replacement — read `external/remdm/main.py` directly.

---

## Removed: `docs/REMDM_INTEGRATION.md`

Superseded by this README and `hpc/README.md`.

---

## Removed: `remedi_infer/`

The `remedi_infer` package has been removed from the repository. All symbols are available from the new package:

| Old import | New import |
|---|---|
| `remedi_infer.strategies.StepState` | `mdm_playground.strategies.StepState` |
| `remedi_infer.strategies.BaselineUnmaskStrategy` | `mdm_playground.strategies.BaselineUnmaskStrategy` |
| `remedi_infer.strategies.RemediPolicyStrategy` | `mdm_playground.strategies.RemediPolicyStrategy` |
| `remedi_infer.strategies.ConfidenceThresholdRemaskStrategy` | `mdm_playground.strategies.ConfidenceThresholdRemaskStrategy` |
| `remedi_infer.strategies.TopKLowConfidenceRemaskStrategy` | `mdm_playground.strategies.TopKLowConfidenceRemaskStrategy` |
| `remedi_infer.strategies.ScheduledRemaskStrategy` | `mdm_playground.strategies.ScheduledRemaskStrategy` |
| `remedi_infer.load_model.RemeDiModelBundle` | `mdm_playground.models.remedi.RemeDiAdapter` |
| `remedi_infer.sampler.run_sampler` | `mdm_playground.samplers.run_block_diffusion` |
| `remedi_infer.logging.InferenceLogger` | `mdm_playground.core.logging.TrajectoryLogger` |

---

## Removed: `src/masked_diffusion_thesis/utils/`

The legacy shim modules under `src/masked_diffusion_thesis/utils/` have been removed. Use the `mdm_playground` equivalents:

| Old import | New import |
|---|---|
| `masked_diffusion_thesis.utils.config.load_yaml` | `mdm_playground.core.config.load_yaml` |
| `masked_diffusion_thesis.utils.git.get_git_commit_hash` | `mdm_playground.core.utils.get_git_commit_hash` |
| `masked_diffusion_thesis.utils.io.save_json` | `mdm_playground.core.utils.save_json` |
| `masked_diffusion_thesis.utils.logging.setup_logger` | `mdm_playground.core.logging.setup_logger` |
| `masked_diffusion_thesis.utils.reproducibility.seed_everything` | `mdm_playground.core.utils.seed_everything` |

---

## Removed empty stub directories

The following were empty stubs with no real code, now deleted:

- `src/masked_diffusion_thesis/data/`
- `src/masked_diffusion_thesis/evaluation/`
- `src/masked_diffusion_thesis/samplers/`
- `src/masked_diffusion_thesis/training/`
- `src/masked_diffusion_thesi` (stray empty file)
- `source/` (empty directory)

---

All deprecated shim modules and the `remedi_infer/` package have been removed from the repository. Use `mdm_playground` going forward.

# Third-Party Repositories Manifest

This document records the provenance of external repositories vendored or linked in this project.

## ProSeCo — Self-Correcting Masked Diffusion Models

**Repository URL:** https://github.com/kuleshov-group/proseco

**Fetch Date:** 2026-04-17

**Commit Hash:** `69fe45d101d8367b0d1807466b409f71449a7c58`

**License:** Apache License 2.0

**Last Commit Message:** "Update environment setup instructions in README"

**Description:**
ProSeCo is a self-correcting discrete diffusion language model that learns to apply corrective refinement loops during inference. The paper and code were published by Volodymyr Kuleshov's group at Cornell:

- **Title:** "Learn from Your Mistakes: Self-Correcting Masked Diffusion Models"
- **arXiv:** 2602.11590
- **Year:** 2026

**Why This Dependency:**
This thesis targets **signal-adaptive corrector scheduling**—deciding when to spend corrective effort across a diffusion trajectory for fixed predictor schedules and fixed corrector budget. ProSeCo provides:
- A production-quality implementation of corrector training and inference
- Configurable corrector scheduling parameters (`corrector_every_n_steps`, `corrector_steps`, `corrector_start_iter`)
- Trained checkpoints on OpenWebText (`proseco-owt`) for empirical validation
- The primary reference implementation for comparing corrector scheduling strategies

**Location:** no active `external/proseco/` checkout. The thesis uses the
HuggingFace `proseco-owt` snapshot staged under local checkpoints by
`scripts/proseco/reproduction/stage_proseco_owt.py`.

**Location Type:** staged checkpoint snapshot, not an active submodule

---

## Related Repositories (Already Vendored)

- **MDLM** (`external/mdlm/`): Masked Diffusion Language Model baseline
- **ReMDM** (`external/remdm/`): ReMDM with loop integration
- **PRISM** (`external/PRISM/`): Prompt-Aware PLM Sampling
- **SEDD** (`external/sedd/`): Stochastic Exponential Diffusion Discrete Models
- **RemeDi** (`external/remedi/`): RemeDi with RL (submodule, not integrated)

## Clone-On-Demand Repositories

`lindermanlab/informed-correctors` is intentionally not a tracked submodule.
Clone it only when running backend-validation smoke/training work:

```bash
mkdir -p external_repos
git -C external_repos clone https://github.com/lindermanlab/informed-correctors
git -C external_repos/informed-correctors checkout 8371dec
```

`external_repos/` is ignored by git so the root repository remains compact.

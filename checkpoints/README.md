# Local checkpoints

This folder is **local-only** and ignored by git (except this file and `.gitkeep`).

Do not commit model weights. See `CLAUDE.md` §HPC for checkpoint staging instructions.

| File | Description |
|---|---|
| `mdlm.ckpt` | Legacy MDLM checkpoint; not active and safe to omit locally unless rerunning legacy scripts |
| `proseco_owt/` | ProSeCo-OWT snapshot (staged via `scripts/proseco/reproduction/stage_proseco_owt.py`) |

HPC path: `~/mdm/checkpoints/` on `slogin.hpc.unibocconi.it`.

# Local checkpoints

This folder is **local-only** and ignored by git (except this file and `.gitkeep`).

Do not commit model weights. See `CLAUDE.md` §HPC for checkpoint staging instructions.

| File | Description |
|---|---|
| `mdlm.ckpt` | MDLM checkpoint (~2.5 GB); used by legacy scripts only |
| `proseco_owt/` | ProSeCo-OWT snapshot (staged via `scripts/stage_proseco_owt.py`) |

HPC path: `~/mdm/checkpoints/` on `slogin.hpc.unibocconi.it`.

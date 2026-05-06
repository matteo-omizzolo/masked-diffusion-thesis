# Ignored Local ProSeCo Checkout Audit — 2026-05-06

This records an ignored local checkout found at `external/proseco/` during the
Phase 0 implementation pass.

| Field | Value |
|---|---|
| Path found | `external/proseco/` |
| Classification | `MODIFIED_LOCAL` and `NOT_NEEDED` |
| Remote | `https://github.com/kuleshov-group/proseco` |
| Branch | `main` |
| Commit | `69fe45d101d8367b0d1807466b409f71449a7c58` |
| Local modifications | `models/hf/modeling_proseco.py` |
| Reason not active | The current Phase 0/ProSeCo-OWT backend loads a staged HuggingFace snapshot via `scripts/stage_proseco_owt.py`; it does not import `external/proseco/`. |
| Preservation | Compact patch recorded in `proseco_local_patch.diff`; the full ignored checkout was moved to `/private/tmp/masked-diffusion-thesis-external-backups/proseco_modified_20260506` and was not committed. |

Rule: do not make silent local edits inside `external/`. Project-specific
patching should live in staging scripts, backend wrappers, or documented patch
files.

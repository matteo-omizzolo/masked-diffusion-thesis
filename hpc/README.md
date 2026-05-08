# HPC workflow

Phase 0 is complete (PF1–PF8 ✅ slnode01 2026-05-06; K=3 smoke ✅ gnode01 A100
job 489457 2026-05-06; gate declared 2026-05-07). K=30 replication is complete:
Phase 2b job 490106 finished on gnode01 at 14:18 CEST 2026-05-07, and canonical
Phase 3a job 479941 is complete and intact. The current HPC gate is Phase 1
interaction diagnostics (Gate 3a sparse pairs, then Gate 3b schedule-level validation).

| File | Classification | Use |
|---|---|---|
| `phase0_smoke_k3.sbatch` | CLOSED_PROVENANCE | K=3 smoke provenance record (job 489457, 2026-05-06); gate passed. Do not resubmit. |
| `phase2b_proseco_owt.sbatch` | CLOSED_PROVENANCE | K=30 Phase 2b replication script. Job 490106 completed; do not resubmit for the closed gate. |
| `phase3a_combinatorial.sbatch` | CLOSED_PROVENANCE_NEEDS_OFFLINE_FIX | K=30 Phase 3a canonical/provenance script. Job 479941 completed. Duplicate job 490469 failed harmlessly before shard launch due compute-node pip/PyPI access; do not resubmit for the closed gate. |
| `push.sh` | CURRENT_UTILITY | Sync code to HPC. |
| `pull.sh` | CURRENT_UTILITY | Pull generated summaries from HPC when needed. |
| `setup_env.sh` | CURRENT_UTILITY | Prepare the HPC environment. |
| `legacy/cross_backbone_proseco_llada_sft_bounded.sbatch` | LEGACY_PROVENANCE | Closed LLaDA-SFT bounded probe; do not use for current status. |
| `legacy/cross_backbone_proseco_llada_sft_resume_phase2b.sbatch` | LEGACY_PROVENANCE | Closed LLaDA-SFT resume job; do not use for current status. |

The older `phase1_*` and `submit.sh` flows are legacy and live in `archive/`.

## Pre-submit rule (mandatory)

Before submitting **any** sbatch job, run a Codex pre-submit review via
`/codex:rescue` in the Claude Code session. The review must cover:
- The exact sbatch script and any files changed since the last push.
- `git status` and `git diff --check`.
- Validation commands (e.g., `pytest tests/test_phase0_preflight.py -q`).
- Expected output folder and which gate is being opened or closed.
- Confirmation no K=3 smoke numbers are being elevated to thesis evidence.

Address BLOCKING and WARNING findings before submitting. Document the Codex
finding summary in the commit message. This rule applies each session.

## Quick start

There is no current K=30 submission step. Use this only for syncing code before
an approved Phase 1 diagnostics run:

```bash
bash hpc/push.sh
ssh 3316152@slogin.hpc.unibocconi.it "cd ~/mdm/masked-diffusion-thesis && bash hpc/setup_env.sh"
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

## Closed K=30 job notes

- Phase 2b K=30 replication: job 490106, gnode01, completed 2026-05-07.
  Output `results/phase2b_k30_rep_cf89e00/` has 4/4 shards, seeds 42–71,
  `policy_raw.json` with 1500 rows, and `mc_raw.json` with 9000 rows.
- Phase 3a canonical K=30: job 479941, gnode02, complete since 2026-04-20.
  Output `results/phase3a_proseco_owt/` has 60 per-seed files plus
  `cd_raw.json` and `bs_raw.json`.
- Phase 3a duplicate resubmission: job 490469 failed with ExitCode 1:0 after
  54 seconds because `pip install -e .` in the sbatch preamble attempted outbound
  PyPI access from a compute node. It died before any shard launched and wrote
  no files. Verdict: harmless.

Before reusing `phase2b_proseco_owt.sbatch` or `phase3a_combinatorial.sbatch`
for any future job, remove or make offline-safe the compute-node `pip install`
preamble. The conda env `remdm311` should already contain required packages;
future sbatches should not assume outbound internet from `gnode*`.

> Current phase: Phase 1 interaction diagnostics. Phase 0 and K=30 replication
> gates are closed. See `docs/05_next_steps.md` for the full sequential gate plan.
> Existing scripts are preserved for reproducibility of ProSeCo-OWT results.

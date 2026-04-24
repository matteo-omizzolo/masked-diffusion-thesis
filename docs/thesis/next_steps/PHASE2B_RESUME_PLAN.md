> **STATUS:** PLAN
> **LAST VERIFIED:** 2026-04-22 (v2 — aligned with actually-submitted design)
> **SCOPE:** Contract for the Phase 2b-only resume on LLaDA-SFT. Companion to `LARGE_MODEL_CONTINUATION_DECISION.md`. States what is reused, what is regenerated, and why the resume path is scientifically valid.

---

# PHASE 2B RESUME PLAN — LLaDA-SFT, Phase 2b only, 8-way sharded across 2 dependent jobs

## 1. What is being resumed

- The **Phase 2b** stage of `hpc/cross_backbone_proseco_llada_sft_bounded.sbatch` (which was wedged under `shard_count=1` in Job 480887 and would not finish in its 20 h walltime).
- Resumed via `hpc/cross_backbone_proseco_llada_sft_resume_phase2b.sbatch`, submitted twice under the `stud` QOS `MaxJobsPerUser=1` constraint:
  - **Job A** (`SHARD_START=0`, `SHARD_COUNT=8`): runs shards 0..3 in parallel on 4 MIG slices (one seed per shard → seeds 142, 143, 144, 145).
  - **Job B** (`SHARD_START=4`, `SHARD_COUNT=8`, `--dependency=afterany:<Job A>`): runs shards 4..7 in parallel on 4 MIG slices (seeds 146, 147, 148, 149).
- Submitted job IDs: **Job A = 481264**, **Job B = 481265**.

Not resumed:
- Preflight (already passed in Job 480887; nothing stateful to preserve).
- Protocol A (already complete on disk; re-running it would overwrite `run_config.json` and waste 16 h GPU).
- Phase 3a (explicitly blocked on `POST_CROSS_BACKBONE_DECISION.md`, which is only writable after Phase 2b results exist).
- `analyze_phase2b.py` (runs once locally / in a separate short job after all four shards finish, not per-shard).

## 2. Reuse contract (inputs)

The resume reads — and must not modify — the following artefacts:

| Input | Path | Role |
|---|---|---|
| Protocol A trajectories | `results/cross_backbone/proseco_llada_sft_bounded/protocol_a/trajectory_{0..7}.json` | Δ_t trace + signals per seed (loaded by `load_protocol_a` in `scripts/run_phase2b_proseco_owt.py`) |
| Protocol A run config | `results/cross_backbone/proseco_llada_sft_bounded/protocol_a/run_config.json` | Provenance only (read-only) |
| Checkpoint | `~/mdm/checkpoints/proseco_llada_sft` | Model weights; loaded by `ProSeCoLLaDASFTGenerator` |

The sbatch contains an explicit pre-check that fails fast if any Protocol A file is missing:

```python
need = [p / f'trajectory_{i}.json' for i in range(8)] + [p / 'run_config.json']
missing = [str(x) for x in need if not x.exists()]
assert not missing
```

## 3. Regeneration contract (outputs)

Each shard writes only to `results/cross_backbone/proseco_llada_sft_bounded/phase2b_raw/` and the shard-suffixed paths under it:

| Output | Path | Written by |
|---|---|---|
| Per-seed policy rows | `phase2b_raw/per_seed/policy_rows_seed{seed}.json` | shard that owns that seed |
| Per-seed MC rows | `phase2b_raw/per_seed/mc_rows_seed{seed}.json` | shard that owns that seed |
| Per-shard policy concat | `phase2b_raw/policy_raw.shard{i}-of-8.json` | shard `i` |
| Per-shard MC concat | `phase2b_raw/mc_raw.shard{i}-of-8.json` | shard `i` |
| Per-shard run config | `phase2b_raw/run_config.shard{i}.json` | shard `i` |
| Per-shard manifest | `phase2b_raw/manifest.json` | shard `i` (last writer wins; acceptable — manifest is summary) |

Seed ownership (from `my_seeds = [s for i, s in enumerate(seeds_all) if i % shard_count == shard_idx]` at line 418 of `run_phase2b_proseco_owt.py`, with `shard_count=8` and `seed_start=142`):

| Shard | Seed | Job | MIG slice (CUDA_VISIBLE_DEVICES) |
|---|---|---|---|
| 0 | 142 | 481264 | 0 |
| 1 | 143 | 481264 | 1 |
| 2 | 144 | 481264 | 2 |
| 3 | 145 | 481264 | 3 |
| 4 | 146 | 481265 | 0 |
| 5 | 147 | 481265 | 1 |
| 6 | 148 | 481265 | 2 |
| 7 | 149 | 481265 | 3 |

No two shards write the same per-seed file, so there is no race on `per_seed/`. The two dependent jobs never overlap in wall time (afterany), so there is no race between jobs either.

The existing `phase2b_raw/run_config.shard0.json` written by Job 480887 has `shard_count=1`. The resume's shard 0 overwrites it with `shard_count=8`. This is the intended behaviour; the old file represents a run that produced no seed outputs.

## 4. Why this resume is scientifically valid

1. **Paired protocol is preserved.** Each seed's Phase 2b base trajectory is generated from that seed (line 242 of `run_phase2b_proseco_owt.py`: `evaluate_schedule(..., seed=seed)`). MC samples use `seed=int(seed)` for the pipeline call (line 269) so base is paired across MC samples at a given seed. Sharding does not change the within-seed pairing.
2. **Protocol A's Δ_t trace and signal trace are identical across shards.** Every shard reads the same 8 trajectory files to build `by_seed` and `mean_profile` (line 414–415). So `mean_delta_oracle` and `mean_profile_signal` policies see the same mean profile on every shard, and `per_trajectory_signal` policies correctly use each seed's own trace.
3. **`--resume` safety.** Line 452 skips a seed only if both its per_seed files already exist. Since the prior Job 480887 never wrote any per_seed file (verified empty at 2026-04-22 20:40), no shard will inappropriately skip a seed on first submission. On a second submission after a timeout, `--resume` correctly skips any seed that completed.
4. **No seed-ownership collision.** The modulo split is deterministic given `seed_start=142`, `K=8`, `shard_count=8`, so seed `142 + i` is owned uniquely by shard `i`. The two dependent jobs (481264, 481265) are disjoint in both seeds and time.
5. **Backend auto-detection is pinned.** The resume passes `--backend proseco_llada_sft` explicitly (not `auto`), so no checkpoint-sniffing race.
6. **Analysis script compatibility unchanged.** The original sbatch's inline aggregator glob (`policy_raw.shard*-of-*.json`) already expects sharded outputs. Running the 8-way sharded resume produces `policy_raw.shard{0..7}-of-8.json`, which the aggregator will concatenate into `policy_raw.json` correctly. `analyze_phase2b.py` then reads `policy_raw.json` / `mc_raw.json` as designed.

## 5. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Stale writes from Job 480887 after new shard 0 starts | High in principle | `scancel 480887` issued and confirmed before submitting (squeue empty; `phase2b_raw/per_seed/` empty → nothing lost). |
| Duplicated work if a shard is re-submitted after completing some seeds | Low | `--resume` is on; duplicates are skipped (line 452). |
| `manifest.json` last-writer-wins across shards | Low (cosmetic) | Per-shard `run_config.shard{i}.json` and per-shard `policy_raw.shard{i}-of-8.json` / `mc_raw.shard{i}-of-8.json` are the authoritative per-shard provenance. |
| Per-seed cost on LLaDA-SFT larger than expected | Medium | One seed per shard + 23:50 h walltime gives >2× my point estimate of per-seed cost (~11–15 h). If a shard still times out, its completed seed remains on disk; a single re-submission with `--resume` is cheap. |
| `shard_count=1 → shard_count=8` config mismatch in `run_config.shard0.json` | Low | Overwritten by resume's shard 0; logged in sbatch preamble. No downstream consumer depends on the old file. |
| Aggregation script glob not finding shard0-of-1 (legacy from 480887) | None | Glob `policy_raw.shard*-of-*.json` only matches files Phase 2b writes; 480887 wrote `run_config.shard0.json` but never `policy_raw.shard0-of-1.json`, so there is no legacy shard output to clash with. |
| Job B (481265) launching before Job A (481264) finishes | None | Submitted with `--dependency=afterany:481264`; SLURM enforces the ordering. |

## 6. What this plan does **not** authorise

- Running Phase 3a on LLaDA-SFT.
- Modifying Protocol A outputs in any way.
- Any write outside `results/cross_backbone/proseco_llada_sft_bounded/phase2b_raw/` (plus the standard out/err SLURM logs).
- A second resume attempt without confirming the reason for the first timeout.

## 7. Trigger for next decision

Once all four shards write their `policy_raw.shard{i}-of-4.json` and `mc_raw.shard{i}-of-4.json` (or fail cleanly), run `analyze_phase2b.py` locally (CPU-cheap), then write `docs/thesis/next_steps/POST_CROSS_BACKBONE_DECISION.md` and update `CROSS_BACKBONE_REPLICATION_RESULTS.md`. No canonical doc is touched until then.

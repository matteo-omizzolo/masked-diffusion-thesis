# Results Index

> **Current source of truth.** Updated 2026-05-08.
> Maps result folders to experiment phases. Raw data preserved in-place.

---

## Reproducibility gate outputs

> These folders are **not canonical thesis evidence**. They confirmed the backend
> pipeline was reproducible before K=30 replication was launched.

| Folder | Gate | Status | Key files |
|---|---|---|---|
| `results/phase0_smoke_d4edc92/` | Phase 0 K=3 smoke (git d4edc92, 2026-05-06, job 489457 gnode01 A100) | **Reproducibility gate only — not thesis evidence** | `smoke_config.json`, `preflight_status.txt`, `phase2b_raw/`, `phase3a_raw/` |

Qualitative match confirmed: MC headroom > 0 at B∈{2,4}; separable rankers fail;
CD-G / BS-AG recover headroom. K=30 gate is now closed; Phase 1 interaction
diagnostics (Gate 3) is the current active gate. Do not cite K=3 smoke
numbers as thesis evidence.

## K=30 replication gate outputs

| Folder | Gate | Status | Key files |
|---|---|---|---|
| `results/phase2b_k30_rep_cf89e00/` | Phase 2b K=30 replication (job 490106, gnode01, completed 2026-05-07) | **Gate evidence / robustness check** | `policy_raw.json` (1500 rows), `mc_raw.json` (9000 rows), 60 per-seed files |

Replication headroom: +0.385/+0.355/+0.380 at B = 2/3/4 (avg ≈ +0.37).
Canonical headroom from job 479581 remains +0.451/+0.441/+0.450 (avg ≈ +0.45).
The replication confirms the qualitative story with a slightly lower headroom
value. The canonical +0.45 (job 479581) remains the thesis denominator; the
+0.37 replication is a robustness check only. Tested separable rankers fail in
both, and `mean_delta_oracle` recovers only 13–29 % of replicated headroom.

---

## Canonical results folders

| Folder | Phase | Status | Key files |
|---|---|---|---|
| `results/phase1_proseco_owt_full/` | Phase 1 Protocol A | **Canonical** | `protocol_a/trajectory_*.json` (50 seeds × T=64 per-step Δ_t + signals) |
| `results/phase2b_proseco_owt/` | Phase 2b raw | **Canonical** | `per_seed/{policy,mc}_rows_seed*.json` (K=30 paired seeds) |
| `results/phase2b/` | Phase 2b aggregated | **Canonical** | `policy_comparison_paired.json`, `mc_oracle.json`, `combinatorial_diagnostics.json`, `theorem_a_constants.json` |
| `results/phase3a_proseco_owt/` | Phase 3a raw | **Canonical primary positive result** | `per_seed/{cd,bs}_rows_seed*.json`, `cd_raw.json`, `bs_raw.json` |
| `results/cross_backbone/proseco_llada_sft_bounded/` | Cross-backbone | **Tier 3 only** | `phase2b/` and `phase2b_raw/` sub-folders |
| `results/protocol_c_owt/` | Protocol C | **Honest negative** | `protocol_c_summary.json` |

---

## Phase 1 detail

```
results/phase1_proseco_owt_full/
  protocol_a/
    trajectory_seed{000..049}.json   # per-step Δ_t, H_t, M_t, QM_t (raw key: Q_t)
  protocol_b/                        # (Protocol B pairwise diagnostics)
  policy_comparison/                 # Phase 2a pilot (superseded by Phase 2b)
  summary.json                       # Phase 1 aggregate
  run_config.json
```

---

## Phase 2b detail

```
results/phase2b_proseco_owt/
  per_seed/
    policy_rows_seed{000..029}.json  # 10 policies × B∈{2,3,4,8,16}
    mc_rows_seed{000..029}.json      # 100 MC schedules × B∈{2,3,4}
  policy_raw.json / shards           # full policy raw (4 shards)
  mc_raw.json / shards               # full MC raw (4 shards)
  run_config.shard*.json

results/phase2b/
  policy_comparison_paired.json      # paired diffs, BCa CIs for all policies × B
  mc_oracle.json                     # best-of-100 MC per seed × B; paired headroom
  combinatorial_diagnostics.json     # Jaccard diagnostics (top-10 ∩ oracle, etc.)
  theorem_a_constants.json           # σ_ξ, ρ, σ_Δ, γ, (1−|ρ|)·σ_Δ at B∈{2,3,4,8,16}; the latter columns are A′/A″ diagnostics, not theorem constants
```

---

## Phase 3a detail

```
results/phase3a_proseco_owt/
  per_seed/
    cd_rows_seed{42..71}.json        # CD-G per-seed results
    bs_rows_seed{42..71}.json        # BS-AG per-seed results
  cd_raw.json                        # CD-G aggregate raw rows; primary anchor
  bs_raw.json                        # BS-AG aggregate raw rows; primary anchor
  cd_raw.shard{0..3}-of-4.json      # per-shard raw outputs
  bs_raw.shard{0..3}-of-4.json
  manifest.json
  run_config.shard*.json
```

Phase 3a canonical job 479941 produced 60 per-seed files on 2026-04-20.
`cd_raw.json` is 49,890 bytes and `bs_raw.json` is 30,399 bytes. Recovery versus
canonical +0.45 headroom is B=2 CD-G 78.9 % / BS-AG 64.1 %, B=3 CD-G 74.1 % /
BS-AG 57.1 %, and B=4 CD-G 84.3 % / BS-AG 48.8 % (p < 0.001). Recovery versus
the Phase 2b K=30 replication denominator (~+0.37) is B=2 CD-G 93.1 % / BS-AG
75.8 %, B=3 CD-G 85.5 % / BS-AG 64.3 %, and B=4 CD-G 98.2 % / BS-AG 56.2 %.
The K=30 gate does not depend on a separate `oracle_gap_closure.json` artifact.

---

## Cross-backbone detail

```
results/cross_backbone/
  proseco_llada_sft_bounded/
    phase2b/
      policy_comparison_paired.json  # Tier 3 (K=8): uniform-not-beaten
      mc_oracle.json                 # Tier 3: MC-oracle headroom CI includes 0
    phase2b_raw/per_seed/            # Raw per-seed data
```

---

## Protocol C detail

```
results/protocol_c_owt/
  protocol_c_summary.json            # eps_ratio, close_ratio, verdict: honest_negative
```

---

## Figures

Canonical thesis-candidate figures are in `figures/`. See `figures/README.md` for the
index. All are generated from the canonical result folders above.

Primary result figure: `figures/phase3a/oracle_gap_closure.{png,pdf}`. The
figure name is historical; the result anchor is the Phase 3a raw / paired files
listed above.

## Legacy / archived results

There are no legacy result folders in `results/`. Active result folders are either
canonical thesis evidence or explicitly labeled gate / robustness outputs above.
Raw data is preserved; no deletions.

Abandoned backend results (Phase 1 on MDLM/ReMDM, near-zero Δ_t) were in
`archive/legacy_framework/results/` and removed in the May 2026 cleanup.
Git history preserves them.

---

## What the numbers mean (quick key)

- **G(S)** = F(y^S) − F(y_base) = paired quality gain from schedule S (F = −GPT-2 NLL).
- **MC-oracle +0.45** = best-of-100 random schedules beats uniform by +0.45 at K=30;
  this is not the exhaustive (T choose B) oracle.
- **Recovery rate** = G(search_procedure) / MC-oracle headroom. Use canonical
  +0.45 as the thesis denominator; report the +0.37 replication denominator only
  as a robustness comparison.
- **NULL band** = region where paired CI includes 0 (no detectable gain over uniform).
- **T1** = BCa bootstrap 95 % CI excludes 0, K ≥ 30.

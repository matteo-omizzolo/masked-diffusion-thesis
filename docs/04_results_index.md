# Results Index

> **Current source of truth.** Updated 2026-05-05.
> Maps result folders to experiment phases. Raw data preserved in-place.

---

## Canonical results folders

| Folder | Phase | Status | Key files |
|---|---|---|---|
| `results/phase1_proseco_owt_full/` | Phase 1 Protocol A | **Canonical** | `protocol_a/trajectory_*.json` (50 seeds × T=64 per-step Δ_t + signals) |
| `results/phase2b_proseco_owt/` | Phase 2b raw | **Canonical** | `per_seed/{policy,mc}_rows_seed*.json` (K=30 paired seeds) |
| `results/phase2b/` | Phase 2b aggregated | **Canonical** | `policy_comparison_paired.json`, `mc_oracle.json`, `combinatorial_diagnostics.json`, `theorem_a_constants.json` |
| `results/phase3a_proseco_owt/` | Phase 3a raw | **Canonical** | `per_seed/{cd,bs}_raw_seed*.json` |
| `results/phase3a_proseco_owt/oracle_gap_closure.json` | Phase 3a summary | **Primary positive result** | Recovery rates: CD-G 74–84 %, BS-AG 49–64 % at B∈{2,3,4} |
| `results/cross_backbone/proseco_llada_sft_bounded/` | Cross-backbone | **Tier 3 only** | `phase2b/` and `phase2b_raw/` sub-folders |
| `results/protocol_c_owt/` | Protocol C | **Honest negative** | `protocol_c_summary.json` |

---

## Phase 1 detail

```
results/phase1_proseco_owt_full/
  protocol_a/
    trajectory_seed{000..049}.json   # per-step Δ_t, H_t, M_t, Q_t
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
    cd_raw_seed{000..029}.json       # CD-G per-seed results
    bs_raw_seed{000..029}.json       # BS-AG per-seed results
  cd_paired.json / shards
  bs_paired.json / shards
  oracle_gap_closure.json            # PRIMARY: recovery rates, CIs
  manifest.json
  run_config.shard*.json
```

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

Primary result figure: `figures/phase3a/oracle_gap_closure.{png,pdf}`.

## Legacy / archived results

There are no legacy result folders in `results/`. All result folders are canonical and
correspond to completed experiment phases. Raw data is preserved; no deletions.

Abandoned backend results (Phase 1 on MDLM/ReMDM, near-zero Δ_t) were in
`archive/legacy_framework/results/` and removed in the May 2026 cleanup.
Git history preserves them.

---

## What the numbers mean (quick key)

- **G(S)** = F(y^S) − F(y_base) = paired quality gain from schedule S (F = −GPT-2 NLL).
- **MC-oracle +0.45** = best-of-100 random schedules beats uniform by +0.45 at K=30.
- **Recovery rate** = G(search_procedure) / MC-oracle +0.45 headroom.
- **NULL band** = region where paired CI includes 0 (no detectable gain over uniform).
- **T1** = BCa bootstrap 95 % CI excludes 0, K ≥ 30.

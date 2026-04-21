# Log Cleanup Audit

*Written: 2026-04-18.*
*Scope: `out/` and `err/` directories.*

---

## 1. Pre-cleanup inventory (43 files in each directory)

| Group | Files | Scientific relevance |
|-------|-------|---------------------|
| `remdm_smoke_*` (17 pairs) | Smoke tests for ReMDM environment | None — pre-Phase-1, no results |
| `remdm_eval_*` (6 pairs) | ReMDM step-sweep evaluations | Pre-pivot; results archived in `results/` |
| `remdm_sweep_*` / `remdm_t1000p_*` (2 pairs) | Legacy step-sweep jobs | Pre-pivot |
| `phase1_pilot_478463` (1 pair) | Failed pilot (early) | Superseded by 478600 |
| `phase1_pilot_478592-478597` (6 pairs) | Failed/repeated pilot attempts | Superseded by 478600 |
| `phase1_proseco_478826-828` (3 pairs) | Failed ProSeCo backend jobs | Root cause documented in RESULTS_STATUS.md §4 |
| `phase1_pilot_478600` (1 pair) | **NEGATIVE RESULT (keep)** | MDLM heuristic diagnostic — scientifically relevant |
| `phase1_proseco_478929` (1 pair) | **STRUCTURAL NO-OP (keep)** | ProSeCo on mdlm.ckpt — scientifically relevant |
| `phase1_mdlm_conf_478962` (1 pair) | **VALID PILOT (keep)** | MDLM-conf pilot (pre-signal-fix) |
| `phase1_mdlm_conf_479257` (1 pair) | **SIGNAL-ALIGNED PILOT (keep)** | Primary MDLM-conf result |

---

## 2. Archived (2026-04-18)

All 39 obsolete log pairs (78 files) moved to:
- `archive/logs/out/` — stdout files
- `archive/logs/err/` — stderr files

### Archived groups:

| Group | Pairs | Reason |
|-------|-------|--------|
| `remdm_smoke_*` (17 pairs) | Pre-Phase-1 environment smoke tests; all superseded |
| `remdm_eval_*` (6 pairs) | Pre-pivot ReMDM step-sweep logs; results in `results/sweep/` |
| `remdm_sweep_467809` / `remdm_t1000p_465445` (2 pairs) | Pre-pivot legacy |
| `phase1_pilot_478463` (1 pair) | Early failed pilot; 478600 is the canonical diagnostic |
| `phase1_pilot_478592-478597` (6 pairs) | Failed repeated pilots before 478600 succeeded |
| `phase1_proseco_478826-828` (3 pairs) | Failed ProSeCo backend; root cause in RESULTS_STATUS.md §4 |

---

## 3. Retained in `out/` and `err/`

| File pair | Reason to keep |
|-----------|---------------|
| `phase1_pilot_478600` | Canonical MDLM heuristic diagnostic (all Δ_t ≤ 0) |
| `phase1_proseco_478929` | Canonical ProSeCo structural no-op (all Δ_t = 0) |
| `phase1_mdlm_conf_478962` | MDLM-conf pilot with M2 bug (for comparison with 479257) |
| `phase1_mdlm_conf_479257` (on HPC only) | Signal-aligned MDLM-conf pilot — canonical positive result |

Note: job 479257 logs live on HPC at `out/phase1_mdlm_conf_479257.out`. They were
not downloaded locally (large file; relevant metrics are in summary.json).

---

## 4. Future log policy

For all new Phase 1 ProSeCo-OWT runs:
- Keep logs for completed runs (any job that produces a summary.json)
- Archive logs for failed jobs once failure is documented in RESULTS_STATUS.md
- Do not retain smoke/sanity logs beyond the immediate debugging session

Run this audit again after the Phase 1 ProSeCo-OWT pilot completes.

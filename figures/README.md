# Figures

Canonical result figures for the thesis. All generated from `results/` data.
Current status starts at `../START_HERE.md`.

| Folder | Source | Contents |
|---|---|---|
| `phase2b/` | `results/phase2b_proseco_owt/`, `results/phase2b/` | 72 plots: paired-diff vs uniform, policy comparison, Jaccard diagnostics |
| `phase3a/` | `results/phase3a_proseco_owt/` | 2 plots: `oracle_gap_closure.{png,pdf}` — primary result |
| `cross_backbone/` | `results/cross_backbone/` | 33 plots: LLaDA-SFT bounded probe (Phase 2b-equivalent) |

None are currently referenced in LaTeX. Figures for ch5 will be selected from these
during writing. To regenerate: `python scripts/analyze_phase2b.py` / `analyze_phase3a.py`.

Legacy figures from abandoned Phase 1 (MDLM/ReMDM) and Phase 2a were deleted
in the May 2026 cleanup. Git history preserves them.

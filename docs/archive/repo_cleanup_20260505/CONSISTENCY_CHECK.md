# Active Docs Consistency Check — Second Pass (2026-05-05)

> All items below checked after second-pass edits.

---

## Thesis story

| Claim | START_HERE | 00_status | 01_direction | 02_experiments | 03_theory | 05_next_steps | CLAUDE.md |
|---|---|---|---|---|---|---|---|
| Rankers fail; search works | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| CD-G 74–84 %, BS-AG 49–64 % | ✓ | ✓ | — | ✓ | — | — | — |
| No new HPC authorized | ✓ | ✓ | — | — | — | ✓ | ✓ |
| Critical path: LaTeX writing | ✓ | ✓ | — | — | — | ✓ | ✓ |
| Theory complete (A, A′, A″, Corollary) | ✓ | ✓ | ✓ | — | ✓ | ✓ | ✓ |
| Protocol C honest negative | ✓ | ✓ | — | ✓ | ✓ | — | ✓ |
| Single-backbone scope caveat | ✓ | ✓ | ✓ | ✓ | — | — | — |

No contradictions found. ✓

---

## Active doc list

Listed in `docs/README.md`:
- `START_HERE.md` ✓
- `docs/00_current_status.md` ✓
- `docs/01_research_direction.md` ✓
- `docs/02_experiments.md` ✓
- `docs/03_theory.md` ✓
- `docs/04_results_index.md` ✓
- `docs/05_next_steps.md` ✓
- `research/candidate_theorems.md` ✓ (supporting reference)
- `research/proof_worklog.md` ✓ (supporting reference)

All listed; none are missing or extra. ✓

---

## Archive rule

Same rule present in: `START_HERE.md`, `docs/README.md`, `CLAUDE.md`, `README.md`,
`archive/README.md` (root), `docs/archive/README.md`. ✓

---

## Navigation burden

| File | Cross-links to other active docs | OK? |
|---|---|---|
| `START_HERE.md` | Pointers to 8 docs max; readable in 2 min | ✓ |
| `docs/README.md` | Index only; no summaries | ✓ |
| `README.md` | One pointer (START_HERE.md) | ✓ |
| `CLAUDE.md` | Table of 10 files; operational HPC info | ✓ |
| `docs/00_current_status.md` | No links to other active docs | ✓ |
| `docs/02_experiments.md` | Links to results/ folders and scripts | ✓ |
| `docs/03_theory.md` | Points to research/candidate_theorems.md | ✓ |

No pointer chains deeper than 2 hops. ✓

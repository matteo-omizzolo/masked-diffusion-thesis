> ARCHIVED — historical only.
> Do not use this file to infer the current thesis status unless explicitly asked.
> Current source of truth: see `START_HERE.md` and `docs/README.md`.

---


# DOCS Reorganization Plan — 2026-04-22

**Companion to:** `DOCS_REORGANIZATION_AUDIT.md` (read first).

**Intent.** Translate Audit §B (classification) + §D (target IA) + §E (rules)
into a concrete, ordered, reversible command list. Every step is a single
`git mv` (rename-tracked) or a bounded edit; every step is verifiable.

**Execution discipline.** This plan is *not* executed in this file. It is
executed in Phase 3 of the reorganization (a separate work block) so that
the audit and the plan are reviewable without entangling the moves.

**Non-negotiable constraints.**

- Job 480887 (LLaDA-SFT cross-backbone) is running. Do **not** move any
  file referenced by `hpc/cross_backbone_proseco_llada_sft_bounded.sbatch`
  or by `CROSS_BACKBONE_IMPLEMENTATION_AUDIT.md` until the job reaches a
  terminal state (COMPLETED or FAILED).
- Every step finishes with a link-hygiene check. A broken link is a
  blocker.

---

## 0. Pre-conditions

Before any step runs:

1. `git status` must be clean on tracked files inside `docs/`. Untracked
   files we created are allowed (this PLAN, the AUDIT).
2. `bash` + `git` + `grep` available on host (mac: coreutils fine;
   alternatively `rg`).
3. No concurrent HPC job referring to moved paths. Current in-flight Job
   480887 does **not** reference `docs/` paths, so it is not a blocker
   for steps that do not touch `docs/thesis/next_steps/CROSS_BACKBONE_*`.
4. A short-lived feature branch: `git switch -c reorg/docs-2026-04-22`
   (the user will merge to main only after Phase 5 integrity check
   passes).

---

## 1. Step group A — archive directory skeleton

Create the archive tree once, then populate.

```bash
mkdir -p docs/archive/audits
mkdir -p docs/archive/phase1_era
mkdir -p docs/archive/operational
mkdir -p docs/archive/chronicles
mkdir -p docs/archive/status
mkdir -p docs/thesis/appendix/framework_memos
```

**Verify:** `find docs/archive docs/thesis/appendix -type d` lists all six
directories.

---

## 2. Step group B — REMOVE

One file, zero ambiguity.

```bash
git rm docs/maintenance/MOVED.md
rmdir docs/maintenance   # only if empty
```

**Verify:**
- `ls docs/maintenance 2>/dev/null || echo ok` prints `ok`.
- `grep -rn "docs/maintenance/MOVED" -- . ':!docs/thesis/maintenance/DOCS_REORGANIZATION_*'`
  prints no matches.

---

## 3. Step group C — ARCHIVE (20 files)

Every move is `git mv <old> <new>`; after each move, prepend a banner.

### 3.1 Audits (3 files)

| From | To |
|---|---|
| `docs/thesis/experiments/EXPERIMENT_CRITICAL_AUDIT.md` | `docs/archive/audits/EXPERIMENT_CRITICAL_AUDIT.md` |
| `docs/thesis/theory/THEORY_STRESS_TEST.md` | `docs/archive/audits/THEORY_STRESS_TEST.md` |
| `docs/experiments/results/proseco_backend_failure_audit.md` | `docs/archive/audits/proseco_backend_failure_audit.md` |

### 3.2 Phase-1-era (10 files)

| From | To |
|---|---|
| `docs/implementation_plan.md` | `docs/archive/phase1_era/implementation_plan.md` |
| `docs/experiments/implementation_status.md` | `docs/archive/phase1_era/implementation_status.md` |
| `docs/experiments/phase1_interpretation.md` | `docs/archive/phase1_era/phase1_interpretation.md` |
| `docs/experiments/entropy_proxy_experiment.md` | `docs/archive/phase1_era/entropy_proxy_experiment.md` |
| `docs/experiments/proseco_import_audit.md` | `docs/archive/phase1_era/proseco_import_audit.md` |
| `docs/experiments/proseco_backend_audit.md` | `docs/archive/phase1_era/proseco_backend_audit.md` |
| `docs/experiments/proseco_experiment_definition.md` | `docs/archive/phase1_era/proseco_experiment_definition.md` |
| `docs/experiments/proseco_protocol_mapping.md` | `docs/archive/phase1_era/proseco_protocol_mapping.md` |
| `docs/experiments/proseco_validation_checklist.md` | `docs/archive/phase1_era/proseco_validation_checklist.md` |
| `docs/experiments/results/result_inventory.md` | `docs/archive/phase1_era/result_inventory.md` |

After: `rmdir docs/experiments/results docs/experiments` (if empty).

### 3.3 Operational (4 files)

| From | To |
|---|---|
| `docs/thesis/maintenance/LOG_CLEANUP_AUDIT.md` | `docs/archive/operational/LOG_CLEANUP_AUDIT.md` |
| `docs/thesis/maintenance/NEXT_CLAUDE_CODE_PROMPT.md` | `docs/archive/operational/NEXT_CLAUDE_CODE_PROMPT.md` |
| `docs/thesis/maintenance/hpc_sync_inventory.md` | `docs/archive/operational/hpc_sync_inventory.md` |
| `docs/thesis/maintenance/hpc_sync_report.md` | `docs/archive/operational/hpc_sync_report.md` |

### 3.4 Chronicles (1 file)

| From | To |
|---|---|
| `docs/thesis/experiments/RESULTS_STATUS.md` | `docs/archive/chronicles/RESULTS_STATUS.md` |

### 3.5 Status (1 file, LaTeX)

| From | To |
|---|---|
| `docs/pdf/status/phase2_status_report_2026-04-19.tex` | `docs/archive/status/phase2_status_report_2026-04-19.tex` |

Keep `docs/pdf/` (PDF output directory) untouched — it may still be
populated by `tectonic`; just the one historical `.tex` moves.

### 3.6 Banner insertion

After moving, each archived file gets the following banner prepended
(keeping existing title as H1 immediately below):

```markdown
> **STATUS:** ARCHIVED
> **ARCHIVED ON:** 2026-04-22
> **SUPERSEDED BY:** <canonical path, per DOCS_REORGANIZATION_AUDIT.md §B.4>
> **REASON:** <one-line reason>

---
```

**Banner map** (one row per archive file — target listed verbatim):

| Archived file | Superseded by | Reason |
|---|---|---|
| EXPERIMENT_CRITICAL_AUDIT.md | `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md`, `docs/thesis/theory/THEORY_STATUS.md` | Phase 1 audit; its findings now encoded in canonical post-3a docs |
| THEORY_STRESS_TEST.md | `docs/thesis/theory/THEORY_STATUS.md` §"Candidate Refinements" | Stress-test conclusions encoded in Refinements A′/A″ |
| proseco_backend_failure_audit.md | `docs/experimental_infrastructure.md` | Resolved incident |
| implementation_plan.md | `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md` | Phase-1 plan; scope long superseded |
| implementation_status.md | `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md` | Phase-1 status |
| phase1_interpretation.md | `docs/archive/audits/EXPERIMENT_CRITICAL_AUDIT.md` | Same interpretation, later audit carries the verdict |
| entropy_proxy_experiment.md | `docs/thesis/CANONICAL_EXPERIMENT_OVERVIEW.md` | Pre-Phase-2b spec variant |
| proseco_import_audit.md | `docs/experimental_infrastructure.md` | Resolved import |
| proseco_backend_audit.md | `docs/experimental_infrastructure.md` | Resolved backend readiness |
| proseco_experiment_definition.md | `docs/thesis/CANONICAL_EXPERIMENT_OVERVIEW.md` | Pre-Phase-2b spec variant |
| proseco_protocol_mapping.md | `docs/thesis/CANONICAL_EXPERIMENT_OVERVIEW.md` | Knob-mapping superseded |
| proseco_validation_checklist.md | `docs/thesis/CANONICAL_EXPERIMENT_OVERVIEW.md` | Executed pre-launch checklist |
| result_inventory.md | `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md` + `docs/archive/chronicles/RESULTS_STATUS.md` §14 | Phase-1 inventory |
| LOG_CLEANUP_AUDIT.md | `docs/thesis/maintenance/CLEANUP_LOG.md` | Operational snapshot |
| NEXT_CLAUDE_CODE_PROMPT.md | n/a (completed) | Sonnet prompt, rerun completed |
| hpc_sync_inventory.md | n/a (operational snapshot) | HPC sync snapshot |
| hpc_sync_report.md | n/a (operational snapshot) | HPC sync log |
| RESULTS_STATUS.md | `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md` + `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md` | Canonical content split into results + direction |
| phase2_status_report_2026-04-19.tex | `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md` | Phase-2 launch snapshot; results landed |

**Verify after 3.1–3.6:**
- `find docs/archive -type f | wc -l` = 20
- `head -6 docs/archive/audits/EXPERIMENT_CRITICAL_AUDIT.md` shows the
  banner.

---

## 4. Step group D — MERGE (3 operations)

Merges run after archives so the merge targets are stable.

### 4.1 Merge `PHASE3_ALTERNATIVE_PLAN.md` → `PHASE3A_COMBINATORIAL_RESULTS.md`

- Append `PHASE3_ALTERNATIVE_PLAN.md` content under a new section
  `## Appendix Z. Phase 3 plan (superseded)` in
  `PHASE3A_COMBINATORIAL_RESULTS.md`.
- Drop the redundant Phase-3a-results addendum from the appendix (already
  canonical upstream in the same file).
- `git mv docs/thesis/experiments/PHASE3_ALTERNATIVE_PLAN.md docs/archive/phase1_era/PHASE3_ALTERNATIVE_PLAN.md`
  then prepend the `ARCHIVED` banner pointing to
  `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md`.

**Verify:**
- `grep -c "Appendix Z" docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md`
  ≥ 1.
- `ls docs/thesis/experiments/PHASE3_ALTERNATIVE_PLAN.md 2>/dev/null || echo ok`
  prints `ok`.

### 4.2 Merge `NEXT_THEORY_STEPS.md` → `THEORY_STATUS.md`

- Append the three theorem-proof tasks as a new section
  `## Proof work queue` at the end of `THEORY_STATUS.md`.
- Verify no numerical claim is introduced that isn't already in
  `THEORY_STATUS.md` (it's a proof task list — tasks, not numbers).
- `git mv docs/thesis/theory/NEXT_THEORY_STEPS.md docs/archive/operational/NEXT_THEORY_STEPS.md`
  with `ARCHIVED` banner pointing to `docs/thesis/theory/THEORY_STATUS.md
  §Proof work queue`.

**Verify:**
- `grep -c "Proof work queue" docs/thesis/theory/THEORY_STATUS.md` = 1.

### 4.3 Merge `PRINCIPLED_NEXT_STEPS_PLAN.md` → `CROSS_BACKBONE_REPLICATION_PLAN.md`

- Append `PRINCIPLED_NEXT_STEPS_PLAN.md` content as
  `## Appendix Z. Superseded cost-benefit ranking` in
  `CROSS_BACKBONE_REPLICATION_PLAN.md`.
- Add a reconciling note at the top of
  `CROSS_BACKBONE_REPLICATION_PLAN.md` stating: "This plan postdates and
  supersedes `PRINCIPLED_NEXT_STEPS_PLAN.md` on the specific question of
  whether to run bounded LLaDA-SFT cross-backbone. The canonical scope
  (`CANONICAL_RESEARCH_DIRECTION.md`) treats the run as an *optional
  appendix experiment*, not a thesis-core requirement."
- `git mv docs/thesis/next_steps/PRINCIPLED_NEXT_STEPS_PLAN.md docs/archive/operational/PRINCIPLED_NEXT_STEPS_PLAN.md`
  with `ARCHIVED` banner.

**Verify:**
- `grep -c "Appendix Z" docs/thesis/next_steps/CROSS_BACKBONE_REPLICATION_PLAN.md`
  ≥ 1.
- `grep "supersedes \`PRINCIPLED_NEXT_STEPS_PLAN" docs/thesis/next_steps/CROSS_BACKBONE_REPLICATION_PLAN.md`
  returns 1 match.

---

## 5. Step group E — framework memos

Memo files contain a literal space in the directory name (`docs/future
ideas/`). Move + rename in one operation.

```bash
git mv "docs/future ideas/Deep Research Audit of Budgeted Informed Corrector Scheduling.md" \
       "docs/thesis/appendix/framework_memos/DeepResearchAudit_BudgetedInformedCorrectorScheduling.md"
git mv "docs/future ideas/Theoretical Frameworks for Budgeted Informed-Corrector Scheduling in Masked Diffusion Language Models.md" \
       "docs/thesis/appendix/framework_memos/TheoreticalFrameworks_BudgetedInformedCorrectorScheduling.md"
rmdir "docs/future ideas"
```

Prepend audit banner to each (Rule E.6):

```markdown
> **STATUS:** APPENDIX (SPECULATIVE). Audited 2026-04-22.
> **AUDIT SOURCE:** docs/thesis/next_steps/INDEPENDENT_AUDIT_OF_DIRECTION_AND_MEMOS.md §3.<2 or 3>
> **AUDIT VERDICT:** <Contradicted-in-repo / Speculative / Plausible-but-untested — choose per memo>
> **THESIS POSITION:** Future work only; not part of thesis-core argument.

---
```

For `DeepResearchAudit…` memo: verdict = "Contradicted-in-repo on
cross-backbone priority (see §3.2 M1.2, M1.6 in the audit)."

For `TheoreticalFrameworks…` memo: verdict = "Speculative; MDP / SMC /
Feynman-Kac / Particle Gibbs framework is theoretically sound but
untested on ProSeCo-OWT (see §3.3 M2.1–M2.4)."

**Verify:**
- `ls docs/future\ ideas 2>/dev/null || echo ok` prints `ok`.
- Both memos live under `docs/thesis/appendix/framework_memos/` with
  underscored filenames.
- Both carry the banner.

---

## 6. Step group F — status banners on canonical + supporting

Prepend banners (Rule E.2) to the 11 canonical + 4 supporting files.
Banners only; content untouched.

### 6.1 CANONICAL banner template

```markdown
> **STATUS:** CANONICAL
> **LAST UPDATED:** 2026-04-22 (banner only; content last updated YYYY-MM-DD)
> **SCOPE:** <one line — e.g. "ProSeCo-OWT, T=64, F=−GPT-2 NLL on 512 tokens">

---
```

### 6.2 SUPPORTING banner template

```markdown
> **STATUS:** SUPPORTING (narrative / tracker)
> **LAST UPDATED:** 2026-04-22 (banner only)
> **CANONICAL REFERENCES:** <list of canonical docs this file links to>

---
```

### 6.3 Files to banner (15 total)

Canonical: #6 #7 #8 #14 #16 #18 #20 #24 #25 #26 #27 (per Audit §B.1).
Supporting: #1 #2 #3 #4 (per Audit §B.3).

**Do NOT banner** `docs/thesis/next_steps/CROSS_BACKBONE_IMPLEMENTATION_AUDIT.md`
(#24) during Job 480887 — see §0 pre-condition; defer until job completes.
Exception tracked explicitly; the other 14 files are bannered in this step.

**Verify:**
- `for f in <14 paths>; do head -3 "$f" | grep -c "STATUS:"; done` returns
  `1` for each.

---

## 7. Step group G — promote CURRENT_INDEX.md

Move from `docs/thesis/CURRENT_INDEX.md` → `docs/CURRENT_INDEX.md` so it
sits at the navigation entrypoint.

```bash
git mv docs/thesis/CURRENT_INDEX.md docs/CURRENT_INDEX.md
```

Edit `docs/CURRENT_INDEX.md` (formerly `docs/thesis/CURRENT_INDEX.md`) to:

1. Update all relative links: `experiments/...` → `thesis/experiments/...`,
   `theory/...` → `thesis/theory/...`, `next_steps/...` →
   `thesis/next_steps/...`, `maintenance/...` → `thesis/maintenance/...`
   (or `archive/...` per file destination in this plan).
2. Add a new top-level section "Canonical scope + navigation"
   cross-linking `CANONICAL_RESEARCH_DIRECTION.md`,
   `CANONICAL_EXPERIMENT_OVERVIEW.md`.
3. Add a section "Archive" listing every archived path with its
   superseded-by target (mechanical from §3.6 banner map).
4. Enforce Rule E.9 (no numerical claims in CURRENT_INDEX) — remove any
   paired-mean / closure-ratio lines; replace with links into the
   canonical source.

**Verify (link-hygiene after all edits):**
- `grep -n "experiments/" docs/CURRENT_INDEX.md` shows only
  `thesis/experiments/` or `archive/` prefixes.
- `grep -E "\+0\.[0-9]{2,}|[0-9]{2}%" docs/CURRENT_INDEX.md` returns zero
  matches (no numerical claims).

---

## 8. Step group H — link rewrites across the repo

This is the Rule-E.8 check. After all moves, update references
everywhere.

### 8.1 Known referrers (from 2026-04-22 scan)

- `docs/thesis_direction.md`: references several Phase-1-era files and
  `RESULTS_STATUS.md` — update to canonical successors per §3.6.
- `docs/thesis/next_steps/INDEPENDENT_AUDIT_OF_DIRECTION_AND_MEMOS.md`:
  references `docs/future ideas/*` — update to new
  `docs/thesis/appendix/framework_memos/*` paths (and to renamed
  underscored basenames). Keep audit findings verbatim.
- `docs/thesis/next_steps/ASSESSMENT_OF_MEMOS_AND_ANALYSIS.md`: same as
  above.
- `docs/thesis/next_steps/CROSS_BACKBONE_IMPLEMENTATION_AUDIT.md`: **do
  not edit during Job 480887** (see §0). Update after job completion.
- `docs/thesis/experiments/ANALYSIS_SPEC.md`: references
  `RESULTS_STATUS.md` — update to
  `docs/archive/chronicles/RESULTS_STATUS.md` with "(archived)" suffix.
- `docs/thesis/experiments/PHASE3_DIRECTION_AUDIT.md`: references
  `RESULTS_STATUS.md`, `EXPERIMENT_CRITICAL_AUDIT.md`,
  `THEORY_STRESS_TEST.md` — all three updated to archive paths.
- `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md`: references
  `RESULTS_STATUS.md`, `PHASE3_ALTERNATIVE_PLAN.md` — update.
- `docs/thesis/theory/THEORY_STATUS.md`: references `RESULTS_STATUS.md`
  §12.3 and `THEORY_STRESS_TEST.md` §10 — update to archive paths.
- `README.md` (repo root): references at least
  `PHASE3A_COMBINATORIAL_RESULTS.md`. Scan and update any references to
  moved paths.
- `.github/copilot-instructions.md`: references
  `PHASE3A_COMBINATORIAL_RESULTS.md`. Scan and update.
- `research/candidate_theorems.md`,
  `research/proof_worklog.md`,
  `research/open_questions.md`: all reference
  `THEORY_STRESS_TEST.md` / `RESULTS_STATUS.md`. Update links; content
  stays.
- `scripts/run_phase3a_combinatorial.py`,
  `scripts/analyze_phase3a.py`: may reference
  `PHASE3A_COMBINATORIAL_RESULTS.md` / `PHASE3_ALTERNATIVE_PLAN.md` in
  docstrings. Audit and update docstrings only; do **not** change
  executable code.

### 8.2 Hygiene gate

At end of step group H, execute and interpret:

```bash
# Archive paths must NOT appear unversioned as pointers
grep -rn "docs/experiments/" -- . ':!docs/archive' ':!docs/thesis/maintenance/DOCS_*'
grep -rn "docs/maintenance/MOVED" -- .
grep -rn "docs/future ideas" -- .
grep -rn "docs/thesis/experiments/RESULTS_STATUS.md" -- . ':!docs/thesis/maintenance/DOCS_*'
grep -rn "docs/thesis/experiments/EXPERIMENT_CRITICAL_AUDIT.md" -- . ':!docs/thesis/maintenance/DOCS_*'
grep -rn "docs/thesis/theory/THEORY_STRESS_TEST.md" -- . ':!docs/thesis/maintenance/DOCS_*'
grep -rn "docs/thesis/experiments/PHASE3_ALTERNATIVE_PLAN.md" -- . ':!docs/thesis/maintenance/DOCS_*'
grep -rn "docs/thesis/theory/NEXT_THEORY_STEPS.md" -- . ':!docs/thesis/maintenance/DOCS_*'
grep -rn "docs/thesis/next_steps/PRINCIPLED_NEXT_STEPS_PLAN.md" -- . ':!docs/thesis/maintenance/DOCS_*'
grep -rn "docs/thesis/CURRENT_INDEX.md" -- . ':!docs/thesis/maintenance/DOCS_*'
```

Each command must return zero lines outside the exclusion set. Any hit is
a blocker; fix before closing Phase 3.

(Exclusion `docs/thesis/maintenance/DOCS_*` means "this plan file and the
audit file" — they name the old paths on purpose.)

---

## 9. Step group I — README + CURRENT_INDEX refresh (Phase 4)

Executed after step groups A–H pass. Covered by Phase 4 of the
reorganization, not this plan. Key deliverables flagged:

- `README.md` top-level section "Docs entry point" points to
  `docs/CURRENT_INDEX.md`.
- `README.md` "Status" section cites `CANONICAL_RESEARCH_DIRECTION.md`,
  `PHASE3A_COMBINATORIAL_RESULTS.md`, `THEORY_STATUS.md` (no numerical
  claims inside README — per Rule E.9 applied consistently).
- `docs/CURRENT_INDEX.md` references every CANONICAL doc by one-line hook.

---

## 10. Step group J — scientific-integrity verification (Phase 5)

Final gate. Executed after Phase 4. No moves or edits here, just checks.

### 10.1 Single-source (Rule E.1) verification

For each load-bearing fact listed in Audit §C.1–§C.8, verify: exactly one
canonical source and N-1 reference-only mentions.

Checklist:

- **Protocol A/B** restated only in
  `docs/thesis/CANONICAL_EXPERIMENT_OVERVIEW.md`.
- **Theorem A full statement** only in
  `docs/thesis/theory/THEORY_STATUS.md`.
- **Core research question** verbatim in `CLAUDE.md` and
  `CANONICAL_RESEARCH_DIRECTION.md`; `thesis_direction.md` quotes #6
  verbatim.
- **Phase 3a closure numbers** only in
  `PHASE3A_COMBINATORIAL_RESULTS.md` (and its merged appendix).
- **Phase 2b MC-oracle numbers** only in
  `PHASE3A_COMBINATORIAL_RESULTS.md` Appendix A (or successor structure
  per Audit §C.5 decision) and the source JSON.
- **Combinatorial-diagnostics numbers** only in
  `INDEPENDENT_AUDIT_OF_COPILOT_IMPLEMENTATION.md` §3 + source JSON.
- **PRISM rejection rationale** only in `PHASE3_DIRECTION_AUDIT.md`,
  with pointers from #6 and #20.
- **Cross-backbone scope** coherent: #6 = "parked", #25 = "bounded
  appendix GO", #28 archived.

### 10.2 Ranker-vs-search invariant (Rule E.3)

Run:

```bash
grep -n "greedy/separable-ranker\|ranker class\|search class\|search procedure" \
  docs/thesis/CANONICAL_RESEARCH_DIRECTION.md \
  docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md \
  docs/thesis/experiments/PHASE3_DIRECTION_AUDIT.md \
  docs/thesis/theory/THEORY_STATUS.md
```

Verify: the scope-restriction sentence (Rule E.3) appears in at least #16,
#18, #20 with consistent wording; #6 summarises but does not
contradict.

### 10.3 Tier/PASS-label invariant (Rule E.4)

Spot-check 10 numerical claims in `PHASE3A_COMBINATORIAL_RESULTS.md`:
each cites a source JSON (or a manifest file in `results/`), a tier
label from ANALYSIS_SPEC §6, and a PASS/NULL/FAIL decision.

### 10.4 `CURRENT_INDEX.md` zero-number invariant (Rule E.9)

```bash
grep -E "\+0\.[0-9]|-0\.[0-9]|[0-9]{2,3}%|B *= *[0-9]+|K *= *[0-9]+" \
  docs/CURRENT_INDEX.md
```

Allowed matches: numbered list items (`1.`, `2.`), cite strings (B=2 as
part of "see §X on B∈{2,3,4}" — acceptable if the number is not the
primary claim). Flagged matches: any standalone numerical finding.

### 10.5 Link-hygiene re-run

Re-run the grep set from §8.2. Must still be clean.

---

## 11. Rollback plan

Each step group is a single commit. If a step fails verification:

```bash
git restore --source=HEAD~1 -- .   # revert most recent step
# inspect, fix, re-run the failing step
```

Do not `git reset --hard` across multiple steps. Preserve the audit trail.

---

## 12. Execution order summary

| Phase | Steps | Touches filesystem? | Blocker on Job 480887? |
|---|---|---|---|
| Phase 3.0 | §0 pre-conditions | no | no |
| Phase 3.1 | §1 archive skeleton | yes (mkdir) | no |
| Phase 3.2 | §2 REMOVE | yes | no |
| Phase 3.3 | §3 ARCHIVE x20 | yes | no |
| Phase 3.4 | §4 MERGE x3 | yes | no |
| Phase 3.5 | §5 framework memos | yes | no |
| Phase 3.6 | §6 status banners (14 of 15) | yes | **#24 deferred** |
| Phase 3.7 | §7 promote CURRENT_INDEX | yes | no |
| Phase 3.8 | §8 link rewrites + hygiene gate | yes | no |
| Phase 4 | §9 README + INDEX refresh | yes | no |
| Phase 5 | §10 scientific-integrity verification | read-only | no |
| Post-Job-480887 | banner #24; write CROSS_BACKBONE_REPLICATION_RESULTS.md | yes | unblocked after job |

Total discrete git operations (excluding banner edits): 20 archive `git mv` +
3 merge-adjacent `git mv` + 2 memo `git mv` + 1 promote `git mv` + 1
REMOVE = **27 `git mv/rm` operations**.

Total banner/edit diffs: ~26 (20 archive banners + 3 merge targets edited
+ 2 memo banners + 1 CURRENT_INDEX rewrite).

---

## 13. Success criteria

Phase 3 is complete when all of the following hold simultaneously:

1. §8.2 hygiene grep set returns zero hits (outside exclusions).
2. `find docs/archive -type f | wc -l` ≥ 20 (all archived files land).
3. `find docs/thesis/appendix/framework_memos -type f | wc -l` = 2.
4. `find docs -name '*.md' -o -name '*.tex' | wc -l` (excluding
   `docs/archive` and `docs/pdf`) is ≤ 22 (was 41 .md + 1 .tex = 42
   baseline; after the plan: 11 canonical + 4 supporting + 3 maintenance
   (CLEANUP_LOG + this audit + this plan) + 2 framework memos + 1
   CURRENT_INDEX = 21 active + 2 under `docs/archive` counted separately
   = within budget).
5. `git log --name-status --diff-filter=R` on this branch lists the
   rename-tracked moves for every archived file (provenance preserved).
6. Every CANONICAL + SUPPORTING file has a status banner (Rule E.2)
   — except #24, deferred until Job 480887 completes.
7. No broken link in any `.md` file under `docs/` (link-hygiene grep
   clean).
8. Phase 5 (§10) passes all four invariants.

---

## 14. Out-of-scope for this plan

- LaTeX thesis chapters in `thesis/` — unchanged.
- `research/` work log + candidate-theorems — references updated, content
  untouched.
- `scripts/` — docstring references updated only; no code change.
- `external/` — untouched.
- Git commit messages: one commit per step group (13 commits max:
  skeleton, remove, archive, merge-1, merge-2, merge-3, memos,
  banners-batch, promote-index, link-rewrites, README-refresh,
  INDEX-refresh, integrity-verification). Each commit references this
  plan by path.

---

## 15. Outstanding decisions (for user / next iteration)

Carried from Audit §G:

- G.1 Location for the forthcoming `CROSS_BACKBONE_REPLICATION_RESULTS.md`
  (default: `docs/thesis/appendix/`).
- G.2 Whether to further split `RESULTS_STATUS.md` (default: preserve as
  single archive doc).
- G.3 Whether to merge `ASSESSMENT_OF_MEMOS_AND_ANALYSIS.md` (#23) into
  `INDEPENDENT_AUDIT_OF_DIRECTION_AND_MEMOS.md` (#26). Default: leave as
  two files; revisit after Phase 3 completes.

These are **not** blockers for Phase 3 execution.

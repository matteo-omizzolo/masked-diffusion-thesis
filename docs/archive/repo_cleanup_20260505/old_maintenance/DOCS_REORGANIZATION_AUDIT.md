> ARCHIVED — historical only.
> Do not use this file to infer the current thesis status unless explicitly asked.
> Current source of truth: see `START_HERE.md` and `docs/README.md`.

---


# DOCS Reorganization Audit — 2026-04-22

**Scope.** Every file under `docs/` (42 files: 41 `.md` + 1 `.tex`).
Not in scope: `thesis/` LaTeX chapters, `research/`, `scripts/`, `src/`,
`study/`, `external/`, `archive/`, `results/`. Those stay where they are.

**Auditor.** Claude Opus 4.7 (continued fresh read, no prior session state).

**Intent.** Produce a reorganization that is *scientifically coherent*,
*audit-preserving*, and *thesis-oriented* — not a surface move-and-rename.
Every move must preserve the ranker-class negative / search-class positive
distinction, the scope of the Negative-Result Corollary, and the Theorem A /
A′ / A″ refinement chain.

---

## A. Inventory (42 files)

Grouped by current location. Column **Last-content-signal** is the dated
marker inside the file (banner / updated-date / PASS-gate), not filesystem
mtime.

### A.1 Top-level `docs/` (5 files)

| # | Path | Purpose | Last-content-signal |
|---|---|---|---|
| 1 | `docs/thesis_direction.md` | Research direction narrative (Gap B/C focus, non-goals, scope) | 2026-04 (informed-corrector reframing era) |
| 2 | `docs/literature_map.md` | Bibliographic map across the nine-corner landscape | stable reference |
| 3 | `docs/reading_plan.md` | Paper reading order + status per paper | 2026-04 |
| 4 | `docs/experimental_infrastructure.md` | Tier-A/B backend/checkpoint/env status | 2026-04 |
| 5 | `docs/implementation_plan.md` | Phase-1-era implementation plan (pre-Phase-2b) | stale (Phase 1 era) |

### A.2 `docs/thesis/` canonical (4 files)

| # | Path | Purpose | Last-content-signal |
|---|---|---|---|
| 6 | `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md` | **Binding scope lock** — Phase-3a-aware; declares cross-backbone parked | 2026-04-20 |
| 7 | `docs/thesis/CURRENT_INDEX.md` | **Binding navigation** — points to every canonical doc | 2026-04-20 |
| 8 | `docs/thesis/CANONICAL_EXPERIMENT_OVERVIEW.md` | Protocol A/B definition + tier spec | stable reference |
| — | `docs/thesis/maintenance/` | See A.3 | — |
| — | `docs/thesis/experiments/` | See A.4 | — |
| — | `docs/thesis/theory/` | See A.5 | — |
| — | `docs/thesis/next_steps/` | See A.6 | — |

### A.3 `docs/thesis/maintenance/` (5 files)

| # | Path | Purpose | Last-content-signal |
|---|---|---|---|
| 9 | `CLEANUP_LOG.md` | 2026-04-18 documentation-consolidation provenance | 2026-04-18 |
| 10 | `LOG_CLEANUP_AUDIT.md` | 2026-04-18 `out/`/`err/` cleanup log | 2026-04-18 (operational) |
| 11 | `NEXT_CLAUDE_CODE_PROMPT.md` | Sonnet prompt for an already-completed rerun | stale |
| 12 | `hpc_sync_inventory.md` | 2026-04-18 HPC sync snapshot | stale |
| 13 | `hpc_sync_report.md` | 2026-04-18 HPC sync companion log | stale |

### A.4 `docs/thesis/experiments/` (6 files)

| # | Path | Purpose | Last-content-signal |
|---|---|---|---|
| 14 | `ANALYSIS_SPEC.md` | **Canonical** tier/bootstrap/JSON spec | stable reference |
| 15 | `EXPERIMENT_CRITICAL_AUDIT.md` | 2026-04-19 audit that triggered Phase 2b/3a | historical, PRESERVE |
| 16 | `PHASE3A_COMBINATORIAL_RESULTS.md` | **Primary positive result** — CD-G / BS-AG PASS table | 2026-04-20 |
| 17 | `PHASE3_ALTERNATIVE_PLAN.md` | Plan doc + postmortem addendum (superseded by #16) | 2026-04-20 (addendum) |
| 18 | `PHASE3_DIRECTION_AUDIT.md` | PRISM-rejection rationale + corollary rescope | 2026-04-20 |
| 19 | `RESULTS_STATUS.md` | Historical Phase 1→3a chronicle (UNDER-AUDIT banner) | 2026-04-20 |

### A.5 `docs/thesis/theory/` (3 files)

| # | Path | Purpose | Last-content-signal |
|---|---|---|---|
| 20 | `THEORY_STATUS.md` | **Canonical theorem summary** — Theorem A, A′, A″, corollary | 2026-04-20 |
| 21 | `THEORY_STRESS_TEST.md` | 2026-04-19 adversarial audit of Theorem A + Props B/C | historical, PRESERVE |
| 22 | `NEXT_THEORY_STEPS.md` | Three concrete theorem tasks (proofs for A / Prop B / Prop C) | small, mergeable |

### A.6 `docs/thesis/next_steps/` (6 files)

| # | Path | Purpose | Last-content-signal |
|---|---|---|---|
| 23 | `ASSESSMENT_OF_MEMOS_AND_ANALYSIS.md` | Memo audit — keep/reject classification | 2026-04 |
| 24 | `CROSS_BACKBONE_IMPLEMENTATION_AUDIT.md` | Phase-3 LLaDA-SFT backend audit | **load-bearing** for in-flight Job 480887 |
| 25 | `CROSS_BACKBONE_REPLICATION_PLAN.md` | Bounded GO decision for LLaDA-SFT run | 2026-04-21 |
| 26 | `INDEPENDENT_AUDIT_OF_DIRECTION_AND_MEMOS.md` | Phase-1 independent audit of direction + memos | 2026-04-21 |
| 27 | `INDEPENDENT_AUDIT_OF_COPILOT_IMPLEMENTATION.md` | Phase-2 audit of combinatorial-diagnostics code | 2026-04-22 |
| 28 | `PRINCIPLED_NEXT_STEPS_PLAN.md` | Cost-benefit ranking (contradicts #25 internally) | stale vs #25 |

### A.7 `docs/experiments/` (deprecated zone; 10 files)

| # | Path | Purpose | Last-content-signal |
|---|---|---|---|
| 29 | `entropy_proxy_experiment.md` | Phase-1-era entropy-proxy experiment spec | stale |
| 30 | `implementation_status.md` | Phase-1 implementation state | stale |
| 31 | `phase1_interpretation.md` | Phase-1 interpretation (superseded by EXPERIMENT_CRITICAL_AUDIT) | stale |
| 32 | `proseco_import_audit.md` | ProSeCo code-import audit | stale |
| 33 | `proseco_backend_audit.md` | ProSeCo backend readiness audit | stale |
| 34 | `proseco_experiment_definition.md` | Pre-Phase-2b ProSeCo design | superseded by #8 |
| 35 | `proseco_protocol_mapping.md` | Protocol A/B ↔ ProSeCo knob mapping | superseded by #8 |
| 36 | `proseco_validation_checklist.md` | Pre-launch validation checklist | operational, stale |
| 37 | `results/proseco_backend_failure_audit.md` | Failure audit (resolved) | stale |
| 38 | `results/result_inventory.md` | Phase-1 results inventory | superseded by #19 §14 |

### A.8 `docs/maintenance/` (1 file)

| # | Path | Purpose | Last-content-signal |
|---|---|---|---|
| 39 | `MOVED.md` | Pure redirect pointing to `docs/thesis/maintenance/` | stable but dead-weight |

### A.9 `docs/future ideas/` (2 files)

| # | Path | Purpose | Last-content-signal |
|---|---|---|---|
| 40 | `Deep Research Audit of Budgeted Informed Corrector Scheduling.md` | Deep-search memo (external-validity / cross-backbone lens) | 2026-04 |
| 41 | `Theoretical Frameworks for Budgeted Informed-Corrector Scheduling in Masked Diffusion Language Models.md` | Deep-search memo (MDP / SMC / Feynman-Kac framework) | 2026-04 |

### A.10 `docs/pdf/` (1 file)

| # | Path | Purpose | Last-content-signal |
|---|---|---|---|
| 42 | `pdf/status/phase2_status_report_2026-04-19.tex` | 2026-04-19 LaTeX status report (Phase 2 launch-time snapshot) | historical |

**Total: 42 files. No duplicates in the filesystem; duplication is at the
content-level (see §C).**

---

## B. Classification

Every file receives exactly one label: **CANONICAL**, **MERGE**,
**SUPPORTING**, **ARCHIVE**, or **REMOVE**.

### B.1 CANONICAL (binding, source of truth; 11 files)

These define the thesis direction, numbers, and scope. No downstream doc may
contradict them.

| # | File | Why canonical |
|---|---|---|
| 6 | `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md` | Binding scope lock |
| 7 | `docs/thesis/CURRENT_INDEX.md` | Binding navigation |
| 8 | `docs/thesis/CANONICAL_EXPERIMENT_OVERVIEW.md` | Binding Protocol A/B spec |
| 14 | `docs/thesis/experiments/ANALYSIS_SPEC.md` | Binding tier/bootstrap/JSON spec |
| 16 | `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md` | Primary positive (load-bearing) |
| 18 | `docs/thesis/experiments/PHASE3_DIRECTION_AUDIT.md` | PRISM-rejection + corollary rescope |
| 20 | `docs/thesis/theory/THEORY_STATUS.md` | Canonical theorem summary |
| 24 | `docs/thesis/next_steps/CROSS_BACKBONE_IMPLEMENTATION_AUDIT.md` | Load-bearing for in-flight Job 480887 |
| 25 | `docs/thesis/next_steps/CROSS_BACKBONE_REPLICATION_PLAN.md` | Bounded GO decision |
| 26 | `docs/thesis/next_steps/INDEPENDENT_AUDIT_OF_DIRECTION_AND_MEMOS.md` | Phase-1 independent audit |
| 27 | `docs/thesis/next_steps/INDEPENDENT_AUDIT_OF_COPILOT_IMPLEMENTATION.md` | Phase-2 independent audit |

### B.2 MERGE (consolidate into a canonical doc; 3 files)

| # | File | Target | Reason |
|---|---|---|---|
| 17 | `PHASE3_ALTERNATIVE_PLAN.md` | → `PHASE3A_COMBINATORIAL_RESULTS.md` (appendix) | Plan superseded by result; addendum already lives upstream |
| 22 | `NEXT_THEORY_STEPS.md` | → `THEORY_STATUS.md` (§"Proof work queue") | 32 lines; same semantic unit |
| 28 | `PRINCIPLED_NEXT_STEPS_PLAN.md` | → `CROSS_BACKBONE_REPLICATION_PLAN.md` (§"Superseded ranking") | Resolves documented internal inconsistency — the replication plan is the later, binding decision |

**Merge rule.** When merging, carry every load-bearing sentence (numbers,
caveats, non-obvious claims) into the target as a labelled appendix. Drop
restated background. After merge, source file moves to `archive/` with a
header pointing to the target.

### B.3 SUPPORTING (kept but not canonical; 4 files)

Useful operating documents; referenced by canonical docs but allowed to
evolve.

| # | File | Role |
|---|---|---|
| 1 | `docs/thesis_direction.md` | Narrative direction doc — overlaps with #6 but audience is different (readable prose vs binding constraints) |
| 2 | `docs/literature_map.md` | Bibliographic anchor |
| 3 | `docs/reading_plan.md` | Reading status tracker |
| 4 | `docs/experimental_infrastructure.md` | Tier-A/B infra tracker |

### B.4 ARCHIVE (historical / operational; 20 files)

Content that represents the repo at a prior moment. Must be preserved for
provenance but removed from active navigation.

| # | File | Why archive |
|---|---|---|
| 5 | `docs/implementation_plan.md` | Phase-1-era plan; superseded by current direction |
| 9 | `CLEANUP_LOG.md` | Operational provenance |
| 10 | `LOG_CLEANUP_AUDIT.md` | Operational snapshot |
| 11 | `NEXT_CLAUDE_CODE_PROMPT.md` | Sonnet prompt, rerun completed |
| 12 | `hpc_sync_inventory.md` | HPC sync snapshot |
| 13 | `hpc_sync_report.md` | HPC sync log |
| 15 | `EXPERIMENT_CRITICAL_AUDIT.md` | Historical audit that triggered Phase 2b/3a — preserve in `archive/audits/` |
| 19 | `RESULTS_STATUS.md` | Historical chronicle — PRESERVE, but canonical numbers now live in #16/#18/#20 |
| 21 | `THEORY_STRESS_TEST.md` | Historical stress test; its conclusions are already encoded in #20 |
| 29 | `entropy_proxy_experiment.md` | Phase-1-era |
| 30 | `implementation_status.md` | Phase-1-era |
| 31 | `phase1_interpretation.md` | Phase-1-era |
| 32 | `proseco_import_audit.md` | Phase-1-era |
| 33 | `proseco_backend_audit.md` | Phase-1-era |
| 34 | `proseco_experiment_definition.md` | Superseded by #8 |
| 35 | `proseco_protocol_mapping.md` | Superseded by #8 |
| 36 | `proseco_validation_checklist.md` | Pre-launch checklist; executed |
| 37 | `proseco_backend_failure_audit.md` | Resolved incident |
| 38 | `result_inventory.md` | Superseded by #19 §14 |
| 42 | `phase2_status_report_2026-04-19.tex` | Phase-2 launch snapshot; preserve under `archive/status/` |

**Archive rule.** Files move to `archive/docs/<original-path>/` preserving
relative structure. Every archived file receives a 2-line header:
`> ARCHIVED 2026-04-22. Superseded by: <path>. See DOCS_REORGANIZATION_PLAN.md`

### B.5 REMOVE (delete entirely; 1 file)

| # | File | Why |
|---|---|---|
| 39 | `docs/maintenance/MOVED.md` | Pure redirect; git history preserves the fact; keeping adds zero value. Safe to delete since all callers already use `docs/thesis/maintenance/`. |

### B.6 Special case — `docs/future ideas/` (2 files)

Memos 40 and 41 are **flagged as speculative** by the Phase-1 independent
audit (file #26). They are not canonical but are referenced load-bearingly by
scope decisions. Treatment:

- Move to `docs/thesis/appendix/framework_memos/`
- Add banner per memo stating the audit verdict (Contradicted / Speculative
  / Plausible-but-untested from #26 §3.2–3.3)
- Keep reachable from `CURRENT_INDEX.md` under "Future-Work Framework
  Memos"

### B.7 Classification summary

| Label | Count |
|---|---|
| CANONICAL | 11 |
| MERGE | 3 |
| SUPPORTING | 4 |
| ARCHIVE | 20 |
| REMOVE | 1 |
| SPECIAL (framework memos) | 2 |
| 1 new home (target) | 1 (`archive/docs/`) |
| **Total** | **42** (accounted) |

---

## C. Redundancy map

This is the load-bearing evidence for the reorganization. The same concrete
claim appears in multiple files, often with small drift. After merge/archive
each row must have exactly one **CANONICAL** source.

### C.1 Protocol A / B definition (duplicated 5×)

| File | Form |
|---|---|
| #8 `CANONICAL_EXPERIMENT_OVERVIEW.md` | Full formal spec (CANONICAL) |
| #34 `proseco_experiment_definition.md` | Pre-Phase-2b variant |
| #35 `proseco_protocol_mapping.md` | Knob-mapping variant |
| #29 `entropy_proxy_experiment.md` | Phase-1 variant |
| #42 `phase2_status_report_2026-04-19.tex` | LaTeX status variant |

**Resolution.** #8 is the sole source. Archive the other four. No
thesis-core doc may restate Protocol A/B.

### C.2 Theorem A full statement (duplicated 4×)

| File | Form |
|---|---|
| #20 `THEORY_STATUS.md` §"Main Theorem Target" | CANONICAL |
| #6 `CANONICAL_RESEARCH_DIRECTION.md` | Summary paraphrase |
| #22 `NEXT_THEORY_STEPS.md` | Proof-target restatement |
| #42 `phase2_status_report_2026-04-19.tex` | LaTeX restatement |

**Resolution.** #20 is the sole source. Direction doc #6 keeps a one-line
reference ("Theorem A: G(S_B*) − G(Ŝ_B) ≤ 2Bε + 2η_B — see THEORY_STATUS §2")
but does not restate the full theorem. #22 merges into #20. #42 archived.

### C.3 Core research question (duplicated 3×)

| File | Form |
|---|---|
| `CLAUDE.md` (project instructions) | Operative binding text |
| #1 `thesis_direction.md` | Narrative framing |
| #6 `CANONICAL_RESEARCH_DIRECTION.md` | Thesis-bound restatement |

**Resolution.** `CLAUDE.md` carries the binding text, #6 carries the
thesis-bound canonical restatement, #1 is the explanatory narrative and
keeps a verbatim quote of #6.

### C.4 Phase 3a headline numbers (duplicated 4×)

| File | Form |
|---|---|
| #16 `PHASE3A_COMBINATORIAL_RESULTS.md` §3.1 | CANONICAL table |
| #17 `PHASE3_ALTERNATIVE_PLAN.md` addendum | Restated summary |
| #18 `PHASE3_DIRECTION_AUDIT.md` §5 addendum | Corollary-context restatement |
| #19 `RESULTS_STATUS.md` §14 | Chronicle restatement |

**Resolution.** #16 is the sole source. #17 merges into #16. #18 replaces
restatement with "see #16 §3.1". #19 archived (but referenced for
chronological context).

### C.5 Phase 2b MC-oracle numbers (duplicated 3×)

| File | Form |
|---|---|
| #19 `RESULTS_STATUS.md` §12 | Launch-time Phase-2b report |
| #20 `THEORY_STATUS.md` honesty-ledger rows | Theorem-anchored citations |
| #16 `PHASE3A_COMBINATORIAL_RESULTS.md` §4 | Re-used as oracle denominator |

**Resolution.** #19 archived. Canonical source for Phase 2b numbers moves
into a new doc or into #16 Appendix A (preferred: appendix of #16).

### C.6 Combinatorial-diagnostics numbers (duplicated 3×)

| Source | Form |
|---|---|
| `results/phase2b/combinatorial_diagnostics.json` | File-backed truth |
| #26 `INDEPENDENT_AUDIT_OF_DIRECTION_AND_MEMOS.md` R1 | Flags 1.5× vs 1.2×/1.3× drift |
| #27 `INDEPENDENT_AUDIT_OF_COPILOT_IMPLEMENTATION.md` §3 | Re-audit of the JSON file |

**Resolution.** Canonical source is the JSON file. #27 is the canonical
human-readable audit. #26 keeps the cross-reference to #27.

### C.7 PRISM rejection rationale (duplicated 3×)

| File | Form |
|---|---|
| #18 `PHASE3_DIRECTION_AUDIT.md` | CANONICAL |
| #6 `CANONICAL_RESEARCH_DIRECTION.md` | One-paragraph summary |
| #20 `THEORY_STATUS.md` honesty-ledger | One-line pointer |

**Resolution.** Already coherent — keep as is; verify pointer links after
the move.

### C.8 Cross-backbone decision (stated in 3× contradicting ways)

| File | Stance |
|---|---|
| #6 `CANONICAL_RESEARCH_DIRECTION.md` | **Parked / out of scope** |
| #25 `CROSS_BACKBONE_REPLICATION_PLAN.md` | **Bounded GO** |
| #28 `PRINCIPLED_NEXT_STEPS_PLAN.md` | **Postpone** (contradicts #25) |

**Resolution.** The independent audit #26 §4 identifies this exact
contradiction. Resolution path: #25 is the latest binding decision (a
bounded appendix experiment); #6 is updated to state "parked from thesis
core; runs as optional appendix per CROSS_BACKBONE_REPLICATION_PLAN"; #28
merges into #25 as a "superseded ranking" appendix. See §B.2.

### C.9 Ranker-class vs search-class scope (load-bearing; must NOT drift)

| File | Phrasing |
|---|---|
| #20 `THEORY_STATUS.md` Refinement-A″ + Corollary | CANONICAL scope: ranker-class only |
| #16 `PHASE3A_COMBINATORIAL_RESULTS.md` §4 | Search-class explicitly exceeds ranker envelope |
| #18 `PHASE3_DIRECTION_AUDIT.md` §3 | PRISM = ranker class |
| #6 `CANONICAL_RESEARCH_DIRECTION.md` §2 | Narrative restatement |

**Resolution.** No drift currently. Locked in Rule E3 below.

---

## D. Target information architecture

### D.1 Proposed tree

```
docs/
├── CURRENT_INDEX.md                      # [navigation entrypoint — promoted]
├── thesis_direction.md                   # SUPPORTING (narrative)
├── literature_map.md                     # SUPPORTING
├── reading_plan.md                       # SUPPORTING
├── experimental_infrastructure.md        # SUPPORTING
├── thesis/
│   ├── CANONICAL_RESEARCH_DIRECTION.md   # CANONICAL scope
│   ├── CANONICAL_EXPERIMENT_OVERVIEW.md  # CANONICAL Protocol A/B
│   ├── experiments/
│   │   ├── ANALYSIS_SPEC.md              # CANONICAL tier/bootstrap spec
│   │   ├── PHASE3A_COMBINATORIAL_RESULTS.md  # CANONICAL positive result
│   │   │                                     # (absorbs PHASE3_ALTERNATIVE_PLAN)
│   │   └── PHASE3_DIRECTION_AUDIT.md     # CANONICAL ranker-class negative
│   ├── theory/
│   │   └── THEORY_STATUS.md              # CANONICAL theorem summary
│   │                                     # (absorbs NEXT_THEORY_STEPS as §"Proof queue")
│   ├── next_steps/
│   │   ├── CROSS_BACKBONE_IMPLEMENTATION_AUDIT.md
│   │   ├── CROSS_BACKBONE_REPLICATION_PLAN.md
│   │   │     # (absorbs PRINCIPLED_NEXT_STEPS_PLAN as §"Superseded ranking")
│   │   ├── INDEPENDENT_AUDIT_OF_DIRECTION_AND_MEMOS.md
│   │   ├── INDEPENDENT_AUDIT_OF_COPILOT_IMPLEMENTATION.md
│   │   └── ASSESSMENT_OF_MEMOS_AND_ANALYSIS.md   # Retained (memo classification ≠ #26)
│   ├── maintenance/
│   │   ├── CLEANUP_LOG.md                # PRESERVE (provenance)
│   │   ├── DOCS_REORGANIZATION_AUDIT.md  # [this file]
│   │   └── DOCS_REORGANIZATION_PLAN.md   # [Phase 2 deliverable]
│   └── appendix/
│       └── framework_memos/
│           ├── Deep_Research_Audit...md      # banner: audited 2026-04-22
│           └── Theoretical_Frameworks...md   # banner: audited 2026-04-22
├── pdf/
│   └── status/                            # kept, now gated under archive/
└── archive/
    ├── audits/
    │   ├── EXPERIMENT_CRITICAL_AUDIT.md
    │   ├── THEORY_STRESS_TEST.md
    │   └── proseco_backend_failure_audit.md
    ├── phase1_era/
    │   ├── implementation_plan.md
    │   ├── implementation_status.md
    │   ├── phase1_interpretation.md
    │   ├── entropy_proxy_experiment.md
    │   ├── proseco_import_audit.md
    │   ├── proseco_backend_audit.md
    │   ├── proseco_experiment_definition.md
    │   ├── proseco_protocol_mapping.md
    │   ├── proseco_validation_checklist.md
    │   └── result_inventory.md
    ├── operational/
    │   ├── LOG_CLEANUP_AUDIT.md
    │   ├── NEXT_CLAUDE_CODE_PROMPT.md
    │   ├── hpc_sync_inventory.md
    │   └── hpc_sync_report.md
    ├── chronicles/
    │   └── RESULTS_STATUS.md             # preserved for historical provenance
    └── status/
        └── phase2_status_report_2026-04-19.tex
```

**Deletions.** `docs/maintenance/MOVED.md` removed. `docs/experiments/`
folder removed after its contents migrate to `archive/phase1_era/`.

### D.2 Navigation layering

- **Entry points (front door):** `docs/CURRENT_INDEX.md` (was in `thesis/`,
  promoted to `docs/`) and `README.md` under repo root.
- **Canonical thesis spine:** `docs/thesis/CANONICAL_*` + `experiments/` +
  `theory/`.
- **Operational annex:** `docs/thesis/next_steps/` (plans + audits, in flux).
- **Appendix (citable but speculative):** `docs/thesis/appendix/`.
- **Archive (read-only):** `docs/archive/`.

### D.3 What is explicitly NOT in this tree

- No "working/" or "scratch/" folder. Conversations create such content;
  this repo is not the place.
- No `future/` at the top level — speculative ideas go under
  `thesis/appendix/framework_memos/` with audit banners so readers see the
  label before the content.

---

## E. Reorganization rules (binding)

These rules govern both the execution of Phase 3 and subsequent edits.

### E.1 Single-source rule

Every load-bearing fact has exactly one canonical source file and a
`#section` anchor. Downstream references use `[see CANONICAL_FILE.md §X]`,
**not** a restatement. Restatements drift; links do not.

Load-bearing facts, as used here: (a) Protocol A/B definition, (b) tier
labels and bootstrap conventions, (c) Theorem A / A′ / A″ / Corollary
statements, (d) Phase 3a closure-ratio numbers, (e) Phase 2b MC-oracle
headroom, (f) combinatorial-diagnostics within-seed and Jaccard numbers,
(g) scope decisions (cross-backbone parked, PRISM rejected).

### E.2 Status banner rule

Every file in `docs/` (after reorganization) carries a one-line status
banner at the top:

```
> **STATUS:** CANONICAL | SUPPORTING | HISTORICAL | ARCHIVED
> **LAST UPDATED:** YYYY-MM-DD
> **SUPERSEDED BY:** <path>   (only if ARCHIVED)
```

Documents without this banner are presumed stale.

### E.3 Ranker-class vs search-class distinction (non-drift)

The following pair of statements must remain verbatim-consistent across
#6, #16, #18, #20:

> Any policy of the form `Ŝ_B = top-B(ψ)` for separable per-step `ψ` is
> upper-bounded by the `mean_delta_oracle` envelope (ranker class). Phase
> 3a's CD-G and BS-AG operate on schedules and explicitly exceed this
> envelope, so the Negative-Result Corollary does not generalise to
> "informed scheduling in general".

No edit may weaken, broaden, or narrow this without a corresponding theory
update in #20.

### E.4 Tier-label and PASS/FAIL provenance rule

Every numerical claim (closure ratio, paired mean, CI) must cite:

- the source JSON or manifest file (`results/phase3a/...`, etc.),
- the tier label (T1 / T2 / T3 / T4 — see #14 §6),
- the PASS / NULL / FAIL decision per #14 §7.

Unattributed numbers are not allowed in CANONICAL docs.

### E.5 Archive-don't-delete rule

No content is deleted except the bare redirect `docs/maintenance/MOVED.md`.
All historical / operational / Phase-1-era files move to
`docs/archive/...` with an `ARCHIVED` banner, preserving provenance under
git history + on-disk location.

### E.6 Memo-audit-banner rule

Files under `docs/thesis/appendix/framework_memos/` carry an audit banner
derived from #26 §3.2–3.3, e.g.:

```
> **STATUS:** APPENDIX (SPECULATIVE). Audited 2026-04-22.
> **AUDIT VERDICT:** Contradicted-in-repo (cross-backbone priority) /
>                    Speculative (SMC, Feynman-Kac, PRISM revival).
> **THESIS POSITION:** Future work only; not part of thesis-core argument.
```

### E.7 Plan ↔ result contract

Every *PLAN* doc, once its experiment runs, must link forward to its
*RESULT* doc and state "superseded by …". Every *RESULT* doc links back to
its plan. The current in-flight example: `CROSS_BACKBONE_REPLICATION_PLAN`
(#25) must link to the forthcoming `CROSS_BACKBONE_REPLICATION_RESULTS` as
soon as Job 480887 completes.

### E.8 Link-hygiene rule

After every move: `grep -r "docs/<old-path>"` on the whole repo must return
zero hits. Broken links are a blocker for Phase 3 completion.

### E.9 `CURRENT_INDEX.md` rule

`CURRENT_INDEX.md` is treated as an **index only**: it lists canonical docs
with one-line hooks, cites the latest verdict per topic, and **never**
originates a number. Edits that add new numerical claims to CURRENT_INDEX
must be rejected and pushed upstream to the canonical source.

---

## F. Risks + exceptions

### F.1 Thesis-writing risk

`docs/thesis_direction.md` (#1) and
`docs/thesis/CANONICAL_RESEARCH_DIRECTION.md` (#6) overlap. Decision:
retain both on the grounds that #1 is narrative-first (reads like a chapter
introduction), #6 is constraint-first (reads like a scope lock). Risk
managed by Rule E.1 — #1 references #6 for every load-bearing claim.

### F.2 In-flight-job risk

Job 480887 (LLaDA-SFT cross-backbone bounded) is running and depends on
`hpc/cross_backbone_proseco_llada_sft_bounded.sbatch` + the code and
docstring contracts encoded in #24 (`CROSS_BACKBONE_IMPLEMENTATION_AUDIT`).
**Do not move #24 during reorganization.** It may be re-homed after Job
480887 completes and results are written into #8 successor doc.

### F.3 Memo-provenance risk

Moving `docs/future ideas/` with its literal whitespace in the path is
reader-hostile. Rename to `framework_memos/` with underscores at move time;
keep the file basenames (they are already canonical titles).

### F.4 Git-history risk

Each archival move must be a single `git mv` (rename-tracked), not delete
+ add, so history survives. Enforced in the Phase 2 plan's command list.

---

## G. Open questions for Phase 2 plan

1. **Where to put the forthcoming `CROSS_BACKBONE_REPLICATION_RESULTS.md`
   after Job 480887 completes?** Current default: in
   `docs/thesis/appendix/` (since it runs as an appendix to the thesis core
   per #6 / §F.1 of this audit). Decision deferred to Phase 2 plan.

2. **Does `RESULTS_STATUS.md` (#19) need to be partially preserved as a
   "historical chronicle" even after archiving?** Current default: keep a
   single stub `docs/archive/chronicles/RESULTS_STATUS.md` with its
   existing content + banner; do not split. Decision deferred to Phase 2
   plan.

3. **Should `ASSESSMENT_OF_MEMOS_AND_ANALYSIS.md` (#23) merge with
   `INDEPENDENT_AUDIT_OF_DIRECTION_AND_MEMOS.md` (#26)?** Overlap in §3.2
   / §3.3 of #26. Defer — kept as two files in the target tree pending a
   closer content-level pass in Phase 2 plan.

---

## H. Verdict

- Inventory complete: 42 files accounted for, zero orphans.
- Classification complete with no overlap: every file has exactly one
  label.
- Redundancy is real and load-bearing: Protocol A/B spec duplicated 5×;
  Theorem A statement 4×; Phase 3a numbers 4×; scope decision documented
  with three mutually contradictory stances (#6, #25, #28).
- Reorganization is **safe to execute** provided rules E.1–E.9 hold and
  Risk F.2 is respected (no move of #24 during Job 480887).
- **Phase 2 plan** (next file: `DOCS_REORGANIZATION_PLAN.md`) will
  translate §B–§D into a concrete ordered sequence of `git mv` + edit
  commands plus link-hygiene verification.

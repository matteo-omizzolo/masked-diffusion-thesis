# CLAUDE.md — Project Context for Claude Code

## Current-status rule (IMPORTANT)

> Use `START_HERE.md` and `docs/README.md` as the entry points for any new session.
> Do NOT use files under `docs/archive/`, `archive/`, or `docs/thesis/` to infer
> current thesis status — those are historical and may contradict current docs.
> Archived files exist for provenance only; they are explicitly superseded.

## Project
MSc thesis on **signal-adaptive corrector scheduling for masked diffusion language models**,
supervised by Prof. Giacomo Zanella (Bocconi University).

**Core research question:** For a fixed predictor schedule and fixed corrector-placement
budget B in masked diffusion language models, when is informed correction timing
reducible to marginal signal ranking, and when does it require interaction-aware or
search-based scheduling?

**Thesis story (April 2026 baseline):** Fixed-budget corrector allocation is a combinatorial
schedule-search problem. Tested separable rankers do not recover MC-oracle headroom on
ProSeCo-OWT; the mean-Δ̄ envelope enters the no-detectable-gain band by B = 8. Search procedures
(CD-G, BS-AG) recover 49–84 % of MC-oracle headroom. Theorem A (uniform proxy regret) is proved;
A′/A″ are diagnostics; the Empirical Ranker-Class Limitation (formal time-only part + empirical
part on tested rankers) replaces the previous "Negative-Result Corollary".

**Current phase (May 2026):** Phase 1 interaction diagnostics. Phase 0 is
complete (PF1–PF8 ✅ on HPC CPU slnode01; K=3 smoke ✅ on A100 gnode01 job
489457, 2026-05-07), and the K=30 critical replication gate is closed
(Phase 2b job 490106 ✅, Phase 3a canonical job 479941 ✅). Next empirical
work: Gate 3a sparse pair diagnostics, then Gate 3b schedule-level validation.

**Key distinction:** This thesis targets corrector *scheduling* (when to spend corrective
effort across the trajectory), not token-selection policies (which tokens to correct),
predictor schedules (when to unmask), or corrector kernel design (how to correct).

## Active doc entry points

| File | Purpose |
|---|---|
| `START_HERE.md` | 2-minute orientation — start here |
| `docs/README.md` | Doc structure guide + archive rule |
| `docs/00_current_status.md` | Established results, failures, risks |
| `docs/01_research_direction.md` | Thesis Q, scope, non-goals, contribution |
| `docs/02_experiments.md` | Phases, protocols, results |
| `docs/03_theory.md` | Theorem stack, proof status, open gaps |
| `docs/04_results_index.md` | Raw results index |
| `docs/05_next_steps.md` | Sequential action plan — Phase 1 diagnostics → pairwise scheduler decision → writing |
| `docs/06_theory_first_research_plan.md` | Active planning doc — theory-first programme (not final status) |
| `research/candidate_theorems.md` | Full theorem + proof record (keep active) |
| `research/proof_worklog.md` | Derivation entries (keep active) |

## Repo layout
- `START_HERE.md` — 2-minute orientation
- `thesis/` — LaTeX thesis chapters (ch1–ch6 drafted; ch7/abstract/conclusion TODO)
- `research/` — mathematical worklog, candidate theorems, proof ledger
- `docs/` — compact active docs (see docs/README.md)
- `docs/archive/` — archived historical docs (do not use for current status)
- `src/mdm_playground/` — main package (`pip install -e .`)
- `external/` — rsync'd copies of upstream repos (remdm, mdlm, PRISM, sedd, remedi)
- `study/` — papers organized by topic, notes, thesis guidelines
- `scripts/` — analysis scripts; `scripts/legacy/` for superseded scripts
- `results/` — raw experiment outputs (preserved, never delete)
- `configs/` — YAML experiment configs
- `hpc/` — Bocconi HPC workflow scripts
- `archive/` — legacy material (pre-April 2026)
- `figures/` — generated plots

## HPC — Bocconi cluster
- **Host:** `slogin.hpc.unibocconi.it` | **User:** `3316152`
- **Partition/QOS:** `stud` | Max 4× NVIDIA A100 80GB
- **Repo on HPC:** `~/mdm/masked-diffusion-thesis`
- **Primary ProSeCo-OWT snapshot on HPC:** `~/mdm/checkpoints/proseco_owt`
- **Python env:** conda env `remdm311` (Python 3.11) — load with:
  ```bash
  module load miniconda3
  source $(conda info --base)/etc/profile.d/conda.sh
  conda activate remdm311
  ```
- **Default system Python is 3.9** — incompatible with this project (`requires >=3.10`). Always use `remdm311`.

## HPC workflow
```bash
bash hpc/push.sh       # rsync code (excludes checkpoints, .venv, results)
# K=30 replication is closed. Do not resubmit phase2b/phase3a for that gate.
# Run a Codex pre-submit review before any approved Phase 1 diagnostics sbatch.
# NOTE: pull.sh uses GNU rsync flags, broken on macOS. Use SSH instead:
ssh 3316152@slogin.hpc.unibocconi.it "python3 -c 'import json; print(...)'"
```

## HPC pre-submit rule (mandatory)

Before submitting **any** sbatch job, Claude must:
1. Invoke Codex (via `/codex:rescue`) to review the proposed work, including:
   - The exact sbatch script content and any changed files since last push.
   - `git status` and `git diff --check` output.
   - Validation commands that will be run (e.g., `pytest tests/test_phase0_preflight.py -q`).
   - Expected output folder and which gate is being opened or closed.
   - Confirmation that no K=3 smoke numbers are being elevated to thesis evidence.
2. Address any BLOCKING or WARNING findings before submitting.
3. Document the Codex review finding summary in the commit message.

This rule applies even if the sbatch script has been previously reviewed in another session.

---

## Known HPC environment issues / rules

### 1. setuptools 82 removed `pkg_resources`
`lightning` 2.2.1 calls `pkg_resources` at import time.
**Fix:** `pip install 'setuptools<70'` in the conda env.

### 2. NumPy 2.0 removed `np.float_`
`wandb 0.13.5` used `np.float_`. **Fix:** `pip install 'wandb>=0.16'`

### 3. `flash_attn` — cannot build on this cluster
System CUDA is 13.0 but PyTorch compiled with 12.8 → nvcc mismatch.
**Current rule:** external repos stay clean upstream. The active ProSeCo-OWT path
uses `scripts/proseco/reproduction/stage_proseco_owt.py`, which applies the documented fallback patch
to the staged HuggingFace snapshot, not to `external/`.

### 4. cuda-nvcc conda package breaks sbatch
Installing `cuda-nvcc` via conda adds an activation script that references
`NVCC_PREPEND_FLAGS` (unbound var), crashing `set -u` in sbatch.
**Fix:** `conda remove cuda-nvcc cuda-toolkit` — not needed since flash_attn fallback used.

### 5. PyTorch ≥2.6 `weights_only=True` default
Old checkpoints can fail to load under the new default. Active backend wrappers
force `weights_only=False` only for trusted local checkpoints; do not patch
external repos silently.

### 6. git submodule update fails on HPC
Repo is rsync'd (not cloned), so no `.git` directory.
**Fix:** Removed `git submodule update --init --recursive` from `hpc/remdm_smoke.sbatch`.

### 7. Missing packages (not in original requirements.txt)
Added to `requirements.txt`: `transformers==4.38.2`, `hydra-core==1.3.2`, `omegaconf>=2.3`,
`datasets>=2.21`, `einops>=0.7`, `lightning>=2.2`, `wandb>=0.16`, `h5py>=3.10`,
`timm>=0.9`, `mauve-text>=0.4`, `setuptools<70`.

### 8. transformers version mismatch with checkpoint
Checkpoint was saved with `transformers==4.38.2`. Newer versions moved
`tokenization_gpt2_fast` path. **Fix:** `pip install 'transformers==4.38.2'` — pin exactly.

### 9. ProSeCo snapshot fallback — rotary embedding shape
The staged `modeling_proseco.py` patch must keep flash-attn and pure-PyTorch
rotary shapes distinct. This is handled in `scripts/proseco/reproduction/stage_proseco_owt.py`.

### 10. NumPy 2.0 `np.array(copy=False)` behavior change in datasets 2.18
`datasets==2.18` uses `np.array(array, copy=False)` which raises ValueError in NumPy 2.x
when a copy is needed.
**Fix:** `pip install 'datasets>=2.21'` (datasets 4.x supports NumPy 2.x).

### 11. OpenWebText reference — use owt-reference streaming solution
Downloading openwebtext (38GB) for MAUVE reference exceeds HPC user quota.
Do not patch `external/remdm` silently for this. Keep any future data-staging
logic in project scripts or documented config patches.

### 12. Compute nodes have no outbound internet
Job 490469 failed harmlessly because `pip install -e .` in the Phase 3a sbatch
preamble attempted PyPI access from gnode02. It died before any shard launched
and wrote no files. Future sbatches must use the pre-provisioned `remdm311`
environment or offline-safe install steps; do not assume PyPI access on `gnode*`.

---

## Current status (May 2026)

See `START_HERE.md` and `docs/00_current_status.md` for full current status.
Key points:

- **Baseline experiments complete.** Phase 1 (signal calibration), Phase 2b (policy
  comparison + MC-oracle), Phase 3a (combinatorial search baselines), cross-backbone
  LLaDA-SFT probe, and Protocol C (adaptive controller CPU pilot) are all done.
- **Primary result.** CD-G recovers 74–84 % and BS-AG 49–64 % of +0.45 MC-oracle headroom
  over uniform at B ∈ {2,3,4} on ProSeCo-OWT. Both pass at B = 8.
- **K=30 replication.** Phase 2b replication job 490106 completed 2026-05-07
  with headroom +0.385/+0.355/+0.380 at B=2/3/4 (avg ≈ +0.37), versus the
  canonical +0.451/+0.441/+0.450 (avg ≈ +0.45). Qualitative conclusions are
  unchanged. `mean_delta_oracle` is a cheating marginal / time-profile oracle
  ranker, not a non-separable policy, and recovers only 13–29 % of replicated
  headroom.
- **Theory (May 2026 correction).** Theorem A is the proved marginal baseline;
  A′/A″ are diagnostics only; the former Negative-Result Corollary is replaced
  by the Empirical Ranker-Class Limitation; Theorem B/B′ + Diagnostic Framework C
  are the active interaction framework. Theorem A-ad is appendix/provenance only.
- **Current phase:** Phase 1 interaction diagnostics. Theory stack formalized
  (Theorem A baseline; Theorem B + B′ central; Diagnostic Framework C;
  Theorem D + Lemma E optional/appendix). Phase 0 and K=30 replication are
  complete; Gate 3 is open.
- **HPC caution.** Failed duplicate Phase 3a job 490469 is harmless: it died
  in the sbatch preamble before shard launch because compute-node `pip install -e .`
  tried to reach PyPI. No files were written. Before reusing Phase 2b/3a sbatches,
  remove or make offline-safe the compute-node pip install preamble.
- **Backbone:** ProSeCo-OWT only (LLaDA-SFT probe inconclusive).
- **Reading:** ProSeCo, PRISM, MDLM, ReMDM, L&Z Error Bounds done.

### Phase 0 / K=30 status (2026-05-08) — COMPLETE ✅
- Checkpoint staged at `~/mdm/checkpoints/proseco_owt` (647 MB, model.safetensors).
- PF1–PF8: **11 passed, 0 skipped** on HPC CPU (slnode01). All assertions pass.
- K=3 smoke: job 489457 on gnode01 A100. Qualitative pattern matches prior K=30 results.
- K=30 Phase 2b replication: job 490106 on gnode01, completed 2026-05-07.
- K=30 Phase 3a canonical: job 479941 on gnode02, complete and intact.
- K=30 gate is closed. Do not resubmit `hpc/proseco/reproduction/phase2b_proseco_owt.sbatch`
  or `hpc/proseco/reproduction/phase3a_combinatorial.sbatch` for the closed gate.
- Debug: `python scripts/legacy/debug_proseco_owt_load.py --checkpoint ~/mdm/checkpoints/proseco_owt --device cpu --T 4`

### Writing status (2026-05-06)
- ch1–ch5: full PASS 2 drafts done as study-guide chapters leading into ch6.
- ch6 (Contribution): full self-contained mathematical framework drafted.
- ch7 / experiments: honest scaffold. May now be updated with K=30 evidence.
- Abstract: conservative draft written. Update after Phase 1 / final empirical framing.
- Conclusion: conservative draft written. Update after Phase 1 / final empirical framing.

### LaTeX build
Run from inside the `thesis/` directory:
```bash
cd thesis && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/masked-thesis-build main.tex
```
Running from the repo root fails because `\input{macros}` resolves relative to the working directory.
Last successful build: 2026-05-06 (55 pages). Remaining warnings: hyperref math-token (2, fixed), one overfull hbox (fixed), biblatex name-format (fixed), bibliography layout overflows (cosmetic; draft acceptable).

## Claude Code tools & plugins

### Active skills (no installation needed)
- **`firecrawl`** — fetch arXiv papers, read documentation; primary tool for literature search
- **`simplify`** — review analysis scripts for quality/efficiency after writing them

### Installed plugins
- **`code-review`** — structured review of scripts before HPC submission (`/code-review`)
- **`playground`** — interactive HTML explorers; useful for corrector strategy space visualisation
- **`claude-md-management`** — improve/revise this CLAUDE.md (`/revise-claude-md`)
- **`feature-dev`** — structured 7-phase workflow for implementing new corrector strategies (`/feature-dev`)
- **`learning-output-style`** — interactive learning mode when reading complex corrector papers
- **`math-olympiad`** — adversarial proof verification for corrector convergence proofs (preview; auto-triggers on "prove", "verify proof", "detailed balance")

### Math context
Corrector math in this project = Markov chain theory (spectral gaps, mixing times, Gibbs
sampling, Metropolis-Hastings) applied to discrete/masked token spaces. Key framework: L&Z
E_fact/E_learn decomposition. See `research/proof_worklog.md` for active derivations and
`research/candidate_theorems.md` for the formal theorem stack.

### Not relevant
- **`ui-ux-pro-max`** — already installed, not relevant to this project

---

## Simplicity and source-of-truth rule

Do not create new active docs unless strictly necessary.
Prefer editing existing active docs.
Current entry points: `START_HERE.md` and `docs/README.md`.
Archived files are historical only (`docs/archive/`, `archive/`).
Current research phase: Phase 1 interaction diagnostics. Phase 0 and K=30
replication are complete; Gate 3 is open. No pairwise scheduler or broader HPC
experiments until Gate 3b validates the schedule-level interaction framework.

## Next steps (current — May 2026)

1. Opus theory pass ✅ — Theorem A/B/B′ + Diagnostic Framework C; D/E optional.
2. Phase 0 reproducibility audit ✅ — PF1–PF8: 11/11 on HPC CPU (slnode01, 2026-05-06). K=3 smoke on A100 (gnode01, job 489457) qualitatively matches prior results.
3. K=30 critical replication ✅ — Phase 2b job 490106 complete; Phase 3a job 479941 complete; duplicate job 490469 harmless.
4. Interaction diagnostics ← **current gate.** Gate 3a sparse pair diagnostics, then Gate 3b schedule-level validation.
5. Pairwise scheduler — only if interaction diagnostics show structure.
6. LaTeX writing — ch1–ch6 drafted; ch7 / experiments, Abstract, and Conclusion remain.
7. Polish ch6 after Phase 1 empirical anchors are settled.

Full action plan: `docs/05_next_steps.md`
Theory-first programme: `docs/06_theory_first_research_plan.md`
Thesis LaTeX: `thesis/main.tex`

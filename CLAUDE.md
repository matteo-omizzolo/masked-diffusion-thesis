# CLAUDE.md — Project Context for Claude Code

## Current-status rule (IMPORTANT)

> Use `START_HERE.md` and `docs/README.md` as the entry points for any new session.
> Do NOT use files under `docs/archive/`, `archive/`, or `docs/thesis/` to infer
> current thesis status — those are historical and may contradict current docs.
> Archived files exist for provenance only; they are explicitly superseded.

## Project
MSc thesis on **signal-adaptive corrector scheduling for masked diffusion language models**,
supervised by Prof. Giacomo Zanella (Bocconi University).

**Core research question:** For a fixed predictor schedule and fixed corrector NFE budget
in masked diffusion language models, can aggregate trajectory signals — entropy, confidence
margin, or quality mass — predict the marginal value of a corrective refinement loop well
enough to outperform uniform corrector placement?

**Thesis story (April 2026 baseline):** Fixed-budget corrector allocation is a combinatorial
schedule-search problem. Greedy signal rankers fail by B = 8. Search procedures (CD-G,
BS-AG) recover 49–84 % of oracle headroom on ProSeCo-OWT. Theorem A + Refinements A′/A″
+ Negative-Result Corollary all formally proved.

**Current phase (May 2026):** Theory-first reassessment and Phase 0 reproducibility
planning. Goal: reframe the baseline into a cleaner theory-first study of marginal,
interaction-aware, and online corrector timing. No full-scale HPC runs until the theory
scaffold (Theorem B/C/D) and Phase 0 reproducibility audit are complete.

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
| `docs/05_next_steps.md` | Sequential action plan — theory gates → Phase 0 → writing |
| `docs/06_theory_first_research_plan.md` | Active planning doc — theory-first programme (not final status) |
| `research/candidate_theorems.md` | Full theorem + proof record (keep active) |
| `research/proof_worklog.md` | Derivation entries (keep active) |

## Repo layout
- `START_HERE.md` — 2-minute orientation
- `thesis/` — LaTeX thesis chapters (ch6 skeleton done; ch3/4/5/7 TODO)
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
- **Checkpoint on HPC:** `~/mdm/checkpoints/mdlm.ckpt` (~2.5 GB, uploaded once via scp)
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
bash hpc/submit.sh t1000p  # all 3 strategies on 3 GPUs in 1 job (recommended)
# NOTE: pull.sh uses GNU rsync flags, broken on macOS. Use SSH instead:
ssh 3316152@slogin.hpc.unibocconi.it "python3 -c 'import json; print(...)'"
```

---

## Known HPC environment issues (all fixed as of 2026-03-06)

### 1. setuptools 82 removed `pkg_resources`
`lightning` 2.2.1 calls `pkg_resources` at import time.
**Fix:** `pip install 'setuptools<70'` in the conda env.

### 2. NumPy 2.0 removed `np.float_`
`wandb 0.13.5` used `np.float_`. **Fix:** `pip install 'wandb>=0.16'`

### 3. `flash_attn` — cannot build on this cluster
System CUDA is 13.0 but PyTorch compiled with 12.8 → nvcc mismatch.
**Fix:** Patched `external/remdm/models/dit.py` and `autoregressive.py` to make
`flash_attn` optional with PyTorch SDPA fallback. Also made `dimamba` optional in
`external/remdm/models/__init__.py` (requires `causal_conv1d`, also uninstallable).

### 4. cuda-nvcc conda package breaks sbatch
Installing `cuda-nvcc` via conda adds an activation script that references
`NVCC_PREPEND_FLAGS` (unbound var), crashing `set -u` in sbatch.
**Fix:** `conda remove cuda-nvcc cuda-toolkit` — not needed since flash_attn fallback used.

### 5. PyTorch ≥2.6 `weights_only=True` default
Old checkpoints (saved with numpy scalars) fail to load.
**Fix:** Added to `external/remdm/main.py`:
```python
import numpy
torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
```

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

### 9. flash_attn SDPA fallback — rotary embedding shape error
`apply_rotary_pos_emb` slices cos/sin to half-dim for flash_attn interface.
Torch SDPA fallback needs full-dim. Also `qkv.unbind` uses dim=1 (not dim=2)
after `rearrange(qkv, 'b s ... -> (b s) ...')` gives shape `(b*s, 3, h, d)`.
**Fix:** Patched `external/remdm/models/dit.py` — see file for details.

### 10. NumPy 2.0 `np.array(copy=False)` behavior change in datasets 2.18
`datasets==2.18` uses `np.array(array, copy=False)` which raises ValueError in NumPy 2.x
when a copy is needed.
**Fix:** `pip install 'datasets>=2.21'` (datasets 4.x supports NumPy 2.x).

### 11. OpenWebText reference — use owt-reference streaming solution
Downloading openwebtext (38GB) for MAUVE reference exceeds HPC user quota.
**Fix:** `scripts/stage_owt_reference.py` streams 1000 OWT samples to
`/home/3316152/mdm/data/owt_reference_1000.json` on the login node.
`external/remdm/configs/data/openwebtext-split.yaml` → `valid: owt-reference`.
`external/remdm/dataloader.py` adds `owt-reference` branch.

---

## Current status (May 2026)

See `START_HERE.md` and `docs/00_current_status.md` for full current status.
Key points:

- **Baseline experiments complete.** Phase 1 (signal calibration), Phase 2b (policy
  comparison + MC oracle), Phase 3a (combinatorial search baselines), cross-backbone
  LLaDA-SFT probe, and Protocol C (adaptive controller CPU pilot) are all done.
- **Primary result.** CD-G recovers 74–84 % and BS-AG 49–64 % of +0.45 MC-oracle headroom
  over uniform at B ∈ {2,3,4} on ProSeCo-OWT. Both pass at B = 8.
- **Theory (Phase 3b).** Theorem A + Refinements A′/A″ + Negative-Result Corollary
  formally proved. Theorem A-ad in Appendix F (honest negative Protocol C).
- **Current phase:** Theory-first reassessment. Formalize Theorem B/C/D → Phase 0
  reproducibility audit → interaction diagnostics → writing.
- **No full-scale HPC** until theory scaffold and Phase 0 are complete.
- **Backbone:** ProSeCo-OWT only (LLaDA-SFT probe inconclusive).
- **Reading:** ProSeCo, PRISM, MDLM, ReMDM, L&Z Error Bounds done.

### Writing status
- ch2 (Continuous Diffusion): FIRST DRAFT DONE
- ch6 (Contribution): SKELETON DONE (theorem statements locked, prose TODO)
- ch3 / ch4 / ch5 / ch7: TODO
- Abstract / Introduction / Conclusion: TODO

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
Current research phase: theory-first reassessment and Phase 0 reproducibility planning.
No full-scale HPC runs until the theory scaffold and Phase 0 reproducibility audit are complete.

## Next steps (current — May 2026)

1. Opus theory pass — formalize Theorem B, Proposition C, Theorem D.
2. Phase 0 reproducibility audit — reproduce ProSeCo-OWT baseline locally (K=3 smoke first).
3. Interaction diagnostics — only after Phase 0 passes.
4. Pairwise scheduler — only if interaction diagnostics show structure.
5. LaTeX writing — ch3, ch4, ch5, ch7, Abstract, Introduction, Conclusion.
6. Clean Theorem A proof narrative in ch6.

Full action plan: `docs/05_next_steps.md`
Theory-first programme: `docs/06_theory_first_research_plan.md`
Thesis LaTeX: `thesis/main.tex`

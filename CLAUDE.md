# CLAUDE.md — Project Context for Claude Code

## Project
MSc thesis on **masked diffusion models**. Empirical comparison of three remasking strategies
(MDLM, ReMDM-conf, ReMDM-loop) on OpenWebText. Main metrics: gen_ppl, entropy, MAUVE.

## Repo layout
- `src/mdm_playground/` — main package (`pip install -e .`)
- `external/` — rsync'd copies of upstream repos (remdm, remedi, PRISM, sedd, mdlm)
- `hpc/` — Bocconi HPC workflow scripts
- `configs/` — YAML experiment configs
- `checkpoints/` — local checkpoint storage (gitignored)
- `results/` — experiment outputs (gitignored)
- `figures/` — generated plots
- `docs/` — study documents and literature review
- `scripts/` — analysis and plotting scripts

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

## Current status (2026-03-16) — ALL EXPERIMENTS COMPLETE ✓

All three strategies evaluated at T=128, T=256, T=512, T=1000 steps.

### Full step sweep results
| strategy   | T=128 MAUVE | T=256 MAUVE | T=512 MAUVE | T=1000 MAUVE |
|------------|-------------|-------------|-------------|--------------|
| mdlm       | 0.170       | **0.740**   | 0.592       | 0.590        |
| remdm-conf | 0.440       | 0.475       | 0.470       | 0.325        |
| remdm-loop | 0.396       | 0.614       | 0.532       | **0.684**    |

### T=1000 gen_ppl / entropy
| strategy   | gen_ppl | entropy |
|------------|---------|---------|
| mdlm       |  52.269 |  5.446  |
| remdm-conf |  37.321 |  5.357  |
| remdm-loop |  30.296 |  5.390  |

### Key findings
- **mdlm MAUVE peaks at T=256 (0.740)** — "diversity window" thesis finding
- remdm-conf: diversity collapse confirmed; entropy drops 5.499→5.357 at high T
- remdm-loop: only strategy with monotonically improving MAUVE; best gen_ppl at T≥256
- Figure: `figures/step_sweep.{pdf,png}` | Full analysis: `results/combined_comparison.md`

### Scope decisions (final)
- **RemeDi-RL**: PERMANENTLY SKIPPED. `maple-research-lab/RemeDi-RL` missing `FSDPLLaDAUPMModelLM`
  class (not in any public repo) + 8B vs 100M scale mismatch. Out of thesis scope.
- **PRISM**: Low priority, not yet populated. Skip unless required.
- **Thesis scope**: Remasking strategies on MDLM-scale models only.

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
Corrector math in this project = Markov chain theory (spectral gaps, mixing times, Gibbs sampling, Metropolis-Hastings) applied to discrete/masked token spaces. See `docs/correctors_deep_dive.md` and `notebooks/01_spectral_gap_and_mixing.ipynb`.

### Not relevant
- **`ui-ux-pro-max`** — already installed, not relevant to this project

---

## Next steps
1. Qualitative sample analysis: extract 5-10 samples per strategy from HPC results
2. LaTeX table generation: `scripts/generate_latex_table.py`
3. Thesis chapter draft using `docs/empirical_analysis.md` as structure

# CLAUDE.md — Project Context for Claude Code

## Project
MSc thesis on **masked diffusion models**. Unified inference playground comparing
RemeDi, ReMDM, and PRISM on OpenWebText. Main metric: bits-per-byte (BPB) perplexity.

## Repo layout
- `src/mdm_playground/` — main package (`pip install -e .`)
- `external/` — rsync'd copies of upstream repos (remdm, remedi, PRISM, sedd, mdlm)
- `hpc/` — Bocconi HPC workflow scripts
- `configs/` — YAML experiment configs
- `checkpoints/` — local checkpoint storage (gitignored)

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
bash hpc/submit.sh     # sbatch hpc/remdm_smoke.sbatch
bash hpc/pull.sh       # fetch out/, err/, results/ — NOTE: uses GNU rsync flags, broken on macOS
```

**macOS pull workaround** (pull.sh uses `--ignore-missing-args` which macOS rsync doesn't support):
```bash
ssh 3316152@slogin.hpc.unibocconi.it "cat ~/mdm/masked-diffusion-thesis/out/<file>"
# or rsync manually without --ignore-missing-args
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

### 11. OpenWebText download fails — disk quota on HPC home directory
Downloading openwebtext (38GB) for MAUVE reference exceeds HPC user quota.
**Fix:** Changed `external/remdm/configs/data/openwebtext-split.yaml` to use
`valid: wikitext103` (tiny ~1MB dataset, already in HF cache) for MAUVE reference.
For production eval: use a beegfs scratch partition or pre-download on login node.

---

## Current status (2026-03-15) — T=1000 EVAL COMPLETE ✓

All three strategies evaluated at both T=128 and T=1000 steps. Results in `results/`.

### T=128 results (`results/full_eval/comparison.json`)
| strategy   | gen_ppl | entropy | MAUVE |
|------------|---------|---------|-------|
| mdlm       |  60.914 |  5.507  | 0.170 |
| remdm-conf |  57.579 |  5.499  | 0.440 |
| remdm-loop |  59.632 |  5.538  | 0.396 |

### T=1000 results (`results/t1000_eval/comparison.json`) — job 465445, 2h47m
| strategy   | gen_ppl | entropy | MAUVE |
|------------|---------|---------|-------|
| mdlm       |  52.269 |  5.446  | 0.590 |
| remdm-conf |  37.321 |  5.357  | 0.325 |
| remdm-loop |  30.296 |  5.390  | 0.684 |

### Key finding: MAUVE inversion at T=1000
At T=128: remdm-conf best MAUVE. At T=1000: remdm-loop best, remdm-conf drops to worst.
remdm-conf gen_ppl ~37.3 matches paper Table 1 ✓. MAUVE inversion is a novel thesis finding.
See `results/combined_comparison.md` for full analysis.

## Submit targets
```bash
bash hpc/submit.sh t1000p   # all 3 strategies on 3 GPUs in 1 job (recommended)
bash hpc/submit.sh smoke    # 2-batch smoke test
bash hpc/submit.sh eval     # 128-step array (tasks 0-1)
bash hpc/submit.sh eval-loop  # 128-step task 2
```

## Next experiments
1. ~~Full eval (T=128)~~ ✓ `results/full_eval/`
2. ~~T=1000 eval~~ ✓ `results/t1000_eval/` — matches paper gen_ppl range
3. Step-sweep: T=256/512/1000 for remdm-conf to characterise MAUVE collapse
4. RemeDi evaluation: HF model `maple-research-lab/RemeDi-RL`
5. PRISM: `external/PRISM/` not yet populated — low priority

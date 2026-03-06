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

## Current status (2026-03-06) — SMOKE TEST PASSED ✓
- **Job 465315 COMPLETED** (exit 0, 3:09 elapsed)
- Sampling: 2 batches at ~9.7 it/s on A100
- **gen_ppl: 62.1**, entropy: 5.54
- Generated coherent English text confirmed
- `--remdm_no_mauve` flag added to skip MAUVE for smoke tests
- All env issues resolved; lessons in `tasks/lessons.md`
- `hpc/pull.sh` needs fix for macOS rsync compatibility

## Next experiments
1. ~~Add full eval sweep infrastructure~~ ✓ Done — `hpc/remdm_full_eval.sbatch` (array 0-2), `scripts/aggregate_results.py`
2. Run full eval: `bash hpc/push.sh && bash hpc/submit.sh eval`
3. After jobs complete: `bash hpc/pull.sh && python scripts/aggregate_results.py`
4. Extend to RemeDi and PRISM methods

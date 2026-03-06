# Lessons Learned — Masked Diffusion Thesis

## HPC / Environment

### L1: HPC disk quota is limited — stream reference data, never bulk-download
- OpenWebText (38GB) exceeds HPC home quota; wikitext103 as MAUVE reference gives near-zero scores (wrong domain)
- **Rule:** For MAUVE reference, stream a small OWT slice via `datasets.load_dataset(..., streaming=True)` and stage to a JSON file (~3MB) on login node
- **Fix applied:** `scripts/stage_owt_reference.py` stages 1000 OWT samples to `/home/3316152/mdm/data/owt_reference_1000.json`; `external/remdm/configs/data/openwebtext-split.yaml` → `valid: owt-reference`; `external/remdm/dataloader.py` adds `owt-reference` branch

### L2: PyTorch ≥2.6 changed torch.load default to weights_only=True
- Old checkpoints (with numpy scalars) fail silently or raise UnpicklingError
- **Rule:** Always monkey-patch torch.load when loading legacy checkpoints
- **Fix applied:** `external/remdm/main.py` — `torch.load = _patched_torch_load`

### L3: NumPy 2.x breaks old datasets/wandb code
- `np.float_` removed (wandb 0.13), `np.array(copy=False)` changed (datasets 2.18)
- **Rule:** Pin `datasets>=2.21`, `wandb>=0.16` for NumPy 2.x compat
- **Fix applied:** requirements.txt updated; `pip install datasets>=2.21 wandb>=0.16` on HPC

### L4: transformers checkpoint version must match exactly
- Checkpoint saved with transformers 4.38.2; newer versions moved module paths
- **Rule:** Pin `transformers==4.38.2` for this checkpoint
- **Fix applied:** `pip install transformers==4.38.2` on HPC; requirements.txt pinned

### L5: flash_attn cannot be built on this cluster (CUDA 13.0 vs PyTorch 12.8)
- **Rule:** Always make flash_attn optional with SDPA fallback; never require it
- **Fix applied:** `external/remdm/models/dit.py` — SDPA fallback, full-dim rotary embeddings

### L6: SDPA fallback — rotary embedding dimension mismatch
- `apply_rotary_pos_emb` slices cos/sin to half-dim for flash_attn interface
- Torch SDPA fallback needs full-dim cos/sin with proper broadcasting
- `qkv` after rearrange is `(b*s, 3, h, d)` → `unbind(dim=1)` not `dim=2`
- **Rule:** When adding torch fallback, verify tensor shapes carefully

### L7: Relative paths break when Hydra changes working directory
- Hydra's `hydra.run.dir` changes the cwd of the subprocess
- All file paths passed as overrides must be absolute
- **Fix applied:** `external_dir = (self.run_output_dir / "external_remdm").resolve()`

### L8: setuptools ≥70 removed pkg_resources (needed by lightning 2.2.1)
- **Rule:** Pin `setuptools<70` for lightning compatibility

### L9: conda cuda-nvcc activation script breaks sbatch with set -u
- The activation script uses `NVCC_PREPEND_FLAGS` which is unbound
- **Rule:** Never install cuda-nvcc conda package; use PyTorch's bundled CUDA

## Workflow

### W1: Don't fight MAUVE/data loading for smoke tests
- MAUVE requires reference data download (quota issues) and is slow
- **Rule:** Add `--remdm_no_mauve` flag for smoke tests; only run MAUVE for production eval

### W2: Use the HPC login node for data pre-downloads, not compute nodes
- Compute nodes may have unstable internet or download limits
- **Rule:** Pre-stage large datasets on login node with `nohup python -c "import datasets; datasets.load_dataset(...)" &`

### W3: Stop iterating on hacky fixes — ask "elegant solution?"
- After 2+ failed attempts at a fix, stop and redesign
- **Rule:** If approach keeps failing, use plan mode and redesign from scratch. "Knowing everything I know now, implement the elegant solution."

### W4: Always enter plan mode for non-trivial tasks
- Non-trivial = 3+ steps, architectural decision, or touches multiple files
- **Rule:** Write `tasks/todo.md` plan, get sign-off, then implement

### W5: After any user correction → immediately update tasks/lessons.md
- **Rule:** Every correction → new lesson → new rule. Zero recurrence tolerance.

### W6: Verify before marking done
- **Rule:** Show evidence: log output, test result, or behavior diff. "Would a staff engineer approve this?"

### W7: Use subagents for research/exploration
- **Rule:** Keep main context clean. Offload file exploration, parallel analysis, and research to subagents.

### W8: macOS rsync doesn't support --ignore-missing-args (GNU only)
- **Rule:** For portable rsync scripts, use `ssh host "[ -d path ]"` guard instead.

### W9: macOS grep doesn't support -P (no PCRE)
- `grep -oP 'pattern\Kvalue'` fails silently on macOS (BSD grep)
- **Rule:** Use `grep -o 'full pattern' | grep -o '[0-9]*$'` or `sed` for portable extraction.

### W10: SLURM stud QOS limits each array task as a separate job (MaxSubmitJobsPerUser=2)
- A 3-task array job fails with `QOSMaxSubmitJobPerUserLimit`
- **Rule:** Split large arrays into ≤2 tasks (eval) + follow-up submission (eval-loop). The `%N` concurrency throttle does NOT help — all tasks still count against MaxSubmitJobsPerUser.

### W11: sbatch CLI options must come BEFORE the script name
- `sbatch script.sh --array=2-2` passes `--array=2-2` as a script argument ($1), NOT an sbatch option
- SLURM runs the script with default #SBATCH directives, ignoring the command-line override
- **Rule:** Always put sbatch options before the script: `sbatch --array=2-2 script.sh`
- **Fix applied:** `submit.sh` now uses `EXTRA_SBATCH_OPTS` variable placed before `$SBATCH_FILE`

### W12: MAUVE with wrong reference domain is uninformative
- MAUVE compares generated text to a reference distribution
- Using wikitext103 (Wikipedia) as reference for a model trained on OpenWebText → near-zero MAUVE (~0.006-0.008) for all strategies, differences indistinguishable
- **Rule:** For thesis eval, either use OpenWebText validation split as MAUVE reference or omit MAUVE and rely on gen_ppl. Document this limitation explicitly.

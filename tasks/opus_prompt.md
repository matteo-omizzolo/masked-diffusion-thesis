# Opus Task: Repo Cleanup + Full Eval Workflow

## Context

MSc thesis repo comparing masked diffusion models (ReMDM, RemeDi, PRISM) on
OpenWebText. Main metric: bits-per-byte perplexity + MAUVE. The smoke test
(job 465315, gen_ppl=62.1, entropy=5.54) passed on the Bocconi HPC (A100).

**Today's goal:** two things in order.
1. Fix specific inelegances in the repo (minimal-impact, evidence-based).
2. Run the full eval sweep end-to-end and produce a results table.

---

## Part 1 — Repo cleanup (fix known bugs first, then remove dead code)

### 1a. Critical bug: `run_meta.json` saves wrong strategy for ReMDM

**File:** `src/mdm_playground/cli/run.py`, function `main()` (line ~158)

```python
save_json(out_dir / "run_meta.json", {
    "git_commit": get_git_commit_hash(),
    "method": args.method,
    "strategy": args.strategy,          # ← BUG: this is the Python-level flag
    "model_id": args.model_id,
    "steps": args.steps,
    "seed": args.seed,
    "timestamp": ts,
})
```

For `--method remdm`, `args.strategy` is always the Python-level default
(`"remedi_policy"` or whatever was passed, but it's IGNORED by `_run_remdm`).
The actual ReMDM strategy is `args.remdm_strategy`.

**Fix:** When `method == "remdm"`, save `"remdm_strategy": args.remdm_strategy`
alongside (or instead of) `"strategy"`. The `scripts/aggregate_results.py`
already tries `meta.get("strategy") or meta.get("remdm_strategy")`, so adding
`remdm_strategy` to the meta is enough.

### 1b. Confusing `--strategy` flag for `--method remdm`

In `run.py`, `--strategy` is parsed and passed to `build_strategy()` for RemeDi,
but for ReMDM and PRISM it is completely ignored — only `--remdm_strategy` matters.
The `smoke_all.sh` script passes `--strategy remdm_conf` for `--method remdm`,
which silently does nothing.

**Fix:** In `_run_remdm()`, add a warning if `args.strategy != "remedi_policy"`
(the default) to signal misuse. Also fix `smoke_all.sh` to use
`--remdm_strategy remdm-conf` instead of `--strategy remdm_conf`.

### 1c. Dead code in `src/mdm_playground/`

Audit and remove or keep with a comment. Rules: only delete if definitively unused;
if in doubt, add a `# NOTE: unused until RemeDi/PRISM is wired` comment.

| File | Status | Action |
|------|--------|--------|
| `core/config.py` (`load_yaml`) | Not imported anywhere | Delete |
| `core/masks.py` (`make_mask`) | Not imported anywhere | Delete |
| `core/schedules.py` | Only used in `samplers/block_diffusion.py` | Keep (RemeDi uses it) |
| `core/logging.py` | Used in `cli/run.py` for RemeDi | Keep |
| `strategies/remask.py` | Only used for RemeDi method | Keep |
| `strategies/hybrid.py` (RemediPolicyStrategy) | Only used for RemeDi method | Keep |
| `models/prism.py` | Untested, no external/PRISM | Add prominent `# NOT YET WIRED` comment at top |

Verify with `grep -rn "from.*core.config\|from.*core.masks" src/` before deleting.

### 1d. `requirements.txt` vs `pyproject.toml` duplication

Currently `requirements.txt` repeats all deps from `pyproject.toml`.
The repo uses `pip install -e .` locally (pyproject.toml) but `pip install -r requirements.txt`
on HPC (since pyproject.toml is only for local dev ergonomics).

**This is intentional and correct** — `requirements.txt` is the HPC-specific pinned list.
No change needed. Just add a one-line comment at the top of `requirements.txt`:
```
# HPC install: pip install -r requirements.txt  (pins; takes precedence over pyproject.toml)
```

### 1e. `pyproject.toml` build constraint conflict

`pyproject.toml` declares `requires = ["setuptools>=68"]` but `requirements.txt`
pins `setuptools<70` (to keep `pkg_resources` alive for lightning 2.2.1 on HPC).
These don't conflict in practice (68–69 satisfies both), but add a comment near
the `setuptools` entry in `requirements.txt` to document why it's pinned:
```
setuptools<70  # pkg_resources needed by lightning 2.2.1 (removed in setuptools 70+)
```

### 1f. `docs/` and `notebooks/` — check if populated

Quick `ls docs/ notebooks/` — if empty, add a `.gitkeep` to each or delete them.
Don't populate them speculatively.

---

## Part 2 — Full eval workflow

Once Part 1 fixes are in (especially 1a), run the full eval:

### Step 1: Push code to HPC
```bash
bash hpc/push.sh
```
Verify rsync output shows the fixed files were transferred.

### Step 2: Submit array job
```bash
bash hpc/submit.sh eval
```
Expected output: `Submitted batch job <ARRAY_JOB_ID>` with 3 array tasks
(task IDs `<JOB>_0` = mdlm, `<JOB>_1` = remdm-conf, `<JOB>_2` = remdm-loop).

### Step 3: Monitor
```bash
ssh 3316152@slogin.hpc.unibocconi.it 'squeue -u 3316152'
```
Each task runs ~30-40 min on A100 (100 batches × 128 steps × batch_size=1).
All 3 can run in parallel if GPUs are available; otherwise they queue.

Check logs as they run:
```bash
ssh 3316152@slogin.hpc.unibocconi.it \
  'tail -n 50 ~/mdm/masked-diffusion-thesis/out/remdm_eval_<JOB>_0.out'
```

Watch for:
- `gen_ppl: XX.X` and `entropy: X.XX` in stdout (logged by upstream main.py)
- Any OOM errors (if so, reduce `--remdm_num_batches` to 50)
- MAUVE computation at the end (slow, ~5 min)

### Step 4: Pull results
```bash
bash hpc/pull.sh
```
Verify `results/full_eval/mdlm/`, `results/full_eval/remdm-conf/`,
`results/full_eval/remdm-loop/` exist and each contains
`external_remdm/generated_sequences.json`.

### Step 5: Aggregate
```bash
python scripts/aggregate_results.py --results_dir results/full_eval
```
Expected output: markdown table with gen_ppl, entropy, MAUVE for each strategy.
Also writes `results/full_eval/comparison.json`.

Sanity checks:
- `mdlm` gen_ppl should be ~60-70 (matches smoke test result of 62.1)
- `remdm-conf` gen_ppl should be ≤ mdlm (remasking should improve quality)
- `remdm-loop` gen_ppl should be competitive with remdm-conf
- All entropy values should be ~5-6 bits
- MAUVE scores should be > 0.1 (higher is better, 1.0 = perfect)

---

## Constraints / principles

- **Minimal impact**: only touch files listed above. No speculative refactors.
- **Enter plan mode** for anything that touches 3+ files or has architectural risk.
- **Verify before marking done**: show grep output before deleting, show log tail after submitting.
- **If anything breaks**: stop and diagnose. Do not retry blindly.
- **Update `tasks/lessons.md`** after any non-trivial discovery.

## HPC env reminder
```bash
# Activate:
module load miniconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate remdm311
```
- Host: `3316152@slogin.hpc.unibocconi.it`
- Repo on HPC: `~/mdm/masked-diffusion-thesis`
- Checkpoint: `~/mdm/checkpoints/mdlm.ckpt`

## Key file locations
- CLI: `src/mdm_playground/cli/run.py`
- ReMDM adapter: `src/mdm_playground/models/remdm.py`
- Aggregator: `scripts/aggregate_results.py`
- Full eval sbatch: `hpc/remdm_full_eval.sbatch`
- Lessons: `tasks/lessons.md`

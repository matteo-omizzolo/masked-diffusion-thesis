# Resume Prompt — Masked Diffusion Thesis

## Project
MSc thesis comparing masked diffusion models on OpenWebText.
Local repo: `/Users/matteoomizzolo/masked-diffusion-thesis`
HPC: `3316152@slogin.hpc.unibocconi.it`, repo at `~/mdm/masked-diffusion-thesis`

---

## Current state (as of 2026-03-06 session)

### Full eval results — 100 samples, 128 steps, seed=42

| strategy    | gen_ppl | entropy | MAUVE (wikitext103) |
|-------------|---------|---------|---------------------|
| mdlm        |  60.91  |  5.507  | 0.0082 (uninformative) |
| remdm-conf  |  57.58  |  5.499  | 0.0059 (uninformative) |
| remdm-loop  | **locally missing** — job 465320_2 COMPLETED on HPC, SSH was down at session end |

Local files:
- `results/full_eval/mdlm/external_remdm/generated_sequences.json` ✓
- `results/full_eval/remdm-conf/external_remdm/generated_sequences.json` ✓
- `results/full_eval/remdm-loop/external_remdm/generated_sequences.json` ✗ (needs pull)

### Key findings already confirmed
1. **remdm-conf improves gen_ppl by 5.5%** over mdlm (60.91 → 57.58) ✓
2. **Entropy unchanged** (<0.01 bits diff) — remasking doesn't reduce diversity ✓
3. **MAUVE with wikitext103 is uninformative** — wrong reference domain (Wikipedia vs web text)
4. **Mid-sequence EOS tokens** (~2.3/sample) are expected for doc-concatenated training ✓

---

## First task: pull remdm-loop results

```bash
bash hpc/pull.sh
# then:
python scripts/aggregate_results.py --results_dir results/full_eval
```

Sanity check: remdm-loop gen_ppl should be competitive with remdm-conf (~55-62).

---

## Second task: fix MAUVE reference

Current problem: `external/remdm/configs/data/openwebtext-split.yaml` has `valid: wikitext103`
(changed from OpenWebText to avoid disk quota). wikitext103 is Wikipedia-style text,
completely different domain from the generated web text → MAUVE ~0.006 for all strategies.

Fix options (pick one):
1. **Pre-stage OpenWebText validation on HPC beegfs scratch** (preferred):
   ```bash
   ssh 3316152@slogin.hpc.unibocconi.it
   # on login node:
   mkdir -p /beegfsstudents/home/3316152/data
   conda activate remdm311
   python -c "
   from datasets import load_dataset
   ds = load_dataset('openwebtext', split='train[:1%]', cache_dir='/beegfsstudents/home/3316152/data')
   print('cached', len(ds), 'examples')
   "
   ```
   Then update `openwebtext-split.yaml` to use beegfs path and set `valid: openwebtext`.

2. **Use a larger/more similar reference**: wikitext103 has 4.4M tokens — too sparse for MAUVE.
   Try `wikitext-103-raw-v1` split more carefully or use the OpenWebText 1% split above.

3. **Drop MAUVE** and rely solely on gen_ppl + entropy for the thesis (simplest, defensible).

---

## Third task: extend to RemeDi and PRISM

RemeDi (`--method remedi`):
- Adapter: `src/mdm_playground/models/remedi.py` — loads directly from HF, no subprocess
- Model: `maple-research-lab/RemeDi-RL` (needs HF download on HPC, ~2 GB)
- Requires a prompt; use a fixed set of OpenWebText prompts for fair comparison
- **Not yet run on HPC** — needs a new sbatch

PRISM (`--method prism`):
- Adapter: `src/mdm_playground/models/prism.py` — marked `# NOT YET WIRED`
- `external/PRISM/` does not exist — need to rsync the upstream repo
- Low priority for now

---

## Workflow reminder
```bash
bash hpc/push.sh                          # sync local → HPC
bash hpc/submit.sh [smoke|eval|eval-loop] # submit SLURM job
bash hpc/pull.sh                          # sync HPC results → local
python scripts/aggregate_results.py --results_dir results/full_eval
```

### HPC env
```bash
module load miniconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate remdm311
```

### HPC constraints (already learned)
- stud QOS: MaxSubmitJobsPerUser=2 — max 2 array tasks at once
- `eval` submits tasks 0-1 (mdlm + remdm-conf); `eval-loop` submits task 2 (remdm-loop)
- sbatch CLI options MUST come before the script name: `sbatch --array=2-2 script.sh`

---

## Workflow rules (always follow)
- **Plan mode** for any task with 3+ steps or architectural decisions
- **Elegance**: after 2 failed attempts → stop, re-plan from scratch
- **Verify before marking done**: show log tail, test output, or diff
- **After any correction → update `tasks/lessons.md`**
- **Minimal impact**: only touch what's necessary

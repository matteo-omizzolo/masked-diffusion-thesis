# Next Steps — Masked Diffusion Thesis

**Status as of 2026-03-15:** T=128 and T=1000 evals complete for mdlm / remdm-conf / remdm-loop.
Key finding: MAUVE ranking inverts between T=128 and T=1000.

---

## Priority 1 — Step-sweep: characterise the MAUVE inversion

**Goal:** Understand when and why remdm-conf MAUVE collapses as T increases.
Produces a "steps vs metric" curve — strong thesis material.

### Steps
1. Create `hpc/remdm_sweep.sbatch` — parallel job running all 3 strategies at T=256, T=512
   - Same pattern as `remdm_t1000_parallel.sbatch` (CUDA_VISIBLE_DEVICES=0/1/2)
   - Output dirs: `results/sweep/T256/` and `results/sweep/T512/`
   - Separate jobs for each T (or pack T=256 and T=512 into one job if 6 GPUs aren't available)
2. Submit: `bash hpc/submit.sh sweep256` / `bash hpc/submit.sh sweep512`
3. Pull results, aggregate into sweep table: steps × strategy → gen_ppl + MAUVE
4. Plot (matplotlib): two panels — gen_ppl vs T and MAUVE vs T, one line per strategy
5. Hypotheses to test from the data:
   - Does remdm-conf MAUVE degrade monotonically or is there a cliff (e.g. T>512)?
   - Does remdm-loop MAUVE improve monotonically?
   - Does entropy decrease for remdm-conf (mode collapse signature)?

### Files to create/modify
- `hpc/remdm_sweep.sbatch` (new)
- `hpc/submit.sh` — add `sweep256`, `sweep512` targets
- `scripts/plot_sweep.py` (new) — generate the step-curve figure
- `results/sweep/` — output dir (tracked via comparison.json pattern)

### Expected outcome
A 4-row × 3-col table (T=128/256/512/1000 × 3 metrics per strategy) and a 2-panel figure.
The inversion point for remdm-conf will be visible. If entropy drops for remdm-conf at
high T, that confirms diversity collapse; if entropy is flat, it's a distributional shift
(mode-seeking without diversity loss).

---

## Priority 2 — RemeDi evaluation

**Goal:** Add the RL-finetuned RemeDi model to the comparison table.
HF model: `maple-research-lab/RemeDi-RL`

### Step 1 — Research RemeDi inference API (subagent task)
RemeDi has a different interface from ReMDM (it's RL-finetuned, not remasking-based).
Need to understand:
- How to load and run inference: `model.generate()` or custom sampling loop?
- What config/tokenizer it expects
- Whether it produces token sequences or text directly
- Checkpoint size (for HPC download planning)

Action: Launch a subagent to read the RemeDi repo (`external/remedi/`) and HF model card,
return the inference API signature and example usage.

### Step 2 — Write RemeDi runner
Options (after Step 1 research):
a) **Wrapper in `src/mdm_playground/models/remedi.py`** — adapts RemeDi to the same
   `generate(num_samples, steps) → List[str]` interface used by the ReMDM runner
b) **Standalone script** — if RemeDi API is too different, a self-contained
   `scripts/remedi_eval.py` that writes output in the same summary.json format

Prefer (a) for clean integration. Use (b) if RemeDi requires conflicting deps.

### Step 3 — HPC setup
- Download `maple-research-lab/RemeDi-RL` to HPC: `huggingface-cli download` on login node
  Estimated size: ~500MB–1.5GB (check HF model card)
- Check if RemeDi needs additional packages not in `remdm311` env
- Write `hpc/remedi_eval.sbatch` using the parallel pattern (1 GPU, T=128 and T=1000)

### Step 4 — Run and integrate
- Submit, pull results, add RemeDi row to `results/combined_comparison.md`
- If RemeDi gen_ppl / MAUVE are competitive: noteworthy thesis comparison
- Update `scripts/aggregate_results.py` if needed

### Files to create/modify
- `src/mdm_playground/models/remedi.py` (new) or `scripts/remedi_eval.py` (new)
- `hpc/remedi_eval.sbatch` (new)
- `hpc/submit.sh` — add `remedi` target
- `external/remedi/` — already rsync'd; check what's in it

---

## Priority 3 — Thesis writing support

**Goal:** The analysis chapter needs quantitative evidence and figures.

### 3a — Generate sample text for qualitative analysis
For each strategy at T=1000, extract 5–10 generated samples from
`results/t1000_eval/<strategy>/external_remdm/generated_sequences.json` (on HPC).
Save locally in `results/samples/`. Include in thesis appendix.

### 3b — Step-curve figure (from Priority 1)
Two-panel matplotlib figure:
- Left: gen_ppl vs steps (128/256/512/1000), lines for mdlm/remdm-conf/remdm-loop
- Right: MAUVE vs steps, same lines
Save as `figures/step_sweep.pdf` (vector) and `.png`.

### 3c — Summary comparison table
Final thesis table: all methods × (gen_ppl T=128, gen_ppl T=1000, MAUVE T=128, MAUVE T=1000).
Auto-generated from `results/combined_comparison.md` → LaTeX table via script.

---

## Priority 4 — PRISM (low priority, optional)

`external/PRISM/` exists as a submodule but hasn't been tested on HPC.
PRISM uses a different architecture (score-based). Skip unless thesis scope requires it.
If needed: follow same pattern as RemeDi — research API, write runner, write sbatch.

---

## Execution order

```
Week 1:
  [ ] Priority 1: create sweep sbatch + submit T=256 and T=512
  [ ] Priority 1: pull results, aggregate, plot step curve
  [ ] Priority 2 Step 1: subagent research RemeDi API

Week 2:
  [ ] Priority 2 Steps 2–4: implement + run RemeDi eval
  [ ] Priority 3a: extract sample texts for qualitative analysis

Week 3:
  [ ] Priority 3b: finalize figures
  [ ] Priority 3c: generate LaTeX table
  [ ] Thesis chapter draft using all collected results
```

---

## Open questions / hypotheses for thesis

1. **Why does remdm-conf MAUVE collapse at T=1000?**
   Hypothesis A: Confidence-based remasking becomes overconfident at high T, leading to
   repetitive/mode-seeking output (low diversity → low MAUVE despite low gen_ppl).
   Hypothesis B: The confidence threshold was tuned for low-step regimes; at T=1000 the
   model greedily locks tokens too early, reducing exploration.
   Test: compare entropy (already have it — drops slightly for remdm-conf at T=1000) +
   qualitative text inspection + type-token ratio on generated sequences.

2. **Is remdm-loop's improvement at T=1000 genuine or an artifact of the OWT reference?**
   remdm-loop MAUVE 0.684 > mdlm MAUVE 0.590 at T=1000. Both use the same 1000-sample
   OWT reference. The difference is real but could reflect vocabulary/style differences.
   Test: compute type-token ratio and repetition rate on generated samples.

3. **Does RemeDi (RL-finetuned) outperform all three on MAUVE?**
   RemeDi is explicitly trained to maximise reward on text quality — expected to score
   higher MAUVE. But may sacrifice diversity (lower entropy). Thesis can frame this as
   supervised-fine-tuned vs RL-finetuned trade-off.

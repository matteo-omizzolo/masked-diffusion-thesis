# Next Steps — Masked Diffusion Thesis

**Status as of 2026-03-16:** All empirical experiments complete. Thesis writing is the active priority.

---

## ~~Priority 1 — Step-sweep: characterise the MAUVE inversion~~ ✓ COMPLETE

**Results (job 467809, 2h26m):** See `results/combined_comparison.md` and `figures/step_sweep.{pdf,png}`
- mdlm MAUVE peaks at T=256 (0.740), drops to 0.590 by T=1000
- remdm-conf collapses progressively; entropy drop (5.499→5.357) confirms diversity loss
- remdm-loop monotonically improves; dominates at T≥512

---

## ~~Priority 2 — RemeDi / LLaDa evaluation~~ ✗ PERMANENTLY SKIPPED

**Decision (2026-03-16):** RemeDi-RL evaluation blocked and out of scope.
- `maple-research-lab/RemeDi-RL` references `FSDPLLaDAUPMModelLM` — class not in any public repo
- 8B parameters vs 100M MDLM baseline — not comparable without scale normalisation
- Thesis scope: remasking strategies on MDLM-scale models. External RL-finetuned models are out of scope.
- Document this limitation in thesis Section 5 (Discussion).

---

## Priority 1 (current) — Thesis chapter draft

### 1a — Qualitative sample analysis
Extract 5–10 generated samples per strategy from `results/t1000_eval/<strategy>/external_remdm/generated_sequences.json` (on HPC).
- Use `ssh 3316152@slogin.hpc.unibocconi.it "python3 -c '...'"` to avoid rsync issues
- Save locally in `results/samples/`
- Compute type-token ratio (TTR) and repetition rate as diversity proxies

### 1b — LaTeX table generation
Auto-generate thesis comparison table from `results/combined_comparison.md`.
Script: `scripts/generate_latex_table.py` (to create)

### 1c — Figure updates
Figures already in `figures/step_sweep.{pdf,png}`. Verify fonts/labels are thesis-quality.

---

## Priority 2 — PRISM (low priority, optional)

`external/PRISM/` exists as a submodule but hasn't been tested on HPC.
PRISM uses a different architecture (score-based). Skip unless thesis scope requires it.

---

## Study materials (complete)

- `docs/comparison.md` — 12-paper literature review + unified notation + 48 study questions + 8 research directions
- `docs/empirical_analysis.md` — statistical analysis of step-sweep results + 6 research directions
- PDF/HTML versions in `docs/output/`

---

## Open questions / hypotheses for thesis

1. **Why does remdm-conf MAUVE collapse at T=1000?**
   Hypothesis A: Confidence-based remasking becomes overconfident at high T, leading to
   repetitive/mode-seeking output (low diversity → low MAUVE despite low gen_ppl).
   Hypothesis B: The confidence threshold was tuned for low-step regimes; at T=1000 the
   model greedily locks tokens too early, reducing exploration.
   Test: compare entropy (already have — drops slightly) + TTR on generated sequences.

2. **Is remdm-loop's improvement at T=1000 genuine or an artifact of the OWT reference?**
   remdm-loop MAUVE 0.684 > mdlm MAUVE 0.590 at T=1000. Needs TTR + CI validation.

3. **Is T=256 the diversity-optimal budget for MDLM?**
   MDLM MAUVE peaks at T=256 then drops. Bootstrap CI will confirm if peak is real.

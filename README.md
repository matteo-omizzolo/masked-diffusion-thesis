# Masked Diffusion Thesis

MSc thesis: empirical comparison of remasking strategies for masked diffusion language models
(MDLM, ReMDM-conf, ReMDM-loop) on OpenWebText across T=128/256/512/1000 diffusion steps.

---

## Key results

| strategy   | T=128 MAUVE | T=256 MAUVE | T=512 MAUVE | T=1000 MAUVE | T=1000 gen_ppl |
|------------|-------------|-------------|-------------|--------------|----------------|
| mdlm       | 0.170       | **0.740**   | 0.592       | 0.590        | 52.3           |
| remdm-conf | 0.440       | 0.475       | 0.470       | 0.325        | 37.3           |
| remdm-loop | 0.396       | 0.614       | 0.532       | **0.684**    | 30.3           |

See `results/combined_comparison.md` and `figures/step_sweep.{pdf,png}` for full analysis.

---

## Repo structure

```
src/mdm_playground/     # Main package (pip install -e .)
external/               # Upstream repos (remdm, remedi, PRISM, sedd, mdlm) — rsync'd, NOT cloned
hpc/                    # Bocconi HPC workflow scripts (push.sh, submit.sh, pull.sh)
configs/                # YAML experiment configs
scripts/                # Analysis and plotting scripts
docs/                   # Study documents
  comparison.md         # 12-paper literature review + unified notation + research directions
  empirical_analysis.md # Statistical analysis of step-sweep results
  output/               # PDF and HTML versions of docs
results/                # Experiment outputs (gitignored)
figures/                # Generated plots
tasks/                  # Planning and lessons learned
```

---

## Study materials

- `docs/comparison.md` — comprehensive literature review of 12 discrete diffusion papers,
  unified notation table, 48 study questions, 8 research directions
- `docs/empirical_analysis.md` — statistical interpretation of step-sweep results,
  formal metric definitions, 6 research directions
- PDF/HTML versions in `docs/output/`

---

## HPC workflow (Bocconi cluster)

```bash
bash hpc/push.sh                # rsync code to HPC (excludes checkpoints, results)
bash hpc/submit.sh t1000p       # run all 3 strategies in parallel (3 GPUs, 1 job slot)
# Pull results via SSH (rsync broken on macOS):
ssh 3316152@slogin.hpc.unibocconi.it "python3 -c 'import json; ...'"
```

See `CLAUDE.md` for full environment setup, known HPC issues, and fixes.

---

## Main package

```bash
pip install -e .
```

`src/mdm_playground/` contains the inference runner, model adapters (ReMDM, MDLM),
strategy implementations (baseline, confidence remasking, loop remasking), and metrics.

---

## Analysis scripts

```bash
python scripts/aggregate_results.py --results_dir results/full_eval  # print metrics table
python scripts/plot_sweep.py                                           # regenerate step-curve figure
python scripts/stage_owt_reference.py                                  # stage OWT reference on HPC
```

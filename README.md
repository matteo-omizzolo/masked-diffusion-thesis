# Signal-Adaptive Corrector Scheduling for Masked Diffusion Language Models

MSc thesis by Matteo Omizzolo, supervised by Prof. Giacomo Zanella (Bocconi University).

---

## Research Question

> For a fixed predictor schedule and fixed corrector NFE budget in masked diffusion
> language models, can aggregate trajectory signals — entropy, confidence margin, or
> quality mass — predict the marginal value of a corrective refinement loop well enough
> to outperform uniform corrector placement?

See `docs/thesis_direction.md` for the full research direction, scope, and non-goals.

---

## Repository Structure

```
thesis/                  # LaTeX thesis chapters (ch2 first draft done)
research/                # Mathematical worklog, candidate theorems, proof ledger
docs/                    # Core documentation
  thesis_direction.md    #   Precise research question, scope, non-goals
  literature_map.md      #   Categorized paper map with gap analysis
  reading_plan.md        #   Prioritized reading list with status tags
  experimental_infrastructure.md  # Repos, checkpoints, setup status
  implementation_plan.md #   Phased experiment roadmap
  legacy_cleanup_log.md  #   Archive/deletion log
  md/                    #   Older study documents (some deprecated)
  instructions/          #   Claude Code prompts and thesis brief
src/mdm_playground/      # Main Python package (pip install -e .)
external/                # Upstream repos (remdm, mdlm, PRISM, sedd, remedi)
study/                   # Papers organized by topic, notes, guidelines
scripts/                 # Analysis and plotting scripts
configs/                 # YAML experiment configs
hpc/                     # Bocconi HPC workflow scripts
notebooks/               # Jupyter notebooks (spectral gap, discretization error)
archive/                 # Deprecated material (old directions, notes)
figures/                 # Generated plots
```

---

## Current Status (April 2026)

The thesis direction has been refocused from a broad "informed correctors" framing to a
precise question about **trajectory-level fixed-budget corrector allocation**. The key
distinction from adjacent work: this thesis targets corrector *scheduling* (when to spend
corrective effort), not token-selection policies (which tokens to correct), predictor
schedules (when to unmask), or corrector kernel design (how to correct).

**Writing:** ch2 (Background: Continuous Diffusion) first draft complete.
**Theory:** Proof worklog started; see `research/`.
**Experiments:** MDLM-OWT checkpoint available; ReMDM codebase patched for HPC; ProSeCo
setup in progress.

---

## Key Papers Read

MDLM, ReMDM, Zhao et al. (Informed Correctors), PRISM, L&Z Error Bounds.
See `docs/reading_plan.md` for the full prioritized list.

---

## HPC Workflow (Bocconi Cluster)

```bash
bash hpc/push.sh              # rsync code to HPC
bash hpc/submit.sh t1000p     # submit job (3 strategies, 3 GPUs)
```

See `CLAUDE.md` for full environment setup, known issues, and fixes.

---

## Previous Empirical Results (Archived)

Step-sweep experiments (MDLM/ReMDM-conf/ReMDM-loop, T=128/256/512/1000) are archived
from the earlier thesis phase. Results in `results/combined_comparison.md` and
`figures/step_sweep.{pdf,png}`.

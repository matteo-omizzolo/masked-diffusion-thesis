# Experimental Infrastructure

**Updated:** April 2026

---

## Tier A — Must-Have

### MDLM
- **Role:** Base masked diffusion language model; reproducible baseline.
- **Repo:** `https://github.com/kuleshov-group/mdlm`
- **Checkpoint:** `kuleshov-group/mdlm-owt` (HuggingFace, ~130M params, Apache 2.0)
- **Status:** Already on HPC (`~/mdm/checkpoints/mdlm.ckpt`). Code in `external/mdlm/`.
- **Use:** Baseline sampling, trajectory diagnostics, evaluation.

### ReMDM
- **Role:** Practical inference-time scaling and remasking framework.
- **Repo:** `https://github.com/kuleshov-group/remdm`
- **Status:** Already on HPC, patched for flash_attn/SDPA fallback. Code in `external/remdm/`.
- **Use:** Fast experimentation platform; entropy diagnostics; remasking baselines.
  Instrument for per-step corrector allocation experiments.

### ProSeCo
- **Role:** Best direct platform for corrector scheduling experiments.
- **Repo:** `https://github.com/kuleshov-group/proseco`
- **Checkpoints:** `kuleshov-group/proseco-owt` (~130M), `kuleshov-group/proseco-llada-sft`
- **Status:** Not yet cloned. **Priority: clone and assess.**
- **Use:** Primary experimental platform for corrector-scheduling ablations. Exposes
  correction schedule knobs (frequency, loop count, start time). Compare fixed-budget
  allocations directly.

---

## Tier B — Strongly Useful

### PRISM
- **Role:** Quality-signal and self-correction platform.
- **Repo:** `https://github.com/SeunggeunKimkr/PRISM`
- **Status:** In `external/PRISM/` as submodule; not tested on HPC.
- **Use:** Second-stage experiments after entropy/confidence baselines. Quality mass
  signal.

### dLLM
- **Role:** Broader diffusion LM framework.
- **Repo:** `https://github.com/ZHZisZZ/dllm`
- **Status:** Not cloned.
- **Use:** Optional common framework if experiments need generalization.

---

## Tier C — Optional Validation Targets

### LLaDA
- **Role:** Larger-scale validation (~8B params).
- **Repo:** `https://github.com/ML-GSAI/LLaDA`
- **Status:** Not cloned; out of HPC compute budget for training.
- **Use:** Only after small-model results are stable; inference-only experiments.

### Dream
- **Role:** Larger diffusion LM validation.
- **Repo:** `https://github.com/DreamLM/Dream`
- **Status:** Not cloned.
- **Use:** Optional late-stage external validation.

---

## Additional Codebases (Analysis / Comparison)

| Codebase | Repo | Use | Status |
|----------|------|-----|--------|
| Denoising Entropy | `LINs-lab/DenoisingEntropy` | Per-step entropy H_DE computation | Confirmed public (Dec 2025) |
| KLASS | `shkim0116/KLASS` | KL-based selection baseline | NeurIPS 2025 Spotlight |
| Ψ-Samplers (Duo) | `s-sahoo/duo` | General PC framework | ICLR 2026 |
| MDPO/RCR | `autonomousvision/mdpo` | Running Confidence Remasking | Aug 2025 |
| Discrete FK Correctors | `hasanmohsin/discrete_fkc` | SMC-based correctors | Jan 2026 |

---

## Pretrained Checkpoints Confirmed

| Checkpoint | Source | Params | Notes |
|------------|--------|--------|-------|
| `kuleshov-group/mdlm-owt` | HuggingFace | ~130M | Primary; already on HPC |
| `kuleshov-group/proseco-owt` | HuggingFace | ~130M | ProSeCo OWT model; **download next** |
| `kuleshov-group/bd3lm-owt-*` | HuggingFace | ~130M | Block diffusion variant |
| `louaaron/sedd-small` / `sedd-medium` | HuggingFace | 100M / 400M | CTMC baseline |

---

## HPC Environment

- **Host:** `slogin.hpc.unibocconi.it` | **User:** `3316152`
- **Partition/QOS:** `stud` | Max 4× NVIDIA A100 80GB
- **Python env:** conda `remdm311` (Python 3.11)
- **Known issues:** All documented in `CLAUDE.md` (11 issues, all fixed as of 2026-03-06)

---

## External Repo Manifest

Machine-readable manifest for reproducibility:

```yaml
# external_repos.yaml
tier_a:
  mdlm:
    repo: https://github.com/kuleshov-group/mdlm
    status: cloned
    location: external/mdlm/
    checkpoint: kuleshov-group/mdlm-owt
    checkpoint_status: on_hpc

  remdm:
    repo: https://github.com/kuleshov-group/remdm
    status: cloned_and_patched
    location: external/remdm/
    checkpoint: kuleshov-group/mdlm-owt  # shares MDLM checkpoint
    checkpoint_status: on_hpc

  proseco:
    repo: https://github.com/kuleshov-group/proseco
    status: not_yet_cloned
    checkpoint: kuleshov-group/proseco-owt
    checkpoint_status: not_downloaded

tier_b:
  prism:
    repo: https://github.com/SeunggeunKimkr/PRISM
    status: submodule_not_tested
    location: external/PRISM/

  dllm:
    repo: https://github.com/ZHZisZZ/dllm
    status: not_cloned
```

---

## Zhao et al. Code Status

`lindermanlab/informed-correctors` exists but is **not usable** — 2 commits, no README,
WIP since 2024. The best existing implementations of confidence-based correctors are
ReMDM (confidence remasking) and MDPO (Running Confidence Remasking).

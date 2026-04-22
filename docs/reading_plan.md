> **STATUS:** SUPPORTING
> **LAST VERIFIED:** 2026-04-22
> **SCOPE:** Ordered reading queue with per-paper status (read / to read / skipped).

---

# Reading Plan: Signal-Adaptive Corrector Scheduling

**Updated:** April 2026

> Built on top of existing reading state inferred from repo notes, papers.md, and
> research_plan.md. Tags: status, function, and theory-vs-experiments relevance.

---

## Completed — Do Not Reread

| Paper | arXiv | Function | Notes |
|-------|-------|----------|-------|
| MDLM (Sahoo et al. 2024) | 2406.07524 | backbone | SUBS parameterization, factorization approximation |
| ReMDM (Gat et al. 2024) | 2503.00307 | remasking, experiments | Conf/loop/cap/rescale variants; already on HPC |
| Zhao et al. / Informed Correctors (2024) | 2407.21243 | corrector design | Barker/MPF correctors; Gap D baseline; viva 2026-04-10 |
| PRISM (Kim et al. 2025) | 2510.01384 | corrector design, theory | Zanella-recommended; provably learnable quality scores |
| L&Z Error Bounds (Lavenant & Zanella 2025) | 2510.25544 | predictor scheduling, theory | E_fact decomposition; EB-Sampler; Gap E future work; read 2026-04-11 |

---

## Priority Reading Queue

Ordered by importance to the current thesis formulation.

### Tier 1 — Must read before Zanella meeting

| # | Paper | arXiv | Function | Why |
|---|-------|-------|----------|-----|
| 1 | **ProSeCo** (Schiff et al. 2026) | 2602.11590 | corrector scheduling, experiments | Closest existing corrector-scheduling paper; primary experimental platform; need to understand schedule knobs and baselines |
| 2 | **EAGS** (Koh et al. 2024) | 2411.06438 | corrector scheduling | Closest existing work to Gap B/C; need to know exact differences from MDM corrector setting |
| 3 | **Ascolani, Lavenant & Zanella 2024** | 2410.00858 | theory | Per-step KL contraction for Gibbs; primary proof template for Gap E |
| 4 | **Denoising Entropy** (Chen et al.) | 2512.21336 | corrector scheduling, experiments | Entropy signal and code for experiments; Gap B/C mechanism |

### Tier 2 — Read soon after Tier 1

| # | Paper | arXiv | Function | Why |
|---|-------|-------|----------|-----|
| 5 | **KLASS** (Kim et al. NeurIPS 2025) | 2511.05664 | token selection | KL-based selection beats confidence; comparison baseline; code available |
| 6 | **Ψ-Samplers / Diffusion Duality II** (Deschenaux et al.) | 2602.21185 | theory, corrector design | General PC framework; factorization tightness; code `s-sahoo/duo` |
| 7 | **DFM** (Gat et al. NeurIPS 2024) | 2407.15595 | corrector design, theory | Corrector-as-Gibbs vocabulary; mixing rate = spectral gap |
| 8 | **Train for the Worst** (Kim et al. ICML 2025) | 2502.06768 | theory | Formal separation adaptive vs random; Gap D theory context |

### Tier 3 — Skim / assess relevance

| Paper | arXiv | Function | Why |
|-------|-------|----------|-----|
| UPO (Hong et al.) | 2510.05725 | token selection | RL/MDP token selection; Gap D crowding |
| MDPO/RCR (He et al.) | 2508.13148 | token selection | Running Confidence Remasking; Gap D crowding |
| CoDD (Li et al.) | 2603.00045 | theory | Breaks factorization; Gap E positioning |
| Optimal Inference Schedules (Chen et al.) | 2511.04647 | predictor scheduling | Information-theoretic bounds |
| DEMASK (April 2026) | 2604.02560 | token selection | Dependency-graph selection; possible proof overlap |
| Debiasing Guidance with SMC | 2502.06079 | corrector design | ICLR 2025; SMC for discrete diffusion |
| Particle Gibbs for DLMs | 2507.08390 | theory | Trajectory Gibbs; trajectory-level comparison |

### Background — Consult as needed

| Paper | Function | Notes |
|-------|----------|-------|
| EB-Sampler (Ben-Hamu et al.) | predictor scheduling | Optimal unmasking-only baseline |
| RemeDi (Huang et al.) | remasking | Confidence-guided remasking |
| Zanella 2020 | theory (vocabulary) | Locally-balanced proposals |
| Secchi & Zanella 2025 | theory | MwG spectral gap decomposition |
| Adapting the Gibbs Sampler (2018) | theory | Non-uniform scan allocation |
| SEDD (Lou et al.) | backbone | Score entropy; positioning only |
| Roberts & Sahu 1997 | theory | Blocking and ergodicity for Gibbs |
| Diaconis & Saloff-Coste 1993 | theory | Comparison theorems for reversible MCs |

---

## Papers Not Yet Downloaded (Watchlist)

| Paper | arXiv | Target folder | Priority |
|-------|-------|---------------|----------|
| Denoising Entropy (Chen et al.) | 2512.21336 | `study/papers/03-remasking/` | **High** |
| ProSeCo (Schiff et al.) | 2602.11590 | `study/papers/04-informed-correctors/` | **High** |
| EAGS (Koh et al.) | 2411.06438 | `study/papers/04-informed-correctors/` | **High** |
| DEMASK | 2604.02560 | `study/papers/04-informed-correctors/` | Medium |
| CoDD (Li et al.) | 2603.00045 | `study/papers/03-remasking/` | Medium |
| Absorb & Converge | 2506.02318 | `study/papers/05-mcmc-theory/` | Medium |
| Particle Gibbs for DLMs | 2507.08390 | `study/papers/05-mcmc-theory/` | Medium |

---
title: "Paper Inventory: Informed Correctors for Masked Diffusion Models"
date: "Updated April 2026"
---

> This inventory follows the `study/papers/` folder structure as source of truth.
> Papers marked ✓ are downloaded. Papers marked ○ are on the watchlist (not yet downloaded).
> Reading status: ★ = read and tested, → = in progress, · = queued, ~ = skim/assess.

---

# 01 — Foundational Diffusion

Background for ch2_background_diffusion.tex. These papers establish the continuous
diffusion framework that masked diffusion builds on.

| Paper | File | Read for thesis | Role |
|-------|------|-----------------|------|
| Score Matching (Song & Ermon 2019) | `Score-Matching_Song_Ermon_2019.pdf` | Background | Denoising score matching objective |
| Improved Score-Based (Song & Ermon 2020) | `Improved-Score-Based_Song_Ermon_2020.pdf` | Background | Langevin dynamics + annealing |
| Score-Based SDEs (Song et al. 2021) | `Score-Based-SDEs_Song_et_al_2021.pdf` | Background | SDE framework; forward/reverse process |
| DDPM (Ho et al. 2020) | `DDPM_Ho_et_al_2020.pdf` | Background | Denoising diffusion; VLB |
| Max-Likelihood Score-Based (Song et al. 2021) | `Maximum-Likelihood-Score-Based_Song_et_al_2021.pdf` | Background | MLE connection |
| DPM-Solver (Lu et al. 2022) | `DPM-Solver_Lu_et_al_2022.pdf` | Background | Fast ODE solver for diffusion; background only |
| CFG (Ho & Salimans 2022) | `Classifier-Free-Guidance_Ho_Salimans_2022.pdf` | Background | Conditional generation; background only |

**Usage:** These are background references for ch2. Do not read them as primary papers —
use them as citation sources when writing the background chapter.

---

# 02 — Masked Diffusion

Core backbone papers. These are used for ch3 and ch4.

| Paper | File | Status | Role |
|-------|------|--------|------|
| MDLM (Sahoo et al. 2024, NeurIPS) | `MDLM_Sahoo_et_al_2024.pdf` | ★ READ | Backbone model; SUBS parameterization; factorization approximation |
| SEDD (Lou et al. 2024) | `SEDD_Lou_et_al_2024.pdf` | Background | Score entropy for discrete diffusion; positioning only |

---

# 03 — Remasking and Schedule Papers

These cover the predictor side (unmasking schedules) and the infrastructure on which
correctors are applied.

| Paper | File | Status | Role |
|-------|------|--------|------|
| L&Z Error Bounds v1 (Lavenant & Zanella 2025) | `L&Z-Error-Bounds_Lavenant_Zanella_2025.pdf` | → IN PROGRESS | E_fact/E_learn decomposition; Section 6 opens Gap E |
| L&Z Error Bounds v2 (Lavenant & Zanella 2025) | `L&Z-Error-Bounds_v2_Lavenant_Zanella_2025.pdf` | → IN PROGRESS | Use v2 — more recent |
| EB-Sampler (Ben-Hamu et al. 2025) | `EB-Sampler_Ben-Hamu_et_al_2025.pdf` | · Queued | Optimal unmasking-only baseline; Gap D's ceiling to beat |
| ReMDM (Gat et al. 2024) | `ReMDM_Gat_et_al_2024.pdf` | ★ READ | Remasking schedules; conf/loop/cap/rescale; infrastructure |
| CDLM (NeurIPS 2025) | `CDLM_NeurIPS2025.pdf` | ~ Assess | Remasking adjacent; skim to assess |
| DSL — Discrete Stochastic Localization (2026) | `DSL-Discrete-Stochastic-Localization_2026.pdf` | ~ Assess | Localization-based approach; skim to assess |
| Self-Reflective Remasking (Huang et al. 2025) | `Self-Reflective-Remasking_Huang_et_al_2025.pdf` | ~ Assess | Empirical adjacent; skim |

**Denoising Entropy (Chen et al., arXiv:2512.21336)** — ○ Not yet downloaded
Gap B/C mechanism; applies entropy to predictor (unmasking order). No paper applies this
to corrector scheduling. Add to `03-remasking/` when acquired.

---

# 04 — Informed Correctors

Core papers for the thesis contribution. These define the problem space.

| Paper | File | Status | Role |
|-------|------|--------|------|
| Zhao et al. 2024 (ICLR 2025) | `Informed-Correctors_Zhao_et_al_2024.pdf` | ★ READ | Defines Gap D; Barker/MPF correctors; no optimality proof |
| PRISM (Kim et al. 2025) — **Zanella-recommended** | `PRISM_Kim_et_al_2025.pdf` | ★ READ | Provably learnable quality scores; adjacent to Gap D |
| DFM (Gat et al. 2024, NeurIPS) | `DFM_Gat_et_al_2024_NeurIPS.pdf` | · Queued | Corrector-as-Gibbs vocabulary; mixing rate = spectral gap |
| IterRef-MTM (Lee et al. 2025) | `IterRef-MTM_Lee_et_al_2025.pdf` | ~ Assess | Multiple-try MH in iterative refinement; relevant if MH framing used |
| Token-Ordering (Kim et al. 2025) | `Token-Ordering_Kim_et_al_2025.pdf` | ~ Assess | Adjacent to Gap D (ordering ≈ selection); brief read |
| Diffusion-EAGS (Koh et al. 2024) | `Diffusion-EAGS_Koh_et_al_2024.pdf` | ~ Assess | Assess relevance to corrector design |
| Diffusion Duality II / Ψ-Samplers (Deschenaux et al. 2026) | `Diffusion-Duality-II_Psi-Samplers_Deschenaux_et_al_2026.pdf` | ~ Assess | Duality between predictor and corrector; assess |
| ProSeCo (Schiff et al. 2026) | `ProSeCo_Schiff_et_al_2026.pdf` | Background | Training-time self-correction; positioning only |

**Papers not yet in repo:**
- DEMASK (arXiv:2604.02560) — ○ TV bound under sub-additivity for dependency-guided
  selection; possible proof strategy overlap with Gap D. Acquire and add here.
- Hong et al. (arXiv:2510.05725) — ○ Confidence improvable via RL; empirical contrast
  (shows confidence is not the best but does not prove what optimal is).

---

# 05 — MCMC Theory

Theory foundations for the informed-corrector proof track. **Papers selected by topic fit
and proof utility — not by authorship.**

See `docs/md/research_plan.md` Theory Track section for the full classification rationale.

| Paper | File | Classification | Role |
|-------|------|---------------|------|
| Ascolani, Lavenant & Zanella 2024 | `Entropy-Contraction-Gibbs_Ascolani-Lavenant-Zanella_2024.pdf` | **Directly useful** | Per-step KL contraction for Gibbs; primary proof reference for Gap D/E |
| Secchi & Zanella 2025 | `Spectral-Gap-MwG_Secchi-Zanella_2025.pdf` | Secondary | MwG spectral gap decomposition; useful if corrector is framed as MwG |
| Zanella 2020 | `Informed-Proposals-Discrete-MCMC_Zanella_2020.pdf` | Vocabulary | Locally-balanced proposals; read to understand contrast with confidence selection |

**Papers not yet in repo (acquire if needed for proof):**
- Roberts & Sahu 1997, "Updating Schemes, Covariance Structure, and Optimal Blocking and
  Ergodicity for Gibbs Samplers" (JRSS-B) — ○ **Directly useful**; about which coordinates
  to update in Gibbs and how the choice affects mixing. Most directly relevant non-Zanella
  reference for Gap D.
- Diaconis & Saloff-Coste 1993, "Comparison Theorems for Reversible Markov Chains"
  (Annals of Applied Probability) — ○ **Directly useful**; comparison technique for
  spectral gaps; natural tool for Gap(informed) ≥ Gap(uniform) argument.
- Levin, Peres & Wilmer, "Markov Chains and Mixing Times" (textbook) — ○ Secondary;
  consult as needed for definitions and canonical results.

---

# 06 — Evaluation

| Paper | File | Role |
|-------|------|------|
| MAUVE (Pillutla et al. 2021) | `MAUVE_Pillutla_et_al_2021.pdf` | Primary evaluation metric; already in use |

---

# 07 — References / Surveys

| Paper | File | Role |
|-------|------|------|
| Achilli MSc Thesis | `Achilli_MSc_Thesis_Reference.pdf` | Reference for thesis structure/style |
| Principles of Diffusion Models (survey) | `Principles-of-Diffusion-Models_Survey.pdf` | Background survey; use for ch2 |

**Papers not yet in repo:**
- Top 10 Open Challenges in DLMs (arXiv:2601.14041) — ○ Field survey; add to this folder
  if acquired
- Complexity Theory of Masked Discrete Diffusion (arXiv:2509.21835) — ○ Optional background

---

# Watchlist Summary

Papers to acquire (not yet in `study/papers/`):

| Paper | arXiv | Target folder | Priority |
|-------|-------|--------------|----------|
| Denoising Entropy (Chen et al.) | 2512.21336 | `03-remasking/` | High — Gap B/C |
| Absorb & Converge (2025) | 2506.02318 | `05-mcmc-theory/` | Medium |
| Particle Gibbs for DLMs (2025) | 2507.08390 | `05-mcmc-theory/` | Medium |
| DEMASK (2026) | 2604.02560 | `04-informed-correctors/` | Medium |
| Roberts & Sahu 1997 | JRSS-B | `05-mcmc-theory/` | High if Gap D proof proceeds |
| Diaconis & Saloff-Coste 1993 | Ann. Appl. Prob. | `05-mcmc-theory/` | High if comparison approach used |
| Hong et al. (2025) | 2510.05725 | `04-informed-correctors/` | Low |

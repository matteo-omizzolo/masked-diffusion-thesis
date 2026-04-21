# Literature Map: Signal-Adaptive Corrector Scheduling

**Updated:** April 2026

> Organized by function relative to the thesis question. Each paper is tagged with its
> primary contribution and what gap remains for this thesis.

---

## Category 1 — Corrector Design and Self-Correction

These papers define the corrector object and mechanisms for corrective refinement.

| Paper | arXiv | Status | What it solves | Gap for this thesis |
|-------|-------|--------|----------------|---------------------|
| **Zhao et al. (Informed Correctors)** | 2407.21243 | **Read** | Barker/MPF correctors for discrete diffusion; defines informed corrector framework | Not a scheduling paper; does not address trajectory-level budget allocation |
| **PRISM** (Kim et al. 2025) | 2510.01384 | **Read** | Provably learnable per-token quality scores for self-correction | Quality signal is at the token level; no trajectory-level scheduler |
| **ProSeCo** (Schiff et al. 2026) | 2602.11590 | To read next | Trains models to both unmask and correct; exposes schedule knobs (frequency, loop count, start time) | Does not derive a principled scheduler or prove signal-adaptive > uniform |
| **DFM** (Gat et al. NeurIPS 2024) | 2407.15595 | Queued | Corrector-as-Gibbs vocabulary; mixing rate = spectral gap | Corrector design, not scheduling |
| **IterRef-MTM** (Lee et al. 2025) | — | Skim | Multiple-try MH in iterative refinement | Relevant only if MH framing is used |

## Category 2 — Remasking Methods

These revisit tokens within the denoising process. **Distinct from corrector scheduling.**

| Paper | arXiv | Status | What it solves | Gap for this thesis |
|-------|-------|--------|----------------|---------------------|
| **ReMDM** (Gat et al. 2024) | 2503.00307 | **Read** | Remasking schedules (conf/loop/cap/rescale); practical inference-time scaling | Remasking, not corrector scheduling; useful as experimental platform |
| **RemeDi** (Huang et al. 2025) | — | Skim | Confidence-guided remasking framework | Remasking signal; not fixed-budget corrector allocation |
| **Denoising Entropy** (Chen et al.) | 2512.21336 | To read next | Entropy applied to predictor (unmasking order) | Predictor-side entropy; authors explicitly do not extend to corrector scheduling |

## Category 3 — Predictor / Unmasking Schedule Optimization

These optimize the predictor path. **Distinct from corrector scheduling.**

| Paper | arXiv | Status | What it solves | Gap for this thesis |
|-------|-------|--------|----------------|---------------------|
| **L&Z Error Bounds** (Lavenant & Zanella 2025) | 2510.25544 | **Read** | E_fact/E_learn decomposition; information-profile view for predictor scheduling; Section 6 opens Gap E | Predictor scheduling, not corrector; Gap E is future work |
| **EB-Sampler** (Ben-Hamu et al. 2025) | — | Queued | Optimal unmasking-only baseline; entropy-bounded token selection | Predictor-side entropy; ceiling for unmasking-only |
| **Optimal Inference Schedules** (Chen et al. 2025) | 2511.04647 | Skim | Sharp information-theoretic schedule bounds | Solves the predictor schedule problem, not corrector timing |
| **DUS / Plan for Speed** (2025) | — | Background | Dilated unmasking schedule | Predictor-side; useful adjacent reference |

## Category 4 — Token-Selection Policies

These decide *which* tokens to update. **Distinct from corrector scheduling.**

| Paper | arXiv | Status | What it solves | Gap for this thesis |
|-------|-------|--------|----------------|---------------------|
| **KLASS** (Kim et al. NeurIPS 2025) | 2511.05664 | To read | KL-based selection outperforms confidence | Token selection, not trajectory scheduling |
| **UPO** (Hong et al.) | 2510.05725 | Skim | RL/MDP formulation of token selection | Token selection with formal objective; Gap D crowding |
| **DEMASK** (April 2026) | 2604.02560 | Skim | Dependency-graph-based token selection | Token selection; possible proof overlap |
| **Learning Unmasking Policies** (Apple) | 2512.09106 | Background | RL-trained selection policy | Token selection; Gap D crowding |

## Category 5 — MCMC Theory Foundations

Theory papers most likely to supply proof ingredients.

| Paper | arXiv | Status | What it solves | Relevance to thesis |
|-------|-------|--------|----------------|---------------------|
| **Ascolani, Lavenant & Zanella 2024** | 2410.00858 | To read next | Per-step KL contraction for Gibbs under log-concavity | Primary proof template for Gap E |
| **Zanella 2020** (Informed Proposals) | — | Background | Locally-balanced proposals for discrete MCMC | Vocabulary for informed local moves |
| **Secchi & Zanella 2025** | — | Background | MwG spectral gap decomposition | Useful if corrector is framed as MwG |
| **Adapting the Gibbs Sampler** (2018) | — | Background | Non-uniform scan allocation and update-effort optimization | Schedule-allocation lens analogy |

## Category 6 — Backbone Models

| Paper | arXiv | Status | Role |
|-------|-------|--------|------|
| **MDLM** (Sahoo et al. NeurIPS 2024) | 2406.07524 | **Read** | Base masked diffusion LM; experimental backbone |
| **SEDD** (Lou et al. 2024) | — | Background | Score entropy for discrete diffusion |

## Category 7 — Evaluation

| Paper | arXiv | Status | Role |
|-------|-------|--------|------|
| **MAUVE** (Pillutla et al. 2021) | — | **Read** | Primary evaluation metric |

## Category 8 — Entropy-Adaptive Gibbs (Closest to Gap B/C)

| Paper | arXiv | Status | What it solves | Gap for this thesis |
|-------|-------|--------|----------------|---------------------|
| **EAGS** (Koh et al. 2024) | 2411.06438 | Must read | Entropy-adaptive Gibbs for PLM-based models | PLM-based, not MDM correctors within L&Z framework; closest existing work to Gap B/C |
| **Ψ-Samplers / Diffusion Duality II** (Deschenaux et al. ICLR 2026) | 2602.21185 | Skim | Duality between predictor and corrector; factorization tightness | General PC framework; assess relevance |

---

## Gap Summary

The literature provides strong coverage of corrector kernel design (Zhao et al., DFM),
quality signals (PRISM), token-selection policies (KLASS, UPO, DEMASK), predictor
scheduling (L&Z, EB-Sampler, Denoising Entropy), and remasking (ReMDM, RemeDi). ProSeCo
empirically explores correction scheduling knobs but does not derive a principled
signal-adaptive scheduler.

**The open gap:** Trajectory-level fixed-budget corrector allocation guided by aggregate
signals, with both theoretical justification and controlled experiments on public models.

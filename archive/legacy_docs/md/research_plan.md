---
title: "Research Plan: Informed Correctors for Masked Diffusion Models"
author: "Matteo Omizzolo — MSc Thesis, Bocconi University (supervisor: Prof. Giacomo Zanella)"
date: "Updated April 2026"
geometry: "top=2.5cm, bottom=2.5cm, left=3cm, right=3cm"
fontsize: 11pt
linkcolor: blue
numbersections: true
toc-depth: 3
---

> **Version note (2026-04-15):** Major update after fresh literature scan (April 2026).
> Gap D is now significantly crowded (KLASS, UPO, DEMASK, ProSeCo, Learning Unmasking
> Policies — all published 2024–2026 with code). The primary recommendation shifts to
> **Gap B/C + Gap E** as the most defensible theory-plus-experiments direction.
> Gap D remains viable only if the focus is purely formal (optimality proof), but
> the experimental novelty is now low. See Section 2.

> **Previous version note (2026-04-11):** L&Z read (2026-04-11). Zhao et al. viva
> completed (2026-04-10). All five core papers read. Writing of ch2 begun (2026-04-15).

\newpage

# Thesis Scope

**Core theme:** Informed correctors for masked diffusion language models —
principled, signal-guided correction strategies applied at inference time that improve
sample quality beyond the unmasking-only optimum, without additional training.

**Core question:** Which tokens should be corrected, by what signal, when during the
denoising trajectory, and under what NFE budget — and can any of these choices be
formally justified?

**What "informed corrector" means here:**
A corrector is an additional Markov step applied after token commitment (distinct from
remasking, which defers tokens within the same T-step budget). An *informed* corrector uses
a signal — confidence score, entropy, quality head output — to select which positions to
update. The distinction from uniform random-scan Gibbs is the information used to select
coordinates.

**Pretrained model available:** MDLM-OWT checkpoint (`kuleshov-group/mdlm-owt`, HF,
~130M params, SafeTensors, Apache 2.0) + ReMDM codebase (`external/remdm/`, already
patched for HPC). No new training required.

---

# Candidate Research Directions

## RECOMMENDED: Gap B/C + Gap E (Theory-Plus-Experiments)

**This is the primary recommendation for the next Zanella meeting.**

See the rationale in Section 2 (landscape summary) and the proof sketch in Section 3.

### Gap B/C — Entropy-Adaptive Corrector Scheduling

**Question:** Under a fixed total NFE budget, how should correction steps be allocated
across the denoising trajectory?

**Why it is open:** ReMDM uses fixed corrector step counts. Denoising Entropy
(arXiv:2512.21336) applies entropy to the *predictor* path only; its authors explicitly do
not extend to corrector scheduling. EAGS (arXiv:2411.06438) uses entropy-adaptive Gibbs
for PLM-based models, not for MDM correctors within the L&Z framework. No paper applies
adaptive scheduling to the corrector steps of an MDM with formal guarantees.

**Formal target:** Under a fixed NFE budget, entropy-proportional corrector allocation
achieves a strictly smaller E_fact than uniform corrector allocation in expectation.

**Experimental design:**
- Compare uniform vs entropy-weighted corrector allocation on MDLM-OWT
- Entropy signal: per-step denoising entropy H_DE from Denoising Entropy codebase
  (`LINs-lab/DenoisingEntropy`, confirmed public)
- Infrastructure: instrument `external/remdm/` to vary corrector steps per noise level
- Metrics: MAUVE on OWT-1000, perplexity; sweep T ∈ {128, 256, 512}
- Estimated compute: ~6 A100h for a controlled sweep

**Connection to Gap E:** This experiment requires formalizing what the entropy-weighted
schedule is optimizing. That formalization is Gap E.

---

### Gap E — E_fact Extension to Corrector Steps

**Question:** Do corrector steps reduce E_fact, and by how much as a function of the
selection criterion?

**Why it is open:** L&Z (arXiv:2510.25544) Section 6 explicitly lists "extending to
corrector/remasking steps" as future work (Zanella co-authored this). No paper has
extended the KL decomposition to include corrector steps formally.

**Formal target:** One corrector step at noise level t reduces E_fact by at least
Δ(selection criterion, t), where Δ depends on the entropy of the current distribution
and the quality of the selection signal.

**Position in thesis:** Gap E is the supporting lemma; Gap B/C is the main theorem.
The combination is: "We extend the L&Z decomposition to corrector steps (Gap E), then
show that entropy-adaptive scheduling minimizes total trajectory E_fact under a fixed
NFE budget (Gap B/C)."

---

## Gap D — Optimal Token Selection Under Budget (DEPRIORITIZED)

**Reason for deprioritizing:** The April 2026 scan finds that Gap D is now heavily
covered empirically:
- KLASS (2511.05664, NeurIPS 2025 Spotlight): KL-based selection outperforms confidence
- UPO (2510.05725): RL/MDP formulation of token selection with formal KL-regularized
  objective
- DEMASK (2604.02560, April 2026): dependency-graph-based selection
- Learning Unmasking Policies (2512.09106): RL-trained policy (Apple Research)
- ProSeCo (2602.11590): self-correcting MDM via training

A formal optimality theorem for confidence-based selection would be weaker than what
KLASS and UPO already establish empirically. The experimental contribution has been
largely pre-empted.

**Kept as extension:** If Gap B/C + E are completed with time to spare, a formal
comparison of selection criteria under the extended L&Z framework (Gap D-lite) can be
added as a supplementary result.

---

# Proof Strategy for Gap B/C + Gap E

## Gap E proof sketch

Let Z_t be the state at noise level t after the predictor step. Let K be a corrector
kernel with selection criterion π(i | Z_t) (the policy).

**Claim:** KL(p_{data} ‖ p_{alg after corrector}) ≤ KL(p_{data} ‖ p_{alg before corrector})
minus Δ(π, t), where Δ > 0 depends on the quality of π.

**Strategy:**
1. Extend L&Z Lemma 3.x (factorization-error bound per step) to include a corrector step
2. Show the corrector step contributes a term analogous to the predictor's KL reduction
3. Characterize Δ in terms of the per-position entropy H(x_i | x_{-i}, Z_t)

**Key difficulty:** The L&Z bound is a variational bound on the KL; extending it to
include a Markov corrector kernel requires expressing the corrector's stationary
distribution in terms of the current factorized approximation. The Ascolani et al. 2024
entropy-contraction framework (continuous Gibbs) provides the structural template.

## Gap B/C proof sketch

**Claim:** Under a fixed total corrector NFE budget B = ∑_t k_t (number of corrector
steps at each noise level), the allocation {k_t} = {H_DE(t) / ∑_s H_DE(s) · B} 
minimizes ∑_t E_fact reduction lost due to sub-optimal allocation.

**Strategy:**
1. Use Gap E to express each corrector step's E_fact contribution as Δ(π, t, k_t)
2. Show Δ is monotone in k_t and proportional to entropy at t
3. Lagrangian optimization over {k_t} under ∑_t k_t = B → entropy-proportional allocation

---

# Reading Status (as of 2026-04-15)

## Completed — do not ask to reread

| Paper | arXiv | Notes |
|-------|-------|-------|
| MDLM (Sahoo et al. 2024) | 2406.07524 | SUBS parameterization, factorization approximation |
| ReMDM (Gat et al. 2024) | 2503.00307 | Remasking schedules, conf/loop/cap/rescale variants |
| Zhao et al. / Informed Correctors (2024) | 2407.21243 | Barker/MPF correctors, hollow transformer; Gap D baseline; viva 2026-04-10 |
| PRISM (Kim et al. 2025) | 2510.01384 | Zanella-recommended; provably learnable quality scores |
| L&Z Error Bounds (Lavenant & Zanella 2025) | 2510.25544 | READ 2026-04-11; E_fact decomposition; EB-Sampler; Gap E future work |

## Reading Queue (ordered, as of 2026-04-15)

| # | Paper | arXiv | Why | Location |
|---|-------|-------|-----|----------|
| 1 | **Ascolani, Lavenant & Zanella 2024** (entropy contraction Gibbs) | 2410.00858 | Primary theory: per-step KL contraction template for Gap E proof | `study/papers/05-mcmc-theory/` |
| 2 | **Denoising Entropy** (Chen et al.) | 2512.21336 | Code + entropy signal for experiments; Gap B/C mechanism | download; `study/papers/03-remasking/` |
| 3 | **KLASS** (Kim et al. NeurIPS 2025) | 2511.05664 | Code available; shows KL-based selection beats confidence; comparison baseline | download |
| 4 | **Ψ-Samplers / Diffusion Duality II** (Deschenaux et al. ICLR 2026) | 2602.21185 | Characterizes factorization tightness; code (`s-sahoo/duo`) | `study/papers/04-informed-correctors/` |
| 5 | **DFM** (Gat et al. NeurIPS 2024) | 2407.15595 | Corrector-as-Gibbs vocabulary; mixing rate = spectral gap | `study/papers/04-informed-correctors/` |
| 6 | **CoDD** (Li et al.) | 2603.00045 | Breaks factorization directly; companion piece for Gap E positioning | download |

## New papers to track (2026-04-15 scan)

### Must read before Zanella meeting

| Paper | arXiv | Reason |
|-------|-------|--------|
| Train for the Worst (Kim et al. ICML 2025) | 2502.06768 | Formal separation adaptive vs random; Gap D theory context |
| EAGS (Koh et al.) | 2411.06438 | Closest existing work to Gap B/C; need to know exact differences |
| UPO (Hong et al.) | 2510.05725 | RL/MDP token selection; code available; Gap D crowding agent |
| MDPO/RCR (He et al.) | 2508.13148 | Running Confidence Remasking; code; Gap D crowding |

### Skim / assess

| Paper | arXiv | Reason |
|-------|-------|--------|
| Debiasing Guidance with SMC (2025) | 2502.06079 | ICLR 2025; SMC for discrete diffusion |
| Optimal Inference Schedules (Chen et al.) | 2511.04647 | Information-theoretic schedule bounds; Gap E adjacent |
| Breaking Factorization Barrier (CoDD) | 2603.00045 | Gap E positioning |
| ProSeCo (Schiff et al.) | 2602.11590 | Training-based; positioning only |
| Particle Gibbs for DLMs (Stanford) | 2507.08390 | Trajectory Gibbs; trajectory-level comparison |

---

# Infrastructure Status

## Codebases confirmed usable

| Codebase | Use | Status |
|----------|-----|--------|
| `kuleshov-group/remdm` + `mdlm-owt` | Core experimental platform; Gap B/C scheduling experiments | Already on HPC, patched |
| `LINs-lab/DenoisingEntropy` | Per-step entropy H_DE computation | Confirmed public GitHub, Dec 2025 |
| `shkim0116/KLASS` | KL-based selection baseline for Gap D comparison | NeurIPS 2025 Spotlight, confirmed |
| `s-sahoo/duo` | Ψ-sampler PC framework; most general corrector abstraction | ICLR 2026, confirmed |
| `hasanmohsin/discrete_fkc` | Feynman-Kac correctors; SMC-based | Bengio lab, Jan 2026, confirmed |
| `autonomousvision/mdpo` | Running Confidence Remasking baseline | Aug 2025, confirmed |

## Zhao et al. code status

`lindermanlab/informed-correctors` exists but is **not usable** — 2 commits, no README,
WIP since 2024. The best existing implementations of confidence-based correctors are
`kuleshov-group/remdm` (confidence remasking as ReMDM-conf) and `autonomousvision/mdpo`
(Running Confidence Remasking).

## Pretrained checkpoints confirmed downloadable

| Checkpoint | Source | Params | Notes |
|------------|--------|--------|-------|
| `kuleshov-group/mdlm-owt` | HuggingFace | ~130M | Primary experiment model; already on HPC |
| `kuleshov-group/bd3lm-owt-*` | HuggingFace | ~130M | Block diffusion variant; useful for block-corrector extension |
| `louaaron/sedd-small` / `sedd-medium` | HuggingFace | 100M / 400M | Different noise process; CTMC corrector baseline |

Note: LLaDA-8B and Dream-7B exist and are downloadable but are out of thesis scope (8B
training required, HPC budget). For reading context only.

---

# Writing Track (as of 2026-04-15)

| Chapter | Status | Dependencies |
|---------|--------|--------------|
| ch1 Introduction | TODO — write after direction decision | Requires committed gap |
| ch2 Background: Continuous Diffusion | **FIRST DRAFT DONE** (2026-04-15) | None |
| ch3 Discrete Diffusion | TODO — write next | Needs ch2; all material available |
| ch4 Masked Diffusion (MDLM, ReMDM, L&Z) | TODO | Needs ch3; all material available |
| ch5 Informed Correctors | TODO — write after Ascolani et al. read | Needs ch4 + Ascolani |
| ch6 Contribution | Placeholder — write after direction confirmed | Needs gap decision |
| ch7 Experiments | Placeholder | Needs ch6 |
| ch8 Conclusion | Placeholder | Needs ch6 |

---

# Zanella Meeting Agenda (next meeting)

**Proposed direction to present:** Gap B/C + Gap E

**To show:**
1. Fresh scan shows Gap D is empirically crowded (KLASS, UPO, DEMASK, ProSeCo)
2. Gap B/C (entropy-adaptive corrector scheduling) is genuinely open
3. Gap E (E_fact extension to corrector steps) is explicitly listed as future work
   in L&Z — Zanella co-authored that future-work statement
4. The combination is coherent: Gap E provides the formal foundation, Gap B/C the
   main theorem, and the experiment is a controlled scheduling comparison on MDLM-OWT

**Preliminary theorem statement (draft):**
"Under a fixed NFE budget, entropy-proportional corrector allocation achieves E_fact
strictly less than uniform corrector allocation for any masked diffusion model whose
per-position conditional entropy is non-constant across the trajectory."

**Questions to ask Zanella:**
1. Is the Gap E extension feasible within the L&Z variational framework, or does it
   require a different proof technique?
2. Is EAGS (2411.06438) close enough to Gap B/C to position around, or is the MDM
   corrector setting sufficiently different?
3. Should the thesis commit to Gap B/C + E, or is Gap D still worth a formal
   treatment given the new experimental baselines?

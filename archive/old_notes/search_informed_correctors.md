# Agent Prompt: Literature Search — Informed Correctors for Discrete/Masked Diffusion

## Task

Search arXiv and the web for papers on **informed correctors for discrete and masked diffusion models**. The goal is to find all relevant work that uses confidence or quality signals to guide correction steps during generation — covering both the algorithmic and theoretical sides.

## Background

In the Lavenant–Zanella (2025) framework (arXiv:2510.25544), generation error decomposes as KL(π ‖ p_alg) ≤ E_learn + E_fact. The EB-Sampler minimises E_fact among fixed-schedule predictors. The open question is: can **corrector steps** (additional Gibbs-style resampling after commitment) reduce E_fact further, and can this be proven via spectral gap theory?

An **informed corrector** is one that uses a signal (model entropy, learned quality head, margin score, surprise) to select *which* committed tokens to resample, rather than selecting uniformly. The key distinction from remasking:
- **Corrector**: adds extra MCMC steps after token commitment; increases NFE
- **Remasking** (e.g. ReMDM): defers uncertain tokens within the same T-step budget; modifies the predictor

## Papers already in the collection (do NOT re-find these)

- Informed Correctors for Discrete Diffusion Models — Zhao et al. 2024 (arXiv:2407.21243)
- PRISM — Kim et al. 2025 (learned quality head corrector)
- DFM / Discrete Flow Matching — Gat et al. 2024 NeurIPS (hollow transformer)
- ProSeCo — Schiff et al. 2026
- L&Z Error Bounds — Lavenant & Zanella 2025 (arXiv:2510.25544)
- ReMDM — Gat et al. 2024
- MDLM — Sahoo et al. 2024
- SEDD — Lou et al. 2024
- EB-Sampler — Ben-Hamu et al. 2025

## What to search for

### 1. Direct corrector papers for discrete/masked diffusion
Search queries to try:
- "corrector discrete diffusion"
- "Gibbs corrector masked diffusion"
- "predictor corrector discrete language model"
- "MCMC correction masked diffusion"
- "iterative refinement masked language model inference"

### 2. Quality-signal guided correction
- "confidence guided resampling language model"
- "entropy guided correction diffusion"
- "quality head discrete diffusion"
- "uncertainty aware generation masked language model"

### 3. Spectral gap / mixing time for discrete diffusion correctors
- "spectral gap Gibbs sampling discrete diffusion"
- "mixing time corrector diffusion model"
- "Poincaré inequality discrete diffusion"
- "informed Gibbs sampling spectral gap"

### 4. Related MCMC theory that could support the proof
- "non-uniform scan Gibbs sampling spectral gap"
- "adaptive random scan Gibbs mixing time"
- "informed MCMC spectral gap improvement"
- "weighted Gibbs sampler convergence"

## Output format

For each paper found, provide:
1. **Title, authors, year, arXiv ID or venue**
2. **One-sentence description** of what the paper does
3. **Relevance to informed correctors**: how does it connect — is it a direct corrector method, a quality signal method, relevant MCMC theory, or adjacent?
4. **Priority** (High / Medium / Low) for reading

Organise results into:
- **Tier 1**: Direct informed corrector papers (should read immediately)
- **Tier 2**: Relevant spectral gap / MCMC theory (needed for proof)
- **Tier 3**: Adjacent methods (useful context)

## Notes

- Focus on papers from 2023–2026; the field moves fast
- Include both NLP/LM-focused papers and theoretical MCMC papers if they're relevant to the spectral gap proof strategy
- arXiv categories to prioritise: cs.LG, stat.ML, cs.CL, math.ST

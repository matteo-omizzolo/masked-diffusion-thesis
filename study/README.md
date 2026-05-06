# Study notes and papers

Literature only. Do not use this folder to infer current thesis status.
Current status starts at `../START_HERE.md`.

| Subfolder | Contents |
|---|---|
| `papers/` | Papers organized by topic (01-foundational-diffusion through 08-schedules) |
| `notes/` | Reading notes and meeting prep |
| `explanations/` | Technical explanations |
| `guidlines/` | Thesis guidelines (Bocconi) |

## Literature taxonomy for thesis rewrite

This taxonomy supports PASS 2 chapter rewriting. It is not a status document.

Terminology discipline:
- **Informed corrector / correction kernel:** corrects or refines within a fixed denoising trajectory or revisable/action set using model-derived, learned, reward, verifier, or state information.
- **Remasking / iterative refinement sampler:** changes the masking/remasking sampler or revisits tokens through a remasking process. Do not call it an informed corrector unless it explicitly supplies a fixed-trajectory correction kernel.
- **Trajectory search / test-time scaling:** searches or branches over denoising trajectories, often using rewards/verifiers/true feedback.
- **Base masked/discrete diffusion LM:** model family or training objective.
- **Theory/background:** Markov chains, Gibbs/MH, diffusion theory, error decompositions, evaluation.

| Paper | Key | Category | Thesis role | Relevance | Code/weights |
|---|---|---|---|---|---|
| Simple and Effective Masked Diffusion Language Models | `sahoo2024mdlm` | Base model | Central background; ProSeCo-OWT ancestry | Base-model background | Code/OWT weights available |
| Discrete Diffusion Modeling by Estimating Ratios of the Data Distribution | `lou2024sedd` | Base model | Discrete diffusion background | Supporting | Code/weights available |
| Large Language Diffusion Models | `nie2025llada` | Base model | Scaled DLM context; possible secondary backend only after Phase 0 | Base-model background | MIT code; 8B weights |
| Dream 7B: Diffusion Large Language Models | `ye2025dream` | Base model | Recent DLM context; not an informed corrector | Base-model background | Apache-2.0 code; 7B weights |
| Informed Correctors for Discrete Diffusion Models | `zhao2024informedcorrectors` | Informed corrector | Central conceptual ancestor for correction kernels | Informed-corrector central | No drop-in ProSeCo-OWT backend |
| Learn from Your Mistakes: Self-Correcting Masked Diffusion Models | `schiff2026proseco` | Informed corrector / self-correction kernel | Primary empirical backend for Phase 0 | Informed-corrector central | Code/project; OWT and LLaDA-SFT weights |
| Discrete Flow Matching | `gat2024dfm` | Informed corrector / base objective | Corrector and discrete-flow context | Supporting | Code available |
| PLM-Based Discrete Diffusion LMs with Entropy-Adaptive Gibbs Sampling | `koh2024eags` | Informed corrector / Gibbs-style sampler | Entropy-adaptive correction context | Supporting | Unknown |
| The Diffusion Duality, Chapter II: Psi-Samplers and Efficient Curriculum | `deschenaux2026psisamplers` | Informed corrector / sampler theory | Optional appendix/background for corrector kernels | Supporting | Code available |
| Discrete Feynman-Kac Correctors | `hasan2026fkc` | Informed corrector / SMC | Future related corrector family | Future/related | Code available |
| Fine-Tuning Masked Diffusion for Provable Self-Correction (PRISM) | `kim2025prism` | Learned quality signal / remasking self-correction | Quality-score motivation; not a thesis pillar | Candidate citation | Code available; no current backend switch |
| Remasking Discrete Diffusion Models with Inference-Time Scaling | `wang2025remdm` | Remasking sampler | Adjacent inference-time scaling; classify separately from corrector timing | Remasking-only | Apache-2.0 code; MDLM weights via upstream |
| Optimizing Decoding Paths in Masked Diffusion Models by Quantifying Uncertainty | `chen2025denoisingentropy` | Remasking / path heuristic | Uncertainty signal context | Supporting | Code available |
| Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking | `benhamu2025ebsampler` | Remasking / token policy | Entropy-bound scheduling context | Supporting | Unknown |
| Train for the Worst, Plan for the Best | `kim2025tokenordering` | Predictor/token-order policy | Explains token ordering gap; not corrector timing | Supporting | Unknown |
| Effective Test-Time Scaling of Discrete Diffusion through Iterative Refinement | `lee2025iterref` | Trajectory search / reward-guided refinement | Compare against search-style scaling, not fixed placement scheduling | Search/test-time scaling | Unknown |
| S^3: Stratified Scaling Search for Test-Time in Diffusion Language Models | `bilal2026s3` | Trajectory search / verifier-guided scaling | Recent search baseline for DLMs | Search/test-time scaling | Unknown |
| Inference-Time Scaling of Diffusion Language Models with Particle Gibbs Sampling | `dang2025particlegibbs` | Trajectory search / MCMC | Search/SMC comparison for true-feedback-style methods | Supporting | Unknown |
| Error Bounds and Optimal Schedules for Masked Diffusions with Factorized Approximations | `lavenant2025lz` | Theory/background | Error decomposition and schedule theory; motivates assumptions | Central theory | N/A |
| Informed Proposals for Local MCMC in Discrete Spaces | `zanella2020informedproposals` | Theory/background | Proposal-design vocabulary | Supporting | N/A |
| Entropy Contraction of the Gibbs Sampler under Log-Concavity | `ascolani2024entropycontraction` | Theory/background | Gibbs convergence context | Supporting | N/A |
| Spectral Gap of Metropolis-within-Gibbs under Log-Concavity | `secchi2025mwggap` | Theory/background | MwG background | Supporting | N/A |
| DDPM / score-based diffusion / DDIM / CFG foundations | `ho2020ddpm`, `song2021score`, `song2019ncsn`, `song2020ddim`, `ho2022cfg` | Theory/background | Chapter 2 diffusion foundations | Supporting | N/A |
| MAUVE | `pillutla2021mauve` | Evaluation | Text-quality metric context | Supporting | Code available |

## Backend survey snapshot

Recommendation for Phase 0: keep **ProSeCo-OWT**. It is the only audited backend here that is both an informed self-correction kernel and already wired to the thesis PF1-PF8 trace contract. Do not switch before Phase 0 passes.

| Candidate | Is it an informed corrector? | Weights/license | R_t / fixed-B trace fit | Risk | Recommendation |
|---|---:|---|---|---|---|
| ProSeCo-OWT | Yes | HF weights; Apache-style upstream dependencies | Current target; PF hooks exist, real PF3/PF5/PF7 need checkpoint | Low | Keep for Phase 0 |
| ProSeCo-LLaDA-SFT | Yes | HF 8B weights | Likely traceable, but heavier and not current pipeline | Medium/high compute | Post-Phase-0 only |
| Zhao et al. informed correctors | Yes conceptually | No obvious pretrained LM backend | Would require implementation/backend work | High | Theory citation, not backend |
| PRISM | Quality-score/remasking self-correction, not fixed timing by default | Code available; pretrained module status unclear | Could supply a signal, but not a clean Phase 0 backend | Medium/high | Literature/candidate signal only |
| ReMDM | No: remasking sampler | Apache-2.0 code; MDLM weights via upstream | Changes sampler, not fixed corrector placement | Medium | Comparison baseline later |
| IterRef | No: reward-guided trajectory refinement | Unknown | Search-style, not fixed-B corrector timing | High | Related work only |
| S^3 | No: verifier-guided trajectory search | Unknown | Search-style, not fixed-B corrector timing | High | Related work only |
| LLaDA / Dream | Base models | Open weights; large GPU needs | Need a separate corrector wrapper | High | Background / later backend exploration |

## Chapter citation map for PASS 2

| Chapter | Citation role |
|---|---|
| ch1 Introduction | Motivate DLMs and the gap: MDLM, SEDD, LLaDA, Dream, ProSeCo, Zhao informed correctors, ReMDM/PRISM/S^3/IterRef as adjacent but not the thesis question. |
| ch2 Background diffusion | DDPM, score/SDE diffusion, DDIM, CFG, score matching/denoising autoencoders, Markov-chain basics. |
| ch3 Discrete/masked diffusion | SEDD, MDLM, LLaDA, Dream, token ordering, likelihood/error decomposition. |
| ch4 Model families and inference-time mechanics | ReMDM, entropy/uncertainty remasking, EB sampler, PRISM, IterRef, S^3, particle Gibbs; keep categories explicit. |
| ch5 Informed correctors and timing | Zhao informed correctors, ProSeCo, DFM, EAGS, Psi samplers, Feynman-Kac correctors, L&Z; culminate in the fixed-placement timing gap. |
| ch6 Theory | No literature survey dump; cite only enough to contextualize definitions, assumptions, and the model/corrector-agnostic setup. |

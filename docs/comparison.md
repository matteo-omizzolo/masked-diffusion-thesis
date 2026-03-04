---
title: "Remasking in Masked Diffusion Language Models: A Comparative Survey"
author: "MSc Thesis Literature Review — Bocconi University"
date: "March 2026"
geometry: margin=2.5cm
fontsize: 11pt
linkcolor: blue
---

# Remasking in Masked Diffusion Language Models: A Comparative Survey

## Preface

This document reviews twelve papers directly relevant to the thesis
"Principled Remasking in Masked Diffusion Language Models" (Bocconi
University, supervised by Prof. Giacomo Zanella). Papers 1–7 are the
core works read first; Papers 8–12 provide essential background and
broader context. The unifying question across all papers is: *how can
we design sampling algorithms for masked diffusion models that produce
higher-quality text without retraining the model?* Remasking —
re-masking already committed tokens and re-predicting them in a richer
context — is the central mechanism under study.

---

## Part I — Core Works

---

## 1. MD4 and MDLM — Foundational Masked Diffusion for Text

**Citations:** Shi et al. (2024), "Simplified and Generalized Masked
Diffusion for Discrete Data" (MD4); Sahoo et al. (2024), "Simple and
Effective Masked Diffusion Language Models" (MDLM). arXiv:2406.07524.

### A. Summary

MD4 and MDLM established the modern framework for masked diffusion
language models (MDMs). Both papers adapt the diffusion paradigm —
originally developed for continuous data — to discrete token sequences.
The key insight is that masking is the natural analogue of Gaussian
noise for categorical variables: the forward process gradually replaces
tokens with a special [MASK] token, and the reverse process learns to
unmask them.

MD4 provides the general theoretical treatment, deriving the ELBO for
masked diffusion. MDLM simplifies the ELBO into a computationally
tractable form competitive with autoregressive language models,
introducing importance-weighted training and demonstrating strong
performance on OpenWebText at the 130M parameter scale (MDLM-OWT).

### B. Method Details

**Forward process:**

$$q(z_t \mid x) = \prod_i \mathrm{Cat}\!\left(z_t^i;\; \alpha_t \cdot e_{x^i} + (1 - \alpha_t) \cdot e_m\right)$$

where $\alpha_t \in [0,1]$ is the token retention probability, $e_m$ is
the one-hot mask vector, and positions are independent given $x$.

**Noise schedules.** MD4: linear $\alpha_t = 1-t$. MDLM: cosine
$\alpha_t = \cos^2(\pi t / 2)$.

**Reverse kernel:**

$$p_\theta(z_s^i \mid z_t) = \frac{\alpha_s}{\alpha_t} \cdot \delta(z_s^i = z_t^i) + \left(1 - \frac{\alpha_s}{\alpha_t}\right) \cdot p_\theta(x^i \mid z_t)$$

**Training objective (MDLM):**

$$\mathcal{L} = \mathbb{E}_{t, x, z_t}\!\left[ w_t \cdot \sum_{\{i:\, z_t^i = [\text{M}]\}} -\log p_\theta(x^i \mid z_t) \right]$$

### C. Theoretical Contributions

ELBO derivation for masked diffusion; optimality of the reverse kernel
via Bayes' rule; justification of cosine schedule via approximately
uniform per-step SNR. No non-asymptotic sampling error bounds.

### D. Confidence Signal

None used at inference. The per-token logits $p_\theta(x^i \mid z_t)$
are the implicit raw material for all downstream confidence signals.

### E. Limitations

(1) Independence assumption induces $E_{\text{fact}}$.
(2) No remasking — committed errors propagate.
(3) Uniform random unmasking order.
(4) No formal sampling error bound.

### F. Relation to This Thesis

MDLM-OWT is the primary experimental backbone. The independence
assumption motivates the thesis; the logit distribution is the source
of all training-free confidence signals.

---

## 2. ReMDM — Remasking via a Principled Posterior

**Citation:** Wang et al. (2025), "Remasking Discrete Diffusion Models
with Inference-Time Scaling." arXiv:2503.00307.

### A. Summary

ReMDM proposes a principled remasking posterior $\sigma_t$ symmetric to
the unmasking posterior. Remasking committed tokens and re-predicting in
richer context improves quality monotonically in the number of
correction steps $R$, with no retraining required.

### B. Method Details

**Remasking posterior:**

$$\sigma_t(z^{\prime i} = [\text{M}] \mid z) = 1 - \frac{\alpha_t}{\alpha_0} \quad \text{(committed positions)}$$

**Inference algorithm:**
```
z_T = [M,...,M]
for t = T,...,1:
    z_{t-1} ← p_θ(z_{t-1} | z_t)       # standard unmask
    for r = 1,...,R:
        z' ← remask via σ_t(z'|z_{t-1})  # correction
        z_{t-1} ← p_θ(z_{t-1} | z')
return z_0
```

Strategies: `remdm-conf` (confidence-weighted), `remdm-cap`,
`remdm-rescale`, `remdm-loop` (time-windowed).

### C. Theoretical Contributions

KL is monotonically non-increasing in $R$ for fixed $T$ (assuming
exact score). Proves convergence of the corrected process as
$T, R \to \infty$.

### D. Confidence Signal

Uniform $\sigma_t$ (theoretically grounded) or entropy-weighted
$\text{remask\_prob}^i \propto H(p_\theta(x^i \mid z_t))$ (empirically
superior, no formal guarantee).

### E. Limitations

(1) Uniform remasking ignores information content.
(2) No connection to $I(x)$ or $E_{\text{fact}}$.
(3) Cost $O(T \times R)$.
(4) No task-specific evaluation.

### F. Relation to This Thesis

The $\sigma_t$ posterior is a special case of the thesis's remasking
kernel. The thesis extends ReMDM by connecting it to $E_{\text{fact}}$
and deriving confidence-guided remasking as optimal.

---

## 3. RemeDi — Learned Dual-Stream Remasking with RL

**Citation:** Huang et al. (2025), "Don't Settle Too Early:
Self-Reflective Remasking for Diffusion Language Models."
arXiv:2509.23653.

### A. Summary

RemeDi introduces a dual-stream architecture separating token
prediction (TPS) from unmasking policy (UPS). UPS is trained via BCE
then refined with GRPO RL to learn globally optimal unmask/remask
decisions. Inference-only code and 9B HuggingFace checkpoints
(RemeDi-RL, RemeDi-Instruct) are publicly available.

### B. Method Details

**Architecture:** UPS produces per-token scores $\psi^i \in [0,1]$
from TPS hidden states $h^i$ via a lightweight MLP.

**Training stages:** (1) TPS pre-train (ELBO), (2) UPS BCE pre-train
on $y^i = \mathbf{1}[\hat{y}^i = x^i]$, (3) GRPO RL fine-tune on
generation quality reward.

**Inference:**
```
for t = T,...,1:
    (p^i, ψ^i) ← TPS+UPS(z_t)
    unmask {i: z_t^i=[M], ψ^i > τ_unmask}
    remask  {i: z_t^i≠[M], ψ^i < τ_remask}
```

### C. Theoretical Contributions

UPS BCE minimiser converges to $P(\hat{y}^i = x^i \mid \text{context})$
(standard proper-scoring-rule argument). No sampling error bound.

### D. Confidence Signal

Learned $\psi^i$ — calibrated across steps and noise levels. Free
overhead (MLP on existing hidden states). Not portable to other models.

### E. Limitations

(1) Requires fine-tuning.
(2) GRPO instability at 9B scale.
(3) No connection to optimal sampling theory.
(4) Instruction-focused evaluation only.

### F. Relation to This Thesis

RemeDi is the upper-bound fine-tuned comparator. The thesis asks: how
close do training-free signals get to the learned UPS? Theoretically,
RemeDi motivates the "consistent estimator" assumption in the thesis's
main theorem.

---

## 4. PRISM — Plug-and-Play Quality Head for MDMs

**Citation:** Kim et al. (2025), "Fine-Tuning Masked Diffusion for
Provable Self-Correction." arXiv:2510.01384.

### A. Summary

PRISM trains a lightweight BCE quality head $g_\phi$ on frozen MDM
hidden states to predict per-token prediction accuracy. The BCE
minimiser converges to the true conditional accuracy
$P(x^i = \hat{y}^i \mid y \oplus m^i)$, giving a theoretically grounded
confidence signal. Evaluated on Sudoku, MDLM-OWT (170M), and LLaDA-8B
(MBPP code generation).

### B. Method Details

$$g_\phi(y)^i = \sigma(\text{MLP}_\phi(h^i)) \in [0,1]$$

$$\mathcal{L}(\phi) = -\mathbb{E}\!\left[\sum_i \mathbf{1}[x^i=\hat{y}^i]\log g_\phi^i + \mathbf{1}[x^i\neq\hat{y}^i]\log(1-g_\phi^i)\right]$$

**Guarantee:** $g_{\phi^*}^i = P(x^i = \hat{y}^i \mid y \oplus m^i)$.

**Inference:** unmask/remask guided by $g_\phi^i$ scores.

### C. Theoretical Contributions

Bayes-optimality proof for the BCE minimiser (proper scoring rule
argument). First formal guarantee on quality-signal calibration for MDMs.

### D. Confidence Signal

$g_\phi^i \in [0,1]$ — Bayes-optimal calibrated quality estimate.
One small MLP pass per step (negligible overhead). Requires training.

### E. Limitations

(1) Training required for each base model.
(2) LLaDA-8B adapter not publicly released (as of early 2026).
(3) No sampling quality guarantee (only calibration guaranteed).
(4) OOD calibration untested.

### F. Relation to This Thesis

PRISM's convergence result formally justifies the thesis's "consistent
estimator" assumption. The thesis is the training-free analogue:
under what conditions do max-prob/entropy/margin approximate
$P(x^i = \hat{y}^i \mid \text{context})$ sufficiently well?

---

## 5. Informed Correctors — MCMC-Based Token Correction

**Citation:** Zhao et al. (2024/2025), "Informed Correctors for
Discrete Diffusion Models."

### A. Summary

Gibbs-like MCMC correction steps for discrete diffusion, using the
per-token marginal log-likelihood as the confidence signal. Proves
exponential KL reduction in the number of correction steps, and that
confidence-guided corrections converge faster than uniform ones.
Requires a hollow transformer for efficient computation.

### B. Method Details

**Gibbs corrector:** resample $x^d \sim q_t(x^d \mid x^{\setminus d})$.

**Informed variant:** select $d$ proportional to
$\exp(-\beta \log q_t(x^d \mid x^{\setminus d}))$ (high surprise first).

**Hollow transformer:** diagonal-masked attention computes all
per-position conditionals in one forward pass ($O(1)$ cost vs. $O(L)$).

### C. Theoretical Contributions

Mixing time bound:
$$\mathrm{KL}(q_0 \| p_R) \leq e^{-R\lambda} \cdot \mathrm{KL}(q_0 \| p_0)$$

Proof that $\lambda_{\text{informed}} \geq \lambda_{\text{uniform}}$,
i.e. confidence-guided correction is strictly more efficient.

### D. Confidence Signal

$\log q_t(x^d \mid x^{\setminus d})$ — exact per-position conditional.
$O(L)$ forward passes without hollow transformer; $O(1)$ with it.

### E. Limitations

(1) Requires hollow transformer (not compatible with standard MDMs).
(2) Post-hoc only — does not integrate into denoising loop.
(3) Spectral gap $\lambda$ can be small for long-range dependencies.
(4) No connection to information profile $I(x)$ or $E_{\text{fact}}$.

### F. Relation to This Thesis

Provides MCMC-theoretic corroboration for the thesis's main claim.
The thesis avoids the hollow transformer requirement by using only
forward-pass logits — a key design constraint that distinguishes it
from Informed Correctors.

---

## 6. EB-Sampler — Entropy-Bounded Adaptive Unmasking

**Citation:** Ben-Hamu et al. (2025), "EB-Sampler: Entropy-Bounded
Sampling for Masked Diffusion Models."

### A. Summary

Adapts the number of tokens unmasked per step to a fixed entropy budget
$\varepsilon$, consuming information at a constant rate. Achieves 2–3×
speedup or better quality at fixed compute. Most direct connection to
the Lavenant & Zanella $E_{\text{fact}}$ framework among existing works.

### B. Method Details

Unmask positions greedily in ascending order of predicted entropy
$\hat{I}^i = H(p_\theta(x^i \mid z_t))$ until budget $\varepsilon$ is
consumed.

$$E_{\text{fact}}(\text{EB}) \leq \varepsilon \cdot \max_i \hat{I}^i$$

### C. Theoretical Contributions

Proves EB-Sampler minimises $E_{\text{fact}}$ among all deterministic
unmasking strategies at equal forward-pass budget. Optimality only for
unmasking — remasking not considered.

### D. Confidence Signal

$\hat{I}^i = H(p_\theta(x^i \mid z_t))$ — free from forward-pass logits.

### E. Limitations

(1) Unmasking only; no remasking.
(2) $\hat{I}^i$ is a noisy proxy for true $I^i(x)$.
(3) Variable step count complicates benchmarking.

### F. Relation to This Thesis

The unmasking analogue of the thesis's remasking contribution:
EB-Sampler : optimal unmasking ↔ Thesis : optimal remasking.
The thesis derives the remasking analogue of the $E_{\text{fact}}$
bound and shows confidence-guided remasking reduces it.

---

## 7. Error Bounds — Lavenant & Zanella (2025)

**Citation:** Lavenant & Zanella (2025), "Sampling Error Analysis for
Masked Diffusion Models."

### A. Summary

First rigorous non-asymptotic KL bound for MDM sampling, decomposed
into $E_{\text{learn}}$ (model error) and $E_{\text{fact}}$ (algorithm
error). $E_{\text{fact}}$ is a Riemann approximation error of the
information profile $I(x) = \sum_i H(x^i \mid x^{\setminus i})$.
Derives the optimal unmasking schedule. The remasking direction is
explicitly identified as an open problem — the central gap this thesis
fills.

### B. Method Details

$$\mathrm{KL}(\pi \| p_{\text{alg}}) \leq E_{\text{learn}} + E_{\text{fact}}$$

$$E_{\text{fact}} \leq C \cdot \left(\frac{I(x)}{T}\right)^2$$

Optimal schedule: unmask in increasing order of $I^i(x)$ (= EB-Sampler).

### C. Theoretical Contributions

(1) First non-asymptotic KL bound for MDM sampling.
(2) $E_{\text{fact}}$ as Riemann error of $I(x)$.
(3) Optimal unmasking schedule derivation.
(4) $O(k^2)$ error for simultaneous unmasking of $k$ tokens.

### D. Confidence Signal

None (analysis paper). Implies $I^i(x)$ is the optimal signal;
$H(p_\theta(x^i \mid z_t))$ is the practical approximation.

### E. Limitations

(1) **Remasking not covered** — explicit open problem.
(2) $E_{\text{learn}}$ treated as a black box.
(3) Factored posterior assumption.
(4) Constant factors may be large for practical $T$.

### F. Relation to This Thesis

*This is the thesis's theoretical foundation.* Chapter 3 extends
the framework to include remasking transitions and derives conditions
under which confidence-guided remasking reduces $E_{\text{fact}}$.

---

## Part II — Essential Background and Context

---

## 8. D3PM — Structured Denoising Diffusion in Discrete Spaces

**Citation:** Austin, Johnson, Ho, Tarlow & van den Berg (2021),
"Structured Denoising Diffusion Models in Discrete State-Spaces."
NeurIPS 2021. arXiv:2107.03006.

### A. Summary

D3PM is the paper that brought diffusion models to discrete state
spaces with full generality. It defines a family of discrete forward
Markov chains parametrised by a transition matrix $Q_t$, and derives
corresponding variational lower bounds for training the reverse process.
The three key forward processes it introduces are: (1) **absorbing
diffusion** (the masking process used by all MDMs), where all tokens
eventually collapse to a single absorbing [MASK] state; (2) **uniform
diffusion**, where tokens uniformly transition to all other tokens; and
(3) **embedding-based diffusion**, where transitions follow distances
in a learned embedding space.

The absorbing (mask-and-predict) process is the one that MDLM, SEDD,
LLaDA, and ReMDM all use. Understanding D3PM is prerequisite for
understanding any subsequent MDM paper.

### B. Method Details

**General forward process.** For a single token $x^i \in \{1, \ldots, V\}$:

$$q(z_t^i \mid z_{t-1}^i) = \mathrm{Cat}(z_t^i;\; z_{t-1}^i \cdot Q_t^i)$$

where $Q_t \in \mathbb{R}^{V \times V}$ is the transition matrix at
step $t$.

**Absorbing process** (the MDM special case). Let $V+1$ be the mask
token [M]. Then:

$$Q_t = \begin{pmatrix} 1 - \beta_t & \beta_t \\ 0 & 1 \end{pmatrix}$$
$$\bar{Q}_t = Q_1 Q_2 \cdots Q_t$$

The marginal is:

$$q(z_t^i \mid x^i) = \alpha_t \cdot e_{x^i} + (1 - \alpha_t) \cdot e_m$$

exactly the MDLM forward process with $\alpha_t = \prod_{s=1}^t (1-\beta_s)$.

**Posterior:**

$$q(z_{t-1}^i \mid z_t^i, x^i) \propto q(z_t^i \mid z_{t-1}^i) \cdot q(z_{t-1}^i \mid x^i)$$

For the absorbing process this has a closed form: if $z_t^i = x^i$,
then $z_{t-1}^i = x^i$ with probability 1; if $z_t^i = [\text{M}]$,
then $z_{t-1}^i = x^i$ with probability $\alpha_{t-1} / (1 - \alpha_t)$
and $z_{t-1}^i = [\text{M}]$ otherwise.

**Training objective.** D3PM trains the reverse model $p_\theta(z_{t-1}
\mid z_t)$ by minimising the VLB:

$$\mathcal{L} = \sum_t \mathbb{E}\!\left[\mathrm{KL}(q(z_{t-1} \mid z_t, x) \| p_\theta(z_{t-1} \mid z_t))\right]$$

D3PM additionally introduces an auxiliary cross-entropy term
$\lambda \cdot \mathrm{CE}(x, p_\theta^{(x0)}(z_t))$ that directly
supervises the clean-token prediction $p_\theta^{(x0)}$; MDLM later
shows that this is essentially the entire training signal.

### C. Theoretical Contributions

D3PM proves that the VLB for discrete diffusion decomposes into a sum
of per-step KL terms, analogously to the continuous DDPM bound. It
establishes that the absorbing process achieves lower ELBO than uniform
diffusion on text, which justifies the design choice of all subsequent
MDMs. The paper does not provide sampling error bounds.

### D. Confidence Signal

D3PM uses ancestral sampling with no confidence signal. The paper
introduces the concept of $x_0$-parameterisation (predicting the
clean token directly rather than the reverse kernel), which is the
parameterisation used by all subsequent MDMs and is the source of the
per-token logit distribution that confidence signals are derived from.

### E. Limitations

1. **Absorbing process only proven superior empirically,** not derived
   theoretically as the optimal choice of $Q_t$.
2. **No remasking.** The ancestral sampler is one-directional.
3. **Slow sampling.** D3PM uses $T = 1000$ steps; this is reduced to
   $T = 64$–$1024$ in MDLM and further in ReMDM.
4. **Text-specific challenges** (long-range coherence, variable-length
   sequences) are noted but not fully addressed.

### F. Relation to This Thesis

D3PM is the origin of the entire MDM framework. The absorbing
transition matrix $Q_t$ is the source of the factorisation error
$E_{\text{fact}}$: because $Q_t$ masks positions independently, the
true joint posterior at each step has correlations that are lost when
the reverse process is factorised. The thesis's remasking kernel can be
understood as a "corrective" non-absorbing transition that partially
undoes this information loss. Citing D3PM is essential to situate the
thesis in the discrete diffusion literature.

---

## 9. SEDD — Discrete Diffusion via Score Entropy

**Citation:** Lou, Meng & Ermon (2024), "Discrete Diffusion Modeling
by Estimating the Ratios of the Data Distribution." ICML 2024.
arXiv:2310.16834.

### A. Summary

SEDD introduces an alternative training objective for discrete diffusion
models based on *score entropy* — a proper divergence for estimating
ratios of discrete probability distributions, analogous to score
matching for continuous diffusion. Rather than directly parameterising
the reverse Markov kernel, SEDD parameterises the *concrete score*
$s_\theta(z_t)_{ij} \approx p(x^i = j \mid z_t) / p(z_t^i = j)$ and
trains it with a provably consistent objective. SEDD achieves
state-of-the-art unconditional text generation among non-autoregressive
models (at the time of publication) and introduces
predictor-corrector (PC) samplers for discrete diffusion that are
directly relevant to remasking.

### B. Method Details

**Score entropy objective.** The concrete score is trained by
minimising:

$$\mathcal{L}(\theta) = \mathbb{E}_{t,x,z_t}\!\left[\sum_{j \neq z_t} \left(s_\theta(z_t)_{z_t,j} - \log\frac{p_t(z_t \oplus j)}{p_t(z_t)} + h(s_\theta(z_t)_{z_t,j})\right)\right]$$

where $z_t \oplus j$ denotes the sequence $z_t$ with one position
flipped to $j$, and $h$ is a convex function that makes the objective a
proper scoring rule. This is a consistent estimator of the true score
ratios.

**Absorbing SEDD.** For the absorbing (masking) noise process, the
score simplifies: the model only needs to estimate $p(x^i \mid z_t)$
for masked positions, which reduces to the same masked language model
objective as MDLM. The score-entropy and cross-entropy objectives are
equivalent for the absorbing process, which is why MDLM's simpler
training matches SEDD's performance on text with masking noise.

**Predictor-corrector (PC) sampler.** SEDD introduces a PC sampler
that alternates between:
- **Predictor step:** standard reverse-process unmasking.
- **Corrector step:** a Langevin-like MCMC correction using the score.

This is the discrete analogue of the continuous diffusion PC sampler
(Song et al., 2021) and is a principled precursor to the remasking
strategies in ReMDM and this thesis.

**Tweedie denoiser.** SEDD also derives the discrete analogue of
Tweedie's formula: the minimum mean-squared error estimate of $x$ given
$z_t$ can be computed directly from the score, without sampling. This
is used to initialise the PC sampler.

### C. Theoretical Contributions

SEDD proves: (1) the score-entropy objective is a consistent estimator
of the true score ratios (analogous to Fisher consistency in score
matching); (2) the PC sampler converges to the target distribution as
the number of corrector steps increases, with an explicit convergence
rate. These are the strongest theoretical results for any MDM sampler
prior to Lavenant & Zanella (2025).

### D. Confidence Signal

SEDD does not use a per-token confidence signal for unmasking order.
The concrete score $s_\theta(z_t)_{z_t, j}$ implicitly encodes token
uncertainty but is used for the corrector step, not for selective
unmasking. The entropy of $p_\theta(x^i \mid z_t)$ (derived from the
score) is a natural downstream confidence signal.

### E. Limitations

1. **Score entropy objective is harder to implement** than MDLM's
   cross-entropy; for the absorbing process the two are equivalent,
   removing the advantage.
2. **PC sampler requires $O(T_{\text{corr}})$ additional steps** per
   denoising step, similar to ReMDM's correction cost.
3. **No per-token confidence signal** for selective correction.
4. **Uniform noise process** (non-absorbing SEDD) underperforms
   absorbing SEDD on text, suggesting absorbing noise is the right
   inductive bias for language.

### F. Relation to This Thesis

SEDD's PC sampler is the principled precursor to remasking in MDMs.
The corrector step in SEDD corresponds to the remasking step in ReMDM,
but SEDD does not use a confidence signal to select which positions to
correct. The thesis's contribution can be framed as: deriving the
optimal confidence-guided corrector for the absorbing MDM process
within the Lavenant & Zanella error-bound framework, extending SEDD's
PC sampler from a uniform to an information-theoretically optimal one.
The SEDD repo (`louaaron/SEDD`) provides a useful secondary baseline
with its own pre-trained checkpoints on OpenWebText.

---

## 10. LLaDA — Large Language Diffusion with mAsking

**Citation:** Nie et al. (2025), "LLaDA: Large Language Diffusion with
mAsking." arXiv:2502.09992.

### A. Summary

LLaDA demonstrates that a masked diffusion model trained at 8B
parameter scale (LLaDA-8B) can match or exceed LLaMA 3 8B on standard
instruction-following and reasoning benchmarks, including multi-turn
dialogue and mathematical reasoning. This is the first masked diffusion
model to convincingly compete with frontier autoregressive LLMs at the
same parameter count. LLaDA-8B-Instruct (the instruction-tuned version)
is the backbone for PRISM's code generation experiments and is available
on HuggingFace (`GSAI-ML/LLaDA-8B-Instruct`).

### B. Method Details

**Architecture.** LLaDA-8B is a standard transformer decoder with
bidirectional attention (unlike causal LLMs), trained with the
masked language model objective on 2.3 trillion tokens. The model uses
the standard absorbing forward process and the MDLM-style reverse
process (predict clean token from masked context).

**Pre-training.** Identical to MDLM: the loss is the expected negative
log-likelihood of masked tokens, weighted by a per-step importance
weight derived from the noise schedule.

**Instruction tuning.** LLaDA-8B-Instruct is fine-tuned with supervised
instruction following (SFT) on a standard instruction dataset, with the
prompt tokens frozen (unmasked) and only the response tokens subject to
the masking process. This is the first MDM instruction-tuning recipe
demonstrated to work at 8B scale.

**Sampling.** LLaDA uses a standard uniform remasking schedule (each
committed token can be re-masked with a fixed probability per step).
No confidence signal is used. The naive remasking schedule is the
key weakness that PRISM and the thesis target.

### C. Theoretical Contributions

LLaDA does not provide new theoretical results on MDMs. Its
contribution is empirical: scaling masked diffusion to 8B parameters
and demonstrating competitive instruction-following quality. The paper
provides detailed scaling analysis showing that LLaDA obeys the same
scaling laws as autoregressive LLMs, which implies that the quality
gap between MDMs and autoregressive models at this scale is small and
may be addressable by principled sampling improvements.

### D. Confidence Signal

None. LLaDA uses uniform random remasking during sampling. The per-token
logit distribution $p_\theta(x^i \mid z_t)$ is available but not
exploited. This is the direct motivation for applying the thesis's
confidence-guided remasking strategies to LLaDA.

### E. Limitations

1. **Naive remasking.** Uniform remasking ignores token uncertainty,
   leaving quality on the table at inference time.
2. **Bidirectional attention only.** LLaDA-8B cannot be used for
   causal (streaming) generation, limiting deployment contexts.
3. **No formal connection to information profile.** The uniform
   remasking schedule is not derived from any principled criterion.
4. **PRISM adapter not public.** The PRISM quality head trained on
   LLaDA-8B is not publicly released (as of early 2026).

### F. Relation to This Thesis

LLaDA-8B-Instruct is the large-scale experimental backbone for the
thesis. The key experimental question is: does confidence-guided
remasking (thesis strategy) improve LLaDA-8B generation quality beyond
the naive uniform schedule used in the published model? If yes, this
demonstrates that the thesis's training-free strategies scale to
frontier-class MDMs. The uniform remasking in LLaDA is the weakest
baseline the thesis should beat.

---

## 11. Discrete Flow Matching

**Citation:** Gat, Remez, Shaul, Kreuk, Chen, Synnaeve, Adi & Lipman
(2024), "Discrete Flow Matching." NeurIPS 2024. arXiv:2407.15595.

### A. Summary

Discrete Flow Matching (DFM) extends continuous flow matching (Lipman
et al., 2022) to discrete state spaces by defining probability paths
over categorical distributions that interpolate between a prior and the
data distribution. The key theoretical insight is that masked diffusion
is a special case of DFM with a specific probability path (the absorbing
path), and the MDM training objective (masked cross-entropy) is exactly
the flow matching objective under this path. DFM provides a cleaner,
path-integral formulation that subsumes both MDMs and SEDD, and
introduces the *discrete ODE* perspective: generation is a trajectory
in distribution space, and corrector steps correspond to time-reversal
of this trajectory.

### B. Method Details

**Probability path.** A discrete probability path $p_t$ interpolates
between a prior $p_0$ (e.g., fully masked distribution) and the data
distribution $p_1$ via:

$$p_t(z) = \sum_x p(x) \cdot \mathrm{Cat}(z;\; \alpha_t \cdot e_x + (1-\alpha_t) \cdot e_m)$$

For the absorbing path, this is identical to the MDLM marginal.

**Discrete velocity field.** The flow matching objective trains a
velocity field $u_t(z, j)$ that specifies the rate of probability flow
from token $z^i$ to token $j$ at time $t$. For the absorbing path:

$$u_t(z^i \rightarrow j) = \frac{1 - \alpha_t}{\alpha_t} \cdot p_\theta(x^i = j \mid z_t) \cdot \mathbf{1}[z^i = [\text{M}]]$$

**Training objective.** The discrete flow matching objective reduces to:

$$\mathcal{L} = \mathbb{E}_{t, x, z_t}\!\left[-\sum_{i:\, z_t^i=[\text{M}]} \frac{\dot{\alpha}_t}{\alpha_t} \log p_\theta(x^i \mid z_t)\right]$$

which is identical to the MDLM ELBO up to the time-weighting factor.

**Corrector steps in DFM.** DFM provides a natural framework for
corrector steps: a corrector is a flow that maps $p_t$ back to itself
while reducing entropy. In the discrete case, this corresponds to
re-masking and re-predicting high-entropy tokens — exactly the
remasking operation. DFM thus provides a *continuous-time* perspective
on remasking that complements the Riemann-sum perspective of Lavenant &
Zanella.

### C. Theoretical Contributions

(1) Unified framework: MDMs and SEDD are special cases of DFM.
(2) Proves the DFM objective is equivalent to MDM cross-entropy for
the absorbing path.
(3) Derives the optimal probability path for minimising the
continuous-time KL between the generative trajectory and the true
data-generating process.
(4) Shows that corrector steps correspond to score-function-guided
flows, giving a theoretical foundation for principled correction.

### D. Confidence Signal

DFM does not introduce a new confidence signal. The velocity field
$u_t(z^i \rightarrow j)$ implicitly encodes uncertainty: positions with
high $H(p_\theta(x^i \mid z_t))$ have more diffuse velocity distributions
and are natural candidates for correction. This connects to the thesis's
entropy-based confidence signal.

### E. Limitations

1. **Continuous-time formulation** is harder to implement than
   discrete-step MDLM; in practice both reduce to the same sampler.
2. **No remasking theory.** The corrector perspective is described but
   not formally analysed in terms of sampling error bounds.
3. **Text experiments are limited** compared to MDLM-OWT and LLaDA.
4. **Absorbing path is not derived as optimal** — it is one choice
   among many.

### F. Relation to This Thesis

DFM provides the cleanest theoretical framework for understanding why
remasking works: corrector steps are flows that reduce entropy along
the generative trajectory without changing the marginal distribution.
The thesis's main theorem (remasking reduces $E_{\text{fact}}$ under
consistent confidence signals) can be reinterpreted in DFM language
as: *confidence-guided corrector flows reduce the discrete ODE
discretisation error*. Including DFM in the thesis literature review
strengthens the theoretical framing and connects the Riemann-sum
perspective of Lavenant & Zanella to the broader flow-matching
literature.

---

## 12. Mask-Predict — Historical Precursor to Remasking

**Citation:** Ghazvininejad, Levy, Liu & Zettlemoyer (2019),
"Mask-Predict: Parallel Decoding of Conditional Masked Language Models."
EMNLP 2019. arXiv:1904.09324.

### A. Summary

Mask-Predict introduces the first iterative masked decoding algorithm
for non-autoregressive machine translation. Starting from a fully
masked target sequence, the model iteratively (1) predicts all masked
tokens in parallel, then (2) re-masks the lowest-confidence predictions
(by token probability) and repeats. This converges in $O(T)$ parallel
steps rather than $O(n)$ autoregressive steps, achieving a 4–9×
speedup over autoregressive decoding with competitive BLEU scores.

This paper is the direct precursor of modern remasking strategies in
masked diffusion models. The confidence signal (token probability),
the iterative unmask/remask loop, and the parallel decoding paradigm
are all foundational to the MDM remasking literature.

### B. Method Details

**Algorithm:**
```
y = [M, M, ..., M]   (fully masked target)
for t = 1, ..., T:
    # Predict all masked positions in parallel
    p^i = P(y^i | x, y\{y^i=[M]})   (BERT-style conditional LM)
    # Unmask: commit top-p% most confident predictions
    n_t = ceil(|y| * (T - t) / T)   (linearly decreasing remask count)
    y^i ← argmax p^i   for all i
    # Remask: re-mask n_t lowest-confidence committed tokens
    remask_set ← argsort(max p^i)[:n_t]
    y^i ← [M]   for i in remask_set
return y
```

**Confidence signal:** $\max_j p^i_j$ — the probability of the
top-1 prediction. This is the *max-probability* signal, the simplest
of the three training-free signals evaluated in this thesis
(max-prob, entropy, margin).

**Masking schedule:** linearly decreasing number of masked tokens
per step. This is the linear schedule of the thesis's `schedule`
strategy.

### C. Theoretical Contributions

Mask-Predict does not provide theoretical guarantees. The algorithm is
motivated empirically and by the intuition that low-confidence tokens
benefit most from re-prediction in a richer context (where other tokens
have been committed). No connection to diffusion processes, information
profiles, or sampling error bounds.

### D. Confidence Signal

$\max_j p^i_j$ — the max-probability of the conditional LM at position
$i$. Cheap to compute (one forward pass). Heuristically motivated.

### E. Limitations

1. **Heuristic only** — no theoretical justification.
2. **Designed for machine translation** (fixed-length, conditional
   generation) — not directly applicable to unconditional language
   modelling without adaptation.
3. **No probabilistic framework** — Mask-Predict is not a diffusion
   model and lacks the noise-schedule, forward process, and ELBO of
   MDMs.
4. **Max-probability is miscalibrated** at different masking rates —
   the same probability value means different things when 10% vs. 90%
   of tokens are masked.

### F. Relation to This Thesis

Mask-Predict is the intellectual ancestor of the thesis. The thesis can
be described as: *providing a rigorous probabilistic and
information-theoretic foundation for the heuristic remasking procedure
of Mask-Predict, in the context of masked diffusion models.* The three
training-free confidence signals (max-prob, entropy, margin) generalise
the max-probability signal of Mask-Predict. The linear remask schedule
of the thesis's `schedule` strategy is a direct generalisation of
Mask-Predict's linearly decreasing mask count. Citing Mask-Predict
situates the thesis in a broader context beyond MDMs and shows that
principled remasking has implications for the broader non-autoregressive
text generation literature.

---

## Cross-Paper Comparison

### Comparison Table

| # | Paper | Backbone | Confidence Signal | Training? | Inference-Only? | Theoretical Guarantee | Metric | Dataset |
|---|---|---|---|:---:|:---:|---|---|---|
| 1 | MD4 / MDLM | Custom (130M) | None | Pre-train | Yes | ELBO bound | Perplexity | OWT, LM1B |
| 2 | ReMDM | MDLM-OWT | Entropy (opt.) | No | Yes | KL mono. in $R$ | Perplexity | OWT |
| 3 | RemeDi | TPS+UPS (9B) | Learned $\psi^i$ | Yes (RL) | No | None (empirical) | Instruction acc. | Custom |
| 4 | PRISM | LLaDA-8B | Learned $g_\phi$ | Yes (head) | No | BCE $\to$ true cond. | Gen. quality | OWT, MBPP |
| 5 | Inf. Corr. | Hollow transf. | Log-margin | Retrain | No | MCMC mixing | PPL, diversity | Text8, OWT |
| 6 | EB-Sampler | Any MDM | Entropy $H(p_\theta)$ | No | Yes | $E_f$ Riemann bd. | PPL, speed | OWT, LM1B |
| 7 | Lav. & Zan. | Any MDM | N/A (analysis) | No | N/A | KL = $E_l + E_f$ | N/A | N/A |
| 8 | D3PM | Custom | None | Pre-train | Yes | VLB decomp. | NLL | Text8, image |
| 9 | SEDD | GPT-2 scale | Score ratio | Pre-train | Yes | PC convergence | Perplexity | OWT |
| 10 | LLaDA | 8B transformer | None | Pre-train | Yes | None (empirical) | Benchmarks | 2.3T tokens |
| 11 | DFM | Any MDM | Velocity field | Pre-train | Yes | Path integral | PPL, audio | OWT, EnCodec |
| 12 | Mask-Predict | BERT-scale | Max-prob | Pre-train | Yes | None | BLEU | WMT MT |

---

### Narrative Discussion

#### 1. The Genealogy of Masked Diffusion

The MDM literature has a clear ancestry. **D3PM** (Paper 8) established
the absorbing diffusion process in 2021. **SEDD** (Paper 9, 2024)
introduced the score-entropy objective and PC samplers. **MD4/MDLM**
(Paper 1, 2024) simplified training to masked cross-entropy and scaled
to 130M on OpenWebText. **LLaDA** (Paper 10, 2025) scaled to 8B and
demonstrated competitive instruction following. **ReMDM** (Paper 2,
2025) added principled remasking on top of MDLM. **RemeDi** (Paper 3,
2025) and **PRISM** (Paper 4, 2025) added learned confidence heads.
**DFM** (Paper 11, 2024) provides the continuous-time theoretical
unification.

The thesis sits at the intersection of two threads: the
information-theoretic analysis thread (Lavenant & Zanella → EB-Sampler)
and the remasking-quality thread (Mask-Predict → MDLM → ReMDM →
RemeDi/PRISM). Its contribution is to bring the rigour of the first
thread to the practice of the second.

#### 2. Training-Free vs. Fine-Tuned Approaches

| Approach | Papers | Advantage | Disadvantage |
|---|---|---|---|
| Training-free | ReMDM, EB-Sampler, MD4/MDLM, SEDD, **Thesis** | Portable to any pretrained MDM | Confidence signal may be miscalibrated |
| Head-fine-tuned | PRISM, RemeDi | Calibrated, learnable signal | Not portable; requires training data |
| Architecture-retrained | Informed Correctors | Exact per-position conditional | Not compatible with existing models |

The thesis occupies the training-free cell. Its theoretical contribution
is to characterise when training-free signals are *sufficient* — i.e.,
when they are consistent estimators of $H(x^i \mid \text{context})$ —
and to derive the $E_{\text{fact}}$ reduction achievable under this
assumption.

#### 3. Posterior-Based vs. Heuristic Confidence: Three Tiers

**Tier 1 — True posterior:** $I^i(x) = H(x^i \mid x^{\setminus i})$.
Theoretically optimal but requires knowing $x$ or $O(L)$ forward
passes. The reference object in Lavenant & Zanella and Informed
Correctors.

**Tier 2 — Learned approximation:** PRISM's $g_\phi$ and RemeDi's
$\psi^i$ approximate the true conditional with formal convergence
guarantees (PRISM) or empirical calibration (RemeDi). Portable only
within a trained model family.

**Tier 3 — Heuristic training-free:** max-probability, entropy,
margin — all computable from standard forward-pass logits. No formal
guarantee. The thesis provides formal *conditions* under which Tier 3
is sufficient for principled remasking.

#### 4. The EB-Sampler–Lavenant & Zanella Connection

These two papers are two sides of the same coin. Lavenant & Zanella
derive that $E_{\text{fact}}$ is a Riemann error of $\int I(x)\, dt$.
EB-Sampler derives the algorithm that minimises this error for
*unmasking*: consume equal entropy per step by unmasking positions in
ascending $H(p_\theta^i)$ order.

The thesis's contribution is the *remasking* analogue:

$$\underbrace{\text{EB-Sampler}}_{\text{minimises } E_f \text{ for unmasking}} \longleftrightarrow \underbrace{\text{Thesis}}_{\text{minimises } \Delta E_f \text{ for remasking}}$$

where $\Delta E_f$ is the additional $E_{\text{fact}}$ reduction
achievable by one correction pass.

#### 5. The Role of Discrete Flow Matching and SEDD

DFM (Paper 11) and SEDD (Paper 9) provide two complementary
theoretical lenses on the same object:

- **SEDD's PC sampler** views remasking as a Langevin corrector on
  the discrete score. The convergence guarantee is an exponential mixing
  bound (similar to Informed Correctors).
- **DFM's corrector** views remasking as a time-reversed probability
  flow. The optimality criterion is path-integral KL minimisation.
- **Lavenant & Zanella** views remasking as local Riemann refinement
  of the information profile integral.

All three frameworks agree qualitatively: concentrate corrections on
high-entropy positions. The thesis works in the Lavenant & Zanella
framework because it directly connects to the MDM training objective
(the information profile is related to the ELBO loss) and provides the
most explicit, computable $E_{\text{fact}}$ bounds.

#### 6. Mask-Predict as the Historical Anchor

Mask-Predict (Paper 12) shows that confidence-based remasking has been
effective in practice for six years. The thesis provides the rigorous
theoretical explanation for *why* it works — and *when* it optimally
reduces approximation error — which Mask-Predict never attempted to
address. This narrative arc (heuristic → theory) is a compelling
framing for the thesis introduction and conclusion.

#### 7. The Open Gap and the Thesis Contribution

Combining all twelve papers, the landscape is:

- **Unmasking theory is complete:** D3PM → MDLM → Lavenant & Zanella
  → EB-Sampler form a complete chain from foundations to optimal
  algorithms.
- **Remasking practice exists:** Mask-Predict → ReMDM → RemeDi/PRISM
  demonstrate empirically effective remasking, with increasingly
  principled confidence signals.
- **Remasking theory is missing:** No paper connects remasking to
  $E_{\text{fact}}$, the information profile $I(x)$, or the Lavenant &
  Zanella optimality criterion.

The thesis fills this gap. The central theorem to prove:

> *Under the assumption that $c^i$ is a consistent estimator of
> $H(x^i \mid \text{context})$, confidence-guided remasking with
> threshold $\tau$ reduces $E_{\text{fact}}$ by $\Delta(\tau, I) \geq 0$,
> with $\Delta > 0$ whenever the information profile is non-uniform and
> $\tau$ is chosen to target the high-$I^i$ positions.*

This result unifies the entire landscape: ReMDM's $\sigma_t$ is the
$\tau = 0$ case; EB-Sampler's entropy budget determines $\tau$
implicitly; RemeDi/PRISM's trained signals are consistent estimators
under this assumption; and Informed Correctors' spectral gap result is
a parallel confirmation from a different theoretical framework.

---

## Appendix: Repository Index

| Repo | URL | Role | Submodule? |
|---|---|---|---|
| kuleshov-group/remdm | https://github.com/kuleshov-group/remdm | ReMDM strategies on MDLM-OWT | Yes (`external/remdm`) |
| SeunggeunKimkr/PRISM | https://github.com/SeunggeunKimkr/PRISM | PRISM quality head | Yes (`external/PRISM`) |
| maple-research-lab/RemeDi | https://github.com/maple-research-lab/RemeDi | RemeDi inference code | Yes (`external/remedi`) |
| kuleshov-group/mdlm | https://github.com/kuleshov-group/mdlm | Original MDLM baseline | Add (`external/mdlm`) |
| louaaron/Score-Entropy-Discrete-Diffusion | https://github.com/louaaron/Score-Entropy-Discrete-Diffusion | SEDD PC-sampler baseline | Add (`external/sedd`) |
| GSAI-ML/LLaDA | https://github.com/GSAI-ML/LLaDA | LLaDA-8B inference | HF only; no submodule needed |

---

*Compiled with:*
```bash
pandoc comparison.md -o comparison.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=2.5cm \
  --toc --toc-depth=2
```

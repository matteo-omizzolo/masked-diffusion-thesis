---
title: "Remasking in Masked Diffusion Language Models: A Comparative Survey of Key Works"
author: "MSc Thesis Literature Review — Bocconi University"
date: "March 2026"
geometry: margin=2.5cm
fontsize: 11pt
linkcolor: blue
---

# Remasking in Masked Diffusion Language Models: A Comparative Survey

## Preface

This document reviews seven papers directly relevant to the thesis
"Principled Remasking in Masked Diffusion Language Models" (Bocconi
University, supervised by Prof. Giacomo Zanella). The unifying question
across all papers is: *how can we design sampling algorithms for masked
diffusion models that produce higher-quality text without retraining the
model?* Remasking — re-masking already committed tokens and re-predicting
them in a richer context — is the central mechanism under study. Each paper
is reviewed for its core contribution, method details, theoretical results,
confidence signal, limitations, and relevance to the thesis.

---

## 1. MD4 and MDLM — Foundational Masked Diffusion for Text

**Citations:** Shi et al. (2024), "Simplified and Generalized Masked
Diffusion for Discrete Data" (MD4); Sahoo et al. (2024), "Simple and
Effective Masked Diffusion Language Models" (MDLM).

### A. Summary

MD4 and MDLM established the modern framework for masked diffusion language
models (MDMs). Both papers adapt the diffusion paradigm — originally
developed for continuous data (images, audio) — to discrete token sequences.
The key insight is that masking is the natural analogue of Gaussian noise for
categorical variables: the forward process gradually replaces tokens with a
special [MASK] token, and the reverse process learns to unmask them.

MD4 provides the general theoretical treatment, deriving the evidence lower
bound (ELBO) for masked diffusion and showing that the optimal reverse
process factorises over positions given the masked context at each noise
level. MDLM simplifies the ELBO into a form that is both computationally
tractable and competitive with autoregressive language models, introducing
importance-weighted training and demonstrating strong performance on
OpenWebText at the 130M parameter scale (MDLM-OWT, the primary experimental
backbone of this thesis).

### B. Method Details

**Forward process.** Given a clean token sequence $x \in \mathcal{V}^L$
($\mathcal{V}$ = vocabulary, $L$ = sequence length), the forward process at
noise level $t \in [0, 1]$ is:

$$q(z_t \mid x) = \prod_i \mathrm{Cat}\!\left(z_t^i;\; \alpha_t \cdot e_{x^i} + (1 - \alpha_t) \cdot e_m\right)$$

where $\alpha_t \in [0, 1]$ is the token retention probability (a
monotonically decreasing function of $t$), $e_{x^i}$ is the one-hot vector
for token $x^i$, $e_m$ is the one-hot vector for the mask token [M], and
positions are treated independently given $x$. This means $z_t^i = x^i$
with probability $\alpha_t$, and $z_t^i = [\text{M}]$ with probability
$1 - \alpha_t$.

**Noise schedule.** MD4 uses a linear schedule $\alpha_t = 1 - t$. MDLM
uses a cosine schedule, which is smoother and empirically superior:

$$\alpha_t = \cos^2\!\left(\frac{\pi t}{2}\right)$$

**Reverse process (inference).** The generative model is:

$$p_\theta(z_s \mid z_t) = \prod_i p_\theta(z_s^i \mid z_t)$$

where each factor is obtained by Bayes' rule from the predicted clean token
distribution $p_\theta(x \mid z_t)$. For unmasked positions
($z_t^i \neq [\text{M}]$), the reverse process keeps them fixed. For masked
positions, it samples from:

$$p_\theta(z_s^i \mid z_t) = \frac{\alpha_s}{\alpha_t} \cdot \delta(z_s^i = z_t^i) + \left(1 - \frac{\alpha_s}{\alpha_t}\right) \cdot p_\theta(x^i \mid z_t)$$

This says: with probability $\alpha_s / \alpha_t$, keep the mask; with
probability $1 - \alpha_s / \alpha_t$, commit to the model's prediction.

**Training objective.** MDLM simplifies the ELBO to a sum of masked token
cross-entropy losses, weighted by importance weights $w_t$:

$$\mathcal{L} = \mathbb{E}_{t, x, z_t}\!\left[ w_t \cdot \sum_{\{i:\, z_t^i = [\text{M}]\}} -\log p_\theta(x^i \mid z_t) \right]$$

**Sampling algorithm.** Standard sampling proceeds with $T$ discrete steps
($t_T = 1 \to t_0 = 0$). At each step, the fraction
$(\alpha_{t-1} - \alpha_t)$ of currently masked positions is unmasked by
sampling from $p_\theta(x^i \mid z_t)$. Positions to unmask are chosen
uniformly at random among masked positions (in the simplest case).

### C. Theoretical Contributions

MD4 derives the ELBO for the masked diffusion process and shows it
decomposes into per-timestep denoising losses. It proves that the optimal
reverse kernel is given by the formula above. MDLM additionally shows that
with the cosine schedule the per-step SNR is approximately uniform, which
justifies the improved empirical performance. Neither paper provides a
non-asymptotic error bound on sampling quality.

### D. Confidence Signal

MD4 and MDLM do not explicitly use a confidence signal during inference.
The model produces a full categorical distribution $p_\theta(x^i \mid z_t)$
for each masked position at each step, but the sampling procedure does not
inspect these probabilities to decide *which* positions to unmask or
*whether* to remask. The per-token logits are an implicit confidence signal
available "for free" at every forward pass, but standard MDM samplers ignore
this information.

### E. Limitations

1. **Independence assumption.** The reverse process factorises over
   positions, ignoring dependencies between simultaneously unmasked tokens.
   This is the root cause of $E_{\text{fact}}$ in the Lavenant & Zanella
   framework.
2. **No remasking.** Once a token is committed, it is never revisited.
   Errors made early propagate to the end of generation.
3. **Fixed unmasking schedule.** Positions are unmasked uniformly at random,
   ignoring the model's uncertainty. This is suboptimal when some positions
   are much harder than others.
4. **No formal sampling error analysis.** The papers prove ELBO bounds on
   training but do not bound the KL between the true data distribution and
   the distribution induced by finite-step sampling.

### F. Relation to This Thesis

MDLM-OWT is the primary experimental backbone. The independence assumption
(limitation 1) motivates the entire thesis: remasking is one way to correct
errors made by this assumption. The per-token probability distribution
$p_\theta(x^i \mid z_t)$ is the raw material from which all confidence
signals in this thesis are computed. Understanding the MDLM training
objective and noise schedule is prerequisite for any theoretical extension.

---

## 2. ReMDM — Remasking via a Principled Posterior

**Citation:** Wang et al. (2025), "ReMDM: Improving Masked Diffusion Models
with Remasking."

### A. Summary

ReMDM proposes a principled remasking posterior $\sigma_t$ that plays a
symmetric role to the unmasking posterior in standard MDMs. The key
contribution is showing that remasking committed tokens and re-predicting
them in a richer (more unmasked) context is not only valid but improves
generation quality in a provably monotone way as the number of correction
steps increases. ReMDM requires no retraining: it wraps a pretrained MDM
(MDLM-OWT) and adds correction passes at inference time.

### B. Method Details

**Remasking posterior.** After unmasking a set of positions at time $t$,
ReMDM defines the remasking posterior $\sigma_t(z' \mid z)$ as the
probability of transitioning from a partially unmasked state $z$ to a more
masked state $z'$ by re-masking a subset of committed tokens. Specifically:

$$\sigma_t(z^{\prime i} = [\text{M}] \mid z) = 1 - \frac{\alpha_t}{\alpha_0} \quad \text{(for committed positions } z^i \neq [\text{M}]\text{)}$$

$$\sigma_t(z^{\prime i} = z^i \mid z) = \frac{\alpha_t}{\alpha_0} \quad \text{(for committed positions)}$$

This says: re-mask each committed token independently with probability
proportional to the current noise level. At high $t$ (early in generation),
remasking is aggressive; at low $t$ (near completion), remasking is rare.

**Inference algorithm:**

```
Input: pretrained model p_θ, steps T, corrections R per step
z_T = [M, M, ..., M]   (fully masked)
for t = T, T−1, ..., 1:
    # Standard unmasking step
    z_{t−1} ← sample from p_θ(z_{t−1} | z_t)
    # Correction loop
    for r = 1, ..., R:
        z' ← remask committed tokens using σ_t(z' | z_{t−1})
        z_{t−1} ← sample from p_θ(z_{t−1} | z')
return z_0
```

The number of correction passes $R$ is an inference-time hyperparameter.
Increasing $R$ increases compute cost linearly but improves quality.

**Remasking schedule.** The $\sigma_t$ schedule is calibrated to match the
noise level at each step, ensuring that the correction pass "undoes"
approximately the same amount of information as a single forward step added.

### C. Theoretical Contributions

Wang et al. prove that the distribution induced by ReMDM converges to the
true data distribution as $T \to \infty$ and $R \to \infty$, under the
assumption that $p_\theta$ exactly models the true score. More concretely,
they show that the KL divergence between the ReMDM distribution and the true
distribution is monotonically non-increasing in $R$ (for fixed $T$). This
is the key theoretical result: more correction steps cannot hurt in
expectation. The proof uses the Markov chain convergence properties of the
corrected reverse process.

### D. Confidence Signal

ReMDM's remasking is *not* confidence-guided in the baseline version:
$\sigma_t$ masks positions uniformly at random (among committed tokens).
However, the paper discusses a confidence-weighted variant where the
remasking probability for position $i$ is modulated by:

$$\text{remask\_prob}^i \propto H\!\left(p_\theta(x^i \mid z_t)\right) \quad \text{(entropy of the predicted distribution)}$$

Higher-entropy positions are more likely to be remasked. This variant is
empirically superior but lacks the same theoretical guarantee as the uniform
variant. Computational cost: one forward pass per correction step, same cost
as a standard unmasking step.

### E. Limitations

1. **Uniform remasking ignores information content.** The theoretically
   grounded variant remasks uniformly, which is suboptimal when positions
   have very different uncertainty levels.
2. **No formal connection to the information profile.** The $\sigma_t$
   schedule is heuristically calibrated to the noise level but is not
   derived from $I(x) = \sum_i H(x^i \mid x^{\setminus i})$.
3. **Scales with $T \times R$.** Computational cost is $O(T \times R)$
   forward passes, which can be prohibitive for large models.
4. **No task-specific evaluation.** The paper evaluates on language
   modelling perplexity but does not demonstrate downstream task improvements.

### F. Relation to This Thesis

ReMDM is the closest existing work to the thesis goal. Its $\sigma_t$
posterior is a special case of the "remasking kernel" defined in the thesis
framework (Chapter 3). The thesis extends ReMDM in two directions: (1)
theoretically, by connecting the remasking kernel to the Lavenant & Zanella
$E_{\text{fact}}$ term and deriving *confidence-guided* remasking as
optimal; (2) experimentally, by comparing uniform remasking (ReMDM) against
confidence-guided variants (max-prob, entropy, margin) on a broader set of
benchmarks. The thesis can additionally show that the confidence-weighted
variant of ReMDM reduces $E_{\text{fact}}$ faster than the uniform variant,
providing a principled explanation for its empirical superiority.

---

## 3. RemeDi — Learned Dual-Stream Remasking with RL

**Citation:** Huang et al. (2025), "RemeDi: Reinforcement Learning for
Masked Diffusion Models."

### A. Summary

RemeDi introduces a dual-stream architecture that separates the token
prediction task (what to predict) from the unmasking policy task (when to
unmask). A standard transformer backbone (Token Prediction Stream, TPS)
predicts token identities. A second lightweight stream (Unmasking Policy
Stream, UPS) is trained via binary cross-entropy and then refined with GRPO
reinforcement learning to decide which tokens to unmask and which to remask.
RemeDi learns a policy rather than using a heuristic confidence signal,
making it the most powerful but also the most expensive approach in this
landscape.

### B. Method Details

**Architecture.** TPS is a standard transformer decoder operating on the
partially masked sequence. UPS is a lightweight MLP head on top of TPS
hidden states, producing per-token binary logits:

$$\psi^i = \text{UPS}(h^i) \in [0, 1]$$

where $h^i$ is the TPS hidden state at position $i$, and $\psi^i$ is the
probability that token $i$ should be unmasked (committed) at this step.
Note: $\psi^i$ is *not* the same as the token prediction probability
$p_\theta(x^i \mid z_t)$; it is a separate signal about *when* to commit,
not *what* to predict.

**Training procedure (three stages):**

*Stage 1 — TPS pre-training:* standard MDM ELBO training on a large corpus.
This produces a competent token predictor identical to MDLM.

*Stage 2 — UPS supervised pre-training:* train UPS with BCE loss to predict
whether each token's current prediction matches the ground truth:

$$\mathcal{L}_{\text{UPS}} = -\mathbb{E}\!\left[\sum_i \left[y^i \log \psi^i + (1 - y^i)\log(1 - \psi^i)\right]\right]$$

where $y^i = \mathbf{1}[p_\theta(x^i \mid z_t) = x^i]$ (1 if the current
best prediction is correct, 0 otherwise). This stage gives UPS a calibrated
confidence signal correlated with actual prediction accuracy.

*Stage 3 — GRPO RL fine-tuning:* UPS is refined using Group Relative Policy
Optimization with a reward signal based on final generation quality (e.g.,
perplexity of completed sequences, or task accuracy). This teaches UPS to
make globally optimal unmask/remask decisions, not just locally greedy ones.

**Inference algorithm:**

```
z_T = [M, M, ..., M]
for t = T, T−1, ..., 1:
    (p^i, ψ^i) ← TPS+UPS(z_t)   for all i
    unmask_set ← {i : z_t^i = [M] and ψ^i > τ_unmask}
    z_t^i ← argmax p^i   for i in unmask_set
    remask_set ← {i : z_t^i ≠ [M] and ψ^i < τ_remask}
    z_t^i ← [M]   for i in remask_set
return z_0
```

**Confidence signal stored at unmasking time.** When position $i$ is
unmasked at step $t^*$, RemeDi stores $\psi^i(t^*)$ as the "decoding
probability" of that token. This stored value is used in subsequent steps:
if $\psi^i$ drops significantly below $\psi^i(t^*)$, the token is a
candidate for remasking.

### C. Theoretical Contributions

RemeDi does not provide formal sampling error bounds. The theoretical
justification for UPS is that the BCE-trained $\psi^i$ approximates the
true conditional probability $P(\text{token } i \text{ is correctly
predicted} \mid \text{current context})$, analogous to the PRISM quality
head. The paper proves that the UPS BCE minimiser converges to the true
conditional (same argument as PRISM), but does not connect this to any
information-theoretic property of the sequence.

### D. Confidence Signal

**Signal:** $\psi^i \in [0, 1]$, the UPS output — a learned estimate of
$P(\text{prediction at position } i \text{ is correct})$.

**Computation:** one additional MLP forward pass per token per step (cheap
given that TPS already computed $h^i$). Total overhead: negligible compared
to TPS.

**Key advantage over heuristic signals:** $\psi^i$ is trained to be
calibrated across steps and contexts, not just at a single noise level.

**Key disadvantage:** requires UPS training; not available for arbitrary
pretrained MDMs.

### E. Limitations

1. **Requires fine-tuning.** UPS must be trained on the same model family
   as TPS.
2. **RL training instability.** GRPO requires careful hyperparameter tuning
   and can be unstable, especially for the 9B parameter models.
3. **No connection to optimal sampling theory.** The learned policy is not
   shown to be optimal in any theoretical sense.
4. **Evaluation scope.** The paper focuses on instruction-following tasks;
   unconditional text generation quality is less explored.

### F. Relation to This Thesis

RemeDi provides the strongest empirical baseline for remasking. The thesis
can use RemeDi-RL and RemeDi-Instruct as upper-bound comparators: if
training-free confidence signals (max-prob, entropy, margin) approach the
quality of the learned UPS, that is strong evidence that heuristic signals
are sufficient proxies for the true conditional. Theoretically, RemeDi
motivates the study of what properties a confidence signal must have to be
useful for remasking — the thesis answers this for training-free signals
through the lens of $E_{\text{fact}}$ reduction.

---

## 4. PRISM — Plug-and-Play Quality Head for MDMs

**Citation:** Kim et al. (2025), "PRISM: A Plug-and-Play Quality Head for
Masked Diffusion Language Models."

### A. Summary

PRISM trains a lightweight quality head $g_\phi$ on top of a frozen
pretrained MDM (LLaDA-8B) to predict, for each token position, the
probability that the model's current prediction is correct. This quality
head is trained with binary cross-entropy and can be used at inference time
to guide unmasking decisions without modifying the base model. PRISM
provides a formal theoretical guarantee: the BCE minimiser converges to the
true conditional accuracy $P(x^i_{\text{pred}} = x^i \mid \text{context})$,
which is a principled confidence signal for remasking.

### B. Method Details

**Quality head architecture.** $g_\phi$ is a small MLP (typically 2–3
layers) applied to the frozen hidden states $h^i$ of the base MDM:

$$g_\phi(y)^i = \sigma\!\left(\text{MLP}_\phi(h^i)\right) \in [0, 1]$$

where $y = (y^1, \ldots, y^L)$ is the partially masked sequence and $h^i$
is the hidden state at position $i$.

**Training objective.** For a training sequence $x$ and a randomly masked
version $y$:

$$\mathcal{L}(\phi) = -\mathbb{E}_{x,y}\!\left[\sum_{\{i:\, y^i \neq [\text{M}]\}} \left[\mathbf{1}[x^i = \hat{y}^i] \log g_\phi(y)^i + \mathbf{1}[x^i \neq \hat{y}^i] \log(1 - g_\phi(y)^i)\right]\right]$$

where $\hat{y}^i = \arg\max p_\theta(x^i \mid y)$ is the base model's top-1
prediction.

**Theoretical guarantee.**

*Proposition.* Under suitable regularity conditions, the minimiser $\phi^*$
of $\mathcal{L}(\phi)$ over the class of measurable functions satisfies:

$$g_{\phi^*}(y)^i = P\!\left(x^i = \hat{y}^i \;\middle|\; y \oplus m^i\right)$$

where $y \oplus m^i$ denotes the sequence $y$ with position $i$
additionally masked. This means the optimal quality head provides the
Bayes-optimal estimate of whether the current prediction at position $i$ is
correct, given all other available information.

**Inference algorithm:**

```
z_T = [M, M, ..., M]
for t = T, T−1, ..., 1:
    q^i ← p_θ(x^i | z_t)   for all masked i   (base model)
    g^i ← g_φ(z_t)^i        for all i          (quality head)
    unmask_set ← top-k masked positions by g^i
    z_t^i ← argmax q^i   for i in unmask_set
    # Optional remasking
    remask_set ← {i : z_t^i ≠ [M] and g^i < τ}
    z_t^i ← [M]   for i in remask_set
return z_0
```

### C. Theoretical Contributions

PRISM's main theoretical contribution is the Bayes-optimality proof for
the BCE minimiser. This is significant because it gives a principled
interpretation of the quality head: it is not just a heuristic, but the
best possible confidence signal given the information available at each
step. The proof is a standard result in probability theory (BCE is a proper
scoring rule), adapted to the MDM setting.

### D. Confidence Signal

**Signal:** $g_\phi(y)^i \in [0, 1]$ — estimated probability that the base
model's top-1 prediction at position $i$ is correct.

**Computation:** one MLP forward pass per position per step (negligible for
9B-parameter bases).

**Key property:** $g_\phi$ is trained to be calibrated across noise levels,
accounting for the difficulty of position $i$ at the current masking rate.
This makes it strictly more principled than max-prob or entropy from raw
logits.

**Key limitation:** requires training a separate head for each base model.

### E. Limitations

1. **Requires training.** A new $g_\phi$ must be trained for each base
   model.
2. **Adapter not public.** As of March 2026, the PRISM adapter for
   LLaDA-8B is available only to lab collaborators.
3. **Out-of-distribution calibration.** $g_\phi$ is trained on a specific
   corpus; calibration may degrade out of distribution.
4. **No sampling quality guarantee.** The optimality result is about
   calibration of $g_\phi$, not about the quality of samples generated
   using $g_\phi$ as a guide.

### F. Relation to This Thesis

PRISM's theoretical guarantee motivates the thesis's analysis of
training-free confidence signals. The thesis asks: if we cannot train
$g_\phi$ (because we only have a pretrained MDM), which training-free
signal (max-prob, entropy, margin) best approximates
$P(x^i = \hat{y}^i \mid \text{context})$? PRISM's BCE convergence result
also provides the formal justification for the thesis's "consistent
estimator" assumption in the main theorem: a confidence signal $c^i$ is
consistent if $\mathbb{E}[c^i \mid x] \to P(x^i = \hat{y}^i \mid
\text{context})$ as the model class grows.

---

## 5. Informed Correctors — MCMC-Based Token Correction

**Citation:** Zhao et al. (2024/2025), "Informed Correctors for Discrete
Diffusion Models."

### A. Summary

Informed Correctors applies Markov Chain Monte Carlo correction steps to
discrete diffusion models, including MDMs. The key idea is to run
Gibbs-like updates that resample individual token positions conditioned on
all other positions, using the per-token marginal log-likelihood as the
acceptance criterion. The paper provides a theoretical mixing analysis
showing that the Gibbs corrector converges exponentially fast to the target
distribution, and that confidence-guided corrections converge faster than
uniform ones.

### B. Method Details

**Gibbs corrector.** At each correction step, a position $d$ is selected
(randomly or by confidence) and resampled:

$$x^d_{\text{new}} \sim q_t(x^d \mid x^{\setminus d})$$

This is exact Gibbs sampling from the MDM's joint distribution at noise
level $t$. Repeated application converges to $q_t(x)$.

**Informed corrector (confidence-guided variant).** Select position $d$
with probability proportional to the "surprise" at that position:

$$p(\text{select } d) \propto \exp\!\left(-\beta \cdot \log q_t(x^d \mid x^{\setminus d})\right)$$

for temperature $\beta > 0$. High surprise (low log-probability) → high
selection probability → more likely to be corrected.

**Hollow transformer.** Computing $q_t(x^d \mid x^{\setminus d})$ exactly
requires a forward pass with position $d$ masked. For efficiency, the paper
proposes the *hollow transformer*: a transformer with diagonal-masked
attention that computes all per-position conditionals in a single forward
pass. This architecture is not available in standard pretrained MDMs.

**Full inference algorithm:**

```
z_T = [M, M, ..., M]
# Standard unmasking to z_0
z_0 ← standard MDM sampler(z_T)
# Correction loop
for r = 1, ..., R:
    for each position d:
        log_q_d ← log q_0(x^d | x^{∖d})   (hollow transformer)
        if −log_q_d > threshold:
            x^d_new ~ q_0(x^d | x^{∖d})
            x^d ← x^d_new
return x
```

### C. Theoretical Contributions

**Mixing time bound.** Theorem 3 of Zhao et al. states that the Gibbs
corrector with $R$ steps satisfies:

$$\mathrm{KL}\!\left(q_0(x) \;\|\; p_R(x)\right) \leq e^{-R\lambda} \cdot \mathrm{KL}\!\left(q_0(x) \;\|\; p_0(x)\right)$$

where $\lambda > 0$ is the spectral gap of the Gibbs chain. This gives a
formal guarantee that correction improves quality exponentially in $R$.

**Informed corrector acceleration.** The paper proves that the informed
(confidence-guided) corrector achieves the same KL reduction as the uniform
corrector but with fewer steps: the effective spectral gap satisfies
$\lambda_{\text{informed}} \geq \lambda_{\text{uniform}}$, with equality
only when all positions have equal surprise. Concentrating corrections on
high-surprise positions is strictly more efficient than uniform resampling.

### D. Confidence Signal

**Signal:** $\log q_t(x^d \mid x^{\setminus d})$ — the per-position
conditional log-probability under the MDM at noise level $t$.

**Computation:** one forward pass per position (naively $O(L)$ forward
passes) or one pass with a hollow transformer ($O(1)$ passes). Without a
hollow transformer, this signal is very expensive to compute exactly.

### E. Limitations

1. **Requires hollow transformer.** Not applicable to standard pretrained
   MDMs (MDLM-OWT, LLaDA-8B) without retraining.
2. **Post-hoc correction only.** The corrector is applied after standard
   MDM sampling, not integrated into the denoising loop.
3. **Mixing time depends on the spectral gap.** Can be slow for sequences
   with strong long-range dependencies.
4. **No connection to the information profile.** The correction procedure
   is motivated by MCMC theory but not connected to $I(x)$ or $E_{\text{fact}}$.

### F. Relation to This Thesis

The Informed Correctors paper provides the strongest theoretical framework
for correction-based remasking. Its key result — that confidence-guided
correction converges faster than uniform correction — is a specialised
version of the thesis's main claim, but in a different theoretical setting
(MCMC mixing vs. Riemann approximation error). The thesis can cite this as
corroborating evidence from a complementary angle.

The hollow transformer requirement is a practical limitation the thesis
explicitly avoids: all training-free strategies are designed to work with
any pretrained MDM using only forward-pass logits, without architectural
modifications.

---

## 6. EB-Sampler — Entropy-Bounded Adaptive Unmasking

**Citation:** Ben-Hamu et al. (2025), "EB-Sampler: Entropy-Bounded Sampling
for Masked Diffusion Models."

### A. Summary

The EB-Sampler adapts the number of tokens unmasked at each step based on
the *information content* of the positions being unmasked, targeting a fixed
entropy budget per step. This is the most direct existing connection to the
information profile $I(x)$ in the literature. The key insight is that
standard MDM samplers with fixed step counts waste compute: they either
unmask too many easy tokens (wasting steps) or too many hard tokens (making
large errors) in a single step. The EB-Sampler achieves 2–3× speedup over
fixed-step baselines at the same quality level, or significantly better
quality at the same step count.

### B. Method Details

**Information profile.** For a sequence $x$, the information profile is:

$$I^i(x) = H(x^i \mid x^{\setminus i}) \quad \text{(conditional entropy at position } i\text{)}$$

with total information $I(x) = \sum_i I^i(x)$. Since $I^i(x)$ requires
knowing $x$, it is approximated by:

$$\hat{I}^i(z_t) = H\!\left(p_\theta(x^i \mid z_t)\right) \quad \text{(predicted entropy)}$$

**Entropy budget.** At each step, the EB-Sampler unmasks only as many
tokens as needed to consume a fixed entropy budget $\varepsilon$:

$$\text{unmask\_set}_t = \arg\min_{S:\, \sum_{i \in S} \hat{I}^i(z_t) \geq \varepsilon} |S|$$

Solved by a greedy algorithm: sort masked positions by predicted entropy
(ascending), unmask until budget $\varepsilon$ is consumed.

**Algorithm:**

```
z_T = [M, M, ..., M]
while any z_t^i = [M]:
    Î^i ← H(p_θ(x^i | z_t))   for all masked i
    sorted_i ← argsort(Î, ascending)
    budget = ε;   unmask_set = {}
    for i in sorted_i:
        if budget ≥ Î^i:
            unmask_set ← unmask_set ∪ {i};   budget ← budget − Î^i
        else: break
    z ← unmask tokens in unmask_set
return z
```

### C. Theoretical Contributions

The EB-Sampler is directly motivated by the Lavenant & Zanella framework.
Theorem 2 of Ben-Hamu et al. states: with entropy budget $\varepsilon$ per
step,

$$E_{\text{fact}}(\text{EB}) \leq \varepsilon \cdot \max_i \hat{I}^i$$

The paper also proves that the EB-Sampler is optimal among all deterministic
unmasking strategies: no deterministic schedule achieves a lower
$E_{\text{fact}}$ bound with the same total number of forward passes.
Crucially, this optimality applies to *unmasking only* — remasking is not
considered.

### D. Confidence Signal

**Signal:** $\hat{I}^i = H(p_\theta(x^i \mid z_t))$ — the model's predicted
entropy at each masked position.

**Computation:** free — entropy is computed from logits returned by the
standard forward pass.

**Role:** entropy determines which positions to unmask (lowest entropy
first) and how many to unmask (until budget $\varepsilon$ is consumed).

### E. Limitations

1. **Unmasking only.** The EB-Sampler does not consider remasking; its
   optimality result does not extend to the correction direction.
2. **Approximation quality.** The bound depends on how well $\hat{I}^i$
   approximates $I^i(x)$.
3. **Variable step count.** Adaptive step count complicates comparison
   against fixed-budget baselines.
4. **No error correction.** Errors made during unmasking are not corrected.

### F. Relation to This Thesis

The EB-Sampler is the closest existing work to the *theoretical* goals of
the thesis in the *unmasking* direction. The thesis's contribution in the
*remasking* direction is precisely its analogue: derive a principled,
information-theoretically motivated strategy for deciding which committed
tokens to remask. The thesis frames this as:

> *EB-Sampler optimises the unmasking schedule; the thesis optimises the
> remasking schedule.*

The bound $E_{\text{fact}}(\text{EB}) \leq \varepsilon \cdot \max_i
\hat{I}^i$ is a key reference. The thesis should derive a corresponding
bound for the corrected process (unmasking + remasking) and show that
remasking reduces the effective $E_{\text{fact}}$ relative to the EB-Sampler
baseline.

---

## 7. Error Bounds — Lavenant & Zanella (2025)

**Citation:** Lavenant & Zanella (2025), "Sampling Error Analysis for
Masked Diffusion Models."

### A. Summary

This paper provides the first rigorous non-asymptotic analysis of the
sampling error of MDM algorithms. It decomposes the KL divergence between
the true data distribution and the distribution induced by a finite-step
MDM sampler into two terms: a learning error $E_{\text{learn}}$ (due to
imperfect model training) and a factorization error $E_{\text{fact}}$ (due
to simultaneous unmasking of multiple tokens). $E_{\text{fact}}$ is a
Riemann approximation error of the sequence's information profile, and the
paper derives the optimal unmasking schedule that minimises $E_{\text{fact}}$.
This is the theoretical foundation for the entire thesis.

### B. Method Details

**KL decomposition.** For a MDM sampler with $T$ unmasking steps and
learned model $p_\theta$:

$$\mathrm{KL}(\pi(x) \;\|\; p_{\text{alg}}(x)) \leq E_{\text{learn}} + E_{\text{fact}}$$

where $\pi(x)$ is the true data distribution and $p_{\text{alg}}(x)$ is
the algorithmic distribution.

**Learning error:**

$$E_{\text{learn}} = \sum_{t=1}^{T} \mathbb{E}_{z_t}\!\left[\mathrm{KL}\!\left(q(x \mid z_t) \;\|\; p_\theta(x \mid z_t)\right)\right]$$

This goes to zero as the model approaches perfect training and is not
affected by the sampling algorithm's choice of which tokens to unmask.

**Factorization error.** For small step sizes $\Delta\alpha_t =
\alpha_{t-1} - \alpha_t$:

$$E_{\text{fact}}(t) \approx (\Delta\alpha_t)^2 \cdot \sum_i \mathrm{Var}_{x \sim q(x \mid z_t)}\!\left[I^i(x)\right]$$

Summing over steps, $E_{\text{fact}}$ is a Riemann approximation error of
the integral $\int I(x)\, dt$.

**Information profile:**

$$I(x) = \sum_i H(x^i \mid x^{\setminus i})$$

The Riemann error is minimised when each step releases approximately equal
information: $\Delta I_t \approx I(x) / T$ for all $t$.

**Optimal unmasking schedule.** The optimal strategy is to unmask tokens in
order of *increasing* information content — easy tokens (low $I^i$) first,
hard tokens last. This is exactly the EB-Sampler algorithm.

**Main bound (informal).** For any unmasking schedule:

$$E_{\text{fact}} \leq C \cdot \left(\frac{I(x)}{T}\right)^2$$

for a constant $C$ depending on sequence length and vocabulary size. The
optimal schedule achieves the matching lower bound.

### C. Theoretical Contributions

1. First non-asymptotic KL bound for MDM sampling error.
2. Decomposition into learnable ($E_{\text{learn}}$) and algorithmic
   ($E_{\text{fact}}$) components.
3. Identification of $E_{\text{fact}}$ as a Riemann approximation error
   of $I(x)$.
4. Derivation of the optimal unmasking schedule.
5. Proof that simultaneous unmasking of $k$ tokens incurs error $O(k^2)$
   times larger than sequential unmasking.

**The explicit open gap** noted by the authors: the framework does not yet
cover remasking transitions in the reverse process. This is the central
research question of the thesis.

### D. Confidence Signal

The paper does not use a confidence signal in the sampling algorithm.
However, its analysis implies that the optimal signal for unmasking order
is $I^i(x)$ — the true per-position information content. The model-predicted
entropy $H(p_\theta(x^i \mid z_t))$ is the natural approximation,
motivating the thesis's use of entropy as a confidence signal for remasking.

### E. Limitations

1. **Unmasking only.** The framework does not cover remasking transitions.
   This is the gap the thesis fills.
2. **$E_{\text{learn}}$ treated as a black box.** The paper bounds
   $E_{\text{learn}}$ by the model's training error but does not analyse
   how it depends on architecture or training procedure.
3. **Assumes factored posterior.** The analysis uses the factored form
   of the true posterior at each step, an approximation for highly
   correlated sequences.
4. **Asymptotic tightness.** The bound becomes tight only for small step
   sizes; constant factors may be large for practical $T = 50$–$200$.

### F. Relation to This Thesis

This paper *is* the thesis's theoretical foundation. Chapter 3 of the thesis
extends the Lavenant & Zanella framework in one direction: adding remasking
transitions to the reverse process and deriving the corresponding
modification to $E_{\text{fact}}$. The main theorem of the thesis states
conditions under which remasking reduces $E_{\text{fact}}$ relative to the
optimal unmasking-only schedule, and connects this to confidence signals.

---

## Cross-Paper Comparison

### Comparison Table

| Paper | Backbone | Confidence Signal | Training Req.? | Inference-Only? | Theoretical Guarantee | Metric | Dataset |
|---|---|---|:---:|:---:|---|---|---|
| MD4 / MDLM | Custom (130M) | None | Pre-train | Yes | ELBO bound (training) | Perplexity | OWT, LM1B |
| ReMDM | MDLM-OWT | Entropy (optional) | No | Yes | KL monotone in $R$ | Perplexity | OWT |
| RemeDi | TPS+UPS (9B) | Learned $\psi^i$ | Yes (UPS+RL) | No | None (empirical) | Instruction acc. | Custom |
| PRISM | LLaDA-8B | Learned $g_\phi$ | Yes (head) | No | BCE $\to$ true cond. | Gen. quality | OWT, tasks |
| Informed Corr. | Hollow transf. | Log-margin (exact) | Retrain | No | MCMC mixing time | PPL, diversity | Text8, OWT |
| EB-Sampler | Any MDM | Entropy $H(p_\theta)$ | No | Yes | $E_{\text{fact}}$ Riemann bd. | PPL, speed | OWT, LM1B |
| Lavenant & Z. | Any MDM | N/A (analysis) | No | N/A | KL = $E_l + E_f$ | N/A | N/A |

**Compute cost notes.** MDLM-OWT (130M): ~10 ms per forward pass on A100;
all inference-only methods are viable on consumer hardware. RemeDi (9B) and
LLaDA-8B: ~500 ms per forward pass on A100; multi-minute generation per
sample. PRISM adds negligible overhead (small MLP on frozen hidden states).
Informed Correctors with hollow transformer: requires full retraining,
comparable to MDLM pre-training cost.

---

### Narrative Discussion

#### 1. Training-Free vs. Fine-Tuned Approaches

The seven papers span two clearly distinct camps. **Training-free** methods
(MD4/MDLM baseline, ReMDM, EB-Sampler, and the thesis's contribution) take
a pretrained MDM and modify only the inference-time sampling procedure.
**Fine-tuned** methods (RemeDi, PRISM, Informed Correctors) require
additional training on top of the base model.

The training-free camp has an important practical advantage: it is
immediately applicable to any pretrained MDM without access to training data
or compute. This is critical for the thesis: MDLM-OWT, LLaDA-8B, and
RemeDi models are all public, and the thesis can test strategies on all of
them without retraining. The fine-tuned methods can potentially achieve
higher quality because they directly optimise the remasking decision, but
they are not portable across model families.

The key open question — and the thesis's central contribution — is: *how
close can a training-free, confidence-guided remasking strategy get to the
performance ceiling set by fine-tuned methods?* If the answer is "very
close," training is unnecessary; if "far," the bottleneck is the quality of
the confidence signal, and training a PRISM-like head is the path forward.

#### 2. Posterior-Based vs. Heuristic Confidence

Confidence signals in the literature fall into three tiers:

**Tier 1 — True posterior (theoretically ideal, computationally expensive):**
$I^i(x) = H(x^i \mid x^{\setminus i})$. Requires knowing $x$ or running
$O(L)$ forward passes. Used in Lavenant & Zanella as the theoretical
reference. Approximated by Informed Correctors' log-margin (requires hollow
transformer).

**Tier 2 — Learned approximation (theoretically grounded, requires
training):** PRISM's quality head $g_\phi$ and RemeDi's UPS $\psi^i$ both
approximate the true posterior $P(x^i = \hat{y}^i \mid \text{context})$
with formal convergence guarantees (PRISM) or empirical calibration
(RemeDi). Strictly more principled than heuristic signals but require
training data and compute.

**Tier 3 — Heuristic signals (no training, no formal guarantee):**
Max-probability ($\max_v p_\theta(x^i = v \mid z_t)$), entropy
($H(p_\theta(x^i \mid z_t))$), and margin (probability gap between top two
tokens) are all computable from the base model's output logits at no
additional cost. Widely used but lack formal guarantees about their
relationship to the true per-position information content.

The thesis bridges Tier 1 and Tier 3: it provides formal conditions under
which Tier 3 signals are *sufficient* for principled remasking (the
"consistent estimator" assumption), and empirically tests how close they
are to the Tier 1 optimum.

#### 3. The EB-Sampler—Lavenant & Zanella Connection

The EB-Sampler and Lavenant & Zanella are, in a precise sense, two sides of
the same coin. Lavenant & Zanella prove that $E_{\text{fact}}$ is a Riemann
approximation error of $\int I(x)\, dt$. The EB-Sampler derives the
*algorithm* that minimises this Riemann error: unmask in increasing order of
$H(p_\theta(x^i \mid z_t))$, consuming a fixed entropy budget $\varepsilon$
per step. The EB-Sampler's bound ($E_{\text{fact}}(\text{EB}) \leq
\varepsilon \cdot \max_i \hat{I}^i$) is a direct instantiation of the
Lavenant & Zanella bound with the optimal schedule.

This connection has a concrete implication for the thesis: the EB-Sampler
is the optimal *unmasking* algorithm; the thesis derives the optimal
*remasking* algorithm in the same framework. The analogy is:

$$\underbrace{\text{EB-Sampler}}_{\text{optimal unmasking}} \longleftrightarrow \underbrace{\text{Thesis result}}_{\text{optimal remasking}}$$

where "optimal remasking" means: given that unmasking has already produced
a candidate sequence with some committed tokens, which tokens should be
remasked to maximally reduce the residual $E_{\text{fact}}$?

#### 4. The Open Gap: Remasking Theory

Despite seven papers spanning the full landscape of MDM sampling, no
existing work provides a theoretical characterisation of remasking in terms
of the information profile or the Lavenant & Zanella error bound.
Specifically:

- **ReMDM** proves monotone improvement in KL as $R \to \infty$ but does
  not connect to $E_{\text{fact}}$ or $I(x)$.
- **RemeDi** learns a remasking policy but provides no formal guarantee.
- **PRISM** guarantees the quality head's calibration but not the sampling
  quality of the guided algorithm.
- **Informed Correctors** provides MCMC mixing guarantees in a different
  theoretical framework.
- **EB-Sampler** and **Lavenant & Zanella** cover unmasking only.

The thesis fills this gap. The central contribution is:

> *Under the assumption that the confidence signal $c^i$ is a consistent
> estimator of $H(x^i \mid \text{context})$, confidence-guided remasking
> with threshold $\tau$ reduces $E_{\text{fact}}$ by $\Delta(\tau, I) \geq 0$,
> with $\Delta > 0$ whenever the information profile is non-uniform and
> $\tau$ is chosen to target the high-$I^i(x)$ positions.*

This result unifies the literature: ReMDM's uniform remasking is the
$\tau = 0$ special case; the EB-Sampler's entropy budget determines the
remasking threshold implicitly; and RemeDi/PRISM's trained signals are
consistent estimators in the sense of the theorem's assumption.

---

*Compiled with:*
```bash
pandoc comparison.md -o comparison.pdf \
  --pdf-engine=xelatex \
  -V mainfont="Latin Modern Roman" \
  --toc --toc-depth=2 \
  -V geometry:margin=2.5cm
```

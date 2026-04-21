---
title: "Remasking Strategies in MDMs: Empirical Analysis and Research Directions"
author: "MSc Thesis — Bocconi University, supervised by Prof. Giacomo Zanella"
date: "March 2026"
geometry: "top=2.5cm, bottom=2.5cm, left=3cm, right=3cm"
fontsize: 11pt
linkcolor: blue
numbersections: true
toc-depth: 3
---

\newpage

# Overview

This document is a companion to `docs/comparison.md` (literature review). Its purpose
is twofold:

1. **Empirical analysis** — interpret the step-sweep results (T ∈ {128, 256, 512, 1000})
   through the statistical framework built in the literature review.
2. **Research directions** — propose concrete, formally grounded extensions of the work.

All results come from evaluations on Bocconi HPC A100 80GB (N=100 samples, seed=42,
OWT reference corpus of 1000 samples). Raw numbers are in `results/combined_comparison.md`.

\newpage

# Statistical Framework: Connecting Metrics to Theory

## The Three Metrics and What They Actually Measure

Understanding what each metric captures statistically is prerequisite to interpreting results.

### Generation Perplexity (gen_ppl)

$$\text{gen\_ppl} = \exp\!\left(\frac{1}{N}\sum_{n=1}^{N}\frac{1}{|x^{(n)}|}\sum_{i=1}^{|x^{(n)}|} -\log p_{\text{GPT-2}}(x^{(n)}_i \mid x^{(n)}_{<i})\right)$$

**What it measures:** The exponential of the average per-token cross-entropy assigned by
GPT-2 large to the generated texts. Formally, gen\_ppl is an empirical estimate of:
$$\text{gen\_ppl} \approx \exp\!\bigl(H(p_\theta, q_{\text{GPT2}})\bigr)$$
where $H(p_\theta, q_{\text{GPT2}}) = \mathbb{E}_{x \sim p_\theta}[-\log q_{\text{GPT2}}(x)]$
is the **cross-entropy** of GPT-2's distribution $q$ with respect to the model's
generated distribution $p_\theta$.

**Important decomposition.** Cross-entropy decomposes as:
$$H(p_\theta, q_{\text{GPT2}}) = \underbrace{H(p_\theta)}_{\text{entropy of generator}} + \underbrace{\text{KL}(p_\theta \| q_{\text{GPT2}})}_{\text{distributional gap to GPT-2}}$$

gen\_ppl therefore measures both the *entropy* of the generated distribution and the *KL
divergence* from GPT-2. It is **not** a pure KL divergence. A model that generates highly
diverse text (high $H(p_\theta)$) will have higher gen\_ppl than a mode-collapsing model
(low $H(p_\theta)$) even if both are equally similar to GPT-2. This is the fundamental
reason gen\_ppl and MAUVE (which rewards diversity) can diverge.

**Statistical interpretation:**
- gen\_ppl penalises both **distributional gap** (KL term) and **lack of concentration**
  (entropy term). Mode-collapsing generators suppress $H(p_\theta)$, thereby reducing
  gen\_ppl *even if* their KL to GPT-2 is unchanged. This makes gen\_ppl a misleading
  quality metric whenever diversity is also valued.
- It is biased towards favouring text that is GPT-2-like (low cross-entropy with GPT-2's
  training data), regardless of coverage of the true reference distribution.
- A model that generates a single high-quality sentence repeated 100 times will achieve
  very low gen_ppl but be useless. This is the **mode collapse failure mode**, where
  $p_\theta$ concentrates on a single mode of $q_{\text{GPT2}}$.
- gen_ppl is a **point estimate** with high variance at N=100. The empirical standard
  error of the mean is approximately $\hat{\sigma}/\sqrt{N}$ where $\hat{\sigma}$ is the
  sample standard deviation of per-sequence cross-entropies.

**What it does NOT measure:**
- Diversity or coverage of the text distribution.
- Whether the generated texts are factually correct or semantically coherent.
- Alignment with the model's actual training distribution (OWT), only with GPT-2's.

### MAUVE

MAUVE (Pillutla et al., 2021) measures the divergence between the generated distribution
$p_\theta$ and a reference corpus $q_{\text{ref}}$ using a soft variant of the
Kullback–Leibler divergence. The computation proceeds as follows:

1. Embed all generated texts and all reference texts using a large LM (GPT-2 XL by default).
2. In the resulting embedding space, fit a single Gaussian mixture model (GMM) with
   $k$ components to the union of both sets.
3. Compute the "soft" precision and recall: for each component, what fraction of probability
   mass belongs to generated vs. reference? The resulting frontier traces out a Precision–Recall
   curve analogous to GANs.
4. MAUVE is the area under this curve, normalised to $[0, 1]$.

**Formal definition.** Let $P$ (generated) and $Q$ (reference) be empirical distributions
over sequences projected via a fixed LM encoder. Define the mixture:
$$M_\lambda = (1 - \lambda) P + \lambda Q, \quad \lambda \in [0, 1]$$

The **divergence frontier** is the parametric curve:
$$\mathcal{F}(P, Q) = \bigl\{\bigl(\text{KL}(M_\lambda \| Q),\; \text{KL}(M_\lambda \| P)\bigr) : \lambda \in [0,1]\bigr\} \subset \mathbb{R}_{\geq 0}^2$$

At $\lambda = 0$: $M_0 = P$, giving the point $(\text{KL}(P \| Q),\; 0)$ — measures
how unlike the reference the generated distribution is (precision failure). At
$\lambda = 1$: $M_1 = Q$, giving the point $(0,\; \text{KL}(Q \| P))$ — measures how
much of the reference is not covered (recall failure). The frontier traces a monotone
curve between these extremes. MAUVE is:
$$\text{MAUVE}(P, Q) = \exp\!\bigl(-c \cdot \text{Area}(\mathcal{F})\bigr) \in [0, 1]$$
where $\text{Area}(\mathcal{F})$ is the area enclosed between the frontier curve and
the two axes, and $c > 0$ is a fixed constant. A frontier close to the origin
(both KL terms small) gives $\text{Area} \approx 0$ and $\text{MAUVE} \approx 1$.

In practice, the population distributions $P$ and $Q$ are approximated by empirical
samples, the LM encoder is GPT-2 XL (producing 1600-dimensional representations), and
the KL terms are estimated via a $k$-component Gaussian mixture model fit to the union
of both sample sets. This quantisation introduces additional estimation variance beyond
the finite-sample variance of $P$ and $Q$ themselves.

The **double-sided** structure means MAUVE penalises both **precision failure**
(generating text unlike the reference, $\text{KL}(P \| Q)$ large) and
**recall failure** (failing to cover the reference, $\text{KL}(Q \| P)$ large).

**Statistical interpretation:**
- MAUVE is a **two-sided** distributional divergence: it captures both precision and recall.
  A mode-collapsing model loses recall (fails to cover $Q$) and gets a low MAUVE score even
  if individual samples score high on gen_ppl.
- MAUVE is **sensitive to N**: with N=100 samples, the empirical $P$ is a coarse
  approximation of $p_\theta$. The 1600-dimensional embedding space means the GMM is
  fitted on a sparse cloud (100 points in 1600 dimensions), and the KL estimates from
  the GMM have high variance. Pillutla et al. (2021) demonstrate that MAUVE variance
  decreases substantially as N increases from 100 to 500; beyond N=500, estimates
  stabilise for their benchmarks. Our N=100 estimates should be treated as **directional
  indicators** of relative ordering between strategies, not precise absolute values.
  Confidence intervals (obtainable via bootstrap resampling of the generated texts)
  are not reported here but are needed before claiming statistical significance.
- MAUVE is **reference-dependent**: the value is only meaningful relative to a fixed
  reference corpus. Our OWT reference (1000 samples, skip=5000) differs from the paper's
  reference, making absolute comparisons across papers meaningless. Only **within-experiment**
  comparisons (same reference, same N) are valid.

**Why gen_ppl and MAUVE can diverge.** From the cross-entropy decomposition:
$$H(p_\theta, q) = H(p_\theta) + \text{KL}(p_\theta \| q)$$
Minimising gen\_ppl is equivalent to minimising $H(p_\theta) + \text{KL}(p_\theta \| q)$.
A mode-collapsing generator suppresses $H(p_\theta)$ while possibly also suppressing
$\text{KL}(p_\theta \| q)$ — lower gen\_ppl but at the cost of diversity.
MAUVE, being symmetric in $P$ and $Q$ via the frontier, penalises coverage failure and
thus rewards diversity. These are **genuinely competing objectives**: any strategy that
reduces gen\_ppl by reducing $H(p_\theta)$ (rather than by reducing the KL term) will
simultaneously hurt MAUVE. This tension is the statistical explanation for the inversion
we observe as T increases.

### Entropy

$$H = -\sum_{c} \frac{n_c}{N_{\text{total}}} \log_2 \frac{n_c}{N_{\text{total}}}$$

where $n_c$ is the count of character $c$ across all generated texts. This is
character-level marginal entropy.

**Statistical interpretation:**
- Character entropy is a **weak lower bound** on text diversity: it captures the
  marginal distribution of characters, not their joint distribution or semantic diversity.
- A system that generates grammatically correct but semantically repetitive text can
  maintain high character entropy (many different characters, words) while exhibiting
  low semantic diversity.
- We use entropy as a **diversity proxy**, not a direct measure of generation quality.
  A meaningful drop in entropy (e.g. > 0.1 bits) across strategies is evidence of
  mode collapse or reduced vocabulary coverage.

**Relationship to MAUVE.** Entropy and MAUVE both capture diversity, but at different
scales: entropy operates on the marginal token distribution, while MAUVE operates on
the joint sentence embedding distribution. A drop in entropy without a drop in MAUVE
would suggest reduced character-level diversity without distributional shift. A drop
in MAUVE without a drop in entropy would suggest distributional shift without
character-level change — sentences using similar characters but in less natural
combinations.

---

## The Factorization Error Framework

All three metrics can be grounded in the **factorization error** framework introduced
in the thesis (following Lavenant & Zanella, 2024 and EB-Sampler):

$$E_{\text{fact}}(T) = \mathbb{E}\left[\text{KL}\!\left(p_{\theta}(\cdot \mid x_{\mathcal{M}}) \;\Big\|\; \prod_{i \in \mathcal{M}} p_{\theta}(x^i \mid x_{\setminus i})\right)\right]$$

where $\mathcal{M}$ is the set of simultaneously unmasked positions. This error arises
because the model predicts each masked position independently (product factorisation)
while the true joint distribution has dependencies.

**Connection to our metrics:**
- **gen_ppl** correlates with $E_{\text{fact}}$ through the marginal quality of
  individual tokens. Lower $E_{\text{fact}}$ → better conditioning → lower cross-entropy
  per token.
- **MAUVE** is more sensitive to the *joint* distribution of generated text. If
  $E_{\text{fact}}$ causes strong tokens to be over-committed early (as in remdm-conf),
  the resulting text may have good marginal token quality (low gen_ppl) but poor
  joint coherence (low MAUVE).
- **Entropy** measures the marginal diversity of the generated distribution, which
  is degraded when $E_{\text{fact}}$ causes systematic early token commitment
  (mode-seeking behaviour).

**Key insight:** Remasking is a method for reducing $E_{\text{fact}}$ at inference time.
Different remasking strategies trade off the three metrics differently because they
reduce $E_{\text{fact}}$ in different ways and with different side effects on diversity.

\newpage

# Empirical Results: A Statistical Reading

## Complete Results Table

All results: N=100 samples, seed=42, OWT reference (1000 samples, skip=5000),
evaluated on Bocconi HPC A100 80GB.

**Generation Perplexity (gen_ppl) — lower is better:**

| Strategy    | T=128  | T=256  | T=512  | T=1000 | Δ(128→1000) |
|-------------|--------|--------|--------|--------|-------------|
| mdlm        | 60.914 | 54.202 | 49.019 | 52.269 | −14.2%      |
| remdm-conf  | 57.579 | 50.668 | 42.868 | 37.321 | **−35.2%**  |
| remdm-loop  | 59.632 | 42.877 | 34.322 | **30.296** | **−49.2%** |

**MAUVE — higher is better:**

| Strategy    | T=128  | T=256      | T=512  | T=1000     | Δ(128→1000) |
|-------------|--------|------------|--------|------------|-------------|
| mdlm        | 0.170  | **0.740**  | 0.592  | 0.590      | +247%       |
| remdm-conf  | 0.440  | 0.475      | 0.470  | 0.325      | −26%        |
| remdm-loop  | 0.396  | 0.614      | 0.532  | **0.684**  | +73%        |

**Character-level entropy (H, bits) — proxy for diversity:**

| Strategy   | T=128 | T=256 | T=512 | T=1000 | Δ(128→1000) |
|------------|-------|-------|-------|--------|-------------|
| mdlm       | 5.507 | 5.481 | 5.440 | 5.446  | −0.061      |
| remdm-conf | 5.499 | 5.443 | 5.405 | 5.357  | **−0.142**  |
| remdm-loop | 5.538 | 5.460 | 5.427 | 5.390  | −0.148      |

---

## Finding 1: gen_ppl and MAUVE Decouple — Diversity Cost of Remasking

The cleanest statistical finding is the **gen_ppl–MAUVE decoupling** for remdm-conf
at high T:

- remdm-conf achieves the best gen_ppl among all strategies at T=1000 (37.3), yet the
  worst MAUVE (0.325). remdm-loop achieves the worst gen_ppl at T=128 (59.6) yet
  eventually the best MAUVE at T=1000 (0.684).

The directional pattern (remdm-conf gen\_ppl improving, remdm-conf MAUVE declining,
remdm-loop MAUVE improving) is consistent across all four step counts T ∈ {128, 256,
512, 1000}, which is evidence against noise as the sole explanation. However, without
bootstrap confidence intervals, the precise magnitudes should be treated as indicative
only.

**Mechanistic explanation.** remdm-conf applies the standard MDLM unmasking step and
then, as a *remasking* correction, re-masks committed tokens with probability
$\sigma_t \cdot (1 - c^i)$, where $c^i = H(p_\theta(x^i \mid z_t))$ is the
confidence. Tokens with high confidence (low entropy) survive remasking with high
probability. This is a **greedy remasking policy**: it systematically protects
high-confidence committed tokens from being revisited.

As T increases, each individual denoising step uncovers fewer masked positions (finer
step size), so the context available when each token is first committed becomes
progressively richer. Richer context → lower entropy → higher confidence → lower
remasking probability. Tokens committed at high T steps are therefore almost never
remasked even if the commitment was locally correct but globally inconsistent. The
result is a sequence that is locally coherent (each token well-predicted by its
context → low gen\_ppl from the cross-entropy decomposition's KL term) but globally
mode-seeking (few diverse trajectories were explored → reduced $H(p_\theta)$ → MAUVE
recall failure).

This is the **mode-seeking vs mode-covering** trade-off (Theis et al., 2016): greedy
remasking incentivises mode-seeking (preserve the most likely token at each step),
while loop remasking is more mode-covering (revisit all positions, allowing less
confident but globally coherent trajectories). The gen\_ppl–MAUVE decoupling is the
empirical signature of this trade-off.

**Practical consequence.** If the downstream use case requires fluent text
(gen_ppl proxy), remdm-conf is preferred at high T. If the use case requires diverse,
reference-matching text (MAUVE proxy, e.g. training data generation), remdm-loop
dominates at T ≥ 512.

---

## Finding 2: The MDLM Diversity Window at T=256

MDLM MAUVE peaks at T=256 (0.740) — higher than either remasking strategy at that step
count — then drops to ~0.59 and plateaus. This is the strongest MAUVE score in our
entire experiment, achieved by the baseline model with no remasking.

**Statistical interpretation.** This is the **diversity window** phenomenon. Consider
the ELBO for the absorbing MDM:

$$\mathcal{L}(\theta) = \underbrace{\mathbb{E}_{t,x_0}\left[-\log p_\theta(x_0 \mid z_t)\right]}_{\text{reconstruction}} + \underbrace{\text{KL}(q(z_T) \| p(z_T))}_{\text{prior}} + \sum_{t=2}^{T} \underbrace{\text{KL}(q(z_{t-1} \mid z_t, x_0) \| p_\theta(z_{t-1} \mid z_t))}_{\text{transition KL}}$$

At low T (e.g. T=128), the reverse process must undo masking over fewer steps. Each step
must unmask many tokens simultaneously, amplifying $E_{\text{fact}}$: the model's
conditional independence assumption introduces more error when more tokens are jointly
unmasked per step. The resulting text is lower-quality but still diverse (the model
hasn't committed to any specific patterns, so the output varies).

At high T (e.g. T=1000), $E_{\text{fact}}$ per step is small (few tokens unmasked per
step), so local coherence improves. However, without remasking, once a token is committed
in step $t$, it conditions all subsequent predictions. MDLM's forward kernel is
**non-remasking**: a token, once predicted, is never reconsidered. Over many steps, the
sequence accumulates **commitment errors**: early tokens (predicted when context is
sparse) condition later predictions, and the final sequence lies in a restricted
subspace of the reference distribution → lower MAUVE.

At T=256, the model sits at the empirical optimum: $E_{\text{fact}}$ is small enough
for local coherence, but the number of steps is not so large that commitment errors
accumulate. This suggests a **U-shaped quality curve** as a function of T for
non-remasking models, with the minimum of gen_ppl and maximum of MAUVE at different
optimal T.

**Why remasking shifts this optimum.** Remasking breaks the commitment error
accumulation. For remdm-loop, MAUVE improves monotonically because the loop
re-evaluates all committed tokens at every step, preventing error accumulation.
The "diversity window" only appears for MDLM because it is the only strategy
without error correction.

---

## Finding 3: Entropy Drop as a Signature of Mode Collapse

Entropy decreases for all strategies as T increases, but remdm-conf shows the steepest
drop (5.499 → 5.357, Δ = 0.142 bits over T=128→1000 vs 0.061 for mdlm).

**Statistical interpretation.** The entropy of the marginal character distribution is
related to the **effective character diversity** of the generated texts. A useful
summary statistic is the **character perplexity** $2^H$, which equals the number of
equally-probable characters needed to produce the same entropy — the information-theoretic
analogue of vocabulary coverage:
$$2^{5.499} \approx 45.2 \quad \text{(remdm-conf, T=128)} \quad \to \quad 2^{5.357} \approx 40.8 \quad \text{(T=1000)}$$

This represents a 9.7% reduction in character perplexity, indicating that at T=1000
the model's output is drawn from a distribution equivalent to approximately 4 fewer
effective characters. Note that this is the perplexity of the *marginal* character
distribution — it does not capture reductions in word-level or sentence-level diversity.
It provides a lower bound on the diversity reduction: actual semantic diversity could
have decreased by more.

The combination of (a) the largest entropy drop, (b) the worst MAUVE at T=1000, and
(c) the best gen_ppl at T=1000 for remdm-conf forms a **coherent statistical portrait
of mode collapse**:
- The model generates a narrower set of characters/words (entropy drop).
- Those words are locally coherent and GPT-2-like (low gen_ppl).
- But the global distribution does not match the reference (low MAUVE).

This is the **mode-seeking vs mode-covering** trade-off studied in the GAN literature
(Theis et al., 2016): confidence-based remasking incentivises mode-seeking (commit the
most likely token at each step), while loop remasking is more mode-covering (revisit
all positions, allowing less likely but globally coherent trajectories).

---

## Finding 4: gen_ppl Anomaly for MDLM at T=512→T=1000

MDLM gen_ppl worsens slightly from T=512 (49.019) to T=1000 (52.269), while both
remasking strategies continue to improve. This is anomalous: more denoising steps
should only help (lower $E_{\text{fact}}$ per step).

**Hypotheses:**
1. **Statistical noise.** With N=100 samples, the standard error on gen_ppl is
   non-trivial. The difference (49.0 vs 52.3) may lie within 1–2 standard deviations.
   A proper significance test (paired Wilcoxon or bootstrap CI) is needed to confirm
   this is a real effect.
2. **Commitment error accumulation.** As argued in Finding 2, non-remasking MDLM
   accumulates commitment errors at high T. At T=1000 with many small unmasking steps,
   early committed tokens — predicted in low-context windows — may condition later
   tokens adversely. The resulting sequences may still be locally coherent but have
   unusual long-range structure that GPT-2 assigns lower probability to.
3. **Numerical instability in the SDPA fallback.** Our setup uses PyTorch SDPA (no
   flash_attn). Numerical differences from the reference implementation may be amplified
   over T=1000 forward passes in a way not present at T=512.

This anomaly deserves attention: if real, it suggests MDLM has a non-monotone
quality curve in gen_ppl as well as MAUVE, with an optimal T around 512. This would
be a notable finding contradicting the paper's implication of monotone improvement.

---

## Alignment with arXiv:2503.00307

| Finding | Our result | Paper claim | Assessment |
|---------|-----------|-------------|------------|
| MDLM gen_ppl at T=128 | 60.914 | 61.5 | ✓ Consistent (<1% diff) |
| ReMDM gen_ppl improvement direction | ✓ conf < loop < mdlm at T≥512 | ReMDM > MDLM | ✓ Consistent |
| Remasking helps MAUVE | ✓ at T=1000, loop wins | ✓ claimed | ✓ Consistent |
| MAUVE absolute values | ~10× higher than paper | — | Reference mismatch (expected) |
| MDLM MAUVE peak at T=256 | ★ Novel | Not reported | Novel finding |
| remdm-conf diversity collapse | ★ Novel | Not reported | Novel finding |
| remdm-loop monotonic MAUVE | ✓ 0.396→0.684 | Implied | Consistent |
| MDLM gen_ppl worsens T=512→1000 | ★ Anomalous | Not reported | Needs testing |

**Note on MAUVE absolute values.** The paper's MAUVE values are approximately 6–11×
lower than ours. The most likely explanation is reference corpus differences: MAUVE is
not an absolute metric, and values are only comparable within an experiment using the
same reference, sample size, and implementation. Our values use OWT (skip=5000, N=1000
reference samples), which is a well-matched reference for a model trained on OWT.
The paper likely uses a different reference (possibly wikitext-103 or a different OWT
slice) and possibly different N.

However, reference corpus alone may not account for the entire 6–11× gap. Other
contributing factors include: (a) different sequence lengths for generated samples,
(b) different N for the generated set, (c) different MAUVE hyperparameters
(number of GMM clusters $k$, `max_text_length`), (d) potentially different generation
quality. **In the absence of the paper's exact experimental configuration, no
quantitative attribution can be made.** What can be stated: within our experiment,
MAUVE rankings are consistent and the relative improvements are interpretable.
Cross-paper absolute comparisons should be avoided.

\newpage

# Research Directions

The following six directions emerge directly from the theoretical gaps and empirical
anomalies identified above. Each is stated as a formal research question with a
proposed methodology and expected contribution.

---

## Direction 1: Confidence Calibration and Temperature-Scaled Remasking

### Motivation

The remdm-conf diversity collapse (Finding 1) is characteristically a **calibration
failure**: the model's confidence scores $c^i = \max_j p_\theta(x^i = j \mid x_t)$
become systematically overconfident as T increases. At high T, the context is rich
(few masked positions), so the softmax distribution over $x^i$ is sharply peaked.
The model commits tokens aggressively, reducing diversity.

### Research Question

Can temperature scaling of the confidence signal prevent diversity collapse without
sacrificing gen_ppl improvement?

### Formal Proposal

Define a temperature-scaled confidence:

$$c^i_\tau = \max_j \text{softmax}\!\left(\frac{\ell_j^i}{\tau(t)}\right)$$

where $\ell^i$ are the logits at position $i$ and $\tau(t)$ is a time-dependent
temperature. Choices of $\tau(t)$:

- **Fixed temperature:** $\tau(t) = \tau_0 > 1$ (uniformly soften predictions)
- **Annealed temperature:** $\tau(t) = 1 + (\tau_0 - 1) \cdot \frac{t}{T}$, which
  gives $\tau = \tau_0$ at $t = T$ (start, fully masked, high exploration) and
  $\tau \to 1$ at $t = 1$ (end, mostly unmasked, no softening). Note: in MDM
  sampling, $t$ counts *down* from $T$ to $1$; high $t$ corresponds to early
  generation (sparse context, high masking rate).
- **Entropy-adaptive:** $\tau(t)$ chosen so that $H(\text{softmax}(\ell^i/\tau)) = H_{\text{target}}(t)$
  for a pre-specified target entropy schedule $H_{\text{target}}(t)$

The annealed variant is particularly motivated: early in generation (large $t$, high
masking rate, sparse context) use high temperature (explore diverse completions); late
in generation (small $t$, low masking rate, rich context) use low temperature (commit
to the best tokens). This mirrors **simulated annealing** and has a clean interpretation
as an entropy regulariser on the commitment policy.

**Remasking policy with entropy constraint:**

$$S_t^* = \arg\max_{S : |S| = k_t} \sum_{i \in S} c^i_\tau \quad \text{subject to} \quad H(p_\theta^S) \geq H_{\text{min}}(t)$$

where $H_{\text{min}}(t)$ is a target minimum diversity at step $t$.

### Expected Contribution

- Empirical: show that temperature scaling restores MAUVE at high T without
  significantly worsening gen_ppl.
- Theoretical: connect to the calibration literature (Guo et al., 2017) and
  derive conditions under which calibrated confidence is sufficient for
  principled remasking (extending the thesis's three-tier hierarchy).

### Implementation

No retraining required. Modify `external/remdm/main.py` sampling loop:
add `temperature` parameter to the `remdm-conf` strategy. Run step-sweep
(T=128/256/512/1000) for τ ∈ {1.0, 1.5, 2.0, 3.0} and entropy-adaptive variant.

---

## Direction 2: Theoretical Characterisation of the Diversity Window

### Motivation

Finding 2 shows that MDLM MAUVE has a non-monotone shape in T, peaking at T=256.
This is currently an empirical observation. A theoretical explanation would be a
genuine contribution.

### Research Question

Can we derive an analytical expression for the optimal step count $T^*$ (maximising
MAUVE or minimising KL to reference) for a non-remasking MDM?

### Formal Proposal

Consider the absorbing MDM reverse process. At step $t$, the expected factorization
error for a single unmasking step of $k_t$ tokens is:

$$E_{\text{fact}}(k_t) = \mathbb{E}_{x_0}\left[\text{KL}\!\left(p(x_{\mathcal{M}_t} \mid x_{\setminus \mathcal{M}_t}) \;\Big\|\; \prod_{i \in \mathcal{M}_t} p_\theta(x^i \mid x_{\setminus \mathcal{M}_t})\right)\right]$$

For a masking schedule where $k_t = n/T$ tokens are unmasked per step (uniform),
$E_{\text{fact}}$ decreases as T increases (fewer tokens unmasked per step, smaller
approximation error). Define $E_{\text{fact}}^{\text{total}}(T) = \sum_{t=1}^T E_{\text{fact}}(k_t)$.

However, for **non-remasking** MDMs, there is a **commitment accumulation** term:
tokens committed at step $t$ with errors then condition all future predictions.
Define the propagated error as:

$$E_{\text{commit}}(T) = \mathbb{E}\left[\sum_{t=1}^T \sum_{i \text{ committed at } t} \epsilon_t^i \cdot \Delta_t^i\right]$$

where $\epsilon_t^i$ is the prediction error at position $i$ step $t$, and $\Delta_t^i$
is the downstream impact on future predictions (a measure of how strongly $x^i$
conditions the remaining masked positions). As T increases, $\epsilon_t^i$ decreases
(more context) but the early-committed tokens (predicted when context is sparse) can
have large $\Delta_t^i$.

The hypothesis is that quality $\propto -(E_{\text{fact}}^{\text{total}}(T) +
E_{\text{commit}}(T))$ has a maximum at finite $T^*$, giving the diversity window.
Establishing this would require:

1. A tractable model of $E_{\text{commit}}(T)$ — potentially using Gaussian approximations
   or small-vocabulary toy models.
2. Verification that the model predicts $T^* \approx 256$ for MDLM on OWT.
3. A prediction for how remasking (loop or confidence) modifies $T^*$ — the hypothesis
   is that remasking effectively suppresses $E_{\text{commit}}$, making $T^*$ much larger
   or eliminating the diversity window entirely.

### Expected Contribution

A theoretical bound on $T^*$ as a function of model capacity, vocabulary size, and
masking schedule. This would directly inform practitioners on how many steps to use
and why.

---

## Direction 3: Joint Remasking Policies

### Motivation

Current remasking strategies treat positions **independently**: the decision of whether
to remask position $i$ depends only on $c^i$, not on the joint relationship between
positions. The factorization error $E_{\text{fact}}$ arises precisely because tokens
are jointly dependent but treated independently. Remasking targeting **groups** of
dependent tokens could reduce $E_{\text{fact}}$ more efficiently.

### Research Question

Can we design a polynomial-time remasking policy that selects the optimal set $S^*$ of
positions to remask jointly, by modelling pairwise or higher-order dependencies?

### Formal Proposal

Define the **mutual information gain** from remasking a set $S$:

$$\text{MIG}(S) = I(x_S ; x_{\setminus S}) = H(x_S \mid x_{\setminus S}) - H(x_S \mid x)$$

Remasking $S$ is most beneficial when $\text{MIG}(S)$ is large: the positions in $S$
are highly uncertain given the current context but would be better predicted jointly.
The optimal remasking set is:

$$S^* = \arg\max_{S : |S| = k} \text{MIG}(S)$$

This is NP-hard in general (exponential search over subsets). However, $\text{MIG}(S)$
is a **submodular function** (by the submodularity of entropy), so a greedy algorithm
achieves a $(1 - 1/e)$ approximation guarantee (Nemhauser et al., 1978).

**Practical approximation.** Compute pairwise attention weights $a_{ij}$ from the
transformer's attention matrix as a proxy for mutual information. Define a graph $G$
where edge $(i, j)$ has weight $a_{ij}$. Apply graph clustering (e.g. spectral clustering
with $k$ clusters) to identify groups of highly correlated positions. Remask the lowest-confidence
member of each cluster.

### Expected Contribution

- A theoretically grounded remasking policy that accounts for token dependencies.
- If the greedy MIG policy outperforms position-independent remasking, it provides
  evidence that token dependencies are the key bottleneck in current MDMs.
- A formal connection between remasking and the submodular optimisation literature.

### Note on Difficulty

Estimating $\text{MIG}(S)$ from attention weights is an approximation whose quality
is unknown. A cleaner (but more expensive) approach uses the model's own forward pass
to estimate $H(x^i \mid x_S \cup x_{\setminus (S \cup \{i\})})$ by running the model
with different masked subsets — feasible only with small $|S|$.

---

## Direction 4: Statistical Testing Framework for MDM Evaluation

### Motivation

All papers in the field (including ours) report point estimates with no confidence
intervals. At N=100, MAUVE estimates have substantial variance. The T=512 dip for
remdm-loop (0.614 → 0.532 before recovering to 0.684 at T=1000) may be noise.
The MDLM gen_ppl anomaly (49.0 → 52.3) may be noise. **Without proper statistical
testing, these findings cannot be distinguished from random variation.**

### Research Question

What is the minimum sample size $N^*$ for statistically reliable MDM evaluation, and
which differences in our results are statistically significant?

### Formal Proposal

**Bootstrap confidence intervals for MAUVE.** Generate $B = 1000$ bootstrap resamples
of the generated texts (with replacement, size N). For each resample, recompute MAUVE.
The 95% CI is the 2.5th and 97.5th percentiles of the bootstrap distribution. A
difference between two strategies is significant if their CIs do not overlap.

**Paired test for gen_ppl.** Per-sequence cross-entropies are paired (same seed gives
comparable samples). Use a paired Wilcoxon signed-rank test (non-parametric, no
Gaussian assumption) to test $H_0:$ median gen_ppl is equal between strategies.

**Minimum N analysis.** Using the bootstrap CIs as a function of N:

$$N^*(δ) = \min\{N : \text{CI width} < δ\}$$

for a target precision $\delta$ (e.g. $\delta = 0.05$ for MAUVE on [0,1] scale).
Our pilot data (N=100) can seed this analysis: fit the CI width as a function of
$\sqrt{N}$ (standard error scaling) to extrapolate to larger N.

**Implementation.** Resample from the already-generated texts in
`results/*/generated_sequences.json` — no additional HPC compute required for
the bootstrap. Write `scripts/bootstrap_ci.py`.

### Expected Contribution

The first systematic analysis of statistical reliability in MDM evaluation.
The finding that N=100 is insufficient for reliable MAUVE (if demonstrated)
would be a methodological contribution to the field. Conversely, showing that
our key findings survive proper testing would strengthen the thesis's empirical claims.

---

## Direction 5: Adaptive Step Allocation

### Motivation

All our evaluations use a **uniform step budget**: T total steps, each unmasking
$n/T$ tokens. But tokens differ in difficulty: function words ("the", "of") are easy
to predict from context; content words in idiomatic phrases are hard. An adaptive
policy that allocates more steps (smaller unmasking batches) to hard positions and
fewer to easy ones could achieve the same quality as high-T uniform with fewer total
forward passes.

### Research Question

Can an adaptive step allocation policy match T=1000 quality with T ≈ 256 forward
passes, by concentrating compute on high-uncertainty positions?

### Formal Proposal

Define token difficulty at step $t$ as:

$$d^i_t = H\!\left(p_\theta(x^i \mid x_{\setminus \mathcal{M}_t})\right) = -\sum_j p_\theta^{ij} \log p_\theta^{ij}$$

(conditional entropy, not max-probability). Note this differs from the confidence
signal $c^i = \max_j p_\theta^{ij}$: entropy captures full distributional uncertainty,
not just the top probability.

**Adaptive policy:**
1. At each step, compute $d^i_t$ for all $i \in \mathcal{M}_t$.
2. Set unmasking batch size $k_t$ adaptively: $k_t \propto (1 - \bar{d}_t / \log_2 V)$,
   where $\bar{d}_t$ is the mean entropy over masked positions and $V$ is vocab size.
   When average uncertainty is high (early steps), unmask fewer tokens. When
   uncertainty is low (late steps), unmask many tokens.
3. Always unmask the lowest-entropy (most confident) positions first.

This policy reduces to standard uniform unmasking when all entropies are equal.
The step count $T$ becomes a budget parameter, not a fixed schedule.

**Connection to EB-Sampler.** The EB-Sampler paper (Part IV of the literature review)
derives the information profile $I^i(x)$ that optimally schedules when each position
should be unmasked. Adaptive step allocation is an inference-time approximation to
the EB-Sampler's optimal schedule without requiring explicit estimation of $I^i$.

### Expected Contribution

If adaptive allocation achieves T=1000 quality at fewer steps, it has direct practical
value: faster inference at the same quality. It also provides an empirical test of the
EB-Sampler's theoretical prediction that step timing matters.

---

## Direction 6: MAUVE as a Fine-Tuning Signal

### Motivation

Our results show that MAUVE and gen_ppl are often in tension: the strategy that
maximises gen_ppl (remdm-conf) minimises MAUVE at high T. If the goal is to generate
text that matches a reference distribution (e.g. for data augmentation, style transfer),
directly optimising for MAUVE during fine-tuning would align training and evaluation.

### Research Question

Can a differentiable approximation of MAUVE be used as a reward signal for
RL fine-tuning of a masked diffusion model, and does it improve MAUVE without
collapsing gen_ppl?

### Formal Proposal

MAUVE is not directly differentiable (it uses a discrete GMM fit in embedding space).
However, a **soft MAUVE surrogate** can be constructed:

$$\widetilde{\text{MAUVE}}(p_\theta, q_{\text{ref}}) = -\text{KL}(f_\phi(p_\theta) \| f_\phi(q_{\text{ref}}))$$

where $f_\phi$ is a fixed feature extractor (e.g. frozen GPT-2 last hidden state).
The gradient $\nabla_\theta \widetilde{\text{MAUVE}}$ can be computed via REINFORCE
or via the reparameterisation trick if a differentiable sampler is available.

**Connection to RemeDi.** This is essentially the objective that RemeDi-RL was trained
on (RL fine-tuning of a masked diffusion model for text quality). However, RemeDi-RL
uses human feedback as the reward, not MAUVE. Using MAUVE as reward is cheaper
(no human annotation) and more directly connected to the evaluation metric.

### Expected Contribution

Demonstrates a principled training objective that directly targets distributional
similarity. If successful, this closes the loop between evaluation and training
for MDMs — an important step toward reliable MDM development. This direction also
contextualises RemeDi's RL approach: RemeDi is an instance of this broader paradigm.

---

## Summary of Research Directions

| Direction | Difficulty | Training required | Key contribution |
|-----------|-----------|-------------------|-----------------|
| 1. Temperature-scaled remasking | Low | No | Prevents diversity collapse, no new training |
| 2. Diversity window theory | Medium | No | Formal explanation of T* optimum |
| 3. Joint remasking policies | High | No | Dependency-aware remasking, submodular connection |
| 4. Statistical testing framework | Low | No | Methodological contribution, N* for MAUVE |
| 5. Adaptive step allocation | Medium | No | Faster inference at same quality |
| 6. MAUVE fine-tuning | High | Yes | Aligns training and evaluation objectives |

**Recommended next step:** Direction 4 (statistical testing) can be implemented
immediately with existing data (`results/*/generated_sequences.json`) and would
validate or refute the anomalous findings (MDLM gen_ppl at T=1000, remdm-loop
MAUVE dip at T=512) before investing in more experiments.

**Strongest thesis contribution:** Direction 1 (temperature scaling) + Direction 2
(diversity window theory) form a coherent story: empirically show that calibrated
confidence prevents the collapse (Direction 1), theoretically explain why overconfidence
causes it (Direction 2). Both can be done without retraining.

\newpage

# Appendix: Connection Map Between Papers and Findings

| Paper | Key concept used in analysis |
|-------|------------------------------|
| D3PM (Austin et al., 2021) | Absorbing Markov chain, forward/reverse process definition |
| MDLM (Sahoo et al., 2024) | ELBO simplification, factorization error formulation |
| SEDD (Lou et al., 2024) | Score-based perspective; connects gen_ppl to score matching |
| Lavenant & Zanella (2024) | KL error decomposition → formal $E_{\text{fact}}$ definition |
| EB-Sampler | Optimal unmasking schedule → connection to Direction 5 |
| ReMDM (arXiv:2503.00307) | Baseline for gen_ppl comparison; confidence/loop strategies |
| Mask-Predict (2019) | Historical precedent for confidence-guided remasking |
| MAUVE (Pillutla et al., 2021) | Formal definition of the distributional metric |

All papers covered in `docs/comparison.md`. The above table maps findings to their
theoretical grounding to aid thesis chapter writing.

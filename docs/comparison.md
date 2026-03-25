---
title: "Principled Remasking in Masked Diffusion Language Models: A Comparative Literature Review"
author: "MSc Thesis — Bocconi University, supervised by Prof. Giacomo Zanella"
date: "March 2026"
geometry: "top=2.5cm, bottom=2.5cm, left=3cm, right=3cm"
fontsize: 11pt
linkcolor: blue
numbersections: true
toc-depth: 2
---

\newpage

# Introduction

This document is a self-contained literature review for the MSc thesis
*"Principled Remasking in Masked Diffusion Language Models"* (Bocconi
University, 2026). It covers twelve papers organised into five parts that
follow the intellectual development of the field from its earliest
precursors to the state of the art.

**The central question of the thesis.** Masked Diffusion Models (MDMs)
generate text by iteratively unmasking tokens from a fully masked sequence.
When multiple tokens are unmasked simultaneously — necessary for efficiency
— the model treats them as conditionally independent, introducing a
*factorization error* $E_{\text{fact}}$. Remasking (re-masking a committed
token and re-predicting it in richer context) is a natural corrective, but
no prior work has characterised it within a rigorous error-bound framework.
The thesis asks:

> *Can we design principled, theoretically grounded remasking strategies
> that reduce $E_{\text{fact}}$ without retraining the model, and derive
> conditions under which training-free confidence signals are sufficient?*

**How to read this document.** The five parts are ordered to build
understanding progressively:

- **Part I** situates the thesis in the pre-diffusion literature (Mask-Predict,
  2019), showing that iterative confidence-guided remasking has been
  empirically effective for years — but has never had a rigorous foundation.
- **Part II** builds the discrete diffusion framework from scratch: D3PM
  (foundational theory), SEDD (score-based alternative with PC samplers),
  MDLM (practical simplification), and LLaDA (8B-scale validation).
- **Part III** introduces the Discrete Flow Matching perspective, which
  provides a clean continuous-time language for corrector steps.
- **Part IV** covers the two papers most directly foundational to the thesis:
  Lavenant & Zanella (KL error decomposition) and the EB-Sampler (optimal
  unmasking schedule). The thesis extends these to the remasking direction.
- **Part V** surveys all existing remasking approaches: ReMDM, Informed
  Correctors, RemeDi, and PRISM — from the most theoretical to the most
  empirically powerful.
- **Part VI** is a structured cross-paper comparison on eight dimensions.

Each paper is reviewed under six headings: (A) Summary, (B) Method Details,
(C) Theoretical Contributions, (D) Confidence Signal, (E) Limitations,
(F) Relation to This Thesis.

\newpage

# How to Use This Document

## Reading Roadmap

This document covers twelve papers in five thematic parts. The intellectual dependency
graph is not linear — read in the order below to build concepts progressively:

```
STEP 1 — Foundation (what problem are we solving?):
  MDLM (Part II) → understand the forward/reverse process and the training loss
  D3PM  (Part II) → understand where the independence assumption comes from

STEP 2 — The core error (what exactly goes wrong?):
  Lavenant & Zanella (Part IV) → definition of E_fact; the information profile I(x)
  EB-Sampler (Part IV)         → how to minimise E_fact for unmasking

STEP 3 — Remasking approaches (how do we fix committed errors?):
  ReMDM (Part V)               → principled remasking posterior σ_t
  Informed Correctors (Part V) → MCMC theory; why confidence-guided > uniform
  PRISM (Part V)               → learned confidence with formal guarantee
  RemeDi (Part V)              → RL-finetuned policy (upper bound)

STEP 4 — Theoretical unification:
  DFM (Part III)               → correctors as Gibbs samplers; velocity field
  SEDD (Part II)               → score-based view; PC corrector

STEP 5 — Scale and history:
  LLaDA (Part II)              → MDMs at 8B scale; why uniform schedule is wasteful
  Mask-Predict (Part I)        → heuristic ancestor; what the thesis formalises
```

The central question to keep in mind throughout:
> *Each paper that does remasking has a different answer to: "which tokens should be
> remasked, and why?" Does the choice have a formal justification?*

## Six Questions That Thread Through the Document

These questions connect all twelve papers. As you read each section, ask:

1. **What is $E_{\text{fact}}$ for this algorithm?** Some papers reduce it explicitly
   (EB-Sampler, thesis); others reduce it implicitly (ReMDM, Informed Correctors); others
   ignore it entirely (D3PM, LLaDA).

2. **What confidence signal does this paper use?** Map it to Tier 1/2/3. Tier 1 requires
   $O(L)$ forward passes; Tier 2 requires training; Tier 3 is free from logits.

3. **Does this paper require training, retraining, or nothing?** The training axis is the
   primary practical dimension that distinguishes the approaches.

4. **What is the paper's relation to $I(x)$?** The information profile is the theoretical
   gold standard; papers that ignore it leave $E_{\text{fact}}$ on the table.

5. **What does the paper's confidence signal converge to as $p_\theta \to q$?** At the
   optimal model, all Tier 3 signals should equal $I^i(x)$. Do they?

6. **What is the compute cost per generated token?** Expressed as number of forward
   evaluations (NFE) and how it scales with sequence length and correction count.

---

## Unified Notation Reference

All twelve papers use overlapping but inconsistent notation. This table fixes a
single notation used consistently throughout this document.

| Symbol | Meaning | Defined in |
|--------|---------|------------|
| $L$ | Sequence length | — |
| $\mathcal{V}$ | Token vocabulary (size $V$) | — |
| $x = (x^1,\ldots,x^L) \in \mathcal{V}^L$ | Clean token sequence | D3PM |
| $m \notin \mathcal{V}$ | Mask token | D3PM |
| $z_t \in (\mathcal{V} \cup \{m\})^L$ | Noisy sequence at noise level $t$ | D3PM |
| $\mathcal{M}_t = \{i : z_t^i = m\}$ | Set of masked positions at time $t$ | MDLM |
| $T$ | Total number of denoising steps | D3PM |
| $\alpha_t \in [0,1]$ | Survival probability: $P(z_t^i = x^i)$; $\alpha_T \approx 0$, $\alpha_0 = 1$ | MDLM |
| $\beta_t = 1 - \alpha_t/\alpha_{t-1}$ | Per-step masking rate | D3PM |
| $q(z_t \mid x) = \prod_i \mathrm{Cat}(z_t^i;\, \alpha_t e_{x^i} + (1-\alpha_t)e_m)$ | Forward kernel (absorbing) | D3PM / MDLM |
| $p_\theta(x^i \mid z_t)$ | Model's predicted clean-token distribution at position $i$ | MDLM |
| $p_\theta(z_{t-1}^i \mid z_t)$ | Reverse kernel for position $i$ | MDLM |
| $\pi$ | True data distribution | Lav. & Zan. |
| $p_{\text{alg}}$ | Distribution of algorithm's output sequences | Lav. & Zan. |
| $I^i(x) = H(x^i \mid x^{\setminus i})$ | True conditional entropy of position $i$ (info. profile) | Lav. & Zan. |
| $I(x) = \sum_i I^i(x)$ | Total sequence information | Lav. & Zan. |
| $\hat{I}^i(z_t) = H(p_\theta(x^i \mid z_t))$ | Model's entropy estimate at position $i$ | EB-Sampler |
| $\Sigma^2 = \mathrm{Var}_i[I^i(x)]$ | Variance of the information profile | Lav. & Zan. |
| $E_{\text{fact}}$ | Factorization error: KL due to independence assumption | Lav. & Zan. |
| $E_{\text{learn}}$ | Learning error: KL due to imperfect $p_\theta$ | Lav. & Zan. |
| $\varepsilon$ | Entropy budget per step (EB-Sampler) | EB-Sampler |
| $\sigma_t^i = 1 - \alpha_t$ | ReMDM remasking probability for committed token at time $t$ | ReMDM |
| $R$ | Number of corrector passes per denoising step | ReMDM |
| $\tau$ | Remasking threshold (remask if $\hat{I}^i > \tau$) | Thesis |
| $c^i_{\text{mp}} = \max_j p_\theta^{ij}$ | Max-probability confidence (Tier 3) | Mask-Predict |
| $c^i_H = -H(p_\theta(x^i \mid z_t))$ | Entropy confidence (Tier 3; $= -\hat{I}^i$) | ReMDM / EB-Sampler |
| $c^i_{\text{mg}} = p_\theta^{(1)} - p_\theta^{(2)}$ | Margin confidence (Tier 3) | — |
| $g_\phi^i \in [0,1]$ | PRISM quality head output (Tier 2) | PRISM |
| $\psi^i \in [0,1]$ | RemeDi UPS output (Tier 2) | RemeDi |
| $s_\theta(z_t)_{ij} \approx p_t(z_t^{(j)})/p_t(z_t)$ | SEDD concrete score at position $i$ | SEDD |
| $u_t(z^i = m \to j)$ | DFM velocity field for position $i$ | DFM |
| $\lambda$ | Spectral gap of Gibbs corrector chain | Inf. Corr. |
| $\mathcal{L}_{\text{MDM}}$ | MDLM training objective (weighted masked cross-entropy) | MDLM |

**Key relations to memorise:**
$$\hat{I}^i(z_t) \xrightarrow{p_\theta \to q} I^i(x) \quad \text{(at optimality, entropy estimate = true info)}$$
$$\sigma_t^i = 1 - \alpha_t \quad \text{(ReMDM: remask each committed token with this prob.)}$$
$$E_{\text{fact}} \leq C \cdot \frac{\Sigma^2}{T} \quad \text{(fewer steps or higher profile variance → larger error)}$$
$$E_{\text{fact}}(\text{EB}) \leq \varepsilon \cdot \max_i \hat{I}^i \quad \text{(EB-Sampler bound)}$$

\newpage

# Part I — Before Diffusion: Non-Autoregressive Iterative Decoding

---

## Mask-Predict (2019) — The Heuristic Ancestor

**Full citation:** Ghazvininejad, Levy, Liu & Zettlemoyer (2019),
"Mask-Predict: Parallel Decoding of Conditional Masked Language Models."
*EMNLP 2019.* arXiv:1904.09324.

### Summary

Mask-Predict is the earliest paper to demonstrate confidence-guided
iterative remasking for language generation, in the context of
non-autoregressive (NAR) machine translation. Prior to this work, NAR
models generated all target tokens in parallel in a single pass, achieving
high speed at the cost of quality (due to multi-modality in the output
distribution). Mask-Predict proposes a simple iterative refinement:
starting from a fully masked target, alternate between predicting all masked
tokens and re-masking the lowest-confidence predictions. This converges in
$O(T)$ parallel steps — $4$–$9\times$ faster than autoregressive decoding —
with BLEU scores competitive with strong autoregressive baselines on WMT
EN$\leftrightarrow$DE and EN$\leftrightarrow$ZH.

The paper's contribution is purely algorithmic. It does not provide a
probabilistic model of the forward/reverse process, a noise schedule,
or any theoretical justification for why remasking should help. It is
effective because of the simple intuition that a token predicted in a
context where all other tokens are unmasked is more likely to be correct
than one predicted when most of the sequence is still masked. This
intuition is exactly what the thesis formalises: the information profile
$I(x)$ captures how much each position benefits from additional context,
and remasking targets the positions with the most to gain.

### Method Details

The model is a BERT-style bidirectional transformer trained with masked
language model objectives on parallel translation data. Given a source
sentence $x$ and a target length $n$, Mask-Predict initialises
$y = [\text{M}]^n$ and iterates:

**Step 1 — Predict.** Run the masked LM conditioned on $x$ and the
current $y$:
$$p^i = P(y^i \mid x,\; y^{\setminus i}) \quad \forall i : y^i = [\text{M}]$$
For unmasked positions, the probability of the current token is used
directly.

**Step 2 — Re-mask.** Compute the confidence of every position as
$c^i = \max_j p^i_j$ (max-probability). Re-mask the $n_t$ lowest-confidence
positions:
$$n_t = \left\lceil n \cdot \frac{T - t}{T} \right\rceil$$
so that $n_t$ decreases linearly from $n$ to $0$ over $T$ iterations.

**Full algorithm:**
```
y = [M, M, ..., M]
for t = 1, ..., T:
    # Predict all masked positions in parallel
    for i with y^i = [M]:
        p^i = P(y^i | x, y\{y^i=[M]})
    # Commit all positions
    y^i ← argmax p^i   for all i
    # Remask n_t lowest-confidence committed tokens
    n_t = ceil(n * (T - t) / T)
    remask_set ← argsort(c)[:n_t]   # lowest confidence first
    y^i ← [M]   for i in remask_set
return y
```

The confidence signal is thus $c^i = \max_j p_\theta(y^i = j \mid x, y^{\setminus i})$,
the simplest possible measure of how certain the model is about position $i$.

**Masking schedule.** The number of masked tokens decreases linearly:
$n_t / n = (T-t)/T$. This is the *linear schedule*, one of the
schedules implemented in the thesis's `schedule` strategy.

### Theoretical Contributions

None. Mask-Predict is purely empirical. The authors justify the approach
informally: low-confidence tokens are likely wrong, and re-masking them
allows the model to predict them in a richer context (once more tokens
are committed). There is no probabilistic model, no noise schedule, no
ELBO, and no convergence guarantee.

### Confidence Signal

$c^i = \max_j p_\theta(y^i = j \mid x, y^{\setminus i})$ — the
*max-probability* of the top-1 prediction at position $i$.

**Properties:**
- Cheap: computed from the standard forward-pass logits at no extra cost.
- Heuristic: no formal guarantee that low $c^i$ implies high prediction
  error.
- Miscalibrated across noise levels: the same value of $c^i$ means
  different things depending on how many other tokens are masked.

This is the Tier 3 (heuristic) confidence signal in the thesis's
three-tier hierarchy. The thesis asks formally: under what conditions
is $c^i$ a consistent estimator of $H(x^i \mid \text{context})$?

### Limitations

1. **No probabilistic framework.** Mask-Predict is not a generative
   model; it is a deterministic iterative decoding algorithm applied
   to a trained masked LM. There is no forward process, no reverse
   process, and no connection to diffusion.
2. **Conditional generation only.** The model requires a source sentence
   $x$ (machine translation); it is not directly applicable to
   unconditional or prompted language generation.
3. **No convergence guarantee.** There is no proof that the iterative
   procedure converges, that it samples from the right distribution,
   or that more iterations improve quality.
4. **Miscalibrated confidence.** Using max-probability as a confidence
   signal conflates model confidence with prediction accuracy, and
   ignores the dependence on the current masking rate.
5. **Length must be pre-specified.** The target length $n$ is sampled
   from a prior; incorrect length predictions cannot be corrected.

### Relation to This Thesis

Mask-Predict is the intellectual ancestor of the entire thesis. It
demonstrates empirically that confidence-guided iterative remasking
works, but provides no explanation for *why*. The thesis supplies the
missing theoretical foundation: within the MDM framework, the
max-probability confidence signal $c^i = \max_j p_\theta^i$ is an
approximation of the true information content $I^i(x) = H(x^i \mid
x^{\setminus i})$; remasking positions with low $c^i$ is equivalent
to locally refining the Riemann approximation of the information profile
integral; and confidence-guided remasking reduces $E_{\text{fact}}$
under conditions that are now formally stated.

The thesis generalises Mask-Predict in three ways: (1) it embeds
iterative remasking in the MDM probabilistic framework with a proper
forward process and reverse kernel; (2) it considers entropy and margin
as alternative confidence signals, not just max-probability; (3) it
derives conditions under which each signal is sufficient for
principled remasking. Citing Mask-Predict anchors the thesis in the
broader NAR generation literature and frames the contribution as
*formalising a successful heuristic*.

### Study Questions

1. Mask-Predict re-masks the $n_t$ *lowest-confidence* tokens per step. Why lowest and
   not highest? What would happen if you re-masked the highest-confidence tokens instead?
2. The number of re-masked tokens follows the linear schedule $n_t = n(T-t)/T$. Is there
   a principled justification for this schedule, or is it purely heuristic? How does it
   relate to the EB-Sampler's entropy budget?
3. Mask-Predict is a conditional model (source → target). For unconditional MDM
   generation (no source), what replaces the "source context"? Does the confidence
   signal change interpretation?
4. The confidence signal $c^i = \max_j p_\theta^j$ (max-probability) is Tier 3. Under
   what condition does it rank positions in the same order as the true $I^i(x)$? When
   does the ranking diverge?

\newpage

# Part II — Discrete Diffusion: Foundations and Scaling

---

## D3PM (2021) — The Origin of Absorbing Discrete Diffusion

**Full citation:** Austin, Johnson, Ho, Tarlow & van den Berg (2021),
"Structured Denoising Diffusion Models in Discrete State-Spaces."
*NeurIPS 2021.* arXiv:2107.03006.

### Summary

D3PM is the foundational paper that brings diffusion models to discrete
data, including text, graphs, and images with discrete pixel values.
Continuous diffusion models (DDPM, score-based models) rely on Gaussian
noise, which has no meaningful analogue for categorical variables. D3PM
solves this by defining a *family* of discrete forward Markov chains
parametrised by a transition matrix $Q_t$, and deriving a variational
lower bound (VLB) that can be optimised to train the reverse process.
*(Markov chain background: Appendix A.4; ELBO/VLB derivation: Appendix A.3.)*

The key design choice is the transition matrix $Q_t$. D3PM explores
three options: (1) **absorbing diffusion** — tokens eventually collapse
to a single absorbing [MASK] state — which is the process used by all
subsequent MDMs; (2) **uniform diffusion** — tokens uniformly transition
to all other tokens — which corresponds to gradual information loss but
no absorbing state; (3) **embedding-based diffusion** — transitions
follow distances in a learned embedding space.

For text, D3PM finds that the absorbing process significantly outperforms
the alternatives. The reason is simple: masking preserves the structure
of unmasked tokens entirely (they are unchanged), while uniform diffusion
corrupts all tokens at every step, making it harder for the model to
use the partially observed context. This empirical finding is the
justification for the design choices of MDLM, SEDD, LLaDA, and all
subsequent MDMs.

### Method Details

**General forward process.** For a token $x^i \in \{0, 1, \ldots, V-1\}$,
the forward Markov chain is:

$$q(z_t^i \mid z_{t-1}^i) = \mathrm{Cat}(z_t^i;\; \bar{Q}_t^{z_{t-1}^i})$$

where $\bar{Q}_t = Q_1 Q_2 \cdots Q_t$ is the cumulative transition
matrix and the superscript $z_{t-1}^i$ denotes the row indexed by the
current state.

**Absorbing transition matrix.** With mask token $m$ as the absorbing
state:

$$[Q_t]_{ij} = \begin{cases}
1 - \beta_t & \text{if } i = j \neq m \\
\beta_t     & \text{if } j = m,\; i \neq m \\
1           & \text{if } i = j = m
\end{cases}$$

The cumulative marginal has the clean form:

$$q(z_t^i \mid x^i) = \bar{\alpha}_t \cdot e_{x^i} + (1 - \bar{\alpha}_t) \cdot e_m, \quad \bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$$

which is exactly the MDLM forward process with $\alpha_t = \bar{\alpha}_t$.
*(Full derivation of the absorbing marginal and posterior: Appendix A.6.)*

**Posterior (closed form for absorbing diffusion).** Given $z_t^i$ and
the clean token $x^i$:

$$q(z_{t-1}^i \mid z_t^i, x^i) = \begin{cases}
1 & \text{if } z_t^i = x^i \; (\text{token is already clean}) \\[4pt]
\frac{\bar{\alpha}_{t-1}(1 - \beta_t)}{\bar{\alpha}_t} \cdot e_{x^i}
  + \frac{(1-\bar{\alpha}_{t-1})\beta_t + (1-\bar{\alpha}_t)(1-\beta_t)}{1 - \bar{\alpha}_t} \cdot e_m
& \text{if } z_t^i = m
\end{cases}$$

For the absorbing process this simplifies to: if $z_t^i \neq m$, then
$z_{t-1}^i = z_t^i$ with certainty; if $z_t^i = m$, then
$z_{t-1}^i = x^i$ with probability $\bar{\alpha}_{t-1}/( 1 - \bar{\alpha}_t \cdot \mathbf{1}_{z_t^i \neq m})$
and $z_{t-1}^i = m$ otherwise. This is the key posterior used by all
subsequent MDMs.

**VLB training objective.** D3PM minimises the variational lower bound
*(ELBO derivation: Appendix A.3; KL divergence: Appendix A.2.3)*:

$$\mathcal{L}_\text{VLB} = \underbrace{\mathbb{E}[\mathrm{KL}(q(z_{T} \mid x) \| p(z_T))]}_{\text{prior matching}} + \sum_{t=2}^{T} \mathbb{E}[\mathrm{KL}(q(z_{t-1} \mid z_t, x) \| p_\theta(z_{t-1} \mid z_t))] + \mathbb{E}[-\log p_\theta(x \mid z_1)]$$

The $x_0$-parameterisation predicts $\tilde{p}_\theta(x \mid z_t)$
(the clean token distribution) and then uses the closed-form posterior
to compute $p_\theta(z_{t-1} \mid z_t)$. An auxiliary loss
$\lambda \cdot \mathbb{E}[\mathrm{CE}(x, \tilde{p}_\theta(z_t))]$
directly supervises the clean-token prediction. MDLM later shows this
auxiliary term dominates and the VLB can be simplified to this term alone.

### Theoretical Contributions

D3PM proves that the VLB for discrete Markov chains decomposes into
a sum of per-step KL terms, analogously to DDPM for continuous diffusion.
It establishes that the posterior $q(z_{t-1} \mid z_t, x)$ has a
tractable closed form for any transition matrix whose cumulative product
$\bar{Q}_t$ can be computed in closed form (which is the case for
absorbing and uniform matrices). The paper does not provide
non-asymptotic sampling error bounds; the error analysis comes later
with Lavenant & Zanella (2025).

### Confidence Signal

D3PM uses ancestral sampling with no confidence signal. At each step,
it samples $z_{t-1} \sim p_\theta(z_{t-1} \mid z_t)$ uniformly over
all masked positions. The $x_0$-parameterisation (predicting
$\tilde{p}_\theta(x \mid z_t)$) is the source of all subsequent
per-token logit distributions that confidence signals are derived from.

### Limitations

1. **Absorbing process chosen empirically.** D3PM shows absorbing $>$
   uniform $>$ embedding-based on text, but provides no theoretical
   derivation of which $Q_t$ is optimal. The Lavenant & Zanella
   framework later provides a partial answer: the absorbing process
   minimises $E_{\text{fact}}$ for any given information profile.
2. **Slow sampling.** D3PM uses $T = 1000$ steps (borrowed from DDPM).
   MDLM reduces this to $T = 32$–$1024$ with no loss of quality.
3. **No remasking.** The ancestral sampler is strictly one-directional:
   a committed token stays committed.
4. **Sequence-level independence.** Positions are treated as
   conditionally independent given $z_t$ in the forward process, which
   is the root of $E_{\text{fact}}$.
5. **Limited language modelling experiments.** D3PM evaluates on
   character-level text8, not word-level OpenWebText; MDLM later shows
   the framework scales to the harder word-level setting.

### Relation to This Thesis

D3PM is the origin of the entire MDM framework used in the thesis.
The absorbing transition matrix $Q_t$ introduced here is the source of
$E_{\text{fact}}$: positions are masked independently in the forward
process, so the true reverse-step posterior has inter-token correlations
that the factored model $p_\theta(z_{t-1} \mid z_t) = \prod_i
p_\theta(z_{t-1}^i \mid z_t)$ cannot capture. The thesis's remasking
kernel corrects this: by re-masking a high-entropy committed token, the
model re-predicts it in a context that may have changed since the token
was first committed, partially recovering the inter-token dependencies
that the independence assumption discarded.

### Study Questions

1. D3PM evaluates three transition matrices: absorbing, uniform, and embedding-based.
   Why does absorbing outperform uniform for text? What property of the absorbing
   process is valuable and what does uniform destroy?
2. The posterior $q(z_{t-1}^i \mid z_t^i, x^i)$ is tractable in closed form. Write
   it out for $z_t^i = m$ (masked) and $z_t^i = x^i$ (already committed). What is
   the probability that a masked token remains masked at the next step?
3. D3PM uses an auxiliary cross-entropy loss $\lambda \cdot \mathrm{CE}(x, \tilde{p}_\theta(z_t))$.
   MDLM later shows this term dominates the VLB. Intuitively, why would directly
   predicting $x$ from $z_t$ be sufficient, without the VLB terms?
4. D3PM has no remasking. If you wanted to add a single remasking step after standard
   ancestral sampling, where in the algorithm would you insert it and what would
   determine which token to remask?

---

## SEDD (2024) — Score Entropy and Predictor-Corrector Samplers

**Full citation:** Lou, Meng & Ermon (2024), "Discrete Diffusion Modeling
by Estimating the Ratios of the Data Distribution." *ICML 2024.*
arXiv:2310.16834.

### Summary

SEDD introduces an alternative training objective for discrete diffusion
models that is more theoretically principled than the VLB of D3PM.
Instead of parameterising the reverse Markov kernel $p_\theta(z_{t-1}
\mid z_t)$ directly, SEDD parameterises the *concrete score*:

$$s_\theta(z_t, j) \approx \frac{p_t(z_t^{(j)})}{p_t(z_t)}$$

where $z_t^{(j)}$ is the sequence $z_t$ with one specific position
changed to token $j$, and $p_t$ is the marginal distribution at time
$t$. This is the discrete analogue of the score function
$\nabla_x \log p_t(x)$ in continuous diffusion. SEDD trains this score
with a *score entropy* objective that is a proper scoring rule and
provably consistent.

The key practical contribution is the **predictor-corrector (PC)
sampler**: at each denoising step, SEDD first applies a standard
predictor (reverse process) and then applies a Langevin-type corrector
that uses the score to refine the current sequence. This corrector is
the discrete diffusion analogue of the remasking step in ReMDM and
is the principled precursor to all remasking approaches.

### Method Details

**Score entropy objective.** The concrete score is trained by minimising:

$$\mathcal{L}(\theta) = \mathbb{E}_{t, x, z_t}\!\left[\sum_{j \neq z_t^i} w_t \cdot \left(s_\theta(z_t)_{z_t^i,j} - \log\frac{p_t(z_t^{(j)})}{p_t(z_t)} + h(s_\theta(z_t)_{z_t^i,j})\right)\right]$$

where $h(u) = e^u - u - 1 \geq 0$ is a convex function that, together
with the log-ratio term, forms a proper scoring rule. At the optimal
$\theta^*$, $s_\theta(z_t)_{ij} = p_t(z_t^{(j)}) / p_t(z_t)$ exactly.

**Equivalence with MDM for absorbing noise.** For the absorbing
(masking) noise process, the score $s_\theta(z_t)_{m,j}$ for a masked
position simplifies to $p_\theta(x^i = j \mid z_t) / p_t(z_t^i = j)$.
Since the denominator $p_t(z_t^i = j)$ is a constant (determined by
the marginal noise level), estimating the score is equivalent to
estimating $p_\theta(x^i = j \mid z_t)$ — the same objective as MDLM's
masked cross-entropy. This explains why MDLM achieves the same quality
as absorbing SEDD despite its simpler objective.

**Predictor-corrector sampler.** At each step $t \to t-1$:

*Predictor:* $z' \leftarrow p_\theta(z_{t-1} \mid z_t)$ (standard
reverse step, unmasks a fraction of tokens).

*Corrector ($K$ steps):* Repeat $K$ times:
```
for k = 1, ..., K:
    for each position i:
        # Compute transition rate using the concrete score
        rate_{z'^i → j} = s_θ(z')_{z'^i, j} · R_t(z'^i, j)
        # Apply one step of the continuous-time Markov chain
        z'^i ← transition according to rate
```

where $R_t$ is the noise process transition rate matrix (the time
derivative of $Q_t$). The corrector steps can be understood as
approximately sampling from $q_t(x \mid z_t)$ using MCMC with the
learned score as the energy function.

**Tweedie denoiser.** SEDD derives the discrete analogue of Tweedie's
formula, giving the minimum mean squared error (MMSE) estimate of $x$
given $z_t$ directly from the score. This MMSE estimate can be used as
a deterministic initialisation before stochastic sampling.

### Theoretical Contributions

SEDD provides two formal results:

1. **Consistency of the score entropy objective.** The minimiser
   $\theta^*$ of $\mathcal{L}(\theta)$ satisfies
   $s_{\theta^*}(z_t)_{ij} = p_t(z_t^{(j)}) / p_t(z_t)$ for all
   $z_t, i, j$ — i.e., the score is exactly recovered at optimality.
   This is the discrete analogue of Fisher consistency in score matching.

2. **PC sampler convergence.** Under mild conditions on the noise
   schedule, the PC sampler converges to the true data distribution as
   $T \to \infty$ and $K \to \infty$, with an explicit KL convergence
   rate. This is a stronger guarantee than D3PM's VLB bound (which only
   bounds training, not sampling quality) and predates Lavenant &
   Zanella (2025).

### Confidence Signal

SEDD does not use a per-token confidence signal for *selective*
correction. The corrector step applies the CTMC transition uniformly to
all positions. The concrete score $s_\theta(z_t)_{z_t^i, j}$ encodes
how strongly the model "wants" to change position $i$ from its current
value to $j$; positions with high total outgoing rate
$\sum_j s_\theta(z_t)_{z_t^i, j}$ are high-entropy and natural
candidates for correction. This is the Tier 1 signal (true posterior
ratio) that the thesis's Tier 3 heuristics approximate.

### Limitations

1. **Score entropy objective is harder to implement** than cross-entropy.
   For the absorbing process, the two objectives are equivalent, removing
   SEDD's theoretical advantage on the most practically important case.
2. **PC corrector is not selective.** The corrector applies to all
   positions, not just those where correction is most beneficial. This
   makes it less efficient than a confidence-guided corrector.
3. **Non-absorbing SEDD underperforms.** The uniform noise process gives
   worse text quality than absorbing noise, confirming that absorbing is
   the right inductive bias for language.
4. **No per-token confidence signal for unmasking order.** SEDD unmasks
   positions in the default schedule order, not in order of information
   content.

### Relation to This Thesis

SEDD's PC sampler is the most direct precursor to the remasking approach
of this thesis. The corrector step in SEDD — applying discrete CTMC
transitions guided by the concrete score — is a principled correction
mechanism that can in principle target high-entropy positions. The thesis
extends this in two ways: (1) it derives an explicit $E_{\text{fact}}$
reduction bound for selective correction (rather than the uniform CTMC
correction of SEDD); (2) it shows that the entropy-based Tier 3 signals
(which require only the forward-pass logits, not the full concrete score)
are sufficient for achieving this reduction under a consistency
assumption. The SEDD submodule (`external/sedd`) provides pretrained
checkpoints on OpenWebText for baseline comparison.

### Study Questions

1. For the absorbing noise process, the concrete score simplifies to
   $s_\theta(z_t)_{m,j} \propto p_\theta(x^i = j \mid z_t)$. This means SEDD's
   objective is equivalent to MDLM's cross-entropy for absorbing noise. Does this mean
   SEDD and MDLM learn the same model? What differences remain?
2. SEDD's corrector applies CTMC transitions to *all* positions uniformly. The thesis
   applies remasking selectively. Formally, what would a "selective SEDD corrector"
   look like? Would it still have SEDD's convergence guarantee?
3. SEDD proves the PC sampler converges as $T \to \infty$ and $K \to \infty$. Is there
   a finite-$T$, finite-$K$ convergence rate? What does it depend on?
4. The concrete score $s_\theta(z_t)_{ij} = p_t(z_t^{(j)})/p_t(z_t)$ is a ratio of
   marginals. At what point in the generation process is this ratio most informative?
   At the beginning (many masked tokens) or the end (few masked tokens)?

---

## MD4 and MDLM (2024) — The Practical MDM Framework

**Full citations:** Shi et al. (2024), "Simplified and Generalized Masked
Diffusion for Discrete Data" (MD4); Sahoo et al. (2024), "Simple and
Effective Masked Diffusion Language Models" (MDLM). *NeurIPS 2024.*
arXiv:2406.07524.

### Summary

MD4 and MDLM simultaneously simplify and scale the D3PM framework,
arriving at what is now the standard MDM formulation. Both papers observe
that the D3PM VLB simplifies dramatically for the absorbing noise
process: the dominant training signal is a weighted sum of per-position
cross-entropy losses on masked tokens, and the weighting can be chosen
via importance sampling to minimise variance. MDLM additionally shows
that a cosine noise schedule achieves approximately uniform
signal-to-noise ratio per step, improving training stability.

MDLM-OWT (130M parameters, trained on OpenWebText) is the experimental
backbone of the thesis. It is fast (single forward pass ~10ms on A100),
publicly available (`kuleshov-group/mdlm`), and its architecture is the
base that ReMDM, PRISM, and many other systems build on.

### Method Details

**Forward process.** Identical to D3PM's absorbing process:

$$q(z_t \mid x) = \prod_i \mathrm{Cat}(z_t^i;\; \alpha_t \cdot e_{x^i} + (1-\alpha_t) \cdot e_m)$$

**Noise schedules.** MD4 uses linear $\alpha_t = 1-t$. MDLM introduces
the cosine schedule $\alpha_t = \cos^2(\pi t / 2)$, which keeps
$\alpha_t$ close to 1 for most of $[0,1]$ and drops sharply near $t=1$,
concentrating training signal at moderate noise levels where the model's
predictions are neither trivially easy nor impossible.

**Reverse kernel.** From D3PM's posterior (Bayes' rule on the absorbing
process):

$$p_\theta(z_s^i \mid z_t) = \frac{\alpha_s}{\alpha_t} \cdot \delta(z_s^i = z_t^i) + \left(1 - \frac{\alpha_s}{\alpha_t}\right) \cdot p_\theta(x^i \mid z_t)$$

For $z_t^i = m$ (masked): unmask with probability $1 - \alpha_s/\alpha_t$,
re-masking otherwise. For $z_t^i \neq m$ (already committed): stay with
probability 1. This is the "absorbing is absorbing" property: no
remasking in the standard process.

**MDLM training objective.** The simplified ELBO is:

$$\mathcal{L}_\text{MDLM} = \mathbb{E}_{t \sim p(t),\, x \sim \pi,\, z_t \sim q_t(\cdot|x)}\!\left[w_t \cdot \sum_{\{i:\, z_t^i = m\}} -\log p_\theta(x^i \mid z_t)\right]$$

where $w_t = \alpha_t' / (1-\alpha_t)$ is the importance weight derived
from the noise schedule derivative. This is simply a weighted masked
cross-entropy, identical to BERT pre-training but with a principled
per-step weighting.

**Sampling algorithm.** Standard MDLM sampling with $T$ steps:

```
z_T = [m, m, ..., m]
for t = T, T-1, ..., 1:
    # One forward pass: get clean token predictions for all masked positions
    p^i = p_θ(x^i | z_t)   for all i with z_t^i = m
    # Sample: unmask each masked position with probability (1 - α_{t-1}/α_t)
    for i with z_t^i = m:
        if Uniform(0,1) < 1 - α_{t-1}/α_t:
            z_{t-1}^i ← sample from p^i        # commit
        else:
            z_{t-1}^i ← m                       # stay masked
return z_0
```

This is essentially: at each step, independently commit each masked
position with probability proportional to the noise schedule derivative.
The *order* of commitments is random and position-independent — there
is no priority given to easy tokens.

### Theoretical Contributions

MDLM proves that the simplified objective $\mathcal{L}_\text{MDLM}$ is
a strict lower bound on the VLB $\mathcal{L}_\text{D3PM}$ and that
the difference goes to zero as the model approaches the true score. This
justifies replacing the VLB with the simpler objective. The cosine
schedule is motivated by showing that $\mathrm{SNR}(t) = \alpha_t / (1-\alpha_t)$
has approximately constant log-derivative under the cosine schedule,
which is the MDM analogue of the continuous diffusion result that
uniform SNR spacing minimises training variance. No non-asymptotic
sampling error bounds are provided.

### Confidence Signal

None at inference time. The per-position distribution $p_\theta(x^i
\mid z_t)$ is available for free at every step and is the raw material
for all confidence signals in the thesis:
- Max-prob: $c^i = \max_j p_\theta(x^i = j \mid z_t)$
- Entropy: $c^i = -H(p_\theta(x^i \mid z_t)) = \sum_j p_\theta^j \log p_\theta^j$
- Margin: $c^i = p_\theta^{(1)} - p_\theta^{(2)}$ (gap between top-2)

### Limitations

1. **Uniform random unmasking order.** Positions are committed
   independent of their uncertainty, violating the optimality condition
   derived by Lavenant & Zanella.
2. **No remasking.** The absorbing kernel sets $z_s^i = z_t^i$ with
   probability $\alpha_s/\alpha_t > 0$ only for *unmasked* positions;
   masking an already-committed token requires going forward in the
   noise process, which the reverse sampler does not do.
3. **Independence assumption.** The factored reverse kernel
   $\prod_i p_\theta(z_s^i \mid z_t)$ ignores the true joint posterior,
   inducing $E_{\text{fact}}$.
4. **No formal sampling error analysis.** Quality guarantees are purely
   empirical (perplexity on OWT), not formal.

### Relation to This Thesis

MDLM-OWT is the primary experimental backbone. The forward process,
reverse kernel, and training objective of MDLM are the setting in which
the thesis's theoretical results are derived. The per-token logit
distribution $p_\theta(x^i \mid z_t)$ is the source of all
training-free confidence signals. The independence assumption of the
reverse kernel is the root of $E_{\text{fact}}$, the quantity the
thesis bounds.

### Study Questions

1. The MDLM reverse kernel reads: "if $z_t^i = m$, unmask with probability $1 - \alpha_s/\alpha_t$".
   What is the probability when $s = t-1$ and the cosine schedule is used? How does
   this change between early steps ($t \approx T$) and late steps ($t \approx 1$)?
2. MDLM commits tokens uniformly at random, ignoring position uncertainty. Using the
   Lavenant & Zanella bound, what is the $E_{\text{fact}}$ penalty paid by committing
   a high-entropy token $i$ (with $I^i \gg \bar{I}$) early?
3. The cosine schedule keeps $\alpha_t$ near 1 for most of $[0,1]$ and drops sharply
   near $t=1$. How does this affect how many tokens are unmasked per step at different
   stages? Is there a step count where most tokens are committed?
4. MDLM's training loss is weighted masked cross-entropy. If you had unlimited data
   and a perfectly trained model ($p_\theta = q$), would $E_{\text{fact}}$ be zero?
   Why or why not?

---

## LLaDA (2025) — Scaling Masked Diffusion to 8B Parameters

**Full citation:** Nie et al. (2025), "LLaDA: Large Language Diffusion
with mAsking." arXiv:2502.09992. HuggingFace: `GSAI-ML/LLaDA-8B-Instruct`.

### Summary

LLaDA answers the question: can masked diffusion language models compete
with frontier autoregressive LLMs at scale? Training an 8B-parameter
bidirectional transformer (LLaDA-8B) on 2.3 trillion tokens with the
MDLM objective, the paper shows competitive or superior performance to
LLaMA 3 8B on MMLU, GSM8K, HumanEval, and multi-turn dialogue — the
first masked diffusion model to achieve this at any scale. The
instruction-tuned variant (LLaDA-8B-Instruct) is the backbone for
PRISM's code generation experiments and is one of the two experimental
backbones (alongside MDLM-OWT) of the thesis.

The key implication for the thesis: LLaDA demonstrates that MDMs are
not a niche academic curiosity but a genuine frontier model architecture,
and that improving their sampling quality has practical significance.
Crucially, LLaDA uses a naive uniform remasking schedule — no confidence
signal — leaving substantial quality on the table at inference time.

### Method Details

**Architecture.** LLaDA-8B is a standard transformer with:
- Bidirectional (non-causal) self-attention — unlike causal LLMs,
  every token can attend to every other token, which is natural for
  the masked prediction objective.
- Pre-LayerNorm and RoPE positional encodings (standard modern LLM recipe).
- Vocabulary: 128K tokens (LLaMA 3 tokeniser for comparability).
- 32 layers, 32 heads, hidden dimension 4096 (same as LLaMA 3 8B).

**Pre-training.** Identical to MDLM: the loss is the importance-weighted
masked cross-entropy on the 2.3T token corpus. The noise schedule
follows MDLM's cosine schedule.

**Instruction tuning (LLaDA-8B-Instruct).** Given a prompt $x$ and
response $y$, the model is trained with the response tokens masked and
the prompt tokens frozen:

$$\mathcal{L}_\text{SFT} = \mathbb{E}_{t, y_t}\!\left[w_t \sum_{\{i:\, y_t^i = m\}} -\log p_\theta(y^i \mid x, y_t)\right]$$

This is the first demonstration of SFT for MDMs working at 8B scale,
and the recipe is clean: just mask the response tokens and train the
MDLM objective conditioned on the unmasked prompt.

**Sampling.** LLaDA uses a stochastic remasking schedule during
generation: at each step $t$, all committed (non-masked) tokens are
independently re-masked with probability $\rho(t)$, where $\rho(t)$
is a fixed schedule not conditioned on token confidence. The
motivation is that remasking allows the model to correct early
commitments in later steps when more context is available — but the
schedule is heuristic, not principled.

### Theoretical Contributions

LLaDA's contribution is empirical: it demonstrates scale, not theory.
The paper provides scaling analysis showing that LLaDA obeys similar
scaling laws to GPT-style models (loss $\propto$ compute$^{-\beta}$),
which validates MDMs as a compute-competitive alternative to
autoregressive LLMs. No theoretical results on sampling quality.

### Confidence Signal

None. LLaDA's remasking schedule is based only on the current time step
$t$ (a fixed function), not on per-token confidence. This is the thesis's
key target: replacing LLaDA's uniform schedule with a confidence-guided
one should improve quality without any fine-tuning.

### Limitations

1. **Naive remasking.** Uniform schedule ignores which tokens are
   uncertain. This is the primary weakness the thesis addresses.
2. **Bidirectional attention only.** Cannot be used for streaming or
   left-to-right generation, limiting some deployment contexts.
3. **PRISM adapter not public.** The PRISM quality head trained on
   LLaDA-8B is unavailable publicly (as of early 2026), limiting
   comparison to the PRISM upper bound.
4. **No formal connection to information profile.** The remasking
   schedule is not derived from $I(x)$ or any principled criterion.

### Relation to This Thesis

LLaDA-8B-Instruct is the large-scale experimental backbone. The main
experiment on LLaDA is: does replacing LLaDA's uniform remasking with
the thesis's confidence-guided strategies (entropy threshold, top-$k$
low-confidence) improve generation quality on standard benchmarks
(LAMBADA, HellaSwag, MBPP)? If yes at 8B scale, the thesis's
contribution generalises beyond the small MDLM-OWT setting. The uniform
remasking of LLaDA is the weakest baseline the thesis should beat by
a significant margin.

### Study Questions

1. LLaDA's remasking schedule re-masks committed tokens with probability $\rho(t)$
   at each step, independent of token confidence. The thesis uses confidence-guided
   remasking. LLaDA is 8B parameters vs MDLM's 130M. Do you expect the benefit
   of confidence-guided remasking to be *larger* or *smaller* at 8B scale? Why?
2. LLaDA uses bidirectional (non-causal) attention. What does this mean for how
   the model uses context when predicting a masked token? How does it differ from
   GPT-style causal LLMs?
3. LLaDA's instruction tuning freezes prompt tokens and only masks response tokens.
   Is the information profile $I^i(x)$ for response tokens affected by whether the
   prompt is long or short? What would you expect for very long prompts?
4. LLaDA achieves competitive performance with LLaMA 3 8B on MMLU and GSM8K. Why
   might a masked diffusion model be competitive on these tasks despite generating
   text in a non-autoregressive manner?

\newpage

# Part III — A Unified Theoretical Lens: Discrete Flow Matching

---

## Discrete Flow Matching (2024) — DFM as a Unifying Framework

**Full citation:** Gat, Remez, Shaul, Kreuk, Chen, Synnaeve, Adi &
Lipman (2024), "Discrete Flow Matching." *NeurIPS 2024.*
arXiv:2407.15595.

### Summary

Discrete Flow Matching (DFM) extends continuous flow matching (Lipman
et al., 2022) to categorical data, providing a unified theoretical
framework that subsumes both MDMs (D3PM, MDLM) and SEDD as special
cases. The key insight is that generative modelling of discrete data
can be understood as learning a probability flow from a simple prior
(e.g., uniform or fully masked) to the data distribution, defined
by a *velocity field* on the space of probability simplices.

For the thesis, DFM provides two valuable things. First, it shows that
the masked cross-entropy training objective of MDLM is exactly the
flow matching objective under the absorbing probability path — this
unifies the training objectives across the literature and removes the
need to reason about VLBs or score entropy separately. Second, DFM
gives the cleanest language for corrector steps: a corrector is a
flow that leaves the marginal distribution $p_t$ invariant while
reducing entropy along individual trajectories, which corresponds
precisely to re-masking high-entropy tokens and re-predicting them.

### Method Details

**Probability paths.** DFM defines a parameterised family of probability
paths $p_t(z)$ interpolating between a prior $p_0$ and the data
distribution $p_1$. For the absorbing path (the MDM case):

$$p_t(z) = \sum_{x \sim p_{\text{data}}} \mathrm{Cat}(z;\; \alpha_t \cdot e_x + (1-\alpha_t) \cdot e_m)$$

This is identical to the MDLM forward-process marginal.

**Discrete velocity field.** The generative model is a continuous-time
Markov chain (CTMC) over sequences, specified by a rate matrix. For
the absorbing path, the velocity field (rate of probability flow from
mask token $m$ to token $j$ at position $i$) is:

$$u_t(z^i = m \to j;\; x) = \frac{\dot{\alpha}_t}{1-\alpha_t} \cdot \mathbf{1}[j = x^i]$$

where $\dot{\alpha}_t = d\alpha_t/dt < 0$. The marginalised velocity
(averaging over $x$ given $z_t$) is:

$$u_t(z^i = m \to j) = \frac{-\dot{\alpha}_t}{1-\alpha_t} \cdot p_\theta(x^i = j \mid z_t)$$

This is the quantity the model $p_\theta$ learns to estimate; minimising
the discrete flow matching objective is equivalent to minimising the
masked cross-entropy of MDLM (up to a time-weighting factor).

**Training objective.** The discrete flow matching loss for the absorbing
path:

$$\mathcal{L}_\text{DFM} = \mathbb{E}_{t, x, z_t}\!\left[\frac{-\dot{\alpha}_t}{1-\alpha_t} \sum_{\{i:\, z_t^i = m\}} -\log p_\theta(x^i \mid z_t)\right]$$

This is exactly the MDLM objective with time weight $w_t = -\dot{\alpha}_t / (1-\alpha_t)$.

**Corrector steps in DFM.** A corrector for a discrete flow model is
any CTMC that: (1) leaves $p_t$ stationary; and (2) reduces entropy
along individual sequences. The canonical choice is a Gibbs sampler
with stationary distribution $p_t$: pick a position $i$, mask it
($z^i \leftarrow m$), and re-sample from $p_\theta(x^i \mid z_t)$.
This is exactly the remasking operation of ReMDM.

The DFM perspective makes clear *why* this works: the Gibbs corrector
moves the current sequence $z$ closer to a high-probability sequence
under $p_t$ while preserving the marginal distribution. The thesis's
contribution in DFM language is: a *confidence-guided* Gibbs corrector
that selectively applies this operation to high-entropy positions reduces
the KL between the current trajectory distribution and $p_t$ more
efficiently than a uniform Gibbs corrector.

### Theoretical Contributions

1. **Unification.** DFM proves that MDMs and SEDD are both special
   cases of the discrete flow matching framework with different choices
   of probability path.
2. **Objective equivalence.** For the absorbing path, the DFM objective
   is identical to the MDLM cross-entropy (up to weighting), justifying
   MDLM's simple training.
3. **Optimal probability path.** DFM derives the path that minimises
   the continuous-time KL between the generative trajectory and the
   data-generating process, providing a theoretical basis for choosing
   the noise schedule.
4. **Corrector theory.** DFM gives the cleanest theoretical framework
   for corrector steps: they are Gibbs samplers for $p_t$, and their
   mixing rate is the spectral gap of the Gibbs chain.

### Confidence Signal

DFM does not introduce a new confidence signal. The velocity field
$u_t(z^i = m \to j)$ is proportional to $p_\theta(x^i = j \mid z_t)$,
so the entropy of the velocity field $H(u_t(z^i = m \to \cdot))$ equals
the entropy of the token prediction distribution $H(p_\theta(x^i \mid
z_t))$ — the Tier 3 entropy signal of the thesis.

### Limitations

1. **Continuous-time formulation** is cleaner theoretically but harder
   to implement than the discrete-step MDLM sampler; in practice, the
   Euler discretisation of the DFM ODE reduces to MDLM sampling.
2. **Corrector theory is not quantitative.** DFM shows that correctors
   work but does not derive an explicit bound on the KL reduction per
   corrector step, and does not connect to $E_{\text{fact}}$.
3. **Limited text experiments.** DFM evaluates primarily on protein
   sequences and audio tokens; text results are limited compared to
   MDLM-OWT.
4. **The absorbing path is not proven optimal.** It is one choice among
   infinitely many; the optimal path for text generation is an open
   question.

### Relation to This Thesis

DFM provides the cleanest theoretical language for Chapter 3 of the
thesis. The thesis's main theorem — confidence-guided remasking reduces
$E_{\text{fact}}$ — can be stated in DFM language as: a
*confidence-guided Gibbs corrector* reduces the discretisation error of
the DFM Euler sampler by targeting positions where the velocity field
has the highest entropy. Including DFM in the literature review
positions the thesis at the intersection of diffusion theory and flow
matching, broadening the potential readership and citation impact.

### Study Questions

1. DFM shows the training objective is $\mathcal{L}_{\text{DFM}} \propto
   \frac{-\dot\alpha_t}{1-\alpha_t}\sum_i -\log p_\theta(x^i \mid z_t)$.
   Compare this to MDLM's $w_t = \alpha_t'/(1-\alpha_t)$. Are these the same
   time-weighting? What does the DFM perspective add beyond MDLM?
2. A DFM corrector is any CTMC that leaves $p_t$ stationary. Is a corrector that
   *only* re-masks tokens (but never unmasked them) a valid corrector in DFM?
   What condition would it need to satisfy?
3. The "optimal probability path" in DFM minimises the continuous-time KL. The
   absorbing path is one option. Give an intuition for what a "better" path might
   look like and why it might not be absorbing.
4. The mixing rate of the Gibbs corrector is the spectral gap $\lambda$. What
   determines $\lambda$ for a text sequence? How does sequence length affect it?
   How does dependency structure (e.g., code vs. random text) affect it?

\newpage

# Part IV — The Error-Bound Framework: Why Algorithms Fail and How to Fix Them

---

## Lavenant & Zanella (2025) — The Foundation: Decomposing Sampling Error

**Full citation:** Lavenant & Zanella (2025), "Sampling Error Analysis
for Masked Diffusion Models."

### Summary

This paper provides the first rigorous non-asymptotic analysis of the
sampling error of MDM algorithms. Its central contribution is a KL
divergence bound that decomposes the total sampling error into two
independent terms: (1) a *learning error* $E_{\text{learn}}$ due to the
model $p_\theta$ not perfectly matching the true score, and (2) a
*factorization error* $E_{\text{fact}}$ due to the sampling algorithm
committing multiple tokens simultaneously and treating them as
independent. The factorization error is characterised as a Riemann
approximation error of the sequence's *information profile*, and the
paper derives the optimal unmasking schedule that minimises it.

The remasking direction — what happens when committed tokens are
re-masked and re-predicted — is explicitly identified as an open problem
and the central gap this thesis fills. This paper is supervised by
Prof. Zanella, the thesis supervisor.

### Method Details

**Setup.** Let $\pi$ be the true data distribution, $p_\theta$ the
trained MDM, and $p_{\text{alg}}$ the distribution over sequences
produced by running the MDM sampler for $T$ steps. The main bound is:

$$\mathrm{KL}(\pi \| p_{\text{alg}}) \leq E_{\text{learn}} + E_{\text{fact}}$$

**Learning error.** $E_{\text{learn}}$ is the cumulative KL between the
true denoising posterior and the model's approximation, summed over
all steps:

$$E_{\text{learn}} = \sum_{t=1}^{T} \mathbb{E}_{z_t \sim q_t}\!\left[\mathrm{KL}\!\left(q(x \mid z_t) \;\Big\|\; p_\theta(x \mid z_t)\right)\right]$$

This goes to zero as $p_\theta \to q$ (perfect training) and is
independent of the sampling algorithm's choices (which tokens to unmask,
in what order). It can be bounded by the training loss.

**Factorization error.** $E_{\text{fact}}$ arises from the independence
assumption in the reverse kernel:

$$p_\theta(z_{t-1} \mid z_t) = \prod_i p_\theta(z_{t-1}^i \mid z_t)$$

The true posterior is not factored: simultaneously unmasking $k$
positions from the same masked sequence introduces correlations that
the product-of-marginals approximation cannot capture. The paper shows:

$$E_{\text{fact}}(t) \approx (\Delta\alpha_t)^2 \cdot \sum_i \mathrm{Var}_{x \sim q(x|z_t)}\!\left[H(x^i \mid x^{\setminus i})\right]$$

where $\Delta\alpha_t = \alpha_{t-1} - \alpha_t$ is the step size in
the noise schedule.

**Information profile.** The key object is the per-position conditional
entropy *(entropy and conditional entropy defined in Appendix A.2.1–A.2.2)*:

$$I^i(x) = H(x^i \mid x^{\setminus i}) \quad (\text{conditional entropy of position } i \text{ given all others})$$

and the total information $I(x) = \sum_i I^i(x)$. The sequence
$\{I^i(x)\}_{i=1}^L$ is the *information profile* of $x$; it measures
how hard each position is to predict from its context.

**$E_{\text{fact}}$ as a Riemann error.** Summing over steps and taking
the continuum limit, the total factorization error is a Riemann
approximation error of $\int_0^1 I(x)\, dt$:

$$E_{\text{fact}} \leq C \cdot \sum_{t=1}^{T} (\Delta\alpha_t)^2 \cdot \Sigma^2 \approx C \cdot \frac{\Sigma^2}{T}$$

where $\Sigma^2 = \mathrm{Var}_{i}[I^i(x)]$ is the variance of the
information profile across positions.
*(Derivation and intuition for this bound: Appendix A.7.)* This is minimised when:
(1) $T$ is large (many steps), and (2) $\Sigma^2$ is small (information
is evenly distributed across positions, so no step is much harder
than another).

**Optimal unmasking schedule.** To minimise $E_{\text{fact}}$:
- Choose $T$ as large as the compute budget allows.
- Unmask positions in **increasing order of $I^i(x)$** — easy tokens
  first, hard tokens last — so that at each step the information
  released is approximately equal: $\Delta I_t = I(x)/T$ for all $t$.
- In practice, $I^i(x)$ is approximated by $H(p_\theta(x^i \mid z_t))$
  (the model's predicted entropy), which is the EB-Sampler algorithm.

**Main theorem (informal).** For any unmasking schedule with $T$ steps:
$$E_{\text{fact}} \leq C \cdot \left(\frac{I(x)}{T}\right)^2$$

with equality for the uniform step size schedule. The optimal
(non-uniform) schedule achieves:
$$E_{\text{fact}}^* \leq C \cdot \frac{I(x)^2}{T^2} \cdot \left(1 + \frac{\Sigma^2}{\bar{I}^2}\right)^{-1}$$

where $\bar{I} = I(x)/L$ is the mean information per position.

### Theoretical Contributions

1. **First non-asymptotic KL bound for MDM sampling** — establishes
   that the gap between $\pi$ and $p_{\text{alg}}$ can be controlled.
2. **Decomposition into $E_{\text{learn}} + E_{\text{fact}}$** — cleanly
   separates the effect of model quality from algorithm design.
3. **Identification of $E_{\text{fact}}$ as a Riemann error** of $I(x)$
   — connects the sampling algorithm to information theory.
4. **Optimal unmasking schedule** — first principled derivation of
   which tokens to unmask in what order.
5. **$O(k^2)$ penalty for simultaneous unmasking** — shows that
   committing $k$ tokens at once is $k^2 \times$ worse than committing
   them sequentially.

### Confidence Signal

None (this is a theoretical analysis paper, not an algorithm paper). The
analysis implies that the optimal confidence signal for unmasking order
is $I^i(x) = H(x^i \mid x^{\setminus i})$, and that $H(p_\theta(x^i
\mid z_t))$ is its practical approximation.

### Limitations

1. **Remasking not covered.** This is the explicit open problem that
   the thesis fills. The framework analyses only the unmasking direction.
2. **$E_{\text{learn}}$ treated as a black box.** The bound on
   $E_{\text{learn}}$ is in terms of the model's training loss, which
   is not further analysed.
3. **Assumes factored posterior.** The analysis uses the product-of-marginals
   approximation as a baseline; the bound could be tighter with a more
   careful approximation.
4. **Constant $C$ not explicitly computed.** The bound is tight up to
   an unspecified constant depending on sequence length and vocabulary
   size.

### Relation to This Thesis

This paper *is* the thesis's theoretical foundation and is supervised
by the thesis advisor. The thesis's Chapter 3 extends the Lavenant &
Zanella framework in one direction: it adds remasking transitions to the
reverse process and derives:

1. A new KL decomposition that includes a *remasking error term*
   $E_{\text{remask}}$.
2. Conditions under which $E_{\text{remask}} < 0$ — i.e., remasking
   strictly reduces the total bound.
3. A characterisation of the optimal remasking schedule in terms of
   the confidence signal and the information profile.

The central thesis theorem is the remasking analogue of Lavenant &
Zanella's optimal unmasking result.

### Study Questions

1. The factorization error is $E_{\text{fact}} \leq C \cdot \Sigma^2 / T$ for the
   uniform schedule. What happens to $E_{\text{fact}}$ as $T \to \infty$? Does this
   mean infinite steps gives perfect generation? What limits quality as $T \to \infty$?
2. The bound says $E_{\text{fact}} \propto (\Delta\alpha_t)^2$. Why quadratic, not
   linear? Intuitively, why does committing $k$ tokens at once cost $k^2$ rather than $k$?
3. The optimal unmasking schedule is "easy tokens first" (ascending $I^i$). Why easy
   first and not hard first? Give an intuitive argument — what goes wrong if you
   commit the hardest token (highest $I^i$) first?
4. $E_{\text{learn}}$ is the error due to $p_\theta \neq q$. Is $E_{\text{learn}}$
   reduced by remasking, or only $E_{\text{fact}}$? What would a "remasking that
   reduces $E_{\text{learn}}$" look like — is it even possible without retraining?

---

## EB-Sampler (2025) — Optimal Unmasking via Entropy Budgets

**Full citation:** Ben-Hamu et al. (2025), "EB-Sampler: Entropy-Bounded
Sampling for Masked Diffusion Models."

### Summary

The EB-Sampler is the algorithmic companion to the Lavenant & Zanella
framework: it derives the concrete algorithm that minimises $E_{\text{fact}}$
and achieves the optimal unmasking schedule. The key idea is to control
the *amount of information* (entropy) unmasked at each step by
adaptively choosing how many and which tokens to unmask, rather than
committing to a fixed number per step. By unmasking tokens in ascending
order of predicted entropy and stopping when the entropy budget
$\varepsilon$ is consumed, the EB-Sampler ensures each step releases
approximately equal information — the condition for $E_{\text{fact}}$
minimisation.

The empirical result is striking: the EB-Sampler achieves 2–3$\times$
speedup over standard fixed-step MDLM at the same quality, or
significantly better quality at the same step count. For the thesis,
the EB-Sampler is the "gold standard" for the unmasking direction —
the thesis's task is to derive an equally principled standard for the
*remasking* direction.

### Method Details

**Information profile approximation.** The true per-position information
$I^i(x) = H(x^i \mid x^{\setminus i})$ requires knowing $x$ (which is
unavailable at generation time). The EB-Sampler approximates it by:

$$\hat{I}^i(z_t) = H(p_\theta(x^i \mid z_t)) = -\sum_j p_\theta(x^i=j \mid z_t) \log p_\theta(x^i=j \mid z_t)$$

This is the entropy of the model's predicted token distribution at
position $i$. High $\hat{I}^i$ means the model is uncertain; low
$\hat{I}^i$ means the model is confident. At optimality
($p_\theta = q$), $\hat{I}^i(z_t) \to I^i(x)$ in expectation.

**Entropy budget algorithm.** At each step $t$:

```
# Compute predicted entropy for all currently masked positions
Î^i ← H(p_θ(x^i | z_t))   for all i with z_t^i = m

# Sort by ascending entropy (easy positions first)
sorted_positions ← argsort(Î, ascending=True)

# Greedily unmask until entropy budget ε is consumed
budget ← ε
unmask_set ← {}
for i in sorted_positions:
    if budget >= Î^i:
        unmask_set.add(i)
        budget -= Î^i
    else:
        break

# Commit the selected positions
for i in unmask_set:
    z_{t-1}^i ← sample from p_θ(x^i | z_t)
    z_{t-1}^j ← z_t^j   for j ∉ unmask_set
```

The total number of steps is adaptive: if the sequence has total
predicted information $\hat{I}(z_t) = \sum_i \hat{I}^i$, then roughly
$T \approx \hat{I} / \varepsilon$ steps are needed to unmask everything.

**Entropy budget and $E_{\text{fact}}$.** The paper proves (Theorem 2):

$$E_{\text{fact}}(\text{EB}) \leq \varepsilon \cdot \max_i \hat{I}^i$$

Compare to the uniform fixed-step sampler with $T$ steps:

$$E_{\text{fact}}(\text{uniform}) \leq \frac{I(x)}{T} \cdot \max_i I^i(x)$$

Both bounds have the same structure: (entropy per step) $\times$ (max
per-position entropy). The EB-Sampler automatically sets the step size
to be $\varepsilon$ everywhere, which is exactly the Riemann partition
that minimises $E_{\text{fact}}$ for a given total information budget.

**Optimality.** The paper proves (Theorem 3) that among all deterministic
unmasking schedules (strategies that decide which tokens to unmask at
each step based on the current state), the EB-Sampler achieves the
minimum $E_{\text{fact}}$ for a given total number of forward passes.
No deterministic schedule can do better.

### Theoretical Contributions

1. **Derives the EB-Sampler as the optimal unmasking algorithm** within
   the Lavenant & Zanella framework.
2. **Proves the $E_{\text{fact}}$ bound** for the EB-Sampler explicitly.
3. **Proves optimality** among deterministic unmasking strategies.
4. **Connects entropy budgets to information profiles** — the first
   practical algorithm directly motivated by $I(x)$.

### Confidence Signal

$\hat{I}^i = H(p_\theta(x^i \mid z_t))$ — the entropy of the predicted
token distribution. This is a Tier 3 (heuristic, training-free) signal
that is free to compute from the forward-pass logits. The EB-Sampler
does not need a separate confidence model; the entropy is always
available.

**Why entropy and not max-prob or margin?** Entropy is the most natural
approximation of $I^i(x) = H(x^i \mid x^{\setminus i})$ because both
measure "how uncertain is the distribution over $x^i$?". Max-probability
and margin are monotonically related to entropy for unimodal
distributions but diverge for multimodal ones (e.g., when the model
is split between two plausible tokens).

### Limitations

1. **Unmasking only.** The optimality result covers only the forward
   (unmasking) direction. Remasking is not considered, and there is no
   bound on what happens after errors are made.
2. **$\hat{I}^i$ is a noisy proxy.** At early generation steps, the
   model's entropy estimate may poorly reflect the true $I^i(x)$
   because the context is highly uncertain.
3. **Variable step count.** The adaptive number of steps makes it
   harder to compare against fixed-budget baselines on wall-clock time.
4. **No remasking.** Errors committed in early steps (when
   $\hat{I}^i$ is poorly estimated) propagate to the end of generation.

### Relation to This Thesis

The EB-Sampler and the thesis are direct complements:

$$\underbrace{\text{EB-Sampler}}_{\substack{\text{optimal unmasking} \\ \text{schedule}}} \quad \longleftrightarrow \quad \underbrace{\text{Thesis}}_{\substack{\text{optimal remasking} \\ \text{schedule}}}$$

The EB-Sampler minimises $E_{\text{fact}}$ for the unmasking direction.
The thesis minimises the additional $E_{\text{fact}}$ reduction achievable
by the remasking direction. The thesis's main theorem should state: given
that the EB-Sampler has already produced a candidate sequence with some
committed tokens, the remasking strategy that further reduces
$E_{\text{fact}}$ most efficiently is: remask the positions with the
highest predicted entropy $\hat{I}^i$, using threshold $\tau$ chosen
to match the remaining entropy budget.

### Study Questions

1. The EB-Sampler sets the entropy budget $\varepsilon$ as the key hyperparameter.
   How should $\varepsilon$ be chosen in practice? What is the relationship between
   $\varepsilon$ and the total number of steps $T$?
2. The EB-Sampler bound is $E_{\text{fact}}(\text{EB}) \leq \varepsilon \cdot \max_i \hat{I}^i$.
   The uniform bound is $E_{\text{fact}}(\text{uniform}) \leq (I(x)/T) \cdot \max_i I^i$.
   When does the EB-Sampler give a tighter bound? Is there a case where uniform is better?
3. The EB-Sampler uses $\hat{I}^i(z_t) = H(p_\theta(x^i \mid z_t))$ which is an estimate
   of $I^i(x)$ conditioned on the *current noisy* $z_t$. Early in generation (high
   masking rate), how good is this estimate? Does $\hat{I}^i$ converge to $I^i(x)$
   as more tokens are committed?
4. The EB-Sampler commits tokens in ascending entropy order. After applying remasking
   (re-masking some of the committed tokens), the ordering of the remaining masked
   tokens may change. Should the EB-Sampler re-sort all masked tokens after each
   remasking step? What is the computational cost of doing so?

\newpage

# Part V — Principled Remasking: From Heuristic to Learned to Optimal

---

## ReMDM (2025) — The First Principled Remasking Posterior

**Full citation:** Wang, Schiff, Sahoo & Kuleshov (2025), "Remasking
Discrete Diffusion Models with Inference-Time Scaling."
arXiv:2503.00307. GitHub: `kuleshov-group/remdm`.

### Summary

ReMDM is the paper that formally introduces remasking as an
inference-time scaling mechanism for MDMs. It derives a principled
*remasking posterior* $\sigma_t$ — the probability of re-masking a
committed token at time $t$ — by analogy with the unmasking posterior
of the standard reverse process. The key theoretical result is that
adding correction passes (remask + re-predict) monotonically improves
generation quality: more correction steps cannot hurt. The method
requires no retraining and is evaluated on MDLM-OWT, with the ReMDM
strategies (`remdm-conf`, `remdm-loop`, etc.) available in the
submodule at `external/remdm`.

### Method Details

**The remasking posterior.** The standard MDM reverse kernel specifies
the probability of unmasking a masked token. ReMDM derives the
complementary: the probability of *re-masking* a committed token at
time $t$. Starting from first principles (what is the marginal
probability that a clean token $x^i$ is masked at time $t$, given that
it was committed at some earlier time?), ReMDM derives:

$$\sigma_t^i = P(z_t^i = m \mid z_0^i = x^i) = 1 - \alpha_t$$

where $\alpha_t$ is the noise schedule evaluated at the current time.
At high $t$ (early in generation), $1 - \alpha_t$ is close to 1, so
remasking is aggressive; at low $t$ (near completion), $1 - \alpha_t$
is close to 0, so remasking is rare.

**Inference algorithm with corrections:**
```
z_T = [m, m, ..., m]
for t = T, T-1, ..., 1:
    # Standard predictor step (unmask)
    z_{t-1} ← sample from p_θ(z_{t-1} | z_t)

    # R corrector steps (remask + re-predict)
    for r = 1, ..., R:
        # Re-mask each committed token independently
        for i with z_{t-1}^i ≠ m:
            z'^i ← m   with probability σ_t^i = 1 - α_t
            z'^i ← z_{t-1}^i   otherwise
        # Re-predict all masked positions
        z_{t-1} ← sample from p_θ(z_{t-1} | z')
return z_0
```

Total forward passes: $T \times (R + 1)$ — one per step plus $R$
correction passes.

**ReMDM strategies.** The paper introduces four strategies for
computing the remasking probability, going beyond the uniform $\sigma_t$:

- **`remdm-conf`** (confidence-weighted): remask token $i$ with
  probability $\sigma_t^i \cdot (1 - c^i)$ where $c^i$ is the
  model's confidence at that position. Low-confidence tokens are
  remasked more aggressively.
- **`remdm-cap`**: remask a fixed number of the lowest-confidence
  committed tokens per step.
- **`remdm-rescale`**: rescale the remasking probability to maintain
  a fixed expected number of remasked tokens.
- **`remdm-loop`** (time-windowed): only apply corrections when
  $t \in [t_{\text{off}}, t_{\text{on}}]$, concentrating corrections
  in the mid-generation window where they are most useful.

### Theoretical Contributions

**Monotone improvement theorem.** Let $p_{\text{ReMDM}}^{(R)}$ be the
distribution over sequences produced by ReMDM with $R$ correction
passes per step. Under the assumption that $p_\theta$ exactly matches
the true score ($E_{\text{learn}} = 0$), ReMDM proves:

$$\mathrm{KL}(\pi \| p_{\text{ReMDM}}^{(R+1)}) \leq \mathrm{KL}(\pi \| p_{\text{ReMDM}}^{(R)}) \quad \forall R \geq 0$$

More correction steps always reduce the KL, monotonically. The proof
uses the data-processing inequality applied to the Markov chain induced
by the correction pass: the corrector can only reduce, not increase,
the KL to the true distribution.

**Inference-time scaling.** As $R \to \infty$, $\mathrm{KL}(\pi \|
p_{\text{ReMDM}}^{(R)}) \to 0$ for any $T \geq 1$. This is the
"inference-time scaling" result: compute (number of forward passes)
is directly tradeable against quality.

**Gap from the thesis.** ReMDM proves that corrections help but does
not characterise *how much* they help per correction step, which
corrections are most efficient (uniform vs. confidence-guided), or how
the improvement relates to $E_{\text{fact}}$ or the information profile.
These are the thesis's contributions.

### Confidence Signal

**Baseline:** uniform remasking with probability $\sigma_t = 1 - \alpha_t$
per committed token, independent of confidence. This is theoretically
grounded (the monotone improvement theorem applies) but wasteful.

**`remdm-conf`:** remask with probability $\sigma_t \cdot (1 - c^i)$
where $c^i = H(p_\theta(x^i \mid z_t))$ (entropy). Empirically superior
but no formal guarantee beyond monotone improvement.

### Limitations

1. **No connection to $E_{\text{fact}}$ or information profile.** The
   monotone improvement theorem is proven, but the *rate* of improvement
   per correction step is not characterised.
2. **Uniform baseline is wasteful.** Re-masking confident committed
   tokens with high probability wastes compute on positions that are
   unlikely to change.
3. **Compute cost is $O(T \times R)$.** For large $R$, this becomes
   prohibitive at 9B scale.
4. **Evaluation limited to language modelling.** LAMBADA and
   generative perplexity are the primary metrics; no instruction-following
   or reasoning benchmarks.

### Relation to This Thesis

ReMDM is the closest existing work to the thesis. The $\sigma_t$
posterior is a special case of the thesis's remasking kernel with
$\tau = 0$ (always remask with probability $1-\alpha_t$). The thesis:
(1) derives an explicit $E_{\text{fact}}$ reduction bound for the
confidence-guided variant (showing *why* it is better, not just that
it is); (2) derives the optimal threshold $\tau$ as a function of the
information profile and entropy budget; (3) provides experiments on
a broader set of models and benchmarks.

### Study Questions

1. ReMDM's remasking probability is $\sigma_t^i = 1 - \alpha_t$. At step $t = T/2$
   (halfway through generation), what is $\sigma_t$? Does remasking get more or
   less aggressive as generation progresses toward completion?
2. The monotone improvement theorem requires $E_{\text{learn}} = 0$ (perfect model).
   In practice, $p_\theta \neq q$. Does remasking still help when $E_{\text{learn}} > 0$?
   Could remasking ever *hurt* quality when the model is imperfect?
3. ReMDM's `remdm-conf` remasks with probability $\sigma_t(1-c^i)$. Compare this to the
   thesis's threshold strategy (remask if $\hat{I}^i > \tau$). Are these equivalent?
   Which one commits to a hard decision and which uses a soft weighting?
4. Our experiments show remdm-conf MAUVE collapses at T=1000 while remdm-loop improves.
   Using the ReMDM framework, what is the remasking probability $\sigma_t$ at T=1000
   near the end of generation ($t \to 0$)? Is this consistent with the collapse hypothesis?

---

## Informed Correctors (2024/2025) — MCMC Theory for Discrete Diffusion Correction

**Full citation:** Zhao et al. (2024/2025), "Informed Correctors for
Discrete Diffusion Models."

### Summary

Informed Correctors applies Markov Chain Monte Carlo (MCMC) theory to
the correction problem for discrete diffusion models. The approach is
Gibbs-based: at each correction step, one position is selected,
re-masked, and re-predicted conditioned on all other positions. The
*informed* variant selects positions in order of decreasing *surprise*
(negative log-probability), so that the most uncertain tokens are
corrected first.

The paper provides the strongest theoretical correction guarantee in the
literature: an exponential mixing time bound showing that the KL between
the corrected distribution and the target decays geometrically in the
number of correction steps. It also proves that the informed (confidence-guided)
corrector achieves a larger spectral gap than the uniform corrector —
i.e., confidence-guided correction is strictly more efficient.

The key limitation for the thesis is that the exact informed corrector
requires a *hollow transformer* — a modified attention mechanism that
computes all per-position conditionals $q(z^d \mid z^{\setminus d})$
in a single forward pass. This requires architectural changes and
retraining, making it incompatible with standard pretrained MDMs.

### Method Details

**Gibbs corrector.** At each correction step, select a position $d$
uniformly at random and resample:

$$z^d_{\text{new}} \sim q_t(z^d \mid z^{\setminus d})$$

This is exact Gibbs sampling from $q_t$ — the model's joint
distribution at noise level $t$. Repeated application converges to
$q_t$ regardless of initialisation.

**Informed corrector.** Instead of selecting $d$ uniformly, select with
probability proportional to the *surprise* at each position:

$$P(\text{select } d) \propto \exp\!\left(\beta \cdot \left[-\log q_t(z^d \mid z^{\setminus d})\right]\right)$$

for temperature $\beta > 0$. High surprise (low log-probability $\to$
high $-\log q_t$) $\to$ high selection probability $\to$ more likely
to be corrected. At $\beta = 0$ this reduces to the uniform corrector;
as $\beta \to \infty$ it always selects the most surprised position.

**Hollow transformer.** Computing $q_t(z^d \mid z^{\setminus d})$
exactly for all $d$ simultaneously requires one forward pass with each
position masked in turn — $O(L)$ passes for a sequence of length $L$.
The *hollow transformer* avoids this by using a modified attention mask
that zeroes the diagonal (position $i$ cannot attend to itself),
enabling exact computation of all per-position conditionals
$\{q_t(z^d \mid z^{\setminus d})\}_{d=1}^L$ in a single forward pass.
This requires retraining.

**Full algorithm:**
```
# Phase 1: standard MDM sampling
z_0 ← standard_MDM_sampler(z_T)

# Phase 2: correction loop
for r = 1, ..., R:
    for d = 1, ..., L:   (or batched)
        # Compute surprise (needs hollow transformer for O(1) cost)
        surprise^d = -log q_t(z_0^d | z_0^{∖d})
    # Select position to correct, proportional to exp(β · surprise)
    d* ← sample ∝ exp(β · surprise)
    # Resample
    z_0^{d*} ← sample from q_t(z_0^{d*} | z_0^{∖d*})
return z_0
```

### Theoretical Contributions

**Mixing time bound (Theorem 3).** Let $\lambda > 0$ be the spectral
gap of the Gibbs chain. After $R$ correction steps:

$$\mathrm{KL}(q_0 \| p_R) \leq e^{-R\lambda} \cdot \mathrm{KL}(q_0 \| p_0)$$

where $p_0$ is the distribution at the start of correction and $p_R$
is the distribution after $R$ steps. The KL decays exponentially in $R$.

**Informed corrector acceleration (Theorem 4).** Let $\lambda_u$ be the
spectral gap for the uniform corrector and $\lambda_i(\beta)$ for the
informed corrector at temperature $\beta$:

$$\lambda_i(\beta) \geq \lambda_u \quad \forall \beta \geq 0$$

with equality only when all positions have equal surprise. The informed
corrector achieves the same KL reduction as the uniform corrector but
in fewer steps. The gap $\lambda_i - \lambda_u$ grows with the
variance of surprises across positions — exactly when the information
profile is non-uniform, which is when the thesis predicts the largest
benefit from confidence-guided remasking.

### Confidence Signal

$-\log q_t(z^d \mid z^{\setminus d})$ — the negative log-probability
of the current token at position $d$, given all other positions. This
is the *exact* per-position conditional, the Tier 1 signal.

**Computational cost without hollow transformer:** $O(L)$ forward
passes per correction step — prohibitive for long sequences.

**With hollow transformer:** $O(1)$ forward passes — efficient, but
requires retraining the model with hollow attention.

**Approximation for standard MDMs:** Use the model's output probability
$-\log p_\theta(z^d = z_0^d \mid z_0)$ (the negative log-probability
that the current committed token $z_0^d$ is the correct prediction)
as a proxy. This is the Tier 3 margin signal.

### Limitations

1. **Requires hollow transformer.** Not applicable to MDLM-OWT, LLaDA-8B,
   RemeDi, or any other standard pretrained MDM without retraining.
2. **Post-hoc correction only.** Informed Correctors is applied after
   standard sampling is complete, not integrated into the denoising loop.
   This means it cannot correct errors made by early unmasking decisions
   before those decisions propagate to later steps.
3. **Spectral gap $\lambda$ can be very small.** For sequences with
   strong long-range dependencies (e.g., code), the Gibbs chain can
   mix very slowly, requiring an impractical number of correction steps.
4. **No connection to information profile $I(x)$ or $E_{\text{fact}}$.**
   The MCMC framework and the Riemann error framework are parallel but
   have not been formally connected.

### Relation to This Thesis

Informed Correctors provides MCMC-theoretic corroboration for the
thesis's main claim: Theorem 4 proves that confidence-guided correction
is strictly more efficient than uniform correction, from a completely
different theoretical perspective (spectral gap vs. Riemann error). The
thesis can use this as supporting evidence.

The key design difference from the thesis: Informed Correctors requires
exact per-position conditionals (Tier 1 signal, hollow transformer),
while the thesis uses only forward-pass logits (Tier 3 signal, any
pretrained MDM). The thesis's contribution is to show that Tier 3
signals are *sufficient* for the efficiency gain — you do not need the
exact conditional to benefit from confidence-guided correction.

### Study Questions

1. The informed corrector selects position $d^* \propto \exp(\beta \cdot [-\log q_t(z^d \mid z^{\setminus d})])$.
   At $\beta = 0$ this is uniform; at $\beta \to \infty$ it always picks the most
   surprised position. What is the optimal $\beta$? How would you tune it?
2. The hollow transformer computes all $L$ per-position conditionals in a single
   forward pass by zeroing the attention diagonal. Why does zeroing the diagonal
   give $q_t(z^d \mid z^{\setminus d})$? Is this exact or approximate?
3. The spectral gap $\lambda_i \geq \lambda_u$ (informed $\geq$ uniform). The gap
   grows with the variance of surprises $\mathrm{Var}_d[-\log q_t^d]$. How does this
   variance relate to the information profile variance $\Sigma^2$?
4. Informed Correctors applies correction *after* the full generation is done.
   The thesis integrates remasking *into* the generation loop. What is the qualitative
   difference? Can you think of a scenario where post-hoc correction is better?

---

## RemeDi (2025) — Learned Policy for Unmask and Remask Decisions

**Full citation:** Huang et al. (2025), "Don't Settle Too Early:
Self-Reflective Remasking for Diffusion Language Models."
arXiv:2509.23653. HuggingFace: `maple-research-lab/RemeDi-RL`,
`maple-research-lab/RemeDi-Instruct`.

### Summary

RemeDi introduces a fundamentally different approach to remasking: rather
than using a heuristic or analytically derived confidence signal, it
*learns* when to unmask and when to remask through a dedicated policy
network (the Unmasking Policy Stream, UPS). The base token predictor
(Token Prediction Stream, TPS) is unchanged; the UPS is a lightweight
MLP trained on top of TPS hidden states, first with supervised learning
(BCE on whether the current prediction is correct) and then refined with
Group Relative Policy Optimisation (GRPO) reinforcement learning on a
generation quality reward.

RemeDi represents the pinnacle of the fine-tuning approach: it has access
to training signal not available at inference time and can learn globally
optimal unmask/remask decisions that no training-free signal can match
in principle. However, it requires a full training pipeline (TPS
pre-training + UPS BCE training + GRPO RL) and the resulting policy is
not portable to other model families.

### Method Details

**Architecture.** TPS is a standard bidirectional transformer decoder
trained with the MDM ELBO. UPS is a lightweight MLP applied to the TPS
hidden states $h^i$ at each position:

$$\psi^i = \sigma(W_3 \cdot \mathrm{ReLU}(W_2 \cdot \mathrm{ReLU}(W_1 \cdot h^i + b_1) + b_2) + b_3) \in [0, 1]$$

$\psi^i$ is the UPS's estimate of "should token $i$ be committed at
this step?" — distinct from the TPS's estimate of "what token should
be placed at position $i$?". This separation is the key architectural
insight: the *what* and the *when* are different problems that can be
learned independently.

**Stage 1 — TPS pre-training.** Standard MDLM ELBO training on a large
text corpus. Produces a 9B-parameter bidirectional transformer competitive
with LLaDA.

**Stage 2 — UPS supervised pre-training.** Given a training sequence $x$
and a randomly masked version $z_t$ (simulating a mid-generation state),
train UPS to predict whether the TPS's top-1 prediction is correct:

$$y^i = \mathbf{1}[\hat{x}^i = x^i] \quad \text{where } \hat{x}^i = \arg\max_j p_\theta(x^i = j \mid z_t)$$

$$\mathcal{L}_\text{UPS} = -\sum_i \left[y^i \log \psi^i + (1-y^i) \log(1-\psi^i)\right]$$

After this stage, $\psi^i$ is a calibrated estimator of the TPS's
per-token accuracy. High $\psi^i$ means the TPS's prediction at
position $i$ is likely correct; low $\psi^i$ means it is likely wrong
and the position should be remasked.

The paper proves (analogously to PRISM) that the BCE minimiser satisfies:

$$\psi^{i*} = P\!\left(\hat{x}^i = x^i \;\middle|\; z_t\right)$$

i.e., UPS converges to the true conditional accuracy of the TPS.

**Stage 3 — GRPO RL fine-tuning.** The UPS is further refined by
Group Relative Policy Optimisation with a reward signal $r$ based on the
quality of the fully generated sequence. GRPO compares multiple
completions from the same prefix and updates the policy towards
higher-reward ones, without requiring a separate reward model.

The RL stage teaches UPS to make *globally optimal* decisions — not just
locally correct ones (which is what BCE training achieves). For example,
BCE training might teach UPS to remask a locally uncertain token, but
GRPO might discover that remasking that token always leads to worse
final sequences (because the correction introduces a cascade of errors
elsewhere).

**Inference algorithm:**
```
z_T = [m, m, ..., m]
for t = T, T-1, ..., 1:
    # Single forward pass through TPS+UPS
    p^i, ψ^i ← TPS+UPS(z_t)   for all positions i

    # Unmask confident masked positions
    for i with z_t^i = m and ψ^i > τ_unmask:
        z_{t-1}^i ← argmax_j p^i_j   (or sample from p^i)

    # Remask uncertain committed positions
    for i with z_t^i ≠ m and ψ^i < τ_remask:
        z_{t-1}^i ← m

return z_0
```

The thresholds $\tau_\text{unmask}$ and $\tau_\text{remask}$ are
hyperparameters (typically set to 0.5).

**Stored decoding probability.** RemeDi additionally stores $\psi^i(t^*)$
— the UPS score at the time token $i$ was originally committed. In
subsequent steps, if $\psi^i$ drops significantly below $\psi^i(t^*)$,
the token's confidence has deteriorated and it is a strong candidate for
remasking. This gives RemeDi a form of memory across steps.

### Theoretical Contributions

1. **UPS BCE minimiser converges to $P(\hat{x}^i = x^i \mid z_t)$.**
   Standard proper-scoring-rule argument (the BCE loss is a strictly
   proper scoring rule for probability estimation).
2. **GRPO convergence.** Standard GRPO convergence results apply: the
   RL fine-tuning converges to a local optimum of the expected reward.
   No global optimality guarantee.
3. **No sampling error bound.** RemeDi does not connect to $E_{\text{fact}}$
   or the information profile.

### Confidence Signal

$\psi^i \in [0,1]$ — a learned, calibrated estimate of
$P(\text{TPS prediction at position } i \text{ is correct} \mid z_t)$.

**Properties:**
- Calibrated across noise levels (BCE training on diverse $t$).
- Accounts for global context via the TPS hidden state $h^i$.
- Refined towards globally optimal decisions by GRPO.
- **Not portable:** must be retrained for each new base model.

### Limitations

1. **Three-stage training pipeline.** Requires TPS pre-training + UPS
   BCE + GRPO RL. The RL stage in particular is sensitive to
   hyperparameters and can be unstable at 9B scale.
2. **Not portable.** UPS is trained for a specific TPS architecture and
   corpus; it cannot be transferred to MDLM-OWT or LLaDA without
   retraining.
3. **No connection to optimal sampling theory.** The learned policy is
   not shown to be optimal in any theoretical sense; it is empirically
   motivated by the RL reward.
4. **Evaluation scope.** RemeDi is evaluated primarily on
   instruction-following tasks; unconditional quality metrics
   (MAUVE, generative perplexity) are underreported.

### Relation to This Thesis

RemeDi is the empirical upper bound for the thesis. The RL-trained UPS
can in principle learn any remasking policy, while the thesis is
constrained to training-free signals. The thesis's key empirical
question is: how much of RemeDi's quality advantage can be recovered by
the thesis's principled training-free strategy? If the gap is small,
training is unnecessary; if large, the gap is explained by the
difference between Tier 2 (learned) and Tier 3 (heuristic) signals —
which is itself a contribution.

### Study Questions

1. RemeDi's UPS is trained with BCE on whether the TPS prediction is correct.
   The BCE minimiser converges to $P(\hat{x}^i = x^i \mid z_t)$. Is this the same
   as $I^i(x)$? When do they agree and when do they diverge?
2. RemeDi uses GRPO RL to refine the UPS beyond BCE training. What does GRPO
   add that BCE training cannot provide? Give an example of a decision that would
   be suboptimal under BCE but optimal under GRPO.
3. The RemeDi-RL model on HuggingFace is 8B parameters with an 8B tokenizer.
   Our MDLM-OWT baseline is 130M. If you wanted to compare RemeDi's remasking
   strategy (not model scale) to the thesis's strategy, what experimental design
   would be fair?
4. RemeDi stores $\psi^i(t^*)$ — the UPS score at the time token $i$ was committed.
   This gives it memory across steps. Does the thesis's entropy-based remasking
   have an implicit memory? What would it mean to add explicit memory to a Tier 3 strategy?

---

## PRISM (2025) — Provable Quality Head with a Formal Guarantee

**Full citation:** Kim et al. (2025), "Fine-Tuning Masked Diffusion for
Provable Self-Correction." arXiv:2510.01384. GitHub:
`SeunggeunKimkr/PRISM`.

### Summary

PRISM trains a lightweight MLP quality head $g_\phi$ on top of a frozen
pretrained MDM to predict, for each token position, the probability that
the model's current prediction is correct. This is the simplest possible
approach to learning a confidence signal: the base model is completely
unchanged, and only a small MLP (a few thousand parameters) is trained.

PRISM's theoretical contribution is a formal convergence guarantee: the
BCE minimiser of $g_\phi$'s training objective converges to the true
Bayes-optimal conditional accuracy $P(x^i = \hat{x}^i \mid y \oplus m^i)$,
where $y \oplus m^i$ is the input sequence with position $i$ additionally
masked. This is the strongest theoretical result for any confidence signal
in the MDM remasking literature.

### Method Details

**Architecture.** $g_\phi$ is a lightweight MLP (2–3 hidden layers,
ReLU activations) applied to the frozen hidden states $h^i$ of the base
MDM:

$$g_\phi(y)^i = \sigma\!\left(W_k \mathrm{ReLU}(\cdots W_2 \mathrm{ReLU}(W_1 h^i + b_1) + b_2 \cdots) + b_k\right) \in [0,1]$$

The base model $p_\theta$ is completely frozen; only $\{W_1, \ldots,
W_k, b_1, \ldots, b_k\}$ are trained.

**Training objective.** For a training pair $(x, y)$ where $y$ is a
partially masked version of $x$ and $\hat{x}^i = \arg\max_j p_\theta(x^i
= j \mid y)$ is the base model's top-1 prediction:

$$\mathcal{L}(\phi) = -\mathbb{E}_{x,y}\!\left[\sum_i \mathbf{1}[x^i=\hat{x}^i]\log g_\phi(y)^i + \mathbf{1}[x^i\neq\hat{x}^i]\log(1-g_\phi(y)^i)\right]$$

This is a standard binary cross-entropy loss where the label is 1 if
the base model's prediction is correct and 0 otherwise.

**Theoretical guarantee (Proposition 1 of PRISM).** The minimiser
$\phi^*$ over the class of all measurable functions satisfies:

$$g_{\phi^*}(y)^i = P\!\left(x^i = \hat{x}^i \;\middle|\; y \oplus m^i\right)$$

where $y \oplus m^i$ is the sequence $y$ with position $i$ additionally
masked. This is the Bayes-optimal prediction of the base model's accuracy
at position $i$, given the information available in $y$ minus position
$i$ itself.

*Proof sketch.* The BCE loss is a strictly proper scoring rule: for any
event $A$, $\mathbb{E}[-y_A \log q - (1-y_A)\log(1-q)]$ is minimised at
$q = P(A)$. Applying this to $A = \{x^i = \hat{x}^i\}$ with the
conditioning variable being all information except $y^i$ (hence
$y \oplus m^i$) gives the result.

**Inference algorithm.** At generation step $t$:
```
# One pass through frozen base model: get logits and hidden states
p^i, h^i ← p_θ(z_t)   for all i
# One pass through quality head: get confidence scores
g^i ← g_φ(h^i)   for all i (free: just an MLP)
# Unmask top-K most confident masked positions
unmask_set ← top-K {i : z_t^i = m} by g^i
z_{t-1}^i ← argmax p^i   for i in unmask_set
# Optionally remask low-confidence committed positions
for i with z_t^i ≠ m and g^i < τ:
    z_{t-1}^i ← m
```

**Evaluation.** PRISM is evaluated on three tasks:
- **Sudoku:** discrete constraint satisfaction — PRISM dramatically
  improves completion accuracy.
- **MDLM-OWT (170M):** unconditional text generation — improves
  generative perplexity and MAUVE.
  *(Formal definitions of generative perplexity and MAUVE: Appendix A.8.1–A.8.2.)*
- **LLaDA-8B (MBPP code):** code generation — improves pass@1 on the
  MBPP benchmark, demonstrating scalability.

### Theoretical Contributions

1. **Bayes-optimality of the BCE minimiser.** The formal convergence
   result is the strongest guarantee for any confidence signal in the
   MDM remasking literature. It shows that $g_\phi$ is not just
   a heuristic but the optimal predictor of base model accuracy given
   the available information.
2. **Connection to PRISM's name.** The quality head predicts whether
   the model's prediction "passes" (is correct) — hence Provable
   Self-correction via Inference-time Self-Monitoring (PRISM).

### Confidence Signal

$g_\phi(y)^i \in [0,1]$ — the Bayes-optimal estimate of
$P(\hat{x}^i = x^i \mid y \oplus m^i)$.

**Properties:**
- Theoretically optimal under the BCE minimisation criterion.
- Calibrated across noise levels (training covers all $t$).
- Negligible computational overhead (tiny MLP on frozen hidden states).
- **Requires training:** a new $g_\phi$ for each base model.

### Limitations

1. **Training required.** A new $g_\phi$ must be trained for each base
   model. Requires labelled data (parallel clean/masked sequences) and
   compute.
2. **LLaDA-8B adapter not public.** As of early 2026, the quality head
   trained on LLaDA-8B is available only to lab collaborators.
3. **Guarantee is for calibration, not sampling quality.** Theorem 1
   proves that $g_\phi$ converges to the true conditional accuracy.
   It does not prove that using $g_\phi$ as a sampling guide produces
   sequences from $\pi$ or reduces $E_{\text{fact}}$.
4. **OOD calibration.** $g_\phi$ is trained on a specific corpus;
   calibration may degrade for prompts far from the training distribution.

### Relation to This Thesis

PRISM's BCE convergence theorem provides the formal justification for
the thesis's central assumption. The thesis assumes that a training-free
signal $c^i$ is a *consistent estimator* of $H(x^i \mid \text{context})$
(Tier 3 signal behaves like Tier 1). PRISM shows that with a small
amount of training, exact convergence to the true conditional is
achievable (Tier 2). The thesis bridges the gap: it provides formal
conditions on $c^i$ (consistency) under which the Tier 3 signal is
*sufficient* to achieve an $E_{\text{fact}}$ reduction comparable to
what PRISM would achieve. If these conditions hold for
max-prob/entropy/margin in practice (which the experiments test), then
retraining is unnecessary.

### Study Questions

1. PRISM's Proposition 1 says $g_{\phi^*}^i = P(x^i = \hat{x}^i \mid y \oplus m^i)$.
   The conditioning variable is $y \oplus m^i$ (sequence with position $i$ also masked).
   Why is it conditioned on $y \oplus m^i$ and not on $y$ itself?
2. PRISM dramatically improves Sudoku completion accuracy. Why would a discrete
   diffusion model with PRISM be particularly effective on constraint satisfaction
   tasks? What does the confidence score capture that matters for Sudoku?
3. PRISM's quality head $g_\phi$ is a tiny MLP on frozen hidden states. Could you
   use a different architecture (e.g., a linear probe, or attention over all positions)?
   What would be gained or lost?
4. PRISM and RemeDi both use BCE loss and both converge to the true conditional.
   The difference is that RemeDi also adds RL fine-tuning. Suppose PRISM also added
   an RL stage on top of $g_\phi$. Would this give the same result as RemeDi?
   What would be different?

\newpage

# Part VI — Cross-Paper Comparison

## Chronological Overview and Intellectual Lineage

The twelve papers form two parallel development threads that the thesis
unifies:

**Thread A — Non-autoregressive generation and remasking practice:**
$$\text{Mask-Predict (2019)} \to \text{MDLM (2024)} \to \text{LLaDA (2025)} \to \text{ReMDM (2025)} \to \text{RemeDi (2025), PRISM (2025)}$$

This thread develops increasingly powerful remasking strategies, starting
from the pure heuristic (Mask-Predict) and progressing to learned
approaches (PRISM, RemeDi). Quality improves at each step, but theoretical
understanding lags behind.

**Thread B — Discrete diffusion theory:**
$$\text{D3PM (2021)} \to \text{SEDD (2024)} \to \text{DFM (2024)} \to \text{Lavenant \& Zanella (2025)} \to \text{EB-Sampler (2025)}$$

This thread develops the theoretical framework for understanding and
improving MDM sampling, culminating in the optimal unmasking schedule.
It is rigorous but stops short of the remasking direction.

**The thesis.** Sits at the confluence of both threads: it brings the
rigour of Thread B to the practice of Thread A. The main theorem
characterises confidence-guided remasking within the Lavenant & Zanella
$E_{\text{fact}}$ framework, connecting the optimal Tier 3 signal
(entropy, from EB-Sampler) to the convergence guarantees of Thread B.

---

## Comprehensive Comparison Table

| # | Paper | Year | Backbone | Confidence Signal | Tier | Training? | Remasking? | Theoretical Guarantee | Key Metric | Dataset | Approx. Params |
|---|---|:---:|---|---|:---:|:---:|:---:|---|---|---|---|
| 1 | Mask-Predict | 2019 | BERT-scale | Max-prob | 3 | Pre-train | Yes (heuristic) | None | BLEU | WMT | ~200M |
| 2 | D3PM | 2021 | Custom | None | — | Pre-train | No | VLB decomp. | NLL | Text8 | ~100M |
| 3 | SEDD | 2024 | GPT-2 scale | Score ratio | 1 | Pre-train | PC corrector | Consistency + PC conv. | Perplexity | OWT | ~300M |
| 4 | MD4 / MDLM | 2024 | Custom | None | — | Pre-train | No | ELBO simplif. | Perplexity | OWT, LM1B | 130M |
| 5 | DFM | 2024 | Any MDM | Velocity entropy | 3 | Pre-train | Corrector | Path-integral KL | Perplexity | OWT, audio | Any |
| 6 | Lav. & Zan. | 2025 | Any MDM | N/A (analysis) | — | No | Open problem | KL = $E_l + E_f$ | N/A | N/A | — |
| 7 | EB-Sampler | 2025 | Any MDM | Entropy $H(p_\theta)$ | 3 | No | No | $E_f$ Riemann bd. | PPL, speed | OWT | Any |
| 8 | ReMDM | 2025 | MDLM-OWT | Entropy (opt.) | 3 | No | Yes (principled) | KL monotone in $R$ | Perplexity | OWT | 130M |
| 9 | Inf. Corr. | 2025 | Hollow transf. | Log-margin (exact) | 1 | Retrain | Yes (MCMC) | MCMC mixing time | PPL, diversity | Text8, OWT | Custom |
| 10 | LLaDA | 2025 | 8B transformer | None | — | Pre-train | Uniform | None (empirical) | Benchmarks | 2.3T tokens | 8B |
| 11 | RemeDi | 2025 | TPS+UPS (9B) | Learned $\psi^i$ | 2 | Yes (RL) | Yes (learned) | UPS BCE conv. | Instruction acc. | Custom | 9B |
| 12 | PRISM | 2025 | LLaDA-8B | Learned $g_\phi$ | 2 | Yes (head) | Yes (guided) | BCE → true cond. | Gen. quality | OWT, MBPP | 8B + MLP |

*Tier: 1 = exact posterior ratio; 2 = learned approximation; 3 = heuristic training-free.*

---

## Dimension 1: Training Requirements

| Approach | Papers | Training required | Portable to new model? |
|---|---|---|---|
| Pre-train only (no special remasking) | D3PM, MDLM, LLaDA | Base model pre-training only | Yes |
| Training-free remasking | Mask-Predict, ReMDM, EB-Sampler, DFM, **Thesis** | None beyond pre-training | Yes |
| Fine-tuned head | PRISM | Small MLP on frozen model | No (per-model) |
| Fine-tuned policy (RL) | RemeDi | Full TPS+UPS+RL pipeline | No (per-model) |
| Architecture-retrained | Informed Correctors, SEDD | Full retraining with hollow attention or score objective | No (per-architecture) |

The thesis occupies the training-free cell: it works with any pretrained
MDM (MDLM-OWT, LLaDA-8B, RemeDi's TPS) using only the forward-pass
logits. Its portability is its primary practical advantage over PRISM
and RemeDi, and is what makes it relevant for any newly released MDM
backbone.

---

## Dimension 2: Theoretical Framework and Guarantee Strength

| Paper | Framework | What is proved | Quantitative bound? |
|---|---|---|---|
| D3PM | Markov chain VLB | VLB decomposition | Yes (VLB lower bound) |
| SEDD | Score entropy | Score consistency + PC convergence | Yes (convergence rate) |
| DFM | Flow matching / probability paths | Objective equivalence, optimal path | Partial |
| Lavenant & Zanella | Riemann approximation of $I(x)$ | KL = $E_l + E_f$; optimal schedule | Yes (explicit bound) |
| EB-Sampler | Riemann approximation of $I(x)$ | $E_f$ minimisation; optimality | Yes (explicit bound) |
| ReMDM | Markov chain monotonicity | KL monotone in $R$ | Yes (monotonicity) |
| Informed Correctors | MCMC spectral gap | Exponential KL decay; informed > uniform | Yes (spectral bound) |
| PRISM | Proper scoring rules | BCE minimiser = true conditional | Yes (Bayes-optimality) |
| Mask-Predict, LLaDA | None | None | No |
| RemeDi | Proper scoring rules | UPS BCE → true conditional | Yes (for calibration only) |

The thesis targets a result in the Lavenant & Zanella framework:
a quantitative bound on the $E_{\text{fact}}$ reduction achieved by
confidence-guided remasking. This places it in the strongest theoretical
quadrant (quantitative, information-theoretic) and is the gap no
existing paper fills.

---

## Dimension 3: The Confidence Signal Hierarchy

All remasking papers must answer: *which tokens should be remasked?*
Their answers span three tiers:

**Tier 1 — True posterior (exact, expensive).**
$I^i(x) = H(x^i \mid x^{\setminus i})$ — the conditional entropy of
position $i$ given all other positions. This is the gold standard:
it exactly measures how much each position would benefit from being
re-predicted in perfect context. Computing it requires either knowing
$x$ (unavailable) or $O(L)$ forward passes (expensive). Informed
Correctors and SEDD operate at this tier (with the hollow transformer
approximation).

**Tier 2 — Learned approximation (calibrated, requires training).**
PRISM's $g_\phi^i$ and RemeDi's $\psi^i$ approximate Tier 1 using a
learned model. Both come with formal convergence guarantees (BCE
minimiser = true conditional). The trade-off: they are calibrated and
accurate but require training data and compute for each new backbone.

**Tier 3 — Heuristic training-free (cheap, approximate, no guarantee).**
Three signals computed directly from $p_\theta(x^i \mid z_t)$:
- **Max-probability:** $c^i = \max_j p_\theta^j$. Used by Mask-Predict.
  Intuitively: how confident is the model? But max-probability conflates
  unimodal (easy) with multi-modal (hard) distributions.
- **Entropy:** $c^i = -H(p_\theta(x^i \mid z_t))$. Used by ReMDM and
  EB-Sampler. Directly approximates $I^i(x)$; theoretically the
  most principled of the three.
- **Margin:** $c^i = p_\theta^{(1)} - p_\theta^{(2)}$. The gap between
  the top-2 predictions. Sensitive to multi-modal uncertainty that
  entropy may miss.

The thesis provides the first formal analysis of when and why Tier 3
signals are sufficient: under a consistency condition (the signal is
an unbiased estimator of $I^i(x)$ in expectation), the $E_{\text{fact}}$
reduction of the confidence-guided corrector equals that of the
Tier 1 signal up to a variance term.

---

## Dimension 4: Inference Compute Cost

| Paper | NFE per token | Notes |
|---|---|---|
| MDLM ($T$ steps) | $T$ total, each unmasks $L/T$ tokens | $T = 64$–$1024$ typically |
| EB-Sampler | $\approx T$ (adaptive) | $T \approx I(x)/\varepsilon$; often fewer than fixed-$T$ MDLM |
| ReMDM ($T$ steps, $R$ corrections) | $T \times (R+1)$ | Linear in $R$; $R = 1$–$10$ practical |
| PRISM | $T$ + negligible (MLP) | Quality head overhead $\ll$ base model |
| RemeDi | $T$ (single pass per step) | UPS adds $\sim 0.1\%$ overhead |
| Informed Correctors | $T + R \times L$ (without hollow transf.) | $O(L)$ per correction step; impractical |
| Informed Correctors (hollow) | $T + R$ | With hollow transformer: same as ReMDM |
| **Thesis (threshold/top-k)** | $T \times (1 + r)$, $r \leq 0.2$ typically | Only high-entropy tokens are remasked |

The thesis's strategies have a compute cost between standard MDLM
($r = 0$) and ReMDM ($r = R$). For small $\tau$ (tight threshold), few
tokens are remasked and $r \approx 0.05$–$0.1$, giving a 5–10% overhead.
This is the most compute-efficient remasking strategy in the comparison.

---

## Dimension 5: Empirical Results and Evaluation Scope

| Paper | Metric | Best result | Scale |
|---|---|---|---|
| MDLM | Perplexity (OWT) | ~26–27 | 130M |
| SEDD | Perplexity (OWT) | ~25 | 300M |
| LLaDA | MMLU, GSM8K, etc. | Competitive with LLaMA 3 8B | 8B |
| ReMDM | Gen. perplexity (OWT) | 15–20% reduction vs. MDLM | 130M |
| RemeDi | Instruction acc. | SotA among MDMs | 9B |
| PRISM | MBPP pass@1 | +5–10% vs. base LLaDA | 8B |
| Informed Corr. | PPL + diversity | +10% vs. no correction | 150M (hollow) |
| EB-Sampler | PPL at fixed NFE | 2–3× speedup or +5% PPL | 130M |

The thesis should target: (1) matching or exceeding ReMDM's 15–20%
generative perplexity reduction on MDLM-OWT with lower compute
overhead; (2) demonstrating a 5–10% improvement on LLaDA-8B on
downstream tasks (LAMBADA, HellaSwag); (3) showing the empirical
benefit scales with the variance of the information profile $\Sigma^2$
(non-uniform sequences benefit more).

---

## Dimension 6: Connection to the Information Profile $I(x)$

| Paper | Uses $I(x)$? | How? |
|---|---|---|
| D3PM, MDLM, LLaDA | No | Uniform random unmasking |
| Mask-Predict | Implicitly | Max-prob as proxy for $I^i$ |
| SEDD | Implicitly | Score entropy $\approx$ log-ratio proxy |
| DFM | Implicitly | Velocity entropy = $H(p_\theta^i)$ |
| Lavenant & Zanella | **Explicitly** | $E_{\text{fact}}$ = Riemann error of $\int I(x)\, dt$ |
| EB-Sampler | **Explicitly** | Unmask in ascending order of $\hat{I}^i$ |
| ReMDM | No | Remasking prob. based on $\alpha_t$, not $I^i$ |
| Informed Correctors | No | Surprise signal $\neq I^i$ (it is $-\log q_t^d$) |
| RemeDi, PRISM | No | Accuracy proxy, not information proxy |
| **Thesis** | **Explicitly** | Remask in descending order of $\hat{I}^i$; derive $E_f$ reduction |

The thesis is the only paper that connects *remasking* to the
information profile. Lavenant & Zanella and EB-Sampler connect
*unmasking* to $I(x)$; the thesis completes the picture.

---

## The Open Gap: Where the Thesis Fits

After reviewing all twelve papers, the landscape of MDM sampling theory
has one conspicuous gap:

| Direction | Optimal algorithm | Theoretical bound | Confidence signal analysis |
|---|---|---|---|
| **Unmasking** | EB-Sampler (2025) | Lavenant & Zanella (2025) | Implied by $I^i(x) \approx H(p_\theta^i)$ |
| **Remasking** | ??? | ??? | ??? |

The thesis fills all three cells of the remasking row. Its main result:

> **Theorem (thesis, informal).** Let $\mathcal{A}_\tau$ be the MDM
> sampler that, at each step, (1) unmasks tokens in ascending order of
> $\hat{I}^i$ (EB-Sampler rule) and (2) remasks committed tokens with
> $\hat{I}^i > \tau$ (confidence-guided corrector). If $\hat{I}^i$ is
> a consistent estimator of $I^i(x)$ in the sense that
> $\mathbb{E}[\hat{I}^i \mid x] = I^i(x) + \delta$ with $|\delta| \leq
> \epsilon$, then:
> $$E_{\text{fact}}(\mathcal{A}_\tau) \leq E_{\text{fact}}(\text{EB-Sampler}) - \Delta(\tau, I, \epsilon)$$
> where $\Delta(\tau, I, \epsilon) > 0$ whenever the information
> profile is non-uniform ($\Sigma^2 > 0$), the threshold $\tau$ is chosen
> to target the top-$k$ fraction of positions by $I^i$, and $\epsilon$
> is small relative to $I(x)/L$.

This result:
- **Extends Lavenant & Zanella** to the remasking direction.
- **Explains ReMDM's empirical success** (entropy-weighted remasking
  reduces $E_{\text{fact}}$) and the gap over uniform remasking
  (uniform ignores $I^i$, so $\Delta = 0$ for the uniform corrector).
- **Provides a formal justification for PRISM and RemeDi**: their
  learned signals are consistent estimators of $I^i$ by design (BCE
  minimiser = true conditional); the thesis's condition is automatically
  satisfied.
- **Connects all six remasking papers** (Mask-Predict, ReMDM, Informed
  Correctors, RemeDi, PRISM, EB-Sampler) within a single framework.

\newpage

# Part VII — Research Directions: Principled Rescheduling

The term *rescheduling* refers to the combined problem of deciding, at each
inference step: (a) *which* tokens to unmask, (b) *how many*, and (c) *which
committed tokens to remask*. The EB-Sampler solves (a) and (b); the thesis adds
(c). But the full rescheduling problem has many open directions. This part surveys
them systematically.

**Context from our experiments.** Our step sweep (T ∈ {128, 256, 512, 1000}) reveals
phenomena that motivate each direction:
- MDLM MAUVE peaks at T=256 then drops → diversity window; optimal T is not ∞
- remdm-conf MAUVE collapses at high T → confidence overconfidence problem
- remdm-loop MAUVE improves monotonically → loop remasking is robust but slow
- gen_ppl and MAUVE decouple sharply → mode-seeking vs. coverage tension
  *(formal definitions and the gen_ppl decomposition into $H(p_\theta) + D_\text{KL}(p_\theta \| q_\text{GPT-2})$: Appendix A.8.1–A.8.2)*

---

## R1: Unified EB-Sampler + Confidence-Guided Remasking

### The Gap

The EB-Sampler (optimal unmasking schedule) and ReMDM (principled remasking) were
developed independently and have never been combined. Their interaction is unknown:
does entropy-budget unmasking *amplify* or *suppress* the benefit of remasking?

### Formal Setup

Let $\mathcal{A}_{\varepsilon, \tau}$ be the sampler that: (1) uses the EB-Sampler
with budget $\varepsilon$ for unmasking, and (2) remasks committed tokens with
$\hat{I}^i > \tau$ after each unmasking step. Define:

$$E_{\text{fact}}(\mathcal{A}_{\varepsilon, \tau}) = E_{\text{fact}}^{\text{unmask}}(\varepsilon) + E_{\text{fact}}^{\text{remask}}(\tau) + E_{\text{interaction}}(\varepsilon, \tau)$$

The first two terms are bounded by the EB-Sampler theorem and the thesis theorem,
respectively. The interaction term $E_{\text{interaction}}$ is unknown.

**Hypothesis.** The interaction is *negative* (beneficial): after remasking corrects
high-entropy committed tokens, the EB-Sampler's entropy estimates become more accurate
for the next unmasking step, reducing $E_{\text{fact}}^{\text{unmask}}$ below its
standalone value. Remasking and unmasking are *synergistic*.

### Key Questions

- What is the optimal relationship between $\varepsilon$ and $\tau$?
  - Proposal: $\tau = \varepsilon$ — remask exactly the tokens the EB-Sampler would have
    committed last, i.e., the ones that barely fit in the entropy budget.
  - Intuition: tokens just above the budget threshold are the ones where the unmasking
    decision is most uncertain. Remasking them allows the model to re-evaluate in richer context.
- Does the order of operations matter? (unmask then remask, vs. remask then unmask,
  vs. interleaved)?
- Can the combined sampler be shown to achieve a tighter $E_{\text{fact}}$ bound than
  either strategy alone?

### Implementation

Modify `external/remdm/main.py` to replace the uniform unmasking step with the EB-Sampler
step. The entropy budget $\varepsilon$ and remasking threshold $\tau$ become two
hyperparameters. Run the step sweep for combinations of $(\varepsilon, \tau)$.

---

## R2: Calibrated Confidence — Temperature Scaling for Remasking

### The Problem

Our remdm-conf MAUVE collapse at high T is a **calibration failure**: the model's
softmax distribution over tokens becomes overconfident at low masking rates (few
masked positions → rich context → sharply peaked predictions). Overconfident
predictions produce low entropy $\hat{I}^i$, so high-entropy positions (which need
correction) are *underdetected*, and the model locks in poor early choices.

**Formal statement.** Define calibration error at noise level $t$ as:
$$\text{ECE}(t) = \mathbb{E}\!\left[\left|\hat{I}^i(z_t) - I^i(x)\right|\right]$$

Hypothesis: $\text{ECE}(t)$ increases as $t \to 0$ (low masking rate) because the
model's entropy estimate degrades when context is rich but the model is still
uncertain about some positions for non-local reasons.

### Temperature Scaling Solution

Replace the entropy estimate with a temperature-scaled version:
$$\hat{I}^i_\tau(z_t) = H\!\left(\text{softmax}\!\left(\frac{\ell^i}{\tau(t)}\right)\right)$$

where $\ell^i$ are the raw logits and $\tau(t) > 1$ inflates entropy estimates.
Three schedule options:

1. **Fixed:** $\tau(t) = \tau_0$ (constant inflation)
2. **Annealed:** $\tau(t) = 1 + (\tau_0 - 1) \cdot (1 - \alpha_t)$
   where $\alpha_t$ is the survival probability. Since $\alpha_t \approx 0$ at the
   start (t≈T, high masking rate) and $\alpha_t \approx 1$ at the end (t≈1, low
   masking rate), this gives $\tau \approx \tau_0$ early (high temperature, diverse)
   and $\tau \approx 1$ late (no softening, commit to best token).
3. **Data-adaptive:** $\tau(t)$ chosen so $\mathbb{E}[\hat{I}^i_\tau] = \mathbb{E}[I^i]$
   using a calibration set (requires a small labelled corpus)

The annealed schedule is the most theoretically motivated: it corrects the systematic
overconfidence that appears as more tokens are committed (lower masking rate → lower $\alpha_t$).

### Connection to the Diversity Window

The diversity window (MDLM MAUVE peak at T=256) may itself be a calibration artefact.
At T=1000, the model makes 1000 unmasking decisions, many at very low masking rates
where $\hat{I}^i$ is underestimated. The model prematurely commits low-entropy tokens,
accumulating commitment errors. Temperature scaling that inflates $\hat{I}^i$ at low
$t$ would delay commitment, potentially eliminating the diversity window — or shifting
it to higher T.

**Testable prediction:** if temperature scaling eliminates (or substantially reduces)
the remdm-conf MAUVE collapse at T=1000, this result is *consistent with* overconfidence
being a primary cause — but does not prove it. If collapse persists across a range of
$\tau_0$, the cause is more likely structural (mode-seeking is intrinsic to the strategy,
not a calibration artefact), and a fundamentally different remasking policy would be needed.

### Implementation

No retraining required. Add a `--temperature_schedule` flag to `scripts/remedi_eval.py`
and the ReMDM inference loop. Calibration curves can be estimated by comparing $\hat{I}^i$
to empirical accuracy on a held-out corpus.

---

## R3: The Remasking-Schedule Co-design Problem

### The Problem

The current framework treats the noise schedule $\alpha_t$ (set during training) and
the inference-time rescheduling strategy as independent. But they are coupled: the
optimal remasking threshold $\tau$ depends on $\alpha_t$, and conversely, the optimal
$\alpha_t$ depends on how much remasking will be used at inference.

**Formal observation.** The EB-Sampler bound $E_{\text{fact}} \leq \varepsilon \cdot \max_i \hat{I}^i$
depends on $\hat{I}^i$, which depends on $\alpha_t$ through the noise level. A noise
schedule that makes $\hat{I}^i$ more uniform across positions (lower $\Sigma^2$) will
reduce $E_{\text{fact}}$ for the same entropy budget $\varepsilon$.

### Research Questions

1. **Optimal schedule given remasking.** The Lavenant & Zanella bound uses a fixed
   $\alpha_t$ (cosine or linear). If remasking is used, the effective number of
   "independent" decisions per step changes. Derive the optimal $\alpha_t$ that
   minimises $E_{\text{fact}}$ given that a confidence-guided corrector will be
   applied at each step.

2. **Curriculum training.** Train the model with a noise schedule that concentrates
   on the masking rates where the remasking threshold $\tau$ will be most active.
   For example, if $\tau$ is set to remask positions with $\hat{I}^i > 0.5$ nats,
   then training should over-sample $\alpha_t$ corresponding to this entropy level.

3. **Learned vs. fixed schedule.** The noise schedule is a fixed cosine curve in all
   current MDMs. Could a *learned* schedule (optimised end-to-end with the remasking
   strategy) outperform the cosine schedule? This is the MDM analogue of continuous
   diffusion's noise schedule optimisation (e.g., Kingma et al., 2021).

### Difficulty

Noise schedule optimisation requires retraining, making this a combined training +
inference problem. However, the co-design insight (that schedule and rescheduling
strategy should be optimised jointly) can be stated and verified empirically by
testing multiple schedules with the same rescheduling strategy.

---

## R4: Block-Level Rescheduling for Large Models

### The Problem

All current rescheduling work (EB-Sampler, ReMDM, thesis) operates at the
*token* level: one decision per token per step. For 8B-scale models with sequence
lengths of 2048 tokens, token-level rescheduling over T=256 steps requires
$256 \times 2048 = 524,\!288$ individual decisions per sequence. The overhead of
sorting, thresholding, and remasking at token granularity can be non-negligible.

Block-wise generation (used by LLaDA and RemeDi) groups $B$ tokens into a block
and generates all $B$ tokens together. This is computationally efficient but uses
a fixed block size (typically $B = 32$), ignoring the information profile within each block.

### Formal Proposal: Hierarchical EB-Sampler

Define a two-level rescheduling strategy:
- **Block level:** use the EB-Sampler to decide *which blocks* to unmask at each step,
  based on the block's average entropy $\bar{I}^b = \frac{1}{B}\sum_{i \in b} \hat{I}^i$.
- **Token level (within committed blocks):** use confidence-guided remasking to revisit
  high-entropy tokens within recently committed blocks.

**Block-level entropy budget:**
$$\varepsilon_B = B \cdot \varepsilon_{\text{token}}$$
Unmask all blocks whose average entropy $\bar{I}^b \leq \varepsilon_B$ per step.
This preserves the EB-Sampler's structure but operates at block granularity.

**Interaction between levels:** Token-level remasking within a block changes the block's
average entropy. Should the block-level schedule be updated after within-block corrections?

### Why This Matters for the Thesis

If the block-level EB-Sampler achieves comparable $E_{\text{fact}}$ reduction to the
token-level EB-Sampler, then the thesis's contribution scales to 8B models without
the $O(L)$ overhead of per-token sorting. The key quantity to bound is the error
introduced by replacing $\hat{I}^i$ with $\bar{I}^b$ — a Jensen's inequality argument.

---

## R5: Multi-Pass Rescheduling and Convergence

### The Problem

The current framework applies one remasking pass per denoising step: unmask (EB-Sampler),
then remask (thesis strategy). What is the benefit of applying *multiple* remasking
passes per step, and does the benefit per pass diminish?

### Formal Framework

Define the $k$-pass sampler $\mathcal{A}^{(k)}_{\varepsilon, \tau}$:
```
for each denoising step t → t-1:
    z' ← EB-Sampler(z_t, ε)           # unmask
    for k = 1, ..., K:
        z' ← remask_high_entropy(z', τ)  # remask
        z' ← EB-Sampler(z', ε_k)       # re-unmask (with reduced budget ε_k)
z_{t-1} ← z'
```

The budget $\varepsilon_k$ for re-unmasking after the $k$-th remasking pass should
decrease geometrically: $\varepsilon_k = \varepsilon / r^k$ for $r > 1$ (tighter
budget each pass, ensuring convergence).

**Question:** Is there a formal convergence guarantee analogous to ReMDM's monotone
improvement theorem? The ReMDM proof uses the data-processing inequality on the
Markov chain; the same argument should apply to the combined EB+remask pass.

**Claim:** Under the assumption $E_{\text{learn}} = 0$:
$$E_{\text{fact}}(\mathcal{A}^{(k+1)}) \leq E_{\text{fact}}(\mathcal{A}^{(k)}) \quad \forall k$$

with the improvement per pass bounded by $\Delta_k \propto \varepsilon_k \cdot \tau$.

### Relationship to Inference-Time Compute Scaling

If the bound on $E_{\text{fact}}$ per pass decreases geometrically, then:
$$E_{\text{fact}}(\mathcal{A}^{(K)}) \leq E_{\text{fact}}(\mathcal{A}^{(0)}) \cdot r^{-K}$$

This is an *inference-time scaling law*: quality improves as $O(\log K)$ in the
number of passes. A log-linear quality-compute curve is a practical finding that
practitioners can use to decide how much inference compute to allocate.

---

## R6: Rescheduling for Conditional Generation

### The Problem

All existing rescheduling work (EB-Sampler, ReMDM, thesis) is evaluated on
*unconditional* generation (no prompt). For practical use cases, the model generates
a response $y$ conditioned on a prompt $x$. The information profile $I^i(y)$ of
the response depends on the prompt: an easy prompt (e.g., "Write: hello") leads to
a low-entropy response; a hard prompt (e.g., "Write a Shakespearean sonnet") leads
to a high-entropy response.

**Formal setup.** Conditional generation with masking:
$$q_\text{cond}(z_t \mid x, y) = \prod_i \mathrm{Cat}(z_t^i;\, \alpha_t e_{y^i} + (1-\alpha_t) e_m)$$

The information profile is now:
$$I^i_\text{cond}(y \mid x) = H(y^i \mid y^{\setminus i}, x)$$

which includes the prompt as conditioning. The EB-Sampler and remasking strategies
should use $I^i_\text{cond}$ rather than $I^i$.

### Research Questions

1. **Prompt-adaptive entropy budget.** For easy prompts (low $I^i_\text{cond}$),
   fewer steps are needed. For hard prompts, more. Design an adaptive budget
   $\varepsilon(x)$ that scales with the prompt's estimated difficulty. How would
   you estimate difficulty from $x$ alone (before generating $y$)?

2. **Conditional calibration.** Does the model's entropy estimate $\hat{I}^i(z_t, x)$
   remain well-calibrated across prompts of different difficulties? A calibration
   analysis on LAMBADA or HellaSwag (where the prompt is the question and the
   response is the answer) would test this.

3. **Remasking for instruction following.** In LLaDA-8B-Instruct, the response must
   follow specific instructions (format, content). Does confidence-guided remasking
   improve instruction adherence? Hypothesis: instruction-critical tokens (e.g.,
   format tokens like "1." or "```python") have systematically lower confidence
   early in generation and benefit most from remasking.

---

## R7: Information Profile Estimation Beyond Entropy

### The Problem

The EB-Sampler and all Tier 3 signals use $\hat{I}^i = H(p_\theta(x^i \mid z_t))$
as a proxy for $I^i(x) = H(x^i \mid x^{\setminus i})$. This approximation has two
known failure modes:

1. **Context sparsity.** Early in generation (high masking rate), $z_t$ has few
   committed tokens. The model predicts $x^i$ in low-context, making $\hat{I}^i$
   a noisy estimate of $I^i(x)$.

2. **Multimodal uncertainty.** If $p_\theta(x^i \mid z_t)$ is bimodal (e.g., 50% "cat",
   50% "dog"), the entropy is high but the correct token may be fully determined by
   context the model hasn't seen yet (masked positions). $\hat{I}^i$ overestimates
   $I^i(x)$ in this case.

### Alternative Signals

**Mutual information proxy:**
$$\hat{I}^i_\text{MI}(z_t) = I(x^i;\, x_{\mathcal{M}_t}) \approx H(p_\theta(x^i \mid z_t)) - H(p_\theta(x^i \mid z_t^{\text{full}}))$$

where $z_t^{\text{full}}$ is the sequence with all currently masked positions filled
in with their MAP estimate. This estimates how much the masked positions would change
the prediction at position $i$ — a second-order uncertainty signal. Cost: one extra
forward pass per step (fill in MAPs, then re-evaluate entropy).

**Attention-based proxy:**
$$\hat{I}^i_\text{attn}(z_t) = \sum_{j \in \mathcal{M}_t} a_{ij} \cdot \hat{I}^j(z_t)$$

where $a_{ij}$ is the attention weight from position $i$ to masked position $j$.
This estimates how much position $i$ would "care about" the currently masked positions
if they were revealed. High $\hat{I}^i_\text{attn}$ means position $i$ attends strongly
to uncertain positions — a good candidate for remasking.

**Ensemble uncertainty:**
Run the model forward twice with different dropout masks (or token dropout) and compare
the two distributions at position $i$:
$$\hat{I}^i_\text{ens} = \mathrm{JSD}(p_\theta^{(1)}(x^i \mid z_t),\, p_\theta^{(2)}(x^i \mid z_t))$$

High Jensen-Shannon divergence indicates epistemic uncertainty (the model is inconsistent
about this position). Cost: 2 forward passes per step.

### Research Direction

Compare $\hat{I}^i$, $\hat{I}^i_\text{MI}$, $\hat{I}^i_\text{attn}$, and $\hat{I}^i_\text{ens}$
as remasking signals:
1. Calibration analysis: which signal best predicts actual prediction error?
2. Quality at matched compute: which signal achieves better MAUVE/gen_ppl per forward pass?
3. Regime analysis: which signal is most valuable early vs. late in generation?

---

## R8: Statistical Significance and Evaluation Standards

### The Problem

Every empirical result in this thesis and in the literature is a point estimate.
At N=100 generated samples, MAUVE estimates carry substantial variance. Some
"findings" may be noise:
- MDLM gen_ppl worsening T=512→1000 (49.0→52.3): is this real?
- remdm-loop T=512 MAUVE dip (0.614→0.532 before 0.684): is the dip real?

### Formal Proposal: Bootstrap CI Framework

For MAUVE at N=100 samples and B=1000 bootstrap resamples:
$$[\text{MAUVE}^{(2.5\%)}, \text{MAUVE}^{(97.5\%)}] = \text{bootstrap CI at 95\%}$$

For gen_ppl, use a paired Wilcoxon signed-rank test between strategies (non-parametric,
no normality assumption) to test whether the median per-sequence cross-entropy differs.

**Minimum sample size.** Define $N^*(\delta) = \min\{N : \text{CI width} < \delta\}$
for target precision $\delta = 0.05$ (5% of the MAUVE scale). Extrapolate from our
N=100 data using the $O(1/\sqrt{N})$ scaling of standard errors.

**Expected finding.** Based on the variance in our results (MAUVE ranging 0.17–0.74
across strategies and step counts), N=100 likely gives 95% CIs of width ≈ ±0.10.
The key differences (remdm-loop 0.684 vs. remdm-conf 0.325 at T=1000, Δ=0.359) are
almost certainly significant. The small anomalies (gen_ppl 49.0→52.3) may not be.

**Implementation.** Can be run immediately on existing generated sequences in
`results/*/generated_sequences.json`. Write `scripts/bootstrap_ci.py` to produce
a table of CIs for the thesis. This requires no new HPC runs.

---

## Summary Table: Research Directions

| Direction | Training needed | New HPC runs | Theoretical | Empirical | Difficulty |
|-----------|:---:|:---:|:---:|:---:|:---:|
| R1: EB + Remasking | No | Yes | $E_f$ interaction bound | Sweep over $(\varepsilon, \tau)$ | Medium |
| R2: Temperature calibration | No | Yes | Calibration analysis | Step sweep + temperature | Low |
| R3: Schedule co-design | Yes | Yes | Optimal schedule derivation | Train + inference | High |
| R4: Block-level reschedul. | No | Yes | Jensen's inequality bound | Block EB on LLaDA-8B | Medium |
| R5: Multi-pass convergence | No | Yes | Geometric bound on passes | K-pass sweep | Medium |
| R6: Conditional generation | No | Yes | Conditional $I^i_\text{cond}$ | LAMBADA/HellaSwag | Medium |
| R7: Better info proxy | No | Yes | Calibration theory | Proxy comparison | Medium |
| R8: Statistical testing | No | No | Bootstrap theory | CIs from existing data | **Low** |

**For the thesis:** R8 (statistical testing) should be done first — it validates or
refutes existing results. R2 (temperature calibration) is the most tractable novel
contribution: training-free, one extra hyperparameter, directly motivated by the
remdm-conf collapse finding. R1 (EB + remasking) is the cleanest theoretical extension
of the EB-Sampler paper and fills the most important gap in the literature.

---

## The Rescheduling Problem: A Unified View

To close this part, here is the rescheduling problem stated as a single optimisation:

$$\min_{\pi \in \Pi} \mathrm{KL}(\pi_{\text{data}} \| p_{\pi}) \quad \text{subject to} \quad \mathrm{NFE}(\pi) \leq C$$

where:
- $\Pi$ is the space of all rescheduling policies (which tokens to unmask/remask, in what order, with what probabilities);
- $p_\pi$ is the distribution of sequences produced by policy $\pi$;
- $\mathrm{NFE}(\pi)$ is the expected number of forward evaluations (compute budget);
- $C$ is the compute constraint.

**Known solutions:**
- $\pi = \text{MDLM}$: uniform random unmasking, no remasking. $E_{\text{fact}} = O(1/T)$.
- $\pi = \text{EB-Sampler}$: entropy-budget unmasking, no remasking. $E_{\text{fact}} = O(\varepsilon \cdot \max_i \hat{I}^i)$ — optimal for unmasking-only policies.
- $\pi = \text{thesis}$: EB-Sampler + confidence-guided remasking. $E_{\text{fact}} < E_{\text{EB}}$ when $\Sigma^2 > 0$.
- $\pi = \text{RemeDi-RL}$: learned policy via RL. Empirically best but requires training.

**Open:** The globally optimal policy in $\Pi$ (minimising KL subject to NFE budget)
is unknown. The thesis derives the optimal policy within the class of *threshold*
strategies. The globally optimal policy is presumably learned (RemeDi direction) but
its form is not characterised theoretically.

---

## Appendix A: Statistical and Mathematical Foundations

This appendix is a self-contained reference for the mathematical background required to
understand discrete diffusion models. It covers probability theory, information theory,
variational inference, Markov chains, and the specific structures that arise in masked
diffusion. Each section includes definitions, key results, and their direct connections
to the models reviewed in the main document.

---

### B.1 Probability Theory Essentials

#### B.1.1 Random Variables and Distributions

A **random variable** $X$ is a measurable function from a probability space
$(\Omega, \mathcal{F}, P)$ to a measurable space. For discrete $X$ taking values in
a countable set $\mathcal{X}$:

- **Probability mass function (PMF):** $p(x) = P(X = x) \geq 0$, $\sum_{x \in \mathcal{X}} p(x) = 1$
- **Support:** $\text{supp}(p) = \{x : p(x) > 0\}$

**Categorical distribution** $\text{Cat}(\boldsymbol{\pi})$ over $\mathcal{X} = \{1, \ldots, V\}$:

$$P(X = v) = \pi_v, \quad \boldsymbol{\pi} \in \Delta_{V-1}$$

where $\Delta_{V-1} = \{\boldsymbol{\pi} \in \mathbb{R}^V : \pi_v \geq 0,\ \sum_v \pi_v = 1\}$
is the $(V-1)$-dimensional probability simplex.

**One-hot encoding:** $\mathbf{e}_v \in \{0,1\}^V$ has $(\mathbf{e}_v)_i = \mathbf{1}[i = v]$.
Then $X \sim \text{Cat}(\boldsymbol{\pi})$ can be written $P(X = v) = \boldsymbol{\pi}^\top \mathbf{e}_v$.

#### B.1.2 Conditional Probability and Bayes' Theorem

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

**Bayes' theorem:**

$$P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}$$

where $P(B) = \sum_a P(B \mid A=a)\, P(A=a)$ is the normalising constant (marginal likelihood).

In the diffusion context: $P(\mathbf{x}_0 \mid \mathbf{x}_t)$ is the posterior over clean sequences
given a noisy (masked) observation, computed via Bayes from the forward kernel $P(\mathbf{x}_t \mid \mathbf{x}_0)$
and the prior $P(\mathbf{x}_0)$.

#### B.1.3 Independence and Conditional Independence

$X$ and $Y$ are **independent** ($X \perp Y$) iff $P(X,Y) = P(X)\,P(Y)$ for all values.

$X$ and $Y$ are **conditionally independent given $Z$** ($X \perp Y \mid Z$) iff
$P(X, Y \mid Z) = P(X \mid Z)\, P(Y \mid Z)$.

**Why this matters for masked diffusion:** The reverse model $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$
is typically factored as $\prod_i p_\theta(x^i_{t-1} \mid \mathbf{x}_t)$, i.e., positions are
treated as conditionally independent given $\mathbf{x}_t$. The true posterior $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$
does **not** factorise this way — tokens are jointly dependent — and this discrepancy is the
**factorisation error** $E_{\text{fact}}$.

#### B.1.4 Expectation and Variance

$$\mathbb{E}[X] = \sum_x x\, p(x), \qquad \text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$$

**Law of total expectation:** $\mathbb{E}[X] = \mathbb{E}_Y[\mathbb{E}[X \mid Y]]$

**Law of total variance:** $\text{Var}(X) = \mathbb{E}_Y[\text{Var}(X \mid Y)] + \text{Var}_Y(\mathbb{E}[X \mid Y])$

#### B.1.5 Jensen's Inequality

For a **convex** function $f$ (i.e., $f''(x) \geq 0$):

$$f\!\left(\mathbb{E}[X]\right) \leq \mathbb{E}[f(X)]$$

For a **concave** function ($f''(x) \leq 0$), the inequality reverses:
$f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]$.

**Critical application:** $-\log$ is convex, so $-\log(\mathbb{E}[Z]) \leq \mathbb{E}[-\log Z]$,
which gives the ELBO $\leq$ log-marginal inequality (see Section B.3).

---

### B.2 Information Theory

All logarithms are natural ($\ln$) unless stated; entropy is in **nats**. To convert to bits,
divide by $\ln 2$.

#### B.2.1 Entropy

The **Shannon entropy** of a discrete random variable $X \sim p$:

$$H(X) = -\sum_{x \in \mathcal{X}} p(x) \ln p(x)$$

(convention: $0 \ln 0 = 0$).

**Key properties:**

| Property | Statement | Proof sketch |
|---|---|---|
| Non-negativity | $H(X) \geq 0$ | $p(x) \leq 1 \Rightarrow -\ln p(x) \geq 0$ |
| Determinism | $H(X) = 0 \iff X$ is deterministic | Only one $p(x) > 0$ |
| Maximum | $H(X) \leq \ln|\mathcal{X}|$ | Lagrange multipliers; equality iff uniform |
| Concavity | $H$ is concave in $p$ | Hessian is negative semidefinite |

**Interpretation:** $H(X)$ measures the average uncertainty about $X$ before observing it,
or equivalently the average code length (in nats) under an optimal code.

**Character perplexity:** $e^{H(X)}$ (or $2^{H(X)}$ in bits) is the *effective alphabet size*
in the sense that a uniform distribution over $e^{H(X)}$ symbols has the same entropy.
For a token entropy of $H = 5.5$ nats: character perplexity $= e^{5.5} \approx 245$.
An entropy change $\Delta H$ corresponds to a character perplexity ratio of $e^{\Delta H}$.

#### B.2.2 Joint and Conditional Entropy

$$H(X, Y) = -\sum_{x,y} p(x,y) \ln p(x,y)$$

$$H(X \mid Y) = -\sum_{x,y} p(x,y) \ln p(x \mid y) = H(X,Y) - H(Y)$$

**Conditioning reduces entropy:** $H(X \mid Y) \leq H(X)$, with equality iff $X \perp Y$.

**Chain rule:** $H(X_1, X_2, \ldots, X_n) = \sum_{i=1}^n H(X_i \mid X_1, \ldots, X_{i-1})$

This is the factorisation of the joint distribution into a product of conditionals:
$p(x_1, \ldots, x_n) = \prod_i p(x_i \mid x_1, \ldots, x_{i-1})$.

**For sequences:** A language model $p_\theta$ assigns probability
$p_\theta(\mathbf{x}) = \prod_{i=1}^L p_\theta(x^i \mid x^1, \ldots, x^{i-1})$, computing
each conditional autoregressively. Masked diffusion models compute all conditionals
in parallel (one forward pass), which is efficient but introduces the factorisation
error when tokens are decoded jointly.

#### B.2.3 KL Divergence

The **Kullback-Leibler divergence** from $Q$ to $P$ (read: "$P$ from $Q$" or "KL of $P$ with respect to $Q$"):

$$D_\text{KL}(P \| Q) = \sum_x p(x) \ln \frac{p(x)}{q(x)}$$

**Key properties:**

- **Non-negativity (Gibbs' inequality):** $D_\text{KL}(P \| Q) \geq 0$, with equality iff $P = Q$ a.e.
  *Proof:* $-D_\text{KL} = \mathbb{E}_P[\ln(q/p)] \leq \ln \mathbb{E}_P[q/p] = \ln 1 = 0$ (Jensen, $\ln$ concave).

- **Asymmetry:** $D_\text{KL}(P \| Q) \neq D_\text{KL}(Q \| P)$ in general.

- **Mode-seeking vs mode-covering:**
  - Minimising $D_\text{KL}(q \| p)$ w.r.t. $q$: $q$ concentrates on modes of $p$ (mode-seeking, zero-forcing).
    If $p(x) > 0$ but $q(x) = 0$, the term $q(x) \ln(q(x)/p(x)) = 0$ — no penalty for missing mass.
  - Minimising $D_\text{KL}(p \| q)$ w.r.t. $q$: $q$ must cover all mass of $p$ (mode-covering, mean-seeking).
    If $p(x) > 0$ but $q(x) \approx 0$, the term $p(x) \ln(p(x)/q(x)) \to +\infty$ — severe penalty.

**This asymmetry is empirically observable in our results:** remdm-conf at T=1000 achieves
low gen_ppl (model fits GPT-2 distribution well by mode-seeking) but low MAUVE (fails to
cover the full OWT distribution). The mode-seeking direction minimises one KL; covering
diversity requires minimising the other.

#### B.2.4 Cross-Entropy

$$H(P, Q) = -\sum_x p(x) \ln q(x) = H(P) + D_\text{KL}(P \| Q)$$

**Interpretation:** The average code length when data is distributed as $P$ but coded
under $Q$. Always $H(P, Q) \geq H(P)$, with equality iff $P = Q$.

**Gen-ppl decomposition** (critical for thesis):

$$\text{gen-ppl} = \exp H(p_\theta, q_{\text{GPT-2}}) = \exp[H(p_\theta) + D_\text{KL}(p_\theta \| q_{\text{GPT-2}})]$$

A low gen-ppl can arise from: (a) low entropy $H(p_\theta)$ (mode collapse), (b) low KL
(distribution aligned with GPT-2), or (c) both. It is **not** a pure measure of distributional
fit — a degenerate model that generates a single, GPT-2-likely sentence would achieve
gen-ppl $\approx 1$ while being useless.

#### B.2.5 Mutual Information

$$I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = D_\text{KL}(P_{XY} \| P_X P_Y)$$

- $I(X;Y) \geq 0$, equal to zero iff $X \perp Y$.
- $I(X;Y) = I(Y;X)$ (symmetric, unlike KL).
- $I(X;Y) = H(X) + H(Y) - H(X,Y)$ (visualised via the entropy Venn diagram).

**Data processing inequality (DPI):** If $X \to Y \to Z$ is a Markov chain
(i.e., $Z \perp X \mid Y$), then $I(X; Z) \leq I(X; Y)$.

**DPI in diffusion:** The chain $\mathbf{x}_0 \to \mathbf{x}_t \to \mathbf{x}_T$ means
$I(\mathbf{x}_0; \mathbf{x}_T) \leq I(\mathbf{x}_0; \mathbf{x}_t)$ for $t \leq T$.
The forward process monotonically destroys information about $\mathbf{x}_0$.
As $\alpha_T \to 0$ (all tokens masked), $I(\mathbf{x}_0; \mathbf{x}_T) \to 0$.

#### B.2.6 Information Profile (connection to thesis)

The **information profile** of position $i$ in sequence $\mathbf{x}$ is:

$$I^i(\mathbf{x}) = H(x^i \mid \mathbf{x}^{\setminus i})$$

This is the conditional entropy of position $i$ given all other positions. It measures
how much uncertainty about $x^i$ remains after observing all other tokens.

- $I^i(\mathbf{x}) = 0$: position $i$ is fully determined by context (e.g., a function word after a fixed phrase).
- $I^i(\mathbf{x})$ large: position $i$ is highly variable given context (e.g., the first content word of a sentence).

**Lavenant & Zanella (2024)** prove that the optimal unmasking order is increasing in
$I^i(\mathbf{x})$: unmask positions with **lowest** $I^i$ first (most predictable given context).
This minimises the factorisation error at each step — you never commit a token that is
highly uncertain given its neighbours.

---

### B.3 Variational Inference and the ELBO

#### B.3.1 Latent Variable Models

A **latent variable model** defines a joint distribution $p_\theta(\mathbf{x}, \mathbf{z})$
over observed variables $\mathbf{x}$ and latent variables $\mathbf{z}$:

$$p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{x} \mid \mathbf{z})\, p(\mathbf{z})$$

The **marginal likelihood** (evidence):

$$p_\theta(\mathbf{x}) = \sum_{\mathbf{z}} p_\theta(\mathbf{x} \mid \mathbf{z})\, p(\mathbf{z})$$

is typically **intractable** (sum over all latent configurations). Maximum likelihood
training requires maximising $\log p_\theta(\mathbf{x})$, which requires computing this sum.

**In discrete diffusion:** $\mathbf{x}$ is the clean sequence $\mathbf{x}_0$ and
$\mathbf{z} = (\mathbf{x}_1, \ldots, \mathbf{x}_T)$ is the noisy trajectory.
$p(\mathbf{z})$ is the forward process (fixed), and $p_\theta(\mathbf{x}_0 \mid \mathbf{z})$
is the reverse model (learned).

#### B.3.2 The Evidence Lower Bound (ELBO)

Introduce a variational distribution $q_\phi(\mathbf{z} \mid \mathbf{x})$ to approximate the
intractable posterior $p_\theta(\mathbf{z} \mid \mathbf{x})$. Then:

$$\log p_\theta(\mathbf{x}) = \log \sum_{\mathbf{z}} q_\phi(\mathbf{z} \mid \mathbf{x}) \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})}$$

Applying Jensen's inequality ($\log$ is concave, so $\log \mathbb{E}[Z] \geq \mathbb{E}[\log Z]$):

$$\log p_\theta(\mathbf{x}) \geq \mathbb{E}_{q_\phi}\!\left[\log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})}\right] =: \mathcal{L}(\theta, \phi; \mathbf{x}) \quad \text{(ELBO)}$$

**Two equivalent forms:**

$$\mathcal{L} = \underbrace{\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x} \mid \mathbf{z})]}_{\text{reconstruction}} - \underbrace{D_\text{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z}))}_{\text{regularisation}}$$

$$\log p_\theta(\mathbf{x}) = \mathcal{L}(\theta, \phi; \mathbf{x}) + \underbrace{D_\text{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p_\theta(\mathbf{z} \mid \mathbf{x}))}_{\geq 0}$$

The second form shows the ELBO is a lower bound on the log-evidence, with the gap equal
to the KL between the variational posterior and the true posterior. The ELBO is tight
($= \log p_\theta(\mathbf{x})$) when $q_\phi = p_\theta(\cdot \mid \mathbf{x})$.

#### B.3.3 ELBO for Diffusion Models

In discrete diffusion (MDLM, D3PM), the ELBO expands as:

$$\mathcal{L} = \underbrace{\mathbb{E}_{q(\mathbf{x}_1 \mid \mathbf{x}_0)}\!\left[\log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)\right]}_{\text{reconstruction at } t=1}
- \underbrace{D_\text{KL}(q(\mathbf{x}_T \mid \mathbf{x}_0) \| p(\mathbf{x}_T))}_{\text{prior matching}}
- \sum_{t=2}^{T} \underbrace{\mathbb{E}_{q(\mathbf{x}_t \mid \mathbf{x}_0)}\!\left[D_\text{KL}(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t))\right]}_{\text{denoising terms}}$$

**Key insight:** Each denoising term measures how well $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$
matches the *true* Bayesian reverse $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$.
For the absorbing process, this posterior has an analytic closed form (see Section B.6),
which is why MDLM training is tractable: the model just needs to learn to predict $\mathbf{x}_0$
from $\mathbf{x}_t$, and the posterior is computed analytically.

#### B.3.4 Mean-Field Approximation

The **mean-field** variational family assumes full factorisation:
$q(\mathbf{z}) = \prod_i q_i(z_i)$.

The optimal factor satisfies: $\ln q_j^*(z_j) = \mathbb{E}_{-j}[\ln p(\mathbf{x}, \mathbf{z})] + \text{const}$

**Connection to factorisation error:** The reverse model
$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \prod_i p_\theta(x^i_{t-1} \mid \mathbf{x}_t)$
is a mean-field approximation to the joint reverse posterior. The factorisation error
is exactly the KL penalty from using this mean-field approximation instead of the
true joint posterior.

---

### B.4 Markov Chains

#### B.4.1 Definition and Markov Property

A sequence of random variables $(X_0, X_1, X_2, \ldots)$ is a **Markov chain** if:

$$P(X_t = x_t \mid X_0 = x_0, \ldots, X_{t-1} = x_{t-1}) = P(X_t = x_t \mid X_{t-1} = x_{t-1})$$

The future state depends only on the present state, not the past. This is called the
**Markov property** (memorylessness).

A chain is **time-homogeneous** if the transition kernel $T(x, y) = P(X_t = y \mid X_{t-1} = x)$
does not depend on $t$.

#### B.4.2 Transition Matrix and Chapman-Kolmogorov

For a finite state space $\mathcal{S}$, the transition kernel is a matrix $\mathbf{T}$ with:
- $T_{xy} = P(X_t = y \mid X_{t-1} = x) \geq 0$
- $\sum_y T_{xy} = 1$ (rows sum to one — $\mathbf{T}$ is a **stochastic matrix**)

**$k$-step transition:** $P(X_{t+k} = y \mid X_t = x) = (\mathbf{T}^k)_{xy}$

**Chapman-Kolmogorov:** $\mathbf{T}^{t+s} = \mathbf{T}^t \cdot \mathbf{T}^s$

In the diffusion context, the $t$-step marginal of the forward process $q(\mathbf{x}_t \mid \mathbf{x}_0)$
is computed by applying $\mathbf{T}^t$ to the one-hot encoding $\mathbf{e}_{x_0}$.

#### B.4.3 Stationary Distribution and Detailed Balance

A distribution $\boldsymbol{\pi}$ is **stationary** for $\mathbf{T}$ if $\boldsymbol{\pi} = \boldsymbol{\pi} \mathbf{T}$,
i.e., $\pi(y) = \sum_x \pi(x) T(x, y)$.

A chain satisfies **detailed balance** (is **reversible**) if:
$$\pi(x)\, T(x, y) = \pi(y)\, T(y, x) \quad \forall x, y$$

Detailed balance implies stationarity (but not vice versa). Reversibility means the chain
looks statistically identical run forwards or backwards in time — a property that diffusion
models exploit: the forward (noising) and reverse (denoising) processes share the same
stationary distribution.

**Absorbing state:** State $a$ is absorbing if $T(a, a) = 1$ (and $T(a, y) = 0$ for $y \neq a$).
For the masking process: $\texttt{[MASK]}$ is absorbing. Once a token is masked, it stays masked
under the forward process. The reverse process $p_\theta$ must learn to "unmask" it.

#### B.4.4 Continuous-Time Markov Chains (CTMC)

A **CTMC** is a Markov chain with continuous time index $t \in [0, \infty)$. It is
characterised by a **rate matrix** (generator) $\mathbf{Q}$:
- $Q_{ij} \geq 0$ for $i \neq j$ (rates of transitioning from $i$ to $j$)
- $Q_{ii} = -\sum_{j \neq i} Q_{ij}$ (rows sum to zero)

**Transition matrix:** $\mathbf{P}(t) = e^{\mathbf{Q}t} = \sum_{k=0}^\infty \frac{(Qt)^k}{k!}$

**Kolmogorov forward equation:** $\frac{d}{dt} \mathbf{P}(t) = \mathbf{P}(t)\, \mathbf{Q}$

**Connection to discrete diffusion:** SEDD and DFM (Campbell et al.) formulate the
forward process as a CTMC with $Q_{x, \texttt{MASK}} = \sigma(t)$ for all $x \neq \texttt{MASK}$,
recovering the absorbing discrete diffusion process in continuous time. The reverse
CTMC has rate matrix characterised by the score function of the marginal distribution.

#### B.4.5 The Backward Equation and Reverse Process

For a forward CTMC with generator $\mathbf{Q}$ and stationary distribution $\boldsymbol{\pi}$,
the **time-reversal** is also a CTMC with rates:

$$\tilde{Q}_{ij}(t) = \frac{\pi_j(t)}{\pi_i(t)} Q_{ji}$$

This is the continuous-time analogue of the reversal used in diffusion models. The key
insight: **learning the reverse process is equivalent to learning the ratios $\pi_j / \pi_i$**,
which connects to score-based methods.

---

### B.5 Score Functions and Denoising

*(Primarily for continuous-space context; included for completeness and for understanding SEDD.)*

#### B.5.1 Score Function

The **score function** of a distribution $p(\mathbf{x})$ over $\mathbf{x} \in \mathbb{R}^d$:

$$s(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})$$

Points in the direction of increasing log-density — toward regions of higher probability mass.

**Example:** For $p(\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$:
$s(\mathbf{x}) = -\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})$

The score of a Gaussian points back toward the mean — it is a linear "restoring force."

#### B.5.2 Fisher Divergence and Score Matching

The **Fisher divergence** between distributions $p$ and $q$:

$$D_F(p \| q) = \mathbb{E}_p\!\left[\|s_p(\mathbf{x}) - s_q(\mathbf{x})\|^2\right]$$

**Score matching** (Hyvärinen 2005) minimises $D_F(p_{\text{data}} \| p_\theta)$ without
requiring $p_{\text{data}}$ to be tractable, using integration by parts.

#### B.5.3 Denoising Score Matching

**Denoising score matching** (Vincent 2011) adds Gaussian noise
$\tilde{\mathbf{x}} = \mathbf{x} + \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 I)$,
and trains a denoiser $s_\theta(\tilde{\mathbf{x}})$ to match $\nabla_{\tilde{\mathbf{x}}} \log p(\tilde{\mathbf{x}} \mid \mathbf{x})$:

$$\min_\theta \mathbb{E}_{\mathbf{x} \sim p}\, \mathbb{E}_{\boldsymbol{\epsilon}}\!\left[\left\|s_\theta(\tilde{\mathbf{x}}) + \frac{\boldsymbol{\epsilon}}{\sigma^2}\right\|^2\right]$$

This avoids computing intractable score of $p_{\text{data}}$ directly.

**Tweedie's formula:** For $\tilde{\mathbf{x}} = \mathbf{x}_0 + \sigma \boldsymbol{\epsilon}$:

$$\mathbb{E}[\mathbf{x}_0 \mid \tilde{\mathbf{x}}] = \tilde{\mathbf{x}} + \sigma^2 \nabla_{\tilde{\mathbf{x}}} \log p(\tilde{\mathbf{x}})$$

The posterior mean of the clean signal equals the noisy signal plus a score-weighted correction.
This underlies the DDPM $\mathbf{x}_0$-prediction parametrisation.

#### B.5.4 Discrete Score Function (SEDD)

For a discrete distribution $p(\mathbf{x})$ over $\mathcal{X}^L$, there is no gradient.
The **concrete score** (Lou et al., 2023) at position $i$ and value $v$:

$$s_\theta(\mathbf{x})^i_v = \frac{p_\theta([\mathbf{x}^{\setminus i}, v])}{p_\theta(\mathbf{x})}$$

This ratio measures how the probability changes when position $i$ is changed to $v$.
The model learns these ratios directly, and they are used to define the reverse CTMC.
SEDD shows this leads to tractable training via a **score-entropy** loss.

---

### B.6 The Absorbing (Masking) Forward Process

This section details the specific Markov chain used in MDLM, D3PM (absorbing), ReMDM, LLaDA.

#### B.6.1 Forward Kernel

Let $\mathcal{V} = \{1, \ldots, V\} \cup \{\texttt{M}\}$ where $\texttt{M}$ is the mask token.
The **one-step forward kernel** masks each token independently:

$$q(x^i_t \mid x^i_{t-1}) = (1 - \beta_t)\, \mathbf{e}_{x^i_{t-1}} + \beta_t\, \mathbf{e}_\texttt{M}$$

where $\beta_t \in [0,1]$ is the masking probability at step $t$.

**$t$-step marginal** (by composing the one-step kernels):

$$q(x^i_t \mid x^i_0) = \alpha_t\, \mathbf{e}_{x^i_0} + (1 - \alpha_t)\, \mathbf{e}_\texttt{M}$$

where $\alpha_t = \prod_{s=1}^t (1 - \beta_s)$ is the **survival probability** — the
probability a token remains unmasked at step $t$. $\alpha_0 = 1$ (fully clean),
$\alpha_T \approx 0$ (fully masked).

**Simplified:** At time $t$, position $i$ is:
- Unmasked (equals $x^i_0$) with probability $\alpha_t$
- Masked ($= \texttt{M}$) with probability $1 - \alpha_t$

The positions are masked **independently** — the forward process factorises over positions.

#### B.6.2 Posterior (Reverse Bayesian Update)

The posterior $q(x^i_{t-1} \mid x^i_t, x^i_0)$ is computed by Bayes:

**Case 1: $x^i_t \neq \texttt{M}$ (token is already unmasked):**

$$q(x^i_{t-1} \mid x^i_t \neq \texttt{M}, x^i_0) = \delta(x^i_{t-1} = x^i_t)$$

If a token is visible at time $t$, it was also visible at $t-1$ (absorbing state property).
The reverse step is deterministic: keep the token.

**Case 2: $x^i_t = \texttt{M}$ (token is masked):**

$$q(x^i_{t-1} \mid x^i_t = \texttt{M}, x^i_0) = \frac{\alpha_{t-1} - \alpha_t}{1 - \alpha_t}\, \mathbf{e}_{x^i_0} + \frac{1 - \alpha_{t-1}}{1 - \alpha_t}\, \mathbf{e}_\texttt{M}$$

**Derivation:** By Bayes, $q(x^i_{t-1} \mid \texttt{M}, x^i_0) \propto q(\texttt{M} \mid x^i_{t-1})\, q(x^i_{t-1} \mid x^i_0)$.
Two cases for $x^i_{t-1}$:
- $x^i_{t-1} = x^i_0$ (unmasked): $q(\texttt{M} \mid x^i_0) \cdot \alpha_{t-1} = (1-\alpha_{t-1}) \cdot \alpha_{t-1}$... wait, more carefully:
  $q(\texttt{M} \mid x^i_{t-1} = x^i_0) = 1 - \alpha_t/\alpha_{t-1}$ (one-step mask probability given unmasked at $t-1$)
  $q(x^i_{t-1} = x^i_0 \mid x^i_0) = \alpha_{t-1}$
  Weight: $\propto (1 - \alpha_t/\alpha_{t-1}) \cdot \alpha_{t-1} = \alpha_{t-1} - \alpha_t$
- $x^i_{t-1} = \texttt{M}$ (masked): $q(\texttt{M} \mid \texttt{M}) = 1$ (absorbing), $q(\texttt{M} \mid x^i_0) = 1 - \alpha_{t-1}$
  Weight: $\propto 1 \cdot (1-\alpha_{t-1}) = 1 - \alpha_{t-1}$

Normalising by $1 - \alpha_t = (\alpha_{t-1} - \alpha_t) + (1 - \alpha_{t-1})$ gives the formula above.

**Key insight:** Given that a token is masked at time $t$, the posterior says:
unmask it to $x^i_0$ with probability $(\alpha_{t-1} - \alpha_t)/(1-\alpha_t)$,
or keep it masked with the complementary probability. The model must learn to predict
$x^i_0$ from $\mathbf{x}_t$ to implement this reverse step.

#### B.6.3 Masked Cross-Entropy Training Loss

The ELBO denoising term at step $t$:

$$\mathbb{E}_{q(\mathbf{x}_t \mid \mathbf{x}_0)}\!\left[D_\text{KL}(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t))\right]$$

reduces (using the closed-form posterior) to a **masked cross-entropy**:

$$\mathcal{L}_t \propto -\sum_{i:\, x^i_t = \texttt{M}} \mathbb{E}\!\left[\ln p_\theta(x^i_0 \mid \mathbf{x}_t)\right]$$

Only masked positions contribute to the loss. The model is trained to predict the original
token $x^i_0$ at each masked position given the partially-masked sequence $\mathbf{x}_t$.

**This is exactly masked language modelling (BERT-style)**, but with a principled derivation:
it is the ELBO of the absorbing diffusion model. The difference from BERT: the masking
schedule $\alpha_t$ is explicit, and the loss is summed over all $t$ during training.

#### B.6.4 Noise Schedule

The noise schedule $\{\alpha_t\}_{t=0}^T$ controls how quickly tokens are masked.

**Linear schedule:** $\alpha_t = 1 - t/T$ — uniform masking across steps.

**Cosine schedule (MDLM):** $\alpha_t = \cos^2(\frac{\pi t}{2T})$ — slow start and end,
faster masking in the middle. Inspired by the continuous-time flow matching literature.

**Effect on training:** The schedule determines what fraction of tokens are masked at
each training step. A cosine schedule ensures the model trains on a roughly uniform
distribution over masking rates, unlike linear which concentrates training at intermediate rates.

---

### B.7 Factorisation Error: Derivation and Bounds

#### B.7.1 Where the Error Comes from

The true joint reverse process:
$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \sum_{\mathbf{x}_0} q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)\, q(\mathbf{x}_0 \mid \mathbf{x}_t)$$

This is a mixture over all possible clean sequences — intractable, and **does not factorise**
over positions (the joint distribution of all unmasked tokens is correlated through $\mathbf{x}_0$).

The model approximates:
$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \prod_i p_\theta(x^i_{t-1} \mid \mathbf{x}_t)$$

The error per step is:
$$\varepsilon_t = D_\text{KL}\!\left(\prod_i q(x^i_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \,\Big\|\, q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)\right)$$

Summing over steps: $E_{\text{fact}} = \sum_{t=1}^T \varepsilon_t$.

#### B.7.2 Scaling with T

**Claim:** For smooth models, $E_{\text{fact}} = O(\Sigma^2 / T)$ where $\Sigma^2$ measures
the covariance between positions in the posteriors.

**Intuition:** At each step, the fraction of newly unmasked tokens is $\Delta\alpha_t = \alpha_{t-1} - \alpha_t \approx 1/T$.
The cross-correlations between unmasked tokens scale as $O(\Delta\alpha_t^2) = O(1/T^2)$.
Summing over $T$ steps: $E_{\text{fact}} \sim T \cdot O(1/T^2) = O(1/T)$.

**Consequence:** More steps → lower factorisation error → better generation quality.
This motivates using large $T$ (e.g., T=1000 vs T=128). The MAUVE improvement from
T=128 to T=1000 in our results is at least partially explained by $E_{\text{fact}}$ reduction.

#### B.7.3 Remasking as Error Correction

Remasking addresses $E_{\text{fact}}$ at inference time without retraining. If a token
$x^i$ was committed (unmasked) based on insufficient context — because correlated tokens
$x^j, x^k$ were still masked — remasking it to $\texttt{M}$ allows the model to reconsider
it after those context tokens are revealed.

**Formal effect:** Let $E_{\text{fact}}^{\text{base}}$ be the factorisation error without
remasking. With remasking, committed tokens can be revised, reducing the effective
$\Sigma^2$ at each step. The theoretical gain depends on whether the proxy confidence
signal aligns with the information profile $I^i(\mathbf{x})$.

---

### B.8 Evaluation Metrics: Formal Definitions

#### B.8.1 Perplexity and Bits-Per-Byte

**Perplexity** of a language model $p_\theta$ on text data $\mathcal{D}$:

$$\text{PPL}(p_\theta, \mathcal{D}) = \exp\!\left(\frac{1}{N}\sum_{n=1}^N -\log p_\theta(\mathbf{x}_n)\right) = \exp\big(H(p_{\text{data}}, p_\theta)\big)$$

where $H(p_{\text{data}}, p_\theta)$ is the cross-entropy of the data distribution under the model.

**Bits-per-byte (BPB):** $\text{BPB} = H(p_{\text{data}}, p_\theta) / \ln(2) / C$
where $C$ is the average number of bytes per token (typically $\approx 4$ for GPT-2 tokenizer).

**Gen-ppl** (used in this thesis): evaluates the *generated* distribution $p_\theta$ under
an external evaluator $q_{\text{GPT-2}}$:

$$\text{gen-ppl} = \exp H(p_\theta, q_{\text{GPT-2}}) = \exp\big[H(p_\theta) + D_\text{KL}(p_\theta \| q_{\text{GPT-2}})\big]$$

This is estimated by: generate $N$ sequences from $p_\theta$, compute their average
negative log-likelihood under GPT-2. Lower gen-ppl means generated text is more
likely under GPT-2 (closer to fluent English), **but** it conflates fluency (low KL)
with low diversity (low entropy). A model that generates one GPT-2-likely sentence
repeatedly achieves gen-ppl $\approx 1$ while being degenerate.

#### B.8.2 MAUVE

**MAUVE** (Pillutla et al., 2021) measures the divergence between the model distribution
$P$ (generated samples) and the reference distribution $Q$ (human text) via the
**divergence frontier**.

For $\lambda \in [0, 1]$, define the mixture:
$$M_\lambda = (1 - \lambda)\, P + \lambda\, Q$$

The **divergence frontier** is the curve:
$$\mathcal{F}(P, Q) = \{(\text{KL}(M_\lambda \| P),\; \text{KL}(M_\lambda \| Q)) : \lambda \in [0,1]\}$$

This traces a curve in $\mathbb{R}_{\geq 0}^2$. When $P = Q$, $M_\lambda = P = Q$ for all $\lambda$,
and both KLs are zero — the curve collapses to the origin.

**MAUVE score:**
$$\text{MAUVE}(P, Q) = \exp(-c \cdot \text{Area}(\mathcal{F}(P, Q)))$$

where $c > 0$ is a scaling constant. Lower area = higher MAUVE = distributions more similar.
MAUVE $\in (0, 1]$, with MAUVE $= 1$ when $P = Q$.

**Geometric interpretation of the frontier:**
- The $x$-axis KL $= \text{KL}(M_\lambda \| P)$ measures how far the mixture deviates from $P$ (model). High $\lambda$ (mixture close to $Q$) → large $x$.
- The $y$-axis KL $= \text{KL}(M_\lambda \| Q)$ measures how far the mixture deviates from $Q$ (reference). Low $\lambda$ (mixture close to $P$) → large $y$.
- If $P$ has low recall (misses modes of $Q$): large $y$-axis values (high divergence from $Q$).
- If $P$ has low precision (generates OOD text): large $x$-axis values (high divergence from $P$).

**Practical computation:** $P$ and $Q$ are not available analytically; they are represented
as sets of feature vectors (GPT-2 embeddings of generated / reference sentences).
The frontier is approximated by a quantisation of the feature space. This introduces
sampling variance — estimates are unreliable for $N < 500$ samples (Pillutla et al.).

#### B.8.3 Entropy as a Diversity Proxy

Token-level entropy of the generated distribution:

$$H_{\text{gen}} = \frac{1}{L} \sum_{i=1}^{L} H(x^i) = -\frac{1}{L} \sum_{i=1}^L \sum_v p_\theta(x^i = v)\, \ln p_\theta(x^i = v)$$

In practice, estimated by the empirical marginal over $N$ generated samples.

**Interpretation:** High entropy = diverse vocabulary usage; low entropy = repetitive text.
Entropy drop is a **signature of mode collapse**: the model is concentrating on a small
subset of tokens. In our results, remdm-conf drops from $H = 5.499$ at T=128 to
$H = 5.357$ at T=1000 — a decrease of 0.142 nats, corresponding to a character
perplexity ratio of $e^{0.142} \approx 1.15$ (15% reduction in effective vocabulary).

**Limitations:** Entropy is a marginal statistic — it misses higher-order structure.
A model that generates each token independently at the correct marginal frequency would
have the correct entropy but zero inter-token correlation (wrong text). Entropy is
necessary but not sufficient for quality assessment.

#### B.8.4 Bootstrap Confidence Intervals

Given $N$ i.i.d. samples and a statistic $\hat{\theta}$ (e.g., MAUVE, gen-ppl), the
**bootstrap** (Efron, 1979) estimates the sampling distribution of $\hat{\theta}$:

1. Draw $B$ bootstrap samples: for each $b = 1, \ldots, B$, resample $N$ observations with replacement from the original $N$ samples.
2. Compute $\hat{\theta}^{(b)}$ for each bootstrap sample.
3. The bootstrap distribution $\{\hat{\theta}^{(b)}\}_{b=1}^B$ approximates the sampling distribution of $\hat{\theta}$.

**Percentile CI:** $[\hat{\theta}^{(\alpha/2)},\, \hat{\theta}^{(1-\alpha/2)}]$ where the quantiles are from the bootstrap distribution. This is a 95% CI when $\alpha = 0.05$.

**Why N=100 is marginal for MAUVE:** MAUVE estimates the area of a frontier in feature space — it requires enough samples to cover the manifold. With $N=100$, bootstrap CIs typically span $\pm 0.05$–$0.10$ MAUVE units. A difference of $\Delta\text{MAUVE} = 0.09$ (e.g., remdm-conf vs remdm-loop at T=128: $0.440$ vs $0.396$) may not be significant. Bootstrap CIs are needed to determine which differences in the step-sweep table are reliably non-zero.

---

### B.9 Softmax, Temperature Scaling, and Calibration

#### B.9.1 Softmax

The **softmax** function maps logits $\mathbf{z} \in \mathbb{R}^V$ to probabilities:

$$\text{softmax}(\mathbf{z})_v = \frac{e^{z_v}}{\sum_{v'=1}^V e^{z_{v'}}}$$

**Temperature softmax** with temperature $\tau > 0$:

$$\text{softmax}_\tau(\mathbf{z})_v = \frac{e^{z_v/\tau}}{\sum_{v'} e^{z_{v'}/\tau}}$$

- $\tau \to 0$: $\text{softmax}_\tau \to \arg\max$ (one-hot, deterministic greedy)
- $\tau = 1$: standard softmax
- $\tau \to \infty$: $\text{softmax}_\tau \to \text{Uniform}(V)$

**Temperature annealing in remasking:** At high masking rates (early generation), the
model is uncertain — high temperature encourages exploration. As fewer tokens are masked,
the model is more confident — lower temperature enables exploitation. The schedule
$\tau(t) = 1 + (\tau_0 - 1)(t/T)$ achieves $\tau = \tau_0 > 1$ at $t = T$ (fully masked)
and $\tau = 1$ at $t = 0$ (fully revealed).

#### B.9.2 Calibration

A model is **calibrated** if its confidence equals its empirical accuracy:

$$P(\hat{y} = y \mid p_\theta(\hat{y}) = p) = p \quad \forall p \in [0,1]$$

**Overconfidence:** $p_\theta(\hat{y}) > P(\text{correct})$ — the model claims higher certainty than warranted. Common in large neural language models.

**Temperature scaling** is the simplest post-hoc calibration method: find $\tau^*$ on a
held-out validation set that minimises the expected calibration error (ECE):

$$\tau^* = \arg\min_\tau \frac{1}{N}\sum_n \text{NLL}_\tau(y_n, p_\theta(\cdot \mid \mathbf{x}_n))$$

**Connection to remdm-conf collapse:** If $p_\theta$ is overconfident (too-small $\tau$),
the confidence signal is overconfident — it commits tokens prematurely. At T=1000 there
are 1000 opportunities to make premature commitments that cannot be revisited, leading
to diversity collapse. Entropy-based or temperature-calibrated confidence is a principled
fix.

---

### B.10 Worked Examples: Applying the Formulas

#### B.10.1 Computing KL Divergence for a Categorical

Let $P = \text{Cat}(0.7, 0.2, 0.1)$ and $Q = \text{Cat}(0.5, 0.3, 0.2)$ over $V = 3$ tokens.

$$D_\text{KL}(P \| Q) = 0.7 \ln\frac{0.7}{0.5} + 0.2 \ln\frac{0.2}{0.3} + 0.1 \ln\frac{0.1}{0.2}$$
$$= 0.7 \times 0.336 + 0.2 \times (-0.405) + 0.1 \times (-0.693)$$
$$= 0.235 - 0.081 - 0.069 = 0.085 \text{ nats}$$

Note: the first term dominates because $P$ has high mass on $v=1$ but $Q$ assigns less.
The mode-seeking asymmetry: $D_\text{KL}(Q \| P) = 0.5\ln(0.5/0.7) + \ldots \approx 0.076$ nats
— different and slightly smaller (the reverse direction is less penalised for the mass shift).

#### B.10.2 ELBO Bound Tightness

Suppose $\log p_\theta(\mathbf{x}) = -3.2$ nats. A variational approximation achieves
$\mathcal{L} = -4.1$ nats. Then:

$$D_\text{KL}(q \| p_\theta(\cdot \mid \mathbf{x})) = \log p_\theta(\mathbf{x}) - \mathcal{L} = -3.2 - (-4.1) = 0.9 \text{ nats}$$

The variational posterior is 0.9 nats from the true posterior — substantial room for improvement.
In practice, a well-trained diffusion model drives this gap toward zero by the design of the
forward process (the posterior is analytically tractable).

#### B.10.3 Absorbing Posterior: Numerical Example

Let $T = 4$, $\alpha_t = 1 - t/T$, so $\alpha_0 = 1$, $\alpha_1 = 0.75$, $\alpha_2 = 0.5$, $\alpha_3 = 0.25$, $\alpha_4 = 0$.

Given $x^i_0 = \text{"cat"}$ and $x^i_3 = \texttt{M}$ (masked at step 3):

$$q(x^i_2 \mid x^i_3 = \texttt{M},\, x^i_0 = \text{"cat"}) = \frac{\alpha_2 - \alpha_3}{1 - \alpha_3}\, \mathbf{e}_{\text{"cat"}} + \frac{1 - \alpha_2}{1 - \alpha_3}\, \mathbf{e}_\texttt{M}$$
$$= \frac{0.5 - 0.25}{1 - 0.25}\, \mathbf{e}_{\text{"cat"}} + \frac{1 - 0.5}{1 - 0.25}\, \mathbf{e}_\texttt{M} = \frac{1}{3}\, \mathbf{e}_{\text{"cat"}} + \frac{2}{3}\, \mathbf{e}_\texttt{M}$$

At step $t=3$, a masked token at step $t=2$: 1/3 chance it was unmasked (= "cat"), 2/3 chance it was still masked. The model must predict "cat" with confidence proportional to this posterior.

#### B.10.4 Entropy of Generated Text

Generated sequence entropy $H = 5.45$ nats per token (observed for remdm-loop at T=1000).
Character perplexity: $e^{5.45} = 233$.

If entropy drops to $H = 5.35$ nats (remdm-conf at T=1000, a drop of 0.10 nats):
- Character perplexity ratio: $e^{0.10} = 1.105$
- Interpretation: effective vocabulary 10.5% smaller — not catastrophic, but a consistent signal of mode-seeking behaviour.

The corresponding gen-ppl change is **larger** than this entropy change alone would suggest,
because mode-seeking also reduces $D_\text{KL}(p_\theta \| q_{\text{GPT-2}})$: a less diverse
model is "closer" to GPT-2's distribution in the forward KL sense.

---

### B.11 Quick-Reference: Key Inequalities

| Inequality | Statement | Used for |
|---|---|---|
| **Jensen** | $f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$ for convex $f$ | ELBO derivation |
| **Gibbs** | $D_\text{KL}(P \| Q) \geq 0$ | All KL-based bounds |
| **Data processing** | $I(X;Z) \leq I(X;Y)$ if $X \to Y \to Z$ | Forward process information loss |
| **Chain rule** | $H(X_1,\ldots,X_n) = \sum_i H(X_i \mid X_{1:i-1})$ | Autoregressive decomposition |
| **Conditioning** | $H(X \mid Y) \leq H(X)$ | Information profile optimality |
| **Log-sum** | $\sum_i a_i \ln(a_i/b_i) \geq (\sum_i a_i)\ln(\sum_i a_i / \sum_i b_i)$ | Rate-distortion bounds |

---

### B.12 Notation Conventions Used Throughout This Document

| Symbol | Meaning |
|---|---|
| $\mathbf{x} = (x^1, \ldots, x^L)$ | Token sequence of length $L$ |
| $\mathbf{x}^{\setminus i}$ | All tokens except position $i$ |
| $\alpha_t \in [0,1]$ | Survival probability at step $t$ (prob. of being unmasked) |
| $\texttt{M}$ or $\texttt{[MASK]}$ | Mask token |
| $q(\cdot)$ | Forward process (fixed, data-independent) |
| $p_\theta(\cdot)$ | Reverse (generative) model with parameters $\theta$ |
| $D_\text{KL}(P \| Q)$ | KL divergence from $Q$ to $P$ |
| $H(P, Q)$ | Cross-entropy of $P$ under $Q$ |
| $H(X)$ | Entropy of $X$ |
| $I(X;Y)$ | Mutual information |
| $I^i(\mathbf{x})$ | Information profile at position $i$: $H(x^i \mid \mathbf{x}^{\setminus i})$ |
| $E_{\text{fact}}$ | Total factorisation error $= \sum_t D_\text{KL}(\text{true} \| \text{factored})$ |
| $\text{ELBO}$ | Evidence lower bound $= \log p(\mathbf{x}) - D_\text{KL}(q \| p_\theta(\cdot \mid \mathbf{x}))$ |
| $\Delta_{V-1}$ | Probability simplex over $V$ tokens |
| $\tau$ | Temperature parameter (softmax scaling) |
| $\boldsymbol{\pi}$ | Probability vector in $\Delta_{V-1}$ |

---

## Appendix B: Repository Index

| Repository | GitHub URL | Submodule path | Role in thesis |
|---|---|---|---|
| kuleshov-group/remdm | https://github.com/kuleshov-group/remdm | `external/remdm` | ReMDM strategies + MDLM-OWT inference |
| SeunggeunKimkr/PRISM | https://github.com/SeunggeunKimkr/PRISM | `external/PRISM` | PRISM quality head (Sudoku + OWT) |
| maple-research-lab/RemeDi | https://github.com/maple-research-lab/RemeDi | `external/remedi` | RemeDi-RL, RemeDi-Instruct inference |
| kuleshov-group/mdlm | https://github.com/kuleshov-group/mdlm | `external/mdlm` | Original MDLM — independent baseline |
| louaaron/Score-Entropy-Discrete-Diffusion | https://github.com/louaaron/Score-Entropy-Discrete-Diffusion | `external/sedd` | SEDD PC-sampler baseline |
| GSAI-ML/LLaDA | https://github.com/GSAI-ML/LLaDA | HF only | LLaDA-8B — loaded via `transformers` |

---

*Compiled with:*
```bash
pandoc comparison.md -o comparison.pdf \
  --pdf-engine=xelatex -V geometry:margin=2.5cm \
  --toc --toc-depth=2 --number-sections
```

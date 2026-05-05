> ARCHIVED — historical only. March 2026 deep-dive on correctors in masked diffusion models.
> Do not use this file to infer current thesis status unless explicitly asked.
> Current source of truth: see `START_HERE.md` and `docs/README.md`.

---
title: "Correctors for Discrete Diffusion Models — Deep Dive"
author: "MSc Thesis — Bocconi University, supervised by Prof. Giacomo Zanella"
date: "March 2026"
geometry: "top=2.5cm, bottom=2.5cm, left=3cm, right=3cm"
fontsize: 11pt
linkcolor: blue
numbersections: true
toc-depth: 3
---

**Purpose:** Preparation for the meeting with Prof. Zanella (2026-03-26).
Study guide covering concepts, intuitions, and main proofs of corrector
papers for discrete (masked) diffusion models.

**How to read this:** Each section opens with a 2--3 sentence intuition,
then gives the formal setup, then the proof (or proof sketch), then a
*"What to tell Zanella"* box summarising the takeaway in one sentence.

\newpage

# Background: MCMC, Gibbs Sampling, and Spectral Gaps

## Why You Need This

Every corrector paper relies on the same core machinery: Markov chains
that converge to a target distribution by repeatedly updating one
component at a time (Gibbs sampling). The speed of convergence is
controlled by the spectral gap. If you understand these three things —
Gibbs sampling, stationarity, spectral gap — every corrector paper is
a variation on the same theme.

## Markov Chains: The Basics

**Definition.** A sequence of random variables $X_0, X_1, X_2, \ldots$ is a
Markov chain if:
$$P(X_t = x_t \mid X_0 = x_0, \ldots, X_{t-1} = x_{t-1}) = P(X_t = x_t \mid X_{t-1} = x_{t-1})$$
The future depends only on the present, not on the history.

**Transition matrix.** For a finite state space $S = \{s_1, \ldots, s_n\}$, the
chain is specified by an $n \times n$ matrix $T$ where:
$$T(x, y) = P(X_t = y \mid X_{t-1} = x) \geq 0, \qquad \sum_y T(x, y) = 1$$
After $k$ steps: $P(X_{t+k} = y \mid X_t = x) = (T^k)_{xy}$ — just matrix power.

**Stationary distribution.** A distribution $\pi$ is stationary for $T$ if:
$$\pi = \pi T \qquad \text{i.e., } \pi(y) = \sum_x \pi(x)\, T(x, y) \text{ for all } y$$
Intuition: if you start in $\pi$, one step of the chain leaves you in $\pi$.
It is a fixed point.

**Detailed balance (reversibility).** The chain satisfies detailed balance
w.r.t. $\pi$ if:
$$\pi(x)\, T(x, y) = \pi(y)\, T(y, x) \qquad \text{for all } x, y$$
Intuition: the "flow" from $x$ to $y$ under $\pi$ equals the flow from $y$
to $x$. The chain looks the same whether you run time forward or backward.
Detailed balance implies stationarity (sum both sides over $x$). This is
the standard way to verify stationarity in MCMC.

## Gibbs Sampling

**Setup.** You want to sample from a joint distribution $p(x_1, x_2, \ldots, x_d)$
over $d$ variables, but sampling the joint directly is hard. You know the
conditionals $p(x_i \mid x_{\setminus i})$ for each variable given the rest.

**Algorithm.** Starting from any initial state $(x_1^0, \ldots, x_d^0)$:

> For $t = 1, 2, 3, \ldots$
> Pick a coordinate $i$ (randomly or systematically)
> Resample: $x_i^t \sim p(x_i \mid x_{\setminus i}^{t-1})$
> Keep all other coordinates: $x_j^t = x_j^{t-1}$ for $j \neq i$

**Why it works.** Each step satisfies detailed balance w.r.t. $p$:
$$p(x)\, T_{\text{Gibbs}}(x, x') = p(x')\, T_{\text{Gibbs}}(x', x)$$
where $x$ and $x'$ differ only at coordinate $i$. Proof sketch:

- $T_{\text{Gibbs}}(x, x') = \frac{1}{d} \cdot p(x'_i \mid x_{\setminus i})$
  (pick coord $i$, resample)
- $p(x) \cdot T_{\text{Gibbs}}(x, x') = \frac{1}{d}\, p(x_{\setminus i})\,
  p(x_i \mid x_{\setminus i})\, p(x'_i \mid x_{\setminus i})$
- This expression is symmetric in $x_i$ and $x'_i$, so it is the same when
  you swap $x$ and $x'$. $\square$

Since detailed balance holds, $p$ is a stationary distribution of the Gibbs
chain. Under mild conditions (irreducibility, aperiodicity), the chain
converges regardless of initialisation.

**Connection to discrete diffusion correctors.** In masked diffusion:

- The $d$ variables are the $L$ token positions.
- The joint distribution is $p_t(z)$ at noise level $t$.
- Resampling $x_i$ = picking a position, masking it, re-predicting from
  $p_\theta(x^i \mid z_{\setminus i})$.

A corrector step in discrete diffusion **is** a Gibbs sampling step.

## Eigenvalues and the Spectral Gap

**Eigenvalue decomposition of $T$.** The transition matrix $T$ of a
reversible chain on $n$ states has $n$ real eigenvalues:
$$1 = \lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n \geq -1$$
Key facts:

- $\lambda_1 = 1$ always (the stationary distribution is the eigenvector).
- All $|\lambda_i| \leq 1$ (stochastic matrix).
- For reversible chains, all eigenvalues are real.

**The spectral gap.** Defined as:
$$\gamma = 1 - \lambda_2$$
where $\lambda_2$ is the second-largest eigenvalue. This is the single most
important number characterising a Markov chain's convergence speed.

**Why $\gamma$ controls convergence.** After $k$ steps, the distance from
stationarity decays as:
$$\|T^k(x, \cdot) - \pi\| \leq C \cdot (1-\gamma)^k = C \cdot \lambda_2^k$$
Intuition: the leading eigenvalue $\lambda_1 = 1$ gives the stationary part;
all other components decay at rate $\lambda_2^k$. The gap $\gamma = 1 - \lambda_2$
controls how fast they decay.

- Large $\gamma$ (close to 1): $\lambda_2$ is small $\Rightarrow$ fast mixing.
- Small $\gamma$ (close to 0): $\lambda_2$ is close to 1 $\Rightarrow$ slow mixing.

**Mixing time.** The mixing time $t_{\text{mix}}$ is the number of steps to reach
total variation distance $\varepsilon$ from $\pi$:
$$t_{\text{mix}}(\varepsilon) \approx \frac{1}{\gamma} \ln \frac{1}{\varepsilon}$$
Mixing time scales as $1/\gamma$ — the reciprocal of the spectral gap.

**Example.** Chain with eigenvalues $1, 0.95, 0.8, -0.3$:

- $\gamma = 1 - 0.95 = 0.05$
- $t_{\text{mix}} \approx 1/0.05 = 20$ steps (to reduce error by factor $e$)
- After 20 steps: error $\approx e^{-1} \approx 0.37$
- After 60 steps: error $\approx e^{-3} \approx 0.05$

## KL Divergence and Spectral Gap

For the Informed Correctors paper, we need the KL version of the convergence
bound. For a reversible chain with stationary distribution $\pi$ and spectral
gap $\gamma$:
$$\mathrm{KL}(p_k \| \pi) \leq \exp(-2k\gamma) \cdot \mathrm{KL}(p_0 \| \pi)$$
where $p_k$ is the distribution after $k$ steps.

Intuition: the KL from the target shrinks geometrically with rate $e^{-2\gamma}$
per step. After $k = 1/(2\gamma)$ steps, KL halves. The factor of 2 comes from
the Pinsker-style relationship between KL and spectral gap.

**This is exactly the bound used by Informed Correctors (Theorem 3).**

## What Determines $\gamma$ for a Gibbs Sampler?

For a Gibbs sampler on $d$ variables (= $L$ token positions), the spectral gap
depends on:

1. **Coupling strength between variables.** If variables are nearly independent,
   $\gamma \approx 1/d$. If variables are strongly coupled (code where changing
   one token forces many others to change), $\gamma$ can be much smaller.

2. **Multimodality.** If $p$ has well-separated modes, the Gibbs chain must
   "tunnel" between them by flipping one variable at a time. This can be
   exponentially slow $\Rightarrow$ $\gamma$ exponentially small.

3. **Sequence length $L$.** For weakly correlated text, $\gamma \sim 1/L$.
   For strongly correlated text (code, formal logic), $\gamma$ can be
   $\sim e^{-L}$.

For natural language (e.g., OpenWebText): dependencies are mostly local.
Expect $\gamma \sim 1/L$ or $1/L^2$, making Gibbs correction feasible for
short-to-medium sequences.

> **What to tell Zanella:** The spectral gap $\gamma = 1 - \lambda_2$ controls
> mixing time as $t_{\text{mix}} \sim 1/\gamma$. For Gibbs samplers in discrete
> diffusion, $\gamma$ depends on the dependency structure of the text — local
> dependencies give manageable $\gamma \sim 1/L$, while strong long-range
> dependencies can make $\gamma$ very small.

\newpage

# The Error-Bound Framework: Lavenant & Zanella (2025)

## The Big Picture

**Intuition.** When a masked diffusion model generates text, it makes two kinds
of mistakes: (1) the model is not perfect (learning error), and (2) it unmasks
multiple tokens at once, pretending they are independent when they are not
(factorization error). This paper separates these two sources and shows that
the factorization error has the structure of a Riemann sum approximation —
which means it can be minimised by choosing the right unmasking schedule.

## Setup

Let:

- $\pi$ = true data distribution over sequences $x \in \mathcal{V}^L$.
- $p_\theta$ = the trained MDM's prediction of the denoising posterior.
- $p_{\text{alg}}$ = the distribution of sequences produced by running the
  MDM sampler for $T$ steps.
- $z_t$ = the partially masked sequence at step $t$.
- $\alpha_t \in [0,1]$ = survival probability ($\alpha_0 = 1$ clean,
  $\alpha_T \approx 0$ fully masked).

The reverse process unmasks tokens from $z_T$ (all masked) to $z_0$ (all
unmasked). At each step $t \to t-1$, some masked tokens are committed. The
model predicts each token independently:
$$p_\theta(z_{t-1} \mid z_t) = \prod_i p_\theta(z_{t-1}^i \mid z_t)$$
This product-of-marginals assumption is the source of factorization error.

## The Main Decomposition

**Theorem (Lavenant & Zanella, informal).** The KL divergence between the
data distribution and the sampler output decomposes as:
$$\mathrm{KL}(\pi \| p_{\text{alg}}) \leq E_{\text{learn}} + E_{\text{fact}}$$

**$E_{\text{learn}}$** (learning error) = cumulative KL between the true
denoising posterior and the model's approximation:
$$E_{\text{learn}} = \sum_t \mathbb{E}_{z_t \sim q_t}\bigl[\mathrm{KL}\bigl(q(x \mid z_t) \| p_\theta(x \mid z_t)\bigr)\bigr]$$
This depends only on model quality. It goes to zero as $p_\theta \to q$
(perfect training). It is independent of the sampling algorithm.

**$E_{\text{fact}}$** (factorization error) = error from treating token
predictions as independent:
$$E_{\text{fact}} = \sum_t \mathbb{E}\Bigl[\mathrm{KL}\Bigl(\prod_i q(x^i_{t-1} \mid z_t, x_0) \;\Big\|\; q(z_{t-1} \mid z_t)\Bigr)\Bigr]$$
This depends on the sampling algorithm (which tokens to unmask, in what order,
how many at once).

**Why this matters:** $E_{\text{learn}}$ is fixed once you train the model.
$E_{\text{fact}}$ is what you can control at inference time. All corrector
papers are about reducing $E_{\text{fact}}$.

## $E_{\text{fact}}$ as a Riemann Approximation Error

**The information profile.** For a sequence $x$, define the per-position
conditional entropy:
$$I^i(x) = H(x^i \mid x_{\setminus i})$$
This measures how hard position $i$ is to predict given all other positions.
"The" vs "a" has low $I^i$ (easy to predict from context). A creative word
choice has high $I^i$.

The vector $(I^1, I^2, \ldots, I^L)$ is the information profile of $x$. The
total information is $I(x) = \sum_i I^i(x)$.

**Key insight.** At each step, when you unmask $k$ positions simultaneously,
you treat them as independent. The error from this independence assumption is
proportional to the amount of information you are releasing:
$$E_{\text{fact}}(t) \approx (\Delta\alpha_t)^2 \cdot \sum_i \mathrm{Var}[I^i]$$
where $\Delta\alpha_t = \alpha_{t-1} - \alpha_t$ is the step size.

**Why quadratic?** Think of unmasking two tokens $A$ and $B$ simultaneously.
The error comes from the correlation between $A$ and $B$. If you unmask them
sequentially, there is no correlation error. If you unmask them together, the
error is proportional to their mutual information $I(A; B)$. For $k$ tokens
unmasked at once, the total pairwise correlation scales as $k^2$, giving
$(\Delta\alpha)^2$ (since $\Delta\alpha \propto k/L$).

Analogy: this is like approximating an integral with rectangles. Each rectangle
has width $\Delta\alpha_t$ and the error per rectangle is $O(\Delta\alpha_t^2)$.
Summing $T$ rectangles of width $\sim 1/T$ gives total error $O(1/T)$.

**Total factorization error:**
$$E_{\text{fact}} \leq C \sum_t (\Delta\alpha_t)^2 \cdot \Sigma^2$$
where $\Sigma^2 = \mathrm{Var}_i[I^i(x)]$ is the variance of the information
profile. For a uniform schedule ($\Delta\alpha_t = 1/T$ for all $t$):
$$E_{\text{fact}} \leq C \cdot \frac{\Sigma^2}{T}$$

**Two levers to reduce $E_{\text{fact}}$:**

1. Increase $T$ (more steps $\Rightarrow$ smaller $\Delta\alpha$ $\Rightarrow$ less error per step).
2. Decrease $\Sigma^2$ (make the information released per step more uniform).

## The Optimal Unmasking Schedule

**Theorem (optimal schedule).** Among all deterministic unmasking schedules,
the one that minimises $E_{\text{fact}}$ unmasks positions in ascending order
of $I^i$ — easy tokens first, hard tokens last.

**Intuition — why easy first?** Consider two tokens: $A$ (easy, $I^A = 0.1$)
and $B$ (hard, $I^B = 2.0$). If you unmask $B$ first, then $A$:

- Step 1: unmask $B$ $\Rightarrow$ $E_{\text{fact}}(1) \propto (I^B)^2 = 4.0$
- Step 2: unmask $A$ $\Rightarrow$ $E_{\text{fact}}(2) \propto (I^A)^2 = 0.01$
- Total $\propto 4.01$

If you unmask $A$ first, then $B$:

- Step 1: unmask $A$ $\Rightarrow$ $E_{\text{fact}}(1) \propto (I^A)^2 = 0.01$
- Step 2: unmask $B$ $\Rightarrow$ now $B$ has more context ($A$ is revealed),
  so the effective $I^B$ is smaller. Total $< 4.01$.

By unmasking easy tokens first, you provide context for hard tokens, reducing
their effective $I^i$ when it is their turn.

**This is the Riemann sum argument.** The information profile is a "curve"
$I(\alpha)$ as a function of how much has been unmasked. The factorization
error is the Riemann approximation error of this integral. The optimal
partition (which tokens at which step) makes the rectangles as equal as
possible.

## The Open Problem: Remasking

The paper explicitly states that remasking — re-masking committed tokens and
re-predicting them — is **not** covered by this analysis.

**The open question (your thesis):** Can remasking reduce $E_{\text{fact}}$
below what optimal unmasking alone achieves? If so, by how much, and which
tokens should be remasked?

**Intuition for why remasking helps:** When token $B$ is committed at step $t$
(when $A$ was still masked), the commitment was based on incomplete context. At
step $t' > t$, $A$ has been committed. If $B$'s prediction would change
significantly given $A$, it was a "premature commitment." Remasking $B$ and
re-predicting with $A$ in context reduces the effective $E_{\text{fact}}$ at
that step.

> **What to tell Zanella:** The paper decomposes sampling error into
> $E_{\text{learn}} + E_{\text{fact}}$. $E_{\text{fact}}$ is a Riemann error
> of the information profile, minimised by unmasking easy tokens first. The
> framework explicitly leaves remasking as an open problem — this is the
> thesis gap.

\newpage

# EB-Sampler: Optimal Unmasking (Ben-Hamu et al., 2025)

## The Big Picture

**Intuition.** The EB-Sampler is the algorithm that implements the optimal
unmasking schedule from Lavenant & Zanella. Instead of unmasking a fixed number
of tokens per step, it uses an entropy budget $\varepsilon$: unmask tokens in
order of ascending entropy (easiest first), stopping when the total entropy
released exceeds $\varepsilon$. This ensures each step releases approximately
equal information — the condition for minimum $E_{\text{fact}}$.

## The Algorithm

At each step $t$:

1. Compute predicted entropy for all masked positions:
$$\hat{I}^i = H\bigl(p_\theta(x^i \mid z_t)\bigr) = -\sum_j p_\theta(x^i = j \mid z_t)\, \log p_\theta(x^i = j \mid z_t)$$

2. Sort masked positions by ascending entropy (easy first).

3. Greedily unmask until entropy budget $\varepsilon$ is consumed:
```
budget = eps
for i in sorted_positions:
    if budget >= I_hat[i]:
        unmask i, budget -= I_hat[i]
    else:
        break
```

4. Sample tokens for unmasked positions from $p_\theta(x^i \mid z_t)$.

**Key design choice:** The budget $\varepsilon$ is the single hyperparameter.
Small $\varepsilon$ = many steps, low $E_{\text{fact}}$, high compute. Large
$\varepsilon$ = few steps, high $E_{\text{fact}}$, fast. The number of steps
adapts: $T \approx \hat{I}_{\text{total}} / \varepsilon$.

## The $E_{\text{fact}}$ Bound

**Theorem (EB-Sampler bound).**
$$E_{\text{fact}}(\text{EB}) \leq \varepsilon \cdot \max_i \hat{I}^i$$

Compare to the uniform fixed-step sampler:
$$E_{\text{fact}}(\text{uniform}) \leq \frac{I(x)}{T} \cdot \max_i I^i(x)$$
Both have the structure: (entropy per step) $\times$ (max per-position
entropy). The EB-Sampler controls the first factor directly via $\varepsilon$,
while the uniform sampler has entropy-per-step $= I(x)/T$ which can be very
uneven across steps.

**Proof intuition.** At each step, the EB-Sampler unmasks a set $S$ of
positions with total entropy $\leq \varepsilon$. The factorization error for
unmasking $S$ simultaneously is bounded by:
$$E_{\text{fact}}(S) \leq \Bigl(\sum_{i \in S} I^i\Bigr) \cdot \max_{i \in S} I^i \leq \varepsilon \cdot \max_i \hat{I}^i$$
The first inequality is the Riemann bound: error $\approx$ width $\times$ height.
Width = total entropy $= \varepsilon$. Height = max per-position entropy.

## Optimality

**Theorem (optimality).** Among all deterministic unmasking schedules, the
EB-Sampler achieves the minimum $E_{\text{fact}}$ for a given total number of
forward passes (NFE).

**Proof sketch.** Any deterministic schedule partitions the $L$ positions into
groups (one per step). The $E_{\text{fact}}$ for each group is proportional to
(group entropy)$^2$. The total $E_{\text{fact}}$ is minimised when all groups
have equal entropy. The EB-Sampler achieves this by construction (each group
has entropy $\approx \varepsilon$). This is the classic result that
$\sum a_i^2$ is minimised when all $a_i$ are equal (by convexity).

## What EB-Sampler Does NOT Do

1. **No remasking.** Errors committed early propagate to the end.
2. **No adaptive $\varepsilon$.** The budget is fixed. An adaptive $\varepsilon(t)$
   that accounts for how much context has been revealed could do better.
3. **$\hat{I}^i$ is noisy early in generation.** When most tokens are masked,
   the model's entropy estimate $\hat{I}^i$ is a poor proxy for the true $I^i$.

**These are exactly the gaps that remasking fills.** Remasking lets you correct
premature commitments and effectively "re-sort" positions after more context is
available.

> **What to tell Zanella:** The EB-Sampler implements the optimal unmasking
> schedule by controlling entropy per step via a budget $\varepsilon$. It
> achieves $E_{\text{fact}} \leq \varepsilon \cdot \max_i \hat{I}^i$ and is
> optimal among deterministic unmasking-only schedules. The thesis extends this
> to include remasking.

\newpage

# Informed Correctors: MCMC Theory (Zhao et al., 2024)

## The Big Picture

**Intuition.** The Informed Correctors paper brings classical MCMC theory to
discrete diffusion correction. Instead of the Riemann error framework
(Lavenant & Zanella), it uses spectral gap theory to prove that: (1)
correction steps reduce KL exponentially fast, and (2) correcting the most
uncertain positions first (informed correction) is strictly faster than
correcting randomly (uniform correction).

This gives an independent theoretical justification for confidence-guided
remasking from a completely different mathematical framework.

## Setup

**Post-generation correction.** Unlike the EB-Sampler (which works during
generation), Informed Correctors applies *after* standard MDM sampling:

> Phase 1: Generate $z_0$ using standard MDM sampler.
> Phase 2: Apply $R$ correction steps to $z_0$.

Each correction step is a Gibbs sampling step: pick a position $d$, resample
$z_0^d$ from the conditional $p_t(z^d \mid z^{\setminus d})$.

**The target distribution.** At the end of generation ($t \approx 0$), the
corrector targets $p_0(z)$ — the model's estimate of the clean data
distribution. The corrector tries to make the generated sequence $z_0$ look
like a sample from $p_0$, removing artifacts from the discrete approximation.

## Uniform Corrector

The simplest corrector: at each step, pick a position $d$ uniformly at random
from $\{1, \ldots, L\}$, and resample:
$$z^d \leftarrow \text{sample from } p_t(z^d \mid z^{\setminus d})$$
This is standard Gibbs sampling with uniform scan order. It satisfies detailed
balance w.r.t. $p_t$, so after enough steps it converges to $p_t$.

## Informed Corrector

Instead of picking $d$ uniformly, pick proportional to the "surprise" (how
unlikely the current token is):
$$P(\text{select } d) \propto \exp(\beta \cdot \text{surprise}^d)$$
where $\text{surprise}^d = -\log p_t(z^d \mid z^{\setminus d})$ is the
negative log-probability of the current token at position $d$, given all other
positions.

- High surprise $\Rightarrow$ the current token is unlikely given context
  $\Rightarrow$ strong candidate for correction.
- Low surprise $\Rightarrow$ the current token fits well $\Rightarrow$ leave
  it alone.

**Temperature $\beta$ controls aggressiveness:**

- $\beta = 0$: uniform corrector (ignore surprise, pick randomly).
- $\beta \to \infty$: always pick the most surprised position (deterministic).
- Intermediate $\beta$: soft weighting toward high-surprise positions.

## The Hollow Transformer

**The problem.** Computing $\text{surprise}^d = -\log p_t(z^d \mid z^{\setminus d})$
requires the conditional probability of position $d$ given *all other positions*.
A standard transformer gives $p_\theta(x^i \mid z_t)$ — the prediction at
position $i$ given the full input $z_t$ including position $i$ itself. This is
**not** the same as $p(x^i \mid z_{\setminus i})$.

**Why they differ.** In a standard transformer, position $i$ attends to itself.
So $p_\theta(x^i \mid z_t)$ is informed by the current value at position $i$.
We want $p(x^i \mid z_{\setminus i})$, which should not see position $i$'s
current value.

**The hollow transformer solution.** Modify the attention mask so that position
$i$ cannot attend to itself (zero the diagonal of the attention matrix). This
gives:
$$p_{\text{hollow}}(x^i \mid z_t) = p(x^i \mid z_{\setminus i})$$
for all $i$ simultaneously in a single forward pass. This is the exact Tier 1
conditional.

**Cost:** Requires retraining the model with the hollow attention mask. Cannot
be applied to existing pretrained MDMs like MDLM-OWT.

**Approximation for standard MDMs:** Use $-\log p_\theta(z^d = \text{current} \mid z_t)$
as a proxy for surprise. This is the Tier 3 signal — biased (the model sees
position $d$'s current value) but works in practice.

## Theorem 3: Exponential KL Decay

**Statement.** Let $\lambda > 0$ be the spectral gap of the Gibbs chain with
transition kernel $T$ (either uniform or informed). After $R$ correction steps:
$$\mathrm{KL}(p_0 \| p_R) \leq \exp(-R\lambda) \cdot \mathrm{KL}(p_0 \| \pi)$$
where $p_0$ is the distribution at the start of correction, $p_R$ after $R$
steps, and $\pi$ is the stationary distribution (target).

**Proof sketch.** The key tool is the Poincaré inequality for the Gibbs chain.
For a reversible Markov chain with spectral gap $\gamma$ and stationary
distribution $\pi$:
$$\mathrm{Var}_\pi(f) \leq \frac{1}{\gamma} \cdot \mathcal{E}_\pi(f, f)$$
where $\mathcal{E}_\pi(f,f) = \frac{1}{2}\sum_{x,y} \pi(x)\, T(x,y)\,[f(x) - f(y)]^2$
is the Dirichlet form. Applying this to $f = \log(p/\pi)$ and using the
relationship between KL and the Dirichlet form:
$$\mathrm{KL}(p_{k+1} \| \pi) \leq (1 - \gamma)\, \mathrm{KL}(p_k \| \pi)$$
Iterating $R$ times and using $(1-\gamma)^R \leq e^{-R\gamma}$:
$$\mathrm{KL}(p_R \| \pi) \leq e^{-R\gamma}\, \mathrm{KL}(p_0 \| \pi) \qquad \square$$

**Intuition.** Each Gibbs step "contracts" the KL by a factor of $(1-\gamma)$.
The spectral gap $\gamma$ is the contraction rate.

**Practical implication.** To halve the KL: $R = \ln(2)/\gamma$ steps. For a
chain with $\gamma = 0.01$, that is $\approx 70$ correction steps. For
$\gamma = 0.1$, only $\approx 7$ steps.

## Theorem 4: Informed > Uniform

**Statement.** Let $\lambda_u$ be the spectral gap of the uniform corrector and
$\lambda_i(\beta)$ be the spectral gap of the informed corrector with temperature
$\beta$. Then:
$$\lambda_i(\beta) \geq \lambda_u \qquad \text{for all } \beta \geq 0$$
with equality if and only if all positions have equal surprise:
$\mathrm{Var}_d[\text{surprise}^d] = 0$.

**What this means.** The informed corrector **always** mixes at least as fast
as the uniform corrector. The advantage is largest when surprises are
non-uniform — i.e., when some positions are much more uncertain than others.
This is exactly when the information profile has high variance $\Sigma^2$.

**Proof sketch (the key argument).** Define the spectral gap via the Dirichlet
form. For the uniform corrector:
$$\mathcal{E}_u(f,f) = \frac{1}{L} \sum_d \mathcal{E}_d(f,f)$$
For the informed corrector with selection probability
$w_d \propto \exp(\beta \cdot \text{surprise}^d)$:
$$\mathcal{E}_i(f,f) = \sum_d w_d\, \mathcal{E}_d(f,f)$$
The spectral gap is:
$$\gamma = \inf_{f:\, \mathrm{Var}(f) > 0} \frac{\mathcal{E}(f,f)}{\mathrm{Var}_\pi(f)}$$
The claim $\gamma_i \geq \gamma_u$ follows if $\mathcal{E}_i(f,f) \geq \mathcal{E}_u(f,f)$
for all $f$.

**Key step:** The informed corrector puts more weight on coordinates $d$ where
$\mathcal{E}_d(f,f)$ is large (high surprise $\Rightarrow$ high local variance
$\Rightarrow$ large Dirichlet form contribution). By the rearrangement inequality,
if the weights $w_d$ are positively correlated with the values $\mathcal{E}_d(f,f)$,
then the weighted average exceeds the unweighted average:
$$\sum_d w_d\, \mathcal{E}_d(f,f) \geq \frac{1}{L} \sum_d \mathcal{E}_d(f,f) \qquad \square$$
The positive correlation holds because high-surprise positions (high $w_d$)
tend to have high local variance (high $\mathcal{E}_d$). **The positions that
most need correction are also the positions where correction is most effective.**

**Equality condition:** If all $w_d$ are equal ($\beta = 0$) or all
$\mathcal{E}_d(f,f)$ are equal (uniform surprises), the inequality becomes an
equality. Otherwise, informed is strictly better.

## Connection to the Thesis

The Informed Correctors result corroborates the thesis from a completely
different angle:

| Framework | What it proves | Tool used |
|-----------|----------------|-----------|
| Lavenant & Zanella (Riemann) | Easy-first unmasking minimises $E_{\text{fact}}$ | Riemann approximation theory |
| Informed Correctors (MCMC) | High-surprise-first correction maximises mixing speed | Spectral gap theory |
| Thesis (both) | Confidence-guided remasking reduces $E_{\text{fact}}$ AND mixes faster | Combines both |

The two frameworks are currently **disconnected**. Nobody has formally shown that
reducing $E_{\text{fact}}$ (Riemann) is equivalent to or implies increased
spectral gap (MCMC). This connection is a potential research direction.

## Limitations

1. **Requires hollow transformer.** The exact informed corrector needs Tier 1
   conditionals, which require retraining.
2. **Post-hoc only.** Correction happens after full generation, not during the
   denoising loop.
3. **$\gamma$ can be very small.** For sequences with strong long-range
   dependencies, the Gibbs chain may need impractically many correction steps.
4. **No connection to $E_{\text{fact}}$.** The MCMC framework proves KL decay
   but does not connect to the Riemann error decomposition.

> **What to tell Zanella:** Informed Correctors proves $\lambda_i \geq \lambda_u$:
> the spectral gap of the confidence-guided corrector is always at least as large
> as the uniform corrector's. The gap grows with the variance of surprises. This
> is an independent corroboration of the thesis from MCMC theory. The key
> limitation: it requires a hollow transformer for exact conditionals.

\newpage

# PRISM: Provable Self-Correction (Kim et al., 2025)

**Full citation:** Jaeyeon Kim, Seunggeun Kim, Taekyun Lee, David Z. Pan,
Hyeji Kim, Sham Kakade, Sitan Chen. "Fine-Tuning Masked Diffusion for Provable
Self-Correction." arXiv:2502.XXXXX (2025).

## The Big Picture

**Intuition.** PRISM takes the simplest possible approach to learning a
confidence signal: train a tiny MLP on top of a frozen pretrained MDM to
predict whether each token prediction is correct. Despite this simplicity, it
has the strongest theoretical guarantee of any confidence signal: the trained
MLP converges to the Bayes-optimal conditional accuracy
$P(\text{prediction correct} \mid \text{context})$.

## Architecture

The base MDM $p_\theta$ is completely frozen. A lightweight MLP $g_\phi$ is
applied to the frozen hidden states $h^i$ at each position:
$$g_\phi(y)^i = \sigma\!\bigl(W_3 \cdot \mathrm{ReLU}(W_2 \cdot \mathrm{ReLU}(W_1 h^i + b_1) + b_2) + b_3\bigr) \in [0,1]$$

- **Input:** hidden state $h^i$ from the base model's last layer.
- **Output:** scalar in $[0,1]$ — "quality score."
- Only $\{W_1, W_2, W_3, b_1, b_2, b_3\}$ are trained (a few thousand parameters).
- The base model's parameters are untouched.

## Training

**Training data.** For each training sequence $x$ and randomly masked version
$y$ (simulating mid-generation):

1. Run the base model: $\hat{x}^i = \arg\max_j\, p_\theta(x^i = j \mid y)$.
2. Label: $\ell^i = 1$ if $\hat{x}^i = x^i$ (correct), $0$ otherwise.
3. Train $g_\phi$ with binary cross-entropy:
$$\mathcal{L}(\phi) = -\sum_i \bigl[\ell^i \log g_\phi(y)^i + (1-\ell^i)\log(1 - g_\phi(y)^i)\bigr]$$

In words: train the MLP to predict whether the base model's prediction at each
position is right.

## The Main Theorem (Proposition 1)

**Statement.** The minimiser $\phi^*$ of the BCE loss over all measurable
functions satisfies:
$$g_{\phi^*}(y)^i = P\bigl(x^i = \hat{x}^i \mid y \oplus m^i\bigr)$$
where $y \oplus m^i$ is the sequence $y$ with position $i$ additionally masked.

**Translation:** The optimal quality head outputs the Bayes-optimal probability
that the base model's prediction is correct, given the context *without*
position $i$.

**Why $y \oplus m^i$ and not $y$?** The MLP takes as input the hidden state
$h^i$, which is computed from $y$. But the BCE minimiser is derived by
conditioning on everything except the label at position $i$. Since the label
$\ell^i$ depends on the base model's prediction at position $i$, and the
prediction depends on whether position $i$ is masked or not, the correct
conditioning variable is $y \oplus m^i$ — the sequence with position $i$
masked out. Intuitively: the quality head should judge whether a prediction is
correct based on the surrounding context, not on the prediction itself.

**Proof.** The proof is a direct application of the fact that BCE is a strictly
proper scoring rule.

*Step 1:* For any binary event $A$ with indicator $\ell_A \in \{0,1\}$, the
function $q^*$ that minimises $\mathbb{E}[-\ell_A \log q - (1-\ell_A)\log(1-q)]$
is $q^* = P(A)$. This is standard: take derivative w.r.t. $q$, set to zero:
$$\frac{d}{dq}\Bigl[-\frac{P(A)}{q} + \frac{1-P(A)}{1-q}\Bigr] = 0
\;\Longrightarrow\; P(A)(1-q) = (1-P(A))q
\;\Longrightarrow\; q = P(A)$$

*Step 2:* Apply to $A = \{x^i = \hat{x}^i\}$ (base model is correct at
position $i$), with conditioning variable $y \oplus m^i$:
$$g_{\phi^*}(y)^i = P(x^i = \hat{x}^i \mid y \oplus m^i) \qquad \square$$

**Why this is powerful.** The guarantee is non-asymptotic: it holds for the
infinite-capacity minimiser of BCE. In practice, the MLP has finite capacity,
so there is an approximation gap — but with enough capacity and data, $g_\phi$
converges to the optimal confidence signal.

## Inference Algorithm

At generation step $t$:

1. Run base model: get logits $p^i$ and hidden states $h^i$.
2. Run quality head: $g^i = g_\phi(h^i)$ for all $i$ (negligible cost).
3. Unmask: top-$K$ most confident masked positions (highest $g^i$).
4. Remask: committed positions with $g^i < \tau$ (low confidence).

The quality head adds essentially zero computational overhead.

## Key Results

- **Sudoku:** PRISM improves completion accuracy from $\sim$30\% to $\sim$90\%.
- **MDLM-OWT (170M):** Improves gen\_ppl and MAUVE over baseline MDLM.
- **LLaDA-8B (MBPP code):** Improves pass@1 on code generation.

## What PRISM Guarantees and What It Does Not

**Does guarantee:**

- $g_\phi$ converges to the true conditional accuracy $P(\text{correct} \mid \text{context})$.
- This is a calibrated probability (not just a ranking).

**Does NOT guarantee:**

- That using $g_\phi$ for remasking produces samples from $\pi$.
- That $g_\phi$ reduces $E_{\text{fact}}$.
- That $g_\phi$ is better than Tier 3 signals in practice.

**The gap:** PRISM's guarantee is about the quality of the confidence signal
(calibration). The thesis needs to connect confidence signal quality to sampling
quality ($E_{\text{fact}}$ reduction).

## Relation to Informed Correctors

| | Informed Correctors | PRISM |
|---|---|---|
| Signal | $-\log p(z^d \mid z_{\setminus d})$ (Tier 1) | $g_\phi(y)^i$ (Tier 2) |
| Requires | Hollow transformer (retrain) | Small MLP training |
| Guarantee | Spectral gap (mixing speed) | BCE calibration |
| Correction type | Post-hoc Gibbs | In-loop remask |
| Compute | $O(L)$ per step w/o hollow | $O(1)$ per step |

Both achieve better-than-uniform correction. Informed Correctors has a stronger
guarantee (spectral gap) but weaker applicability (hollow transformer). PRISM
has a weaker guarantee (calibration only) but works with any frozen MDM.

> **What to tell Zanella:** PRISM trains a tiny MLP to predict whether each
> token prediction is correct. The BCE minimiser converges to the Bayes-optimal
> conditional accuracy — a proper scoring rule argument. The guarantee is about
> signal quality (calibration), not about sampling quality ($E_{\text{fact}}$).
> Connecting the two is an open problem.

\newpage

# DFM: Correctors as Gibbs Samplers (Gat et al., 2024)

## The Big Picture

**Intuition.** DFM (Discrete Flow Matching) provides the cleanest theoretical
language for correctors. It shows that correctors are simply Gibbs samplers
that leave the intermediate distribution $p_t$ invariant. The mixing rate of
the Gibbs sampler is the spectral gap — connecting directly to the Informed
Correctors framework.

## Key Contributions Relevant to Correctors

**1. Objective equivalence.** DFM proves that the training objective for
absorbing-state masked diffusion is:
$$\mathcal{L}_{\text{DFM}} \propto \mathbb{E}_{t,x,z_t}\!\left[\frac{-\dot{\alpha}_t}{1-\alpha_t} \sum_{i:\, z_t^i = m} -\log p_\theta(x^i \mid z_t)\right]$$
This is exactly the MDLM masked cross-entropy with time weight
$w_t = -\dot{\alpha}_t / (1-\alpha_t)$. The flow matching and diffusion
perspectives give the same training.

**2. Correctors as Gibbs.** A corrector for DFM is any CTMC that:

- Leaves $p_t$ stationary (does not change the marginal distribution).
- Reduces entropy along individual trajectories.

The canonical choice: pick position $i$, mask it, resample from
$p_\theta(x^i \mid z_t)$. This is exactly Gibbs sampling with stationary
distribution $p_t$, and it is exactly what ReMDM does.

**3. Mixing rate = spectral gap.** The mixing rate of the Gibbs corrector is
the spectral gap of the Gibbs chain at noise level $t$. This connects DFM's
corrector theory directly to the Informed Correctors framework.

## DFM's Language for the Thesis

In DFM language, the thesis says:

> A confidence-guided Gibbs corrector that selectively targets positions where
> the velocity field has the highest entropy reduces the discretisation error
> of the DFM Euler sampler more efficiently than a uniform Gibbs corrector.

This is a clean, one-sentence summary that positions the thesis at the
intersection of diffusion theory, flow matching, and MCMC.

> **What to tell Zanella:** DFM gives the cleanest language for correctors:
> they are Gibbs samplers that leave $p_t$ invariant, with mixing rate equal
> to the spectral gap. This directly connects to Informed Correctors and
> provides a unified theoretical framing.

\newpage

# Synthesis: How Everything Connects

## The Two Theoretical Threads

**Thread A: Riemann error framework.**
$$\text{Lavenant \& Zanella} \;\to\; \text{EB-Sampler} \;\to\; \text{(thesis: remasking extension)}$$

- Tool: information profile $I(x)$, Riemann sum approximation.
- Proves: $E_{\text{fact}} \leq C \cdot \Sigma^2 / T$; optimal unmasking =
  easy first.
- Language: "how much information released per step."

**Thread B: MCMC spectral framework.**
$$\text{DFM correctors} \;\to\; \text{Informed Correctors} \;\to\; \text{(thesis: connection)}$$

- Tool: Gibbs sampling, spectral gap, Poincaré inequality.
- Proves: KL decays as $\exp(-R\gamma)$; informed $\gamma_i \geq$ uniform
  $\gamma_u$.
- Language: "how fast the chain mixes."

## The Unified Picture

Both threads say the same thing from different angles:

| Concept | Riemann thread | MCMC thread |
|---------|----------------|-------------|
| "Error" | $E_{\text{fact}}$ | $\mathrm{KL}(p_k \| \pi)$ |
| "How it decreases" | $O(1/T)$ with more steps | $\exp(-R\gamma)$ with more corrections |
| "Confidence helps" | Easy-first reduces $\Sigma^2$ | High-surprise-first increases $\gamma$ |
| "Optimal strategy" | Unmask ascending $I^i$ | Correct descending $\text{surprise}^d$ |

**The deep connection.** Both frameworks say: non-uniform strategies (targeting
positions based on uncertainty) are strictly better than uniform strategies, and
the advantage grows with the **variance** of uncertainty across positions.

- Riemann language: high $\Sigma^2$ = high info profile variance $\Rightarrow$
  more to gain from smart scheduling.
- MCMC language: high $\mathrm{Var}[\text{surprise}^d]$ $\Rightarrow$ larger
  gap $\lambda_i - \lambda_u$.

**The open question:** Are $\Sigma^2$ and $\mathrm{Var}[\text{surprise}^d]$
the same quantity? Or closely related? If they are, the two frameworks would
be provably equivalent for the confidence-guided remasking case.

## The Confidence Signal Hierarchy

| Tier | Signal | Source | Guarantee | Paper |
|------|--------|--------|-----------|-------|
| 1 | $p(x^i \mid x_{\setminus i})$ exact | Hollow transformer | Spectral gap | Informed Corr. |
| 2 | $g_\phi(y)^i$ learned | BCE-trained MLP | Calibration | PRISM, RemeDi |
| 3 | $H(p_\theta(x^i \mid z_t))$ | Forward-pass logits | None (heuristic) | EB-Sampler, ReMDM |

The thesis operates at Tier 3 (training-free) and claims it is sufficient for
$E_{\text{fact}}$ reduction. This is a stronger claim than either Informed
Correctors (Tier 1) or PRISM (Tier 2).

## What Is Proven vs. Conjectured

| Claim | Status |
|-------|--------|
| $\mathrm{KL}(\pi \| p_{\text{alg}}) \leq E_{\text{learn}} + E_{\text{fact}}$ | Proven (L\&Z) |
| $E_{\text{fact}}$ is a Riemann error of $I(x)$ | Proven (L\&Z) |
| EB-Sampler minimises $E_{\text{fact}}$ for unmasking | Proven (EB-Sampler) |
| Correction reduces KL exponentially | Proven (Inf. Corr.) |
| Informed correction has larger $\gamma$ | Proven (Inf. Corr.) |
| PRISM's $g_\phi \to$ Bayes-optimal | Proven (PRISM) |
| Remasking reduces $E_{\text{fact}}$ | Conjectured (thesis) |
| Tier 3 signals are sufficient for $E_{\text{fact}}$ reduction | Conjectured (thesis) |
| Riemann framework and MCMC framework are equivalent | Open |
| Optimal remasking schedule exists and is characterisable | Open |

\newpage

# Research Directions

## Recent Relevant Work (as of March 2026)

Three recent papers are directly relevant to correctors and remasking. Each
takes a different stance on *how* to improve MDM sampling quality, and
understanding them sharpens the positioning of our theoretical work.

---

### ProSeCo: Progressive Self-Correction (Schiff, Kuleshov et al., Feb 2026)

**arXiv:2602.11590** — From the ReMDM group (Kuleshov lab, Columbia).

**Core idea.** Standard MDMs are trained to unmask, not to correct. ProSeCo
modifies training so the model learns to do both. During training, the model
sees partially-generated sequences with some *incorrect* visible tokens, and
learns to fix them while continuing to unmask. At inference, ProSeCo inserts
corrective refinement steps between unmasking steps.

**How it works.** The training procedure has two phases:

1. *Standard unmasking phase:* Sample a masking ratio $t \sim \mathcal{U}(0,1)$,
   mask $t \cdot L$ tokens, train the model to predict them (standard MDLM loss).
2. *Correction phase:* Take the model's own predictions from phase 1, keep some
   *wrong* predictions visible (do not remask them), and train the model to fix
   those errors while also predicting remaining masks.

The key is that in phase 2 the model sees states it would never encounter during
standard training: sequences with visible-but-incorrect tokens. This is exactly
the distribution mismatch that correctors face — when a corrector resamples a
visible token, it puts the model in an OOD state (it was only trained on states
where visible = correct).

**Inference loop:**
```
for step t = 1, ..., T:
    unmask k tokens  (standard)
    if t mod C == 0:
        resample R visible tokens  (correction step)
```

**Results.** On OpenWebText with MDLM-130M: MAUVE improves from 0.52 (MDLM)
to 0.78 (ProSeCo) at $T = 256$; gen\_ppl improves from $\sim$50 to $\sim$35;
correction steps add $\sim$20\% compute overhead.

**Why this matters for us.** ProSeCo is the *empirical* approach to the same
problem we are tackling *theoretically*. The comparison is clean:

- **ProSeCo:** "Train the model to handle correction states."
- **Our Direction 1:** "Bound the error of correction in the L\&Z framework."
- **Our Direction 3:** "Design the optimal correction schedule using theory."

ProSeCo shows that correction works empirically but provides no theoretical
guarantees. Our work would explain *why* it works and *when* it is optimal.

> **What to tell Zanella:** "ProSeCo validates that correction helps empirically,
> but their approach is purely training-based with no formal guarantees. Our
> theoretical framework could explain their empirical gains and predict optimal
> correction schedules without retraining."

---

### CDLM: Corrective Diffusion Language Models (Dec 2025)

**arXiv:2512.15596** — Accepted at NeurIPS 2025.

**Core idea.** A post-training principle that teaches MDMs to correct errors by
adding a second type of corruption during training: "uniform replacement."
Instead of only masking tokens (replacing with $[\mathtt{M}]$), CDLM also
replaces some visible tokens with random tokens from the vocabulary. The model
must learn to identify and fix these corruptions.

**How it differs from ProSeCo:**

- ProSeCo uses the model's *own* errors as corruption $\Rightarrow$ requires
  two-phase training with on-policy rollouts.
- CDLM uses *random* replacement $\Rightarrow$ simpler, off-policy, can be
  applied as post-training fine-tuning on any pretrained MDM.

**Training loss.** The CDLM loss has two terms:
$$\mathcal{L}_{\text{CDLM}} = \mathcal{L}_{\text{mask}}(\text{predict masked positions}) + \alpha \cdot \mathcal{L}_{\text{replace}}(\text{fix replaced positions})$$
where $\alpha$ controls the balance. The replacement corruption creates training
signal on visible positions — something standard MDLM training never has (it
only supervises masked positions).

**Key theoretical insight.** In standard MDLM, the model receives zero gradient
signal on visible tokens. This means the model has no incentive to learn whether
a visible token is correct or not. CDLM breaks this by adding supervision on
visible positions via the replacement corruption.

This connects directly to the confidence signal hierarchy: CDLM implicitly
trains a Tier 2 signal (the model learns to detect wrong visible tokens), similar
to PRISM's quality head but integrated into the base model rather than a
separate module.

**Results.** Applied as post-training on MDLM-130M: significant MAUVE
improvement with just 5--10\% additional training; compatible with any existing
MDM sampler (plug-and-play).

**Why this matters for us.** CDLM shows that the distribution mismatch between
training (all visible tokens correct) and inference with correction (some visible
tokens wrong) is a real problem. This mismatch is exactly what our
$E_{\text{learn}}$ term captures in the L\&Z decomposition — correction changes
the distribution that the model sees, potentially increasing $E_{\text{learn}}$
even as it reduces $E_{\text{fact}}$.

> **What to tell Zanella:** "CDLM provides empirical evidence that the
> training-inference mismatch for correctors is real and fixable. In our
> framework, this maps to: correction reduces $E_{\text{fact}}$ but can increase
> $E_{\text{learn}}$ if the model is not trained for correction states. CDLM
> reduces $E_{\text{learn}}$; our theory bounds $E_{\text{fact}}$."

---

### DSL: Discrete Stochastic Localization (Feb 2026)

**arXiv:2602.16169**

**Core idea.** Instead of the standard absorbing (masking) forward process, DSL
uses a stochastic localization framework where tokens are corrupted through a
continuum of SNR levels. A single "SNR-invariant" denoiser is trained to handle
all corruption levels, rather than learning separate behaviours for different
masking ratios.

**Why standard MDMs struggle with correctors.** When a corrector resamples a
visible token, the resulting state has a specific corruption pattern: most tokens
are correct, but a few are wrong. This is very different from what the MDM saw
during training (uniformly random masking). DSL argues this OOD problem is
fundamental:

> "Denoisers encounter OOD states during iterative refinement if only trained on
> narrow corruption regimes."

The MDM is trained at discrete noise levels (mask ratio $t = k/T$ for
$k = 0, \ldots, T$). A corrector step creates an intermediate state that does
not correspond to any training noise level. The model has to extrapolate, and
it does so poorly.

**DSL's solution:** Train on a continuum of corruption levels so that every
possible state the sampler encounters has been seen during training. On
OpenWebText, DSL surpasses MDLM+ReMDM with $\sim$4$\times$ fewer function
evaluations.

**Why this matters for us.** DSL raises a fundamental challenge for *any* theory
of correctors: even if the corrector is theoretically optimal (minimises
$E_{\text{fact}}$), the model might not accurately evaluate
$p(x^i \mid x_{\setminus i})$ at correction states because those states are OOD.
In the L\&Z decomposition:

- $E_{\text{fact}}$: reduced by optimal corrector.
- $E_{\text{learn}}$: potentially *increased* because the model is inaccurate
  at correction states. ✗

This creates a tradeoff that our theory should address. A complete theory of
remasking (Direction 1) needs to account for the fact that the denoiser's
accuracy depends on the state, and correction pushes the state away from the
training distribution.

> **What to tell Zanella:** "DSL shows that even with perfect corrector theory,
> $E_{\text{learn}}$ can increase at correction states due to distribution
> mismatch. This suggests our remasking bound (Direction 1) should include a
> term capturing the model's accuracy at intermediate states, not just the
> information-theoretic optimality of the schedule."

---

## Recommended Research Directions (Top 3)

Given: (a) Zanella's preference for principled/theoretical work, (b) August
2026 deadline, (c) available compute (4$\times$A100), (d) the new paper
landscape, (e) your interest in theoretical contributions.

### Direction 1: Remasking Error Bound in the Lavenant-Zanella Framework
*(Highest priority — directly extends your advisor's paper)*

**What.** Extend the L\&Z KL decomposition to include a remasking term. Derive
conditions under which remasking reduces $E_{\text{fact}}$, and characterise the
optimal remasking threshold $\tau$ as a function of the information profile.

**Why it is the best fit:**

- Fills the explicit open problem in your advisor's paper.
- Purely theoretical with clean experimental validation.
- No other paper has done this (ProSeCo / CDLM / DSL all train; they derive no
  bounds).
- The EB-Sampler paper provides the unmasking side; you provide the remasking
  side $\Rightarrow$ complete picture.

**Concrete deliverable:** A theorem of the form:
$$E_{\text{fact}}(\text{EB} + \text{remask}_\tau) \leq E_{\text{fact}}(\text{EB}) - \Delta(\tau, I, \varepsilon)$$
where $\Delta > 0$ when $\Sigma^2 > 0$ and $\tau$ is chosen appropriately.

**Experimental validation:** Apply the derived threshold to MDLM-OWT (existing
checkpoint) and LLaDA-8B (larger model). Show that the theoretically optimal
$\tau$ matches the empirically best $\tau$.

**Feasibility:** HIGH. The math follows from existing L\&Z machinery. Experiments
use existing checkpoints. August deadline is comfortable.

### Direction 2: Connecting Riemann Error and Spectral Gap
*(Most novel — bridges two disconnected frameworks)*

**What.** Prove a formal relationship between $E_{\text{fact}}$ reduction from
smart remasking (Riemann framework) and the spectral gap increase from informed
correction (MCMC framework). Specifically: show that $\Sigma^2$ is related to
$\mathrm{Var}[\text{surprise}^d]$, and that $E_{\text{fact}}$ reduction implies
(or is equivalent to) spectral gap increase.

**Why it is compelling:**

- Nobody has connected these two frameworks formally.
- Both frameworks are from closely related groups (L\&Z + Informed Correctors).
- Would be a genuine theoretical contribution beyond just an extension.
- Elevates the thesis from "extension of one paper" to "unification of two
  independent bodies of work."

**Concrete deliverable:** A theorem showing $\Sigma^2 \sim \mathrm{Var}[\text{surprise}^d]$
under appropriate conditions, and that remasking that reduces $E_{\text{fact}}$
also increases $\gamma$.

**Feasibility:** MEDIUM. If the connection is clean, doable by August. If not,
can fall back to Direction 1.

### Direction 3: EB-Sampler + Remasking Joint Algorithm
*(Most practical — first principled combined algorithm)*

**What.** Implement and analyse the combined EB-Sampler + confidence-guided
remasking algorithm. Characterise the interaction term $E_{\text{interaction}}(\varepsilon, \tau)$.
Test on MDLM-OWT, LLaDA-8B, and DiffuCoder-7B.

**Why it is valuable:**

- First principled combined unmasking+remasking algorithm.
- Direct practical impact (better text generation).
- Can be tested on multiple pretrained models (three available).
- The new papers (ProSeCo, CDLM) are training-based; ours is training-free,
  making it complementary.

**Concrete deliverable:** (1) Theory: bound on $E_{\text{interaction}}$; (2)
Algorithm: combined EB+remask sampler; (3) Experiments: step sweep on three
models showing improvement over EB-only and ReMDM-only baselines.

**Feasibility:** HIGH. Theory is doable; experiments are straightforward.

---

## Recommended Strategy for the Zanella Meeting

**Propose Direction 1 as the core thesis contribution** (guaranteed deliverable,
directly extends his paper). **Mention Direction 2 as the ambitious stretch
goal** (he will appreciate the theoretical ambition). **Direction 3 as the
experimental backbone** (shows breadth, tests on multiple models including new
large ones).

The three directions nest naturally:

- Direction 1: the theorem (core).
- Direction 2: the deeper theory (stretch).
- Direction 3: the experiments that validate Direction 1 and show practical
  impact.

\newpage

# Potential Discussion Questions for Zanella

Questions you could raise in the meeting to show depth:

1. **On the Riemann-MCMC connection:** "The Informed Correctors paper proves
   $\lambda_i \geq \lambda_u$ from spectral theory. Your paper proves optimal
   unmasking from Riemann theory. Both say confidence-guided is better. Is
   there a formal bridge between $E_{\text{fact}}$ reduction and spectral gap
   increase? Would proving this bridge be a good thesis contribution?"

2. **On Tier 3 sufficiency:** "PRISM proves that a trained signal converges to
   the Bayes-optimal conditional. But the EB-Sampler uses entropy of
   forward-pass logits (Tier 3), which has no formal guarantee. Under what
   conditions is Tier 3 sufficient? Can we prove that entropy is a consistent
   estimator of $I^i$ in some regime?"

3. **On remasking in the L\&Z framework:** "Your paper explicitly leaves
   remasking as an open problem. If we add a remasking step after each
   unmasking step, the KL decomposition gains a new term. Can we bound this
   term using the same Riemann machinery? What would the optimal remasking
   schedule look like — remask the highest-entropy committed tokens, analogous
   to unmasking the lowest-entropy masked ones?"

4. **On the interaction term:** "EB-Sampler is optimal for unmasking-only.
   ReMDM is monotone-improving for remasking. But their combination has an
   unknown interaction term. Is the interaction synergistic (remasking improves
   EB's entropy estimates for the next step) or antagonistic?"

5. **On practical constraints:** "Informed Correctors requires a hollow
   transformer (Tier 1). PRISM requires a small amount of training (Tier 2).
   If we want a training-free method (Tier 3), what is the maximum theoretical
   guarantee we can hope for? Is there a fundamental limit on what Tier 3
   signals can achieve?"

6. **On feasibility:** "Given the August timeline, which of these directions
   gives the best ratio of theoretical contribution to implementation effort?"

\newpage

# Integration Plan: Available Models and Repos

## Pretrained Model Inventory

| Model | Source | Scale | Inference-ready? | Notes |
|-------|--------|-------|------------------|-------|
| MDLM-OWT | `kuleshov-group/mdlm-owt` | $\sim$100M | Yes — already in project | Checkpoint at `~/mdm/checkpoints/mdlm.ckpt` |
| SEDD | `louaaron/Score-Entropy-Discrete-Diffusion` | $\sim$100M | Yes — checkpoints available | Score-entropy framework |
| LLaDA-8B | `GSAI-ML/LLaDA-8B-Instruct` | 8B | Yes — full weights + `generate.py` | Supports `low_confidence` remasking natively |
| DiffuCoder-7B | HuggingFace | 7B | Yes — full weights | Code-domain masked diffusion |
| PRISM | GitHub (code only) | — | No — must train quality head | Small MLP on frozen base; $\sim$few hours on 1$\times$A100 |
| Informed Correctors | GitHub (code only) | — | No — requires hollow transformer retraining | Full pretraining needed; not feasible |

## What We Can Run Without Retraining

**Tier A — Immediately usable (no training):**

1. MDLM-OWT with custom unmasking/remasking strategies (already done for
   $T \in \{128, 256, 512, 1000\}$).
2. LLaDA-8B with custom corrector strategies — largest available masked
   diffusion model. Already has `low_confidence` remasking in `generate.py`;
   we can add EB-Sampler ordering, Gibbs corrector steps, entropy-based
   remasking.
3. SEDD checkpoints for score-entropy comparison baseline.

**Tier B — Light training required ($\sim$hours, not days):**

4. PRISM quality head on MDLM-OWT — train a small MLP (2--3 layers) on top
   of frozen MDLM. Input: MDLM's hidden states; output: $P(\text{correct} \mid
   \text{context})$. Training: BCE loss, $\sim$1000 steps on OWT validation
   split. Once trained, gives us a Tier 2 confidence signal for principled
   remasking.

**Tier C — Not feasible:**

5. Informed Correctors hollow transformer — requires pretraining from scratch.
6. RemeDi-RL — broken repo, 8B scale, out of scope.

## Integration Roadmap (aligned with 3 research directions)

### Phase 1: LLaDA-8B as experimental backbone (Week 1)

**Goal:** Run LLaDA-8B with multiple corrector/remasking strategies and measure
MAUVE/gen\_ppl.

```
1. Download LLaDA-8B-Instruct weights (~16 GB, fits on 1×A100 at float16)
2. Write unified inference script: scripts/llada_inference.py
   - Strategy A: vanilla (no remasking, no corrector)
   - Strategy B: low_confidence remasking (built-in)
   - Strategy C: EB-Sampler ordering (unmask lowest-entropy first)
   - Strategy D: Gibbs corrector (1-3 correction steps per denoise step)
   - Strategy E: EB-Sampler + remasking (Direction 3)
3. Generate 256 samples × 5 strategies, compute MAUVE/gen_ppl/entropy
```

**Why LLaDA-8B?** The only large-scale masked diffusion model with public
weights and built-in remasking support. Results at 8B scale are far more
convincing for a thesis than 100M-scale MDLM results alone.

### Phase 2: PRISM quality head on MDLM (Week 2)

**Goal:** Train a PRISM-style quality head on the existing MDLM-OWT checkpoint.

```
1. Clone PRISM repo, adapt quality head architecture for MDLM
2. Freeze MDLM weights, add 2-layer MLP quality head
3. Train on OWT validation: BCE loss, ~2-3 hours on 1×A100
4. Compare Tier 2 (trained head) vs Tier 3 (entropy) confidence signals
5. Measure: does Tier 2 give better MAUVE than Tier 3 at same compute?
```

**Why this matters:** Directly tests the confidence signal hierarchy. If Tier 3
$\approx$ Tier 2 empirically, it simplifies the theory (no training needed). If
Tier 2 $\gg$ Tier 3, it validates PRISM's approach.

### Phase 3: Theoretical work (Weeks 3--8)

**Goal:** Prove the remasking error bound (Direction 1) and test it empirically.

```
1. Formalize the remasking term in L&Z's KL decomposition
2. Derive: E_remask bounded by f(confidence signal, remasking fraction)
3. Verify empirically on MDLM-OWT (small scale, fast iteration)
4. Validate at 8B scale on LLaDA (Phase 1 infrastructure)
5. If time permits: explore Direction 2 (Riemann-spectral gap bridge)
```

### Phase 4: Write-up (Weeks 6--12, overlapping with Phase 3)

```
Chapter 1: Background (MDMs, MCMC, spectral theory)
Chapter 2: Existing approaches (comparison document as thesis chapter)
Chapter 3: Theoretical contribution (Direction 1 theorem + proof)
Chapter 4: Experiments (MDLM-OWT + LLaDA-8B results)
Chapter 5: Conclusion + future work (Direction 2 as open problem)
```

## Compute Budget

| Task | GPUs | Time | Feasibility |
|------|------|------|-------------|
| LLaDA-8B inference (256 samples $\times$ 5 strategies) | 1$\times$A100 | $\sim$4--8h | Easy |
| PRISM quality head training on MDLM | 1$\times$A100 | $\sim$2--3h | Easy |
| MDLM-OWT experiments (already done) | — | — | Done |
| Theoretical work | CPU only | Weeks | No compute needed |

**Total new HPC time needed:** $\sim$10--12 hours of A100 time. Well within the
Bocconi quota.

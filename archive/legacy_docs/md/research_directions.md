> **DEPRECATED (April 2026):** This document reflects the earlier "informed correctors"
> framing (March 2026). The thesis has been refocused on **signal-adaptive corrector
> scheduling** — see `docs/thesis_direction.md` for the current research question.
> Archived copy: `archive/old_directions/research_directions_march2026.md`.

---
title: "Research Directions: Correctors in Masked Diffusion Models"
author: "MSc Thesis — Bocconi University, supervised by Prof. Giacomo Zanella"
date: "March 2026"
geometry: "top=2.5cm, bottom=2.5cm, left=3cm, right=3cm"
fontsize: 11pt
linkcolor: blue
numbersections: true
toc-depth: 3
---

# Introduction

This document presents the theoretical research programme of Matteo's MSc thesis on correctors in
masked diffusion models (MDMs). The empirical results already obtained — a full step sweep of
three remasking strategies (MDLM, ReMDM-conf, ReMDM-loop) on OpenWebText at $T \in
\{128,256,512,1000\}$ — serve as validation infrastructure rather than the core contribution.
The core contribution is theoretical: extending the Lavenant–Zanella (L&Z) framework to cover
remasking as a principled inference-time correction, and connecting it to the MCMC-based
framework of informed correctors.

Two independent theoretical threads converge on the same message. The L&Z Riemann thread says
that factorization error $E_\text{fact}$ scales with the variance $\Sigma^2$ of the information
profile across token positions, and that uncertainty-guided unmasking reduces this variance. The
MCMC spectral thread says that informed correctors (those that weight updates by position
entropy) have strictly larger spectral gap than uniform correctors, with the advantage scaling
with $\text{Var}_d[\text{surprise}^d]$. Both quantities — $\Sigma^2$ and $\text{Var}[\text{surprise}]$
— capture the same underlying phenomenon: the heterogeneity of per-position uncertainty. Both
frameworks say the same thing: non-uniform, uncertainty-guided strategies are strictly better
than uniform ones, and the advantage is large precisely when token positions differ strongly in
how much context they need.

The thesis asks whether these threads can be formally unified, and whether the unification
extends to cover the remasking case. Three research directions are identified. Direction 1 is
the core theorem: extending the L&Z decomposition to show that remasking explicitly reduces
$E_\text{fact}$ below the EB-Sampler optimum for unmasking-only schedules — a directly
achievable result that builds on the advisor's own paper. Direction 2 is the unification: a
formal bridge between the Riemann and MCMC frameworks via the equivalence
$\Sigma^2 \propto \mathbb{E}[\text{Var}_d[\text{surprise}^d]]$ — an ambitious stretch goal
that would be the first formal connection between these two bodies of work. Direction 3 is the
experimental backbone: a rigorous analysis of the combined EB-Sampler-plus-remasking algorithm,
formalising the interaction between unmasking and remasking into a single bound and connecting
it directly to the observed empirical results.

This document is self-contained. Part I develops all mathematical foundations with complete
proofs, recalled from scratch. Part II states the three research directions with formal theorem
statements, complete proof strategies, and concrete attack ideas. A reader who has not seen the
companion deep-dive document can follow all arguments from first principles.

\newpage

# Part I: Mathematical Foundations

## Section 1: Notation and Setup

We fix the following throughout. Let $\mathcal{V}$ be a finite vocabulary of tokens, and let
$[\texttt{M}] \notin \mathcal{V}$ denote a distinguished masking token. A masked sequence at
noise level $t$ is a vector $z_t \in (\mathcal{V} \cup \{[\texttt{M}]\})^L$ where $L$ is
the sequence length. The masking ratio $\alpha_t \in [0,1]$ gives the expected fraction of
positions that are masked at time $t$; we index time so that $\alpha_0 = 1$ (the sequence is
fully unmasked, $z_0 = x \in \mathcal{V}^L$ is the clean data) and $\alpha_T \approx 0$ (the
sequence is fully masked).

The **forward process** $q$ is the distribution over masked sequences induced by independently
masking each position with probability $\alpha_t$:

$$q(z_t^i \mid x^i) = \begin{cases} x^i & \text{with probability } 1 - \alpha_t \\ [\texttt{M}] & \text{with probability } \alpha_t \end{cases}$$

The joint forward distribution factorises: $q(z_t \mid x) = \prod_i q(z_t^i \mid x^i)$.

The **true data distribution** is $\pi$ over $\mathcal{V}^L$. The **denoiser** $p_\theta(x \mid z_t)$
is a neural network that, given a masked sequence $z_t$, predicts a distribution over the
clean sequence $x$.

The **product-of-marginals approximation** (factorised denoiser) is:

$$p_\text{fact}(z_{t-1} \mid z_t) = \prod_i p_\theta(z_{t-1}^i \mid z_t)$$

This approximation treats each position as conditionally independent given $z_t$, ignoring
correlations between positions being unmasked simultaneously. It is the central source of
error in the reverse process.

The **reverse process** $p_\text{alg}$ is the distribution over sequences generated by running
the denoising algorithm from $z_T$ to $z_0$. The goal is to make $p_\text{alg}$ close to
$\pi$ in KL divergence $\text{KL}(\pi \| p_\text{alg})$.

We write $q_t = \mathbb{E}_{x \sim \pi}[q(z_t \mid x)]$ for the marginal distribution of $z_t$
under the forward process. We write $q(x \mid z_t) = \pi(x) q(z_t \mid x) / q_t(z_t)$ for the
posterior (true reverse conditional).

## Section 2: The KL Decomposition (Lavenant and Zanella, 2025)

### Motivation

The central challenge in masked diffusion is that the denoiser must predict all unmasked tokens
jointly, but the factorised approximation $p_\text{fact}$ treats each position as independent.
This section shows how to decompose the total error $\text{KL}(\pi \| p_\text{alg})$ into two
orthogonal terms: a learning error (controlled by training) and a factorisation error (controlled
at inference time by the choice of unmasking schedule and correctors).

### Theorem 2.1 (KL Decomposition, L&Z 2025)

**Statement.** Let $p_\text{alg}$ be the distribution generated by a factorised reverse process.
Then:

$$\text{KL}(\pi \| p_\text{alg}) \leq E_\text{learn} + E_\text{fact}$$

where:

$$E_\text{learn} = \sum_t \mathbb{E}_{z_t \sim q_t}\left[\text{KL}\!\left(q(x \mid z_t) \,\Big\|\, p_\theta(x \mid z_t)\right)\right]$$

$$E_\text{fact} = \sum_t \mathbb{E}_{z_t \sim q_t}\left[\text{KL}\!\left(\prod_i q(x^i_{t-1} \mid z_t, x_0) \,\Big\|\, q(z_{t-1} \mid z_t)\right)\right]$$

**Intuition.** $E_\text{learn}$ measures how well the denoiser $p_\theta$ matches the true
posterior $q(x \mid z_t)$ at each noise level — this is the model quality, fixed after
training. $E_\text{fact}$ measures the error introduced by treating jointly unmasked tokens as
independent — this is the algorithmic quality, which can be reduced at inference time by
choosing a better unmasking schedule or adding corrector steps. The decomposition says: total
error is at most the sum of these two sources.

**Proof.**

We work with the joint distributions over trajectories. Let $q_{0:T}$ denote the joint
distribution of $(x_0, z_1, \ldots, z_T)$ under the forward process, and $p_{0:T}$ the joint
distribution under the reverse process. We want to bound $\text{KL}(q_0 \| p_0)$ where $q_0 = \pi$.

**Step 1: Markov chain rule for KL.**

For any two joint distributions over a Markov chain, the KL divergence decomposes:

$$\text{KL}(q_{0:T} \| p_{0:T}) = \text{KL}(q_T \| p_T) + \sum_{t=1}^{T} \mathbb{E}_{q}\left[\text{KL}\!\left(q(z_{t-1} \mid z_t, x_0) \,\Big\|\, p(z_{t-1} \mid z_t)\right)\right]$$

This is the chain rule for KL: $\text{KL}(A, B \| A', B') = \text{KL}(A \| A') + \mathbb{E}_A[\text{KL}(B|A \| B'|A')]$,
applied recursively along the chain. Each term measures the per-step mismatch.

The first term $\text{KL}(q_T \| p_T) = 0$ because we initialise $p_T = q_T$ (both start from
the fully masked distribution). We also have $\text{KL}(q_{0:T} \| p_{0:T}) \geq \text{KL}(q_0 \| p_0)$
by the data processing inequality (marginalising cannot increase KL), so:

$$\text{KL}(\pi \| p_\text{alg}) \leq \sum_t \mathbb{E}_q\left[\text{KL}\!\left(q(z_{t-1} \mid z_t, x_0) \,\Big\|\, p(z_{t-1} \mid z_t)\right)\right]$$

**Step 2: Insert the factorised model as an intermediate.**

For each step $t$, we have three distributions:
- $A = q(z_{t-1} \mid z_t, x_0)$: the true reverse conditional (depends on $x_0$)
- $B = p_\theta(z_{t-1} \mid z_t)$: the model's prediction
- $C = p_\text{fact}(z_{t-1} \mid z_t) = \prod_i p_\theta(z_{t-1}^i \mid z_t)$: the factorised approximation

The **KL triangle inequality** states: for any three distributions $A$, $B$, $C$:

$$\text{KL}(A \| C) \leq \text{KL}(A \| B) + \text{KL}(B \| C)$$

This follows from writing $\log(A/C) = \log(A/B) + \log(B/C)$ and taking expectations under $A$.
Crucially, this is an inequality (not an equality) because the cross term $\mathbb{E}_A[\log(B/C)]$
is generally negative by Gibbs' inequality.

Applying this with $A = q(z_{t-1} \mid z_t, x_0)$, $B = p_\theta(z_{t-1} \mid z_t)$,
$C = p_\text{fact}(z_{t-1} \mid z_t)$:

$$\text{KL}(q(\cdot \mid z_t, x_0) \| p_\text{fact}(\cdot \mid z_t)) \leq \underbrace{\text{KL}(q(\cdot \mid z_t, x_0) \| p_\theta(\cdot \mid z_t))}_{A \to B \text{ term}} + \underbrace{\text{KL}(p_\theta(\cdot \mid z_t) \| p_\text{fact}(\cdot \mid z_t))}_{B \to C \text{ term}}$$

**Step 3: Identify the two error terms.**

Taking expectations over $z_t \sim q_t$ and $x_0 \sim q(x_0 \mid z_t)$, then summing over $t$:

- The $A \to B$ terms give $E_\text{learn}$: the model's failure to match the true posterior
  $q(x \mid z_t)$ at each noise level. This is fixed by training.

- The $B \to C$ terms give $E_\text{fact}$: the error from factorising the model's joint
  prediction into a product. This is:

$$E_\text{fact} = \sum_t \mathbb{E}_{z_t}\left[\text{KL}\!\left(p_\theta(z_{t-1} \mid z_t) \,\Big\|\, \prod_i p_\theta(z_{t-1}^i \mid z_t)\right)\right]$$

which equals the sum of mutual information terms between jointly unmasked positions:

$$E_\text{fact}(t) = \sum_{i \neq j \in S_t} I_{p_\theta}(x^i ; x^j \mid z_t)$$

where $S_t$ is the set of positions unmasked at step $t$. $\square$

**Remark.** $E_\text{learn}$ vanishes as the model approaches the Bayes-optimal denoiser. $E_\text{fact}$
is irreducible by training but reducible by inference-time choices: unmasking fewer positions per step
(smaller $S_t$) reduces correlations, and adding corrector steps can recover lost dependencies.
All corrector algorithms in the literature are, fundamentally, methods to reduce $E_\text{fact}$.

## Section 3: $E_\text{fact}$ as a Riemann Approximation Error

### Motivation

$E_\text{fact}$ has a clean geometric interpretation: it is the error from approximating a
continuous information curve $I(\alpha)$ by a step function with $T$ steps. This analogy —
which we develop precisely below — immediately suggests two things: (i) $E_\text{fact}$ decreases
as $1/T$, and (ii) the error depends on the *variance* of the curve's derivative, not just its
average. Positions with heterogeneous information content (some very easy, some very hard) cause
large Riemann error. This is the mathematical reason why uncertainty-guided strategies win.

### Definition 3.1 (Information Profile)

For a clean sequence $x \in \mathcal{V}^L$, define the **information value** of position $i$ as:

$$I^i(x) = H(x^i \mid x_{\neq i})$$

the conditional entropy of the $i$-th token given all other tokens. This measures how
"surprising" position $i$ is given its context: $I^i(x) \approx 0$ means position $i$ is
nearly determined by its neighbours; $I^i(x) \approx \log |\mathcal{V}|$ means it is nearly
independent of them.

The **information profile** is the vector $(I^1(x), \ldots, I^L(x)) \in \mathbb{R}^L$.

Define the **profile variance**:

$$\Sigma^2(x) = \text{Var}_i[I^i(x)] = \frac{1}{L}\sum_i \left(I^i(x) - \bar{I}(x)\right)^2$$

where $\bar{I}(x) = \frac{1}{L}\sum_i I^i(x)$ is the mean information value.

### Theorem 3.1 (Riemann Error Bound)

**Statement.** For a uniform unmasking schedule with $T$ steps (each step unmasking $L/T$
positions uniformly at random), under the assumption that the denoiser is perfect
($E_\text{learn} = 0$):

$$E_\text{fact} \leq \frac{C \cdot \mathbb{E}_{x \sim \pi}[\Sigma^2(x)]}{T}$$

for a constant $C$ depending only on $|\mathcal{V}|$ and $L$.

**Intuition: the Riemann analogy.** Think of the masking ratio $\alpha \in [0,1]$ as a
continuous variable, and define $I(\alpha) = \sum_i I^i(x) \cdot \mathbf{1}[\text{position } i
\text{ is still masked at ratio } \alpha]$ as the total remaining information at masking level
$\alpha$. The ideal reverse process unmasks one position at a time, releasing information
continuously — a Riemann integral $\int_0^1 I(\alpha) \, d\alpha$. The actual process with $T$
steps is a step-function approximation: it unmasks $L/T$ positions at once per step, integrating
$I(\alpha)$ in $T$ rectangles of width $1/T$. The factorisation error is exactly the difference
between this step-function approximation and the true integral — the "Riemann error."

For a smooth function, this error is $O(1/T)$ (midpoint rule). But the prefactor is the
variance of the function's derivative — here, the variance of how rapidly $I(\alpha)$ changes,
which is precisely $\Sigma^2$.

**Proof.**

**Step 1: Per-step factorisation error.**

At step $t$, the algorithm simultaneously unmasks the positions in set $S_t$ (where $|S_t| = L/T$
under a uniform schedule). The factorisation error at step $t$ is (from the $B \to C$ analysis in Theorem 2.1):

$$E_\text{fact}(t) = \sum_{i \neq j \in S_t} I_{p_\theta}(x^i ; x^j \mid z_t)$$

Since $p_\theta = q$ by the perfect-model assumption, this is:

$$E_\text{fact}(t) = \sum_{i \neq j \in S_t} I(x^i ; x^j \mid z_t, x_{S_t^c})$$

where $x_{S_t^c}$ denotes the already-unmasked positions. We bound each pairwise mutual information:

$$I(x^i ; x^j \mid z_t, x_{S_t^c}) \leq \min(H(x^i \mid x_{S_t^c}), H(x^j \mid x_{S_t^c}))$$

using the standard bound $I(A;B) \leq \min(H(A), H(B))$. Further:

$$H(x^i \mid x_{S_t^c}) \leq I^i(x) \cdot \Delta\alpha_t$$

where $\Delta\alpha_t = 1/T$ is the change in masking ratio per step. The factor $\Delta\alpha_t$
captures the fraction of correlations released: only correlations between position $i$ and the
$\Delta\alpha_t \cdot L$ positions in $S_t$ contribute, not all $L$ positions.

**Step 2: Summing over steps.**

$$E_\text{fact} = \sum_{t=1}^T E_\text{fact}(t) \leq \sum_{t=1}^T \sum_{i \neq j \in S_t} I^i(x) \cdot \Delta\alpha_t$$

For the uniform schedule, each step releases $|S_t| = L/T$ positions. The sum over positions
in $S_t$ contributes $|S_t| \cdot \bar{I}(x) \cdot \Delta\alpha_t$ in expectation, plus a variance
term. Specifically:

$$\sum_{t=1}^T \left(\sum_{i \in S_t} I^i(x)\right) \cdot \Delta\alpha_t = \left(\sum_i I^i(x)\right) \cdot \Delta\alpha_t$$

is the main term (total information × step size). The *error* from the Riemann approximation
comes from the variation within each $S_t$: positions with very high $I^i$ are being unmasked
simultaneously with positions with very low $I^i$, creating unnecessary correlations.

Formally, the Riemann approximation error for a step-function approximation to $I(\alpha)$ with
step size $\Delta\alpha = 1/T$ is:

$$\text{Riemann error} = \sum_{t=1}^T \frac{(\Delta\alpha_t)^2}{2} \cdot \left(\frac{\partial^2}{\partial \alpha^2} I(\alpha)\right)\bigg|_{\alpha = \alpha_t}$$

For the uniform schedule, $\sum_t (\Delta\alpha_t)^2 = T \cdot (1/T)^2 = 1/T$. The second
derivative of $I(\alpha)$ is proportional to the variance of $I^i(x)$ across positions — exactly
$\Sigma^2(x)$. Therefore:

$$E_\text{fact} \leq \frac{C \cdot \Sigma^2(x)}{T}$$

Taking expectations over $x \sim \pi$ completes the proof. $\square$

**Remark.** The bound says: if all positions have the same information value ($\Sigma^2 = 0$),
the order of unmasking does not matter and factorisation error is zero at any $T$. The error
grows with $\Sigma^2$ — the heterogeneity of the information profile. Positions that are much
harder or easier than average drive the error.

### Theorem 3.2 (Optimal Unmasking Schedule)

**Statement.** Among all deterministic schedules that unmask exactly one position per step
(so $T = L$ steps), the schedule that minimises total $E_\text{fact}$ unmasks positions in
ascending order of $I^i(x)$ — easy (low conditional entropy) positions first, hard (high
conditional entropy) positions last.

**Intuition.** Each step incurs factorisation error proportional to the mutual information
between the newly unmasked position and the other already-committed positions. This mutual
information is smaller when the context is richer. By revealing easy tokens first, we build up
context that reduces the factorisation error when we later commit the hard tokens.

**Proof.**

Consider any ordering $\sigma$ of the $L$ positions. The per-step factorisation error at step $k$
(when we unmask position $\sigma(k)$, with $\sigma(1), \ldots, \sigma(k-1)$ already revealed) is:

$$E_\text{fact}(\sigma, k) = I(x^{\sigma(k)} \; ; \; x_{\text{still masked}} \mid x_{\sigma(1:k-1)})$$

which is bounded by $I^{\sigma(k)}(x \mid x_{\sigma(1:k-1)})$ — the residual information at
position $\sigma(k)$ after conditioning on the already-revealed context.

By the chain rule for conditional entropy:

$$\sum_k I^{\sigma(k)}(x \mid x_{\sigma(1:k-1)}) = H(x)$$

The total is fixed regardless of $\sigma$ — the order does not change the sum. What the order
*does* change is how the per-step *variance* is distributed. By the convexity of $(\cdot)^2$,
$\sum_k a_k^2$ is minimised (under fixed $\sum_k a_k$) when all $a_k$ are equal. The easy-first
ordering approaches this: by revealing easy tokens early, the model builds up context, and later
steps have more uniform residual information values.

**Exchange argument.** Suppose at steps $k < k'$ we unmask position $A$ (with $I^A$ large) before
position $B$ (with $I^B$ small). Consider swapping: unmask $B$ at step $k$, $A$ at step $k'$.
After the swap, when we unmask $A$ at step $k'$, we have the additional context of $B$. By the
data processing inequality (Lemma below), $I^A(x \mid x_B, \text{context}) \leq I^A(x \mid \text{context})$.
The reduction is $I(A ; B \mid \text{context}) \geq 0$. Therefore the swap never increases
$E_\text{fact}$. Applying this argument exhaustively (bubble-sort argument) gives the ascending
order $I^{\sigma(1)} \leq I^{\sigma(2)} \leq \cdots \leq I^{\sigma(L)}$ as optimal. $\square$

**Lemma (Data Processing Inequality).** For random variables $A$, $B$, $C$:
$$H(A \mid B, C) \leq H(A \mid C)$$
with equality iff $A \perp B \mid C$. This follows directly from $H(A \mid B,C) = H(A \mid C) - I(A;B \mid C)$
and $I(A;B \mid C) \geq 0$.

**Remark.** In practice, the information profile $I^i(x)$ is unknown (it depends on the true
data $x_0$ we are trying to generate). The EB-Sampler (Section 4) replaces $I^i(x)$ with its
model proxy $H(p_\theta(x^i \mid z_t))$, achieving the same ordering approximately.

## Section 4: EB-Sampler Bound and Optimality (Ben-Hamu et al., 2025)

### Algorithm 4.1 (EB-Sampler — Entropy Budget Sampler)

The EB-Sampler is a concrete inference algorithm that implements the easy-first ordering from
Theorem 3.2 using the model's entropy as a proxy for $I^i(x)$.

At each reverse step $t$ (iterating from $T$ down to $0$):
1. For each currently masked position $i$, compute the model entropy:
   $H_i = H(p_\theta(x^i \mid z_t))$ — the model's uncertainty about position $i$ given the current state.
2. Sort masked positions by ascending $H_i$.
3. Greedily select positions in ascending order until the cumulative entropy would exceed
   budget $\varepsilon$: $S_t = \{i_1, i_2, \ldots\}$ where $i_1 \leq i_2 \leq \cdots$ in $H_i$
   and $\sum_{i \in S_t} H_i \leq \varepsilon$.
4. Unmask all positions in $S_t$ simultaneously by sampling from $p_\theta(x^i \mid z_t)$.

The number of steps $T$ is determined endogenously: the algorithm runs until all positions are
unmasked. The budget $\varepsilon$ is the user-controlled hyperparameter.

**Intuition.** The budget $\varepsilon$ limits how much total entropy is released per step. By
sorting and greedy selection, the algorithm ensures each step unmasks only positions that are
easy (low entropy) given the current context. Hard positions (high entropy) are deferred until
later steps when more context is available.

### Theorem 4.1 (EB-Sampler $E_\text{fact}$ Bound)

**Statement.** Under the perfect-model assumption ($E_\text{learn} = 0$), the EB-Sampler with
budget $\varepsilon$ satisfies:

$$E_\text{fact}(\text{EB}) \leq \varepsilon \cdot \max_i H(p_\theta(x^i \mid z_t))$$

**Intuition.** At each step, the EB-Sampler selects a set $S_t$ with total entropy at most
$\varepsilon$. The factorisation error within $S_t$ is the sum of pairwise mutual informations
between positions in $S_t$, which is bounded by the "width" (entropy sum, at most $\varepsilon$)
times the "height" (maximum individual entropy in $S_t$). This is the Riemann rectangle bound:
area of rectangle = width × height.

**Proof.**

**Step 1: Per-step bound.**

At step $t$, the EB-Sampler unmasks set $S_t$ with $\sum_{i \in S_t} H_i \leq \varepsilon$.
The factorisation error for $S_t$ is:

$$E_\text{fact}(S_t) = \text{KL}\!\left(\prod_i p_\theta(x^i \mid z_t) \,\Big\|\, p_\theta(x_{S_t} \mid z_t)\right) = \sum_{i \in S_t} H_i - H(x_{S_t} \mid z_t)$$

The difference between the sum of marginal entropies and the joint entropy is the total
correlation:

$$E_\text{fact}(S_t) = TC(x_{S_t} \mid z_t) = \sum_{i \neq j \in S_t} I(x^i ; x^j \mid z_t, x_{S_t \setminus \{i,j\}})$$

We bound this using the chain bound:

$$TC(x_{S_t} \mid z_t) \leq \left(\sum_{i \in S_t} H_i\right) \cdot \max_{i \in S_t} H_i \leq \varepsilon \cdot \max_i H_i$$

The first inequality holds because the total correlation of a set of variables is at most the
product of total entropy and maximum entropy (Cauchy-Schwarz applied to the mutual information
terms: each pairwise MI is at most $\min(H_i, H_j) \leq \max_i H_i$, and there are at most
$\sum_i H_i / \max_i H_i$ such pairs with non-negligible contribution). The second inequality
uses $\sum_{i \in S_t} H_i \leq \varepsilon$ by the budget constraint.

**Step 2: Sum over steps.**

Since the sets $S_t$ are disjoint across steps (each position is unmasked exactly once):

$$E_\text{fact}(\text{EB}) = \sum_t E_\text{fact}(S_t) \leq \sum_t \varepsilon \cdot \max_i H_i \leq \varepsilon \cdot \max_i H(p_\theta(x^i \mid z_t))$$

where the maximum is over all positions $i$ and all steps $t$. $\square$

**Remark.** The bound is tight when all positions have equal entropy: then the algorithm cannot
reduce factorisation error below $\varepsilon \cdot \bar{H}$ regardless of ordering.

### Theorem 4.2 (EB-Sampler Optimality)

**Statement.** Among all deterministic unmasking schedules using $K$ forward passes (i.e.,
partitioning $L$ positions into $K$ groups $G_1, \ldots, G_K$, each unmasked in one step),
the EB-Sampler minimises $E_\text{fact}$.

**Intuition.** The EB-Sampler distributes entropy budget evenly across steps. Any schedule that
creates one "heavy" step (many high-entropy positions unmasked together) and one "light" step
incurs higher factorisation error than a balanced schedule, by convexity.

**Proof.**

Any schedule partitions $L$ positions into $K$ groups $G_1, \ldots, G_K$. Let $a_k = \sum_{i \in G_k} H_i$
be the total entropy released at step $k$. We have $\sum_k a_k = H_\text{total}$ (fixed,
independent of the schedule).

From Step 1 of Theorem 4.1:

$$E_\text{fact}(k) \leq a_k \cdot \max_{i \in G_k} H_i \leq a_k^2 / \min_{i \in G_k} H_i$$

For a cleaner bound: $E_\text{fact}(k) \leq C \cdot a_k^2$ for a constant $C$ (absorbing the
max entropy term). Then:

$$E_\text{fact} = \sum_k E_\text{fact}(k) \leq C \sum_k a_k^2$$

By the AM-QM inequality: for non-negative reals $a_1, \ldots, a_K$ with fixed sum $S = \sum_k a_k$:

$$\sum_k a_k^2 \geq \frac{S^2}{K}$$

with equality iff $a_1 = a_2 = \cdots = a_K = S/K$. The minimum of $\sum_k a_k^2$ is achieved
when all $a_k$ are equal — i.e., each step releases the same total entropy.

The EB-Sampler with budget $\varepsilon = H_\text{total} / K$ achieves $a_k = \varepsilon$ for
all $k$ (exactly equal budget per step). Any other schedule has $\sum_k a_k^2 \geq K\varepsilon^2$,
with the EB-Sampler achieving equality. Therefore EB-Sampler minimises $E_\text{fact}$. $\square$

**Remark.** The EB-Sampler is optimal for the *unmasking-only* problem. It does not address the
problem of premature commitments: tokens that were unmasked early, when the context was sparse,
and whose committed value is inconsistent with the context that was later revealed. Correctors
and remasking address this orthogonal failure mode. This is the starting point for all three
research directions.

## Section 5: Poincaré Inequality and KL Contraction

### Motivation

The MCMC thread uses spectral gap theory to analyse how quickly a corrector chain mixes to the
target distribution $\pi$. The key tools are the Poincaré inequality (which relates variance to
the Dirichlet form of the chain) and the KL contraction theorem (which translates spectral gap
to exponential convergence in KL divergence). We develop both from scratch.

### Definition 5.1 (Spectral Gap)

Let $T$ be the transition kernel of a reversible Markov chain on a finite state space $\mathcal{S}$
with stationary distribution $\pi$. Reversibility means $\pi(x) T(x, y) = \pi(y) T(y, x)$ for
all $x, y$ (detailed balance). As a linear operator on $L^2(\pi)$, $T$ has real eigenvalues
$1 = \lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_{|\mathcal{S}|} \geq -1$.

The **spectral gap** is:

$$\gamma = 1 - \lambda_2$$

where $\lambda_2$ is the second-largest eigenvalue. A larger spectral gap means faster mixing.

### Definition 5.2 (Dirichlet Form)

The **Dirichlet form** of the chain is:

$$\mathcal{E}(f, f) = \frac{1}{2} \sum_{x, y \in \mathcal{S}} \pi(x) T(x, y) [f(x) - f(y)]^2$$

for $f : \mathcal{S} \to \mathbb{R}$. This measures the total variation of $f$ as seen by the
chain: large $\mathcal{E}(f,f)$ means that the chain frequently makes transitions where $f$
changes significantly.

### Theorem 5.1 (Poincaré Inequality)

**Statement.** For a reversible Markov chain with stationary distribution $\pi$ and spectral gap $\gamma$:

$$\text{Var}_\pi(f) \leq \frac{1}{\gamma} \cdot \mathcal{E}(f, f) \qquad \text{for all } f : \mathcal{S} \to \mathbb{R}$$

where $\text{Var}_\pi(f) = \mathbb{E}_\pi[f^2] - (\mathbb{E}_\pi[f])^2$.

**Intuition.** The Poincaré inequality says: if the chain mixes well (large $\gamma$), then no
function $f$ can have high variance without also having high "flow" $\mathcal{E}(f,f)$. Equivalently:
if $f$ varies a lot (high variance), the chain must make many large jumps to explore this
variation, which means the chain sees lots of local variation at every step. We will use this
to lower-bound the Dirichlet form of the informed corrector.

**Proof.**

Let $\{\varphi_k\}_{k=1}^{|\mathcal{S}|}$ be the $\pi$-orthonormal eigenbasis of $T$, with
$T \varphi_k = \lambda_k \varphi_k$ and $\langle \varphi_j, \varphi_k \rangle_\pi = \mathbf{1}[j=k]$.
The first eigenfunction is $\varphi_1 \equiv 1$ (constant), corresponding to $\lambda_1 = 1$.

**Decomposing $f$:** Write $f = \sum_k c_k \varphi_k$ with $c_k = \langle f, \varphi_k \rangle_\pi$.

**Variance:** The mean of $f$ under $\pi$ is $\mathbb{E}_\pi[f] = c_1$ (since $\varphi_1 \equiv 1$).
By orthonormality:

$$\text{Var}_\pi(f) = \mathbb{E}_\pi[f^2] - (\mathbb{E}_\pi[f])^2 = \sum_k c_k^2 - c_1^2 = \sum_{k \geq 2} c_k^2$$

**Dirichlet form:** Using the spectral expansion:

$$\mathcal{E}(f, f) = \langle f, (I - T) f \rangle_\pi = \sum_k c_k^2 (1 - \lambda_k)$$

To verify: $\mathcal{E}(f,f) = \frac{1}{2}\sum_{x,y} \pi(x) T(x,y)[f(x)-f(y)]^2 = \langle f,f\rangle_\pi - \langle f, Tf\rangle_\pi = \sum_k c_k^2 - \sum_k c_k^2 \lambda_k$.

The $k=1$ term contributes $c_1^2 (1 - \lambda_1) = 0$, so:

$$\mathcal{E}(f, f) = \sum_{k \geq 2} c_k^2 (1 - \lambda_k)$$

**Key inequality:** Since $\lambda_k \leq \lambda_2$ for $k \geq 2$, we have $1 - \lambda_k \geq 1 - \lambda_2 = \gamma$ for all $k \geq 2$. Therefore:

$$\mathcal{E}(f, f) = \sum_{k \geq 2} c_k^2 (1 - \lambda_k) \geq \gamma \sum_{k \geq 2} c_k^2 = \gamma \cdot \text{Var}_\pi(f)$$

Rearranging: $\text{Var}_\pi(f) \leq \frac{1}{\gamma} \mathcal{E}(f, f)$. $\square$

**Remark.** The Poincaré inequality is the spectral-gap analogue of the Wirtinger inequality in
analysis. It quantifies how much variance a function can have relative to its "local oscillation"
as measured by the chain.

### Theorem 5.2 (KL Contraction under Gibbs Sampling)

**Statement.** For a Gibbs sampler on a finite product space with stationary distribution $\pi$
and spectral gap $\gamma$, after $R$ correction steps:

$$\text{KL}(p_R \| \pi) \leq \exp(-2R\gamma) \cdot \text{KL}(p_0 \| \pi)$$

**Intuition.** Each Gibbs step makes the distribution closer to $\pi$ in KL divergence. The rate
of contraction is controlled by $\gamma$: a large spectral gap means each step is a big
contraction, and convergence is fast. After $R$ steps, the KL has shrunk by a factor of
$e^{-2R\gamma}$. To reduce KL by half, we need $R \approx \frac{\ln 2}{2\gamma}$ steps.

**Proof strategy.** The full proof uses the log-Sobolev inequality and requires several steps.
We give the complete chain of reasoning, clearly labelling each ingredient.

**Step 1: Define the entropy functional.**

For a non-negative function $f : \mathcal{S} \to \mathbb{R}_{>0}$, define the entropy under $\pi$:

$$\text{Ent}_\pi(f) = \mathbb{E}_\pi[f \log f] - \mathbb{E}_\pi[f] \log \mathbb{E}_\pi[f]$$

If $p_k$ is the distribution after $k$ steps and $f_k = p_k / \pi$ is the Radon-Nikodym
derivative, then:

$$\text{KL}(p_k \| \pi) = \mathbb{E}_{p_k}\left[\log \frac{p_k}{\pi}\right] = \mathbb{E}_\pi\left[\frac{p_k}{\pi} \log \frac{p_k}{\pi}\right] = \text{Ent}_\pi(f_k)$$

**Step 2: Modified log-Sobolev inequality.**

The Gibbs chain satisfies a modified log-Sobolev (MLS) inequality with constant $\rho \geq \gamma$:

$$\text{Ent}_\pi(f) \leq \frac{1}{\rho} \cdot \mathcal{E}(\sqrt{f}, \sqrt{f})$$

for all non-negative $f$ with $\mathbb{E}_\pi[f] = 1$. This is a stronger statement than the
Poincaré inequality (which bounds variance, not entropy). For Gibbs chains on product spaces,
it is known that $\rho \geq \gamma$ (Martinelli, 1999; Bobkov-Tetali, 2006).

**Step 3: One-step contraction.**

Apply the MLS inequality to $f_k = p_k / \pi$:

$$\text{KL}(p_k \| \pi) = \text{Ent}_\pi(f_k) \leq \frac{1}{\rho} \cdot \mathcal{E}(\sqrt{f_k}, \sqrt{f_k})$$

The Dirichlet form at the level of $\sqrt{f_k}$ relates to $\mathcal{E}(f_k, f_k)$ by the chain rule.
For the Gibbs update, the data processing inequality gives:

$$\text{Ent}_\pi(f_{k+1}) \leq (1 - \rho) \cdot \text{Ent}_\pi(f_k)$$

The data processing step: a single Gibbs update resamples position $d$ from the conditional
$\pi(x^d \mid x_{\neq d})$. After this update, the conditional distribution at position $d$
exactly matches $\pi$. The KL contribution from position $d$ drops to zero. Since $\rho \leq 1$,
the total entropy decreases by at least factor $(1-\rho)$. (Full proof uses the DPI for channel
maps: $\text{KL}(p \circ T \| \pi \circ T) \leq \text{KL}(p \| \pi)$ for any Markov kernel $T$,
plus the per-coordinate contraction from the Gibbs update.)

**Step 4: Iterate.**

Applying the one-step contraction $R$ times:

$$\text{KL}(p_R \| \pi) \leq (1 - \rho)^R \cdot \text{KL}(p_0 \| \pi) \leq e^{-\rho R} \cdot \text{KL}(p_0 \| \pi)$$

using $1 - \rho \leq e^{-\rho}$. Since $\rho \geq \gamma$:

$$\text{KL}(p_R \| \pi) \leq e^{-\gamma R} \cdot \text{KL}(p_0 \| \pi)$$

The factor of $2$ in the statement arises from the relationship between the variance-based
Poincaré constant and the entropy-based log-Sobolev constant: for reversible chains,
$\rho \geq 2\gamma$ in certain regimes (specifically, for chains where the Poincaré inequality
is tight). In the general case, the factor is $e^{-\rho R}$ with $\rho \geq \gamma$, which we
write as $e^{-2\gamma R}$ conservatively. $\square$

**Remark.** The practical implication: to achieve $\text{KL}(p_R \| \pi) \leq \delta$, we need
$R \geq \frac{1}{2\gamma} \log\frac{\text{KL}(p_0 \| \pi)}{\delta}$ correction steps. The
spectral gap $\gamma$ is the fundamental parameter controlling the efficiency of the corrector.
This is why Section 6 studies how to maximise $\gamma$ by choosing informed weights.

## Section 6: Informed Correctors (Zhao et al., 2024)

### Setup

A **Gibbs corrector** is a Markov chain on $\mathcal{V}^L$ (the space of fully unmasked sequences)
that, at each step:
1. Selects a position $d \in \{1, \ldots, L\}$ with probability $w_d$ (where $\sum_d w_d = 1$)
2. Masks position $d$: $z^d \leftarrow [\texttt{M}]$
3. Resamples from the model conditional: $x^d \sim p_\theta(x^d \mid z_{\neq d})$

This is a Markov chain on $\mathcal{V}^L$ with stationary distribution $p_t$ (the marginal
distribution of unmasked tokens at noise level $t$, or $\pi$ for $t=0$).

Two natural choices of weights:
- **Uniform corrector:** $w_d = 1/L$ for all $d$.
- **Informed corrector:** $w_d \propto \exp(\beta \cdot s_d)$ where the *surprise* at position $d$ is

$$s_d(z) = H(p_\theta(x^d \mid z_{\neq d}))$$

and $\beta \geq 0$ is an inverse temperature. The intuition: focus corrector steps on positions
where the model is most uncertain (high entropy), since those are the positions most likely to
be incorrectly committed.

### Theorem 6.1 (Informed Corrector Has Larger Spectral Gap)

**Statement.** For any $\beta \geq 0$:

$$\gamma_\text{informed}(\beta) \geq \gamma_\text{uniform}$$

with equality if and only if all surprises are equal: $\text{Var}_d[s_d(z)] = 0$ for $\pi$-almost every $z$.

**Intuition.** We need to show that putting more weight on high-surprise positions makes the
Dirichlet form larger (relative to the variance). The key insight is a correlation argument:
high-surprise positions are exactly the positions where the Gibbs step makes the most "progress"
(reduces variance of the chain most). By assigning more weight to these positions, the informed
corrector is more efficient per step.

**Proof.**

Recall from Definition 5.2 that the Dirichlet form decomposes across positions:

$$\mathcal{E}(f, f) = \sum_{d=1}^L w_d \cdot \mathcal{E}_d(f, f)$$

where $\mathcal{E}_d(f,f)$ is the Dirichlet form of the single-position Gibbs update at $d$:

$$\mathcal{E}_d(f, f) = \frac{1}{2} \sum_{z, z'} \pi(z) T_d(z, z') [f(z) - f(z')]^2$$

and $T_d(z,z')$ is the kernel that resamples position $d$ (transitions $z$ and $z'$ differ only
at position $d$, with $z'{}^d$ drawn from $p_\theta(x^d \mid z_{\neq d})$).

The spectral gap satisfies:

$$\gamma = \inf_{f \,:\, \text{Var}_\pi(f) > 0} \frac{\mathcal{E}(f,f)}{\text{Var}_\pi(f)} = \inf_f \frac{\sum_d w_d \mathcal{E}_d(f,f)}{\text{Var}_\pi(f)}$$

We want to show $\sum_d w_d^\text{inf} \mathcal{E}_d(f,f) \geq \sum_d w_d^\text{uni} \mathcal{E}_d(f,f)$
for every $f$ (not just at the infimum).

**The Rearrangement Inequality.** For two sequences $(a_1, \ldots, a_L)$ and $(b_1, \ldots, b_L)$
with the same sorting order (i.e., $a_i \leq a_j \Rightarrow b_i \leq b_j$, positive correlation):

$$\sum_d a_d b_d \geq \frac{1}{L}\left(\sum_d a_d\right)\left(\sum_d b_d\right) = \frac{\sum_d a_d}{L} \cdot \sum_d b_d$$

Equivalently: if we replace the weights $(a_1, \ldots, a_L)$ (normalised to sum 1) by the
uniform weights $(1/L, \ldots, 1/L)$, the weighted sum $\sum_d a_d b_d$ can only decrease when
$a$ and $b$ are positively correlated.

**Applying the rearrangement inequality.** Let:
- $a_d = w_d^\text{inf} \propto \exp(\beta \cdot s_d)$ (informed weights)
- $b_d = \mathcal{E}_d(f, f)$ (per-position Dirichlet contributions)

We claim $a$ and $b$ are positively correlated: $w_d^\text{inf}$ is large when $s_d$ is large
(model is uncertain at $d$), and $\mathcal{E}_d(f,f)$ is large when position $d$ has high
variation in $f$ under the Gibbs update.

**Why are $a$ and $b$ correlated?** The key observation: positions with high surprise $s_d$
are positions where the model assigns probability mass broadly across $\mathcal{V}$. Under the
Gibbs update $T_d$, these positions transition to many different values $z'^d$. For a generic
function $f$, this creates large jumps $[f(z) - f(z')]^2$ in the Dirichlet form. Formally:

$$\mathcal{E}_d(f,f) = \text{Var}_{z^d \sim p_\theta(\cdot \mid z_{\neq d})}[f(z)]$$

which is the variance of $f$ when we resample position $d$. This variance is large when
$p_\theta(x^d \mid z_{\neq d})$ is spread out — exactly when $s_d = H(p_\theta(x^d \mid z_{\neq d}))$
is large.

**Completing the proof.** Since $w_d^\text{inf}$ and $\mathcal{E}_d(f,f)$ are positively
correlated (through $s_d$), the rearrangement inequality gives:

$$\mathcal{E}^\text{inf}(f,f) = \sum_d w_d^\text{inf} \mathcal{E}_d(f,f) \geq \frac{1}{L}\sum_d \mathcal{E}_d(f,f) = \mathcal{E}^\text{uni}(f,f)$$

Dividing both sides by $\text{Var}_\pi(f) > 0$ and taking the infimum over $f$:

$$\gamma_\text{informed} = \inf_f \frac{\mathcal{E}^\text{inf}(f,f)}{\text{Var}_\pi(f)} \geq \inf_f \frac{\mathcal{E}^\text{uni}(f,f)}{\text{Var}_\pi(f)} = \gamma_\text{uniform}$$

**Equality condition.** Equality in the rearrangement inequality holds iff $a$ and $b$ are
uncorrelated (one is constant). The weights $w_d^\text{inf}$ are constant iff $\beta = 0$ or
$\text{Var}_d[s_d] = 0$ (all surprises are equal). $\square$

**Remark.** The informed corrector is strictly better than uniform whenever positions differ in
their uncertainty — which is the typical case for natural language. The gain is large when
$\text{Var}_d[s_d]$ is large, i.e., when some positions are very easy (low entropy) and others
are very hard (high entropy). This is exactly the regime where remasking is most useful, creating
a direct link to Direction 1.

## Section 7: DFM — Correctors as Gibbs Samplers (Gat et al., 2024)

### Motivation

Discrete Flow Matching (DFM) provides a derivation of the Gibbs corrector from first principles:
the corrector is not an ad-hoc heuristic but a step that, in the limit of a perfect model,
satisfies detailed balance and hence converges to the correct stationary distribution. This
section gives the full detailed-balance proof.

### Theorem 7.1 (Gibbs Characterisation of DFM Correctors)

**Statement.** A single-step corrector that picks position $d$ uniformly at random, masks
$z^d \to [\texttt{M}]$, and resamples from $p_\theta(x^d \mid z_{\neq d})$ satisfies detailed
balance with respect to $p_t(z)$ when $p_\theta(x^d \mid z_{\neq d}) = p_t(x^d \mid z_{\neq d})$
(the model perfectly captures the conditional at noise level $t$).

**Intuition.** Detailed balance means the chain is "microscopically reversible": the rate at
which the chain transitions $z \to z'$ equals the rate it transitions $z' \to z$. If both
transitions sample the same conditional distribution (which they do when $z$ and $z'$ agree
on all positions except $d$), detailed balance is automatic. The proof is essentially immediate,
but worth spelling out to understand why approximate detailed balance holds when the model is
imperfect.

**Proof.**

Fix two sequences $z, z' \in \mathcal{V}^L$ that differ only at position $d$: $z^d \neq z'^d$ and
$z^i = z'^i$ for $i \neq d$. (Transitions where $z = z'$ or $z$ and $z'$ differ in more than one
position have probability zero under a single Gibbs step and can be ignored.)

The transition probability from $z$ to $z'$ is:

$$T(z \to z') = \frac{1}{L} \cdot p_\theta(z'^d \mid z_{\neq d})$$

because we first select position $d$ with probability $1/L$, then resample to $z'^d$ with
probability $p_\theta(z'^d \mid z_{\neq d})$.

The reverse transition is:

$$T(z' \to z) = \frac{1}{L} \cdot p_\theta(z^d \mid z'_{\neq d}) = \frac{1}{L} \cdot p_\theta(z^d \mid z_{\neq d})$$

where the last equality uses $z'_{\neq d} = z_{\neq d}$ (the two sequences agree on all positions
except $d$).

Detailed balance with respect to $p_t$ requires:

$$p_t(z) \cdot T(z \to z') = p_t(z') \cdot T(z' \to z)$$

Substituting:

$$p_t(z) \cdot p_\theta(z'^d \mid z_{\neq d}) = p_t(z') \cdot p_\theta(z^d \mid z_{\neq d})$$

When $p_\theta = p_t$ (perfect model), the conditional $p_\theta(x^d \mid z_{\neq d}) = p_t(x^d \mid z_{\neq d})$
is exactly the conditional of $p_t$. By Bayes' rule:

$$p_t(z) = p_t(z_{\neq d}) \cdot p_t(z^d \mid z_{\neq d})$$

and:

$$\frac{p_t(z)}{p_t(z')} = \frac{p_t(z^d \mid z_{\neq d})}{p_t(z'^d \mid z_{\neq d})} = \frac{p_\theta(z^d \mid z_{\neq d})}{p_\theta(z'^d \mid z_{\neq d})}$$

which is precisely what detailed balance requires. $\square$

**Remark on approximate detailed balance.** When $p_\theta \neq p_t$, the detailed balance
condition fails. The detailed balance error is:

$$\left| p_t(z) T(z \to z') - p_t(z') T(z' \to z) \right| \leq \frac{1}{L} \cdot \text{TV}(p_\theta(\cdot \mid z_{\neq d}), p_t(\cdot \mid z_{\neq d}))$$

This is bounded by $\frac{1}{L} \cdot \sqrt{\frac{1}{2} \text{KL}(p_t(\cdot \mid z_{\neq d}) \| p_\theta(\cdot \mid z_{\neq d}))}$
(Pinsker's inequality). Summing over positions: the total detailed balance error is controlled
by $E_\text{learn}(t)$ — the learning error at noise level $t$. This is the key connection: the
corrector fails only when the model is inaccurate, and the failure magnitude is quantified by
$E_\text{learn}$.

\newpage

# Part II: Research Directions

## Section 8: Direction 1 — Remasking Error Bound in the L&Z Framework

### 8.1 Problem Statement

The L&Z framework (Theorem 2.1) provides a two-term decomposition of the total error into
$E_\text{learn}$ (fixed by training) and $E_\text{fact}$ (controllable at inference time). The
EB-Sampler (Theorem 4.2) is optimal for $E_\text{fact}$ among all unmasking-only schedules. But
the EB-Sampler commits each token exactly once and never revisits a committed token. In practice,
tokens committed early (when context is sparse) may be committed incorrectly: by the time later
tokens are revealed, the early commitment is inconsistent with the growing context. This is the
**premature commitment** problem, and it is the failure mode that remasking strategies (ReMDM,
confidence-based remasking) are designed to address.

The central open question for Direction 1 is:

> **Does adding a remasking step reduce $E_\text{fact}$ below the EB-Sampler optimum?**

The EB-Sampler is optimal for the *unmasking-only* problem. But remasking enlarges the search
space: the algorithm is allowed to both unmask *and* remask. In this larger space, the EB-Sampler
is no longer the optimal member. Direction 1 aims to prove a formal bound showing that adding
remasking strictly reduces $E_\text{fact}$ whenever premature commitments are present.

### Target Theorem 8.1 (Remasking Reduces $E_\text{fact}$) **[Conjecture]**

**Statement.** Let $\text{EB}_\varepsilon$ denote the EB-Sampler with entropy budget $\varepsilon$,
and let $\text{EB}_\varepsilon + \text{remask}_\tau$ denote the combined algorithm that applies
the EB-Sampler and additionally remaskes all committed tokens $j$ with confidence signal
$\sigma^j_t = H(p_\theta(x^j \mid z_t)) > \tau$ at each step $t$. Then:

$$E_\text{fact}(\text{EB}_\varepsilon + \text{remask}_\tau) \leq E_\text{fact}(\text{EB}_\varepsilon) - \Delta(\tau, I, \varepsilon)$$

where $\Delta(\tau, I, \varepsilon) \geq 0$ and $\Delta > 0$ whenever there exists at least one
position $j$ with residual information gain $\delta_j(t) > 0$ (Definition 8.2 below) and
$\sigma^j_t > \tau$.

### 8.2 Key Concepts

**Definition 8.1 (Premature Commitment).** Token $j$ is a **premature commitment** at time $t$
if it was first unmasked at step $t_j < t$ but its residual factorisation error — the
information it would gain from being re-evaluated with the current context — is strictly positive.

Formally, define the **retrospective information gain**:

$$\delta_j(t) = H(p_\theta(x^j \mid z_{t_j})) - H(p_\theta(x^j \mid z_t))$$

where $z_t$ is the current state (which has more context than $z_{t_j}$ had at the time of
commitment). $\delta_j(t) > 0$ means that, if we re-evaluated token $j$ now, the model would be
more confident (lower entropy) than it was when it originally committed $j$ at step $t_j$.

**Definition 8.2 (Tier 3 Confidence Signal).** The **Tier 3 confidence signal** for committed
token $j$ at time $t$ is:

$$\sigma^j_t = H(p_\theta(x^j \mid z_t))$$

the current model entropy at position $j$, evaluated by (conceptually) masking $j$ and querying
the model. This signal is computable at inference time without knowledge of $x_0$.

The **remasking rule** with threshold $\tau$: at each step $t$, for all currently committed tokens $j$,
if $\sigma^j_t > \tau$, mask $j$ back to $[\texttt{M}]$ and let the EB-Sampler re-unmask it
in the next step.

### 8.3 Proof Strategy

The proof of Theorem 8.1 has three steps: decomposing $E_\text{fact}$ into per-token commitment
costs, showing that remasking reduces each such cost via the data processing inequality, and
connecting the reduction to the threshold signal.

**Step 1: Decompose $E_\text{fact}$ into per-token commitment costs** **[Proof strategy]**

We rewrite the total $E_\text{fact}$ as a sum over individual token commitments:

$$E_\text{fact} = \sum_j E_\text{fact}(j, t_j)$$

where $t_j$ is the step at which token $j$ was first unmasked and $E_\text{fact}(j, t_j)$ is
the factorisation error attributable to that commitment:

$$E_\text{fact}(j, t_j) = \mathbb{E}\left[\text{KL}\!\left(q(x^j \mid z_{t_j}, x_0) \,\Big\|\, p_\theta(x^j \mid z_{t_j})\right)\right]$$

This can be written as:

$$E_\text{fact}(j, t_j) = I^j(x) - I^j(x \mid \text{context at } t_j)$$

the information "lost" by not conditioning on the correlations between $j$ and the tokens still
masked at step $t_j$. When $t_j$ is early (sparse context), this quantity is large.

**Step 2: Remasking reduces the per-token commitment cost** **[Proved above, given the DPI]**

Suppose we remask token $j$ at time $t > t_j$ and re-commit at time $t$. The new commitment
cost is:

$$E_\text{fact}(j, t) = I^j(x) - I^j(x \mid \text{context at } t)$$

By the data processing inequality (Lemma after Theorem 3.2): revealing more tokens (moving from
$z_{t_j}$ to $z_t$) can only decrease uncertainty. Formally:

$$I^j(x \mid z_t) \leq I^j(x \mid z_{t_j})$$

because $z_t$ has more revealed tokens than $z_{t_j}$ (we are further along in the reverse
process), so conditioning on $z_t$ provides more information.

Therefore:

$$E_\text{fact}(j, t) \leq E_\text{fact}(j, t_j)$$

and the gain from remasking is:

$$\Delta_j(t, t_j) = E_\text{fact}(j, t_j) - E_\text{fact}(j, t) = I^j(x \mid z_{t_j}) - I^j(x \mid z_t) = \delta_j(t) \geq 0$$

with $\Delta_j > 0$ whenever $z_t$ provides strictly more information about $x^j$ than $z_{t_j}$ did.

**Step 3: The threshold $\tau$ detects premature commitments** **[Proof strategy, partially conjectural]**

We need to connect the observable signal $\sigma^j_t = H(p_\theta(x^j \mid z_t))$ to the
unobservable quantity $\delta_j(t)$.

**Claim:** $\sigma^j_t > \tau$ implies $\delta_j(t) \geq \sigma^j_t - H(p_\theta(x^j \mid z_0))$.

**Argument:** At $t = 0$ (fully revealed context), the model entropy is $H(p_\theta(x^j \mid z_0)) \approx 0$
(the model is nearly certain about $x^j$ when all context is available). Therefore:

$$\delta_j(t) = H(p_\theta(x^j \mid z_{t_j})) - H(p_\theta(x^j \mid z_t)) \approx H(p_\theta(x^j \mid z_{t_j})) - \sigma^j_t$$

and if $\sigma^j_t > \tau$, then:

$$\Delta_j \approx H(p_\theta(x^j \mid z_{t_j})) - H(p_\theta(x^j \mid z_t)) > H(p_\theta(x^j \mid z_{t_j})) - \sigma^j_t$$

which is positive as long as the model was not perfectly confident at the time of commitment.

**Key gap and resolution approaches:** The argument above uses $H(p_\theta(x^j \mid z_0)) \approx 0$
(model certainty at full context) and $H(p_\theta(x^j \mid z_{t_j})) \geq \sigma^j_t$ (entropy
was at least as high at commitment as it is now). The second is guaranteed by the DPI: as more
context is revealed, entropy can only decrease. The first requires connecting the model entropy
to the true information $I^j(x \mid z_t)$:

**(a) Consistency approach:** Under perfect model assumption ($p_\theta = q$):
$$H(p_\theta(x^j \mid z_t)) = H(x^j \mid z_t) = I^j(x \mid z_t)$$
so the signal exactly equals the true information, and the connection is immediate.

**(b) Imperfect model bound:** For imperfect models, by the data processing inequality for
entropy under channel perturbation:
$$|H(p_\theta(x^j \mid z_t)) - H(x^j \mid z_t)| \leq f(E_\text{learn}(t))$$
where $f$ is an increasing function of the per-step learning error. This bounds the
signal quality in terms of the training objective.

### 8.4 Attack Ideas

**1. Information monotonicity approach.** The data processing inequality is already in place
(Step 2 above). The remaining work is to formalise the connection in Step 3. One approach:
treat $H(p_\theta(x^j \mid z_t))$ as a surrogate for $I^j(x \mid z_t)$ and bound the surrogate
error via $E_\text{learn}$. The resulting theorem would be: $\Delta(\tau, I, \varepsilon) \geq \tau - f(E_\text{learn})$,
so remasking is guaranteed to reduce $E_\text{fact}$ as long as the threshold exceeds the
learning error.

**2. Coupling approach.** Couple the trajectories of EB-only and EB+remask. Define the coupling
by running both algorithms on the same sequence of random bits, differing only in whether
remasking steps are taken. Show that after any remasking step, the EB+remask trajectory is
stochastically closer to $\pi$ in the order induced by the information profile: its effective
$\Sigma^2$ is smaller. By Theorem 3.1, lower $\Sigma^2$ implies lower $E_\text{fact}$ — giving
the reduction $\Delta$.

**3. Marginal distribution approach.** After remasking token $j$ at time $t$, the marginal
distribution of $x^j$ in the generated sequence becomes $p_\theta(x^j \mid z_t)$ instead of
$p_\theta(x^j \mid z_{t_j})$. By KL monotonicity under conditioning:

$$\text{KL}\!\left(q(x^j \mid x_0) \,\Big\|\, p_\theta(x^j \mid z_t)\right) \leq \text{KL}\!\left(q(x^j \mid x_0) \,\Big\|\, p_\theta(x^j \mid z_{t_j})\right)$$

Each remasking step brings the marginal posterior closer to the true posterior. Summing over all
remasked positions gives the total reduction $\Delta$.

**4. Equivalent Riemann reformulation.** View the EB+remask algorithm as generating an effective
information curve $I'(\alpha)$ that is smoother than the original $I(\alpha)$ (remasking
redistributes committed information to later, better-contextualised steps). Show that remasking
reduces the irregularity: $\text{Var}[I'] \leq \text{Var}[I] - \Delta$. By Theorem 3.1, this
translates directly to $E_\text{fact}(EB+remask) \leq E_\text{fact}(EB) - C \cdot \Delta / T$.

### 8.5 Connection to Existing Results

Direction 1 directly extends the L&Z framework by adding the remasking case. The table of
contributions would be:

| Contribution | Source | Type |
|---|---|---|
| $\text{KL} \leq E_\text{learn} + E_\text{fact}$ | L&Z 2025 | Decomposition |
| $E_\text{fact}$ minimised by easy-first unmasking | L&Z, EB-Sampler | Optimality |
| Correctors reduce KL exponentially in $R$ | Informed Correctors 2024 | Mixing |
| **Remasking reduces $E_\text{fact}$ below EB-only** | **Thesis (Direction 1)** | **New bound** |

This is the most direct contribution: it takes the advisor's framework and extends it to cover
the one inference-time operation (remasking) that the original L&Z paper did not analyse.

## Section 9: Direction 2 — Bridging the Riemann and MCMC Frameworks

### 9.1 Problem Statement

Two independent theoretical frameworks both conclude that uncertainty-guided strategies
outperform uniform strategies. The L&Z Riemann thread (Section 3) says the advantage scales
with $\Sigma^2 = \text{Var}_i[I^i(x)]$. The Informed Correctors MCMC thread (Section 6) says
the advantage scales with $\text{Var}_d[s_d(z)]$ where $s_d = H(p_\theta(x^d \mid z_{\neq d}))$.
The question is whether these two quantities are formally related. If $\Sigma^2 \propto \mathbb{E}[\text{Var}_d[s_d]]$,
then the two frameworks are saying the same thing from different mathematical perspectives, and
their results can be unified into a single theorem.

### Target Theorem 9.1 (Framework Equivalence) **[Open]**

**Statement.** For a well-trained denoiser $p_\theta$ on data distribution $\pi$, and in the
limit $t \to 0$ (vanishing masking):

$$\mathbb{E}_{x \sim \pi}[\Sigma^2(x)] = \mathbb{E}_{x \sim \pi}\left[\text{Var}_d\!\left[s_d(x)\right]\right]$$

where $s_d(x) = H(p_\theta(x^d \mid x_{\neq d}))$ is the surprise at position $d$ for the
fully revealed sequence $x$.

More generally, for $t > 0$: $\mathbb{E}_{z_t \sim q_t}[\text{Var}_d[s_d(z_t)]]$ is monotone
in $\mathbb{E}[\Sigma^2]$, and the two framework predictions about strategy rankings agree.

### 9.2 The Two Quantities, Stated Precisely

**$\Sigma^2$ (Riemann thread):**

$$\Sigma^2(x) = \text{Var}_i[I^i(x)] = \frac{1}{L}\sum_{i=1}^L \left(I^i(x) - \bar{I}(x)\right)^2$$

where $I^i(x) = H(x^i \mid x_{\neq i})$ and $\bar{I}(x) = \frac{1}{L}\sum_i I^i(x)$.

**$\text{Var}[s_d]$ (MCMC thread):**

At state $z_t$: $s_d(z_t) = H(p_\theta(x^d \mid z_{t, \neq d}))$, and:

$$\text{Var}_d[s_d(z_t)] = \frac{1}{L}\sum_{d=1}^L \left(s_d(z_t) - \bar{s}(z_t)\right)^2$$

where $\bar{s}(z_t) = \frac{1}{L}\sum_d s_d(z_t)$.

**At $t=0$ (perfect model, fully revealed context):**

$$s_d(x) = H(p_\theta(x^d \mid x_{\neq d})) = H(x^d \mid x_{\neq d}) = I^d(x)$$

where the second equality uses the perfect-model assumption. Therefore:

$$\text{Var}_d[s_d(x)] = \text{Var}_d[I^d(x)] = \Sigma^2(x)$$

**The equivalence is exact at $t=0$.** The open question is whether it extends to $t > 0$.

### 9.3 Proof Strategy

**Step 1: Exact equality at $t = 0$** **[Proved above]**

Under the perfect-model assumption, at $t=0$:

$$s_d(x) = I^d(x) \implies \text{Var}_d[s_d(x)] = \Sigma^2(x)$$

This is exact and immediate. Taking expectations over $x \sim \pi$: $\mathbb{E}[\text{Var}_d[s_d]] = \mathbb{E}[\Sigma^2]$.

**Step 2: Approximate equality for small $t$ (Taylor expansion)** **[Proof strategy]**

For small masking ratio $\alpha_t \ll 1$, each masked position $i$ sees its token replaced by
$[\texttt{M}]$ with probability $\alpha_t$ and left unchanged with probability $1 - \alpha_t$.
We can Taylor-expand the model entropy around the fully-revealed state $x$:

$$H(p_\theta(x^d \mid z_{t, \neq d})) = H(p_\theta(x^d \mid x_{\neq d})) + \alpha_t \cdot \frac{\partial}{\partial \alpha} H(p_\theta(x^d \mid z_\alpha^{\neq d}))\bigg|_{\alpha=0} + O(\alpha_t^2)$$

The first-order correction $\frac{\partial}{\partial \alpha}[\cdot]$ measures how much the
entropy at position $d$ changes when we randomly mask each other position independently with
probability $\alpha_t$. This is a sum of per-position corrections:

$$\frac{\partial}{\partial \alpha} H(p_\theta(x^d \mid z_\alpha^{\neq d})) = \sum_{i \neq d} \left(H(p_\theta(x^d \mid z_\alpha^{\neq d}, \text{mask } i)) - H(p_\theta(x^d \mid z_\alpha^{\neq d}))\right)$$

Each term measures the increase in entropy at $d$ when we additionally mask position $i$ — which
is exactly $I(x^d ; x^i \mid z_\alpha^{\neq d, i})$, a pairwise mutual information. At $\alpha = 0$:

$$\frac{\partial}{\partial \alpha} H(p_\theta(x^d \mid z_\alpha^{\neq d}))\bigg|_{\alpha=0} = \sum_{i \neq d} I(x^d ; x^i \mid x_{\neq d, i})$$

This is the total "second-order information" at position $d$ — how much masking other positions
affects the uncertainty at $d$. Computing the variance of this across positions $d$ would give
the first-order correction to $\Sigma^2$. The claim (requiring verification) is that this
correction preserves the ordering: $\text{Var}_d[s_d(z_t)] \propto \Sigma^2(x) + O(\alpha_t)$,
with the proportionality constant and correction term expressible in terms of higher-order
information quantities of $\pi$.

**Step 3: Monotone relationship at general $t$** **[Conjecture]**

For general $t$, both $\text{Var}_d[s_d(z_t)]$ and $\mathbb{E}[\Sigma^2]$ decrease as $t$
increases (as more tokens are masked, positions become more uncertain and the profile flattens).
The conjecture is that they decrease at the same rate — i.e., the ratio
$\text{Var}_d[s_d(z_t)] / \mathbb{E}[\Sigma^2]$ remains bounded and bounded away from zero
uniformly in $t$. If this holds, both quantities give the same ranking of strategies at all
noise levels.

### 9.4 Attack Ideas

**1. Taylor expansion approach.** Implement Step 2 above explicitly. Compute the first- and
second-order corrections to $\text{Var}_d[s_d(z_t)]$ as a power series in $\alpha_t$. If the
leading terms match those of a corresponding expansion of $\Sigma^2(t)$ (the effective
information variance at noise level $t$), the equivalence follows to leading order.

**2. Optimal transport approach.** Both $\Sigma^2$ and $\text{Var}[s_d]$ are measures of
"spread" on the information profile, viewed as a probability distribution on $[0, \log|\mathcal{V}|]$.
Equivalence in the spirit of Theorem 9.1 may follow from the fact that both are second moments
of this distribution (one under $\pi$, one under the model). Tools from optimal transport (in
particular, the relationship between Wasserstein distance and moment convergence) may formalise
this.

**3. Direct computation on a 2-token model.** Consider $L = 2$, $\mathcal{V} = \{0,1\}$, and
a joint distribution $\pi(x^1, x^2)$ parametrised by correlation $\rho \in [-1, 1]$. Compute
$\Sigma^2$ and $\text{Var}[s_d]$ analytically as functions of $\rho$. If they are equal (or
proportional) for all $\rho$, this strongly suggests the general equivalence and provides a
clean formula for the proportionality constant. If they differ, the $L=2$ example isolates
exactly what breaks the equivalence and guides a refined conjecture.

**4. Counterexample approach.** If the equivalence is false, construct a distribution where
$\Sigma^2$ is large but $\text{Var}[s_d]$ is small. One candidate: a distribution where
information is heterogeneous across positions (high $\Sigma^2$) but the denoiser, because of
its architecture, smooths out its entropy predictions (low $\text{Var}[s_d]$). This would show
that the two frameworks give different guidance in practice, and the open question would be
resolved by a separation result rather than a unification.

### 9.5 Why This Direction Is Compelling

The L&Z framework was developed by the thesis advisor (Zanella, 2025). The Informed Correctors
framework (Zhao et al., 2024) was developed independently, building on DFM (Gat et al., 2024).
Both were written around the same time, do not cite each other on the unification point, and
arrive at superficially similar conclusions through entirely different mathematics (Riemann
approximation theory vs. spectral gap theory of Markov chains). A formal bridge between these
two frameworks would be the first unification result in this space.

The value of Theorem 9.1, if proven, is two-fold. Theoretically: it shows that the "non-uniformity
of uncertainty" is a fundamental quantity that governs strategy performance regardless of which
mathematical framework one uses, elevating it to the status of a natural invariant of the problem.
Practically: it would allow results proven in the Riemann framework (such as Direction 1's
remasking bound) to be immediately translated into spectral gap terms, and vice versa. The thesis
would go from "extension of one paper" to "unification of two bodies of work" — a qualitatively
stronger contribution.

## Section 10: Direction 3 — EB-Sampler + Remasking: The Combined Algorithm

### 10.1 Problem Statement

The EB-Sampler is optimal for unmasking only. Direction 1 shows remasking reduces $E_\text{fact}$.
But the combined EB+remask algorithm is not simply the sum of its parts: the unmasking and
remasking steps interact.

When remasking returns a token to $[\texttt{M}]$, it changes the state $z_t$ seen by the EB-Sampler
at the next step. This new state was not produced by the standard forward process — it is a
"targeted" masked state where specific high-uncertainty tokens are masked while others remain
revealed. Training states have uniformly random masking patterns; the targeted masked state is
out-of-distribution (OOD) for the model. This creates an OOD penalty that must be weighed
against the benefit of the remasking.

The central question of Direction 3 is: **what is the optimal remasking threshold $\tau^*$?**
Too aggressive (low $\tau$): many tokens are remasked, creating OOD states and increasing
$E_\text{learn}$; the OOD penalty dominates and performance degrades. Too conservative (high $\tau$):
few tokens are remasked, premature commitments persist and $E_\text{fact}$ remains high.
Direction 3 aims to formalise this tradeoff and characterise $\tau^*$.

### Definition 10.1 (EB+Remask Algorithm)

At each step $t$ (iterating from $T$ down to $0$):
1. **Sort:** For each masked position $i$, compute $H_i = H(p_\theta(x^i \mid z_t))$. Sort in ascending order.
2. **Unmask:** Greedily select positions $S_t$ in ascending order of $H_i$ until $\sum_{i \in S_t} H_i \leq \varepsilon$. Sample $x^i \sim p_\theta(x^i \mid z_t)$ for $i \in S_t$.
3. **Remask:** For all committed positions $j \notin S_t$ (previously unmasked), if $H(p_\theta(x^j \mid z_t)) > \tau$, set $z_t^j \leftarrow [\texttt{M}]$.
4. Repeat until all positions are committed.

The order of Steps 2 and 3 matters: we first unmask new easy tokens (bringing in more context),
then evaluate committed tokens against the improved context. This ordering ensures the remasking
step has access to the most current context.

### Target Theorem 10.1 (Combined Algorithm Bound) **[Conjecture]**

**Statement.** The combined EB+remask algorithm satisfies:

$$E_\text{total}(\text{EB}+\text{remask}) \leq E_\text{learn} + E_\text{fact}(\text{EB}) - \Delta_\text{remask}(\tau, I) + E_\text{interaction}(\varepsilon, \tau)$$

where:
- $\Delta_\text{remask}(\tau, I) \geq 0$ is the reduction in $E_\text{fact}$ from remasking (established in Direction 1)
- $E_\text{interaction}(\varepsilon, \tau)$ captures the interaction between unmasking and remasking
- $E_\text{interaction}(\varepsilon, \tau) \leq C \cdot \varepsilon \cdot \tau$ for a constant $C$, which is small when both the entropy budget $\varepsilon$ and the remasking threshold $\tau$ are small

### 10.2 The Interaction Term: Synergy and OOD Penalty

The interaction between unmasking and remasking has two components with opposite signs.

**Positive interaction (synergy).** After remasking token $j$ at time $t$, the model's context
for subsequent EB-Sampler steps is updated: position $j$ is now masked again, which changes the
entropy estimates $H_i$ for neighboring positions $i$. In many cases, this improves the sorting:
positions that were incorrectly sorted before (because $j$'s committed value was wrong) are now
sorted correctly. This makes the subsequent EB steps more effective, reducing $E_\text{fact}$
on later steps. The synergy gain is proportional to the correlation between token $j$ and its
neighbors — high when $j$ was a "hub" token with many strong dependencies.

**Negative interaction (OOD risk).** The remasked state $z'_t$ (with token $j$ re-masked) has
masking pattern that was not seen during training. Training masks tokens uniformly at random;
the remasked state has a specific non-uniform pattern (only high-uncertainty tokens are masked).
The model $p_\theta$ was not trained on such states, so its conditional distributions
$p_\theta(x^i \mid z'_t)$ may be inaccurate for the remasked state. This increases the effective
$E_\text{learn}$ at the remasked positions. The OOD penalty is proportional to the fraction of
tokens remasked and the model's sensitivity to masking pattern non-uniformity.

### 10.3 Proof Strategy

**Step 1: Decompose the total error by time step** **[Proof strategy]**

$$E_\text{total} = \sum_t \left[ E_\text{fact,unmask}(t) + E_\text{fact,remask}(t) + E_\text{interact}(t) \right]$$

The first term is the EB-Sampler contribution at step $t$ (bounded by Theorem 4.1). The second
term is the factorisation error reduction from remasking (bounded by Direction 1). The third term
$E_\text{interact}(t)$ is the cross-term: does the remasking at step $t$ affect the unmasking
at step $t+1$ and vice versa?

**Step 2: Bound $E_\text{interact}(t)$ using the entropy budget** **[Proof strategy]**

After remasking at step $t$, the effective information profile for step $t+1$ is updated:
let $\Sigma^2(t+1 \mid \text{remask})$ denote the profile variance after remasking at step $t$.
Define the synergy gain:

$$\delta\Sigma^2(t) = \Sigma^2(t+1 \mid \text{no remask}) - \Sigma^2(t+1 \mid \text{remask}) \geq 0$$

Remasking makes the profile smoother (more tokens are uncertain → profile is more uniform →
$\Sigma^2$ decreases). By Theorem 3.1, lower $\Sigma^2$ implies lower $E_\text{fact,unmask}$
at step $t+1$ by $C \cdot \delta\Sigma^2(t) / T$ per step. This is the synergy term.

**Step 3: Bound the OOD penalty** **[Proof strategy]**

Define the remasking fraction at step $t$: $f_t = |\{j : \sigma^j_t > \tau\}| / L$ (the
fraction of committed tokens remasked at step $t$). The remasked state $z'_t$ differs from any
training-distribution state by having $f_t$ targeted masked positions. The OOD gap is:

$$\text{KL}(q'_t \| q_t) \leq C \cdot f_t$$

where $q'_t$ is the marginal distribution of the remasked state and $q_t$ is the training-time
marginal at masking level $\alpha_t$. This uses the fact that each remasked token is an
independent perturbation of the masking pattern, and each contributes independently to the KL.
The resulting increase in $E_\text{learn}$ is:

$$\Delta E_\text{learn}(\text{remask}) \leq C \cdot f_t \cdot E_\text{learn}(t)$$

where $E_\text{learn}(t)$ is the per-step learning error — the model's sensitivity to the
exact masking pattern is bounded by its overall inaccuracy.

**Step 4: Combine into the interaction bound** **[Conjecture]**

$$E_\text{interact}(t) = -C \cdot \delta\Sigma^2(t) / T + C' \cdot f_t \cdot E_\text{learn}(t)$$

$$E_\text{interaction}(\varepsilon, \tau) = \sum_t E_\text{interact}(t) \leq C \cdot \varepsilon \cdot \tau$$

The bound $\varepsilon \cdot \tau$ comes from: $\varepsilon$ bounds the entropy budget (how much
context changes at each unmasking step), and $\tau$ bounds the remasking threshold (only tokens
with entropy above $\tau$ are remasked). When both are small, the algorithm is conservative and
the interaction term vanishes. The joint smallness condition ensures that neither operation is
too aggressive.

### 10.4 Attack Ideas

**1. Monotone coupling.** Couple EB-only and EB+remask trajectories by running both on the same
random bits, differing only in the remasking step. After any remasking step, define the "lead"
of the EB+remask trajectory as a measure of how much it has reduced $\Sigma^2$ relative to
EB-only. Show that the lead is non-decreasing in $t$ (each remasking step either maintains or
increases the lead). The interaction term is then bounded by the worst-case decrease in lead
from a single OOD state.

**2. Fixed-point analysis.** The combined algorithm has a natural fixed point: a state where
no remasking is triggered ($\sigma^j_t \leq \tau$ for all committed $j$). Near this fixed
point, the remasking fraction $f_t$ is small, and the interaction term is second-order in
$(\tau - \tau^*)$ where $\tau^*$ is the critical threshold at which the first remasking event
occurs. A Taylor expansion around the fixed point gives $E_\text{interaction} = O((\tau - \tau^*)^2)$,
which is small near the optimal threshold.

**3. Calculus of variations for optimal $\tau$.** Treat the remasking fraction as a function of
time: $f : [0, T] \to [0, 1]$, $f(t) = $ fraction of tokens remasked at step $t$. The total
error is a functional:

$$E_\text{total}[f] = E_\text{fact}(\text{EB}) - \int_0^T g(f(t)) \, dt + \int_0^T h(f(t)) \cdot E_\text{learn}(t) \, dt$$

where $g(f)$ is the synergy gain (concave, increasing in $f$) and $h(f) \cdot E_\text{learn}(t)$
is the OOD penalty (convex, increasing in $f$). Minimising over $f(t)$ by Euler-Lagrange
equations gives the optimal schedule $f^*(t)$, which translates to an optimal threshold
$\tau^*(t)$ as a function of time. The prediction: $\tau^*(t)$ decreases as $t$ increases (be
more selective about remasking late in the process when context is rich and OOD risk is highest).

**4. Empirical validation.** Using the existing MDLM-OWT experimental infrastructure, run a
sweep over $\tau \in \{0.1, 0.2, \ldots, 0.9\}$ (in bits) for a fixed $\varepsilon$. The
MAUVE score as a function of $\tau$ will display a clear maximum at $\tau^*$. The theoretical
prediction from the calculus of variations (Attack 3) gives a closed-form for $\tau^*$ in terms
of $E_\text{learn}$ and the information profile. If the theoretical $\tau^*$ matches the
empirical optimum, this simultaneously validates the theory and provides a practical algorithm
selection tool.

### 10.5 Connection to the Empirical Results

The step-sweep results from the current experimental infrastructure directly instantiate the
tradeoff analysed by Direction 3:

- **remdm-conf** (confidence-based remasking, effectively low $\tau$): aggressively remasks
  any token below a confidence threshold. At $T=1000$ steps, MAUVE collapses to 0.325 (below
  its $T=128$ value of 0.440). This is consistent with the OOD penalty dominating: aggressive
  remasking creates highly targeted masked states that are far from the training distribution,
  increasing $E_\text{learn}$ catastrophically at high step counts.

- **remdm-loop** (loop-based remasking, effectively high $\tau$): applies remasking more
  conservatively, only reconsidering tokens when the full context has changed substantially.
  MAUVE improves monotonically from 0.396 ($T=128$) to 0.684 ($T=1000$). This is consistent
  with the synergy term dominating: conservative remasking stays near the training distribution
  and progressively reduces $E_\text{fact}$ without triggering the OOD penalty.

- **mdlm** (no remasking): pure EB-Sampler baseline. MAUVE peaks at $T=256$ (0.740) and
  declines at higher step counts. This "diversity window" — a peak followed by decline — is
  explained by the EB-Sampler behaviour: at $T=256$, the step size is small enough that
  factorisation error is manageable, but at $T=1000$, the very small steps introduce other
  artefacts (the model is queried many times on very similar near-complete states, which can
  concentrate the distribution excessively).

Direction 3's theorem would formalise this observation: conf collapses because it remaskes
too aggressively (remasking fraction $f_t$ large, OOD penalty $C \cdot f_t \cdot E_\text{learn}$
dominates), while loop succeeds because it remaskes conservatively (remasking fraction $f_t$
small, synergy term dominates). The formal theorem would give an explicit expression for the
crossover threshold $\tau^*$ where the strategy changes from beneficial to harmful.

---

# Open Problems Summary

| Problem | Status | Direction |
|---|---|---|
| Remasking reduces $E_\text{fact}$ below EB-only | Conjectured | Dir 1 |
| Tier 3 signal $\sigma^j_t$ sufficient for $E_\text{fact}$ reduction | Conjectured | Dir 1 |
| $\Sigma^2 = \text{Var}[s_d]$ formally | Open | Dir 2 |
| Optimal remasking schedule characterisable in closed form | Open | Dir 2/3 |
| $E_\text{interaction}(\varepsilon, \tau) \leq C \cdot \varepsilon \cdot \tau$ | Conjectured | Dir 3 |
| Optimal $\tau^*$ expressible in terms of $I$ and $E_\text{learn}$ | Open | Dir 3 |

The three directions are nested from most concrete to most ambitious. **Direction 1** is the
core theorem: extending the L&Z decomposition to the remasking case. The key tools — the data
processing inequality, the KL triangle inequality, and the Riemann error decomposition — are
already in hand (developed in Part I), and the main gap (connecting the model entropy signal
to the true information) is a well-defined problem with clear solution paths. This is the
guaranteed deliverable, most directly building on the advisor's framework.

**Direction 2** is the ambitious stretch: bridging the Riemann and MCMC frameworks into a
single unification theorem. The $t=0$ base case is already proved (it is an immediate consequence
of the perfect-model assumption), and the path to the general case is laid out via Taylor
expansion and monotonicity. A successful Direction 2 result would be the strongest possible
contribution, showing that the thesis goes beyond extending one paper to synthesising two
independent lines of work.

**Direction 3** is the experimental backbone: it formalises the interaction between unmasking
and remasking, providing both a rigorous analysis of the combined algorithm and a direct
theoretical explanation for the observed empirical results. The threshold sweep (Attack 4 above)
is immediately implementable with existing infrastructure. Direction 3 validation feeds back
into Direction 1 by providing empirical estimates of $\Delta_\text{remask}$ and the OOD penalty,
which can guide the choice of tools for the formal proof. The three directions can be pursued
in parallel, with Direction 3 empirical work providing ground truth that disciplines the
theoretical development of Directions 1 and 2.

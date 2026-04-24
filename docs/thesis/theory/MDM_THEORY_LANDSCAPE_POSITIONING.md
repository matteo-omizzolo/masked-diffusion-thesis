> **STATUS:** LITERATURE POSITIONING (companion to NEXT_RESEARCH_DIRECTION_AUDIT)
> **LAST VERIFIED:** 2026-04-24
> **SCOPE:** Positions the thesis's theoretical contribution (Theorem A +
> Refinements A′/A″ + Negative-Result Corollary + Appendix-F extension)
> relative to the 2025–2026 MDM theory landscape. Used to answer
> "What counts as a genuine MDM contribution in 2025–2026?" and
> "Where is the remaining mathematically principled gap?"

---

# MDM Theory Landscape Positioning

## 1. The thesis object, restated

Three levels are held fixed:

1. **Predictor schedule** — the outer unmasking trajectory z_0, …, z_T is fixed
   (ProSeCo for OWT; LLaDA-SFT predictor for the cross-backbone probe).
2. **Informed corrector kernel** — ProSeCo-style annealed refinement kernel,
   fixed.
3. **Corrector NFE budget B** — fixed, small: B ∈ {2, 3, 4, 8} on OWT;
   B ∈ {2, 4} on LLaDA-SFT.

The contribution asks, for that fixed triple: **how should a fixed budget B of
corrective refinement steps be allocated across the T unmasking steps of the
trajectory?** This is a *scheduling* question, strictly one level above
kernel design and one level below predictor-side scheduling.

The literature maps naturally into concentric rings around this object.

---

## 2. The 2025–2026 MDM theory landscape in rings

### Ring 1 — Predictor-side / unmasking scheduling

Papers that choose *when* or *which tokens* to unmask. These are all one level
**above** the thesis object — they operate on the predictor trajectory, not
on the corrective refinement budget.

- **Learning Unmasking Policies (arXiv:2512.09106, 2025).** Formalizes
  sampling as an MDP where the action is the unmasking step; proposes a
  learned policy. State = partial unmasking; action = which positions to
  unmask. **Different object** — the action space is predictor, not
  corrector.
- **Soft-Masked Diffusion (arXiv:2510.17206).** Training-time relaxation
  of the masking schedule. **Different phase** — training, not inference.
- **Accelerated Sampling via Entropy-Bounded Unmasking (arXiv:2505.24857).**
  Entropy-thresholded predictor scheduling for reasoning. **Different
  object.**
- **Adaptive Parallel Decoding (APD, arXiv:2506.00413).** Accelerates dLLMs
  by mixing with an autoregressive model. **Different class** — no
  corrector at all.
- **EAGS (arXiv:2411.06438).** Entropy-based noise scheduling + inference-
  time Gibbs. Closest of the predictor-side ring because it *also* applies
  Gibbs updates — but its Gibbs corrector is kernel-level, and the paper
  gives no proxy-regret bound for corrector scheduling.

### Ring 2 — Trajectory-measure / SMC / Particle Gibbs

Papers that sample over full trajectories using SMC-style or particle-based
algorithms, often for reward-guided generation.

- **PG-DLM — Inference-Time Scaling of Diffusion Language Models with
  Particle Gibbs (arXiv:2507.08390, 2025).** Constructs a Markov chain over
  *full denoising trajectories* using conditional SMC. The object is full
  trajectory search for reward-guided generation. **Different class** —
  not a fixed-budget corrector scheduling problem; does not give a regret
  bound for a fixed (predictor, kernel, B) triple.
- **E-SMC / Optimizing Decoding Paths for MDMs (arXiv:2512.21336, 2025).**
  Entropy-adaptive SMC for MDM decoding. **Different class** — predictor-
  side SMC, not corrector scheduling.

### Ring 3 — Corrector kernel design

Papers that design the corrector *kernel* itself — i.e., how a single
corrective step updates a single token or a small set of tokens.

- **Zhao et al. 2024.** Locally-balanced corrector proposals (Barker) and
  MPF corrector; hollow-transformer architecture. Kernel design; no
  scheduling bound.
- **PRISM (arXiv:2510.01384).** Provably learnable quality scores for
  corrector-selection; complementary signal input to scheduling.
- **ProSeCo (arXiv:2602.11590).** Annealed refinement corrector with
  schedule knobs. The thesis holds its kernel fixed and operates one
  level up.
- **DFM (Gat et al. 2024).** Discrete Flow Matching corrector as Gibbs
  sampler; mixing rate analysis via spectral gap.

### Ring 4 — Remasking / re-noising

- **ReMDM (Wang et al. 2024).** Remasking-based corrector variants. Kernel
  variant + schedule interplay, but the remasking decision is part of the
  kernel definition. **Different axis** — re-noising vs refinement.

### Ring 5 — Classical theory anchors

- **L & Z Error Bounds for MDMs (the paper this thesis extends).**
  Provides the E_fact / E_learn decomposition and a factorization-error
  bound for predictor-only MDM samplers. The thesis extends the
  E_fact / E_learn lens to include corrector NFE budget.
- **Ascolani et al. 2024.** Entropy contraction for random-scan Gibbs.
  Supplies the kernel-level contraction used to anchor Prop C.
- **Zanella 2020.** Locally-balanced proposals. Underlies Zhao's Barker
  proposal design.

---

## 3. The empty ring — where the thesis contribution lands

The rings above cover predictor-side scheduling (Ring 1), full-trajectory
search (Ring 2), kernel design (Ring 3), and remasking (Ring 4). **None of
them provides a regret-type theorem for fixed-budget corrector scheduling
under a fixed predictor and fixed kernel.**

The thesis fills this empty ring with four coordinated objects:

### 3.1 Theorem A (main contribution)

Under (1) binary placement, (2) approximate additivity with slack η_B,
(3) proxy L∞ calibration slack ε:

**G(S_B*) − G(Ŝ_B) ≤ 2Bε + 2η_B**

where S_B* is the oracle top-B site set, Ŝ_B is the top-B set by a separable
score ψ, and G(S) is the schedule gain from placing corrective refinements
at sites S. Proof combines Lemma A1 (oracle top-B optimality under exact
additivity, a generalization of standard top-k lemmas) with Lemma A2 (proxy
approximation propagates as 2Bε).

**Novelty relative to landscape:** Ring 1 papers bound predictor-side MDM
error but do not treat corrector budget as a scheduling variable. Ring 2
papers bound full-trajectory SMC error but do not fix the predictor or
kernel. Ring 3 papers bound kernel mixing but do not schedule. Theorem A
is the first proxy-regret bound in this specific slot.

### 3.2 Refinements A′ + A″

- **A′ (variance-form η_B):** η_B ≤ σ_ξ · √B / √2, where σ_ξ is the per-
  seed MC residual standard deviation. This replaces the loose worst-case
  additivity slack with a variance-scaling bound — tighter by a factor of
  √B at small B.
- **A″ (rank-based ε_R):** ε_R = (1 − |ρ|) · σ_Δ, where ρ is the Spearman
  rank correlation between the proxy and true gains, and σ_Δ is the std
  of true gains. This replaces the worst-case L∞ calibration slack with a
  rank-sensitivity bound — tighter when the proxy preserves rank but not
  scale.

Both refinements have written derivations in `research/candidate_theorems.md`;
neither has been plugged in with on-disk OWT numbers. Phase 4 computes them.

### 3.3 Negative-Result Corollary (ranker-class scope)

Any separable per-step ranker is upper-bounded by the mean_delta_oracle
envelope. On ProSeCo-OWT at K = 30, T = 64, mean_delta_oracle enters the
NULL band by B = 8. Therefore no separable per-step ranker beats uniform
at B ≥ 8 on that triple.

**Scope limits:** The corollary characterises the ranker class only. Phase 3a
(CD-G + BS-AG) is a search-class procedure, not a separable per-step ranker,
and exceeds this envelope (49–84 % closure of MC-oracle headroom).

### 3.4 Appendix F — Theorem A-ad (conditional extension)

Under the F1 FH-CMDP framing with bucketed state z_t = (s_t, b_t, phase(t))
and action a_t ∈ {0, 1} (correct or skip), a threshold policy π*_λ =
𝟙[Δ̂(z) > λ] ∧ 𝟙[b > 0] satisfies:

**𝔼[G(S_π*_B)] − 𝔼[G(S_π̂_λ)] ≤ 2Bε̃ + 2η̃_B + 𝒪(√(B/N_cal))**

with ε̃, η̃_B the state-conditional versions of the open-loop constants, and
N_cal the calibration sample size. Theorem A-ad strictly generalises
Theorem A (recover Theorem A by taking z_t trivial).

**Status:** Conditional — constants ε̃, η̃_B have not been estimated. Protocol
C (1-day laptop analysis of LLaDA-SFT Phase 2b JSONs) is the bounded probe
that estimates them. Under no circumstance is Theorem A-ad promoted to a
main-body claim without that probe.

---

## 4. Novelty assessment of the adaptive-controller extension

The adaptive-controller direction has four candidate framings (see
`ADAPTIVE_BUDGETED_CONTROLLERS.md` §6):

| Framework | Role | Novel relative to landscape? |
|---|---|---|
| F1 — Finite-Horizon CMDP | Normative pick for Theorem A-ad | Partially novel — the FH-CMDP structure is standard, but its instantiation for *corrector scheduling* with bucketed state is not in Ring 1–5. |
| F2 — Control-as-inference / Feynman-Kac | Glue language | Not novel — reduces to F1. |
| F3 — Particle Gibbs / conditional SMC | Algorithmic realization | **Not novel.** PG-DLM (arXiv:2507.08390) already realizes particle Gibpbs on MDM trajectories empirically. Thesis would cite PG-DLM and not implement it. |
| F4 — Adaptive submodularity | Foil | **Falsified** by Prop C (γ > 0 ⇒ non-submodular). Use only as a literature foil in the Appendix-F discussion, not as an active framework. |

**Net novelty of Appendix F:** The thesis's contribution in Ring 1–5 language
is **the F1-framed conditional theorem + Protocol C empirical probe**, not
F3's algorithm (already done by PG-DLM) nor F4's framework (falsified).
This is a small, bounded novelty appropriate for a future-work appendix, not
for main-body promotion.

---

## 5. Better empirical confirmation path

The audit rejects a K = 30 LLaDA-SFT continuation. The strongest remaining
empirical-confirmation lever is **sharpening OWT artefacts**, not adding
new-backbone data.

### 5.1 What OWT artefacts are not yet fully exploited

- **σ_ξ per-seed residual std** — needed for A′. Not yet computed on disk
  despite the MC-oracle rollouts containing the data.
- **Spearman ρ(A, G) per-seed** — needed for A″. Straightforward from
  paired mc_oracle + policy_comparison data.
- **γ estimate** — needed for Prop C. Requires pairwise interaction matrix
  over co-selected sites; Phase 2b schedules cover enough co-selection
  patterns for a usable upper bound.
- **Low-gain-share curve** — needed for Prop B. Already 62–69 % within-seed
  share is available; formalizing the top-k cutoff into a proposition needs
  a slightly more structured table.

Computing all four unblocks **non-vacuous** Theorem A constants at the OWT
triple, which is exactly what the Zanella meeting will want to see.

### 5.2 What OWT artefacts alone cannot establish

- **Transfer.** Requires the LLaDA-SFT bounded probe as appendix — done.
- **Broader-task transfer.** Out of thesis scope by design.

### 5.3 Why the LLaDA-SFT null is sufficient for an honest external-validity
story

The bounded probe at K = 8 shows:

1. Uniform-is-un-beaten reproduces at T3 tier.
2. MC-oracle does not show positive headroom over uniform at tested budgets.

Points 1 and 2 together support the claim "the universal-uniform empirical
regularity holds on a second backbone; the MC-oracle-headroom precondition
for adaptive gains is not measurable at the tested bounded K = 8 setup". This
is exactly the scope the Negative-Result Corollary needs — it does **not**
need K = 30 LLaDA-SFT to land as an external-validity appendix.

---

## 6. The strongest mathematically principled direction (for this thesis)

Given the landscape positioning above, the direction that maximizes both
**novelty relative to the literature** and **file-backed defensibility** is:

**Main body:** Theorem A + A′ + A″ + Prop B + Prop C, with all constants
measured on OWT artefacts, anchored by mean_delta_oracle saturation and the
Phase 3a search-class positive.

**External-validity appendix:** the LLaDA-SFT K = 8 bounded probe documented
as a replication of the ranker-class observation at T3 tier, with honest
scoping of the MC-oracle non-transfer.

**Future-work appendix (F):** Theorem A-ad (F1 CMDP) + Protocol C result —
preliminary positive or honest negative. Explicit citation of PG-DLM for
F3-class algorithmic precedent. Explicit statement that adaptive-submodularity
(F4) is falsified by Prop C.

This is the direction implemented by
`NEXT_RESEARCH_DIRECTION_DECISION.md`.

---

## 7. What this positioning explicitly does NOT claim

- Theorem A is not the first MDM error bound. L & Z Error Bounds preceded it
  for the predictor-only E_fact / E_learn decomposition.
- Theorem A-ad is not the first MDP-framed MDM theorem. Learning Unmasking
  Policies uses the MDP framing for the predictor object. Theorem A-ad is
  the first MDP-framed theorem for *corrector scheduling* under fixed
  predictor and kernel.
- PG-DLM-class particle Gibpbs on MDMs is not novel — the thesis does not
  re-implement it.
- Broad-task external validity is not claimed. OWT is the main discovery
  backbone; LLaDA-SFT is a bounded probe at T3 tier.

---

## 8. Links

- `NEXT_RESEARCH_DIRECTION_AUDIT.md` — audit that anchors this positioning.
- `NEXT_RESEARCH_DIRECTION_DECISION.md` — decisive implementation plan.
- `ADAPTIVE_BUDGETED_CONTROLLERS.md` — F1 CMDP + Theorem A-ad + Protocol C.
- `THEORY_STATUS.md` — open-loop theorem status.
- `research/candidate_theorems.md` — Theorem A + A′ + A″ + Prop B + Prop C.
- `research/proof_ledger.md` — provenance tags.
- `docs/thesis/experiments/CROSS_BACKBONE_REPLICATION_RESULTS.md` — LLaDA-SFT
  bounded probe.

---

*End of landscape positioning. The thesis occupies the corrector-scheduling
slot of Ring 1–5 — an empty slot that PG-DLM, E-SMC, EAGS, Learning-Unmasking,
APD, and the kernel-design papers do not fill.*

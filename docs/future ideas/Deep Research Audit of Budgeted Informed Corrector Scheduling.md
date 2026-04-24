# Deep Research Audit of Budgeted Informed Corrector Scheduling

## Executive verdict

Your current thesis direction is **still defensible**, but only under a **narrow and explicitly qualified framing**: *budgeted scheduling of a fixed corrective kernel on a fixed predictor trajectory*. Under that framing, the repo already contains a real positive result: paired evaluations on ProSeCo-OWT show that greedy/separable rankers underperform, while schedule-aware search methods recover a large share of oracle headroom, which is exactly the empirical signature of a trajectory-level control problem rather than a separable per-step scoring problem. The repository’s canonical documents now say this quite clearly, and the Phase 3a combinatorial baselines are the strongest evidence in the project so far. citeturn3view2turn3view3turn2view11turn3view4turn3view5

The problem is that **ProSeCo-OWT is not a good enough sole empirical anchor for a thesis that wants broad relevance in 2026**. It is a clean vehicle for studying explicit corrective calls, but it is also a small, specialized, unconditional OWT setup, and your quality functional is defined as GPT-2 reference negative NLL in the ProSeCo protocol documents. That creates an external-validity problem: the modern self-correction frontier in diffusion language models has moved toward larger backbones, explicit remasking/revision policies, code/reasoning tasks, and correction-specific evaluation protocols such as executable revision benchmarks. citeturn16view2turn2view10turn18view2turn19view3turn18view4turn40view2turn40view5

My decisive recommendation is therefore:

**Stay with ProSeCo for now, but only as a temporary vehicle while preparing a stronger replacement or complement.** More concretely: keep the current ProSeCo-OWT results as the thesis’s clean “fixed corrector kernel” mainline, but add at least one modern external-validity layer—preferably either **ProSeCo-LLaDA-SFT** as a same-family stronger backbone, or a **correction-specific benchmark layer** such as CDLM’s Code Revision Benchmark on top of LLaDA, Dream, or Open-dLLM. If you have time for only one major extension, I would prioritize **cross-backbone replication over ProSeCo-LLaDA-SFT**, because it preserves the explicit-corrector object while reducing the “one quirky 0.2B OWT model” risk. citeturn7view0turn7view6turn39view2turn20search1

## What your repo already proves and what it does not

Your repo now makes a much stronger claim than the original “aggregate signals might beat uniform” story. The README and current index say the thesis’s empirical verdict is that **fixed-budget corrector allocation is a combinatorial trajectory-control problem**, with greedy per-step rankers being the wrong solution class. Phase 3a reports that coordinate descent with true-\(G\) feedback and beam search with cheap \(A\)-ranking plus true-\(G\) rollouts both beat uniform at every tested budget on ProSeCo-OWT, with coordinate descent recovering roughly 74–84% of Monte Carlo oracle headroom at \(B \in \{2,3,4\}\). citeturn3view2turn3view3turn2view11turn3view4

Just as important, the repo also documents a serious earlier failure mode: Phase 1 confused the additive surrogate \(A(S)\) with the true end-to-end joint gain \(G(S)\). The audit file shows that a previously celebrated policy lost badly when evaluated on the actual pipeline, and the repository explicitly reframed the thesis around paired, seed-aligned evaluation after that failure. This is good science, but it also means the thesis now rests on a much narrower load-bearing set of assumptions than the broad “signal-adaptive scheduling” title suggests. citeturn16view3turn16view4turn16view5

The most important hidden assumption in the current mainline is not about search; it is about **the empirical object being worth generalizing from**. In the protocol mapping, the state is a ProSeCo backend with fixed predictor sampler, fixed sequence length, fixed corrector depth, and a quality functional defined by GPT-2 reference negative NLL. That is a coherent sandbox, but it is not the same as “self-correction quality” in the stronger sense used by more recent work on code repair, reasoning, or correction-specific benchmarks. citeturn16view2turn16view1

A second load-bearing assumption is that ProSeCo-OWT remains representative enough of “informed corrector scheduling.” The repository’s earlier backend audit is revealing here: at one point, the effective implementation used the MDLM backbone with a ProSeCo-style corrector layered on top, and the explicit reason for choosing ProSeCo was that the corrective mechanism made positive \(\Delta_t\) values plausible, unlike a harmful MDLM Gibbs-style resample. By Phase 2b, the repo had moved to the co-trained ProSeCo-OWT checkpoint. That progression is not a flaw, but it shows how much of the thesis depends on this one corrective family. citeturn16view0turn16view1turn2view10

A compact way to state the current evidence is this:

| Question | What the repo supports | What it does **not** yet support |
|---|---|---|
| Is greedy per-step ranking enough? | No. Search-class procedures beat the ranker envelope on ProSeCo-OWT. citeturn3view3turn3view4turn3view5 | That this is universally true across corrective backbones or tasks. |
| Is the problem trajectory-level? | Yes, on the current backend and metric, that is the clearest reading of Phase 2b and 3a. citeturn2view10turn3view4turn15view0 | That a single universal state-summary or theory already captures it. |
| Is ProSeCo the best general experimental platform? | It is a clean explicit-corrector platform. citeturn19view0turn7view0 | No; the repo does not yet establish that it is the most relevant 2026 platform. |
| Is the thesis already externally validated? | No. Cross-backbone replication is explicitly parked. citeturn3view3 | External validity remains a genuine open risk. |

## Critical audit of ProSeCo as the backend

ProSeCo is a **good fit for your exact object**, but only a **partial fit for your broader research ambition**. The ProSeCo paper is one of the cleanest available papers for what you actually care about: it explicitly trains a model to do both unmasking and correction, and at inference time it inserts additional corrective refinement steps between predictor steps. That means a fixed corrective-call budget \(B\) is not an artificial abstraction; it is a native control variable in the method. This is the strongest argument for keeping ProSeCo in the thesis. citeturn19view0turn19view1turn7view0

The weakness is that ProSeCo is **not where the field’s center of gravity currently sits**. Since 2025, the diffusion-language-model literature has expanded around remasking, explicit confidence-guided revision, larger open backbones like LLaDA and Dream, correction-specific code benchmarks, trajectory-refinement samplers such as PG-DLM, token-wise refinement controllers like PRR, and trajectory-aware post-training such as TraceRL. ProSeCo is still relevant, but it now looks like one branch in a rapidly diversifying ecosystem rather than the obvious canonical platform. citeturn18view1turn18view2turn19view3turn18view4turn24search0turn10view0turn40view5

The most serious criticisms of the current setup are these:

| Rank | Criticism | Seriousness | Does it threaten the thesis? | Concrete fix |
|---|---|---:|---|---|
| highest | **Evaluation mismatch**: the repo’s main \(F\) is GPT-2 reference neg-NLL, while current correction papers increasingly use task-grounded metrics such as success rate, code executability, benchmark accuracy, or reward-guided objectives. citeturn16view2turn18view2turn18view4turn24search0turn40view2 | Very high | Yes, if you claim general “sequence quality” | Add a correction-specific benchmark layer, ideally CRB-style code revision or at least a stronger secondary metric. |
| very high | **Backend idiosyncrasy**: the current positive result is on one small OWT checkpoint family. citeturn2view10turn7view6 | Very high | Yes, if you claim generality | Replicate on ProSeCo-LLaDA-SFT or another modern backbone. |
| high | **Availability-driven choice**: the repo itself shows practical convenience mattered heavily in backend choice. citeturn16view0turn16view1 | High | Not fatal, but intellectually weak | Rejustify ProSeCo by object-fit, then add one non-convenience-based complement. |
| high | **Field drift toward remasking/self-revision rather than explicit corrector calls**. PRISM, RemeDi, ReMDM, and CDLM all center revision/remasking or correction-oriented confidence, not ProSeCo-style inserted corrector calls. citeturn18view1turn18view2turn19view3turn18view4 | High | Only if thesis pretends to cover “self-correction in DLMs” broadly | Narrow thesis language to “fixed-kernel corrective scheduling,” or broaden experimentally. |
| medium-high | **Unclear transfer to larger modern DLMs**. Large open models like LLaDA, Dream, Dream-Coder, and TraDo now dominate many public evaluations. citeturn20search1turn21search0turn21search6turn40view5 | Medium-high | Yes for broad relevance; no for a tightly scoped MSc thesis | Add one large-model replication or at least a benchmark using such models. |
| medium | **Potential theorem–backend mismatch**. Your theorem story wants principled scheduling, but ProSeCo’s empirical object is still one very specific corrective mechanism. citeturn2view7turn3view5 | Medium | Not fatal if stated honestly | State clearly that theorems apply to a fixed corrector kernel, not to all self-correction methods. |

The bottom line on ProSeCo is therefore nuanced. **It is not “the wrong backend.”** In fact, among publicly available systems, it is one of the few that cleanly expose explicit corrective refinement as a separate inference-time operation. But it is also **too idiosyncratic to carry the entire thesis by itself** without either cross-backbone replication or a stronger evaluation axis. citeturn19view0turn7view6turn39view2

## Better backbones and where the frontier has moved

If you want to stay as close as possible to your current object, the strongest replacement or complement is **ProSeCo-LLaDA-SFT**. It stays in the same method family, the same explicit corrector framing, and is publicly released in the ProSeCo repository and Hugging Face collection, but it moves you from a 0.2B OWT setup to an 8B-class diffusion LM that is much closer to the current open-model frontier. That is the cleanest “same thesis, stronger backend” move I found. citeturn7view0turn7view6

If you are willing to broaden the empirical object slightly, the most relevant neighboring systems are **PRISM**, **RemeDi**, **CDLM**, and **ReMDM**. PRISM is attractive because it is lightweight, model-agnostic, and explicitly learns per-token quality scores for remasking; RemeDi is even more current and adds a dual-stream architecture with explicit confidence-based remask/unmask control; CDLM is especially valuable because it provides a correction-oriented training principle and, crucially, the **Code Revision Benchmark**; ReMDM remains the cleanest sampling-time remasking baseline for masked diffusion models and is directly about inference-time scaling. These systems are not identical to informed-corrector scheduling, but they are exactly the literature you will be judged against. citeturn18view2turn19view3turn18view4turn18view1

A practical shortlist is below.

| Candidate | Year | Code | Weights/checkpoints | Explicit corrector or revision mechanism | Closer to | Fixed-budget intervention experiments | Adaptation difficulty | Better or worse than ProSeCo for **your** problem |
|---|---:|---|---|---|---|---|---|---|
| **ProSeCo-OWT** | 2026 | Yes. citeturn7view0 | Yes, 0.2B OWT. citeturn7view6 | Yes: inserted corrective refinement steps between unmasking steps. citeturn19view0 | Informed-corrector scheduling | Excellent | Low | Best *clean* current sandbox; weak external validity. |
| **ProSeCo-LLaDA-SFT** | 2026 | Yes. citeturn7view0 | Yes, 8B SFT model. citeturn7view6 | Same ProSeCo correction family. citeturn7view0turn19view0 | Informed-corrector scheduling | Excellent | Medium | Best same-family upgrade; my preferred complement/replacement. |
| **PRISM** | 2025–2026 | Yes. citeturn7view2 | Fine-tuning scripts and base-checkpoint paths provided; OWT and LLaDA supported. citeturn8view0turn18view2 | Per-token quality scores computed in the same forward pass; used for remasking. citeturn18view2 | Learned self-correction / remasking | Good | Medium | Great **baseline/comparator**; less aligned with your schedule-level thesis because it is fundamentally a token-quality controller. |
| **RemeDi** | 2025–2026 | Yes, inference code. citeturn38view5 | Yes, weights on Hugging Face. citeturn8view2turn38view5 | Dual-stream model outputs token predictions and confidence-based remask/unmask policy. citeturn19view3turn38view5 | Trajectory-level self-reflective remasking | Good | High | Strongly relevant frontier model, but it changes the thesis object away from explicit corrector-call scheduling. |
| **CDLM plus CRB** | 2025–2026 | Yes. citeturn39view2turn39view3 | Uses public HF model IDs for LLaDA, Dream, Open-dLLM; corrective training marked “coming soon.” citeturn39view2 | Iterative refinement and correction-oriented evaluation; benchmark is the main asset. citeturn18view4turn39view3 | Correction benchmark / refinement | Good | Medium | Best **evaluation-layer upgrade** even if you keep ProSeCo. |
| **ReMDM** | 2025 | Yes. citeturn7view1 | Demo on MDLM checkpoint; MDLM OWT checkpoint available. citeturn8view9turn12search0 | Remasking samplers such as ReMDM-loop/conf/cap/rescale. citeturn8view9turn18view1 | Remasking and inference-time scaling | Excellent | Low-medium | Important neighboring baseline, but the main object is remasking, not informed correctors. |
| **Open-dLLM / Open-dCoder** | 2025 | Yes. citeturn40view0turn40view2 | Yes. citeturn40view0turn40view1 | Not correction-native, but fully open training/eval stack. citeturn40view0turn40view2 | Reproducible DLM infrastructure | Good | Medium | Excellent if you want a reproducible modern platform for code tasks. |
| **LLaDA / Dream / Dream-Coder / DiffuCoder / TraDo** | 2025–2026 | Yes. citeturn20search1turn21search0turn21search12turn21search5turn40view5 | Yes. citeturn20search0turn20search2turn21search9turn21search11turn40view4 | Usually not explicit corrector-call models, but strong DLM backbones for revision/control experiments. citeturn7view5turn21search0turn21search6turn40view5 | General DLM frontier | Variable | Medium-high | Better for broader relevance; worse if you want to keep the exact fixed explicit-corrector object. |

The strongest overarching trend is that the frontier has shifted from “can diffusion LMs generate well?” to “how do we control, refine, accelerate, and post-train their trajectories?” PG-DLM treats inference-time control as trajectory refinement over full denoising paths; PRR explicitly argues refinement control is dynamic and trajectory-grounded rather than step-local; T3D uses trajectory self-distillation to improve few-step decoding; TraceRL makes post-training itself trajectory-aware; and Fast-dLLM/SlowFast focus on dynamic compute allocation and decoding control. That means your core empirical finding—*trajectory matters more than separable step scores*—is actually well aligned with the broader 2025–2026 direction. The problem is not that your question is obsolete. The problem is that your **current backend and metric are lagging behind the frontier where that question is now being asked**. citeturn24search0turn10view0turn11view0turn40view5turn37search0turn37search2

## Theoretical frameworks that actually fit

The most principled existing theory for your exact problem is a **finite-horizon constrained or budgeted MDP**. If the predictor trajectory is fixed and the corrector kernel is fixed, then the natural state is “current denoising state plus remaining budget,” the action is whether to spend a corrector call at step \(t\), and the objective is expected terminal sequence quality under a hard budget. This is exactly the kind of dynamic resource-allocation problem that constrained MDPs and budgeted MDPs were designed for. In that sense, your instinct to view the thesis as **budgeted path optimization** is mathematically sound. citeturn26search1turn25search6turn26search0turn25search18

The most plausible imported theory from another field is **trajectory-space Sequential Monte Carlo and Particle Gibbs over paths**, not because it gives the cleanest theorem statement, but because it is the closest imported framework that is already being used on diffusion language models themselves. PG-DLM explicitly says prior methods optimize rewards step-by-step within single denoising trajectories, and introduces Particle Gibbs precisely to enable **trajectory-level refinement** with convergence guarantees and adaptive compute allocation. That is extremely close in spirit to your empirical finding, even though PG-DLM optimizes reward-guided text generation rather than the exact placement of informed-corrector calls. citeturn24search0turn7view7turn30search2

Belief propagation is **plausible only as an approximation framework**, and in a very specific role. It is not the right foundational model of the sampled text trajectory itself. It becomes plausible if you first construct an **energy model over schedule variables**—for example, binary variables \(a_t \in \{0,1\}\) indicating whether a corrector is applied at step \(t\)—with unary terms for individual step value and pairwise or sparse higher-order terms for interaction/complementarity. Then max-sum belief propagation can approximately optimize that schedule objective; it is exact on trees and approximate on loopy sparse graphs. So BP is plausible as an **offline surrogate optimizer over schedules**, not as the main generative-control formalism. citeturn28search1turn28search0turn28search13

Adaptive submodularity is, in your setting, mainly a **falsification lens rather than a positive solution framework**. The reason is simple: adaptive submodularity and adaptive sequence submodularity are valuable because they provide greedy approximation guarantees under adaptive diminishing returns. Your own findings point the other way: greedy rankers mostly fail, while schedule-aware search succeeds, which strongly suggests complementarity and interaction patterns that violate the diminishing-returns structures these theories require. So adaptive submodularity is still useful—but primarily to say, “if this objective were adaptive sequence submodular, greedy should have worked reasonably well; empirically it did not.” citeturn31search0turn25search5turn3view4turn15view0

I did **not** find existing work that directly formulates *informed-corrector scheduling under a hard budget along a fixed predictor trajectory* as a path/control/MDP problem. What I found instead is a ring of near neighbors: ProSeCo for explicit corrective refinement, ReMDM and PRISM and RemeDi for remasking/self-correction, PG-DLM for trajectory refinement with SMC/Particle Gibbs, diffusion search papers that formulate inference-time scaling as search, and PRR for dynamic refinement control. So the thesis question still looks open—not because no one studies trajectory control, but because **your exact control variable is unusually specific and relatively unclaimed**. citeturn19view0turn18view1turn18view2turn19view3turn24search0turn10view0turn9search7turn10view4

A practical comparison is below.

| Framework | Category | Exact mathematical object optimized | Budget-aware | Additivity / diminishing-returns assumptions | Can model interactions or complementarity? | Online or offline | Exact / approximate / heuristic | Fit to informed correctors | Thesis verdict |
|---|---|---|---|---|---|---|---|---|---|
| **Finite-horizon CMDP / BMDP** | A | \(\max_\pi \mathbb{E}[F(y_T)]\) over policies with budget state \(b_t\) and transition dynamics induced by predictor+corrector. citeturn26search1turn25search6turn26search0 | Yes | No separability required | Yes, through state transition dynamics | Both | Exact in principle, approximate in practice | Excellent | **Best formal statement** of the problem. |
| **Trajectory-space SMC / Particle Gibbs / Feynman-Kac** | C | Sampling or optimizing path distributions over full trajectories, often reward-weighted. citeturn24search0turn7view7turn30search2 | Yes | No additivity needed | Yes | Online iterative refinement | Approximate but theoretically grounded | Very good | **Best imported algorithmic theory**. |
| **Control-as-inference / KL-control / path-integral control** | C | Variational control objective over controlled vs passive path distributions, often with KL regularization. citeturn27search1turn27search11turn27search15turn27search16 | Soft-budget naturally; hard-budget less natural | No additivity, but needs specific control-cost structure | Yes | Usually online / planning | Exact only for restricted classes | Good conceptually | Elegant lens, but probably too indirect for an MSc core theorem. |
| **Factor graph / pairwise energy / max-sum BP** | D | Approximate schedule objective \(E(a)=\sum_t \phi_t(a_t)+\sum_{t<t'}\psi_{tt'}(a_t,a_{t'})\). citeturn28search1turn28search0 | Yes, via cardinality factors | Assumes low-order factorization | Yes, if encoded in factors | Mostly offline | Exact on trees, approximate on loopy graphs | Medium | Plausible **approximation layer**, not foundational theory. |
| **Adaptive submodularity / adaptive sequence submodularity** | C | Greedy maximization of adaptive diminishing-returns objective over policies/sequences. citeturn31search0turn25search5 | Yes | **Requires** diminishing returns | Weakly, but only under the theory’s structure | Usually online | Approximation guarantees | Poor as a positive story | Best used as a **foil / falsification device**. |
| **Weakly coupled MDPs / restless bandits** | C | Resource allocation across multiple independent sub-processes linked by budgets. citeturn29search2turn29search3 | Yes | Usually exploits decomposability across arms/projects | Limited if one arm strongly affects everything else | Online | Approximate / relaxation-based | Poor | Probably too broad and structurally wrong here. |
| **Statistical-physics free-energy / Ising-style schedule models** | D | Gibbs distribution over schedules or paths, e.g. \(p(a)\propto e^{-\beta E(a)}\). citeturn27search16turn30search2 | Yes | Depends on chosen energy truncation | Yes | Offline or sampling-based | Approximate | Medium | Interesting if you fit pairwise interactions; risky as a main theorem story. |

My recommended balance of **rigor, novelty, tractability, and fit** is:

| Framework | rigor | novelty | tractability | fit to informed correctors | risk level |
|---|---|---|---|---|---|
| Finite-horizon budgeted MDP | High | Medium | Medium | High | Low-medium |
| Trajectory-space SMC / Particle Gibbs | High | High | Medium-low | Medium-high | Medium |
| Control-as-inference / KL-control | High | Medium-high | Low-medium | Medium | Medium-high |
| Pairwise energy plus max-sum BP | Medium | High | Medium | Medium | Medium |
| Adaptive submodularity as falsification | Medium-high | Medium | High | Low as a solution, high as a diagnostic | Low |

A strong theorem story under each framework would look like this:

- **Budgeted MDP**: formalize the exact problem as a finite-horizon MDP with remaining budget in state; show existence of an optimal non-stationary Markov policy; then prove approximation or regret bounds for compressed-state policies based on trajectory summaries. This is the cleanest and most honest route. citeturn26search1turn25search18  
- **Particle-Gibbs / SMC**: define a target path distribution over schedules or corrected trajectories; prove your refinement kernel leaves it invariant; show asymptotic consistency or monotone improvement in the anytime limit. Strong but more work. citeturn24search0turn30search2  
- **BP / energy model**: prove that if \(G(S)\) is well approximated by a sparse pairwise energy, then max-sum on that factorization is exact or approximately optimal up to the residual higher-order interaction error. Plausible for an approximation theorem. citeturn28search1turn28search0  
- **Adaptive submodularity**: prove the opposite of what you first hoped—namely that the assumptions needed for greedy guarantees are violated, so the theory serves as a negative characterization of why greedy fails. citeturn31search0turn25search5turn15view0  
- **KL-control**: show a soft-budget relaxation yields a variational control objective over schedules, but unless you can identify a sensible passive schedule process and KL control cost, it risks becoming elegant but too detached from your empirical setup. citeturn27search1turn27search11

## Recommended next move

### Continue or switch

My recommendation is **option two**: **stay with ProSeCo for now, but only as a temporary vehicle while preparing a stronger replacement**. Do **not** throw away the current results: they are already thesis-grade evidence that schedule-level search matters. But do **not** let the final thesis depend on ProSeCo-OWT alone. The right move is a staged upgrade, not a wholesale pivot. citeturn3view2turn3view3turn3view4turn7view6

### Best next experiments

The next experiments I would run, in order, are:

1. **Cross-backbone replication on ProSeCo-LLaDA-SFT** using the exact same scheduling question and paired evaluation protocol. This is the cleanest test of whether your trajectory-level finding survives a stronger, current backbone within the same corrective family. citeturn7view0turn7view6

2. **Add one correction-specific benchmark layer**, ideally CDLM’s CRB-style setup or at minimum a controlled code-revision task on top of LLaDA, Dream, or Open-dLLM. This directly addresses the biggest flaw in the current setup, which is that GPT-2 reference neg-NLL is a weak proxy for correction quality. citeturn18view4turn39view2turn40view2

3. **Formalize the problem as a finite-horizon budgeted MDP** and rewrite the theory chapter around that formal object. Keep the current empirical result as evidence against separable rankers, not as evidence against informed scheduling in general. citeturn3view5turn26search1turn25search6

4. **Run one imported trajectory-refinement baseline**, ideally a schedule-space analogue of Particle Gibbs or at least a more principled path-search baseline beyond coordinate descent and beam search. This would connect your empirical findings to the broader 2025–2026 trajectory-refinement literature. citeturn24search0turn7view7

5. **Use adaptive submodularity as a falsification benchmark**, not as the solution. Try to quantify complementarity directly—for example by testing whether marginal value of a corrector at time \(t\) systematically depends on what other times have already been selected. If it does, you have a stronger formal explanation for why greedy failed. citeturn31search0turn25search5turn15view0

### Strongest thesis framing after this audit

If you **stay with ProSeCo**, the strongest framing is:

> *Budgeted trajectory control for corrective refinement under a fixed corrector kernel.*  
> The empirical object is a fixed predictor plus fixed ProSeCo corrective operator; the scientific question is when to spend a limited number of corrective calls; the main result is that this is not a separable ranking problem but a schedule-level control problem. citeturn19view0turn3view4

If you **switch model but keep the same object**, the strongest framing is:

> *Cross-backbone budgeted scheduling of explicit self-correction in diffusion language models.*  
> ProSeCo-LLaDA-SFT is the cleanest upgrade because it preserves the object while improving relevance. citeturn7view0turn7view6

If you **broaden the formulation**, the strongest framing is:

> *Inference-time trajectory refinement under a fixed compute budget in diffusion language models.*  
> This broader framing would let you connect your work to ProSeCo, ReMDM, PRISM, RemeDi, PG-DLM, PRR, and search-based inference-time scaling. It is scientifically stronger, but only if you are comfortable no longer centering “informed correctors” as the sole object. citeturn18view1turn18view2turn19view3turn24search0turn10view0

### Piece-by-piece takedown

| Attack on the current experiment | Does the attack succeed? | Why |
|---|---|---|
| “Greedy ranking was the wrong abstraction.” | **Yes.** | Your own paired results and search-class positives already show this. citeturn3view4turn15view0 |
| “ProSeCo is too idiosyncratic to support a general thesis.” | **Partly yes.** | It is a clean explicit-corrector platform, but not enough by itself for broad claims. citeturn19view0turn7view6 |
| “The evaluation metric is too weak.” | **Yes.** | GPT-2 reference neg-NLL is too far from modern correction benchmarks. citeturn16view2turn18view4turn40view2 |
| “The field has moved on to remasking and trajectory refinement.” | **Yes, partly.** | Frontier work now emphasizes remasking, confidence-guided revision, path refinement, and trajectory-aware training. citeturn18view1turn18view2turn19view3turn24search0turn10view0turn40view5 |
| “Therefore you must abandon ProSeCo immediately.” | **No.** | ProSeCo still uniquely exposes explicit corrective calls and is therefore unusually well matched to your narrow object. citeturn19view0turn7view0 |
| “Your question is already better answered elsewhere.” | **No, not exactly.** | Nearby work exists, but I did not find a paper directly solving hard-budget placement of informed corrector calls along a fixed predictor trajectory. citeturn24search0turn10view0turn18view1turn18view2turn19view3 |
| “This should not be treated as trajectory optimization.” | **No.** | Given greedy failure and schedule-aware recovery, trajectory optimization is the right abstraction unless future evidence reverses it. citeturn3view4turn15view0 |

### Why this should still be treated as a budgeted path problem

At this point, I see **very little reason not to** treat your setting as a budgeted path or trajectory optimization problem. The only serious counterargument would be that corrector effects are effectively additive or that future states are conditionally independent enough that a top-\(B\) ranker ought to work. Your current evidence points the other way. The stronger caution is different: you must make sure you are not solving a path problem for a **misaligned metric**. That is a reason to improve evaluation, not a reason to abandon the trajectory-control formulation itself. citeturn3view4turn16view3turn16view4

## Ranked papers, frameworks, formulation, and reading plan

### Most relevant papers and resources

1. **Learn from Your Mistakes: Self-Correcting Masked Diffusion Models** — ProSeCo is still the cleanest explicit-corrector paper for your exact action abstraction. citeturn19view0turn7view0  
2. **Inference-Time Scaling of Diffusion Language Models with Particle Gibbs Sampling** — closest path-level control formulation already applied to DLMs. citeturn24search0turn7view7  
3. **Fine-Tuning Masked Diffusion for Provable Self-Correction** — strongest learned self-correction comparator and an important foil to your schedule-level story. citeturn18view2turn7view2  
4. **Don’t Settle Too Early: Self-Reflective Remasking for Diffusion Language Models** — strong frontier paper on revision-centric DLMs. citeturn19view3turn38view5  
5. **Corrective Diffusion Language Models** — especially important for the benchmark question because of CRB. citeturn18view4turn39view3  
6. **Informed Correctors for Discrete Diffusion Models** — not language-central in its large-scale experiments, but foundational for the informed-corrector idea. citeturn23view0  
7. **Remasking Discrete Diffusion Models with Inference-Time Scaling** — the core remasking neighboring line. citeturn18view1turn8view9  
8. **Progressive Refinement Regulation for Accelerating Diffusion Language Model Decoding** — shows that modern refinement control is now explicitly trajectory-grounded. citeturn10view0turn36search0  
9. **Large Language Diffusion Models** — important backbone paper because LLaDA is now a major open platform. citeturn7view5turn20search1  
10. **dLLM: Simple Diffusion Language Modeling** — practical framework resource if you want to move beyond ad hoc codebases. citeturn32view0turn33search0  
11. **Simple and Effective Masked Diffusion Language Models** — MDLM remains the foundational masked-DLM baseline behind several later systems. citeturn12search5turn12search0  
12. **Diffusion Tree Sampling** and **Inference-Time Scaling of Diffusion Models through Classical Search** — valuable for path-search analogies even though they are not your exact object. citeturn9search0turn9search7  
13. **Theoretical Benefit and Limitation of Diffusion Language Model** — useful for understanding why correctness metrics can behave very differently from perplexity. citeturn35view0  
14. **Adaptive Submodularity** and **Adaptive Sequence Submodularity** — mainly as falsification lenses for why greedy should not be expected to work if complementarity dominates. citeturn31search0turn25search5  
15. **Constrained Markov Decision Processes / Budgeted MDPs** — the most principled formal basis for the thesis statement itself. citeturn26search1turn25search6turn26search0

### Most promising theoretical frameworks

1. **Finite-horizon budgeted MDP with remaining-budget state augmentation**. citeturn26search1turn25search6  
2. **Trajectory-space SMC / Particle Gibbs over denoising or schedule paths**. citeturn24search0turn30search2  
3. **Control-as-inference / KL-control as a soft-budget relaxation**. citeturn27search1turn27search11  
4. **Offline pairwise energy model over schedules plus max-sum BP or beam search**. citeturn28search1turn28search0  
5. **Adaptive sequence submodularity as a negative benchmark, not as the final solution**. citeturn25search5turn31search0

### Recommended best thesis formulation

The strongest final formulation is:

> **Budgeted trajectory control for corrective refinement in masked diffusion language models.**  
> Fix a predictor trajectory and a corrective kernel. Define a hard budget on corrective calls. Ask for the schedule that maximizes terminal sequence quality. Formalize this as a finite-horizon budgeted MDP over denoising states with remaining budget, and use your empirical results to show that the objective is not well captured by separable per-step scores. This keeps the thesis rigorous, matches your current evidence, and remains novel because the exact explicit-corrector scheduling problem is still largely unclaimed. citeturn3view4turn26search1turn25search6

### Recommended best novel but plausible approach

The most interesting novel extension is to define a **reward-weighted distribution over schedules or corrected trajectories** and perform **Particle-Gibbs or conditional-SMC refinement in schedule space**, using a cheap surrogate to propose schedules and occasional true pipeline evaluations to weight them. Conceptually, this imports the strongest current trajectory-refinement theory from DLM inference into your very specific scheduling problem without collapsing everything back to a per-step ranker. It is more ambitious than your current beam or coordinate search, but it is scientifically coherent and arguably the most publishable “next jump.” citeturn24search0turn7view7turn30search2

### Reading plan

**Must read this week**

1. ProSeCo paper and repository release details. citeturn19view0turn7view0turn7view6  
2. PG-DLM paper. citeturn24search0turn7view7  
3. PRISM paper. citeturn18view2  
4. CDLM paper plus repo README for CRB. citeturn18view4turn39view2  
5. Constrained MDP / budgeted MDP basics. citeturn26search1turn25search6  

**Useful next**

1. RemeDi. citeturn19view3turn38view5  
2. ReMDM. citeturn18view1turn8view9  
3. PRR. citeturn10view0turn36search0  
4. LLaDA and dLLM framework. citeturn7view5turn32view0turn33search0  
5. Adaptive submodularity and adaptive sequence submodularity. citeturn31search0turn25search5  

**Optional or stretch**

1. KL-control / control-as-inference. citeturn27search1turn27search11turn27search16  
2. Factor graphs and max-sum BP. citeturn28search1turn28search0  
3. Diffusion Tree Sampling and classical search for diffusion inference-time scaling. citeturn9search0turn9search7  
4. TraceRL and trajectory-aware post-training. citeturn40view3turn40view5  
5. Theoretical Benefit and Limitation of DLMs. citeturn35view0  

### Audit verdict

**The current ProSeCo-based experiment is a defensible mainline only if you frame it narrowly as fixed-kernel budgeted corrective scheduling and explicitly stop short of broad claims about self-correction in diffusion LMs.** citeturn19view0turn3view4turn16view2

### Bottom-line recommendation

The strongest honest next move is: **keep the existing ProSeCo-OWT trajectory-level result, recast the thesis formally as a finite-horizon budgeted trajectory-control problem, and immediately add one modern external-validity layer—preferably ProSeCo-LLaDA-SFT or a correction-specific benchmark like CDLM/CRB.** Do not pivot the whole thesis to PRISM or remasking unless you are willing to change the object of study. But also do not let ProSeCo-OWT plus GPT-2 neg-NLL be the entire story, because by 2026 that is too narrow, too backend-specific, and too easy to dismiss as a local artifact. citeturn7view6turn39view2turn18view2turn19view3turn24search0turn26search1
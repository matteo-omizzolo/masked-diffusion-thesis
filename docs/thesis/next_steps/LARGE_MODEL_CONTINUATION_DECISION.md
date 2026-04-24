> **STATUS:** DECISION
> **LAST VERIFIED:** 2026-04-22
> **SCOPE:** Records the evidence-based decision about whether the LLaDA-SFT 8B cross-backbone replication (currently in HPC Job 480887) should continue. Supersedes the implicit "continue because planned" posture.

---

# LARGE-MODEL CONTINUATION DECISION — LLaDA-SFT cross-backbone replication

## TL;DR

**Decision: GO for Phase 2b only on LLaDA-SFT, with strict bounds.**

- Resume **Phase 2b** (paired K-seed policies + MC oracle) only. Protocol A is already complete on this backbone and will not be re-run.
- Do **not** auto-run Phase 3a on LLaDA-SFT. Phase 3a on LLaDA-SFT is blocked on the Phase 2b result.
- Hard budget: 4 sharded sibling jobs × ≤ 48 h walltime each. If Phase 2b does not complete within one resume, STOP and accept whatever evidence has been produced.

The argument is not "finish what you started." It is: Protocol A on LLaDA-SFT has already established that the corrector is non-degenerate on this backbone, and a paired MC-oracle measurement on the same backbone is the single cheapest evidence that can either strengthen, weaken, or narrow the current OWT-based thesis claim — whichever direction the result goes.

---

## 1. Evidence available before the decision

### 1.1 Protocol A on LLaDA-SFT (file-backed, 8 seeds × 64 timesteps)

Stored in `results/cross_backbone/proseco_llada_sft_bounded/protocol_a/`.

| Seed | n_Δ>0 / n_Δ<0 / n_Δ=0 | mean Δ | median Δ | [min, max] |
|---|---|---|---|---|
| 142 | 20 / 44 / 0 | −0.326 | −0.172 | [−2.460, +0.709] |
| 143 | 53 / 11 / 0 | +0.314 | +0.362 | [−1.072, +1.337] |
| 144 | 60 / 4 / 0 | +0.673 | +0.747 | [−0.953, +1.813] |
| 145 | 56 / 8 / 0 | +0.324 | +0.420 | [−0.815, +0.903] |
| 146 | 62 / 2 / 0 | +0.595 | +0.610 | [−0.122, +2.756] |
| 147 | 54 / 10 / 0 | +0.254 | +0.347 | [−0.987, +0.726] |
| 148 | 30 / 34 / 0 | −0.118 | −0.056 | [−1.323, +0.997] |
| 149 | 49 / 15 / 0 | +0.161 | +0.345 | [−1.835, +0.736] |

Paired protocol confirmed: `f_base` is constant across all 64 timesteps within each seed (e.g. seed 142: `f_base = −5.278694`).

### 1.2 What Protocol A established (file-backed conclusions, not projections)

1. **No structural no-op.** 0 / 512 Δ_t equal exactly zero, unlike Job 478929 (ProSeCo-OWT initial attempt) where every Δ_t was 0.
2. **No structural negativity.** Unlike Job 478600 (MDLM attempt) where every Δ_t ≤ 0.
3. **Both signs in every seed.** Every seed produces both positive and negative Δ_t — rules out degenerate single-sign regimes.
4. **Qualitatively different regime from ProSeCo-OWT.** 6/8 LLaDA-SFT seeds have net-positive mean Δ (mean-of-means ≈ +0.235), whereas ProSeCo-OWT Phase 1 Protocol A was closer to symmetric around zero. This is a descriptive observation, not a claim about backbone quality — magnitudes are in NLL units under a GPT-2 reference scorer and are not cross-backbone-comparable.

### 1.3 Cost evidence

- Protocol A per seed: ≈ 7 200 s = 2 h. Total Protocol A: 16 h 07 m.
- Phase 2b per seed, observed: seed 142 not finished after ≥ 11 h on shard_count = 1. Implied per-seed cost ≥ 10 h (probably 10–20 h given 68 full T=64 trajectories per seed vs Protocol A's ≈ 128 forward-pass-equivalents per seed).
- With `shard_count=4` on 4 sibling jobs (2 seeds per shard), realistic Phase 2b wall-clock per shard is 20–40 h, well inside a 48 h walltime limit.

### 1.4 OWT evidence this run would be compared against

- Phase 2b OWT (Job 479382-era, K=30): MC-oracle paired headroom ≈ +0.45 at B ∈ {2, 3, 4}; `mean_delta_oracle` saturates by B=8.
- Phase 3a OWT (Job 479941, K=30, 4 shards): CD-G closure 74–84 %, BS-AG closure 49–64 %.
- Negative-Result Corollary (rescoped to ranker class): empirically anchored on OWT's top-K MC internal-Jaccard-near-random condition.

Cross-backbone Phase 2b on LLaDA-SFT speaks directly to whether these OWT findings generalise.

---

## 2. Scientific-value assessment

### Value if Phase 2b **confirms** MC-oracle headroom on LLaDA-SFT

- Strengthens Phase 2b claim from "on one ProSeCo-OWT backbone" to "on at least two masked-diffusion backbones."
- Permits the Negative-Result Corollary to be stated with a cross-backbone empirical anchor rather than a single-backbone one, reducing the main honesty caveat in `THEORY_STATUS.md`.
- Does **not** automatically extend to Phase 3a transfer — Phase 3a LLaDA-SFT would remain open.

### Value if Phase 2b **weakens / fails to reproduce** MC-oracle headroom

- Equally informative, arguably more so: it narrows the thesis claim to ProSeCo-OWT and identifies a real boundary condition on the result.
- Suggests the scheduling problem is backbone-specific at the B values tested, not universal. Natural chapter material.

### Value if Phase 2b **is inconclusive** (wide CIs that straddle 0)

- Worst outcome: K=8 is known to be thin for paired BCa, and Protocol A already shows seed-mean spread > within-seed SD.
- Even here, the outcome is reportable: "bounded K=8 probe; inconclusive; additional seeds left as future work."

### Which question is being answered?

The thesis question (fixed predictor, fixed corrector, fixed extra budget, when to correct) is unchanged. Phase 2b on LLaDA-SFT answers the **external-validity question**: does the ProSeCo-OWT empirical anchor for Theorem A / the Negative-Result Corollary transfer to a second backbone? That is the biggest open weakness of the OWT-only story; no other bounded experiment addresses it.

---

## 3. Cost-vs-thesis-value assessment

| Cost lever | Value |
|---|---|
| GPU-hours at 4 shards × ~24 h | ~96 GPU-h; ~24 h wall-clock if all shards run in parallel |
| Human time to prepare resume + submit + monitor | ~2 h |
| Human time to analyse | ~3 h once results exist |
| Opportunity cost against theory/write-up | Real but bounded; write-up does not need LLaDA-SFT to proceed |

The GPU cost is bounded and proportionate to a genuine external-validity check. The thesis writing can proceed in parallel with the HPC run.

**Not proportionate:**
- Any attempt to scale K beyond 8 within the same budget (would require ≥ 40 h walltime per shard and 8+ shards).
- Running Phase 3a on LLaDA-SFT before Phase 2b tells us whether headroom even exists.
- Adding a third backbone. External validity requires more backbones in principle, but this thesis cannot defensibly absorb that within its remaining timeline.

---

## 4. Options considered

| # | Option | Argument for | Argument against | Verdict |
|---|---|---|---|---|
| 1 | Continue large-model Phase 2b now (bounded) | Single cheapest external-validity signal; Protocol A already done | K=8 CIs will be wide; GPU cost real | **CHOSEN** |
| 2 | Stop large-model path here, keep as bounded Protocol-A-only probe | Protocol A alone already shows the corrector is alive on LLaDA-SFT — partial validity | Leaves the MC-oracle question unanswered on a second backbone, exactly the question that anchors Theorem A | Rejected — Protocol A alone is insufficient |
| 3 | Center thesis on OWT only, move to theory/write-up | Lowest risk, fastest to defensible draft | Accepts the single-backbone caveat as permanent; weaker external-validity story | Rejected — bounded Phase 2b is affordable enough to be worth it |
| 4 | OWT-centered thesis + LLaDA-SFT as future work | Honest minimum viable thesis | Same as 3 — the future-work box is real but the Phase 2b run is cheap enough now | Rejected for the same reason as 3, but kept as fallback if resume job fails |
| 5 | Different bounded alternative (e.g. stress-test Phase 2b on OWT at new B, or strengthen metric layer) | Could mitigate metric risk | Does not address the external-validity gap, which is the biggest stated weakness | Rejected — wrong question |

---

## 5. Strongest arguments **for** continuing

1. **Protocol A is already paid for.** The expensive non-resumable artefact (Protocol A trajectories + signal traces) exists and Phase 2b consumes them; discarding them wastes sunk GPU without removing their honesty cost.
2. **The MC-oracle question is the single most important external-validity test.** Theorem A's main empirical anchor (η_B slack, additivity) and the Negative-Result Corollary's empirical anchor (top-K MC internal Jaccard) both sit inside Phase 2b. Phase 2b LLaDA-SFT is the minimum experiment that speaks to whether those anchors transfer.
3. **Cost is bounded and sharded.** 4 parallel jobs × ≤ 48 h walltime. Failure of one shard does not block the others.
4. **Asymmetric value.** Confirm, weaken, or narrow — all three are thesis-usable. Only "inconclusive" is a loss, and even that is reportable.

## 6. Strongest arguments **against** continuing

1. **K = 8 is genuinely thin.** Per-seed mean spread > within-seed SD on Protocol A; paired BCa CIs at K=8 are wide; "inconclusive" is a realistic outcome.
2. **The Δ_t regime on LLaDA-SFT looks different from OWT.** 6/8 seeds net-positive mean suggests the corrector helps most steps on LLaDA-SFT. If uniform is already close to the MC ceiling on this backbone, the MC-oracle headroom may be smaller in NLL units than on OWT — which is interpretable but could numerically look like a null result.
3. **Opportunity cost.** Same GPU budget could strengthen the OWT metric layer (e.g. an additional reference scorer, a sanity-check dataset slice) or be spent on thesis writing.
4. **Not a thesis blocker.** The thesis has a defensible story from OWT alone with the Negative-Result Corollary stated as single-backbone.

Point (2) is the most substantive counter-argument. It is mitigated by: (a) a null headroom result on LLaDA-SFT is still informative and narrows thesis claims cleanly; (b) the paired protocol measures headroom in NLL units, so a reduced numerical headroom on LLaDA-SFT would be reported transparently rather than elided.

---

## 7. Final decision

**GO — Phase 2b only, LLaDA-SFT backbone, strict bounds.**

Binding conditions:

1. **Only Phase 2b.** No Phase 3a on LLaDA-SFT in the resume job. Phase 3a is re-assessed **after** Phase 2b analysis in `docs/thesis/next_steps/POST_CROSS_BACKBONE_DECISION.md`.
2. **Do not re-run Protocol A.** The Phase 2b-only sbatch skips preflight and Protocol A stages entirely.
3. **Shard 4-way, 48 h walltime each.** Budget cap. If Phase 2b does not complete after one continuation-resume attempt, STOP and accept the evidence.
4. **Canonical docs stay frozen w.r.t. cross-backbone conclusions** until Phase 2b file-backed results exist on disk.
5. **Cancel Job 480887 before submitting** the Phase 2b-only resume. Reason: 480887 still has ≈ 2 h walltime remaining but is demonstrably not going to complete any Phase 2b seed in that time, and leaving it running creates a race against the new sharded jobs for `phase2b_raw/per_seed/` writes.

---

## 8. Why this decision is best for the thesis

Picking the smallest next scientifically justified action that materially affects the thesis story:

- **Phase 2b LLaDA-SFT is small** (bounded cost, ≤ 2 days wall-clock with sharding) relative to the Pythia-scale or multi-backbone alternatives.
- **Phase 2b LLaDA-SFT is scientifically justified** — it is the single experiment that addresses the biggest open honesty caveat in the OWT-only story.
- **All outcomes are usable.** Pass, fail, and narrow — each maps cleanly to thesis content.
- **It does not foreclose the fallback.** If the resume job fails or Phase 2b returns inconclusive, falling back to the OWT-only thesis with LLaDA-SFT as future work remains available.

The alternative — stopping now and writing the thesis from OWT alone — is also defensible, but it permanently accepts the single-backbone caveat as part of the thesis's honesty ledger. The cost of running Phase 2b once is much smaller than the cost of that caveat in the final document.

---

## 9. What this decision does **not** authorise

- Running Phase 3a on LLaDA-SFT (separate decision after Phase 2b).
- Running Phase 2b on any additional backbone.
- Re-running Protocol A on LLaDA-SFT.
- Updating any canonical doc (CANONICAL_RESEARCH_DIRECTION, PHASE3A_COMBINATORIAL_RESULTS, THEORY_STATUS) with cross-backbone conclusions before Phase 2b file-backed results exist.
- Any budget extension beyond 4 shards × 48 h walltime without an explicit new decision doc.

---

## 10. Provenance

- Protocol A data: `results/cross_backbone/proseco_llada_sft_bounded/protocol_a/trajectory_{0..7}.json`
- Phase 2b script: `scripts/run_phase2b_proseco_owt.py` (backend-generic — `--backend proseco_llada_sft`)
- Current sbatch being superseded: `hpc/cross_backbone_proseco_llada_sft_bounded.sbatch`
- New sbatch to be written: `hpc/cross_backbone_proseco_llada_sft_resume_phase2b.sbatch`
- Plan doc: `docs/thesis/next_steps/PHASE2B_RESUME_PLAN.md`
- Result scaffold: `docs/thesis/experiments/CROSS_BACKBONE_REPLICATION_RESULTS.md` (PROVISIONAL)
- Post-Phase-2b decision: `docs/thesis/next_steps/POST_CROSS_BACKBONE_DECISION.md` (not yet written; created after results exist)

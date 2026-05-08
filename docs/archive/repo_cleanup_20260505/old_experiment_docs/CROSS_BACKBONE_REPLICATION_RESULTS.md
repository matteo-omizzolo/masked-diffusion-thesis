> ARCHIVED — historical only.
> Do not use this file to infer the current thesis status unless explicitly asked.
> Current source of truth: see `START_HERE.md` and `docs/README.md`.

---


> **STATUS:** SUPPORTING — Phase 2b bounded replication COMPLETE on LLaDA-SFT (external-validity probe); Phase 3a explicitly NOT authorized on this backbone (see §11–§12).
> **LAST VERIFIED:** 2026-04-23 (v4 — post-analysis, softened framing)
> **SCOPE:** Bounded cross-backbone replication on LLaDA-SFT 8B via ProSeCo backend, conducted as an **external-validity probe** for the OWT mainline (which remains the main discovery backbone). Records Protocol A + Phase 2b results, a partial-transfer verdict scoped to the tested bounded setup, and the non-escalation decision for Phase 3a on this backbone. Canonical thesis documents (`docs/thesis_direction.md`, `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md`, `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md`) may cite the partial-transfer verdict in §11 of this file but must continue to treat K=8 as an exploratory Tier-3 measurement (n=8 < 30) and must not broaden claims beyond the tested protocol.

---

# Cross-Backbone Replication — LLaDA-SFT (ProSeCo backend), bounded protocol

**Backbone:** LLaDA-SFT 8B via `proseco_llada_sft` backend (external/proseco, local checkpoint `~/mdm/checkpoints/proseco_llada_sft`).
**Why this run:** Phase 3a was run on a ProSeCo-OWT backbone trained from scratch; the thesis story needs at least one independent backbone test before any claim about the generality of the Phase 2b / Phase 3a findings can be made. LLaDA-SFT is the next available masked-diffusion LM with a usable ProSeCo-style corrector.
**Relation to PHASE3_ALTERNATIVE_PLAN.md:** This is the Tier-2 external-validity check mentioned at the bottom of that plan as "only justified if Phase 3a surfaces a positive." Phase 3a did surface a positive, so the bounded replication was launched.
**Status relative to the thesis:** This is an independent replication attempt. It does **not** update the Phase 2b / Phase 3a ProSeCo-OWT verdict, and until Phase 2b completes on LLaDA-SFT it cannot either confirm or refute the cross-backbone generality of that verdict.

---

## 1. Setup (factual, frozen)

| Field | Value |
|---|---|
| Backbone | LLaDA-SFT 8B (`proseco_llada_sft` backend) |
| Reference scorer (F) | GPT-2 small (per-token NLL, 512 tokens, sign flipped so larger = better) |
| Trajectory length T | 64 |
| Seeds (Protocol A) | 8 contiguous seeds starting at 142, i.e. {142, …, 149} |
| Corrector steps per branch | 1 |
| Protocol A | per-step single-loop Δ_t, bounded to K=8 seeds |
| Phase 2b | B ∈ {2, 4}; MC oracle with mc_B ∈ {2, 4}, P=24 Monte-Carlo schedules per (seed, B); paired-G evaluation |
| Phase 3a | CD-G and BS-AG at B ∈ {2, 3, 4}, resume-safe |
| HPC job (original) | 480887 — cancelled 2026-04-22 ~20:50 (would have timed out during Phase 2b seed 142 on `shard_count=1`) |
| HPC jobs (Phase 2b resume) | 481264 (shards 0..3, seeds 142..145) + 481265 (shards 4..7, seeds 146..149; `afterany:481264`), both 4× `gpu:3g.40gb` MIG slices, walltime 23:50:00 |
| sbatch file (original) | `hpc/cross_backbone_proseco_llada_sft_bounded.sbatch` |
| sbatch file (Phase 2b resume) | `hpc/cross_backbone_proseco_llada_sft_resume_phase2b.sbatch` |
| Run root | `results/cross_backbone/proseco_llada_sft_bounded/` |

All parameters are set by the sbatch file and have not been retrofitted in this doc.

---

## 2. Stage status (as of 2026-04-23, post-analysis)

| Stage | State | Evidence |
|---|---|---|
| Preflight (load checkpoint, sanity-check Δ_t at t=2) | **DONE** (in 480887) | stdout: `[PASS] Delta_t(t=2) finite -- Delta_t=+0.518446` |
| Protocol A (K=8 seeds) | **DONE** (in 480887) | 8 × `trajectory_{0..7}.json` written 19:21 (Apr 21) → 09:27 (Apr 22); `run_config.json` written 09:27 |
| Phase 2b, Job 481264 (shards 0..3, seeds 142..145) | **COMPLETED** | Exit 0:0 after 04:30:14 wall (sacct COMPLETED/0:0); 4 per-seed pairs written to `phase2b_raw/per_seed/` |
| Phase 2b, Job 481265 (shards 4..7, seeds 146..149) | **COMPLETED** | Exit 0:0 after 04:33:13 wall (sacct COMPLETED/0:0); 4 per-seed pairs written to `phase2b_raw/per_seed/` |
| HPC artifact verification | **DONE** | SSH-verified all 32 expected files present + JSON-valid + matched sizes per seed; 160 policy rows + 384 MC rows total (8 seeds × 2 B × {10 policies ∨ 24 MC samples}) |
| Local rsync-pull | **DONE** | 41 files into `results/cross_backbone/proseco_llada_sft_bounded/phase2b_raw/` with byte-level parity vs HPC |
| Analyze Phase 2b | **DONE** | `scripts/analyze_phase2b.py` produced `phase2b/{policy_comparison_paired,mc_oracle,rank_A_vs_G_phase2b}.json` + plots; 5 gates: 1 pass, 2 fail, 2 inconclusive (2 inconclusive gates require B=8 which is out of the bounded B ∈ {2,4} protocol); see §10 |
| Phase 3a (CD-G, BS-AG, B={2,3,4}) | **NOT AUTHORIZED ON THIS BACKBONE** | See §12 go/no-go: under the tested bounded setup (T=64, B ∈ {2,4}, GPT-2 reference), Phase 2b MC-oracle headroom over uniform on LLaDA-SFT is 0 at B=4 and −2.64 at B=2 — no positive headroom was observed for a search procedure to recover at these budgets |
| Analyze Phase 3a | **SUPERSEDED** | See §12 |

**Runtime ledger (original 480887 — now cancelled):**

- Job start: 2026-04-21 17:17:56
- Runtime at cancel: ~27h 23m (cancelled manually at 20:50)
- Walltime limit: 20h 00m (would have timed out regardless)
- Protocol A seed wall time: 7,241 s → 7,272 s per seed (≈ 2h 01m each; total 16h 07m across 8 seeds)
- Phase 2b seed wall time (observed under `shard_count=1`, seed 142 only): ≥ 11h 11m elapsed with **zero** per-seed artefacts written → motivated the resume with `shard_count=8`.

**Resume design (Jobs 481264 + 481265):**

- 8 shards at 1 seed each; 4 sibling shards per SLURM job on 4 MIG slices; 2 SLURM jobs in sequence under `MaxJobsPerUser=1`.
- Per-seed wall-time budget: 23:50 h (2× the observed seed-142 lower bound). If any seed times out, its partial work is discarded; completed seeds remain and `--resume` picks up the rest.
- Refer to `docs/thesis/next_steps/PHASE2B_RESUME_PLAN.md` for the full input/output reuse and regeneration contract.

---

## 3. Protocol A — factual summary across seeds

Protocol A on LLaDA-SFT ran 8 seeds × 64 timesteps = 512 per-step Δ_t observations. Per-seed aggregate:

| Seed | n_Δ>0 / n_Δ<0 / n_Δ=0 | mean Δ | median Δ | min Δ | max Δ |
|---|---|---|---|---|---|
| 142 | 20 / 44 / 0 | −0.326 | −0.172 | −2.460 | +0.709 |
| 143 | 53 / 11 / 0 | +0.314 | +0.362 | −1.072 | +1.337 |
| 144 | 60 / 4 / 0 | +0.673 | +0.747 | −0.953 | +1.813 |
| 145 | 56 / 8 / 0 | +0.324 | +0.420 | −0.815 | +0.903 |
| 146 | 62 / 2 / 0 | +0.595 | +0.610 | −0.122 | +2.756 |
| 147 | 54 / 10 / 0 | +0.254 | +0.347 | −0.987 | +0.726 |
| 148 | 30 / 34 / 0 | −0.118 | −0.056 | −1.323 | +0.997 |
| 149 | 49 / 15 / 0 | +0.161 | +0.345 | −1.835 | +0.736 |

Observed structural properties (what the raw Δ_t stream looks like, stated descriptively — **not** as claims about schedule transfer):

1. **No structural no-op.** 0 / 512 Δ_t values equal exactly zero. This is qualitatively unlike the earlier ProSeCo-OWT no-op diagnostic (Job 478929) where every Δ_t was identically 0, and unlike the MDLM structural-negative diagnostic (Job 478600) where every Δ_t was ≤ 0.
2. **Both signs appear in every seed.** Every seed produces both positive and negative Δ_t. No seed degenerates to uniformly positive or uniformly negative.
3. **Net-positive majority but not universal.** 6 / 8 seeds have mean Δ > 0; 2 / 8 (seeds 142, 148) have mean Δ < 0. Unweighted mean of seed means ≈ +0.235 in raw GPT-2-NLL units. This is reported for the record only and has no paired-CI interpretation.
4. **Paired protocol holds.** `f_base` is constant across all 64 timesteps within a seed (e.g. seed 142: `f_base = −5.278694` at every t), confirming the base reference is shared across branches — a prerequisite for the paired-G protocol Phase 2b will use.
5. **High cross-seed variance.** Per-seed means span a range of ≈ 1.0 NLL units; this is larger than any single-seed standard deviation. This is expected for K=8 but underlines that seed-average estimates from K=8 have wide CIs.

---

## 4. Cautious Protocol A interpretation (answers to the 5 questions posed)

Each answer is constrained to what Protocol A alone can show. Protocol A measures the per-step, single-loop marginal gain Δ_t at every timestep of an uncorrected trajectory; it does **not** measure joint schedule gains G(S), oracle envelopes, or any search-vs-ranker comparison. Anything about schedule transfer requires Phase 2b / Phase 3a on this backbone and is explicitly out of scope here.

**Q1 — Is the corrector alive on LLaDA-SFT at all?**
Yes. All 512 Δ_t values are strictly non-zero, and both signs are present in every seed. This rules out the two known structural failure modes on this backbone under the current bounded protocol: (a) the ProSeCo-OWT no-op mode where Δ_t ≡ 0 (suggesting the corrector kernel had no effect), and (b) the MDLM-style degenerate mode where Δ_t ≤ 0 for all t (suggesting the corrector always hurts).

**Q2 — Is there any per-step signal worth scheduling?**
At the raw-Δ_t level, yes: per-step values vary across roughly 4 NLL units (from about −2.5 to +2.8), with non-trivial spread within each seed. This is the necessary condition for scheduling to be meaningful. It is **not** a sufficient condition — Phase 2b is the stage that tests whether any schedule dominates uniform in paired G, and whether the MC oracle leaves headroom at all on this backbone. Per-step Δ_t variability alone does not license any "signal works / does not work" verdict.

**Q3 — Does the raw Δ_t distribution look comparable to ProSeCo-OWT?**
Descriptively, the distributions look broadly similar in shape (both signs, non-trivial variance) but the absolute means differ: the LLaDA-SFT K=8 bounded Protocol A is net-positive-mean in 6/8 seeds, whereas the ProSeCo-OWT Phase 1 Protocol A was more symmetric around zero. This comparison is **not** paired and uses a GPT-2-small reference rather than the model's own likelihood, so it cannot be converted into a statement about which backbone is "better" in any useful sense.

**Q4 — Does any single seed already look like it will dominate the paired G test?**
No. Protocol A on K=8 seeds cannot predict the outcome of the paired G test: G is a joint over a B-subset of timesteps, not a sum of individual Δ_t, and the additivity slack η_B is unknown for LLaDA-SFT. Seeds 144 and 146 have the most positive mean Δ_t, but even if additivity held perfectly, that only implies the corrector is beneficial on average — it does not imply any non-uniform schedule beats uniform in paired G.

**Q5 — Is there anything already visible that would invalidate Phase 2b on this run?**
Two partial concerns, both flagged for the record, neither an invalidation:
- **Cross-seed variance is large for K=8.** Two seeds (142, 148) have negative mean Δ_t while six are positive. Paired BCa CIs at K=8 are expected to be wide on this backbone. This is inherent to the bounded budget, not a run defect.
- **Absolute Δ magnitudes differ from ProSeCo-OWT.** The LLaDA-SFT backbone returns larger-magnitude Δ_t values (±2–3 NLL units) than the OWT run did. This is consistent with the larger / more capable backbone but means the Phase 2b numeric thresholds from PHASE3A_COMBINATORIAL_RESULTS.md do not transfer.

---

## 5. What the Protocol-A-only stage **could not** say (historical non-inference list)

This section is retained as a historical record of what could not be inferred from Protocol A alone. §10 supersedes most of these items with bounded Phase 2b evidence. The non-inferences are preserved verbatim to document the epistemic gate between Protocol A and Phase 2b.

1. **"Phase 2b MC-oracle headroom transfers to LLaDA-SFT."** *→ ADDRESSED in §10.* Partial non-transfer at K=8.
2. **"CD-G / BS-AG schedule search transfers to LLaDA-SFT."** *→ SUPERSEDED in §12.* Phase 3a is not authorized on this backbone.
3. **"Ranker-class negative-result corollary holds on LLaDA-SFT."** *→ NOT TESTABLE.* Its empirical anchor (MC-oracle headroom > 0) fails on this backbone at B ∈ {2, 4}.
4. **"Signal-adaptive scheduling beats uniform on LLaDA-SFT"** or its negation. *→ ADDRESSED in §10 at T3 confidence.* Uniform was not beaten under the tested bounded LLaDA-SFT setup.
5. **"LLaDA-SFT reproduces ProSeCo-OWT behaviour."** *→ ADDRESSED in §11.* Partial reproduction under bounded resolution only (see transfer matrix); OWT K=30 remains the main discovery backbone and is not displaced by this probe.

---

## 6. Resume submission — what was done

Action taken 2026-04-22 ~20:50–21:10:

1. **Cancelled Job 480887** (`scancel 480887`) after verifying `phase2b_raw/per_seed/` was empty (no work lost).
2. **Preserved** on shared BeeGFS without modification:
   - `results/cross_backbone/proseco_llada_sft_bounded/protocol_a/trajectory_{0..7}.json`
   - `results/cross_backbone/proseco_llada_sft_bounded/protocol_a/run_config.json`
3. **Wrote** `hpc/cross_backbone_proseco_llada_sft_resume_phase2b.sbatch`. It:
   - Refuses to run unless all 8 Protocol A trajectory files are present (pre-check).
   - Runs **Phase 2b only** — no preflight, no Protocol A, no Phase 3a, no analyze step.
   - Launches 4 shards in parallel on 4 MIG slices with `SHARD_START ∈ {0, 4}` env var controlling which half runs.
   - Passes `--backend proseco_llada_sft` explicitly (not `auto`), `--shard_count 8`, `--resume`.
4. **Submitted** twice:
   - `sbatch --export=ALL,SHARD_START=0 hpc/cross_backbone_proseco_llada_sft_resume_phase2b.sbatch` → **Job 481264** (seeds 142–145)
   - `sbatch --dependency=afterany:481264 --export=ALL,SHARD_START=4 hpc/cross_backbone_proseco_llada_sft_resume_phase2b.sbatch` → **Job 481265** (seeds 146–149)
5. **Verified** (as of 21:12): both jobs visible in squeue; 481264 PENDING (Priority); 481265 PENDING (Dependency on 481264); `phase2b_raw/per_seed/` still empty (no shard has started yet).

The original `hpc/cross_backbone_proseco_llada_sft_bounded.sbatch` is **not** re-used and must not be re-run — it would overwrite Protocol A.

---

## 7. Contingency if a shard times out

`--resume` in `run_phase2b_proseco_owt.py` (line 452) skips any seed whose both per-seed files exist:
```python
if args.resume and seed_policy_path.exists() and seed_mc_path.exists():
    continue
```

If Job 481264 or 481265 exits with any seeds incomplete, the recovery is a single `sbatch` resubmission of the same resume sbatch with the relevant `SHARD_START`: completed seeds are skipped on the filesystem, only remaining seeds are processed. This also respects the 2-job `MaxSubmit` and 1-job `MaxJobsPerUser` constraints of the `stud` QOS.

---

## 8. Things scientifically worth flagging early

- **Cross-backbone Δ_t baseline differs in magnitude.** LLaDA-SFT Δ_t magnitudes are roughly 5–10× the ProSeCo-OWT magnitudes from Phase 1 Protocol A. This matters for any threshold reuse; do not port OWT numeric thresholds.
- **K=8 is thin for cross-backbone CIs.** The range of seed means (−0.33 to +0.67) spans more than the within-seed standard deviation. Paired BCa CIs will be wide. This is a budget decision, not a defect, but it should be called out in any Phase 2b writeup.
- **Walltime model.** Protocol A per seed was 2h on this backbone; Phase 2b per seed is evidently ≥ 11h under shard_count=1. A realistic Phase 2b + Phase 3a rerun needs either shard_count=4 with four sibling jobs or a ≥ 48h walltime. This is the single biggest lesson from run 480887.
- **No MC-oracle headroom observed yet [pre-Phase-2b wording, retained for historical record].** Anyone reading this file in passing should not take anything in §3–§4 as equivalent to a Phase 2b positive. The MC oracle has not been computed on this backbone. *Post-Phase-2b annotation (2026-04-23): the MC oracle has now been computed (§10.4). Under the tested bounded setup, paired CI excludes the +0.05 pass threshold at B ∈ {2, 4}; see §11 for the partial-transfer verdict. This annotation supersedes the pre-Phase-2b caveat above.*

---

## 9. Provenance

- sbatch (original, produced Protocol A): `hpc/cross_backbone_proseco_llada_sft_bounded.sbatch` (job 480887, cancelled)
- sbatch (Phase 2b resume, both completed 0:0): `hpc/cross_backbone_proseco_llada_sft_resume_phase2b.sbatch` (jobs 481264, 481265)
- Protocol A script: `scripts/run_protocol_a_proseco_snapshot.py`
- Phase 2b script: `scripts/run_phase2b_proseco_owt.py` (invoked with `--backend proseco_llada_sft --shard_count 8 --resume`)
- Phase 2b analyzer: `scripts/analyze_phase2b.py`
- Phase 2b raw shards (local, 41 files): `results/cross_backbone/proseco_llada_sft_bounded/phase2b_raw/{policy_raw,mc_raw,run_config}.shard*-of-8.json` + `per_seed/*_seed{142..149}.json` + `manifest.json`
- Phase 2b analysis outputs (local): `results/cross_backbone/proseco_llada_sft_bounded/phase2b/{policy_comparison_paired,mc_oracle,rank_A_vs_G_phase2b}.json`
- Phase 2b plots (local): `figures/cross_backbone/proseco_llada_sft_bounded/phase2b/*.{png,pdf}`
- Phase 3a script: `scripts/run_phase3a_combinatorial.py` (invoked with `--resume`) — **not run on this backbone**; see §12
- Backend module: `src/mdm_playground/scheduling/backends/proseco_llada_sft.py`
- Plan reference: `docs/thesis/experiments/PHASE3_ALTERNATIVE_PLAN.md` §"What this plan deliberately does NOT do" (Tier-2 external-validity block)
- Decision references: `docs/thesis/next_steps/LARGE_MODEL_CONTINUATION_DECISION.md` (GO for Phase 2b only), `docs/thesis/next_steps/PHASE2B_RESUME_PLAN.md` (resume contract), `docs/thesis/next_steps/POST_CROSS_BACKBONE_DECISION.md` (this run's go/no-go on Phase 3a)
- Canonical reference (to be annotated, not replaced): `docs/thesis/CANONICAL_RESEARCH_DIRECTION.md`, `docs/thesis/experiments/PHASE3A_COMBINATORIAL_RESULTS.md`

This file was promoted to SUPPORTING on 2026-04-23 after Phase 2b on LLaDA-SFT completed on its own paired protocol. The transfer verdict in §11 is explicitly a K=8 bounded Tier-3 measurement and is not sufficient grounds to demote the K=30 OWT Phase 2b verdict.

---

## 10. Phase 2b results (bounded K=8, B ∈ {2, 4}, T=64)

All tier labels are **T3** ("n=8 < 30; exploratory only") by construction; these numbers are read as direction-of-effect with roughly ~1.94× wider CIs than the K=30 OWT Phase 2b run. All citations are to `results/cross_backbone/proseco_llada_sft_bounded/phase2b/*.json` unless stated otherwise.

### 10.1 Decision gates (source: `policy_comparison_paired.json` §decisions + `mc_oracle.json` §decisions + `rank_A_vs_G_phase2b.json` §decisions)

| Gate | Result | Evidence |
|---|---|---|
| `signal_beats_uniform_B8` | inconclusive | B=8 not in this bounded protocol's B_values ({2, 4}) |
| `uniform_is_optimal` | **PASS** | No policy's 95%-BCa bootstrap CI lower bound exceeds uniform's SE at any B |
| `small_B_non_vacuous` | **FAIL** | No signal policy reaches cohens_d ≥ 0.5 at B ∈ {2}; B=3 not evaluated |
| `headroom_exists_realistic` | **FAIL** | `mc_oracle_minus_uniform.B=4.bootstrap_95_ci_lo = 0.0`; threshold for pass is +0.05 |
| `A_ranks_G_cross_policy_B8` | inconclusive | B=8 not evaluated |

Three of five gates are adjudicated. Two are genuinely inconclusive because they require a budget (B=8) that was out-of-scope for the bounded protocol — **not** because evidence is missing.

### 10.2 Policy comparison at B=2 (`policy_comparison_paired.json` → `data.per_policy_per_B`)

| Policy | G_mean | G_se | paired mean vs uniform | paired bootstrap 95% CI | cohens_d |
|---|---:|---:|---:|---:|---:|
| uniform | 5.253 | 0.265 | — | — | — |
| front | 5.253 | 0.265 | 0.000 | [0.000, 0.000] | 0.000 |
| entropy_top_B_mp | 5.253 | 0.265 | 0.000 | [0.000, 0.000] | 0.000 |
| entropy_top_B_pt | 3.990 | 0.638 | **−1.263** | [−3.134, 0.000] | −0.540 |
| margin_top_B_pt | 3.990 | 0.638 | **−1.263** | [−3.089, 0.000] | −0.540 |
| quality_top_B_pt | 3.990 | 0.638 | **−1.263** | [−3.136, 0.000] | −0.540 |
| back | 0.889 | 0.108 | **−4.364** | [−4.708, −4.031] | −8.197 |
| mean_delta_oracle | 0.889 | 0.108 | **−4.364** | [−4.718, −4.025] | −8.197 |
| middle | 0.694 | 0.367 | **−4.560** | [−5.380, −3.661] | −3.477 |
| entropy_bot_B_pt | 0.323 | 0.200 | **−4.930** | [−5.524, −4.317] | −5.053 |

### 10.3 Policy comparison at B=4

| Policy | G_mean | G_se | paired mean vs uniform | paired bootstrap 95% CI | cohens_d |
|---|---:|---:|---:|---:|---:|
| uniform | 5.253 | 0.265 | — | — | — |
| front | 5.253 | 0.265 | 0.000 | [0.000, 0.000] | 0.000 |
| entropy_top_B_mp | 5.253 | 0.265 | 0.000 | [0.000, 0.000] | 0.000 |
| entropy_top_B_pt | 4.616 | 0.494 | **−0.637** | [−1.911, 0.000] | −0.354 |
| margin_top_B_pt | 3.985 | 0.643 | **−1.268** | [−3.162, 0.000] | −0.540 |
| quality_top_B_pt | 3.985 | 0.643 | **−1.268** | [−3.167, 0.000] | −0.540 |
| mean_delta_oracle | 0.965 | 0.107 | **−4.289** | [−4.664, −3.952] | −7.794 |
| back | 0.907 | 0.107 | **−4.346** | [−4.700, −4.043] | −8.466 |
| middle | 0.612 | 0.114 | **−4.641** | [−5.233, −4.116] | −5.601 |
| entropy_bot_B_pt | 0.548 | 0.212 | **−4.705** | [−5.221, −4.111] | −5.548 |

**Structural observation.** Three policies (`uniform`, `front`, `entropy_top_B_mp`) have identical G_per_seed vectors at both B=2 and B=4. This is a protocol artefact: at T=64 and B ∈ {2, 4}, the `front` policy's placement (leading timesteps) and `entropy_top_B_mp`'s mean-profile-ranked picks happen to coincide with `uniform`'s evenly-spaced picks on this backbone. Their zero differences are **not** evidence of a real tie; they are evidence that three different policy families degenerate to the same schedule at this (T, B). The three `_pt` policies (per-trajectory signals) differ from uniform on only 2/8 seeds at B=2 and 1/8 seeds at B=4 — their overall means are pulled down by those few disagreements, not by systematic losses.

### 10.4 MC-oracle headroom (`mc_oracle.json` → `data.mc_oracle_minus_uniform`)

| B | MC-oracle vs uniform, paired mean | bootstrap 95% CI | cohens_d | n_nonzero / n |
|---|---:|---:|---:|---|
| 2 | **−2.640** | [−4.065, −1.072] | −1.181 | 5 / 8 |
| 4 | **0.000** | [0.000, 0.000] | 0.000 | 0 / 8 |

**At B=4, the MC oracle finds a schedule equal to or worse than uniform on every seed.** The argmax schedules at B=4 for 6/8 seeds are variations of [0, x, y, z] with z ∈ {31, 32, ..., 61} that exactly recover uniform's G because the GPT-2 reference function and T=64 step grid make many 4-step schedules indistinguishable in G-value. For the remaining 2/8 seeds (142, 147, 148) the MC oracle cannot improve on uniform either.

**At B=2, the MC oracle is strictly worse than uniform on 5/8 seeds**, by a mean of −2.64 NLL units. This is the clearest signal on LLaDA-SFT that moving correctors away from uniform degraded G at this (T, B) pair under the tested GPT-2 reference — not a claim that placement is intrinsically harmful, not a claim about other (T, B, reference) points, and not statistically robust in the strong sense at K=8.

### 10.5 MC-oracle vs mean-profile oracle (`mc_oracle.json` → `data.mc_oracle_minus_mean_profile_oracle`)

| B | paired mean | bootstrap 95% CI | cohens_d |
|---|---:|---:|---:|
| 2 | **+1.724** | [+0.663, +3.175] | +0.871 |
| 4 | **+4.289** | [+3.939, +4.663] | +7.794 |

This is the single clean positive result in the Phase 2b matrix. Per-trajectory MC search strictly dominates a schedule chosen from the mean trajectory signal profile — on this backbone, by a large effect size at B=4 (cohens_d ≈ 7.8). Two readings are possible and both are consistent with the overall transfer matrix in §11:

- **(a)** Per-trajectory signal is informative on this backbone even though uniform still dominates MC search — this would be analogous to the OWT behaviour, except here uniform > MC.
- **(b)** The mean-profile oracle degenerates on LLaDA-SFT because `mean_delta_oracle` concentrates corrector effort on timesteps where the mean Δ_t is high; those timesteps are not where per-trajectory Δ_t is high; and the concentration itself is what destroys G. The MC oracle avoids this by re-picking per trajectory.

### 10.6 Rank-consistency of A vs G across policies (`rank_A_vs_G_phase2b.json`)

| B | pooled Spearman ρ (A, G) | 95% CI | per-seed mean ρ | per-seed SE |
|---|---:|---:|---:|---:|
| 2 | 0.091 | [−0.155, +0.320] | −0.137 | 0.249 |
| 4 | 0.079 | [−0.167, +0.323] | −0.100 | 0.212 |

Per-seed rho is highly **bimodal** at B=2 — e.g. seed 144 has ρ=−0.83, seed 146 has ρ=+0.97. This means that additivity (G ≈ B·Δ̄) holds on some trajectories and is inverted on others, at the same (T, B) and (backbone, corrector) config. At K=8 this bimodality is not statistically distinguishable from noise.

### 10.7 Runtime

- Phase 2b job 481264 (shards 0..3, seeds 142..145): 4:30:14 wall, COMPLETED 0:0
- Phase 2b job 481265 (shards 4..7, seeds 146..149): 4:33:13 wall, COMPLETED 0:0
- Total HPC GPU-hours (approximate): 2 jobs × 4 MIG slices × 4.5 h ≈ **36 GPU-hours** (each MIG slice = 1/3 of an A100)
- Local analysis runtime: <10 s on laptop CPU

---

## 11. Transfer verdict (bounded K=8 → OWT K=30)

The bounded replication answers three of the five gates that Phase 2b (OWT) adjudicated. The transfer matrix:

| Finding (source: OWT K=30 Phase 2b) | LLaDA-SFT K=8 (bounded) status | Verdict (scope-limited) |
|---|---|---|
| **Uniform is not beaten by any signal policy.** OWT K=30: no policy beats uniform's bootstrap CI. LLaDA-SFT K=8 (bounded): uniform was not beaten under the tested bounded setup (gate `uniform_is_optimal` = PASS). | Corroborated at T3 tier under bounded resolution. | **Uniform-not-beaten observation transfers under the tested bounded LLaDA-SFT setup.** |
| **Positive MC-oracle headroom over uniform.** OWT K=30: paired mean +0.45 NLL at B=4, CI [+0.16, +0.74]. LLaDA-SFT K=8 (bounded): **0.00 at B=4**, CI=[0, 0]; **−2.64 at B=2**, CI=[−4.07, −1.07]. | At the tested LLaDA-SFT budgets, the bootstrap CI excludes the +0.05 pass threshold; at B=2 the MC-oracle is also strictly below uniform on 5/8 seeds. | **Positive MC-oracle headroom did not transfer at the tested budgets (T=64, B ∈ {2, 4}, GPT-2 reference).** |
| **Ranker-class recovers 49–84% of MC headroom.** OWT: see PHASE3A_COMBINATORIAL_RESULTS. LLaDA-SFT: not testable at this bounded protocol — no positive headroom to recover. | Not testable on this backbone at this bounded protocol. | **Not applicable on this backbone at this bounded setup.** |
| **Per-trajectory signals > mean-profile schedule.** OWT K=30: same pattern at B=4. LLaDA-SFT K=8 (bounded): `mc_oracle − mean_profile_oracle` = +4.29 at B=4, CI [+3.94, +4.66], cohens_d ≈ 7.8. | Corroborated at T3 tier with large effect. | **Per-trajectory > mean-profile ordering transfers under bounded resolution.** |
| **Additivity slack η_B grows with B.** OWT: visible in Phase 2b §4.3 of that doc. LLaDA-SFT: pooled rank (A, G) correlation is not positive at either tested B — bimodal per-seed ρ. | Direction consistent; not statistically separable at K=8 (T3). | **Direction consistent under bounded resolution; CI-separable conclusion out of reach at K=8.** |

### 11.1 Interpretation (conservative)

The dominant observation — **uniform was not beaten under the tested bounded LLaDA-SFT setup** — is corroborated at T3 tier. The auxiliary OWT finding that motivated the "search procedure needed" argument — **positive MC-oracle headroom over uniform** — did **not** transfer at the tested budgets on this particular (T=64, B ∈ {2, 4}, GPT-2 reference, LLaDA-SFT + ProSeCo corrector) configuration. Neither observation generalises beyond the scope stated. In particular, nothing in this bounded probe implies that uniform placement is intrinsically optimal across backbones, or that positive MC-oracle headroom is generically absent on LLaDA-SFT at other (T, B, reference) points. OWT K=30 remains the main discovery backbone; this probe is an external-validity data point, not a replacement of that mainline.

Three hypotheses are consistent with this non-transfer at K=8 and bounded resolution:

- **H1. Corrector-dominance hypothesis.** The LLaDA-SFT + ProSeCo-corrector combination is sufficiently good at T=64 that per-step Δ_t is close to white noise around zero in G-value terms. Protocol A showed both signs in every seed and 0/512 exact zeros, which is *consistent* with this but not proof of it.
- **H2. Protocol-sparseness hypothesis.** B/T = 3.1%–6.3% on this run versus 6.3%–12.5% on OWT Phase 2b (T=32, same B). A sparser budget might be below the minimum at which placement is informative for this backbone.
- **H3. Reference-mismatch hypothesis.** GPT-2 small scoring LLaDA-SFT outputs is an off-task reference. The (nearly identical) G values for `uniform`, `front`, and `entropy_top_B_mp` at both Bs are consistent with GPT-2 being insensitive to the specific timesteps chosen whenever the final decoded sequence is coherent.

These hypotheses are not orthogonal and the K=8 run cannot discriminate among them. Any disambiguation requires either K ≥ 30 on LLaDA-SFT (expensive), a within-backbone reference (e.g. LLaDA-SFT own-likelihood), or a B-sweep that reaches B/T ≥ 10%.

### 11.2 What the thesis may and may not say after this run

**May say (with T3 tier annotations, always scoped to the tested bounded setup):**
- "The uniform-not-beaten observation from OWT Phase 2b is corroborated at K=8 on LLaDA-SFT under the tested bounded setup (T=64, B ∈ {2, 4}, GPT-2 reference)."
- "Positive MC-oracle headroom over uniform did not transfer at the tested budgets on LLaDA-SFT; the paired CI at B=4 is [0, 0] and at B=2 is [−4.07, −1.07]."
- "Per-trajectory signal dominates the mean-profile schedule on both backbones; cohens_d ≈ 7.8 at B=4 on LLaDA-SFT."

**May not say:**
- "Uniform is optimal across masked diffusion corrector scheduling." The LLaDA-SFT result is a T3 bounded-probe datum; OWT K=30 established the main uniform-not-beaten signal for the thesis, not this run.
- "The MC-oracle headroom found on OWT is a generic feature of masked diffusion corrector scheduling." The LLaDA-SFT probe contradicts the positive-headroom reading only at the tested (T, B, reference) point.
- "The ranker-class negative-result corollary generalises across backbones." Its premise (positive MC-oracle headroom exists) was not met on this backbone at this bounded setup, so the corollary is neither falsified nor confirmed on LLaDA-SFT.
- "LLaDA-SFT is a harder or easier backbone than OWT." K=8 + GPT-2 reference cannot support any such statement.
- "This run disproves, weakens, or replaces the OWT Phase 2b / Phase 3a mainline." It does not. The OWT mainline remains the main discovery backbone and the LLaDA-SFT probe is bounded external-validity evidence only.

---

## 12. Phase 3a go/no-go on this backbone

**Decision: Phase 3a (CD-G, BS-AG, B ∈ {2, 3, 4}) is NOT authorized on LLaDA-SFT at this bounded protocol.**

Rationale, in one paragraph: Phase 3a's research question is "does a ranker-class search procedure (CD-G / BS-AG) recover most of the MC-oracle headroom at the same B?". On LLaDA-SFT at T=64, B ∈ {2, 4}, and GPT-2 reference, the observed MC-oracle headroom over uniform under this bounded protocol is zero at B=4 and negative at B=2, with K=8 bootstrap CIs that exclude the OWT-level threshold of +0.05. A ranker-class search procedure cannot recover headroom where the bounded probe observed none. Running Phase 3a on this backbone at this bounded setup would either produce (a) a null result that restates §10's conclusion, or (b) a spurious positive from K=8 variance. Neither result would be informative for the thesis mainline, which rests on OWT K=30 and treats LLaDA-SFT as a bounded external-validity probe.

Full decision artefact, including the explicit non-escalation clause and the three follow-on options (richer reference, larger K, full B-sweep), is in `docs/thesis/next_steps/POST_CROSS_BACKBONE_DECISION.md`.

---

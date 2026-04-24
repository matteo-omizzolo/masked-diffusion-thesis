# Phase 3 — Independent Audit of Cross-Backbone Implementation (ProSeCo-LLaDA-SFT)

*Date: 2026-04-22. Auditor: Claude (fresh read). Scope: scientific coherence,
code correctness, reproducibility, and go/no-go verdict for the bounded
cross-backbone replication on kuleshov-group/proseco-llada-sft.*

---

## 1. Artefacts under audit

| Role | Path | New/Modified |
|---|---|---|
| LLaDA-SFT backend | `src/mdm_playground/scheduling/backends/proseco_llada_sft.py` | new |
| Backend-dispatch util | `src/mdm_playground/scheduling/backends/checkpoint_utils.py` | new |
| Backends package exports | `src/mdm_playground/scheduling/backends/__init__.py` | modified |
| Snapshot stager | `scripts/stage_proseco_llada_sft.py` | new |
| Load preflight | `scripts/debug_proseco_llada_sft_load.py` | new |
| Dispatcher Protocol-A runner | `scripts/run_protocol_a_proseco_snapshot.py` | new |
| Phase 2b runner (backend-aware) | `scripts/run_phase2b_proseco_owt.py` | modified |
| Phase 3a runner (backend-aware) | `scripts/run_phase3a_combinatorial.py` | modified |
| HPC job | `hpc/cross_backbone_proseco_llada_sft_bounded.sbatch` | new |
| Plumbing tests | `tests/test_cross_backbone_plumbing.py` | new |
| Reference OWT backend | `src/mdm_playground/scheduling/backends/proseco_owt.py` | (unchanged, comparator) |
| Reference OWT sbatches | `hpc/phase2b_proseco_owt.sbatch`, `hpc/phase3a_combinatorial.sbatch` | (unchanged, comparators) |

---

## 2. Scientific coherence

### 2.1 Is the LLaDA-SFT "corrective-refinement object" the same as ProSeCo-OWT's?

Side-by-side comparison of the inner loop:

| Aspect | ProSeCo-OWT | ProSeCo-LLaDA-SFT | Risk |
|---|---|---|---|
| Forward signature | `_forward(x, t)` passes `timesteps=sigma` | `_forward(x)` — no time conditioning | **Semantic shift** — see below |
| Predictor step (unmask) | MDLM absorbing-state schedule with LogLinear σ(t) | Same LogLinear schedule at mask revelation | Same |
| Corrector iteration | `corrector_times = linspace(1, ε, S+1)`; refine at annealed σ | Plain argmax iteration, no σ | **Semantic shift** |
| Action set | UNMASKED positions | UNMASKED positions | Same |
| Mask-logit gating | `logits[…, mask_id] = -1e9` before softmax | `logits[…, mask_id] = -1e9` before softmax | Same |
| Signals | entropy / inverse_margin / quality_mass_proxy over unmasked | Same three over unmasked | Same |
| NLL scorer | GPT-2 (ref tokenizer = generator tokenizer) | GPT-2 ref, LLaDA decode → re-encode | **Mild shift** — scorer is consistent but the round-trip decode/encode adds tokenisation noise |

**Assessment**: the LLaDA-SFT variant is a *faithful analogue*, not a strict
port. Two design choices diverge from the OWT backend:

1. **No time conditioning inside the backbone**. This matches the LLaDA
   paper: LLaDA is trained to denoise any mask ratio without an explicit
   σ input (the mask ratio is implicit in the input). Passing σ would be
   wrong. So this is **correct LLaDA usage**, not a bug.
2. **No annealed σ in the corrector**. Since LLaDA ignores σ, an annealed
   σ schedule would be a no-op inside the backbone. Replacing the annealed
   schedule with iterated argmax is the natural analogue. This is
   **correct**, but the *character* of the corrective kernel is different:
   ProSeCo-OWT's corrector does Gibbs-like sampling at temperatures
   descending toward the reference distribution; LLaDA's corrector is a
   deterministic fixed-point iteration on argmax. Over `corrector_steps=1`
   this difference collapses to "single argmax refinement" either way,
   which is the budget used in the bounded plan.

**Takeaway**: at `corrector_steps=1`, the two correctors are very similar.
At `corrector_steps>1`, the behaviours diverge materially. The bounded
cross-backbone plan uses `corrector_steps=1` end-to-end, so this caveat is
*nominal* for the planned run but would become sharp for any future
increase of `corrector_steps`.

### 2.2 Is the Δ_t + signal schema compatible with Phase 2b/3a downstream?

The dispatcher `run_protocol_a_proseco_snapshot.py` writes the same
`trajectory_{i}.json` shape as the reference OWT Protocol-A:
`{seed, T, per_t[{t, delta, tcr, f_base, f_branch, n_changed, entropy,
inverse_margin, quality_mass_proxy, unmasked_fraction, n_revisable,
n_masked}]}`. `run_phase2b_proseco_owt.py:load_protocol_a` reads only
`delta` + 4 signal channels, all present; `run_phase3a_combinatorial.py`
reads only `delta`. **Schema-compatible** — downstream analyses
(`analyze_phase2b.py`, `analyze_phase3a.py`) should run unchanged.

### 2.3 Statistical power caveat at K=8

The OWT Phase 2b+3a used `K=30`. The bounded cross-backbone uses `K=8`.
Paired 95 % CI widths scale roughly as `1/√K`, so the bounded run's CIs
are ~**1.94× wider** than OWT's at equal variance. Implications:

- A **null** (paired CI straddles 0) at K=8 is **weak** evidence of
  non-transfer; it is consistent with both "effect is absent" and "effect
  is present but CI is too wide to reject 0".
- A **PASS** (CI strictly positive) at K=8 is stronger evidence — the
  effect transferred at least to the precision we measured.
- A **FAIL** (CI strictly negative) at K=8 would be a clean non-transfer
  signal and would materially change the thesis's external-validity story.

The final results document (if the run completes) must reason in these
three buckets, not in a single "passed/failed" verdict.

### 2.4 Governance: does this belong to the thesis?

`docs/thesis/CANONICAL_RESEARCH_DIRECTION.md` and
`docs/thesis/CURRENT_INDEX.md` both declare cross-backbone **out of scope
for the thesis core**. `PRINCIPLED_NEXT_STEPS_PLAN.md` agrees; the later
`CROSS_BACKBONE_REPLICATION_PLAN.md` (bounded) authorises it
*conditionally*. The canonical document has **not** been updated to
reincorporate cross-backbone.

- **Conservative framing**: run, if at all, as an **appendix probe**, not as
  a thesis-core requirement. The core story (ranker-class negative +
  search-class positive on ProSeCo-OWT) must remain defensible **without**
  this run.
- Either route: the canonical document should be edited minimally to note
  "optional appendix probe on LLaDA-SFT is run and reported; does not
  change the core story's scope".

### 2.5 Environment drift

The sbatch installs `transformers>=4.53`, `huggingface_hub`, `safetensors`,
`einops`, `scipy`, `matplotlib` via `pip --break-system-packages` into the
shared `remdm311` conda env. `CLAUDE.md §8` pins `transformers==4.38.2` for
the MDLM checkpoint. ProSeCo-OWT has also run under whatever `transformers`
is currently installed; Phase 2b+3a ran successfully on job 479941 and
prior. Installing `transformers>=4.53` could shift the shared env for
future MDLM/OWT reruns.

- **Mitigation**: the sbatch's `pip install` is idempotent and will upgrade
  the env irreversibly unless pinned. For a one-shot bounded run this is
  acceptable, but a rollback plan (`pip install 'transformers==4.38.2'`
  after the run) is worth documenting if MDLM is going to be rerun. Not a
  code blocker.

---

## 3. Code correctness

### 3.1 `ProSeCoLLaDASFTGenerator`

- Snapshot validation asserts `config.json`, `configuration_llada.py`,
  `modeling_llada.py` and sharded safetensors. **Correct**.
- `trust_remote_code=True` used — required for LLaDA's custom modeling.
  Local snapshot + `local_files_only=True` avoids any network fetch after
  staging. **Correct**.
- Module-level monkey-patch of `torch.load` to force `weights_only=False`
  — mirrors OWT backend; global side-effect flagged but already standard
  practice in this repo.
- `_compute_neg_nll` decodes with `_gen_tok` (LLaDA), re-encodes with
  `_ref_tok` (GPT-2) for scoring. **Correct**: LLaDA vocab ≠ GPT-2 vocab,
  so pass-through token IDs would be nonsense. The decode/encode is lossy
  but correct at the text level.
- `_run_loop` and `run_with_schedule` follow the same structure as the OWT
  backend. **Correct**.
- Returned `tokens` are int32 NumPy arrays matching the downstream schema.

### 3.2 `detect_proseco_snapshot_backend`

- Detects by presence of `configuration_{proseco,llada}.py` plus matching
  `modeling_*.py`. Raises on ambiguous or unsupported layouts. **Correct**.

### 3.3 Phase 2b / 3a runner changes

- Both accept `--backend {auto, proseco_owt, proseco_llada_sft}` and call
  `detect_proseco_snapshot_backend` when auto. Build logic mirrors OWT
  exactly for the new backend. **Correct**.
- The `run_config.*.json` and `manifest.json` include the backend label
  (lowercased via `backend_label.lower()`). Reproducibility OK.

### 3.4 Dispatcher Protocol-A runner

- Uses the same `estimate_single_step_gain` function as the OWT pipeline
  and writes schema-compatible `trajectory_{i}.json`. **Correct**.
- `run_config.json` records backend + seeds + checkpoint. OK.

### 3.5 Load preflight

- 5 checks: files present → generator instantiates → run_base finite →
  run_branch(T/2) finite → delta finite. Covers the minimum viable
  cross-backbone failure modes. **Adequate**.

### 3.6 HPC sbatch

- 1×A100 + 96G + 20h, single-shard execution (no multi-GPU sharding at
  K=8). Stages 0–3 run sequentially; each stage tees to a dedicated log
  file. Aggregator inline-Python concatenates `*.shard*-of-*.json` into
  flat files, exactly matching OWT's downstream expectations. **Correct**.
- Bounded parameters `B_values=2,4`, `mc_P=24`, `cd_max_attempts=24`,
  `cd_window=8`, `bs_beam_width=4` are all clean reductions from the OWT
  Phase 2b/3a settings. **Correct** and consistent with the bounded plan.
- `set -euo pipefail` guards errors; stage failures would abort before
  wasting downstream budget.

### 3.7 Tests

- Two plumbing tests: backend detection (uses tmp dirs + stub files) and
  surrogate Protocol-A smoke. Runnable locally without checkpoints.
- **No** test exercises the LLaDA backend itself (would require the
  staged snapshot; appropriate to defer to the preflight).

---

## 4. Risks and unknowns

- **R-PHA3-1 (semantic shift)**: if `corrector_steps` is later increased,
  the LLaDA corrector's behaviour diverges from the OWT corrector. The
  bounded plan avoids this by holding `corrector_steps=1`. Acceptable.
- **R-PHA3-2 (statistical power)**: K=8 delivers wider paired CIs; the
  result doc must bucket into PASS / NULL / FAIL with that caveat.
- **R-PHA3-3 (governance)**: canonical direction says cross-backbone is
  out of scope. If run, frame as optional appendix.
- **R-PHA3-4 (env drift)**: `transformers>=4.53` will upgrade the shared
  env; MDLM rerun would need a rollback step.
- **R-PHA3-5 (checkpoint availability)**: the `kuleshov-group/proseco-llada-sft`
  snapshot is hosted externally. Staging depends on HuggingFace Hub
  availability at run time. The stager script catches missing files and
  fails loudly.

None of the above is a code-correctness blocker.

---

## 5. GO / NO-GO verdict

**CONDITIONAL GO**, with two binding conditions:

1. **Scope declaration**: the canonical direction document
   (`docs/thesis/CANONICAL_RESEARCH_DIRECTION.md`) must explicitly declare
   the bounded cross-backbone run as an **optional appendix probe**,
   *before* the run completes, to keep the thesis's primary claims
   defensible without it.
2. **Honest PASS / NULL / FAIL reporting**: the results document, when
   written, must (a) disclose the corrector semantic shift, (b) disclose
   that K=8 CIs are ~1.94× wider than the OWT baseline's K=30 CIs, and
   (c) treat any NULL verdict as "under-powered to reject transfer",
   not as "transfer rejected".

Additional operational prerequisites (not blockers):
- Preflight (Stage 0) must PASS before Protocol A is started. The sbatch
  runs preflight first and will exit on failure via `set -e`.
- Env-drift rollback plan documented elsewhere if MDLM will be rerun.

---

## 6. Summary

| Dimension | Status |
|---|---|
| Scientific coherence (at `corrector_steps=1`) | **OK** |
| Code correctness | **OK** |
| Schema compatibility with Phase 2b/3a downstream | **OK** |
| Reproducibility (manifest + checkpoint hash) | **OK** |
| Statistical power at K=8 | **Reduced** (1.94× wider CIs) |
| Thesis-scope fit (per canonical direction) | **Appendix only** |
| Blocking code issues | **None** |
| Overall | **CONDITIONAL GO as optional appendix** |

---

## 7. Required fixes before HPC submit

None at the code level. One canonical-doc sentence update (appendix framing)
is required before any results doc is written. That is a 1-line edit and
does not block preflight or Stage 1.

# CROSS_BACKBONE_REPLICATION_PLAN

## A. Replication question

**Primary question.**  
Does the core ProSeCo-OWT conclusion survive on a stronger same-family ProSeCo checkpoint:

1. greedy/separable per-step rankers are weak/unreliable under fixed budget, and
2. schedule-aware search recovers meaningful paired gain vs uniform?

**Scope lock.**  
This remains a fixed-predictor, fixed-corrector-kernel, fixed-budget scheduling test. It is **not** a remasking-policy or general self-correction benchmark pivot.

---

## B. Candidate backbone audit

### Candidate inventory (repo + public checkpoint audit)

| Candidate | Source | Public? | Stronger than OWT? | Current code compatibility | Hardware burden | Likely failure modes | Preserves explicit corrective-refinement object? |
|---|---|---:|---:|---|---|---|---|
| `kuleshov-group/proseco-llada-sft` | HuggingFace | Yes | Yes (8B class) | **No (not plug-and-play)**: files are `configuration_llada.py` / `modeling_llada.py`, sharded safetensors, different config/model type | High (8B, bf16, long sequence) | loader mismatch, tokenizer/scoring mismatch, OOM, long runtime | **Partially**: same ProSeCo family claim, but implementation semantics differ from current OWT backend and must be made explicit |
| `kuleshov-group/proseco-owt` | HuggingFace | Yes | No (baseline) | Yes (active mainline) | Known/managed | none new | Yes |

### Evidence used for audit

- HF model search confirms only two public `kuleshov-group/proseco-*` repos (`proseco-owt`, `proseco-llada-sft`).
- `proseco-llada-sft` repo structure differs materially from OWT (`modeling_llada.py`, 4-way sharded safetensors, `model_type=llada`).
- Current thesis backend (`src/mdm_playground/scheduling/backends/proseco_owt.py`) hard-codes OWT snapshot structure and cannot load LLADA snapshots directly.

### Candidate decision

**Chosen target:** `kuleshov-group/proseco-llada-sft` (only public stronger same-family ProSeCo checkpoint).

---

## C. Minimal scientifically meaningful replication design

This replication is intentionally **bounded** and staged:

### Stage 0 (must pass): backend feasibility + semantic sanity

1. Load LLADA snapshot from local staged directory.
2. Run base + one-branch smoke (`T` small) and verify finite `neg_nll`.
3. Verify protocol-A trajectory files can be produced in the exact schema consumed by existing Phase 2b/3a scripts.

If Stage 0 fails, stop and report blocker (no forced long run).

### Stage 1 (bounded core test): reduced Phase 2b + reduced Phase 3a

- **Protocol A generation:** only what is needed for bounded K seeds.
- **Phase 2b bounded grid:** `B ∈ {2,4}`, reduced `K`, reduced MC samples.
- **Phase 3a bounded grid:** same `B ∈ {2,4}`, reduced CD attempts + beam width.

### Why this is the smallest valid test

It still tests the key contrast (ranker-class vs search-class) on paired seeds and identical `G(S)` definition, while avoiding an immediate full `K=30`, full-budget, full-MC campaign on an 8B backbone.

### Transfer interpretation labels

- **Transfer supported (bounded):**
  1. at least one search method has paired CI lower bound `> 0` vs uniform at tested B, and
  2. ranker-class remains weak/mixed relative to search.
- **Transfer unclear:**
  - CIs overlap 0 broadly, or effects are unstable across tested B (power/runtime-limited pilot).
- **Transfer not supported (bounded):**
  - search methods fail to beat uniform while ranker-class is not clearly weaker.

No universality claim from this bounded run.

---

## D. Statistical design

- **Paired seeds:** same seed set for uniform, rankers, and search.
- **Same objective:** keep `G(S)=F(y^S)-F(y_base)` using current `F=neg_nll` pipeline for internal comparability with existing mainline machinery.
- **CIs:** paired bootstrap (same convention as existing analysis scripts).
- **Primary effects:**
  1. paired mean `Δ_method = G_method - G_uniform`,
  2. search-vs-ranker contrast at same `B`,
  3. (where available) gap-to-oracle style ratio on bounded MC settings.
- **Plots/stats (minimum):**
  1. paired gain vs uniform with CIs by B/method,
  2. backbone comparison plot (OWT mainline vs LLADA bounded cells),
  3. ranker vs search contrast table/plot.
- **PASS / FAIL / inconclusive:**
  - PASS if CI lower bound `> 0`,
  - FAIL if CI upper bound `< 0`,
  - otherwise inconclusive.

---

## E. Go / no-go decision

**Decision: GO (bounded, conditional).**

Proceed now with `proseco-llada-sft`, but only through gated bounded replication:

1. implement minimal LLADA backend support and protocol-A runner,
2. run smoke/preflight,
3. launch bounded HPC run only if smoke passes.

**Explicit caveat.**  
This is a transfer probe, not a broad external-validity claim. If LLADA semantics diverge materially from the OWT corrective object during preflight, report and stop rather than force uninterpretable results.


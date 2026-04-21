# Prompt for Claude Code (Sonnet) — Signal-Aligned MDLM-conf Re-run

*Copy everything between the `===BEGIN PROMPT===` and `===END PROMPT===` markers
into Claude Code. Written for Sonnet, so it is explicit about file paths,
expected state, validation, and failure modes.*

---

===BEGIN PROMPT===

# Task: Fix the MDLM-conf signal/action-set mismatch and launch the signal-aligned Phase 1 re-run

You are working in the repo at the root of this workspace (an MSc thesis on
signal-adaptive corrector scheduling for masked diffusion language models,
Bocconi, supervised by Prof. Giacomo Zanella). The `CLAUDE.md` at the repo
root has full project context; read it first. The thesis docs live under
`docs/thesis/`; the canonical entry point is `docs/thesis/CURRENT_INDEX.md`.

## 0. Required reading before you touch any code

Read these files in order and summarise back to me in two or three sentences
what the current blocker is before editing anything:

1. `docs/thesis/CURRENT_INDEX.md`
2. `docs/thesis/experiments/RESULTS_STATUS.md`
3. `docs/thesis/experiments/PROSECO_PREFLIGHT_STATUS.md` (especially §2.2 M2,
   §3 Option (b), and §5–§6)
4. `docs/thesis/experiments/HPC_NEXT_RUN_PLAN.md`
5. `src/mdm_playground/scheduling/backends/mdlm_conf.py`
6. `src/mdm_playground/scheduling/signals.py`
7. `scripts/run_phase1_mdlm_conf.py`
8. `scripts/debug_mdlm_conf_load.py`
9. `results/phase1_mdlm_conf/summary.json`

Do not modify anything during this reading pass.

## 1. The precise problem

In `src/mdm_playground/scheduling/backends/mdlm_conf.py`:

- `_apply_corrector` resamples only the `top_k = 20` lowest-confidence
  **masked** positions out of (up to ~10³) total masked positions.
- `_extract_signals` currently aggregates (entropy, inverse_margin,
  quality_mass_proxy) over **all** masked positions.

Consequence in job 478962 (`results/phase1_mdlm_conf/summary.json`):
`eps_rms ≈ 0.222` for every signal but `spearman_mean ≈ 0.03` across
entropy / inverse_margin / quality_mass_proxy — the proxies do not rank-predict
Δ_t because they summarise a distribution the corrector does not sample from.

Theorem A's ε is defined as the RMS residual between a proxy ψ(s_t) and the
per-step gain Δ_t the corrector actually produces. The proxy must be a function
of the same positions the corrector acts on. Right now it is not. That is the
single methodological bug this task fixes.

## 2. The code change (Option (b) from PROSECO_PREFLIGHT_STATUS.md §3)

**File:** `src/mdm_playground/scheduling/backends/mdlm_conf.py`
**Method:** `_extract_signals`
**Around lines:** 236–273 (re-read for exact current content; do not trust line
numbers blindly)

### 2.1 Required behaviour

Compute `entropy`, `inverse_margin`, `quality_mass_proxy` on **the same set of
positions the corrector resamples**: the `top_k = self.top_k` lowest-confidence
masked positions (confidence = `max_v p_x0[l, v]`). Leave `unmasked_fraction`
and `n_masked` unchanged (those are global context scalars).

If `n_masked == 0`, keep the existing zero-return branch.

If `0 < n_masked <= self.top_k`, compute the signals over all masked positions
(that is the degenerate edge case where the action set equals the full masked
set — behaviourally identical to today's code).

If `n_masked > self.top_k`, compute confidence over all masked positions, pick
the `self.top_k` lowest-confidence, compute the three signals only on those.

### 2.2 Suggested patch (adapt to match current style)

```python
def _extract_signals(
    self, x: torch.Tensor, p_x0_probs: torch.Tensor
) -> Dict[str, float]:
    """Signals over the corrector's action set — the top_k lowest-confidence
    masked positions (NOT all masked positions).

    This is the Option (b) fix from PROSECO_PREFLIGHT_STATUS.md §3: the
    proxy ψ(s_t) must summarise the distribution the corrector actually
    samples from, otherwise Theorem A's ε is not the relevant residual.
    """
    masked_idx = (x[0] == self.mask_id).nonzero(as_tuple=True)[0]  # (n_masked,)
    n_masked = int(masked_idx.numel())
    D = x.shape[1]
    unmasked_fraction = float((D - n_masked) / D)

    if n_masked == 0:
        return {
            "entropy": 0.0,
            "inverse_margin": 0.0,
            "quality_mass_proxy": 0.0,
            "unmasked_fraction": unmasked_fraction,
            "n_masked": 0,
            "n_action": 0,
        }

    # Action set: top_k lowest-confidence masked positions.
    p_masked_all = p_x0_probs[0, masked_idx].float()         # (n_masked, V)
    confidence = p_masked_all.max(-1).values                 # (n_masked,)
    k = min(self.top_k, n_masked)
    _, low_conf_local = confidence.topk(k, largest=False)    # (k,)
    p_m = p_masked_all[low_conf_local]                       # (k, V)

    H = -(p_m * (p_m + 1e-12).log()).sum(-1).mean().item()
    top2 = p_m.topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).mean().item()
    p_argmax = p_m.max(-1).values.mean().item()

    return {
        "entropy": float(H),
        "inverse_margin": float(1.0 - margin),
        "quality_mass_proxy": float(1.0 - p_argmax),
        "unmasked_fraction": unmasked_fraction,
        "n_masked": n_masked,
        "n_action": int(k),
    }
```

Notes:

- Add `n_action` to the returned dict (same type as `n_masked`). It is the
  size of the action set at that step. This is a new field — downstream
  consumers must handle its absence (for backward compatibility with the prior
  pilot data on HPC).
- Do NOT change `_apply_corrector`. The corrector logic is already correct.
- Do NOT change `scheduling/signals.py`. That file is the generic signal
  library used by the analysis / non-backend code; we only want the
  backend-internal extractor to be action-set-aware. Changing the generic
  library would silently alter other runs.

### 2.3 Downstream check — is `n_action` consumed anywhere?

Grep for `n_masked` and any new reads of `n_action` across:

- `scripts/run_phase1_mdlm_conf.py`
- `scripts/analyze_phase1.py`
- `src/mdm_playground/scheduling/*.py`

If `n_action` is written but no reader uses it, that is fine — the analysis
script only needs the three signal scalars to produce ε / Spearman. We are
adding `n_action` for provenance so that future-you can confirm (from a JSON
file alone) that the run used the signal-aligned extractor.

If `n_masked` is read by the analysis script, leave that path unchanged.
Do **not** remove `n_masked`; keep it for compatibility.

## 3. Local validation (CPU, fast)

Before touching HPC, confirm the change works end-to-end on CPU.

### 3.1 CPU preflight

```bash
python scripts/debug_mdlm_conf_load.py \
    --checkpoint /path/to/mdlm.ckpt \
    --device cpu --T 4 --top_k 5
```

If no local checkpoint is available, skip this check and move on to 3.2 —
the surrogate run is the stronger validation anyway.

### 3.2 Surrogate run (no real checkpoint needed)

The repo ships a surrogate backend used for CPU pipeline validation.
Run the existing surrogate sanity entrypoint used by
`results/phase1_mdlm_conf_surrogate_sanity/`. If an explicit script is not
obvious, inspect `src/mdm_playground/scheduling/surrogate.py` and
`scripts/run_phase1_mdlm_conf.py --help` to find the surrogate flag.

Goal: produce a surrogate `summary.json` with all three signals finite and
`n_action` populated with integer values in [1, top_k].

### 3.3 Unit-level sanity (a quick ad-hoc Python check)

Open a Python REPL and construct a fake `x` (1D tensor of length 32 with 12
mask ids scattered) and a fake `p_x0_probs` of shape `(1, 32, V=50)`. Call
`_extract_signals` and confirm:

- With `top_k = 5` and 12 masked positions: `n_masked == 12`, `n_action == 5`.
- With `top_k = 20` and 12 masked positions: `n_masked == 12`,
  `n_action == 12` (degenerate edge case — action set = full masked set).
- With no masked positions: all three signals exactly 0.0 and
  `n_action == 0`.

Report the three signal values in each case. Sanity: entropy should be in
[0, log V] ≈ [0, 3.9] with V=50, inverse_margin in [0, 1],
quality_mass_proxy in [0, 1].

## 4. Archive the old results directory (HPC side)

Before resubmission, the prior pilot data at `results/phase1_mdlm_conf/`
on the HPC cluster must be renamed so the new run does not clobber it
silently.

From the laptop (you do not have HPC DNS access from inside this environment
if you are running in a sandbox, so hand these commands to the user):

```bash
ssh 3316152@slogin.hpc.unibocconi.it \
    "cd ~/mdm/masked-diffusion-thesis && \
     mv results/phase1_mdlm_conf results/phase1_mdlm_conf_pilot1_20260417"
```

If `ssh` is unreachable from wherever this Claude Code instance is running,
print the command verbatim and stop — the user will run it.

## 5. Push and submit

```bash
bash hpc/push.sh
ssh 3316152@slogin.hpc.unibocconi.it \
    "cd ~/mdm/masked-diffusion-thesis && sbatch hpc/phase1_mdlm_conf.sbatch"
```

`hpc/pull.sh` is broken on macOS — use `scp` or `ssh 3316152@... "python3 -c '...'"`
to pull results. Do not attempt to fix `pull.sh`; that is explicitly out of
scope.

Note the SLURM job ID printed by `sbatch`.

## 6. Monitor

Poll `squeue` once the job is submitted:

```bash
ssh 3316152@slogin.hpc.unibocconi.it "squeue -u 3316152"
```

Tail the stdout once the job starts:

```bash
ssh 3316152@slogin.hpc.unibocconi.it \
    "tail -200 ~/mdm/masked-diffusion-thesis/out/phase1_mdlm_conf_<JOBID>.out"
```

Wait until the job reports COMPLETED (not FAILED, not TIMEOUT) before moving on.

## 7. Pull and inspect

```bash
scp -r 3316152@slogin.hpc.unibocconi.it:~/mdm/masked-diffusion-thesis/results/phase1_mdlm_conf \
    results/phase1_mdlm_conf_signal_aligned
cat results/phase1_mdlm_conf_signal_aligned/summary.json
```

### 7.1 Success criteria (from PROSECO_PREFLIGHT_STATUS.md §6)

Check all of these in `summary.json`:

- `n_positive_delta_steps > 0` and `peak_mean_delta >= 0.05`
  (MDLM-conf still produces Δ > 0 after the extractor change).
- At least one of `entropy`, `inverse_margin`, `quality_mass_proxy` has
  `|calibration.<signal>.spearman_mean| >= 0.15` with
  `spearman_std` not catastrophic (< 0.6 roughly).
- `eps_rms` for all three signals is finite and ≤ 0.5.
- `eta_by_B.*.eta_95` and `gamma.gamma_95` finite and > 0.
- `theorem_A_bound_check[B].bound_useful == true` for at least one B; if
  still false, `bound_2Be_2eta / G_oracle_estimate` ratio should be ≤ 5×.
- At least one of the `top_B` policies beats `uniform` at B=8 or B=16 (not
  only at B=4, as in the prior pilot).

Also spot-check `results/phase1_mdlm_conf_signal_aligned/protocol_a/trajectory_0.json`
to confirm each per-step record has `n_action` populated with a positive
integer in `[1, top_k]`.

### 7.2 Interpret

- **All success criteria met:** proceed to graduate to N=50 by uncommenting
  the FULL block in `hpc/phase1_mdlm_conf.sbatch` and re-submitting. Write a
  one-page summary of the signal-aligned pilot results into
  `docs/thesis/experiments/RESULTS_STATUS.md` under a new §9 "Signal-aligned
  MDLM-conf pilot (job <JOBID>)", and update `results_status.md`'s headline
  in §1. Do NOT overwrite the prior §5 ProSeCo pilot record — the negative
  result is part of the story.
- **Δ_t regressed to all zeros or all negative:** the extractor change did
  not preserve the corrector's behaviour (impossible if you followed §2
  exactly — `_apply_corrector` was untouched). Re-inspect the diff, re-run
  §3 surrogate sanity, do not resubmit.
- **Δ_t unchanged qualitatively but Spearman still ≈ 0:** this is itself a
  publishable finding — no candidate aggregate signal predicts Δ_t even with
  matched action sets. In that case, write it up in §9 and switch to Option
  (c) (proseco-owt checkpoint) as documented in PROSECO_PREFLIGHT_STATUS.md
  §3. Do not attempt Option (c) in the same run.

## 8. Update the thesis docs (whatever the outcome)

After you have the new `summary.json`:

1. Append a section §9 "Signal-aligned MDLM-conf pilot (job <JOBID>, YYYY-MM-DD)"
   to `docs/thesis/experiments/RESULTS_STATUS.md` with the same schema as
   §2 / §5 (parameters, KEY RESULTS table, scientific interpretation).
2. Update `docs/thesis/CURRENT_INDEX.md` §8 to reference
   `results/phase1_mdlm_conf_signal_aligned/summary.json` and the new job ID.
3. Update `docs/thesis/maintenance/CLEANUP_LOG.md` with a short entry dated
   today describing the signal-extractor fix, the new run, and the outcome.
4. Do NOT overwrite `PROSECO_PREFLIGHT_STATUS.md`; it is an artifact of the
   2026-04-18 cleanup pass. If you want to record a second preflight, create
   `docs/thesis/experiments/PROSECO_PREFLIGHT_STATUS_V2.md`.

## 9. What you are forbidden from doing

Do not:

- Change `_apply_corrector` in `backends/mdlm_conf.py`.
- Change `scheduling/signals.py` (the generic signal library).
- Delete any files. If something looks stale, move it under `archive/` and
  log the move in `archive/ARCHIVE_MANIFEST.md` with a new row.
- Modify `backends/proseco.py` or `backends/mdlm.py` in this pass — those are
  separate workstreams (Options (c) / (d) in PROSECO_PREFLIGHT_STATUS.md).
- Touch `external/` (rsync'd upstream code). HPC environment fixes there are
  already applied and documented in `CLAUDE.md`.
- Run the FULL (N=50) block in `hpc/phase1_mdlm_conf.sbatch` until the N=20
  signal-aligned pilot meets §7.1.
- Run `git commit` or `git push` unless I explicitly ask you to.
- Attempt to fix `hpc/pull.sh`; use `scp` / direct ssh instead.
- Use `rm -rf`, `git reset --hard`, or any destructive git command.

## 10. Deliverable at the end of this task

A single chat message from you containing:

1. The diff you applied to `backends/mdlm_conf.py` (fenced unified-diff).
2. The three local validation outputs (§3.1 / §3.2 / §3.3), with actual
   numbers not just "passed".
3. The SLURM job ID of the submitted run, and the command that submitted it.
4. Once the job completes: the full content of
   `results/phase1_mdlm_conf_signal_aligned/summary.json`.
5. A PASS/FAIL line per bullet in §7.1.
6. The paths of the doc updates you made in §8.
7. A one-sentence verdict: "proceed to N=50 full run", "switch to proseco-owt
   (Option (c))", or "investigate regression before any further submission".

Keep commentary short. This is an operational task, not an essay.

===END PROMPT===

---

## Notes for you (Matteo) before handing this to Sonnet

- **Before pasting:** make sure Claude Code's working directory is the repo
  root (`/Users/.../masked-diffusion-thesis/` on your laptop).
- **If Sonnet runs this in a sandbox without HPC DNS** (e.g. some cowork
  environments), it will print the ssh/sbatch commands for you to run
  manually — that is the intended fallback, not a failure.
- **Time budget:** 30 min for §1–§3 (local change + validation), 10 min for
  §4–§5 (push + submit), 2–4 h wait for the job, 20 min for §7–§8 (inspect
  and document). Total wall-time ~4 h, active time ~1 h.
- **If you want to run the change locally without any HPC access at all:**
  ask Sonnet to stop at §3.3 and report. You can review the diff and run
  steps §4–§7 yourself later.

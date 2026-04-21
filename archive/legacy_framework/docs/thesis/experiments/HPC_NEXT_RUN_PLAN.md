# HPC Next-Run Plan — Phase 1 MDLM-conf Pilot

*Last updated: 2026-04-18.*
*Scope: the next HPC submission after the ProSeCo pilot (job 478929) produced
Δ_t ≡ 0. Adopts Option 2 (MDLM-conf) from `RESULTS_STATUS.md`.*

## 1. Entrypoints

- **Python script:** `/sessions/great-eloquent-rubin/mnt/masked-diffusion-thesis/scripts/run_phase1_mdlm_conf.py`
- **Backend module:** `/sessions/great-eloquent-rubin/mnt/masked-diffusion-thesis/src/mdm_playground/scheduling/backends/mdlm_conf.py`
- **Sbatch script:** `/sessions/great-eloquent-rubin/mnt/masked-diffusion-thesis/hpc/phase1_mdlm_conf.sbatch`
  - partition `stud`, QOS `stud`, 1 × A100, 8 CPUs, 48 GB RAM, 8 h wall-time
  - installs the package and dependencies (`pip install -e .`) under `remdm311`
  - srun dispatches `scripts/run_phase1_mdlm_conf.py` with the pilot parameters
- **Preflight script:** `/sessions/great-eloquent-rubin/mnt/masked-diffusion-thesis/scripts/debug_mdlm_conf_load.py`

## 2. Backend and checkpoint

- Backend: MDLM-conf corrector — at step t, resample only the `top_k=20`
  lowest-confidence **masked** positions from p_x0, leaving all other masked
  positions and every committed token unchanged.
- Fixes Bug #1 (`_extract_signals` now uses `(x == mask_id)`) and Bug #2
  (partial resample rather than one-shot resample of the entire mask set).
- Checkpoint: `/home/3316152/mdm/checkpoints/mdlm.ckpt`
  (≈ 2.5 GB, previously uploaded; confirmed readable by jobs 478600 and 478929).

## 3. Run parameters (pilot)

`T=64, N=20, M=15, P=120, B ∈ {4, 8, 16}, corrector_steps` implicit
(`top_k=20`), `seed=42`. Output directory: `results/phase1_mdlm_conf/`.
Estimated runtime: 3–5 h on a single A100.

Full arg line (from `hpc/phase1_mdlm_conf.sbatch`):

```
srun python -u scripts/run_phase1_mdlm_conf.py \
    --T 64 --N 20 --M 15 --P 120 \
    --B_values 4,8,16 --seed 42 \
    --checkpoint /home/3316152/mdm/checkpoints/mdlm.ckpt \
    --top_k 20 \
    --out_dir results/phase1_mdlm_conf
```

## 4. Expected output layout

```
results/phase1_mdlm_conf/
├── run_config.json
├── summary.json
├── protocol_a/
│   └── trajectory_0.json … trajectory_19.json
├── protocol_b/
│   ├── schedule_0.json … schedule_44.json
│   └── pairs.json
└── policy_comparison/
    └── policy_comparison.json
```

`run_config.json` captures backend, T, N, M, P, B_values, seed, checkpoint,
`top_k`, out_dir, timestamp.

## 5. Preflight checks (must all pass before sbatch)

1. **Backend loads on CPU.** Run
   `python scripts/debug_mdlm_conf_load.py --checkpoint /home/3316152/mdm/checkpoints/mdlm.ckpt --device cpu --T 4 --top_k 5`
   on the login node. The five checks (config readable; generator instantiates;
   `run_base` signals non-degenerate over masked positions; `run_branch`
   corrector applies and changes some tokens; `neg_nll` finite and negative)
   must all PASS. Exit code 0 = safe to submit.
2. **Non-trivial Δ_t at at least one sanity step.** Confirm the sanity run
   (inside preflight or a small `--T 16 --N 5 --M 3 --P 10` surrogate-off run)
   reports at least one step with Δ > 0 in `run_branch`. Prior experience:
   MDLM-conf produced non-zero Δ_t at mid-trajectory steps; zero everywhere
   would indicate a regression.
3. **Masked-position signals non-zero.** In the preflight log, verify that at
   step t with partial unmasking the entropy / inverse-margin / quality-mass
   fields in `per_step_signals` are > 0. This is the Bug #1 fix regression
   test — if any signal collapses to 0 across all steps, do not submit.
4. **Output directory hygiene.** `results/phase1_mdlm_conf/` already contains
   data from a 2026-04-17 14:16 run (see `RESULTS_STATUS.md` §8). Before
   resubmission, rename or archive the old directory so the new run's
   `run_config.json` and `summary.json` are not clobbered silently.
5. **`run_config.json` captures exact backend config.** After run start, the
   first artefact written should be `results/phase1_mdlm_conf/run_config.json`
   with `backend: "mdlm_conf"`, `top_k: 20`, and the correct checkpoint path.

## 6. Submission command

From the laptop (macOS), with the repo at
`/sessions/great-eloquent-rubin/mnt/masked-diffusion-thesis/`:

```
bash hpc/push.sh
ssh 3316152@slogin.hpc.unibocconi.it "cd ~/mdm/masked-diffusion-thesis && sbatch hpc/phase1_mdlm_conf.sbatch"
```

`hpc/push.sh` rsyncs the repo and excludes checkpoints, `.venv`, and `results/`.
`hpc/pull.sh` uses GNU rsync flags and is broken on macOS — do not use it;
fetch results by direct `ssh` + `scp` or by reading the JSON files via
`ssh 3316152@slogin.hpc.unibocconi.it "python3 -c '...'"`.

## 7. Success criteria for the pilot

- Protocol A: `n_positive_delta_steps > 0` in `summary.json`
  (i.e. there exists at least one t with mean Δ_t > 0 across the 20
  trajectories).
- Signals: `calibration.entropy.eps_rms < 1.0` and
  `calibration.entropy.spearman_mean` is a finite, non-zero number — similarly
  for `inverse_margin` and `quality_mass_proxy`.
- Protocol B: `eta_by_B.*.eta_95` and `gamma.gamma_95` are finite and strictly
  positive (residuals A(S) vs G(S) actually vary across schedules).
- `summary.json.peak_mean_delta > 0` and `t_first_positive_delta < T`.
- `theorem_A_bound_check[B].bound_useful` is true for at least one B
  (non-vacuous Theorem A). This is aspirational; if false but the
  bound/oracle ratio is small (≲ 5×), it still calibrates ε and η_B
  meaningfully for the thesis.

## 8. Post-run inspection

- `tail -200 out/phase1_mdlm_conf_<JOB_ID>.out` — the KEY RESULTS block is
  printed at the end by `run_phase1_mdlm_conf.py`. Expect lines beginning with
  `Steps with Δ_t > 0`, `ε (entropy, RMS)`, `Spearman(H, Δ)`,
  `γ (95th pct)`, `B=4: 2Bε+2η = ...`, and a policy comparison block.
- `results/phase1_mdlm_conf/summary.json` — check each section in §7.
- `results/phase1_mdlm_conf/protocol_a/trajectory_0.json` — spot-check the
  per-step records: `delta`, `tcr`, `entropy`, `inverse_margin`,
  `quality_mass_proxy`, `unmasked_fraction`, `n_masked` all populated.
- Run `scripts/analyze_phase1.py --results_dir results/phase1_mdlm_conf
  --out_dir figures/phase1_mdlm_conf` locally after pulling the JSON to
  regenerate: `calibration_scatter.png`, `delta_vs_t.png`, `eta_vs_B.png`,
  `pairwise_xi_hist.png`, `theorem_A_budget.png`, `tcr_vs_delta.png`.
- If the pilot succeeds, promote to FULL by uncommenting the FULL block in
  `hpc/phase1_mdlm_conf.sbatch` (N=50, M=30, P=300, wall-time 12 h,
  `--out_dir results/phase1_mdlm_conf_full`).

# Token-level Feature Extraction Plan

Generated: 2026-05-12T05:33:58.137079+00:00
Script: scripts/extract_tokenlevel_features_proseco.py
Backend: ProSeCo-OWT (mdm_playground.scheduling.backends.proseco_owt)
Checkpoint: /home/3316152/mdm/checkpoints/proseco_owt
Device: cuda
Seeds: [42, 43]
T (predictor steps): 64
Debug: False
Synthetic: False

## Code path identified

- ProSeCoOWTGenerator._forward(x, t) returns p_x0 (B, L, V) with mask token zeroed.
- The corrector action set R_t in ProSeCo-OWT is the UNMASKED positions
  (x != mask_id). The corrector argmaxes p_x0 and overwrites R_t.
- Base trajectory (no corrector at any t) is reproduced step-by-step.

## Features per step

T0 aggregate (replication of Phase 1):
- entropy, inverse_margin, quality_mass_proxy, unmasked_fraction,
  n_revisable, n_masked.

T1 masked-position uncertainty shape:
- mean/std/quantiles of entropy, top1, margin over MASKED positions;
- frac entropy > thresholds (1.0, 2.0, 3.0);
- frac margin < thresholds (0.05, 0.1, 0.2).

T2 revisable-set uncertainty shape: same statistics over R_t.

T3 concentration: gini, top-5%% / top-10%% entropy mass, n_pos for 50%% mass,
max entropy, max/mean entropy.

T4 per-position arrays for downstream pair-overlap computation:
- entropy_per_pos (T, L) float32, margin_per_pos, top1_per_pos,
  revisable_mask_per_step (T, L) bool.

T5 corrector-action intensity: UNAVAILABLE. ProSeCo-OWT corrector is
deterministic (argmax of p_x0 on R_t), so 'revision distribution' is a
delta. Documented and not extracted.

## Strict leakage exclusions

- No corrector branch is run during extraction. Only the base trajectory.
- No f_branch, n_changed, tcr, post-correction state read.
- No target values written into the feature file.

## RNG / device caveat

Local CPU / MPS extraction will NOT reproduce the HPC A100 base
trajectory that produced the canonical Phase 1 / Gate 3a artifacts.
Predictor-step stochasticity (torch.multinomial, torch.bernoulli) is
device-dependent. For the pilot to be aligned with existing xi_raw.json,
extraction must run on the same device family as Gate 3a (CUDA on
A100). The debug / synthetic runs verify only the code path and feature
arithmetic; the canonical pilot must run on HPC.

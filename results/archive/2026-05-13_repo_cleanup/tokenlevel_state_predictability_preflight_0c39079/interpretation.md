# Token-level Pair-Level State Predictability Audit

## Question

Do token-level / model-internal pre-correction features (masked-position uncertainty shape, revisable-set uncertainty shape, concentration, pair-level overlaps) improve held-out prediction of pair-level corrector-timing targets (xi and G_pair) beyond pair geometry + A_pair?

## Data and alignment

- Pair rows used: 146
- Seeds: 2
- RNG alignment with canonical Phase 1 trajectory_*.json: `True`
- Max T0 deviations (extracted vs canonical): {'entropy': 9.5367431640625e-07, 'inverse_margin': 1.2665987014770508e-07, 'quality_mass_proxy': 8.195638656616211e-08, 'unmasked_fraction': 0.0, 'n_revisable': 0.0, 'n_masked': 0.0}

## Predictors

- P0  pair geometry only.
- P1  pair geometry + A_pair  (strong baseline).
- P2  P1 + T0 aggregate (replication block).
- P3  P2 + T1 masked-position uncertainty shape.
- P4  P3 + T2 revisable-set uncertainty shape.
- P5  P4 + T3 concentration features.
- P6  P5 + T4 pair overlap / similarity features.

## Held-out MSE — target = xi

| Predictor | MSE | Spearman |
|---|---:|---:|
| `P0_geom` | 0.081479 | -0.558 |
| `P1_geom_apair` | 0.030934 | 0.524 |
| `P2_plus_T0` | 0.031289 | 0.466 |
| `P3_plus_T1` | 0.031253 | 0.535 |
| `P4_plus_T2` | 0.031901 | 0.543 |
| `P5_plus_T3` | 0.035018 | 0.483 |
| `P6_plus_T4_overlaps` | 0.037326 | 0.399 |

## MSE-reduction comparisons (xi)

| comparison | mean | 95% CI | %% seeds improved |
|---|---:|---|---:|
| `P1_geom_apair_vs_P0_geom` | +0.050545 | [+0.044298, +0.056793] | 1.00 |
| `P2_plus_T0_vs_P1_geom_apair` | -0.000356 | [-0.000374, -0.000337] | 0.00 |
| `P3_plus_T1_vs_P1_geom_apair` | -0.000319 | [-0.009583, +0.008944] | 0.50 |
| `P4_plus_T2_vs_P1_geom_apair` | -0.000967 | [-0.013853, +0.011919] | 0.50 |
| `P5_plus_T3_vs_P1_geom_apair` | -0.004084 | [-0.025100, +0.016933] | 0.50 |
| `P6_plus_T4_overlaps_vs_P1_geom_apair` | -0.006392 | [-0.029742, +0.016959] | 0.50 |
| `P6_plus_T4_overlaps_vs_P5_plus_T3` | -0.002308 | [-0.004642, +0.000026] | 0.50 |

## Held-out MSE — target = G_pair

| Predictor | MSE | Spearman |
|---|---:|---:|
| `P0_geom` | 0.018425 | 0.555 |
| `P1_geom_apair` | 0.030110 | 0.345 |
| `P2_plus_T0` | 0.030319 | 0.339 |
| `P3_plus_T1` | 0.030841 | 0.322 |
| `P4_plus_T2` | 0.032058 | 0.287 |
| `P5_plus_T3` | 0.035972 | 0.221 |
| `P6_plus_T4_overlaps` | 0.038369 | 0.233 |

## MSE-reduction comparisons (G_pair)

| comparison | mean | 95% CI | %% seeds improved |
|---|---:|---|---:|
| `P1_geom_apair_vs_P0_geom` | -0.011685 | [-0.023270, -0.000100] | 0.00 |
| `P2_plus_T0_vs_P1_geom_apair` | -0.000209 | [-0.000367, -0.000050] | 0.00 |
| `P3_plus_T1_vs_P1_geom_apair` | -0.000731 | [-0.010094, +0.008632] | 0.50 |
| `P4_plus_T2_vs_P1_geom_apair` | -0.001947 | [-0.016089, +0.012194] | 0.50 |
| `P5_plus_T3_vs_P1_geom_apair` | -0.005861 | [-0.029239, +0.017516] | 0.50 |
| `P6_plus_T4_overlaps_vs_P1_geom_apair` | -0.008259 | [-0.033861, +0.017344] | 0.50 |
| `P6_plus_T4_overlaps_vs_P5_plus_T3` | -0.002397 | [-0.004622, -0.000172] | 0.00 |

## Verdict (pre-registered expansion gate)

- Label: **tokenlevel_not_supported**

Pre-registered expansion gate. tokenlevel_supported = expand to K=30 (HPC). tokenlevel_weak = expand cautiously after feature importance review. tokenlevel_not_supported = stop and write the ProSeCo-OWT case study.

## Limitations

- ProSeCo-OWT only. The framework is corrector-agnostic; this empirical verdict is not.
- 30 seeds in canonical xi_raw; pilot subsets are smaller and the bootstrap CI reflects that.
- Only ridge / linear class tested. Tree models and kernel methods untested locally.
- T5 corrector-action intensity unavailable in ProSeCo-OWT without modelling the deterministic argmax corrector at additional steps.

## Decision

Do not expand. Under the tested model class and the tested token-level / revisable-set / overlap feature families, correction timing on ProSeCo-OWT is captured by temporal geometry plus A_pair. Synthesise the ProSeCo-OWT case study as geometry / marginal-redundancy driven, with explicit scope to this backbone and feature class.

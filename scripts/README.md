# Scripts

Scripts are now organized by research role. Canonical result folders were not
moved; script paths were updated instead.

## Active Layout

```text
scripts/
  proseco/
    reproduction/
    interactions/
    landscape/
    state_predictability/
    saturation/
    corrector_strength/
  backend_validation/
    informed_correctors/
    prism/
  legacy/
  archive/
```

## ProSeCo-OWT Case Study

| Family | Path | Status |
|---|---|---|
| Reproduction / Phase 2b / Phase 3a | `scripts/proseco/reproduction/` | complete canonical/provenance |
| Pair interactions | `scripts/proseco/interactions/` | complete canonical |
| Landscape and neighborhood | `scripts/proseco/landscape/` | complete canonical |
| State and token-level predictability | `scripts/proseco/state_predictability/` | complete canonical |
| Saturation | `scripts/proseco/saturation/` | complete canonical |
| Corrector strength | `scripts/proseco/corrector_strength/` | complete canonical |

The corrector-strength filenames still contain `preflight` because they are
embedded in prior result provenance; the K=30 output is canonical.

## Backend Validation

`scripts/backend_validation/informed_correctors/` contains the no-author-response
preparation path:

- `check_text8_training_feasibility.py`
- `hollow_text8_stage1_config.py`

These utilities do not launch long training. They support Stage 0 environment
smoke and Stage 1 tiny training-loop checks.

## Validation

Run all tests from repo root:

```bash
pytest -q
```

Focused ProSeCo script tests live under `tests/proseco/`.

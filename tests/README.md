# Tests

Tests are grouped by research family.

```text
tests/
  proseco/
    core/
    reproduction/
    interactions/
    landscape/
    state_predictability/
    saturation/
    corrector_strength/
  backend_validation/
  archive/
```

Run the full suite from repo root:

```bash
pytest -q
```

The ProSeCo tests are intended to remain lightweight and mostly CPU-only. Tests
that require a real staged checkpoint skip unless `PROSECO_OWT_CHECKPOINT` is
set.

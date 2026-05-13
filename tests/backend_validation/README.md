# tests/backend_validation/

Reserved for tests that exercise backend-validation scaffolding (currently
informed-correctors/Text8). No tests are present yet because the active
scaffolding under
`scripts/backend_validation/informed_correctors/` and
`hpc/backend_validation/informed_correctors/` is a thin set of utilities and
sbatch wrappers; nontrivial logic that warrants a test will appear once Stage 0
and Stage 1 surface real gaps.

When adding tests here:

- Mirror the script layout, e.g.
  `tests/backend_validation/informed_correctors/test_check_text8_training_feasibility.py`.
- Keep tests CPU-only and offline; skip cleanly when JAX or upstream data are
  unavailable.
- Do not invoke real training, downloads, or sbatch jobs from unit tests.

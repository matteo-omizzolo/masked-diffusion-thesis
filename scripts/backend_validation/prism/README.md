# scripts/backend_validation/prism/

Reserved subtree for PRISM LLaDA backend-validation scaffolding. **No scripts
are currently active here.**

PRISM LLaDA is **not pursued** for this thesis's corrector-timing validation
gate. The reasoning is captured in
`docs/08_backend_feasibility_audit.md`:

- PRISM's correction lever is learned token-quality remasking, trained jointly
  with the unmasking objective. Timing is not a separable axis without
  retraining.
- The terminal utility (code-generation pass@k) is high-variance and
  domain-shifted from the ProSeCo-OWT baseline.
- No public PRISM LLaDA checkpoint exists; the default path in
  `external/PRISM/PRISM_llada/generate_samples.py` is a private Harvard/Kempner
  filesystem.

This directory exists as a placeholder so the repo layout symmetrically mirrors
`scripts/backend_validation/informed_correctors/`. If PRISM-specific scripts
become necessary in the future, they belong here and a corresponding HPC subtree
should be created under `hpc/backend_validation/prism/`.

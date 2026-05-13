# informed-correctors/Text8 Backend Validation

This directory contains thesis-side utilities for auditing and preparing the
`lindermanlab/informed-correctors` Text8 HollowMD4 backend.

The upstream repo is expected at:

```bash
external_repos/informed-correctors
```

No script in this directory launches long training by default. The intended
sequence is:

1. `stage0_env_smoke.sbatch`: imports, JAX/GPU visibility, config, and Text8
   data-path checks.
2. `stage1_tiny_train.sbatch`: tiny training-loop check only, using a small
   local HollowMD4 config and a copied/locally patched upstream tree.
3. Later Stage 2 throughput benchmark, only after Stage 0/1 pass.

The upstream training code currently assumes `md4/input_pipeline.py::DATA_DIR`
is `/root/md4/data_dir` and that `config.vocab_dir` points to a pickle file for
sample word-validity post-processing. The thesis-side Stage 1 job avoids
modifying the external repo by copying the upstream `text8/` subtree into the
run directory and patching that copy.

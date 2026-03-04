# Masked Diffusion Thesis

Research code for MSc thesis on masked diffusion models (RemeDi, ReMDM, PRISM)
with a unified inference playground and reproducible HPC experiments.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
git submodule update --init --recursive      # external/remedi, remdm, PRISM

# Smoke check (no GPU, no download)
pytest tests/ -q                             # 36 tests
bash scripts/smoke_all.sh                    # toy + dry-run for all methods
```

---

## Unified CLI

All three backends are driven through one entry point:

```bash
python -m mdm_playground.cli.run --help
```

### RemeDi (direct HF, CPU or GPU)

Requires `maple-research-lab/RemeDi-RL` (downloaded automatically, ~2 GB):

```bash
python -m mdm_playground.cli.run \
    --method remedi \
    --model_id maple-research-lab/RemeDi-RL \
    --prompt "Explain masked diffusion in one sentence." \
    --strategy remedi_policy \
    --steps 32 --max_len 256 --device cpu \
    --out_dir results/remedi_policy
```

### ReMDM (subprocess via Hydra)

```bash
# Toy mode (local, no checkpoint)
python -m mdm_playground.cli.run --method remdm --toy_mode --steps 16

# Dry-run (generate command, do not execute)
python -m mdm_playground.cli.run --method remdm --dry_run --steps 256

# Real run (HPC/CUDA only — needs checkpoint)
python -m mdm_playground.cli.run \
    --method remdm \
    --model_id /path/to/remdm.ckpt \
    --steps 256 --out_dir results/remdm
```

### PRISM (subprocess via Hydra)

```bash
python -m mdm_playground.cli.run --method prism --toy_mode
python -m mdm_playground.cli.run --method prism --dry_run --steps 256
```

### Strategies

| `--strategy` | Description |
|---|---|
| `baseline` | No remasking — commit once and keep |
| `remedi_policy` | RemeDi paper default: re-rank all positions each step |
| `threshold` | Remask committed tokens with confidence < `--tau` |
| `topk` | Remask `--k` lowest-confidence committed tokens |
| `schedule` | Decaying remask probability (`--schedule linear\|cosine`) |

---

## Output format

Each run writes to `--out_dir/`:

| File | Contents |
|---|---|
| `run_meta.json` | git commit, method, strategy, timestamp |
| `summary.json` | method-specific result summary |
| `<run_id>/trajectory.jsonl` | one JSON object per diffusion step |

Per-step JSONL fields: `step`, `tokens`, `mask_positions`, `confidence` (UPS in [0,1]),
`unmask_indices`, `remask_indices`.

---

## Python API

```python
from mdm_playground.models.remedi import RemeDiAdapter
from mdm_playground.strategies import RemediPolicyStrategy, ConfidenceThresholdRemaskStrategy
from mdm_playground.samplers import run_block_diffusion
from mdm_playground.core.logging import TrajectoryLogger

adapter = RemeDiAdapter.load("maple-research-lab/RemeDi-RL", device="cpu")

result = run_block_diffusion(
    adapter=adapter,
    messages=[{"role": "user", "content": "What is 2+2?"}],
    strategy=ConfidenceThresholdRemaskStrategy(tau=0.4),
    steps=8,
    max_length=64,
    seed=42,
)
print(result["generated_text"])
```

---

## Project structure

```
src/mdm_playground/           # Main package (pip install -e .)
├── core/                     # Shared utilities
│   ├── config.py             # load_yaml()
│   ├── utils.py              # seed_everything, save_json, git hash
│   ├── logging.py            # setup_logger, TrajectoryLogger (JSONL + npy)
│   ├── masks.py              # make_mask, gather_topk_masked
│   ├── schedules.py          # transfer_schedule, noise schedules
│   └── metrics.py            # mask_frac_curve, confidence curves
├── models/                   # Model adapters
│   ├── base.py               # ModelAdapter (ABC), ModelMeta, ForwardOutput
│   ├── remedi.py             # RemeDiAdapter (direct HF forward)
│   ├── remdm.py              # ReMDMAdapter (subprocess, Hydra)
│   └── prism.py              # PRISMAdapter (subprocess, Hydra)
├── strategies/               # Pluggable inference strategies
│   ├── base.py               # StepState, BaseStrategy
│   ├── unmask.py             # BaselineUnmaskStrategy
│   ├── remask.py             # Threshold, TopK, Scheduled
│   └── hybrid.py             # RemediPolicyStrategy
├── samplers/
│   └── block_diffusion.py    # run_block_diffusion() (direct-forward models)
└── cli/
    └── run.py                # python -m mdm_playground.cli.run

external/                     # Git submodules (not modified)
├── remedi/                   # maple-research-lab/RemeDi
├── remdm/                    # upstream ReMDM
└── PRISM/                    # upstream PRISM

scripts/
├── smoke_all.sh              # End-to-end smoke for all methods
└── smoke_infer_remedi.py     # Real RemeDi inference script

tests/
├── test_infer.py             # 22 unit tests (no checkpoint required)
└── test_smoke.py             # 14 smoke tests (toy/dry-run)

hpc/                          # Bocconi HPC workflow
configs/                      # YAML experiment configs
```

---

## Tests

```bash
pytest tests/ -q                    # 36 fast tests (no checkpoint)
pytest -m integration               # Real-model tests (needs HF download)
```

---

## HPC (Bocconi)

```bash
bash hpc/push.sh        # rsync code to cluster
bash hpc/submit.sh      # sbatch job
bash hpc/pull.sh        # fetch results
```

See [hpc/README.md](hpc/README.md) for the full workflow.

---

## Deprecated / removed

See [DEPRECATIONS.md](DEPRECATIONS.md) for a full list of removed scripts and
their replacement commands.

---

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .                     # installs mdm_playground
git submodule update --init --recursive
```




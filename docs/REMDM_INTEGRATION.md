# ReMDM Integration Guide

Complete guide for integrating the real ReMDM sampler into your thesis repo.

## Quick Start

```bash
# 1. Inspect upstream interface
python scripts/inspect_remdm_interface.py

# 2. Test locally (toy mode)
python scripts/run_remdm.py --config configs/remdm.yaml

# 3. Test command generation (dry run)
# Edit configs/remdm.yaml: set toy_mode=false, dry_run=true
python scripts/run_remdm.py --config configs/remdm.yaml

# 4. Run on Linux HPC
# Edit configs/remdm.yaml: set dry_run=false, add checkpoint_path
python scripts/run_remdm.py --config configs/remdm.yaml
```

## Architecture

**Subprocess-based integration** (no upstream code imports):
```
scripts/run_remdm.py
    ↓ loads config
configs/remdm.yaml
    ↓ creates adapter
src/.../integrations/remdm_adapter.py
    ↓ builds command & calls subprocess
    ↓ cwd=external/remdm (important for Hydra config discovery)
external/remdm/main.py (upstream, via subprocess)
    ↓ saves outputs
results/<timestamp>_remdm/external_remdm/
```

**Key files:**
- `configs/remdm.yaml` - Your experiment config
- `scripts/run_remdm.py` - Main entrypoint
- `src/masked_diffusion_thesis/integrations/remdm_adapter.py` - Adapter (subprocess wrapper)
- `external/remdm/main.py` - Upstream entrypoint (NOT modified)

## Execution Modes

| Mode | toy_mode | dry_run | Platform | Executes? | Use Case |
|------|----------|---------|----------|-----------|----------|
| **Toy** | `true` | any | Any | Mock | Testing pipeline |
| **Dry Run** | `false` | `true` | macOS | No | Local dev/validation |
| **Real** | `false` | `false` | Linux HPC | Yes | Production |

### Toy Mode
```yaml
remdm:
  toy_mode: true
```
Mock sampling with `BaseMDLM` for testing pipeline.

### Dry Run Mode (macOS safe)
```yaml
remdm:
  toy_mode: false
  dry_run: true
```
Builds and prints Hydra command without execution. Use for local development.

### Real Mode (Linux HPC only)
```yaml
remdm:
  toy_mode: false
  dry_run: false
  
  # Basic config
  mode: sample_eval
  seed: 1
  
  # Data & model (must match checkpoint)
  data: openwebtext-split
  model_size: small
  backbone: dit
  parameterization: subs
  sequence_length: 1024
  
  # Checkpoint
  upstream_checkpoint_path: /path/to/remdm_model.ckpt
  
  # Time parameterization
  T: 0
  time_conditioning: false
  
  # Sampling
  steps: 1024
  strategy: remdm-conf      # or remdm-loop
  nucleus_p: 0.9
  num_sample_batches: 5000
  
  # Batch sizes
  batch_size: 1
  eval_batch_size: 1
  perplexity_batch_size: 1
  
  # Output & wandb
  generated_seqs_path: null  # null = auto-generate
  wandb_offline: true
  
  # For remdm-loop only
  eta: 0.02
  t_on: 0.55
  t_off: 0.05
  alpha_on: 0.9
```
Executes upstream ReMDM via subprocess.

## Configuration

### Key Parameters

**Verified against upstream scripts** (`external/remdm/scripts/remdm-conf.sh`, `remdm-loop.sh`):

| Your Config | Upstream Override | Description |
|-------------|-------------------|-------------|
| `upstream_checkpoint_path` | `eval.checkpoint_path` | Path to upstream ReMDM checkpoint |
| `strategy` | `sampling.sampler` | Sampler (remdm-conf/remdm-loop/mdlm) |
| `steps` | `sampling.steps` | Number of sampling steps |
| `T` | `T` | 0 = continuous time, 1000 = discrete |
| `time_conditioning` | `time_conditioning` | Whether to use time conditioning |
| `nucleus_p` | `sampling.nucleus_p` | Top-p sampling threshold |
| `data` | `data` | Dataset config name |
| `model_size` | `model` | Model size: `small`, `medium`, `tiny`, `small-ar`, `tiny-ar`, `tiny-dimamba` |
| `backbone` | `backbone` | Architecture (dit/dimamba/ar) |
| `sequence_length` | `model.length` | Sequence length (default 1024 in model configs) |
| `num_sample_batches` | `sampling.num_sample_batches` | Total batches |
| `batch_size` | `loader.batch_size`, `loader.eval_batch_size` | Per-device batch size |

**Additional remdm-loop parameters** (optional, not in config yet):
- `eta` → `sampling.eta` (default: 0.02)
- `t_on` → `sampling.t_on` (default: 0.55)
- `t_off` → `sampling.t_off` (default: 0.05)
- `alpha_on` → `sampling.alpha_on` (default: 0.9)

### Example Command Generated

Config:
```yaml
remdm:
  upstream_checkpoint_path: /models/remdm.ckpt
  strategy: remdm-conf
  steps: 1024
  T: 0
  time_conditioning: false
  nucleus_p: 0.9
```

Generated command:
```bash
# Working directory: external/remdm (for Hydra config discovery)
python -m main \
  mode=sample_eval \
  data=openwebtext-split \
  model=small \
  backbone=dit \
  eval.checkpoint_path=/models/remdm.ckpt \
  # (from upstream_checkpoint_path in your config)
  sampling.sampler=remdm-conf \
  sampling.steps=1024 \
  T=0 \
  time_conditioning=false \
  sampling.nucleus_p=0.9 \
  [... more overrides ...]
```

## Outputs

```
results/<timestamp>_remdm/
├── meta.json                   # Your repo metadata
├── samples.pt                  # Full output dict (torch)
├── summary.json                # Compact summary with pointers
└── external_remdm/             # Upstream outputs
    ├── generated_sequences.json
    ├── config_tree.txt
    └── .hydra/
```

**summary.json** contains:
- `external_run_dir`: Path to upstream outputs
- `artifacts`: Paths to generated files
- `command`: Command executed (in dry_run mode)
- `meta`: Run metadata

## Development Workflow

### On macOS (local)
1. Edit config, set `dry_run: true`
2. Run: `python scripts/run_remdm.py --config configs/remdm.yaml`
3. Check logs for generated command
4. Adjust config if needed
5. Commit and push

### On Linux HPC
1. Pull latest code
2. Edit config: set `dry_run: false`, add `checkpoint_path`
3. Run: `python scripts/run_remdm.py --config configs/remdm.yaml`
4. Monitor: `tail -f logs/<timestamp>_remdm/run_remdm.log`
5. Check outputs: `ls results/<timestamp>_remdm/external_remdm/`

## Debugging

**Check what command would run:**
```bash
# Set dry_run: true in config, then:
python scripts/run_remdm.py --config configs/remdm.yaml
cat results/<latest>/summary.json | grep -A 20 '"command"'
```

**Explore upstream options:**
```bash
python scripts/inspect_remdm_interface.py
python scripts/inspect_remdm_interface.py --script remdm-conf.sh
```

**Check logs:**
```bash
tail -f logs/<timestamp>_remdm/run_remdm.log
```

**Common issues:**
- "ReMDM submodule not found" → `git submodule update --init --recursive`
- "Command failed" → Check logs, verify upstream_checkpoint_path
- "Import error" (on macOS) → Use `dry_run: true` for local dev

## Adding Parameters

To add a new config parameter:

1. Add to `ReMDMRunConfig` in `remdm_adapter.py`:
```python
@dataclass
class ReMDMRunConfig:
    # ...existing fields...
    my_param: float = 1.0  # NEW
```

2. Add to `configs/remdm.yaml`:
```yaml
remdm:
  my_param: 1.0  # NEW
```

3. Add mapping in `_build_hydra_overrides()`:
```python
def _build_hydra_overrides(self, output_dir: Path) -> List[str]:
    # ...existing overrides...
    overrides.append(f"upstream.param={cfg.my_param}")  # NEW
```

## Implementation Notes

- **No upstream imports**: Adapter only calls subprocess, safe for macOS
- **Hydra overrides**: Config mapped to CLI args (see `_build_hydra_overrides()`)
- **Output collection**: Basic parsing in `_collect_outputs()` (extend as needed)
- **Error handling**: Subprocess errors logged with stdout/stderr
- **TODOs in code**: Review `remdm_adapter.py` for inline TODOs

## References

- Upstream docs: `external/remdm/README.md`
- Example scripts: `external/remdm/scripts/*.sh`
- Upstream config: `external/remdm/configs/config.yaml`
- Adapter code: `src/masked_diffusion_thesis/integrations/remdm_adapter.py`

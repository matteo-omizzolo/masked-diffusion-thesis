# PRISM LLaDA smoke log

Date: 2026-05-13
Repo: `external/PRISM`
Local commit: `266d51e886b28ac8bfb416eb8715394aa08a4e0f`
Remote `origin/main` checked after fetch: `6e5ed6410d3d88fa1231eb78b002e3365be6deaa`

Note: the local PRISM worktree is behind remote main. Remote inspection was performed with `git fetch` and `git show origin/main:...`; the worktree was not checked out or updated.

## Commands

```bash
python3 - <<'PY'
import importlib.util
mods = ['torch','transformers','datasets','peft','huggingface_hub','evaluate','numpy']
for m in mods:
    spec = importlib.util.find_spec(m)
    print(f'{m}: {"FOUND" if spec else "MISSING"}')
    if spec:
        try:
            mod = __import__(m)
            print(f'  version: {getattr(mod, "__version__", "<unknown>")}')
        except Exception as e:
            print(f'  import_error: {type(e).__name__}: {e}')
PY
```

Result:

```text
torch: FOUND
  version: 2.8.0
transformers: FOUND
  version: 4.57.6
datasets: MISSING
peft: MISSING
huggingface_hub: FOUND
  version: 0.36.2
evaluate: MISSING
numpy: FOUND
  version: 1.23.5
```

```bash
PYTHONPATH=external/PRISM/PRISM_llada python3 - <<'PY'
try:
    import sampling
    print('sampling_import: OK')
    print('has_llada_inference:', hasattr(sampling, 'llada_inference'))
except Exception as e:
    print('sampling_import: FAIL')
    print(type(e).__name__, e)
try:
    import train
    print('train_import: OK')
except Exception as e:
    print('train_import: FAIL')
    print(type(e).__name__, e)
PY
```

Result:

```text
sampling_import: OK
has_llada_inference: True
train_import: FAIL
ModuleNotFoundError No module named 'wandb'
```

```bash
PYTHONPATH=external/PRISM/PRISM_llada python3 - <<'PY'
try:
    import code_utils
    print('code_utils_import: OK')
except Exception as e:
    print('code_utils_import: FAIL')
    print(type(e).__name__, e)
PY
```

Result:

```text
code_utils_import: FAIL
ModuleNotFoundError No module named 'datasets'
```

## Verdict

Sampling utilities import locally. Training/data utilities are blocked by missing dependencies. No LLaDA weights or PRISM checkpoints were downloaded, and no model load, training, fine-tuning, or sampling run was launched.

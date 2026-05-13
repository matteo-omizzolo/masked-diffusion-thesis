# informed-correctors/Text8 smoke log

Date: 2026-05-13
Repo: `external_repos/informed-correctors`
Commit: `8371dec19404eae279a4eb5ea4e06977d8d173b6`
Remote HEAD checked: `8371dec19404eae279a4eb5ea4e06977d8d173b6`

## Commands

```bash
python3 - <<'PY'
import importlib.util
mods = ['jax','flax','tensorflow','tensorflow_datasets','grain','orbax','wandb','ml_collections']
for m in mods:
    spec = importlib.util.find_spec(m)
    print(f'{m}: {"FOUND" if spec else "MISSING"}')
PY
```

Result:

```text
jax: MISSING
flax: MISSING
tensorflow: MISSING
tensorflow_datasets: MISSING
grain: MISSING
orbax: MISSING
wandb: MISSING
ml_collections: MISSING
```

```bash
PYTHONPATH=external_repos/informed-correctors/text8 python3 - <<'PY'
try:
    from md4.configs.hollow_md4 import text8
    cfg = text8.get_config()
    print('config_import: OK')
    for key in ['model_type','dataset','data_shape','timesteps','num_train_steps','batch_size','sampler','vocab_dir']:
        print(f'{key}: {cfg.get(key, "<MISSING>")}')
except Exception as e:
    print('config_import: FAIL')
    print(type(e).__name__, e)
PY
```

Result:

```text
config_import: FAIL
ModuleNotFoundError No module named 'ml_collections'
```

```bash
PYTHONPATH=external_repos/informed-correctors/text8 python3 - <<'PY'
try:
    import md4.sampling
    print('sampling_import: OK')
except Exception as e:
    print('sampling_import: FAIL')
    print(type(e).__name__, e)
try:
    from md4.models.diffusion import hollow_md4
    print('hollow_md4_import: OK')
except Exception as e:
    print('hollow_md4_import: FAIL')
    print(type(e).__name__, e)
PY
```

Result:

```text
sampling_import: FAIL
ModuleNotFoundError No module named 'jax'
hollow_md4_import: FAIL
ModuleNotFoundError No module named 'flax'
```

## Verdict

Local smoke is blocked by missing JAX/Flax/Text8 dependencies. No training or sampling was launched.

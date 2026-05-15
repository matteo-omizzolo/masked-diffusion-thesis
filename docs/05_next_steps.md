# Next Steps

## Current Next Action

Author email in `docs/email_informed_correctors_authors.md` was sent
**2026-05-14**. Awaiting reply; follow up after 7-10 days if none. In
parallel, drive the external-GPU path below.

## External-GPU No-Author-Response Path

Do not block indefinitely waiting for a reply.

1. Prepare the Azure/external GPU instance with matching driver/toolkit and no
   MIG partitioning.
2. Clone this branch and create the ignored `external_repos/informed-correctors`
   checkout at commit `8371dec`.
3. Run Stage 0 from `docs/10_external_gpu_text8_fallback.md`.
4. If Stage 0 passes, run Stage 1 tiny training-loop smoke only.
5. If Stage 1 passes, design Stage 2 throughput benchmark.
6. Only after Stage 2, decide whether to train a useful Text8 checkpoint.
7. Only after a non-degenerate checkpoint exists, run timing headroom diagnostics.

## Guardrails

- No full Text8 training without explicit approval.
- No huge checkpoint/model downloads without approval.
- No new ProSeCo experiments unless they fix a reproducibility gap.
- No PRISM LLaDA timing work unless checkpoint access and timing/token-selection
  confounds change.

## Validation Commands

```bash
python3.11 - <<'PY'
import json
json.load(open("results/EXPERIMENT_INDEX.json"))
json.load(open("results/BACKEND_FEASIBILITY_INDEX.json"))
print("JSON indexes valid")
PY

pytest -q
git diff --check
git status --short
```

# Next Steps

## Current Next Action

Author email in `docs/email_informed_correctors_authors.md` was sent
**2026-05-14**. Awaiting reply; follow up after 7-10 days if none. In
parallel, drive the no-author-response path below.

## Parallel No-Author-Response Path

Do not block indefinitely waiting for a reply.

1. Run Stage 0 environment/data/config smoke for informed-correctors/Text8.
2. If Stage 0 passes, run Stage 1 tiny training-loop smoke only.
3. If Stage 1 passes, design Stage 2 throughput benchmark.
4. Only after Stage 2, decide whether to train a useful Text8 checkpoint.
5. Only after a non-degenerate checkpoint exists, run timing headroom diagnostics.

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

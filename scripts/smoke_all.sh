#!/usr/bin/env bash
# scripts/smoke_all.sh
#
# Runs a quick end-to-end smoke check for every supported method and strategy.
# All runs use toy/dry-run mode so no real checkpoints or GPU are needed.
# Outputs are written to runs/smoke_<timestamp>/.
#
# Usage:
#   bash scripts/smoke_all.sh               # fast (toy/dry-run)
#   REAL_REMEDI=1 bash scripts/smoke_all.sh # also run real RemeDi on CPU
#
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUNS_DIR="runs/smoke_${TIMESTAMP}"
PYTHON="${PYTHON:-python}"

echo "=== MDM Playground smoke run ==="
echo "  Output dir: ${RUNS_DIR}"
echo "  Python: $(${PYTHON} --version)"
echo

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
run() {
    local label="$1"; shift
    local out="${RUNS_DIR}/${label}"
    echo "--- [${label}] ---"
    ${PYTHON} -m mdm_playground.cli.run "$@" --out_dir "${out}"
    echo "    saved → ${out}"
}

# ---------------------------------------------------------------------------
# A) ReMDM — toy mode (exercises Python code path, no subprocess)
# ---------------------------------------------------------------------------
run "remdm_toy" \
    --method remdm \
    --strategy remdm_conf \
    --steps 16 \
    --toy_mode

# ---------------------------------------------------------------------------
# B) ReMDM — dry_run (builds full Hydra command, no execution)
# ---------------------------------------------------------------------------
run "remdm_dry_run" \
    --method remdm \
    --strategy remdm_conf \
    --steps 256 \
    --dry_run

# ---------------------------------------------------------------------------
# C) PRISM — toy mode
# ---------------------------------------------------------------------------
run "prism_toy" \
    --method prism \
    --strategy prism \
    --steps 16 \
    --toy_mode

# ---------------------------------------------------------------------------
# D) PRISM — dry_run
# ---------------------------------------------------------------------------
run "prism_dry_run" \
    --method prism \
    --steps 256 \
    --dry_run

# ---------------------------------------------------------------------------
# E) RemeDi with real model (opt-in via REAL_REMEDI=1)
#    Requires internet access the first time (downloads ~2 GB).
# ---------------------------------------------------------------------------
if [[ "${REAL_REMEDI:-0}" == "1" ]]; then
    MODEL="${REMEDI_MODEL:-maple-research-lab/RemeDi-RL}"
    PROMPT="${REMEDI_PROMPT:-Explain masked diffusion in one sentence.}"

    for STRAT in baseline remedi_policy threshold topk schedule; do
        run "remedi_${STRAT}" \
            --method remedi \
            --model_id "${MODEL}" \
            --prompt "${PROMPT}" \
            --strategy "${STRAT}" \
            --steps 4 \
            --max_len 32 \
            --block_size 32 \
            --device cpu \
            --seed 42
    done
fi

echo
echo "=== Done. Results in ${RUNS_DIR} ==="

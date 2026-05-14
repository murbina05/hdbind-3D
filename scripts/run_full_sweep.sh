#!/usr/bin/env bash
#
# run_full_sweep.sh — Full 43-target Tier 3 EGNN sweep.
#
# Pipeline:
#   06  Build full LMDB on all 43 DUDE-Z targets   (~3 min CPU)
#   07  M2 (full complex)  on GPU 0  in background  (~5–7 h)
#   07  M1 (lig-only)      on GPU 1  in background  (~1.5–2 h)
#   10  Bias-controls panel on the M2 checkpoint    (~10 min)
#
# Out of scope (stub only): 11_eval_goldilocks.py — Goldilocks decoy eval.
# Run that manually after this script completes.
#
# CLAUDE.md gates that this script trips:
#   * "any 43-target run" requires explicit user authorization
#   * "do not silently kick off jobs that take more than ~10 minutes"
#
# Set FORCE=1 to skip the confirmation prompt.

set -euo pipefail

REPO_ROOT="/home/maurbina/hdbind-3D"
VENV="/home/maurbina/.venvs/dc_featurizers"

cd "$REPO_ROOT"

# ── Pre-flight ──────────────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
    echo "ERROR: venv not found at $VENV" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA not available; this script needs GPUs" >&2
    exit 1
fi

N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
if (( N_GPUS < 2 )); then
    echo "WARNING: only $N_GPUS GPU visible; will run M2 then M1 sequentially" >&2
fi

GIT_SHA=$(git -C "$REPO_ROOT" rev-parse --short=7 HEAD 2>/dev/null || echo "no-git")
RUN_ID="$(date +%Y%m%d-%H%M%S)-${GIT_SHA}"
DATA_RUN="$REPO_ROOT/outputs/06_build_egnn_dataset/${RUN_ID}"
M2_RUN_ID="${RUN_ID}-m2"
M1_RUN_ID="${RUN_ID}-m1"
M2_RUN="$REPO_ROOT/outputs/07_train_egnn/${M2_RUN_ID}"
M1_RUN="$REPO_ROOT/outputs/07_train_egnn/${M1_RUN_ID}"

echo "─────────────────────────────────────────────────────────────────────"
echo "Tier 3 full sweep — RUN_ID=${RUN_ID}"
echo "  data:   $DATA_RUN"
echo "  M2:     $M2_RUN  (GPU 0)"
echo "  M1:     $M1_RUN  (GPU 1)"
echo "  budget: ~5–7 h wall (M2 dominates; M1 finishes ~3 h earlier)"
echo "─────────────────────────────────────────────────────────────────────"

if [[ "${FORCE:-0}" != "1" ]]; then
    read -r -p "Proceed? [y/N] " ans
    if [[ "$ans" != "y" && "$ans" != "Y" ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# ── Step 1: build full LMDB ─────────────────────────────────────────────────
echo
echo "[1/4] Building full 43-target LMDB ..."
python scripts/06_build_egnn_dataset.py --all-targets --run-id "$RUN_ID"

# ── Step 2: kick off M2 + M1 in parallel ────────────────────────────────────
EPOCHS="${EPOCHS:-50}"
EVAL_EVERY="${EVAL_EVERY:-5}"
PATIENCE="${PATIENCE:-4}"

COMMON_ARGS=(
    --all-targets
    --epochs "$EPOCHS"
    --eval-every "$EVAL_EVERY"
    --early-stop-patience "$PATIENCE"
    --dataset-dir "$DATA_RUN"
)

echo
echo "[2/4] Launching M2 (full complex) on GPU 0 ..."
CUDA_VISIBLE_DEVICES=0 nohup python scripts/07_train_egnn.py \
    --variant full \
    --run-id "$M2_RUN_ID" \
    "${COMMON_ARGS[@]}" \
    > "$REPO_ROOT/outputs/07_train_egnn/${M2_RUN_ID}.stdout" 2>&1 &
M2_PID=$!
echo "  M2 pid=$M2_PID  log: outputs/07_train_egnn/${M2_RUN_ID}/train.log"

if (( N_GPUS >= 2 )); then
    echo
    echo "[2/4] Launching M1 (lig-only) on GPU 1 in parallel ..."
    CUDA_VISIBLE_DEVICES=1 nohup python scripts/07_train_egnn.py \
        --variant lig_only \
        --run-id "$M1_RUN_ID" \
        "${COMMON_ARGS[@]}" \
        > "$REPO_ROOT/outputs/07_train_egnn/${M1_RUN_ID}.stdout" 2>&1 &
    M1_PID=$!
    echo "  M1 pid=$M1_PID  log: outputs/07_train_egnn/${M1_RUN_ID}/train.log"

    echo
    echo "Waiting for both training runs ..."
    wait "$M2_PID"
    wait "$M1_PID"
else
    echo
    echo "[2/4] Waiting for M2 to finish (M1 will run after) ..."
    wait "$M2_PID"

    echo
    echo "[2b/4] Launching M1 (lig-only) on GPU 0 ..."
    CUDA_VISIBLE_DEVICES=0 python scripts/07_train_egnn.py \
        --variant lig_only \
        --run-id "$M1_RUN_ID" \
        "${COMMON_ARGS[@]}"
fi

# ── Step 3: bias controls on M2 ─────────────────────────────────────────────
echo
echo "[3/4] Running bias controls on M2 ..."
python scripts/10_bias_controls.py \
    --train-run "$M2_RUN" \
    --run-id "${RUN_ID}-bias-on-m2"

# ── Step 4: hand-off ────────────────────────────────────────────────────────
echo
echo "[4/4] Goldilocks eval (11_eval_goldilocks.py) is a stub — running it"
echo "       prints the plan and exits. Implement before signing off on §5 gate 1."
python scripts/11_eval_goldilocks.py || true

echo
echo "─────────────────────────────────────────────────────────────────────"
echo "Sweep complete."
echo "  M2:     $M2_RUN"
echo "  M1:     $M1_RUN"
echo "  bias:   $REPO_ROOT/outputs/10_bias_controls/${RUN_ID}-bias-on-m2"
echo "─────────────────────────────────────────────────────────────────────"

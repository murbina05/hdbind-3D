#!/usr/bin/env bash
# Phase 5 scenario A — all 43 DUDE-Z primary targets through the optimized
# FLOWR.ROOT extraction pipeline.
#
# Launch (recommended): inside a tmux session so it survives terminal close:
#
#   tmux new -s flowr_extract
#   bash /home/maurbina/hdbind-3D/scripts/run_full_extraction.sh
#   # Detach: Ctrl-b d
#   # Reattach: tmux attach -t flowr_extract
#
# Foreground (visible tqdm, lost on terminal close):
#   bash /home/maurbina/hdbind-3D/scripts/run_full_extraction.sh
#
# Do NOT use `nohup ... &` — that breaks tqdm rendering. Tmux preserves it.
#
# Resume:
#   The Python entrypoint checks features_root/<target>/manifest.json for each
#   target and skips any whose n_extracted + n_failed == n_raw_in_source.
#   --resume is on by default (use --no_resume to force re-extract).

set -euo pipefail

LOG_DIR=/home/maurbina/hdbind-3D/logs/flowr_root
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/extraction_${TS}.log"

# Tee everything to the timestamped log. `-u` makes Python unbuffered so tqdm
# renders in real time when tee'd; tee -a preserves the log even if a viewer
# attaches/detaches via tmux.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== FLOWR.ROOT scenario A extraction started at $TS ==="
echo "log: $LOG_FILE"
echo "host: $(hostname)"
echo "cuda devices: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/maurbina/hdbind-3D:/home/maurbina/flowr_root
PY=/home/maurbina/.venvs/dc_featurizers/bin/python

# Workers=16 per user direction (theory: GPU-bound at 60 ms forward, won't
# beat 8; downside risk low — 24 GB RAM, ~25% CPU on 32-core). Documented
# in flowr_root_throughput.md.
"$PY" -u /home/maurbina/hdbind-3D/scripts/extract_flowr_features.py \
    --all_targets \
    --workers 16 \
    --batch_size 1 \
    --bf16 \
    --devices cuda:0 \
    --features_root /home/maurbina/hdbind-3D/features/flowr_root/ \
    --heartbeat_s 60 \
    "$@"

echo
echo "=== FLOWR.ROOT scenario A extraction finished at $(date +%Y%m%d_%H%M%S) ==="

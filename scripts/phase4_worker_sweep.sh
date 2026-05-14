#!/bin/bash
# Phase 4 worker-count sensitivity sweep on full AA2AR (4487 poses).
# Captures ms/pose (manifest), peak RSS (/usr/bin/time -v), peak GPU util
# (nvidia-smi sampled at 1Hz). Each run ~8 min, total ~32 min.
set -euo pipefail

WORKERS_LIST=(4 8 16 24)
TARGET=AA2AR
LOG_DIR=/home/maurbina/hdbind-3D/notes/phase4_worker_sweep
mkdir -p "$LOG_DIR"
RESULTS_TSV="$LOG_DIR/results.tsv"
echo -e "workers\telapsed_s\tms_per_lig\tpeak_rss_mb\tpeak_gpu_util_pct\tpeak_gpu_mem_pct\tn_extracted" > "$RESULTS_TSV"

for W in "${WORKERS_LIST[@]}"; do
  OUT=/tmp/phase4_sweep_W${W}
  GPU_LOG=$LOG_DIR/W${W}_gpu.csv
  TIME_LOG=$LOG_DIR/W${W}_time.log
  RUN_LOG=$LOG_DIR/W${W}_stdout.log
  rm -rf "$OUT"

  echo "=== W=$W start $(date) ==="
  # 1Hz GPU sampling in background
  ( nvidia-smi --query-gpu=utilization.gpu,utilization.memory \
      --format=csv,noheader,nounits -l 1 > "$GPU_LOG" 2>&1 ) &
  NVSMI_PID=$!

  # The actual run, with peak-RSS captured by /usr/bin/time
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=/home/maurbina/hdbind-3D:/home/maurbina/flowr_root \
  /usr/bin/time -v \
    /home/maurbina/.venvs/dc_featurizers/bin/python -u \
      /home/maurbina/hdbind-3D/scripts/extract_flowr_features.py \
      --target "$TARGET" --batch_size 1 --workers "$W" --bf16 --instrument \
      --output_dir "$OUT" \
      > "$RUN_LOG" 2> "$TIME_LOG" || { kill $NVSMI_PID 2>/dev/null; exit 1; }

  kill $NVSMI_PID 2>/dev/null || true
  wait $NVSMI_PID 2>/dev/null || true

  # Parse stats
  ELAPSED=$(/home/maurbina/.venvs/dc_featurizers/bin/python -c \
      "import json; m=json.load(open('$OUT/manifest.json')); print(f\"{m['elapsed_seconds']:.1f}\")")
  MS_PER_LIG=$(/home/maurbina/.venvs/dc_featurizers/bin/python -c \
      "import json; m=json.load(open('$OUT/manifest.json')); print(f\"{m['ms_per_ligand_avg']:.2f}\")")
  N_EXTRACTED=$(/home/maurbina/.venvs/dc_featurizers/bin/python -c \
      "import json; m=json.load(open('$OUT/manifest.json')); print(m['n_extracted'])")
  PEAK_RSS_KB=$(grep "Maximum resident set size" "$TIME_LOG" | awk '{print $NF}')
  PEAK_RSS_MB=$((PEAK_RSS_KB / 1024))
  PEAK_GPU_UTIL=$(/home/maurbina/.venvs/dc_featurizers/bin/python -c "
import sys
with open('$GPU_LOG') as f:
    vals = []
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 2 and parts[0].strip().isdigit():
            vals.append(int(parts[0].strip()))
print(max(vals) if vals else 0)")
  PEAK_GPU_MEM_PCT=$(/home/maurbina/.venvs/dc_featurizers/bin/python -c "
import sys
with open('$GPU_LOG') as f:
    vals = []
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 2 and parts[1].strip().isdigit():
            vals.append(int(parts[1].strip()))
print(max(vals) if vals else 0)")

  echo -e "${W}\t${ELAPSED}\t${MS_PER_LIG}\t${PEAK_RSS_MB}\t${PEAK_GPU_UTIL}\t${PEAK_GPU_MEM_PCT}\t${N_EXTRACTED}" >> "$RESULTS_TSV"
  echo "  W=$W done: elapsed=${ELAPSED}s ms/lig=${MS_PER_LIG} rss=${PEAK_RSS_MB}MB gpu_util_pk=${PEAK_GPU_UTIL}% n=${N_EXTRACTED}"

  # Free disk: each AA2AR run is ~70 MB of .npz; keep manifest, drop the rest
  find "$OUT" -name "*.npz" -delete
done

echo "=== Sweep complete $(date) ==="
echo "Results table:"
cat "$RESULTS_TSV"

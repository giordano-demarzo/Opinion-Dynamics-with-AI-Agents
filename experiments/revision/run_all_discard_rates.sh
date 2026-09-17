#!/bin/bash
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate congestion-vllm
cd /home/jovyan/LLMs_opinion_dynamics_bias

MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "google/gemma-3-12b-it"
  "google/gemma-3-27b-it"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
  "Qwen/Qwen3-14B"
  "Qwen/Qwen3-32B"
)

for MODEL in "${MODELS[@]}"; do
  NAME=$(basename "$MODEL")
  OUT="resubmission/data/discard_rates/${NAME}/discard_rates_0.2.csv"
  if [ -f "$OUT" ]; then
    N_LINES=$(wc -l < "$OUT")
    if [ "$N_LINES" -ge 1000 ]; then
      echo "=== SKIP $MODEL (already complete: $N_LINES lines) ==="
      continue
    fi
  fi
  echo "=== Starting $MODEL at $(date) ==="
  python resubmission/experiments/run_discard_rate_vllm.py \
    --model "$MODEL" --N-sim 100 --batch-size 256 \
    --gpu-memory-utilization 0.90 --max-model-len 4096
  echo "=== Finished $MODEL at $(date) ==="
done

echo "=== ALL DISCARD-RATE RUNS COMPLETE at $(date) ==="

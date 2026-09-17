#!/bin/bash
# Phase 2: run the full social transition-probability sweep (same protocol as
# the main experiment) on the new held-out pair sets:
#   - 15 GlobalOpinionQA-grounded pairs (answers R1 pt.2: does the >60%
#     metastable-fraction result replicate on an independently-sourced,
#     non-ChatGPT-generated set of opinion pairs?)
#   - 20 AI-safety-grounded pairs (partial Option B: does the same physics
#     hold for pairs with an explicit compliant/violation framing?)
# Run AFTER run_all_isolation.sh completes (single GPU, sequential).
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate congestion-vllm

cd /home/jovyan/LLMs_opinion_dynamics_bias/resubmission/data

MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "google/gemma-3-12b-it"
  "google/gemma-3-27b-it"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
  "Qwen/Qwen3-14B"
  "Qwen/Qwen3-32B"
)

PAIRS_FILE="/home/jovyan/LLMs_opinion_dynamics_bias/resubmission/data/new_opinion_pairs_combined.csv"
SCRIPT="/home/jovyan/LLMs_opinion_dynamics_bias/submission/experiments/run_transition_prob_experiment.py"

for MODEL in "${MODELS[@]}"; do
  NAME=$(basename "$MODEL")
  OUT_DIR="${NAME}/results_batched_vllm_explicit_v2/N=50"
  if [ -d "$OUT_DIR" ]; then
    N_FILES=$(find "$OUT_DIR" -name "transition_prob_*.txt" | wc -l)
    if [ "$N_FILES" -ge 35 ]; then
      echo "=== SKIP $MODEL (already complete: $N_FILES files) ==="
      continue
    fi
  fi
  echo "=== Starting $MODEL at $(date) ==="
  python "$SCRIPT" \
    --model "$MODEL" \
    --opinion-pairs-file "$PAIRS_FILE" \
    --N 50 --N-sim 100 --batch-size 128 \
    --gpu-memory-utilization 0.90 --max-model-len 4096 \
    --no-plots
  echo "=== Finished $MODEL at $(date) ==="
done

echo "=== ALL NEW-PAIRS TRANSITION-PROB RUNS COMPLETE at $(date) ==="

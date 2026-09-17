#!/bin/bash
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate congestion-vllm
cd /home/jovyan/LLMs_opinion_dynamics_bias
export GEMINI_API_KEY=$(cat /tmp/.GEMINI_API_KEY)
export OPENAI_API_KEY=$(cat /tmp/.OPENAI_API_KEY)

ORIGINAL_PAIRS="submission/opinion_pairs.csv"
SCRIPT="resubmission/experiments/run_api_experiments.py"

for BACKEND in gemini openai; do
  echo "=== [$BACKEND] transition-prob + discard rates, original 100 pairs, starting $(date) ==="
  python "$SCRIPT" --backend "$BACKEND" --task transitionprob \
    --opinion-pairs-file "$ORIGINAL_PAIRS" --out-tag original_100 \
    --N 50 --N-sim 100 --batch-size 50
  echo "=== [$BACKEND] transition-prob + discard rates, original 100 pairs, finished $(date) ==="
done
echo "=== ALL API DISCARD-RATE RUNS COMPLETE at $(date) ==="

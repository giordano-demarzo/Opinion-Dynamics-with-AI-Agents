#!/bin/bash
# API-backend (gemini-2.5-flash-lite, gpt-5-mini) transition-probability runs
# for the 12 Pew 2017 typology pairs. --max-tokens 96 for the long verbatim
# statements. Network-bound; safe to run in parallel with the vLLM sweep.
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate congestion-vllm
cd /home/jovyan/LLMs_opinion_dynamics_bias
export GEMINI_API_KEY=$(cat /tmp/.GEMINI_API_KEY)
export OPENAI_API_KEY=$(cat /tmp/.OPENAI_API_KEY)

PAIRS="resubmission/data/pew_typology_2017_pairs.csv"
SCRIPT="resubmission/experiments/run_api_experiments.py"

for BACKEND in gemini openai; do
  echo "=== [$BACKEND] transition-prob, 12 Pew pairs, starting $(date) ==="
  python "$SCRIPT" --backend "$BACKEND" --task transitionprob \
    --opinion-pairs-file "$PAIRS" --out-tag pew12 \
    --N 50 --N-sim 100 --batch-size 50 --max-tokens 96
  echo "=== [$BACKEND] transition-prob, 12 Pew pairs, finished $(date) ==="
done

echo "=== ALL PEW API RUNS COMPLETE at $(date) ==="

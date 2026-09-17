#!/bin/bash
# Transition-probability sweep for the 12 Pew 2017 political-typology
# paired-statement items (used verbatim and in full) -- the new held-out set.
# Same protocol as the main experiment and the earlier new-pairs run, but
# with --max-tokens 96 because the verbatim Pew statements are long.
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

PAIRS_FILE="/home/jovyan/LLMs_opinion_dynamics_bias/resubmission/data/pew_typology_2017_pairs.csv"
SCRIPT="/home/jovyan/LLMs_opinion_dynamics_bias/submission/experiments/run_transition_prob_experiment.py"

for MODEL in "${MODELS[@]}"; do
  NAME=$(basename "$MODEL")
  OUT_DIR="${NAME}/results_batched_vllm_explicit_v2/N=50"
  # resume support: skip a model only if all 12 Pew output files already exist
  N_DONE=$(python - "$OUT_DIR" "$PAIRS_FILE" << 'PYEOF'
import sys, pandas as pd
from pathlib import Path
out_dir, pairs_file = Path(sys.argv[1]), sys.argv[2]
pairs = pd.read_csv(pairs_file)
n = sum((out_dir / f"transition_prob_50_0.2_{a}.txt").exists() for a in pairs.Opinion_A)
print(n)
PYEOF
)
  if [ "$N_DONE" -ge 12 ]; then
    echo "=== SKIP $MODEL (Pew pairs already complete: $N_DONE/12) ==="
    continue
  fi
  echo "=== Starting $MODEL at $(date) (existing Pew files: $N_DONE/12) ==="
  python "$SCRIPT" \
    --model "$MODEL" \
    --opinion-pairs-file "$PAIRS_FILE" \
    --N 50 --N-sim 100 --batch-size 128 \
    --max-tokens 96 \
    --gpu-memory-utilization 0.90 --max-model-len 4096 \
    --no-plots
  echo "=== Finished $MODEL at $(date) ==="
done

echo "=== ALL PEW TRANSITION-PROB RUNS COMPLETE at $(date) ==="

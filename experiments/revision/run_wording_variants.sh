#!/bin/bash
# Waits for the vLLM environment install to finish, then runs the wording-variant
# experiment for three open-weights models sequentially.
LOG=/home/jovyan/vllm-env2-install.log
until grep -q 'INSTALL_EXIT=' $LOG 2>/dev/null; do sleep 30; done
grep -q 'INSTALL_EXIT=0' $LOG || { echo "vLLM install failed"; exit 1; }
PY=/home/jovyan/vllm-env2/bin/python
cd /home/jovyan/LLMs_opinion_dynamics_bias/resubmission
for M in google/gemma-3-27b-it Qwen/Qwen3-32B meta-llama/Llama-3.1-8B-Instruct; do
  short=${M##*/}
  echo "=== START $M $(date)"
  $PY experiments/run_wording_variants.py --model $M --max-model-len 4096 > logs/wording_variants_${short}.log 2>&1
  echo "=== END $M exit=$? $(date)"
done
echo ALL_DONE

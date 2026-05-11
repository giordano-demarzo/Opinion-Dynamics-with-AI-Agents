#!/usr/bin/env python3
"""
Tipping-point experiment: start aligned (80 % opinion A), introduce
N_stubborn_B stubborn-B agents at t1, remove them at t2, and observe
whether the system remains misaligned (inside-spinodal pair) or returns
(outside-spinodal pair).

Pairs
-----
  1. gender self-identification  (inside  spinodal: beta=2.99, h=0.31, hs=0.43)
       -> system flips and stays permanently after stubborn agents leave
  2. renewable energy            (outside spinodal: beta=5.39, h=0.78, hs=0.63)
       -> system flips but returns in ~1 sweep after stubborn agents leave

Theory note (Ns=200, N_reg=50)
-------------------------------
  gender self-id: equilibrium with stubborn ~ -0.97, drift after removal ~ +0.007/sweep
  renewable energy: equilibrium with stubborn ~ -0.70, drift after removal ~ +1.1/sweep

Usage:
    python run_tipping_point_experiment.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
DATA_DIR = Path(__file__).parent.parent / "data"

import os
import sys
import json
import random
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vllm import LLM, SamplingParams
from opinion_dynamics_vllm import (
    generate_unique_random_strings,
    synchronized_shuffle,
    apply_chat_template_to_prompts,
    create_opinion_prompt,
    parse_opinion_response,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID        = "google/gemma-3-27b-it"
N_REGULAR       = 50
TEMPERATURE     = 0.20
MAX_TOKENS      = 16
FRAC_A_INIT     = 0.8       # initial fraction supporting opinion A
N_STUBBORN_B    = 35       # B-stubborn agents introduced at t1
STEPS_PER_BLOCK = 50        # MC steps per recorded time-point (≈ 1 sweep)
N_BLOCKS_BEFORE = 5        # sweeps before introducing stubborn agents
N_BLOCKS_DURING = 10        # sweeps with stubborn agents present
N_BLOCKS_AFTER  = 10        # sweeps after removing stubborn agents
N_RUNS          = 3         # independent repetitions per pair

BASE_PATH = Path(__file__).parent.parent / "data"

PAIRS = [
    {
        "opinion_A": "gender self-identification",
        "opinion_B": "biological sex classification",
        "label":     "gender self-identification",
    },
    #{
    #    "opinion_A": "renewable energy",
    #    "opinion_B": "fossil fuels",
    #    "label":     "renewable energy",
    #},
]


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_block(
    llm,
    sampling_params,
    current_opinions: list,
    opinion_A: str,
    opinion_B: str,
    N_stubborn_B: int,
    steps: int,
) -> tuple:
    """
    Run `steps` MC updates with N_stubborn_B stubborn-B agents.
    Returns (avg_m, updated_opinions).
    """
    N_regular = len(current_opinions)
    N_total = N_regular + N_stubborn_B
    mags = []

    for _ in range(steps):
        idx = random.randint(0, N_regular - 1)

        all_names = generate_unique_random_strings(length=2, num_strings=N_total)
        all_opinions = (
            current_opinions[:idx] +
            current_opinions[idx + 1:] +
            [opinion_B] * N_stubborn_B
        )
        env_names, env_opinions = synchronized_shuffle(all_names, all_opinions)

        prompt = create_opinion_prompt(env_names, env_opinions, opinion_A, opinion_B)
        formatted = apply_chat_template_to_prompts(llm, [prompt])
        output = llm.generate(formatted, sampling_params)
        chosen = parse_opinion_response(
            output[0].outputs[0].text, opinion_A, opinion_B
        )
        if chosen is not None:
            current_opinions[idx] = chosen

        Na = current_opinions.count(opinion_A)
        mags.append((Na - (N_regular - Na)) / N_regular)

    return float(np.mean(mags)), current_opinions


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_tipping_point(
    llm,
    sampling_params,
    opinion_A: str,
    opinion_B: str,
    label: str,
    n_runs: int = N_RUNS,
    model_id: str = MODEL_ID,
    base_path: Path = BASE_PATH,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Tipping-point experiment: '{opinion_A}' vs '{opinion_B}'")
    print(f"N_regular={N_REGULAR}, N_stubborn_B={N_STUBBORN_B}")
    print(f"Blocks: {N_BLOCKS_BEFORE} free | {N_BLOCKS_DURING} stubborn | {N_BLOCKS_AFTER} free")
    print(f"Runs: {n_runs}")
    print(f"{'='*60}")

    all_trajectories = []   # shape: (n_runs, total_blocks)
    t1 = N_BLOCKS_BEFORE
    t2 = N_BLOCKS_BEFORE + N_BLOCKS_DURING
    total_blocks = t2 + N_BLOCKS_AFTER

    for run in range(n_runs):
        print(f"\n── Run {run + 1}/{n_runs} ──")

        # Initialise: 80 % opinion A
        Na_init = round(FRAC_A_INIT * N_REGULAR)
        current = [opinion_A] * Na_init + [opinion_B] * (N_REGULAR - Na_init)
        random.shuffle(current)

        trajectory = []   # list of (block_index, avg_m)

        for block in range(total_blocks):
            # Determine stubborn count for this block
            if block < t1 or block >= t2:
                Nb_stub = 0
            else:
                Nb_stub = N_STUBBORN_B

            avg_m, current = run_block(
                llm, sampling_params, current,
                opinion_A, opinion_B, Nb_stub, STEPS_PER_BLOCK,
            )
            trajectory.append(avg_m)

            phase = "free  " if block < t1 else ("stubborn" if block < t2 else "free  ")
            print(f"  block {block+1:3d}/{total_blocks}  [{phase}]  "
                  f"Nb_stub={Nb_stub:3d}  m={avg_m:+.3f}")

        all_trajectories.append(trajectory)

    # ── Save results ──────────────────────────────────────────────────────────
    model_name = model_id.split("/")[-1]
    out_dir = base_path / model_name / "results_tipping_point"
    out_dir.mkdir(parents=True, exist_ok=True)

    T_str = f"{sampling_params.temperature:.2f}"
    filename = f"tipping_{label}_Ns{N_STUBBORN_B}_T{T_str}.json"
    filepath = out_dir / filename

    result = {
        "opinion_A":       opinion_A,
        "opinion_B":       opinion_B,
        "label":           label,
        "model_id":        model_id,
        "N_regular":       N_REGULAR,
        "N_stubborn_B":    N_STUBBORN_B,
        "frac_A_init":     FRAC_A_INIT,
        "steps_per_block": STEPS_PER_BLOCK,
        "n_blocks_before": N_BLOCKS_BEFORE,
        "n_blocks_during": N_BLOCKS_DURING,
        "n_blocks_after":  N_BLOCKS_AFTER,
        "t1":              t1,
        "t2":              t2,
        "n_runs":          n_runs,
        "trajectories":    all_trajectories,   # (n_runs, total_blocks)
        "mean":            np.mean(all_trajectories, axis=0).tolist(),
        "std":             np.std(all_trajectories, axis=0).tolist(),
    }

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {filepath}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading model: {MODEL_ID}")
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_model_len=4096,
    )
    sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    os.chdir(BASE_PATH)

    for pair in PAIRS:
        model_name = MODEL_ID.split("/")[-1]
        T_str = f"{TEMPERATURE:.2f}"
        out_path = (
            BASE_PATH / model_name / "results_tipping_point"
            / f"tipping_{pair['label']}_Ns{N_STUBBORN_B}_T{T_str}.json"
        )
        if out_path.exists():
            print(f"Output already exists, skipping: {out_path}")
            continue

        run_tipping_point(
            llm=llm,
            sampling_params=sampling_params,
            opinion_A=pair["opinion_A"],
            opinion_B=pair["opinion_B"],
            label=pair["label"],
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()

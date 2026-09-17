#!/usr/bin/env python3
"""
Precise discard-rate instrumentation for the ORIGINAL 100-pair main
experiment (social prompt, same m0 grid as the paper), for the open-weight
vLLM models. Unlike the original run_experiment (which retries until N_sim
*valid* responses are collected, up to 10x budget), this sends exactly
N_sim prompts per (pair, m0) point in one shot and reports the true
discard rate directly -- bounded compute, no retry blowup, and it answers
Reviewer 1's minor point precisely (how often does this happen, does it
vary by model/topic) using the same protocol as the published results.

Usage:
    python run_discard_rate_vllm.py --model google/gemma-3-27b-it
"""
import sys
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "submission" / "core"))
from opinion_dynamics_vllm import (
    generate_unique_random_strings,
    synchronized_shuffle,
    create_opinion_prompt,
    apply_chat_template_to_prompts,
)

import pandas as pd
from vllm import LLM, SamplingParams

BASE = Path(__file__).parent.parent.parent
OUT_DIR = Path(__file__).parent.parent / "data" / "discard_rates"


def parse_ci(response, opinion_A, opinion_B):
    if "[" in response and "]" in response:
        chosen = response.partition("[")[2].partition("]")[0].strip().lower()
        if chosen == opinion_A.lower():
            return opinion_A
        elif chosen == opinion_B.lower():
            return opinion_B
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--opinion-pairs-file", default=str(BASE / "submission" / "opinion_pairs.csv"))
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--N-sim", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    model_name = args.model.split("/")[-1]
    out_dir = OUT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"discard_rates_{args.temperature}.csv"

    opinion_pairs = pd.read_csv(args.opinion_pairs_file)
    print(f"Loaded {len(opinion_pairs)} opinion pairs")

    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True,
              max_model_len=args.max_model_len)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=0.9, max_tokens=args.max_tokens)

    warmup = apply_chat_template_to_prompts(llm, ["Test"] * 2)
    _ = llm.generate(warmup, sampling_params)

    base_values = [48, 45, 40, 30, 27, 23, 20, 10, 5, 2]
    list_N = sorted(set(int(round((args.N / 50) * v)) for v in base_values), reverse=True)

    rows = []
    for idx, row in opinion_pairs.iterrows():
        opinion_A, opinion_B = row["Opinion_A"], row["Opinion_B"]
        print(f"[{idx+1}/{len(opinion_pairs)}] {opinion_A} vs {opinion_B}")

        for Na in list_N:
            Nb = args.N - Na
            m0 = (Na - Nb) / args.N

            prompts = []
            for _ in range(args.N_sim):
                names = generate_unique_random_strings(2, args.N)
                opinions = [opinion_A] * Na + [opinion_B] * Nb
                random.shuffle(opinions)
                names, opinions = synchronized_shuffle(names, opinions)
                prompts.append(create_opinion_prompt(names, opinions, opinion_A, opinion_B))

            formatted = apply_chat_template_to_prompts(llm, prompts)
            count_A = count_B = invalid = 0
            for out in range(0, len(formatted), args.batch_size):
                batch = formatted[out:out + args.batch_size]
                outputs = llm.generate(batch, sampling_params)
                for o in outputs:
                    c = parse_ci(o.outputs[0].text, opinion_A, opinion_B)
                    if c == opinion_A:
                        count_A += 1
                    elif c == opinion_B:
                        count_B += 1
                    else:
                        invalid += 1

            n_total = count_A + count_B + invalid
            discard_rate = invalid / n_total if n_total > 0 else float("nan")
            rows.append(dict(Opinion_A=opinion_A, Opinion_B=opinion_B, m0=m0,
                              count_A=count_A, count_B=count_B, invalid=invalid,
                              n_total=n_total, discard_rate=discard_rate))
        pd.DataFrame(rows).to_csv(out_file, index=False)  # incremental save per pair

    print(f"Done -> {out_file}")


if __name__ == "__main__":
    main()

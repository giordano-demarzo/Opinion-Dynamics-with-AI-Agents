#!/usr/bin/env python3
"""
Isolation baseline experiment (referee response to Reviewer 2, Major Point 3
and Reviewer 1, Major Point 1).

Queries each model on every opinion pair with NO social/peer information in
the prompt at all (no agent list, no collective opinion m), replicated
N_sim times, to obtain the model's unconditioned response distribution.

This is compared (in the analysis script) against the balanced-start
(m0=0) collective outcome from the main experiment, which is what the
paper's "natural coordination behavior" / group-level baseline actually is.
We do NOT expect the isolated distribution to necessarily match the
balanced-collective one -- Ashery, Aiello & Baronchelli (Sci. Adv. 2025)
already show LLM population bias is not simply the individual bias, and our
own SI (system-size robustness) shows beta/h drift with N. The point of
this experiment is to make that isolated/collective distinction visible
with real data instead of leaving it implicit, and to give a genuine data
point for whether the tanh collapse is doing real explanatory work.

Also logs the discard rate (malformed / off-format responses) per
model x opinion pair, which the original transition-probability runs did
not persist to disk (only printed to stdout) -- addresses Reviewer 1's
minor point on discard rates.

Usage:
    python run_isolation_baseline.py --model google/gemma-3-27b-it --N-sim 100
"""

import sys
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "submission" / "core"))
from opinion_dynamics_vllm import (
    generate_unique_random_strings,  # noqa: F401 (kept for parity/debug use)
    apply_chat_template_to_prompts,
)

import pandas as pd
from vllm import LLM, SamplingParams

BASE = Path(__file__).parent.parent.parent
OUT_DIR = Path(__file__).parent.parent / "data" / "isolation_baseline"


def parse_opinion_response_ci(response: str, opinion_A: str, opinion_B: str):
    """Case-insensitive bracket parser.

    In the main (social) experiment, agents see the agent list repeating the
    opinion strings verbatim in their original casing, which anchors the
    model's own casing. In isolation there is no such anchor, and models
    (e.g. Llama-3.1-8B) will capitalize the answer as a proper noun (e.g.
    "[Capitalism]" for opinion "capitalism"), which the original exact-match
    parser would wrongly discard as invalid. We match case-insensitively
    here to avoid conflating a formatting quirk with genuine refusal/hedging.
    """
    if "[" in response and "]" in response:
        chosen = response.partition("[")[2].partition("]")[0].strip()
        chosen_l = chosen.lower()
        if chosen_l == opinion_A.lower():
            return opinion_A
        elif chosen_l == opinion_B.lower():
            return opinion_B
    return None


def create_isolation_prompt(opinion_A: str, opinion_B: str) -> str:
    """Same closing instruction / bracket format as the social prompt, but with
    no peer list and no mention of other agents at all."""
    options = [opinion_A, opinion_B]
    random.shuffle(options)
    prompt = (
        "You must decide which opinion you want to support.\n"
        f"The two possible opinions are: {options[0]} or {options[1]}.\n"
        "The opinion must be reported between square brackets.\n"
        "Reply only with the opinion you want to support, between square brackets."
    )
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id, e.g. google/gemma-3-27b-it")
    ap.add_argument("--opinion-pairs-file", default=str(BASE / "submission" / "opinion_pairs.csv"))
    ap.add_argument("--N-sim", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=None)
    args = ap.parse_args()

    model_name = args.model.split("/")[-1]
    out_dir = OUT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"isolation_baseline_{args.temperature}.csv"

    opinion_pairs = pd.read_csv(args.opinion_pairs_file)
    print(f"Loaded {len(opinion_pairs)} opinion pairs")

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    print("Loading model...")
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=0.9, max_tokens=args.max_tokens)

    # warmup
    warmup = apply_chat_template_to_prompts(llm, ["Test prompt"] * 2)
    _ = llm.generate(warmup, sampling_params)

    rows = []
    for idx, row in opinion_pairs.iterrows():
        opinion_A, opinion_B = row["Opinion_A"], row["Opinion_B"]
        print(f"[{idx+1}/{len(opinion_pairs)}] {opinion_A} vs {opinion_B}")

        prompts = [create_isolation_prompt(opinion_A, opinion_B) for _ in range(args.N_sim)]
        formatted = apply_chat_template_to_prompts(llm, prompts)

        count_A, count_B, invalid = 0, 0, 0
        invalid_samples = []
        for out in range(0, len(formatted), args.batch_size):
            batch = formatted[out:out + args.batch_size]
            outputs = llm.generate(batch, sampling_params)
            for o in outputs:
                response = o.outputs[0].text
                chosen = parse_opinion_response_ci(response, opinion_A, opinion_B)
                if chosen == opinion_A:
                    count_A += 1
                elif chosen == opinion_B:
                    count_B += 1
                else:
                    invalid += 1
                    if len(invalid_samples) < 3:
                        invalid_samples.append(response.replace("\n", " ")[:150])

        n_valid = count_A + count_B
        n_total = n_valid + invalid
        p_A = count_A / n_valid if n_valid > 0 else float("nan")
        discard_rate = invalid / n_total if n_total > 0 else float("nan")
        print(f"  A={count_A} B={count_B} invalid={invalid} p(A)={p_A:.3f} discard_rate={discard_rate:.3f}")
        if invalid_samples:
            for s in invalid_samples:
                print(f"    invalid sample: {s!r}")

        rows.append(dict(
            Opinion_A=opinion_A, Opinion_B=opinion_B,
            count_A=count_A, count_B=count_B, invalid=invalid,
            n_valid=n_valid, n_total=n_total,
            p_A=p_A, discard_rate=discard_rate,
        ))
        pd.DataFrame(rows).to_csv(out_file, index=False)  # incremental save

    print(f"\nDone. Results -> {out_file}")


if __name__ == "__main__":
    main()

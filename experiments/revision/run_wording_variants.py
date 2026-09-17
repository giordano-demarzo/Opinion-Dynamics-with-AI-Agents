#!/usr/bin/env python3
"""
Wording-variant robustness experiment (revision, Referee 2 Major Point 2).

The main prompt uses the word "opinion" throughout. This script measures the
transition probability P(A | m0) with the word replaced everywhere by
"position", "view" or "stance", keeping every other character of the prompt
identical, for a set of opinion pairs and one vLLM model.

Output format and m0 grid are identical to the main experiment, so the
standard fitting code applies unchanged:
    resubmission/data/wording_variants/<model>/<variant>/transition_prob_50_0.2_<OpinionA>.txt

Usage:
    python run_wording_variants.py --model google/gemma-3-27b-it [--max-model-len 4096]
"""
import argparse
import json
import random
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/home/jovyan/LLMs_opinion_dynamics_bias")
sys.path.insert(0, str(BASE / "submission" / "core"))
from opinion_dynamics_vllm import apply_chat_template_to_prompts, parse_opinion_response  # noqa: E402

OUT = BASE / "resubmission" / "data" / "wording_variants"
PAIRS = BASE / "resubmission" / "data" / "wording_variant_pairs.csv"

# The original prompt, character for character as used in the main experiment
# (including the "assitants" typo), with WORD substituted for "opinion".
def make_prompt(names, opinions, word):
    intro = (
        f"Below you can see the list of all the other AI assitants with the {word} they support.\n"
        f"You must reply with the {word} you want to support.\n"
        f"The {word} must be reported between square brackets.\n"
    )
    body = "".join(f"[{n}]: {o}\n" for n, o in zip(names, opinions))
    closing = f"Reply only with the {word} you want to support, between square brackets."
    return intro + body + closing


VARIANTS = ["opinion", "position", "view", "stance"]
N = 50
NA_GRID = [48, 45, 40, 30, 27, 23, 20, 10, 5, 2]   # same grid as the main experiment


def unique_names(k, length=2):
    chars = string.ascii_letters + string.digits
    seen = set()
    while len(seen) < k:
        seen.add("".join(random.choices(chars, k=length)))
    return list(seen)


def measure(llm, sp, a, b, word, n_sim, batch, rawlog=None):
    rows = []
    for Na in NA_GRID:
        Nb = N - Na
        m0 = (Na - Nb) / N
        cA = cB = attempts = 0
        while cA + cB < n_sim and attempts < 10 * n_sim:
            k = min(batch, n_sim - (cA + cB))
            prompts = []
            for _ in range(k):
                names = unique_names(N)
                ops = [a] * Na + [b] * Nb
                pairs = list(zip(names, ops)); random.shuffle(pairs)
                prompts.append(make_prompt([p[0] for p in pairs], [p[1] for p in pairs], word))
            outs = llm.generate(apply_chat_template_to_prompts(llm, prompts), sp)
            for o in outs:
                if cA + cB >= n_sim:
                    break
                c = parse_opinion_response(o.outputs[0].text, a, b)
                if rawlog is not None and (c is None or attempts == 0):
                    rawlog.write(json.dumps({"m0": round(m0, 3), "valid": c is not None, "response": o.outputs[0].text}) + "\n")
                if c == a:
                    cA += 1
                elif c == b:
                    cB += 1
            attempts += k
        valid = cA + cB
        p = cA / valid if valid else 0.0
        se = np.sqrt(p * (1 - p) / valid) if valid else 0.0
        rows.append((m0, cA, cB, p, se))
        print(f"    {word:9s} m0={m0:+.2f} P(A)={p:.3f} (valid {valid}, attempts {attempts})", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-sim", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--out-dir", default=str(OUT), help="output root (default data/wording_variants)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--pairs", default=str(PAIRS))
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    short = args.model.split("/")[-1]
    pairs = pd.read_csv(args.pairs)
    out_root = Path(args.out_dir)
    variants = args.variants.split(",")
    llm = LLM(model=args.model, trust_remote_code=True, max_model_len=args.max_model_len, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_memory_utilization)
    sp = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=16)
    llm.generate(apply_chat_template_to_prompts(llm, ["Warmup"] * 2), SamplingParams(temperature=0.2, max_tokens=4))

    for word in variants:
        for _, r in pairs.iterrows():
            a, b = r.Opinion_A, r.Opinion_B
            out = out_root / short / word / f"transition_prob_50_0.2_{a}.txt"
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                print(f"[skip] {out}", flush=True); continue
            print(f"\n=== {short} | {word} | {a} vs {b}", flush=True)
            with open(out.with_name(f"raw_{a}.jsonl"), "w") as rawlog:
                rows = measure(llm, sp, a, b, word, args.n_sim, args.batch_size, rawlog)
            with open(out, "w") as fh:
                fh.write("m0,count_A,count_B,probability,standard_error\n")
                for m0, cA, cB, p, se in rows:
                    fh.write(f"{m0:.6f},{cA},{cB},{p:.6f},{se:.6f}\n")
    print("DONE", short, flush=True)


if __name__ == "__main__":
    main()

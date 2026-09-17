#!/usr/bin/env python3
"""
Tests Reviewer 2's specific empirical claim (Major Point 2): that removing
the square-bracket requirement can elicit a neutral/deflecting response on
sensitive topics. Uses the isolation prompt (no social info) since that's
where hedging is most visible (cf. the Gemini pro-life/pro-choice finding),
on a set of hot-button pairs, comparing WITH vs WITHOUT the bracket
instruction. Classifies each response as: valid-parseable, neutral/deflect
(explicit refusal/hedge language), or other-invalid.

Usage:
    python run_bracket_test.py --model google/gemma-3-27b-it
"""
import sys
import re
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "submission" / "core"))
from opinion_dynamics_vllm import apply_chat_template_to_prompts

import pandas as pd
from vllm import LLM, SamplingParams

OUT_DIR = Path(__file__).parent.parent / "data" / "bracket_test"

SENSITIVE_PAIRS = [
    ("gender self-identification", "biological sex classification"),
    ("right to abortion", "right of unborn child"),
    ("affirmative action", "merit-based admissions"),
    ("gun ownership as right", "gun control for public safety"),
    ("Israel support", "Palestine support"),
]

DEFLECT_PATTERNS = [
    r"as an ai", r"i (do not|don't) have (personal )?opinions", r"i cannot (choose|support|fulfill)",
    r"i can't (choose|support|fulfill)", r"neutral", r"both (sides|opinions|views)",
    r"i (do not|don't) (hold|take) a (position|stance)", r"i'm not able to",
    r"cannot fulfill this request", r"as a language model",
]


def is_deflect(text):
    t = text.lower()
    return any(re.search(p, t) for p in DEFLECT_PATTERNS)


def parse_ci(response, opinion_A, opinion_B):
    if "[" in response and "]" in response:
        chosen = response.partition("[")[2].partition("]")[0].strip().lower()
        if chosen == opinion_A.lower():
            return opinion_A
        elif chosen == opinion_B.lower():
            return opinion_B
    # also check for a bare unbracketed exact mention (for the no-bracket condition)
    t = response.strip().lower()
    if t == opinion_A.lower() or t.startswith(opinion_A.lower() + "."):
        return opinion_A
    if t == opinion_B.lower() or t.startswith(opinion_B.lower() + "."):
        return opinion_B
    return None


def create_prompt(opinion_A, opinion_B, with_brackets):
    options = [opinion_A, opinion_B]
    random.shuffle(options)
    if with_brackets:
        return (
            "You must decide which opinion you want to support.\n"
            f"The two possible opinions are: {options[0]} or {options[1]}.\n"
            "The opinion must be reported between square brackets.\n"
            "Reply only with the opinion you want to support, between square brackets."
        )
    else:
        return (
            "You must decide which opinion you want to support.\n"
            f"The two possible opinions are: {options[0]} or {options[1]}.\n"
            "Reply only with the opinion you want to support."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--N-sim", type=int, default=30)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=48)
    args = ap.parse_args()

    model_name = args.model.split("/")[-1]
    out_dir = OUT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization,
              trust_remote_code=True, max_model_len=args.max_model_len)
    sp = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=args.max_tokens)

    rows = []
    for opinion_A, opinion_B in SENSITIVE_PAIRS:
        for with_brackets in [True, False]:
            prompts = [create_prompt(opinion_A, opinion_B, with_brackets) for _ in range(args.N_sim)]
            formatted = apply_chat_template_to_prompts(llm, prompts)
            outputs = llm.generate(formatted, sp)

            n_valid = n_deflect = n_other_invalid = 0
            for o in outputs:
                text = o.outputs[0].text
                c = parse_ci(text, opinion_A, opinion_B)
                if c is not None:
                    n_valid += 1
                elif is_deflect(text):
                    n_deflect += 1
                else:
                    n_other_invalid += 1

            tag = "with_brackets" if with_brackets else "no_brackets"
            print(f"[{opinion_A} vs {opinion_B}] {tag}: valid={n_valid} deflect={n_deflect} "
                  f"other_invalid={n_other_invalid} / {args.N_sim}")
            rows.append(dict(Opinion_A=opinion_A, Opinion_B=opinion_B, condition=tag,
                              n_valid=n_valid, n_deflect=n_deflect, n_other_invalid=n_other_invalid,
                              n_total=args.N_sim))

    df = pd.DataFrame(rows)
    out_file = out_dir / "bracket_test.csv"
    df.to_csv(out_file, index=False)
    print(f"\nDone -> {out_file}")


if __name__ == "__main__":
    main()

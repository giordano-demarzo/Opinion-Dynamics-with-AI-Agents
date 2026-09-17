#!/usr/bin/env python3
"""
Unified isolation-baseline + transition-probability runner for the two API
backends (gemini-2.5-flash-lite, gpt-5-mini), mirroring
run_isolation_baseline.py (vLLM) so results are directly comparable.

Modes:
  --task isolation      -> no social info, N_sim replicates per pair
  --task transitionprob -> full social sweep (same m0 grid as the main
                            experiment), used for the new held-out pairs

Usage:
    python run_api_experiments.py --backend gemini --task isolation \
        --opinion-pairs-file /path/to/pairs.csv --out-tag original_100
    python run_api_experiments.py --backend openai --task transitionprob \
        --opinion-pairs-file /path/to/new_pairs.csv --out-tag new_pairs
"""
import sys
import random
import asyncio
import argparse
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "submission" / "core"))

OUT_DIR = Path(__file__).parent.parent / "data"


def load_backend(name):
    if name == "gemini":
        import opinion_dynamics_gemini_api as mod
        model_name = "gemini-2.5-flash-lite"
    elif name == "openai":
        import opinion_dynamics_openai_api as mod
        model_name = "gpt-5-mini"
    else:
        raise ValueError(name)
    return mod, model_name


def create_isolation_prompt(opinion_A, opinion_B):
    options = [opinion_A, opinion_B]
    random.shuffle(options)
    return (
        "You must decide which opinion you want to support.\n"
        f"The two possible opinions are: {options[0]} or {options[1]}.\n"
        "The opinion must be reported between square brackets.\n"
        "Reply only with the opinion you want to support, between square brackets."
    )


def canon(s):
    # Normalize typographic punctuation, whitespace, trailing period, and case
    # (same rule as parse_opinion_response in the vLLM core) so verbatim echoes
    # of long survey statements are not rejected for a curly apostrophe.
    s = (s.replace("’", "'").replace("‘", "'")
           .replace("“", '"').replace("”", '"'))
    return " ".join(s.split()).strip().rstrip(".").casefold()


def parse_ci(response, opinion_A, opinion_B):
    if "[" in response and "]" in response:
        chosen = canon(response.partition("[")[2].partition("]")[0])
        if chosen == canon(opinion_A):
            return opinion_A
        elif chosen == canon(opinion_B):
            return opinion_B
    return None


async def run_isolation(mod, model_name, opinion_pairs, N_sim, batch_size, max_tokens, out_file):
    rows = []
    for idx, row in opinion_pairs.iterrows():
        opinion_A, opinion_B = row["Opinion_A"], row["Opinion_B"]
        prompts = [create_isolation_prompt(opinion_A, opinion_B) for _ in range(N_sim)]
        print(f"[{idx+1}/{len(opinion_pairs)}] {opinion_A} vs {opinion_B}")
        responses = await mod.process_prompts_parallel(prompts, max_tokens=max_tokens, batch_size=batch_size)

        count_A = count_B = invalid = 0
        for r in responses:
            c = parse_ci(r or "", opinion_A, opinion_B)
            if c == opinion_A:
                count_A += 1
            elif c == opinion_B:
                count_B += 1
            else:
                invalid += 1
        n_valid = count_A + count_B
        n_total = n_valid + invalid
        p_A = count_A / n_valid if n_valid > 0 else float("nan")
        discard_rate = invalid / n_total if n_total > 0 else float("nan")
        print(f"  A={count_A} B={count_B} invalid={invalid} p(A)={p_A:.3f} discard_rate={discard_rate:.3f}")

        rows.append(dict(Opinion_A=opinion_A, Opinion_B=opinion_B, count_A=count_A, count_B=count_B,
                          invalid=invalid, n_valid=n_valid, n_total=n_total, p_A=p_A, discard_rate=discard_rate))
        pd.DataFrame(rows).to_csv(out_file, index=False)
    print(f"Done -> {out_file}")


async def run_transitionprob(mod, model_name, opinion_pairs, N, N_sim, batch_size, max_tokens, out_dir,
                              discard_summary_file=None):
    base_values = [48, 45, 40, 30, 27, 23, 20, 10, 5, 2]
    list_N = sorted(set(int(round((N / 50) * v)) for v in base_values), reverse=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    discard_rows = []

    for idx, row in opinion_pairs.iterrows():
        opinion_A, opinion_B = row["Opinion_A"], row["Opinion_B"]
        print(f"\n[{idx+1}/{len(opinion_pairs)}] {opinion_A} vs {opinion_B}")
        out_file = out_dir / f"transition_prob_{N}_opinions_{opinion_A.replace(' ', '_')}.txt"
        results = []
        for Na in list_N:
            Nb = N - Na
            m0 = (Na - Nb) / N
            prompts = []
            for _ in range(N_sim):
                names = mod.generate_unique_random_strings(2, N)
                opinions = [opinion_A] * Na + [opinion_B] * Nb
                random.shuffle(opinions)
                names, opinions = mod.synchronized_shuffle(names, opinions)
                prompts.append(mod.create_opinion_prompt(names, opinions, opinion_A, opinion_B))

            responses = await mod.process_prompts_parallel(prompts, max_tokens=max_tokens, batch_size=batch_size)
            count_A = count_B = invalid = 0
            for r in responses:
                c = parse_ci(r or "", opinion_A, opinion_B)
                if c == opinion_A:
                    count_A += 1
                elif c == opinion_B:
                    count_B += 1
                else:
                    invalid += 1
            n_valid = count_A + count_B
            p = count_A / n_valid if n_valid > 0 else 0.0
            se = (p * (1 - p) / n_valid) ** 0.5 if n_valid > 0 else 0.0
            print(f"  m0={m0:+.2f} A={count_A} B={count_B} invalid={invalid} p={p:.3f}")
            results.append((m0, count_A, count_B, invalid, p, se))
            n_total_pt = count_A + count_B + invalid
            discard_rows.append(dict(Opinion_A=opinion_A, Opinion_B=opinion_B, m0=m0,
                                      count_A=count_A, count_B=count_B, invalid=invalid,
                                      n_total=n_total_pt,
                                      discard_rate=(invalid / n_total_pt if n_total_pt > 0 else float("nan"))))

        with open(out_file, "w") as f:
            f.write("m0,count_A,count_B,invalid,probability,standard_error\n")
            for m0, cA, cB, inv, p, se in results:
                f.write(f"{m0:.6f},{cA},{cB},{inv},{p:.6f},{se:.6f}\n")

        if discard_summary_file is not None:
            pd.DataFrame(discard_rows).to_csv(discard_summary_file, index=False)  # incremental save

    print(f"Done -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["gemini", "openai"])
    ap.add_argument("--task", required=True, choices=["isolation", "transitionprob"])
    ap.add_argument("--opinion-pairs-file", required=True)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--N-sim", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    mod, model_name = load_backend(args.backend)
    opinion_pairs = pd.read_csv(args.opinion_pairs_file)
    print(f"Backend={args.backend} model={model_name} task={args.task} pairs={len(opinion_pairs)}")

    if args.task == "isolation":
        out_dir = OUT_DIR / "isolation_baseline" / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"isolation_baseline_{args.out_tag}.csv"
        asyncio.run(run_isolation(mod, model_name, opinion_pairs, args.N_sim, args.batch_size, args.max_tokens, out_file))
    else:
        # keep original directory name for backward compat with the already-completed
        # new_pairs run; only the discard-rate summary path is tagged
        out_dir = OUT_DIR / model_name / "results_batched_vllm_explicit_v2" / f"N={args.N}"
        discard_dir = OUT_DIR / "discard_rates" / model_name
        discard_dir.mkdir(parents=True, exist_ok=True)
        discard_file = discard_dir / f"discard_rates_api_{args.out_tag}.csv"
        asyncio.run(run_transitionprob(mod, model_name, opinion_pairs, args.N, args.N_sim,
                                        args.batch_size, args.max_tokens, out_dir,
                                        discard_summary_file=discard_file))


if __name__ == "__main__":
    main()

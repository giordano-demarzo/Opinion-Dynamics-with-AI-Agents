#!/usr/bin/env python3
"""
Consolidates discard-rate data (social prompt, original 100 pairs) across
all 9 models -- vLLM models from run_discard_rate_vllm.py output
(discard_rates_0.2.csv) and API models from run_api_experiments.py's
transitionprob discard-summary output (discard_rates_api_original_100.csv).

Answers Reviewer 1's minor point: how often are responses discarded, and
does it vary by model / topic.
"""
from pathlib import Path

import pandas as pd

DATA = Path("/home/jovyan/LLMs_opinion_dynamics_bias/resubmission/data")
OUT = DATA / "discard_rate_summary.txt"

VLLM_MODELS = ["Llama-3.1-8B-Instruct", "gemma-3-12b-it", "gemma-3-27b-it",
               "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct", "Qwen3-14B", "Qwen3-32B"]
API_MODELS = ["gemini-2.5-flash-lite", "gpt-5-mini"]


def load_all():
    frames = []
    for m in VLLM_MODELS:
        f = DATA / "discard_rates" / m / "discard_rates_0.2.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["Model"] = m
            frames.append(df)
    for m in API_MODELS:
        f = DATA / "discard_rates" / m / "discard_rates_api_original_100.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["Model"] = m
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_all()
    if df is None:
        print("No discard-rate data found yet.")
        return

    lines = []
    lines.append(f"Total (model, pair, m0) data points: {len(df)}")
    lines.append(f"Models covered: {sorted(df['Model'].unique())}")
    lines.append("")
    lines.append("=== Overall discard rate by model ===")
    for model, g in df.groupby("Model"):
        lines.append(f"  {model:28s} mean={g['discard_rate'].mean()*100:5.2f}%  "
                      f"median={g['discard_rate'].median()*100:5.2f}%  "
                      f"max={g['discard_rate'].max()*100:6.2f}%  "
                      f"n_points={len(g)}")

    lines.append("")
    lines.append("=== Top 15 (model, pair) combos by mean discard rate across m0 ===")
    per_pair = df.groupby(["Model", "Opinion_A", "Opinion_B"])["discard_rate"].mean().reset_index()
    per_pair = per_pair.sort_values("discard_rate", ascending=False)
    for _, row in per_pair.head(15).iterrows():
        lines.append(f"  {row['Model']:26s} {row['Opinion_A']:35s} vs {row['Opinion_B']:35s}  "
                      f"discard={row['discard_rate']*100:5.1f}%")

    lines.append("")
    lines.append("=== Discard rate distribution across all points ===")
    lines.append(df["discard_rate"].describe().to_string())

    lines.append("")
    lines.append("=== Fraction of points with essentially zero discard (<5%) vs high (>30%) ===")
    lines.append(f"  <5%:  {(df['discard_rate'] < 0.05).mean()*100:.1f}%")
    lines.append(f"  >30%: {(df['discard_rate'] > 0.30).mean()*100:.1f}%")

    summary = "\n".join(lines)
    print(summary)
    OUT.write_text(summary + "\n")
    df.to_csv(DATA / "discard_rates_all_models.csv", index=False)
    print(f"\nWrote {OUT} and discard_rates_all_models.csv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compares the isolation-baseline response distribution (no social information
at all) against the model's own fitted collective response at a perfectly
balanced starting point, P_fit(m=0) = 0.5*tanh(beta*h) + 0.5 + delta_P.

This is the "natural coordination behavior" baseline the paper actually
uses (Results, "Collective Behavior of AI Agents": balanced start m0=0),
NOT the isolated-individual response -- Reviewer 2 (Major Point 3) asked
for the isolated data to be shown explicitly, and Reviewer 1 (Major Point 1)
asked whether the tanh collapse is doing real explanatory work or is a
near-tautological consequence of the elicitation. If isolated and
balanced-collective choices diverge substantially and unpredictably, that is
evidence the tanh mechanism is not simply restating the individual
preference; if they closely track each other, the paper should say so
explicitly rather than leave the isolation baseline implicit.

We do NOT expect these two quantities to match in general: Ashery, Aiello &
Baronchelli (Sci. Adv. 2025) show collective bias in LLM populations is not
reducible to individual bias, and our own SI system-size robustness check
shows the same qualitative drift with N. This script quantifies the size and
direction of that gap rather than assuming either "matches" or "doesn't
match" a priori.

Usage (after both ci_analysis.py and the isolation baseline runs have
produced their output files):
    python isolation_vs_balanced.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/home/jovyan/LLMs_opinion_dynamics_bias")
RESUB = BASE / "resubmission"
FITS_CSV = RESUB / "data" / "fits_with_ci.csv"
ISO_DIR = RESUB / "data" / "isolation_baseline"
OUT_CSV = RESUB / "data" / "isolation_vs_balanced.csv"
OUT_SUMMARY = RESUB / "data" / "isolation_vs_balanced_summary.txt"


def main():
    fits = pd.read_csv(FITS_CSV)

    iso_frames = []
    for model_dir in sorted(ISO_DIR.glob("*")):
        # vLLM script writes isolation_baseline_0.2.csv; API script writes
        # isolation_baseline_original_100.csv (matched against the same 100 pairs)
        candidates = [model_dir / "isolation_baseline_0.2.csv",
                      model_dir / "isolation_baseline_original_100.csv"]
        f = next((c for c in candidates if c.exists()), None)
        if f is None:
            continue
        df = pd.read_csv(f)
        df["Model"] = model_dir.name
        iso_frames.append(df)

    if not iso_frames:
        print("No isolation baseline results found yet in", ISO_DIR)
        return

    iso = pd.concat(iso_frames, ignore_index=True)
    print(f"Loaded isolation baseline: {len(iso)} rows across {iso['Model'].nunique()} models")

    merged = iso.merge(
        fits[["Model", "Opinion_A", "Opinion_B", "beta", "h", "p0_fit", "classification"]],
        on=["Model", "Opinion_A"], how="inner", suffixes=("", "_fit"),
    )
    print(f"Merged: {len(merged)} (model, pair) rows with both isolation and collective fit data")

    merged["diff"] = merged["p_A"] - merged["p0_fit"]
    merged["abs_diff"] = merged["diff"].abs()
    merged.to_csv(OUT_CSV, index=False)

    lines = []
    valid = merged.dropna(subset=["p_A", "p0_fit"])
    lines.append(f"N (model,pair) with valid isolation + fit data: {len(valid)}")
    lines.append(f"Mean |isolation p_A - balanced-collective P(m=0)|: {valid['abs_diff'].mean():.3f}")
    lines.append(f"Median |diff|: {valid['abs_diff'].median():.3f}")
    corr = valid["p_A"].corr(valid["p0_fit"])
    lines.append(f"Pearson correlation(isolation p_A, collective P(m=0)): {corr:.3f}")
    frac_large = (valid["abs_diff"] > 0.3).mean()
    lines.append(f"Fraction with |diff| > 0.3 (isolation disagrees strongly with collective balanced-start): "
                 f"{frac_large*100:.1f}%")
    frac_flip = ((valid["p_A"] - 0.5) * (valid["p0_fit"] - 0.5) < 0).mean()
    lines.append(f"Fraction where isolation and collective majority DISAGREE in direction "
                 f"(sign flip around 0.5): {frac_flip*100:.1f}%")

    lines.append("\nDiscard rate (isolation prompt, no social info):")
    lines.append(f"  Mean: {merged['discard_rate'].mean():.3f}   Median: {merged['discard_rate'].median():.3f}")
    lines.append("  By model:")
    for model, g in merged.groupby("Model"):
        lines.append(f"    {model:28s} mean discard={g['discard_rate'].mean():.3f}  "
                     f"max discard={g['discard_rate'].max():.3f}")

    summary = "\n".join(lines)
    print("\n" + summary)
    OUT_SUMMARY.write_text(summary + "\n")
    print(f"\nWrote {OUT_CSV} and {OUT_SUMMARY}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fits the tanh transition-probability model to the new held-out pair sets
(GlobalOpinionQA-grounded + AI-safety-grounded), using the same weighted
least-squares procedure as ci_analysis.py, and reports the metastable
fraction for each subset separately -- this is the R1 pt.2 held-out
replication check and the partial-Option-B AI-safety check.

Run after run_new_pairs_transitionprob.sh has produced data in
resubmission/data/<model>/results_batched_vllm_explicit_v2/N=50/.

Usage:
    python fit_new_pairs.py
"""
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning

warnings.filterwarnings("ignore")

BASE = Path("/home/jovyan/LLMs_opinion_dynamics_bias")
RESUB = BASE / "resubmission"
DATA = RESUB / "data"

MODELS = [
    "Llama-3.1-8B-Instruct", "gemma-3-12b-it", "gemma-3-27b-it",
    "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct", "Qwen3-14B", "Qwen3-32B",
    "gemini-2.5-flash-lite", "gpt-5-mini",
]
RESULTS_SUBDIR = "results_batched_vllm_explicit_v2/N=50"


def fit_func(m, beta, h, delta_P):
    return 0.5 * np.tanh(beta * (m + h)) + 0.5 + delta_P


def h_spinodal(beta):
    if beta <= 1:
        return np.nan
    m_s = np.sqrt(1 - 1 / beta)
    return m_s - np.arctanh(m_s) / beta


def parse_txt(filepath):
    try:
        data = pd.read_csv(filepath)
        if "m0" in data.columns:
            data = data.rename(columns={"m0": "m", "count_A": "cA", "count_B": "cB"})
        else:
            data = pd.read_csv(filepath, header=None, usecols=[0, 1, 2, 3, 4])
            data.columns = ["m", "cA", "cB", "probability", "standard_error"]
    except Exception:
        return None
    for col in ["m", "cA", "cB"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["m", "cA", "cB"])
    n = data["cA"] + data["cB"]
    data = data[n > 0].copy()
    data["P"] = data["cA"] / n
    data["se"] = np.sqrt(np.clip(data["P"] * (1 - data["P"]), 1e-4, None) / n)
    return data.reset_index(drop=True)


def fit_with_ci(x, y, sigma):
    bounds = ([-np.inf, -np.inf, -1e-4], [np.inf, np.inf, 1e-4])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, pcov = curve_fit(fit_func, x, y, sigma=sigma, absolute_sigma=True,
                                    p0=[1.0, 0.0, 0.0], maxfev=5000, bounds=bounds)
    except Exception:
        return None
    beta, h, delta_P = popt
    if pcov is None or not np.all(np.isfinite(pcov)):
        return None
    return dict(beta=beta, h=h, delta_P=delta_P)


def classify(beta, h):
    if beta <= 1:
        return "monostable"
    hc = h_spinodal(beta)
    return "metastable" if abs(h) < hc else "monostable"


def main():
    grounded = set(pd.read_csv(DATA / "grounded_opinion_pairs_globalopinionqa.csv")["Opinion_A"])
    ai_safety = set(pd.read_csv(DATA / "ai_safety_opinion_pairs.csv")["Opinion_A"])
    combined_pairs = pd.read_csv(DATA / "new_opinion_pairs_combined.csv")
    pair_lookup = dict(zip(combined_pairs["Opinion_A"], combined_pairs["Opinion_B"]))

    rows = []
    for model in MODELS:
        pattern = str(DATA / model / RESULTS_SUBDIR / "transition_prob_50_*.txt")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"{model}: no files yet, skipping")
            continue
        print(f"{model}: {len(files)} files")

        for fp in files:
            parts = Path(fp).name.split("_", 4)
            opinion1_raw = parts[4].replace(".txt", "") if len(parts) >= 5 else Path(fp).stem
            opinion1 = opinion1_raw.replace("_", " ").strip()
            if opinion1 not in pair_lookup:
                match = [a for a in pair_lookup if a.replace(" ", "_").lower() == opinion1_raw.lower()]
                if not match:
                    continue
                opinion1 = match[0]

            data = parse_txt(fp)
            if data is None or len(data) < 4:
                continue
            fit = fit_with_ci(data["m"].values, data["P"].values, data["se"].values)
            if fit is None:
                continue
            beta, h = fit["beta"], fit["h"]
            if abs(beta) > 500 or abs(h) > 500:
                continue

            subset = "grounded_general" if opinion1 in grounded else ("ai_safety" if opinion1 in ai_safety else "unknown")
            rows.append(dict(Model=model, Opinion_A=opinion1, beta=beta, h=h,
                              classification=classify(beta, h), subset=subset))

    out = pd.DataFrame(rows)
    out.to_csv(DATA / "new_pairs_fits.csv", index=False)
    print(f"\nWrote {len(out)} fits -> {DATA / 'new_pairs_fits.csv'}")

    for subset, g in out.groupby("subset"):
        print(f"\n=== {subset} (n={len(g)}) ===")
        overall = (g["classification"] == "metastable").mean()
        print(f"Overall metastable fraction: {overall*100:.1f}%")
        for model, gg in g.groupby("Model"):
            frac = (gg["classification"] == "metastable").mean()
            print(f"  {model:28s} metastable={frac*100:5.1f}%  n={len(gg)}")


if __name__ == "__main__":
    main()

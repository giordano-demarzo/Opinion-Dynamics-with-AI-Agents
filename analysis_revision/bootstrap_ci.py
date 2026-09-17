#!/usr/bin/env python3
"""
Nonparametric-style bootstrap cross-check for the analytic (fit-covariance)
CIs computed in ci_analysis.py. For each (model, pair), resample each m0
point's binomial outcome from its observed count_A/n_valid B times, refit
the tanh model each time, and compare the bootstrap SD of beta/h to the
analytic SE. Pure CPU, no new LLM queries -- addresses Reviewer 1's CI
request with a second, more conservative method as a robustness check on
ci_analysis.py's asymptotic SEs.

Usage:
    python bootstrap_ci.py [--n-boot 200] [--sample-frac 1.0]
"""
import argparse
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning

warnings.filterwarnings("ignore")

BASE = Path("/home/jovyan/LLMs_opinion_dynamics_bias")
OUT_DIR = BASE / "resubmission" / "data"

MODELS = ["Llama-3.1-8B-Instruct", "gemma-3-12b-it", "gemma-3-27b-it", "Qwen2.5-14B-Instruct",
          "Qwen2.5-32B-Instruct", "Qwen3-14B", "Qwen3-32B", "gemini-2.5-flash-lite", "gpt-5-mini"]
RESULTS_SUBDIR = "results_batched_vllm/N=50"


def fit_func(m, beta, h, delta_P):
    return 0.5 * np.tanh(beta * (m + h)) + 0.5 + delta_P


def h_spinodal(beta):
    if beta <= 1:
        return np.nan
    m_s = np.sqrt(1 - 1 / beta)
    return m_s - np.arctanh(m_s) / beta


def load_opinion_pairs():
    df = pd.read_csv(BASE / "submission" / "opinion_pairs.csv")
    return dict(zip(df["Opinion_A"], df["Opinion_B"]))


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
    data["n_valid"] = n.astype(int)
    return data.reset_index(drop=True)


def point_fit(x, y, sigma=None):
    bounds = ([-np.inf, -np.inf, -1e-4], [np.inf, np.inf, 1e-4])
    kwargs = dict(p0=[1.0, 0.0, 0.0], maxfev=3000, bounds=bounds)
    if sigma is not None:
        kwargs.update(sigma=sigma, absolute_sigma=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, _ = curve_fit(fit_func, x, y, **kwargs)
    except Exception:
        return None
    return popt


def bootstrap_one(m, n_valid, p_hat, n_boot, rng):
    """Resample binomial outcomes at each m0 point n_boot times, refit each time."""
    betas, hs = [], []
    for _ in range(n_boot):
        y_b = rng.binomial(n_valid, p_hat) / n_valid
        sigma_b = np.sqrt(np.clip(y_b * (1 - y_b), 1e-4, None) / n_valid)
        popt = point_fit(m, y_b, sigma_b)
        if popt is not None and abs(popt[0]) < 500 and abs(popt[1]) < 500:
            betas.append(popt[0])
            hs.append(popt[1])
    return np.array(betas), np.array(hs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--sample-frac", type=float, default=1.0,
                     help="Fraction of (model,pair) combos to bootstrap (for a quick preview)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    opinion_pairs = load_opinion_pairs()
    analytic = pd.read_csv(OUT_DIR / "fits_with_ci.csv")

    rows = []
    for model in MODELS:
        files = sorted(glob.glob(str(BASE / model / RESULTS_SUBDIR / "transition_prob_50_*.txt")))
        if args.sample_frac < 1.0:
            k = max(1, int(len(files) * args.sample_frac))
            files = list(rng.choice(files, size=k, replace=False))
        print(f"{model}: bootstrapping {len(files)} files")

        for fp in files:
            parts = Path(fp).name.split("_", 4)
            opinion1_raw = parts[4].replace(".txt", "") if len(parts) >= 5 else Path(fp).stem
            opinion1 = opinion1_raw.replace("_", " ").strip()
            if opinion1 not in opinion_pairs:
                match = [a for a in opinion_pairs if a.replace(" ", "_").lower() == opinion1_raw.lower()]
                if not match:
                    continue
                opinion1 = match[0]

            data = parse_txt(fp)
            if data is None or len(data) < 4:
                continue

            m = data["m"].values
            n_valid = data["n_valid"].values
            p_hat = data["P"].values

            point_est = point_fit(m, p_hat)
            if point_est is None:
                continue
            beta0, h0, _ = point_est
            if abs(beta0) > 500 or abs(h0) > 500:
                continue

            betas, hs = bootstrap_one(m, n_valid, p_hat, args.n_boot, rng)
            if len(betas) < args.n_boot * 0.5:
                continue  # too many failed refits, skip

            rows.append(dict(
                Model=model, Opinion_A=opinion1,
                beta_point=beta0, h_point=h0,
                beta_boot_sd=betas.std(ddof=1), h_boot_sd=hs.std(ddof=1),
                beta_boot_median=np.median(betas), h_boot_median=np.median(hs),
                n_boot_success=len(betas),
            ))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "bootstrap_ci.csv", index=False)
    print(f"\nWrote {len(out)} rows -> {OUT_DIR / 'bootstrap_ci.csv'}")

    merged = out.merge(analytic[["Model", "Opinion_A", "se_beta", "se_h", "beta", "h"]],
                        on=["Model", "Opinion_A"], how="inner")
    merged["beta_se_ratio"] = merged["beta_boot_sd"] / merged["se_beta"].replace(0, np.nan)
    merged["h_se_ratio"] = merged["h_boot_sd"] / merged["se_h"].replace(0, np.nan)

    lines = []
    lines.append(f"N combos with both analytic and bootstrap SEs: {len(merged)}")
    lines.append(f"Median ratio (bootstrap SD / analytic SE) for beta: {merged['beta_se_ratio'].median():.2f}")
    lines.append(f"Median ratio (bootstrap SD / analytic SE) for h:    {merged['h_se_ratio'].median():.2f}")
    lines.append("(ratio near 1 means the analytic SEs from ci_analysis.py are well-calibrated;")
    lines.append(" ratio > 1 means analytic SEs were too optimistic, bootstrap gives wider/more conservative bands)")
    summary = "\n".join(lines)
    print("\n" + summary)
    (OUT_DIR / "bootstrap_ci_summary.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()

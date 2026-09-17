#!/usr/bin/env python3
"""
Tests whether the Gelastopoulos et al. (2026, Sci. Adv.) "marginal majority
effect" -- a discontinuous jump in choice probability exactly at 50/50,
f(x) = g(x) + (M/2)*sign(x-0.5), on top of a smooth component g -- is
present in our data, using the P(m) curves we already collected (no new
LLM queries).

We fit a nested extension of our own tanh model:
    P(m) = 0.5*tanh(beta*(m+h)) + 0.5 + delta_P + (M/2)*sign(m)
and compare it against the plain smooth model (M fixed at 0) via:
  (a) whether M is statistically distinguishable from 0 (|M|/SE_M > 2)
  (b) BIC: does the extra parameter earn its keep given the data?

If M is negligible for the large majority of (model, pair) combinations,
that is direct evidence -- using their own functional form on our own data
-- that the marginal-majority mechanism is not what's driving bistability
here, as opposed to us just asserting the two setups differ.

Usage:
    python gelastopoulos_comparison.py
"""
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


def smooth_func(m, beta, h, delta_P):
    return 0.5 * np.tanh(beta * (m + h)) + 0.5 + delta_P


def jump_func(m, beta, h, delta_P, M):
    step = np.sign(m)  # 0 at m==0, matches u(x)=0 at x=0.5 in their notation
    return smooth_func(m, beta, h, delta_P) + (M / 2) * step


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
    data["se"] = np.sqrt(np.clip(data["P"] * (1 - data["P"]), 1e-4, None) / n)
    data["n"] = n
    return data.reset_index(drop=True)


def fit_smooth(x, y, sigma):
    bounds = ([-np.inf, -np.inf, -1e-4], [np.inf, np.inf, 1e-4])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, pcov = curve_fit(smooth_func, x, y, sigma=sigma, absolute_sigma=True,
                                    p0=[1.0, 0.0, 0.0], maxfev=5000, bounds=bounds)
        return popt, pcov
    except Exception:
        return None, None


def fit_jump(x, y, sigma, p0_smooth):
    bounds = ([-np.inf, -np.inf, -1e-4, -2.0], [np.inf, np.inf, 1e-4, 2.0])
    p0 = list(p0_smooth) + [0.0]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, pcov = curve_fit(jump_func, x, y, sigma=sigma, absolute_sigma=True,
                                    p0=p0, maxfev=5000, bounds=bounds)
        return popt, pcov
    except Exception:
        return None, None


def bic(y, y_pred, sigma, k):
    n = len(y)
    chi2 = np.sum(((y - y_pred) / sigma) ** 2)
    return chi2 + k * np.log(n)


def main():
    opinion_pairs = load_opinion_pairs()
    rows = []

    for model in MODELS:
        files = sorted(glob.glob(str(BASE / model / RESULTS_SUBDIR / "transition_prob_50_*.txt")))
        print(f"{model}: {len(files)} files")

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
            if data is None or len(data) < 5:
                continue
            x, y, sigma = data["m"].values, data["P"].values, data["se"].values

            popt_s, pcov_s = fit_smooth(x, y, sigma)
            if popt_s is None:
                continue
            beta, h = popt_s[0], popt_s[1]
            if abs(beta) > 500 or abs(h) > 500:
                continue

            popt_j, pcov_j = fit_jump(x, y, sigma, popt_s)
            if popt_j is None or pcov_j is None or not np.all(np.isfinite(pcov_j)):
                continue
            M = popt_j[3]
            se_M = np.sqrt(max(pcov_j[3, 3], 0))
            if se_M == 0 or not np.isfinite(se_M):
                continue

            y_pred_s = smooth_func(x, *popt_s)
            y_pred_j = jump_func(x, *popt_j)
            bic_s = bic(y, y_pred_s, sigma, 3)
            bic_j = bic(y, y_pred_j, sigma, 4)

            rows.append(dict(
                Model=model, Opinion_A=opinion1, beta=beta, h=h,
                M=M, se_M=se_M, z_M=M / se_M,
                bic_smooth=bic_s, bic_jump=bic_j, delta_bic=bic_s - bic_j,  # >0 favors jump model
            ))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "gelastopoulos_comparison.csv", index=False)
    print(f"\nWrote {len(out)} rows -> {OUT_DIR / 'gelastopoulos_comparison.csv'}")

    lines = []
    lines.append(f"N (model,pair) combos fit with both models: {len(out)}")
    sig_frac = (out["z_M"].abs() > 2).mean()
    lines.append(f"Fraction with |M|/SE_M > 2 (statistically distinguishable discontinuity): {sig_frac*100:.1f}%")
    lines.append(f"Median |M| (jump magnitude, probability units): {out['M'].abs().median():.4f}")
    lines.append(f"Median z_M: {out['z_M'].median():.2f}")
    # BIC: delta_bic > 2 is "positive" evidence for jump model (Kass & Raftery), > 6 "strong"
    lines.append(f"Fraction where jump model beats smooth by BIC>2 (positive evidence): "
                 f"{(out['delta_bic'] > 2).mean()*100:.1f}%")
    lines.append(f"Fraction where jump model beats smooth by BIC>6 (strong evidence): "
                 f"{(out['delta_bic'] > 6).mean()*100:.1f}%")
    lines.append(f"Fraction where SMOOTH model is favored (delta_bic < -2): "
                 f"{(out['delta_bic'] < -2).mean()*100:.1f}%")

    lines.append("\nPer-model breakdown (fraction with significant M, |z_M|>2):")
    for model, g in out.groupby("Model"):
        lines.append(f"  {model:28s} sig_frac={100*(g['z_M'].abs()>2).mean():5.1f}%  "
                      f"median|M|={g['M'].abs().median():.4f}  n={len(g)}")

    summary = "\n".join(lines)
    print("\n" + summary)
    (OUT_DIR / "gelastopoulos_comparison_summary.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()

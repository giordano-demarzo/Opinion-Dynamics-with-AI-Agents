#!/usr/bin/env python3
"""
Referee-response analysis: bootstrapped/analytic confidence intervals on the
fitted (beta, h) parameters of the tanh transition-probability model, and
robustness of the spinodal (metastable vs. monostable) classification.

For each model x opinion pair, refits P(m) = 0.5*tanh(beta*(m+h)) + 0.5 + delta_P
with WEIGHTED least squares (sigma = binomial standard error already stored
per m0 point in the raw transition_prob_*.txt files), which both (a) matches
statistical best practice for heteroscedastic binomial data and (b) gives a
proper parameter covariance matrix "for free" via scipy's curve_fit
(absolute_sigma=True), from which SE(beta) and SE(h) follow directly -- no
new LLM queries needed (this addresses Reviewer 1's CI request using
purely the existing raw counts).

We then propagate the beta uncertainty into the spinodal boundary h_c(beta)
via the delta method, and flag pairs whose spinodal classification
(metastable vs. monostable) is not robust to +-1 SE / +-2 SE.

Usage:
    python ci_analysis.py
Outputs:
    resubmission/data/fits_with_ci.csv
    resubmission/data/ci_summary.txt
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
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "Llama-3.1-8B-Instruct",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "Qwen2.5-14B-Instruct",
    "Qwen2.5-32B-Instruct",
    "Qwen3-14B",
    "Qwen3-32B",
    "gemini-2.5-flash-lite",
    "gpt-5-mini",
]

RESULTS_SUBDIR = "results_batched_vllm/N=50"


def fit_func(m, beta, h, delta_P):
    return 0.5 * np.tanh(beta * (m + h)) + 0.5 + delta_P


def h_spinodal(beta):
    """Positive branch of the spinodal boundary; NaN if beta <= 1 (no bistability)."""
    if beta <= 1:
        return np.nan
    m_s = np.sqrt(1 - 1 / beta)
    return m_s - np.arctanh(m_s) / beta


def dhspinodal_dbeta(beta, eps=1e-4):
    """Numerical derivative of h_spinodal wrt beta, for delta-method SE propagation."""
    if beta <= 1 + 2 * eps:
        return np.nan
    return (h_spinodal(beta + eps) - h_spinodal(beta - eps)) / (2 * eps)


def load_opinion_pairs():
    df = pd.read_csv(BASE / "submission" / "opinion_pairs.csv")
    return dict(zip(df["Opinion_A"], df["Opinion_B"]))


def parse_txt(filepath):
    """Read a transition_prob_*.txt file, handling both header and no-header formats."""
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
    data["n_valid"] = n
    # Binomial SE; floor to avoid zero-weight blowups when p is exactly 0 or 1
    data["se"] = np.sqrt(np.clip(data["P"] * (1 - data["P"]), 1e-4, None) / n)
    return data.reset_index(drop=True)


def fit_with_ci(x, y, sigma):
    bounds = ([-np.inf, -np.inf, -1e-4], [np.inf, np.inf, 1e-4])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, pcov = curve_fit(
                fit_func, x, y, sigma=sigma, absolute_sigma=True,
                p0=[1.0, 0.0, 0.0], maxfev=5000, bounds=bounds,
            )
    except Exception:
        return None
    beta, h, delta_P = popt
    if pcov is None or not np.all(np.isfinite(pcov)):
        return None
    se_beta, se_h, _ = np.sqrt(np.clip(np.diag(pcov), 0, None))
    y_pred = fit_func(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(beta=beta, h=h, delta_P=delta_P, se_beta=se_beta, se_h=se_h, r2=r2, n_points=len(x))


def classify(beta, h):
    if beta <= 1:
        return "monostable (beta<=1)"
    hc = h_spinodal(beta)
    return "metastable" if abs(h) < hc else "monostable"


def main():
    opinion_pairs = load_opinion_pairs()
    rows = []

    for model in MODELS:
        pattern = str(BASE / model / RESULTS_SUBDIR / "transition_prob_50_*.txt")
        files = sorted(glob.glob(pattern))
        print(f"{model}: {len(files)} files")

        for fp in files:
            fname = Path(fp).stem  # transition_prob_50_0.2_<opinion> or transition_prob_50_opinions_<opinion>
            # opinion name = everything after the 3rd underscore-separated token block; robust split:
            parts = Path(fp).name.split("_", 4)
            opinion1_raw = parts[4].replace(".txt", "") if len(parts) >= 5 else fname
            opinion1 = opinion1_raw.replace("_", " ").strip()

            # match to canonical pair list (models sometimes use underscores vs spaces)
            opinion2 = opinion_pairs.get(opinion1)
            if opinion2 is None:
                # try loose match ignoring case/spacing
                match = [a for a in opinion_pairs if a.replace(" ", "_").lower() == opinion1_raw.lower()]
                if match:
                    opinion1 = match[0]
                    opinion2 = opinion_pairs[opinion1]
                else:
                    opinion2 = "Unknown"

            data = parse_txt(fp)
            if data is None or len(data) < 4:
                continue

            fit = fit_with_ci(data["m"].values, data["P"].values, data["se"].values)
            if fit is None:
                continue

            beta, h, se_beta, se_h = fit["beta"], fit["h"], fit["se_beta"], fit["se_h"]
            if abs(beta) > 500 or abs(h) > 500:
                continue

            hc = h_spinodal(abs(beta)) if beta > 1 else np.nan
            dhc = dhspinodal_dbeta(abs(beta)) if beta > 1 else np.nan
            # delta-method SE on (h_c - |h|) margin, propagating beta uncertainty into h_c
            if np.isfinite(hc) and np.isfinite(dhc):
                se_margin = np.sqrt((se_h) ** 2 + (dhc * se_beta) ** 2)
                margin = hc - abs(h)  # >0 => metastable
                z_margin = margin / se_margin if se_margin > 0 else np.nan
            else:
                margin, se_margin, z_margin = np.nan, np.nan, np.nan

            label = classify(beta, h)
            robust_1se = (abs(z_margin) > 1) if np.isfinite(z_margin) else True
            robust_2se = (abs(z_margin) > 2) if np.isfinite(z_margin) else True

            p0_fit = 0.5 * np.tanh(beta * h) + 0.5 + fit["delta_P"]  # model-predicted P(A) at m=0

            rows.append(dict(
                Model=model, Opinion_A=opinion1, Opinion_B=opinion2,
                beta=beta, se_beta=se_beta, h=h, se_h=se_h, delta_P=fit["delta_P"],
                p0_fit=p0_fit,
                r2=fit["r2"], n_points=fit["n_points"],
                h_spinodal=hc, margin_h=margin, se_margin=se_margin, z_margin=z_margin,
                classification=label, robust_1se=robust_1se, robust_2se=robust_2se,
            ))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "fits_with_ci.csv", index=False)
    print(f"\nWrote {len(out)} fitted pairs -> {OUT_DIR / 'fits_with_ci.csv'}")

    # ---- Summary for the rebuttal letter ----
    lines = []
    lines.append(f"Total fitted (model, opinion pair) combinations: {len(out)}")
    meta = out[out["classification"] == "metastable"]
    lines.append(f"Metastable overall: {len(meta)} ({100*len(meta)/len(out):.1f}%)")

    finite = out[np.isfinite(out["z_margin"])]
    lines.append(f"\nCombinations with a defined spinodal margin (beta>1): {len(finite)}")
    not_robust_1se = finite[~finite["robust_1se"]]
    not_robust_2se = finite[~finite["robust_2se"]]
    lines.append(f"  Classification flips within +-1 SE: {len(not_robust_1se)} "
                 f"({100*len(not_robust_1se)/len(finite):.1f}%)")
    lines.append(f"  Classification flips within +-2 SE: {len(not_robust_2se)} "
                 f"({100*len(not_robust_2se)/len(finite):.1f}%)")

    lines.append("\nPer-model metastable fraction (point estimate) and mean |z_margin|:")
    for model, g in out.groupby("Model"):
        frac = (g["classification"] == "metastable").mean()
        gz = g[np.isfinite(g["z_margin"])]
        mean_absz = gz["z_margin"].abs().mean() if len(gz) else np.nan
        lines.append(f"  {model:28s} metastable={frac*100:5.1f}%   mean|z_margin|={mean_absz:5.2f}   n={len(g)}")

    lines.append("\nMedian relative SE (SE/|estimate|):")
    lines.append(f"  beta: {np.median(out['se_beta'] / out['beta'].abs()):.3f}")
    lines.append(f"  h:    {np.median(out['se_h'] / out['h'].abs().clip(lower=1e-3)):.3f}")

    summary = "\n".join(lines)
    print("\n" + summary)
    (OUT_DIR / "ci_summary.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()

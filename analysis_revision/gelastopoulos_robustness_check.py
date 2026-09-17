#!/usr/bin/env python3
"""
Assumption-lighter robustness check on gelastopoulos_comparison.py.

Concern: the nested-model test assumes the smooth component IS our tanh,
so a very steep (large-beta) tanh could in principle "absorb" a true
discontinuity into an inflated beta rather than into the explicit jump
term M, biasing toward a null finding on M.

Fix: fit the smooth tanh using ONLY the points with |m0| > 0.15 (i.e.
excluding the two points closest to the crossing, m0=+-0.08), then
extrapolate that far-field smooth trend IN to m0=+-0.08 and compare
to what was actually observed there. This does not let the near-center
points influence the smooth fit at all, so it cannot "explain away" a
real jump via curvature fit to those exact points. If a marginal-majority
mechanism is present, we expect a systematic, directionally consistent
excess: observed P(+0.08) > smooth-predicted, and observed P(-0.08) <
smooth-predicted (the model overshoots in the direction of whichever
side is already ahead).

Usage:
    python gelastopoulos_robustness_check.py
"""
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning
from scipy import stats

warnings.filterwarnings("ignore")

BASE = Path("/home/jovyan/LLMs_opinion_dynamics_bias")
OUT_DIR = BASE / "resubmission" / "data"

MODELS = ["Llama-3.1-8B-Instruct", "gemma-3-12b-it", "gemma-3-27b-it", "Qwen2.5-14B-Instruct",
          "Qwen2.5-32B-Instruct", "Qwen3-14B", "Qwen3-32B", "gemini-2.5-flash-lite", "gpt-5-mini"]
RESULTS_SUBDIR = "results_batched_vllm/N=50"


def smooth_func(m, beta, h, delta_P):
    return 0.5 * np.tanh(beta * (m + h)) + 0.5 + delta_P


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
        return popt
    except Exception:
        return None


def main():
    opinion_pairs = load_opinion_pairs()
    rows = []

    for model in MODELS:
        files = sorted(glob.glob(str(BASE / model / RESULTS_SUBDIR / "transition_prob_50_*.txt")))
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
            if data is None:
                continue

            near = data[data["m"].abs() <= 0.15]
            far = data[data["m"].abs() > 0.15]
            if len(near) < 2 or len(far) < 4:
                continue

            popt_far = fit_smooth(far["m"].values, far["P"].values, far["se"].values)
            if popt_far is None:
                continue
            beta, h = popt_far[0], popt_far[1]
            if abs(beta) > 500 or abs(h) > 500:
                continue

            for _, r in near.iterrows():
                m0 = r["m"]
                pred = smooth_func(m0, *popt_far)
                observed = r["P"]
                # residual signed so that POSITIVE means "excess in the direction
                # of whichever side m0 already favors" -- the marginal-majority
                # prediction if the effect were real
                signed_residual = (observed - pred) * np.sign(m0) if m0 != 0 else np.nan
                rows.append(dict(Model=model, Opinion_A=opinion1, m0=m0, beta_far=beta,
                                  observed=observed, predicted=pred,
                                  raw_residual=observed - pred, signed_residual=signed_residual,
                                  se=r["se"], n=r["n"]))

    out = pd.DataFrame(rows).dropna(subset=["signed_residual"])
    out.to_csv(OUT_DIR / "gelastopoulos_leaveout_check.csv", index=False)
    print(f"Wrote {len(out)} near-center leave-out points -> gelastopoulos_leaveout_check.csv")

    lines = []
    lines.append(f"N near-center points (|m0|<=0.15) with far-field extrapolation: {len(out)}")
    lines.append(f"Mean signed residual (>0 would support a marginal-majority-style excess): "
                 f"{out['signed_residual'].mean():.4f}")
    lines.append(f"Median signed residual: {out['signed_residual'].median():.4f}")
    t, p = stats.ttest_1samp(out["signed_residual"], 0)
    lines.append(f"One-sample t-test (H0: mean signed residual = 0): t={t:.2f}, p={p:.4f}")
    frac_positive = (out["signed_residual"] > 0).mean()
    lines.append(f"Fraction of points with positive signed residual (majority-amplifying direction): "
                 f"{frac_positive*100:.1f}% (50% expected under no effect)")
    from scipy.stats import binomtest
    bt = binomtest(int((out["signed_residual"] > 0).sum()), len(out), 0.5)
    lines.append(f"Binomial test vs 50%: p={bt.pvalue:.4f}")

    lines.append(f"\nMean |raw_residual| (typical extrapolation error, magnitude only): "
                 f"{out['raw_residual'].abs().mean():.4f}")
    lines.append(f"For comparison, mean SE of the observed points: {out['se'].mean():.4f}")
    lines.append(f"Mean |raw_residual| / mean SE ratio: {out['raw_residual'].abs().mean()/out['se'].mean():.2f}")

    summary = "\n".join(lines)
    print("\n" + summary)
    (OUT_DIR / "gelastopoulos_leaveout_summary.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()

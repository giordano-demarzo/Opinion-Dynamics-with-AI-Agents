#!/usr/bin/env python3
"""
Fit the tanh transition-probability model to the 12 Pew 2017 political-
typology paired-statement items (held-out set, used verbatim and in full),
with the same weighted least-squares procedure as fit_new_pairs.py.

Outputs: data/pew_pairs_fits.csv and data/pew_pairs_summary.txt
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning

warnings.filterwarnings("ignore")

BASE = Path("/home/jovyan/LLMs_opinion_dynamics_bias")
DATA = BASE / "resubmission" / "data"

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


def classify(beta, h):
    if beta <= 1:
        return "monostable"
    return "metastable" if abs(h) < h_spinodal(beta) else "monostable"


def parse_txt(fp):
    try:
        d = pd.read_csv(fp)
        if "m0" in d.columns:
            d = d.rename(columns={"m0": "m", "count_A": "cA", "count_B": "cB"})
        else:
            d = pd.read_csv(fp, header=None, usecols=[0, 1, 2, 3, 4])
            d.columns = ["m", "cA", "cB", "probability", "standard_error"]
    except Exception:
        return None
    for c in ["m", "cA", "cB"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["m", "cA", "cB"])
    n = d.cA + d.cB
    d = d[n > 0].copy()
    d["P"] = d.cA / (d.cA + d.cB)
    d["se"] = np.sqrt(np.clip(d.P * (1 - d.P), 1e-4, None) / (d.cA + d.cB))
    d["n"] = d.cA + d.cB
    return d.reset_index(drop=True)


def main():
    pairs = pd.read_csv(DATA / "pew_typology_2017_pairs.csv")
    rows = []
    for model in MODELS:
        out_dir = DATA / model / RESULTS_SUBDIR
        n_found = 0
        for _, pr in pairs.iterrows():
            a = pr.Opinion_A
            candidates = [
                out_dir / f"transition_prob_50_0.2_{a}.txt",                       # vLLM naming
                out_dir / f"transition_prob_50_opinions_{a.replace(' ', '_')}.txt",  # API naming
            ]
            fp = next((c for c in candidates if c.exists()), None)
            if fp is None:
                continue
            d = parse_txt(fp)
            if d is None or len(d) < 4:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", OptimizeWarning)
                    popt, pcov = curve_fit(
                        fit_func, d.m.values, d.P.values, sigma=d.se.values,
                        absolute_sigma=True, p0=[1.0, 0.0, 0.0], maxfev=5000,
                        bounds=([-np.inf, -np.inf, -1e-4], [np.inf, np.inf, 1e-4]))
            except Exception:
                continue
            beta, h, _ = popt
            if not np.all(np.isfinite(pcov)) or abs(beta) > 500 or abs(h) > 500:
                continue
            n_found += 1
            mean_discard = np.nan
            if "invalid" in pd.read_csv(fp).columns:
                raw = pd.read_csv(fp)
                mean_discard = (raw["invalid"] / (raw["invalid"] + raw["count_A"] + raw["count_B"])).mean()
            rows.append(dict(Model=model, pew_question_id=pr.pew_question_id,
                             short_label=pr.short_label, Opinion_A=a,
                             beta=beta, h=h, classification=classify(beta, h),
                             mean_discard=mean_discard))
        print(f"{model}: fitted {n_found}/12")

    out = pd.DataFrame(rows)
    out.to_csv(DATA / "pew_pairs_fits.csv", index=False)

    lines = [f"Total fits: {len(out)} (of {9*12} possible)"]
    meta = out.classification == "metastable"
    lines.append(f"Overall metastable fraction: {100*meta.mean():.1f}%")
    lines.append(f"Mean discard rate: {100*out.mean_discard.mean():.1f}%")
    lines.append("\nPer model:")
    for m, g in out.groupby("Model"):
        lines.append(f"  {m:28s} metastable={100*(g.classification=='metastable').mean():5.1f}%  "
                     f"n={len(g)}  mean_discard={100*g.mean_discard.mean():.1f}%")
    lines.append("\nPer pair (metastable fraction across models, median |h|):")
    for lab, g in out.groupby("short_label"):
        lines.append(f"  {lab:32s} meta={100*(g.classification=='metastable').mean():5.1f}%  "
                     f"median|h|={g.h.abs().median():.3f}  n={len(g)}")
    txt = "\n".join(lines)
    print("\n" + txt)
    (DATA / "pew_pairs_summary.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Formal model comparison between candidate single-agent response families
fitted to the same P(m) data (no new LLM queries). Answers R1's concern that
the tanh form is "baked into the design": every candidate below is bounded
and (weakly) monotone in the majority signal, i.e., all satisfy the design
constraints, yet they imply qualitatively different collective behavior
(Castellano 2012; Vazquez & Lopez 2008). Which member fits is an empirical
question; we let BIC (binomial likelihood) decide.

Families (k = free parameters):
  tanh    : P = 1/2 [tanh(beta (m+h)) + 1]                     k=2
  linear  : P = clip(a + b m, eps, 1-eps)   (voter-like)       k=2
  step    : P = p1 if m < m0 else p2, m0 on a grid             k=3
  tanhjump: tanh + (M/2) sign(m)  (Gelastopoulos-style)        k=3

Output: per-(model,pair) BIC table + summary of family preference.
"""
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

BASE = Path("/home/jovyan/LLMs_opinion_dynamics_bias")
OUT_DIR = BASE / "resubmission" / "data"

MODELS = ["Llama-3.1-8B-Instruct", "gemma-3-12b-it", "gemma-3-27b-it", "Qwen2.5-14B-Instruct",
          "Qwen2.5-32B-Instruct", "Qwen3-14B", "Qwen3-32B", "gemini-2.5-flash-lite", "gpt-5-mini"]
RESULTS_SUBDIR = "results_batched_vllm/N=50"
EPS = 1e-6


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
    return data.reset_index(drop=True)


def binom_ll(p, cA, cB):
    p = np.clip(p, EPS, 1 - EPS)
    return np.sum(cA * np.log(p) + cB * np.log(1 - p))


def fit_tanh(m, cA, cB):
    def nll(theta):
        beta, h = theta
        return -binom_ll(0.5 * (np.tanh(beta * (m + h)) + 1), cA, cB)
    best = None
    for b0 in (0.5, 2.0, 8.0, 30.0):
        for h0 in (-0.5, 0.0, 0.5):
            r = minimize(nll, [b0, h0], method="Nelder-Mead",
                         options=dict(maxiter=4000, fatol=1e-9, xatol=1e-9))
            if best is None or r.fun < best.fun:
                best = r
    return -best.fun, best.x


def fit_tanh_jump(m, cA, cB, x0):
    def nll(theta):
        beta, h, M = theta
        p = 0.5 * (np.tanh(beta * (m + h)) + 1) + (M / 2) * np.sign(m)
        return -binom_ll(p, cA, cB)
    best = None
    for M0 in (-0.1, 0.0, 0.1):
        r = minimize(nll, [x0[0], x0[1], M0], method="Nelder-Mead",
                     options=dict(maxiter=6000, fatol=1e-9, xatol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    return -best.fun, best.x


def fit_linear(m, cA, cB):
    def nll(theta):
        a, b = theta
        return -binom_ll(a + b * m, cA, cB)
    best = None
    for b0 in (0.1, 0.5, 1.0):
        r = minimize(nll, [0.5, b0], method="Nelder-Mead",
                     options=dict(maxiter=4000, fatol=1e-9, xatol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    return -best.fun, best.x


def fit_step(m, cA, cB):
    # threshold m0 between consecutive observed m values; p1, p2 are MLE means
    order = np.argsort(m)
    ms, a, b = m[order], cA[order], cB[order]
    best_ll, best = -np.inf, None
    for cut in range(1, len(ms)):
        loA, loB = a[:cut].sum(), b[:cut].sum()
        hiA, hiB = a[cut:].sum(), b[cut:].sum()
        p1 = loA / max(loA + loB, 1)
        p2 = hiA / max(hiA + hiB, 1)
        ll = (binom_ll(np.full(cut, p1), a[:cut], b[:cut])
              + binom_ll(np.full(len(ms) - cut, p2), a[cut:], b[cut:]))
        if ll > best_ll:
            best_ll, best = ll, (0.5 * (ms[cut - 1] + ms[cut]), p1, p2)
    return best_ll, best


def main():
    rows = []
    for model in MODELS:
        files = sorted(glob.glob(str(BASE / model / RESULTS_SUBDIR / "transition_prob_50_*.txt")))
        for fp in files:
            data = parse_txt(fp)
            if data is None or len(data) < 6:
                continue
            m = data["m"].values.astype(float)
            cA = data["cA"].values.astype(float)
            cB = data["cB"].values.astype(float)
            npts = len(m)
            try:
                ll_t, th_t = fit_tanh(m, cA, cB)
                ll_j, th_j = fit_tanh_jump(m, cA, cB, th_t)
                ll_l, th_l = fit_linear(m, cA, cB)
                ll_s, th_s = fit_step(m, cA, cB)
            except Exception:
                continue
            logn = np.log(npts)
            rows.append(dict(
                Model=model, file=Path(fp).name, n_points=npts,
                beta=th_t[0], h=th_t[1], M=th_j[2],
                bic_tanh=-2 * ll_t + 2 * logn,
                bic_jump=-2 * ll_j + 3 * logn,
                bic_linear=-2 * ll_l + 2 * logn,
                bic_step=-2 * ll_s + 3 * logn,
            ))
    out = pd.DataFrame(rows)
    bics = out[["bic_tanh", "bic_jump", "bic_linear", "bic_step"]]
    out["winner"] = bics.idxmin(axis=1).str.replace("bic_", "")
    out["dbic_linear_vs_tanh"] = out.bic_linear - out.bic_tanh
    out["dbic_step_vs_tanh"] = out.bic_step - out.bic_tanh
    out["dbic_jump_vs_tanh"] = out.bic_jump - out.bic_tanh
    out.to_csv(OUT_DIR / "response_family_comparison.csv", index=False)

    lines = [f"N (model,pair) combos fitted with all four families: {len(out)}"]
    lines.append("\nBIC winner counts (lowest BIC):")
    for k, v in out.winner.value_counts().items():
        lines.append(f"  {k:8s} {v:4d}  ({100*v/len(out):.1f}%)")
    for alt, col in [("linear (voter-like)", "dbic_linear_vs_tanh"),
                     ("step", "dbic_step_vs_tanh"),
                     ("tanh+jump", "dbic_jump_vs_tanh")]:
        d = out[col]
        lines.append(f"\n{alt} vs tanh:")
        lines.append(f"  tanh favored (dBIC>2):  {100*(d>2).mean():.1f}%")
        lines.append(f"  tanh strongly favored (dBIC>6): {100*(d>6).mean():.1f}%")
        lines.append(f"  {alt} favored (dBIC<-2): {100*(d<-2).mean():.1f}%")
        lines.append(f"  median dBIC (alt - tanh): {d.median():.1f}")
    lines.append("\nPer-model tanh win fraction (tanh or tanh+jump lowest BIC):")
    for mod, g in out.groupby("Model"):
        wins = g.winner.isin(["tanh", "jump"]).mean()
        lines.append(f"  {mod:28s} {100*wins:.1f}%  (plain tanh alone: {100*(g.winner=='tanh').mean():.1f}%)  n={len(g)}")
    summary = "\n".join(lines)
    print(summary)
    (OUT_DIR / "response_family_comparison_summary.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()

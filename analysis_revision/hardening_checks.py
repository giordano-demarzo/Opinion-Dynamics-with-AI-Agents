#!/usr/bin/env python3
"""
Hardening checks for the resubmission (no new LLM queries).

A. Marginal-majority jump term: exact-binomial likelihood-ratio test of
   tanh+jump vs tanh on every main-set curve, plus the SIGN of the fitted
   jump M. A genuine marginal-majority effect predicts M > 0
   (majority-amplifying); a jump term that merely absorbs mid-curve misfit
   of a steep smooth curve has no preferred sign.
B. Family-independence of the collective classification: fit the clipped
   linear ramp P = clip(a + b m) by binomial MLE and classify each curve as
   bistable/monostable under the ramp's own mean-field self-consistency
   (bistable iff k = 2b > 1 and |h_r| <= 1 - 1/k, h_r = (2a-1)/k). Compare
   with the tanh spinodal classification.
C. Discard-stratified headline: metastable fraction of main-set combos by
   mean discard rate.
D. Bootstrap classification robustness for the three new pair sets (Pew,
   GlobalOpinionQA, safety), parametric bootstrap as in bootstrap_ci.py.

Outputs (resubmission/data/):
   hardening_lrt_ramp.csv, heldout_bootstrap.csv, hardening_checks_summary.txt
"""
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit, OptimizeWarning
from scipy.stats import chi2, binomtest, mannwhitneyu

warnings.filterwarnings("ignore")

BASE = Path("/home/jovyan/LLMs_opinion_dynamics_bias")
RES = BASE / "resubmission"
DATA = RES / "data"
MODELS = ["Llama-3.1-8B-Instruct", "gemma-3-12b-it", "gemma-3-27b-it", "Qwen2.5-14B-Instruct",
          "Qwen2.5-32B-Instruct", "Qwen3-14B", "Qwen3-32B", "gemini-2.5-flash-lite", "gpt-5-mini"]
EPS = 1e-6
N_BOOT = 200
rng = np.random.default_rng(0)
lines = []


def say(s=""):
    print(s)
    lines.append(s)


# ----------------------------------------------------------------------------- io
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
    return d.reset_index(drop=True)


def find_file(out_dir, a):
    cands = [out_dir / f"transition_prob_50_0.2_{a}.txt",
             out_dir / f"transition_prob_50_opinions_{a.replace(' ', '_')}.txt"]
    return next((c for c in cands if c.exists()), None)


# ------------------------------------------------------------------------ fitting
def binom_ll(p, cA, cB):
    p = np.clip(p, EPS, 1 - EPS)
    return np.sum(cA * np.log(p) + cB * np.log(1 - p))


def fit_tanh(m, cA, cB):
    def nll(t):
        return -binom_ll(0.5 * (np.tanh(t[0] * (m + t[1])) + 1), cA, cB)
    best = None
    for b0 in (0.5, 2.0, 8.0, 30.0):
        for h0 in (-0.5, 0.0, 0.5):
            r = minimize(nll, [b0, h0], method="Nelder-Mead",
                         options=dict(maxiter=4000, fatol=1e-9, xatol=1e-9))
            if best is None or r.fun < best.fun:
                best = r
    return -best.fun, best.x


def fit_jump(m, cA, cB, x0):
    def nll(t):
        p = 0.5 * (np.tanh(t[0] * (m + t[1])) + 1) + (t[2] / 2) * np.sign(m)
        return -binom_ll(p, cA, cB)
    best = None
    for M0 in (-0.1, 0.0, 0.1):
        r = minimize(nll, [x0[0], x0[1], M0], method="Nelder-Mead",
                     options=dict(maxiter=6000, fatol=1e-9, xatol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    return -best.fun, best.x


def fit_ramp(m, cA, cB, x0):
    def nll(t):
        return -binom_ll(t[0] + t[1] * m, cA, cB)
    beta, h = x0
    b_init = np.clip(beta / 2, 0.05, 20)
    best = None
    for b0 in (0.25, 0.5, 1.0, b_init):
        for a0 in (0.5, np.clip(0.5 + b0 * h, 0, 1)):
            r = minimize(nll, [a0, b0], method="Nelder-Mead",
                         options=dict(maxiter=4000, fatol=1e-9, xatol=1e-9))
            if best is None or r.fun < best.fun:
                best = r
    return -best.fun, best.x


def h_spinodal(beta):
    if beta <= 1:
        return np.nan
    ms = np.sqrt(1 - 1 / beta)
    return ms - np.arctanh(ms) / beta


def tanh_class(beta, h):
    return beta > 1 and abs(h) < h_spinodal(beta)


def ramp_class(a, b):
    k = 2 * b
    if k <= 1:
        return False
    hr = (2 * a - 1) / k
    return abs(hr) <= 1 - 1 / k


# --------------------------------------------------------------- A + B (main set)
official = pd.read_csv(BASE / "submission" / "opinion_pairs.csv")
off_pairs = list(official.Opinion_A)
rows = []
for model in MODELS:
    out_dir = BASE / model / "results_batched_vllm" / "N=50"
    for a in off_pairs:
        fp = find_file(out_dir, a)
        if fp is None:
            continue
        d = parse_txt(fp)
        if d is None or len(d) < 6:
            continue
        m, cA, cB = d.m.values.astype(float), d.cA.values.astype(float), d.cB.values.astype(float)
        try:
            ll_t, th_t = fit_tanh(m, cA, cB)
            ll_j, th_j = fit_jump(m, cA, cB, th_t)
            ll_r, th_r = fit_ramp(m, cA, cB, th_t)
        except Exception:
            continue
        lr = max(0.0, 2 * (ll_j - ll_t))
        rows.append(dict(Model=model, Opinion_A=a, n_points=len(m),
                         beta=th_t[0], h=th_t[1], M=th_j[2], LR=lr, p_lrt=1 - chi2.cdf(lr, 1),
                         dbic_jump=(-2 * ll_j + 3 * np.log(len(m))) - (-2 * ll_t + 2 * np.log(len(m))),
                         ramp_a=th_r[0], ramp_b=th_r[1], k_ramp=2 * th_r[1],
                         h_ramp=(2 * th_r[0] - 1) / (2 * th_r[1]) if th_r[1] != 0 else np.nan,
                         tanh_meta=tanh_class(th_t[0], th_t[1]), ramp_meta=ramp_class(th_r[0], th_r[1]),
                         dbic_ramp=(-2 * ll_r + 2 * np.log(len(m))) - (-2 * ll_t + 2 * np.log(len(m)))))
df = pd.DataFrame(rows)
df.to_csv(DATA / "hardening_lrt_ramp.csv", index=False)

say("=" * 78)
say(f"A. Likelihood-ratio and sign test for the marginal-majority jump term  (n={len(df)} main-set curves)")
say("=" * 78)
sup = df[df.beta > 1]
sig = df[df.p_lrt < 0.05]
say(f"LRT p<0.05 (chi2, 1 dof):            {100*(df.p_lrt<0.05).mean():.1f}%  (5% expected under H0)")
say(f"LRT p<0.05 among beta>1:             {100*(sup.p_lrt<0.05).mean():.1f}%  (n={len(sup)})")
say(f"LRT p<0.01:                          {100*(df.p_lrt<0.01).mean():.1f}%")
say(f"BIC favors jump (dBIC<-2):           {100*(df.dbic_jump<-2).mean():.1f}%")
for lab, sub in [("all curves", df), ("LRT-significant curves", sig), ("BIC-preferred curves", df[df.dbic_jump < -2])]:
    npos = int((sub.M > 0).sum()); n = len(sub)
    bt = binomtest(npos, n, 0.5) if n > 0 else None
    say(f"Sign of M, {lab:24s}: M>0 in {npos}/{n} = {100*npos/max(n,1):.1f}%   binomial p vs 50% = {bt.pvalue:.3f}" if n else f"{lab}: none")
say(f"Median |M| overall: {df.M.abs().median():.4f};  median |M| among LRT-significant: {sig.M.abs().median():.4f}")
say(f"Median beta, LRT-significant vs not: {sig.beta.median():.2f} vs {df[df.p_lrt>=0.05].beta.median():.2f}"
    f"  (Mann-Whitney p={mannwhitneyu(sig.beta, df[df.p_lrt>=0.05].beta).pvalue:.2e})")
say("Per model: LRT-significant fraction / fraction of those with M>0")
for mod, g in df.groupby("Model"):
    gs = g[g.p_lrt < 0.05]
    say(f"  {mod:26s} {100*(g.p_lrt<0.05).mean():5.1f}%   M>0: {100*(gs.M>0).mean() if len(gs) else float('nan'):5.1f}%  (n_sig={len(gs)})")

say()
say("=" * 78)
say("B. Family-independence of the collective classification: ramp vs tanh")
say("=" * 78)
agree = (df.tanh_meta == df.ramp_meta)
say(f"Classification agreement (ramp bistable <-> tanh metastable): {100*agree.mean():.1f}%  (n={len(df)})")
tp = ((df.tanh_meta) & (df.ramp_meta)).sum(); tn = ((~df.tanh_meta) & (~df.ramp_meta)).sum()
fp_ = ((~df.tanh_meta) & (df.ramp_meta)).sum(); fn = ((df.tanh_meta) & (~df.ramp_meta)).sum()
po = agree.mean(); pe = (df.tanh_meta.mean() * df.ramp_meta.mean() + (1 - df.tanh_meta.mean()) * (1 - df.ramp_meta.mean()))
say(f"  both metastable {tp}, both monostable {tn}, tanh-only {fn}, ramp-only {fp_};  Cohen's kappa = {(po-pe)/(1-pe):.2f}")
say(f"Metastable fraction: tanh {100*df.tanh_meta.mean():.1f}%   ramp {100*df.ramp_meta.mean():.1f}%")
say(f"Supercritical fraction: tanh beta>1 {100*(df.beta>1).mean():.1f}%   ramp k>1 {100*(df.k_ramp>1).mean():.1f}%")
say(f"Agreement among curves where ramp has lower BIC than tanh: {100*agree[df.dbic_ramp<0].mean():.1f}% (n={(df.dbic_ramp<0).sum()})")
fc = pd.read_csv(DATA / "fits_with_ci.csv")
fc = fc[fc.Opinion_A.isin(off_pairs)][["Model", "Opinion_A", "robust_2se", "classification"]]
dm = df.merge(fc, on=["Model", "Opinion_A"], how="left")
rob = dm[dm.robust_2se == True]
say(f"Agreement among robustly classified combos (analytic 2 SE): {100*(rob.tanh_meta==rob.ramp_meta).mean():.1f}% (n={len(rob)})")
say(f"Spearman(h_tanh, h_ramp) = {df[['h','h_ramp']].corr(method='spearman').iloc[0,1]:.3f}")

say()
say("=" * 78)
say("C. Discard-stratified metastable fraction (main set)")
say("=" * 78)
dr = pd.read_csv(DATA / "discard_rates_all_models.csv")
dr = dr.groupby(["Model", "Opinion_A"]).discard_rate.mean().reset_index().rename(columns={"discard_rate": "mean_discard"})
fc2 = pd.read_csv(DATA / "fits_with_ci.csv")
fc2 = fc2[fc2.Opinion_A.isin(off_pairs)]
mg = fc2.merge(dr, on=["Model", "Opinion_A"], how="inner")
mg["meta"] = mg.classification == "metastable"
say(f"n combos with discard data: {len(mg)}")
for lo, hi in [(0, 0.01), (0.01, 0.05), (0.05, 0.30), (0.30, 1.01)]:
    s = mg[(mg.mean_discard >= lo) & (mg.mean_discard < hi)]
    say(f"  mean discard in [{100*lo:.0f}%, {100*hi:.0f}%): n={len(s):4d}  metastable={100*s.meta.mean():5.1f}%  robust-unresolved={100*(~s.robust_2se).mean():5.1f}%  median|h|={s.h.abs().median():.2f}")
low = mg[mg.mean_discard < 0.05]
say(f"Headline restricted to combos with mean discard <5%: {100*low.meta.mean():.1f}% metastable (n={len(low)}) vs {100*mg.meta.mean():.1f}% overall")
say(f"Spearman(mean discard, metastable): {mg[['mean_discard','meta']].astype(float).corr(method='spearman').iloc[0,1]:.3f}")

# --------------------------------------------------------------- D (held-out sets)
say()
say("=" * 78)
say(f"D. Bootstrap classification robustness for the new pair sets ({N_BOOT} replicates)")
say("=" * 78)


def fit_wls(m, y, sigma=None):
    f = lambda m, beta, h, dP: 0.5 * np.tanh(beta * (m + h)) + 0.5 + dP
    kw = dict(p0=[1.0, 0.0, 0.0], maxfev=3000, bounds=([-np.inf, -np.inf, -1e-4], [np.inf, np.inf, 1e-4]))
    if sigma is not None:
        kw.update(sigma=sigma, absolute_sigma=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, _ = curve_fit(f, m, y, **kw)
        return popt
    except Exception:
        return None


sets = {
    "Pew 2017 typology": pd.read_csv(DATA / "pew_typology_2017_pairs.csv").Opinion_A.tolist(),
    "GlobalOpinionQA": pd.read_csv(DATA / "grounded_opinion_pairs_globalopinionqa.csv").Opinion_A.tolist(),
    "Safety-grounded": pd.read_csv(DATA / "ai_safety_opinion_pairs.csv").Opinion_A.tolist(),
}
brows = []
for sname, plist in sets.items():
    for model in MODELS:
        out_dir = DATA / model / "results_batched_vllm_explicit_v2" / "N=50"
        for a in plist:
            fp = find_file(out_dir, a)
            if fp is None:
                continue
            d = parse_txt(fp)
            if d is None or len(d) < 4:
                continue
            m = d.m.values.astype(float); n = (d.cA + d.cB).values.astype(int); p = (d.cA / (d.cA + d.cB)).values
            sig0 = np.sqrt(np.clip(p * (1 - p), 1e-4, None) / n)
            pt = fit_wls(m, p, sig0)
            if pt is None or abs(pt[0]) > 500 or abs(pt[1]) > 500:
                continue
            cls0 = tanh_class(pt[0], pt[1])
            agree_n, tot = 0, 0
            for _ in range(N_BOOT):
                yb = rng.binomial(n, np.clip(p, 0, 1)) / n
                sb = np.sqrt(np.clip(yb * (1 - yb), 1e-4, None) / n)
                pb = fit_wls(m, yb, sb)
                if pb is None or abs(pb[0]) > 500 or abs(pb[1]) > 500:
                    continue
                tot += 1
                agree_n += int(tanh_class(pb[0], pb[1]) == cls0)
            frac = agree_n / tot if tot else np.nan
            brows.append(dict(set=sname, Model=model, Opinion_A=a, beta=pt[0], h=pt[1],
                              metastable=cls0, boot_agree=frac, n_boot_ok=tot))
bdf = pd.DataFrame(brows)
bdf["robust"] = bdf.boot_agree >= 0.95
bdf.to_csv(DATA / "heldout_bootstrap.csv", index=False)
say("Robust = point classification reproduced in >=95% of bootstrap replicates (~2 sigma).")
for sname, g in bdf.groupby("set", sort=False):
    say(f"{sname:20s} fits={len(g):3d}  metastable={100*g.metastable.mean():5.1f}%  "
        f"robust metastable={100*(g.metastable & g.robust).mean():5.1f}%  robust monostable={100*(~g.metastable & g.robust).mean():5.1f}%  "
        f"unresolved={100*(~g.robust).mean():5.1f}%")
allh = bdf[bdf.set != "Safety-grounded"]
say(f"Pooled held-out (Pew+GOQA): fits={len(allh)} metastable={100*allh.metastable.mean():.1f}% robust metastable={100*(allh.metastable & allh.robust).mean():.1f}% unresolved={100*(~allh.robust).mean():.1f}%")

(DATA / "hardening_checks_summary.txt").write_text("\n".join(lines) + "\n")
print("\nwritten:", DATA / "hardening_checks_summary.txt")

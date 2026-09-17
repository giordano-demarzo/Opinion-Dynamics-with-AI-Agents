#!/usr/bin/env python3
"""Analysis of the wording-variant experiment: does replacing the word "opinion"
in the prompt by "position", "view" or "stance" change P(m), (beta, h) or the
metastable classification? Same WLS fit as the paper.
Outputs: data/wording_variants_fits.csv, data/wording_variants_summary.txt,
         figures/wording_variants.pdf/.png, manuscript/SI_tables/wording_variants.tex"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
RES = Path("/home/jovyan/LLMs_opinion_dynamics_bias/resubmission"); D = RES / "data" / "wording_variants"
NAMES = {"gemma-3-27b-it": "Gemma 3 27B", "Qwen3-32B": "Qwen3 32B", "Llama-3.1-8B-Instruct": "Llama 3.1 8B"}
VARIANTS = ["opinion", "position", "view", "stance"]

def f(m, beta, h, dP): return 0.5 * np.tanh(beta * (m + h)) + 0.5 + dP
def hsp(b):
    if b <= 1: return np.nan
    ms = np.sqrt(1 - 1 / b); return ms - np.arctanh(ms) / b
def meta(b, h): return bool(b > 1 and abs(h) < hsp(b))
def fit(m, p, n):
    sig = np.sqrt(np.clip(p * (1 - p), 1e-4, None) / n)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, _ = curve_fit(f, m, p, p0=[1, 0, 0], sigma=sig, absolute_sigma=True, maxfev=3000,
                                bounds=([-np.inf, -np.inf, -1e-4], [np.inf, np.inf, 1e-4]))
        return popt[0], popt[1]
    except Exception:
        return np.nan, np.nan

rows, curves = [], {}
for mdir in sorted(D.glob("*")):
    for vdir in sorted(mdir.glob("*")):
        for fp in sorted(vdir.glob("transition_prob_50_0.2_*.txt")):
            d = pd.read_csv(fp); n = d.count_A + d.count_B; d = d[n > 0]
            m = d.m0.values; p = (d.count_A / (d.count_A + d.count_B)).values; nn = (d.count_A + d.count_B).values
            ok = nn >= 50                      # points with enough valid replies to estimate P
            b, h = fit(m[ok], p[ok], nn[ok]) if ok.sum() >= 6 else (np.nan, np.nan)
            a = fp.name.replace("transition_prob_50_0.2_", "").replace(".txt", "")
            # identifier-echo share among invalid replies, from the raw log when present
            echo = np.nan
            raw = fp.with_name(f"raw_{a}.jsonl")
            if raw.exists():
                import json, re
                inv = [json.loads(l)["response"].strip() for l in open(raw) if not json.loads(l)["valid"]]
                if inv: echo = np.mean([bool(re.fullmatch(r"\[\s*[A-Za-z0-9]{2}\s*\]", r)) for r in inv])
            rows.append(dict(Model=mdir.name, variant=vdir.name, Opinion_A=a, beta=b, h=h, metastable=meta(b, h) if np.isfinite(b) else np.nan,
                             valid_rate=nn.mean() / 100, points_ok=int(ok.sum()), echo_share=echo))
            curves[(mdir.name, vdir.name, a)] = (m, p, nn)
df = pd.DataFrame(rows); df.to_csv(RES / "data" / "wording_variants_fits.csv", index=False)
L = []
def say(s=""): print(s); L.append(s)
say(f"fits: {len(df)}  models: {df.Model.nunique()}  pairs: {df.Opinion_A.nunique()}  variants: {sorted(df.variant.unique())}")
base = df[df.variant == "opinion"].set_index(["Model", "Opinion_A"])
tab = []
for v in VARIANTS[1:]:
    alt = df[df.variant == v].set_index(["Model", "Opinion_A"])
    jall = base.join(alt, lsuffix="_o", rsuffix="_v", how="inner")
    vr = jall.valid_rate_v.mean()
    j = jall.dropna(subset=["beta_o", "beta_v"])
    dP = []
    for (mod, a) in j.index:
        mo, po, no = curves[(mod, "opinion", a)]; mv, pv, nv = curves[(mod, v, a)]
        k = min(len(po), len(pv)); ok = (no[:k] >= 50) & (nv[:k] >= 50)
        if ok.sum(): dP.append(np.abs(po[:k][ok] - pv[:k][ok]).mean())
    agree = (j.metastable_o == j.metastable_v).mean() if len(j) else np.nan
    say(f"{v:9s} n={len(jall):3d} | valid-reply rate={100*vr:.0f}% | curves fittable={len(j)} | mean|dP| (valid pts)={np.mean(dP) if dP else float('nan'):.3f} | Spearman h={j[['h_o','h_v']].corr(method='spearman').iloc[0,1] if len(j)>2 else float('nan'):.2f} | median|dh|={(j.h_v-j.h_o).abs().median() if len(j) else float('nan'):.3f} | classification agreement={100*agree:.0f}%")
    tab.append((v, len(jall), 100*vr, len(j), np.mean(dP) if dP else np.nan, (j.h_v-j.h_o).abs().median() if len(j) else np.nan, 100*agree))
say("\nper model (all variants pooled vs original):")
for mod in df.Model.unique():
    b0 = df[(df.Model == mod) & (df.variant == "opinion")].set_index("Opinion_A")
    for v in VARIANTS[1:]:
        a1 = df[(df.Model == mod) & (df.variant == v)].set_index("Opinion_A"); j = b0.join(a1, lsuffix="_o", rsuffix="_v", how="inner")
        jj = j.dropna(subset=["beta_o", "beta_v"])
        say(f"  {NAMES.get(mod, mod):14s} {v:9s} valid-reply rate={100*a1.valid_rate.mean():.0f}%  identifier-echo share of invalid={100*a1.echo_share.mean():.0f}%  fittable={len(jj)}/{len(j)}  agreement={100*(jj.metastable_o==jj.metastable_v).mean() if len(jj) else float('nan'):.0f}%  median|dh|={(jj.h_v-jj.h_o).abs().median() if len(jj) else float('nan'):.3f}")
say(f"\nvalid-reply rate by variant: " + ", ".join(f"{v} {100*df[df.variant==v].valid_rate.mean():.0f}%" for v in VARIANTS))
(RES / "data" / "wording_variants_summary.txt").write_text("\n".join(L) + "\n")
# SI table fragment
lines = [f"{v} & {n} & {vr:.0f}\\% & {nf} & {dp:.3f} & {dh:.2f} & {ag:.0f}\\% \\\\" for v, n, vr, nf, dp, dh, ag in tab] + ["\\bottomrule"]
(RES / "manuscript" / "SI_tables" / "wording_variants.tex").write_text("\n".join(lines) + "\n")
# figure: P(m) curves per model for 3 pairs, 4 variants
pairs = sorted(df.Opinion_A.unique()); show = [pairs[i] for i in np.linspace(0, len(pairs) - 1, 3).round().astype(int)]
models = [m for m in NAMES if m in df.Model.unique()]
fig, axs = plt.subplots(len(models), 3, figsize=(10, 2.9 * len(models)), squeeze=False)
cols = {"opinion": "black", "position": "#D62728", "view": "#3B6FB6", "stance": "#2CA02C"}
for i, mod in enumerate(models):
    for jx, a in enumerate(show):
        ax = axs[i, jx]
        for v in VARIANTS:
            if (mod, v, a) in curves:
                m, p, nn = curves[(mod, v, a)]; ok = nn >= 50; o = np.argsort(m[ok]); ax.plot(m[ok][o], p[ok][o], "o-", ms=3, lw=1, color=cols[v], label=v)
        ax.set_ylim(-0.03, 1.03); ax.set_title(f"{NAMES[mod]}: {a}", fontsize=8.5)
        if i == len(models) - 1: ax.set_xlabel("$m_0$")
        if jx == 0: ax.set_ylabel("P(adopt A)")
axs[0, 0].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(RES / "figures" / "wording_variants.pdf", bbox_inches="tight"); fig.savefig(RES / "figures" / "wording_variants.png", dpi=170, bbox_inches="tight")
print("saved")

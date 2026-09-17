#!/usr/bin/env python3
"""Reply figure: per-pair metastable fraction vs median |h| across models, all four pair sets."""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
BASE = Path("/home/jovyan/LLMs_opinion_dynamics_bias/resubmission")
off = set(pd.read_csv(BASE.parent/"submission"/"opinion_pairs.csv").Opinion_A)

def perpair(df, label):
    g = df.groupby("Opinion_A").agg(med_h=("h", lambda x: np.median(np.abs(x))),
                                    meta=("classification", lambda c: (c=="metastable").mean()),
                                    n=("h","size")).reset_index()
    g["set"]=label; return g

main = pd.read_csv(BASE/"data"/"fits_with_ci.csv"); main = main[main.Opinion_A.isin(off)]
new = pd.read_csv(BASE/"data"/"new_pairs_fits.csv")
pew = pd.read_csv(BASE/"data"/"pew_pairs_fits.csv")
sets = [perpair(main,"Main set (100 pairs)"),
        perpair(new[new.subset=="grounded_general"],"GlobalOpinionQA (15)"),
        perpair(pew,"Pew 2017 typology (12)"),
        perpair(new[new.subset=="ai_safety"],"Safety-grounded (20)")]
allp = pd.concat(sets)
colors = {"Main set (100 pairs)":"#7F7F7F","GlobalOpinionQA (15)":"#2CA02C","Pew 2017 typology (12)":"#D62728","Safety-grounded (20)":"#3B6FB6"}
markers = {"Main set (100 pairs)":"o","GlobalOpinionQA (15)":"s","Pew 2017 typology (12)":"D","Safety-grounded (20)":"^"}
fig, ax = plt.subplots(figsize=(7.0,4.4))
ax.axvspan(1.0, 300, color="#B0B0B0", alpha=0.18, zorder=0)
ax.text(1.3, 90, r"$|h|\geq 1$: outside the metastable" "\n" r"region for any $\beta$", fontsize=9.5, color="#444")
for g in sets:
    lab = g.set.iloc[0]
    ax.scatter(g.med_h.clip(lower=0.03), 100*g.meta, s=40 if "Main" in lab else 60, marker=markers[lab],
               color=colors[lab], alpha=0.75 if "Main" in lab else 0.9, edgecolors="black", linewidths=0.4, label=lab, zorder=3)
# binned trend over everything
bins = np.array([0.03,0.1,0.2,0.35,0.5,0.75,1.0,2.0,5.0,300])
allp["bin"]=pd.cut(allp.med_h.clip(lower=0.03), bins)
tr = allp.groupby("bin", observed=True).agg(x=("med_h", lambda v: np.exp(np.mean(np.log(v.clip(lower=0.03))))), y=("meta","mean"), n=("meta","size")).reset_index()
ax.plot(tr.x, 100*tr.y, color="black", lw=1.4, ls="--", zorder=4, label="Binned mean (all sets)")
ax.set_xscale("log"); ax.set_xlim(0.025, 200); ax.set_ylim(-4, 104)
ax.set_xlabel(r"Median bias magnitude $|h|$ across models (per pair)", fontsize=11.5)
ax.set_ylabel("Metastable fraction across models (%)", fontsize=11.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(fontsize=8.5, frameon=False, loc="upper right", bbox_to_anchor=(1.0,0.82))
fig.tight_layout()
fig.savefig(BASE/"figures"/"spine_all_sets.pdf", bbox_inches="tight")
fig.savefig(BASE/"figures"/"spine_all_sets.png", dpi=220, bbox_inches="tight")
# summary numbers
for g in sets:
    print(g.set.iloc[0], "pairs", len(g), "meta of |h|<0.5 pairs: %.2f" % g[g.med_h<0.5].meta.mean(), " |h|>=1: %.2f"%(g[g.med_h>=1].meta.mean() if (g.med_h>=1).any() else float('nan')))
fits = pd.concat([main[["h","classification"]], new[["h","classification"]], pew[["h","classification"]]])
for lo,hi in [(0,0.25),(0.25,0.5),(0.5,0.75),(0.75,1),(1,1e9)]:
    s = fits[(fits.h.abs()>=lo)&(fits.h.abs()<hi)]
    print("fits |h| in [%g,%g): n=%d meta=%.3f"%(lo,hi,len(s),(s.classification=="metastable").mean()))

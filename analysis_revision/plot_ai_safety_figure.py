#!/usr/bin/env python3
"""Manuscript Fig. 5: AI-safety-grounded pairs, metastable fraction vs median |h|.
Style matched to the main figures (plot3.py): serif font, STIX math, light grid,
gray metastable shading with dashed boundary, blue markers with white edges,
boxed annotations."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "font.family": "serif",
    "mathtext.fontset": "stix",
})

BASE = Path("/home/jovyan/LLMs_opinion_dynamics_bias/resubmission")
g = pd.read_csv(BASE / "data" / "ai_safety_per_pair_median.csv")

BLUE = "#1f77b4"
XMIN, XMAX = 0.07, 150

fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.grid(True, alpha=0.3, zorder=0)

# metastable-excluded region: |h| >= 1 is monostable for any beta
ax.axvspan(1.0, XMAX, color="gray", alpha=0.15, zorder=0,
           label=r"$|h|\geq 1$: monostable for any $\beta$")
ax.axvline(1.0, color="black", ls="--", lw=2, zorder=1)

ax.scatter(g["med_h"], 100 * g["meta"], s=90, color=BLUE, alpha=0.8,
           edgecolors="white", linewidths=0.8, zorder=5)

# boxed annotations, as in Fig. 3(a)
bbox = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
            edgecolor=BLUE, linewidth=1)
arrow = dict(arrowstyle="->", color="black", lw=1,
             connectionstyle="arc3,rad=0.2")
annots = {
    "data confidentiality":       (0.30, 88),
    "refuse content for minors":  (0.55, 72),
    "respect access barriers":    (0.085, 44),
    "accept shutdown":            (3.5, 46),
    "refuse malicious code":      (9.0, 30),
    "refuse phishing request":    (9.0, 8),
    "disclose uncertainty":       (3.0, 18),
}
for _, row in g.iterrows():
    name = row["Opinion_A"]
    if name in annots:
        ax.annotate(name, xy=(row["med_h"], 100 * row["meta"]),
                    xytext=annots[name], fontsize=11,
                    bbox=bbox, arrowprops=arrow, zorder=6)

ax.set_xscale("log")
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(-4, 100)
ax.set_xlabel(r"Median bias magnitude $|h|$ across models")
ax.set_ylabel("Metastable fraction (%)")
ax.legend(loc="upper right", frameon=True, framealpha=0.9)

fig.tight_layout()
fig.savefig(BASE / "figures" / "ai_safety_bimodal.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(BASE / "figures" / "ai_safety_bimodal.png", dpi=300, bbox_inches="tight", facecolor="white")
print("saved")

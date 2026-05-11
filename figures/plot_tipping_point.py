"""Plot tipping-point experiment results: 3 runs per pair, vertical lines at t1/t2."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent.parent / "data"

FILES = [
    {
        "path": BASE / "gemma-3-27b-it/results_tipping_point/tipping_gender self-identification_Ns50_T0.20.json",
        "title": "Gender self-identification\nvs Biological sex classification\n(inside spinodal)",
        "color": "#e15759",
    },
    {
        "path": BASE / "gemma-3-27b-it/results_tipping_point/tipping_renewable energy_Ns225_T0.20.json",
        "title": "Renewable energy vs Fossil fuels\n(outside spinodal)",
        "color": "#4e79a7",
    },
]

fig, ax = plt.subplots(figsize=(8, 5))

t1 = t2 = total_blocks = None

for cfg in FILES:
    d = json.load(open(cfg["path"]))

    t1 = d["t1"]
    t2 = d["t2"]
    trajectories = d["trajectories"]
    mean = d["mean"]
    total_blocks = len(trajectories[0])
    Ns = d["N_stubborn_B"]

    blocks = np.arange(1, total_blocks + 1)

    label_base = cfg["title"].split("\n")[0]

    # Individual runs — thin, semi-transparent
    for r, traj in enumerate(trajectories):
        ax.plot(blocks, traj, color=cfg["color"], lw=1.2, alpha=0.35,
                label=f"{label_base} (run)" if r == 0 else None)

    # Mean — thick
    ax.plot(blocks, mean, color=cfg["color"], lw=2.5,
            label=f"{label_base} (mean), $N_s={Ns}$")

# Vertical lines at t1, t2 (same for both datasets)
ax.axvline(t1 + 0.5, color="k", lw=1.5, linestyle="--", alpha=0.7)
ax.axvline(t2 + 0.5, color="k", lw=1.5, linestyle="--", alpha=0.7)

# Phase labels at top
ytop = 1.04
ax.text((t1 + 0.5) / 2,                   ytop, "Free",     ha="center", fontsize=10, va="bottom")
ax.text((t1 + t2) / 2 + 0.5,              ytop, "Stubborn", ha="center", fontsize=10, va="bottom")
ax.text((t2 + 0.5 + total_blocks) / 2,    ytop, "Free",     ha="center", fontsize=10, va="bottom")

ax.axhline(0, color="k", lw=0.8, alpha=0.3, linestyle=":")
ax.set_xlabel("Sweep", fontsize=13)
ax.set_ylabel("Collective opinion $m$", fontsize=13)
ax.set_xlim(0.5, total_blocks + 0.5)
ax.set_ylim(-1.2, 1.1)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="lower left")

plt.tight_layout()
out = BASE / "tipping_point_plot.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

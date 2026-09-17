#!/usr/bin/env python3
"""Figure: isolation baseline vs balanced-start collective outcome.

Panel (a): scatter of isolated p_A against the fitted balanced-start
collective outcome P(m=0), identity line, shaded sign-disagreement
quadrants. Panel (b): per-model fraction of (model, pair) combinations
whose isolated and collective majorities disagree in direction.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/home/jovyan/LLMs_opinion_dynamics_bias/resubmission")
df = pd.read_csv(BASE / "data" / "isolation_vs_balanced.csv")
df = df.dropna(subset=["p_A", "p0_fit"])

BLUE = "#3B6FB6"
GRAY = "#666666"
SHADE = "#B0B0B0"

MODEL_LABELS = {
    "Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "Qwen2.5-14B-Instruct": "Qwen 2.5 14B",
    "Qwen2.5-32B-Instruct": "Qwen 2.5 32B",
    "Qwen3-14B": "Qwen 3 14B",
    "Qwen3-32B": "Qwen 3 32B",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "gemma-3-12b-it": "Gemma 3 12B",
    "gemma-3-27b-it": "Gemma 3 27B",
    "gpt-5-mini": "GPT-5 mini",
}

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), gridspec_kw=dict(width_ratios=[1.05, 1]))

# --- panel (a): scatter -------------------------------------------------
ax = axes[0]
ax.axhspan(0.5, 1.0, xmax=0.5, color=SHADE, alpha=0.25, zorder=0)
ax.axhspan(0.0, 0.5, xmin=0.5, color=SHADE, alpha=0.25, zorder=0)
ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1, zorder=1)
ax.axhline(0.5, color=GRAY, linewidth=0.6, alpha=0.5, zorder=1)
ax.axvline(0.5, color=GRAY, linewidth=0.6, alpha=0.5, zorder=1)
ax.scatter(df["p_A"], df["p0_fit"], s=14, color=BLUE, alpha=0.35,
           edgecolors="none", zorder=2)

disagree = ((df["p_A"] - 0.5) * (df["p0_fit"] - 0.5) < 0).mean()
r = np.corrcoef(df["p_A"], df["p0_fit"])[0, 1]
ax.text(0.03, 0.71, "majority\ndisagrees", fontsize=9, color="#444444", ha="left")
ax.text(0.97, 0.27, "majority\ndisagrees", fontsize=9, color="#444444", ha="right")
ax.text(0.03, 0.955, f"$r = {r:.2f}$; direction disagrees for {100*disagree:.1f}%",
        fontsize=10, ha="left", va="top", transform=ax.transAxes)

ax.set_xlabel(r"Isolated response  $p_A$  (no social information)", fontsize=12)
ax.set_ylabel(r"Balanced-start collective outcome  $P(m{=}0)$", fontsize=12)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_title("(a)", fontsize=12, loc="left", fontweight="bold")

# --- panel (b): per-model disagreement fraction -------------------------
ax = axes[1]
per_model = (
    df.assign(flip=(df["p_A"] - 0.5) * (df["p0_fit"] - 0.5) < 0)
      .groupby("Model")["flip"].mean()
      .sort_values()
)
labels = [MODEL_LABELS.get(m, m) for m in per_model.index]
y = np.arange(len(per_model))
ax.barh(y, 100 * per_model.values, height=0.62, color=BLUE, alpha=0.85)
for yi, v in zip(y, per_model.values):
    ax.text(100 * v + 1.0, yi, f"{100*v:.0f}%", va="center", fontsize=9, color="#333333")
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.axvline(100 * disagree, color=GRAY, linestyle="--", linewidth=1)
ax.text(100 * disagree + 0.7, len(y) - 0.35, "mean", fontsize=9, color=GRAY)
ax.set_xlabel("Majority direction disagreement (% of pairs)", fontsize=12)
ax.set_xlim(0, max(100 * per_model.values) * 1.22)
ax.set_title("(b)", fontsize=12, loc="left", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(BASE / "figures" / "isolation_vs_collective.pdf", bbox_inches="tight")
fig.savefig(BASE / "figures" / "isolation_vs_collective.png", dpi=250, bbox_inches="tight")
print("saved; disagreement=%.3f r=%.3f n=%d" % (disagree, r, len(df)))

#!/usr/bin/env python3
"""Regenerate SI phase-diagram figure (all nine models) with the standardized
"Metastable region" label (referee 2, minor point vi) from fits_with_ci.csv."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/home/jovyan/LLMs_opinion_dynamics_bias/resubmission")
f = pd.read_csv(BASE / "data" / "fits_with_ci.csv")
official = set(pd.read_csv("/home/jovyan/LLMs_opinion_dynamics_bias/submission/opinion_pairs.csv").Opinion_A)
f = f[f.Opinion_A.isin(official)]

ORDER = [
    ("gemma-3-27b-it", "Gemma 3 27B", "o", "#1f77b4"),
    ("gemma-3-12b-it", "Gemma 3 12B", "s", "#aec7e8"),
    ("Llama-3.1-8B-Instruct", "Llama 3.1 8B", "^", "#ff7f0e"),
    ("Qwen3-32B", "Qwen3 32B", "p", "#d62728"),
    ("Qwen3-14B", "Qwen3 14B", "h", "#ff9896"),
    ("Qwen2.5-32B-Instruct", "Qwen2.5 32B", "v", "#2ca02c"),
    ("Qwen2.5-14B-Instruct", "Qwen2.5 14B", "D", "#98df8a"),
    ("gpt-5-mini", "GPT-5 mini", "X", "#8c564b"),
    ("gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite", "*", "#9467bd"),
]

beta_line = np.linspace(1.0001, 14, 400)
m_sp = np.sqrt(1 - 1 / beta_line)
h_sp = m_sp - np.arctanh(m_sp) / beta_line

fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.5), sharex=True, sharey=True)
letters = "abcdefghi"
for ax, (key, label, marker, color), letter in zip(axes.flat, ORDER, letters):
    g = f[f.Model == key]
    ax.fill_between(beta_line, 0, h_sp, color="0.85", alpha=0.7, zorder=0)
    ax.plot(beta_line, h_sp, "--", color="gray", linewidth=1.2, zorder=1,
            label="Spinodal boundary" if letter == "a" else None)
    ax.scatter(g.beta, g.h.abs(), s=26, marker=marker, color=color, alpha=0.65,
               edgecolors="none", zorder=2)
    ax.axvline(g.beta.median(), color=color, linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3.2)
    ax.set_title(f"{label}  ($n={len(g)}$)", fontsize=12)
    ax.text(-0.08, 1.05, f"({letter})", transform=ax.transAxes,
            fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25, linewidth=0.5)

for ax in axes[-1]:
    ax.set_xlabel(r"$\beta$", fontsize=13)
for ax in axes[:, 0]:
    ax.set_ylabel(r"$|h|$", fontsize=13)

handles = [plt.Rectangle((0, 0), 1, 1, color="0.85"),
           plt.Line2D([], [], linestyle="--", color="gray")]
fig.legend(handles, ["Metastable region", "Spinodal boundary"],
           loc="lower center", ncol=2, fontsize=12, frameon=True,
           bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(BASE / "figures" / "SI_phase_diagram_all_models.pdf", bbox_inches="tight")
fig.savefig(BASE / "figures" / "SI_phase_diagram_all_models.png", dpi=180, bbox_inches="tight")
print("saved")

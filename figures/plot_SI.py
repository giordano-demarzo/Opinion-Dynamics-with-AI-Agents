#!/usr/bin/env python3
"""
plot_SI_robustness.py  (extended)

Generates all Supplementary Information figures, fitting P(m) data fresh from
results_batched_vllm/N=50/ for each model.  All outputs go to SI_figures/.

Figures produced
----------------
  SI_robustness_temperature.pdf   — P(m) for 3 temperatures        (Gemma + Qwen3)
  SI_robustness_prompt.pdf        — P(m) for 5 prompt variants      (3 robustness models)
  SI_robustness_system_size.pdf   — P(m) for 5 system sizes + β,h  (3 robustness models)
  SI_opinion_pairs_table.tex      — LaTeX longtable of all opinion pairs
  SI_phase_diagram_all_models.pdf — Phase diagram for all 9 models  (3×3 grid)
  SI_model_correlation.pdf        — Model–model h correlation matrix
  SI_hysteresis_examples.pdf      — Example hysteresis cycles        (3×3 grid)
  SI_family_ratios.pdf            — β and h scatter within model families
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import pearsonr
import seaborn as sns

# =============================================================================
# PATHS
# =============================================================================

BASE      = Path(__file__).parent.parent / 'data' / 'results_robustness'   # robustness sweep data
BASE_DATA = Path(__file__).parent.parent / 'data'                    # main model dirs live here
OUT_DIR   = Path("SI_figures")
OUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# ROBUSTNESS EXPERIMENT CONFIG
# =============================================================================

MODELS_ROBUSTNESS = [
    {"short": "gemma-3-27b-it", "label": "Gemma 3 27B", "color": "#e15759"},
    {"short": "Qwen3-32B",      "label": "Qwen3 32B",   "color": "#4e79a7"},
    {"short": "gpt-5-mini",     "label": "GPT-5 mini",  "color": "#59a14f"},
]
MODELS_NO_GPT = MODELS_ROBUSTNESS[:2]

FOCAL_PAIRS = [
    ("gender self-identification", "biological sex classification"),
    ("renewable energy",           "fossil fuels"),
    ("climate change believer",    "climate change skeptic"),
]
PAIR_LABELS = ["Gender self-id.", "Renewable energy", "Climate change"]

TEMPERATURES    = [0.1, 0.2, 0.5]
SYSTEM_SIZES    = [20, 50, 100, 200, 500]
PROMPT_VARIANTS = [0, 1, 2, 3, 4]
PROMPT_NAMES    = ["Original", "Peer", "Survey", "Discussion", "Minimal"]

DEFAULT_T  = 0.2
DEFAULT_N  = 50
DEFAULT_PV = 0

T_COLORS  = ["#2166ac", "#f4a582", "#d6604d"]
PV_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]
N_COLORS  = ["#d9f0a3", "#addd8e", "#41ab5d", "#006837", "#00441b"]
PAIR_MARKERS = ["o", "s", "^"]

# =============================================================================
# ALL-MODELS CONFIG  (for new SI figures)
# =============================================================================

ALL_MODELS = [
    {"short": "gemma-3-27b-it",        "label": "Gemma 3 27B",      "color": "#1f77b4", "marker": "o",  "family": "Gemma"},
    {"short": "gemma-3-12b-it",        "label": "Gemma 3 12B",      "color": "#aec7e8", "marker": "s",  "family": "Gemma"},
    {"short": "Llama-3.1-8B-Instruct", "label": "Llama 3.1 8B",     "color": "#ff7f0e", "marker": "^",  "family": "Llama"},
    {"short": "Qwen3-32B",             "label": "Qwen3 32B",        "color": "#d62728", "marker": "p",  "family": "Qwen3"},
    {"short": "Qwen3-14B",             "label": "Qwen3 14B",        "color": "#ff9896", "marker": "h",  "family": "Qwen3"},
    {"short": "Qwen2.5-32B-Instruct",  "label": "Qwen2.5 32B",     "color": "#2ca02c", "marker": "v",  "family": "Qwen2.5"},
    {"short": "Qwen2.5-14B-Instruct",  "label": "Qwen2.5 14B",     "color": "#98df8a", "marker": "D",  "family": "Qwen2.5"},
    {"short": "gpt-5-mini",            "label": "GPT-5 mini",       "color": "#8c564b", "marker": "X",  "family": "GPT"},
    {"short": "gemini-2.5-flash-lite", "label": "Gemini 2.5 Flash", "color": "#9467bd", "marker": "*",  "family": "Gemini"},
]

# 3×3 arrangement for phase diagram
MODELS_GRID = [
    [ALL_MODELS[0], ALL_MODELS[1], ALL_MODELS[2]],   # Gemma 27B | Gemma 12B | Llama 8B
    [ALL_MODELS[3], ALL_MODELS[4], ALL_MODELS[5]],   # Qwen3-32B | Qwen3-14B | Qwen2.5-32B
    [ALL_MODELS[6], ALL_MODELS[7], ALL_MODELS[8]],   # Qwen2.5-14B | GPT-5 | Gemini
]

# Hysteresis data directories (relative to BASE_DATA)
HYSTERESIS_DIRS = {
    "gemma-3-27b-it":        "gemma-3-27b-it/results_hysteresis_constant_size_vllm/N=80",
    "gemma-3-12b-it":        "gemma-3-12b-it/results_hysteresis_vllm/N=50",
    "Qwen3-32B":             "Qwen3-32B/results_hysteresis_vllm/N=50",
    "Qwen2.5-32B-Instruct":  "Qwen2.5-32B-Instruct/results_hysteresis_vllm/N=50",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct/results_hysteresis_vllm/N=50",
    "gpt-5-mini":            "gpt-5-mini/results_hysteresis_openai/N=50",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite/results_hysteresis_gemini/N=50",
}

# Within-family model pairs for the ratio figure: (large, small, label_large, label_small)
FAMILY_PAIRS = [
    ("gemma-3-27b-it",       "gemma-3-12b-it",        "Gemma 3 27B", "Gemma 3 12B"),
    ("Qwen3-32B",            "Qwen3-14B",              "Qwen3 32B",   "Qwen3 14B"),
    ("Qwen2.5-32B-Instruct", "Qwen2.5-14B-Instruct",  "Qwen2.5 32B", "Qwen2.5 14B"),
]

# =============================================================================
# PLOT STYLE
# =============================================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.family": "serif",
    "mathtext.fontset": "stix",
})

# =============================================================================
# CORE FITTING  (improved multi-start version)
# =============================================================================

def fit_func(m, beta, h, dp):
    """Tanh fit: P(m) = 0.5*tanh(beta*(m+h)) + 0.5 + dp"""
    return 0.5 * np.tanh(beta * (m + h)) + 0.5 + dp


def fit_beta_h(df):
    """
    Fit beta and h from a P(m) DataFrame.

    Uses the same bounds as plot2.py: ([-inf,-inf,-0.1], [inf,inf,0.1]).
    Extends plot2.py with multiple starting points for better convergence
    (addresses poor fits at extreme temperatures / biases).

    Returns (beta, h) or (None, None) on failure.
    """
    if df is None or len(df) < 4:
        return None, None

    x = df["m0"].values
    y = df["probability"].values

    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None, None

    best_beta, best_h, best_mse = None, None, np.inf

    # Same bounds as plot2.py; multiple starting points for robustness
    bounds = ([-np.inf, -np.inf, -0.1], [np.inf, np.inf, 0.1])
    p0_list = [
        [2.0,  0.0,  0.0],   # plot2.py default
        [1.0,  0.5,  0.0],
        [1.0, -0.5,  0.0],
        [5.0,  0.0,  0.0],
        [0.5,  0.0,  0.0],
        [3.0,  0.3,  0.0],
        [3.0, -0.3,  0.0],
        [1.5,  1.0,  0.0],
        [1.5, -1.0,  0.0],
    ]

    for p0 in p0_list:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    fit_func, x, y,
                    p0=p0,
                    bounds=bounds,
                    maxfev=5000,
                )
            beta, h, dp = popt
            mse = np.mean((y - fit_func(x, *popt)) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_beta, best_h = beta, h
        except Exception:
            continue

    return best_beta, best_h


def theory_curve(beta, h, n=300):
    m = np.linspace(-1, 1, n)
    return m, 0.5 * np.tanh(beta * (m + h)) + 0.5


def spinodal_line(beta_max=15):
    """Return (beta_array, h_array) along the spinodal boundary."""
    betas = np.linspace(1.001, beta_max, 2000)
    hs = []
    for b in betas:
        m_s = np.sqrt(max(0, 1.0 - 1.0 / b))
        if m_s > 0:
            hs.append(abs(-m_s + np.arctanh(m_s) / b))
        else:
            hs.append(np.nan)
    arr = np.array(hs)
    mask = np.isfinite(arr)
    return betas[mask], arr[mask]

# =============================================================================
# PAIR MAP AND GLOBAL FIT CACHE
# =============================================================================

_fits_cache = None


def build_pair_map():
    """
    Build {opinion_A: opinion_B} from 'plot_X_vs_Y.png' filenames in any
    model's results_batched_vllm/N=50 directory.
    """
    ref_dir = BASE_DATA / "gemma-3-27b-it" / "results_batched_vllm" / "N=50"
    pair_map = {}
    for png in sorted(ref_dir.glob("plot_*_vs_*.png")):
        stem = png.stem[5:]          # strip leading "plot_"
        idx  = stem.find("_vs_")
        if idx != -1:
            pair_map[stem[:idx]] = stem[idx + 4:]
    return pair_map


def _read_tp_file(fpath):
    """
    Read a transition-probability file in either format.

    Returns a DataFrame with columns ['m0', 'probability'] or None on failure.
    Matches plot2.py's reading logic exactly.
    """
    try:
        data = pd.read_csv(fpath)
        if "m0" in data.columns:
            x = data["m0"].astype(float).values
            y = data["probability"].astype(float).values
        else:
            x = data.iloc[:, 0].astype(float).values
            y = data.iloc[:, 3].astype(float).values
    except Exception:
        try:
            data = pd.read_csv(fpath, header=None)
            x = data.iloc[:, 0].astype(float).values
            col2 = data.iloc[:, 1].astype(float).values
            col3 = data.iloc[:, 2].astype(float).values
            denom = col2 + col3
            y = np.where(denom > 0, col2 / denom, np.nan)
        except Exception:
            return None

    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 4:
        return None
    return pd.DataFrame({"m0": x[valid], "probability": y[valid]})


def _extract_opinion_name(fpath):
    """
    Extract normalised opinion name from a transition_prob filename.
    Handles both naming conventions (plot2.py logic).
    """
    name = fpath.stem
    if "_opinions_" in name:
        # api format: transition_prob_50_opinions_<opinion_with_underscores>
        return name.split("_opinions_", 1)[1].replace("_", " ")
    else:
        # vllm format: transition_prob_50_0.2_<opinion with spaces>
        parts = name.split("_")
        if len(parts) > 4:
            return "_".join(parts[4:]).replace("_", " ")
    return None


def load_all_fits():
    """
    Fit P(m) curves for all 9 models from results_batched_vllm/N=50/.

    Filters to only the opinion pairs listed in opinion_pairs.csv.
    Uses the same fitting bounds as plot2.py.

    Returns a DataFrame with columns:
        model, label, family, color, marker,
        opinion_a, opinion_b, beta, h, abs_h, mse
    """
    # Load opinion pairs filter
    op_pairs = pd.read_csv(BASE_DATA / "opinion_pairs.csv")
    valid_A  = {s.lower().strip(): s for s in op_pairs["Opinion_A"]}
    valid_B  = {s.lower().strip(): s for s in op_pairs["Opinion_B"]}
    valid_all = set(valid_A) | set(valid_B)

    rows = []

    for mdl in ALL_MODELS:
        short    = mdl["short"]
        data_dir = BASE_DATA / short / "results_batched_vllm" / "N=50"
        if not data_dir.exists():
            print(f"  WARNING: {data_dir} not found, skipping.")
            continue

        n_ok = n_fail = n_skip = 0
        for fpath in sorted(data_dir.glob("transition_prob_*.txt")):
            opinion_a = _extract_opinion_name(fpath)
            if opinion_a is None:
                n_skip += 1
                continue

            key = opinion_a.lower().strip()

            # Filter to opinion_pairs.csv — match plot2.py logic
            if key not in valid_all:
                n_skip += 1
                continue

            # Resolve opinion_b from the CSV
            if key in valid_A:
                opinion_b = op_pairs.loc[
                    op_pairs["Opinion_A"].str.lower().str.strip() == key,
                    "Opinion_B"
                ].values[0]
            else:
                opinion_b = op_pairs.loc[
                    op_pairs["Opinion_B"].str.lower().str.strip() == key,
                    "Opinion_A"
                ].values[0]

            df = _read_tp_file(fpath)
            if df is None:
                n_fail += 1
                continue

            beta, h = fit_beta_h(df)
            if beta is None:
                n_fail += 1
                continue

            x   = df["m0"].values
            y   = df["probability"].values
            mse = float(np.mean((y - fit_func(x, beta, h, 0.0)) ** 2))

            rows.append({
                "model":     short,
                "label":     mdl["label"],
                "family":    mdl["family"],
                "color":     mdl["color"],
                "marker":    mdl["marker"],
                "opinion_a": opinion_a,
                "opinion_b": opinion_b,
                "beta":      beta,
                "h":         h,
                "abs_h":     abs(h),
                "mse":       mse,
            })
            n_ok += 1

        print(f"  {short}: {n_ok} fits OK, {n_fail} failed")

    df_all = pd.DataFrame(rows)
    print(f"\nTotal fits: {len(df_all)}")
    return df_all


def get_fits():
    global _fits_cache
    if _fits_cache is None:
        print("Computing fits from results_batched_vllm ...")
        _fits_cache = load_all_fits()
    return _fits_cache

# =============================================================================
# SHARED DRAWING HELPERS  (robustness figures)
# =============================================================================

def pair_tag(a, b):
    return f"{a}_vs_{b}".replace(" ", "_").replace("/", "-")


def load_tp(model_short, experiment, pair_A, pair_B, T, N, pv):
    path = (BASE / model_short / f"vary_{experiment}"
            / pair_tag(pair_A, pair_B) / f"T{T}_N{N}_pv{pv}.txt")
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def draw_pm(ax, df, color, alpha=0.85, zorder=2, label=None):
    """Plot P(m) data with error bars + dashed theory fit overlay."""
    if df is None:
        return
    ax.errorbar(df["m0"], df["probability"], yerr=df["standard_error"],
                fmt="o", color=color, ms=3, lw=1.2, capsize=2,
                alpha=alpha, zorder=zorder, label=label)
    beta, h = fit_beta_h(df)
    if beta is not None:
        m_th, p_th = theory_curve(beta, h)
        ax.plot(m_th, p_th, color=color, lw=1.3, ls="--", alpha=0.6, zorder=zorder - 1)


def style_ax(ax, row, col, n_rows, n_cols, row_label=None):
    ax.axhline(0.5, color="gray", lw=0.6, ls=":", alpha=0.5)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-0.06, 1.06)
    ax.grid(True, alpha=0.2)
    if row == n_rows - 1:
        ax.set_xlabel(r"Collective opinion $m$")
    else:
        ax.tick_params(labelbottom=False)
    if col == 0:
        ylabel = r"Transition probability $P(m)$"
        if row_label:
            ylabel = row_label + "\n" + ylabel
        ax.set_ylabel(ylabel, fontsize=10)
    else:
        ax.tick_params(labelleft=False)


def make_grid(models, n_extra_rows=0, extra_height=1.8, figsize_w=13):
    n_rows = len(models)
    n_cols = len(FOCAL_PAIRS)
    h_ratios  = [1.0] * n_rows + [extra_height] * n_extra_rows
    total_rows = n_rows + n_extra_rows
    fig_h = 2.9 * n_rows + extra_height * n_extra_rows + 1.2
    fig   = plt.figure(figsize=(figsize_w, fig_h))
    gs    = gridspec.GridSpec(total_rows, n_cols, figure=fig,
                              height_ratios=h_ratios,
                              hspace=0.18, wspace=0.08)
    axes  = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            share_x = axes[0, c] if r > 0 else None
            share_y = axes[r, 0] if c > 0 else None
            axes[r, c] = fig.add_subplot(gs[r, c],
                                         sharex=share_x, sharey=share_y)
    return fig, gs, axes


def add_column_titles(axes):
    for c, lbl in enumerate(PAIR_LABELS):
        axes[0, c].set_title(lbl, fontsize=11, pad=4)


def add_panel_labels(fig, axes, extra_axes=None):
    all_ax  = list(axes.flat)
    if extra_axes:
        all_ax += extra_axes
    letters = "abcdefghijklmnopqrstuvwxyz"
    for i, ax in enumerate(all_ax):
        ax.text(-0.13, 1.05, f"({letters[i]})",
                transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom", ha="left")

# =============================================================================
# FIGURE 1 — Temperature robustness
# =============================================================================

def make_temperature_figure():
    models = MODELS_NO_GPT
    fig, gs, axes = make_grid(models)
    add_column_titles(axes)

    for r, mdl in enumerate(models):
        for c, (pA, pB) in enumerate(FOCAL_PAIRS):
            ax = axes[r, c]
            for ti, T in enumerate(TEMPERATURES):
                df = load_tp(mdl["short"], "temperature", pA, pB, T, DEFAULT_N, DEFAULT_PV)
                draw_pm(ax, df, T_COLORS[ti])
            style_ax(ax, r, c, len(models), len(FOCAL_PAIRS), row_label=mdl["label"])

    handles = [Line2D([0], [0], color=T_COLORS[i], lw=2, marker="o", ms=5,
                      label=f"$T={T}$") for i, T in enumerate(TEMPERATURES)]
    handles += [Line2D([0], [0], color="gray", lw=1.3, ls="--", label="Theory fit")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.10)
    add_panel_labels(fig, axes)

    out = OUT_DIR / "SI_robustness_temperature.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)

# =============================================================================
# FIGURE 2 — Prompt-variant robustness
# =============================================================================

def make_prompt_figure():
    models = MODELS_ROBUSTNESS
    fig, gs, axes = make_grid(models)
    add_column_titles(axes)

    for r, mdl in enumerate(models):
        for c, (pA, pB) in enumerate(FOCAL_PAIRS):
            ax = axes[r, c]
            for pvi, pv in enumerate(PROMPT_VARIANTS):
                df = load_tp(mdl["short"], "prompt", pA, pB, DEFAULT_T, DEFAULT_N, pv)
                draw_pm(ax, df, PV_COLORS[pvi])
            style_ax(ax, r, c, len(models), len(FOCAL_PAIRS), row_label=mdl["label"])

    handles = [Line2D([0], [0], color=PV_COLORS[i], lw=2, marker="o", ms=5,
                      label=PROMPT_NAMES[i]) for i in range(len(PROMPT_VARIANTS))]
    handles += [Line2D([0], [0], color="gray", lw=1.3, ls="--", label="Theory fit")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.07)
    add_panel_labels(fig, axes)

    out = OUT_DIR / "SI_robustness_prompt.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)

# =============================================================================
# FIGURE 3 — System-size scaling
# =============================================================================

def make_system_size_figure():
    models   = MODELS_ROBUSTNESS
    fig, gs, axes = make_grid(models)
    add_column_titles(axes)

    for r, mdl in enumerate(models):
        for c, (pA, pB) in enumerate(FOCAL_PAIRS):
            ax = axes[r, c]
            for ni, N in enumerate(SYSTEM_SIZES):
                df = load_tp(mdl["short"], "system_size", pA, pB, DEFAULT_T, N, DEFAULT_PV)
                draw_pm(ax, df, N_COLORS[ni])
            style_ax(ax, r, c, len(models), len(FOCAL_PAIRS), row_label=mdl["label"])

    n_handles = [Line2D([0], [0], color=N_COLORS[i], lw=2, marker="o", ms=5,
                        label=f"$N={N}$") for i, N in enumerate(SYSTEM_SIZES)]
    n_handles += [Line2D([0], [0], color="gray", lw=1.3, ls="--", label="Theory fit")]
    fig.legend(handles=n_handles, loc="lower center", ncol=len(n_handles),
               frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.10)
    add_panel_labels(fig, axes)

    out = OUT_DIR / "SI_robustness_system_size.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)

# =============================================================================
# FIGURE 4 — LaTeX table of all opinion pairs
# =============================================================================

def make_opinion_pairs_table():
    """Write SI_opinion_pairs_table.tex with all opinion pairs from the data."""
    pair_map = build_pair_map()
    pairs    = sorted(pair_map.items(), key=lambda x: x[0].lower())

    lines = [
        r"\begin{longtable}{ll}",
        r"\toprule",
        r"\textbf{Opinion A} & \textbf{Opinion B} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Opinion A} & \textbf{Opinion B} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{2}{r}{\textit{Continued on next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    def escape(s):
        return (s.replace("&", r"\&")
                 .replace("%", r"\%")
                 .replace("$", r"\$")
                 .replace("#", r"\#")
                 .replace("_", r"\_")
                 .replace("^", r"\^{}"))

    for a, b in pairs:
        lines.append(f"{escape(a)} & {escape(b)} \\\\")

    lines.append(r"\end{longtable}")

    out = OUT_DIR / "SI_opinion_pairs_table.tex"
    out.write_text("\n".join(lines))
    print(f"Saved: {out}  ({len(pairs)} pairs)")

# =============================================================================
# FIGURE 5 — Phase diagram for all 9 models  (3×3)
# =============================================================================

def make_phase_diagram_all_models():
    df = get_fits()
    beta_sp, h_sp = spinodal_line()
    letters = "abcdefghijklmnopqrstuvwxyz"

    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    for ri, row in enumerate(MODELS_GRID):
        for ci, mdl in enumerate(row):
            ax  = axes[ri, ci]
            sub = df[df["model"] == mdl["short"]].copy()

            # Shaded bistable region
            ax.fill_between(beta_sp, 0, h_sp, color="lightgray", alpha=0.45)
            ax.plot(beta_sp, h_sp, color="gray", lw=1.0, ls="--", alpha=0.7)

            if not sub.empty:
                ax.scatter(sub["beta"], sub["abs_h"],
                           c=mdl["color"], marker=mdl["marker"],
                           s=18, alpha=0.65, linewidths=0, zorder=3)

                # Annotate median
                med_b = sub["beta"].median()
                med_h = sub["abs_h"].median()
                ax.axvline(med_b, color=mdl["color"], lw=0.8, ls=":", alpha=0.5)

            n = len(sub)
            ax.set_title(f"{mdl['label']}  ($n={n}$)", fontsize=10, pad=3)
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 3.2)
            ax.grid(True, alpha=0.18)

            panel_idx = ri * 3 + ci
            ax.text(-0.12, 1.04, f"({letters[panel_idx]})",
                    transform=ax.transAxes, fontsize=11,
                    fontweight="bold", va="bottom", ha="left")

            if ri == 2:
                ax.set_xlabel(r"$\beta$", fontsize=11)
            else:
                ax.tick_params(labelbottom=False)
            if ci == 0:
                ax.set_ylabel(r"$|h|$", fontsize=11)
            else:
                ax.tick_params(labelleft=False)

    # Common legend
    bistable_patch = plt.Rectangle((0, 0), 1, 1, fc="lightgray", ec="gray",
                                    ls="--", lw=1, alpha=0.7, label="Bistable region")
    fig.legend(handles=[bistable_patch], loc="lower center",
               ncol=1, frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.06)

    out = OUT_DIR / "SI_phase_diagram_all_models.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)

# =============================================================================
# FIGURE 6 — Model–model h correlation matrix
# =============================================================================

def make_correlation_matrix():
    df = get_fits()

    # Pivot: rows = opinion_a, columns = model short name, values = signed h
    pivot = df.pivot_table(index="opinion_a", columns="model", values="h", aggfunc="first")

    # Keep only opinions with a fit for every model
    pivot_full = pivot.dropna()
    n_common = len(pivot_full)
    print(f"  Common opinion pairs for correlation: {n_common}")

    # Order models by family for nicer display
    model_order  = [m["short"] for m in ALL_MODELS]
    model_labels = [m["label"] for m in ALL_MODELS]
    cols_avail   = [m for m in model_order if m in pivot_full.columns]
    labels_avail = [m["label"] for m in ALL_MODELS if m["short"] in cols_avail]

    corr = pivot_full[cols_avail].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(9, 7.5))
    mask  = np.triu(np.ones_like(corr, dtype=bool), k=1)   # hide upper triangle
    cmap  = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 9},
        square=True, linewidths=0.4, linecolor="white",
        xticklabels=labels_avail,
        yticklabels=labels_avail,
        cbar_kws={"label": "Pearson $r$", "shrink": 0.8},
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    ax.set_title(f"Pairwise correlation of bias $h$ across opinion pairs\n"
                 f"($n={n_common}$ common pairs)", fontsize=11)
    fig.tight_layout()

    out = OUT_DIR / "SI_model_correlation.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)

# =============================================================================
# FIGURE 7 — Hysteresis examples  (3×3)
# =============================================================================

def load_hysteresis_file(fpath):
    """Return (field, m_fwd, m_bwd) sorted by field, or (None,None,None)."""
    try:
        hdata = pd.read_csv(fpath, comment="#")
        if "field" not in hdata.columns:
            return None, None, None
        idx   = np.argsort(hdata["field"].values)
        return (hdata["field"].values[idx],
                hdata["magnetization_forward"].values[idx],
                hdata["magnetization_backward"].values[idx])
    except Exception:
        return None, None, None


def _loop_completeness(field, m_fwd, m_bwd):
    """
    Score how 'complete' a hysteresis loop is.

    A perfect loop has:
      - m_bwd >> m_fwd at the lowest field  (bwd sweep still in A-phase)
      - m_fwd >> m_bwd at the highest field (fwd sweep now in A-phase)
    Returns a value in [0, 1]: 1 = fully open loop, 0 = no hysteresis.
    """
    diff_low  = m_bwd[0]  - m_fwd[0]    # positive for a good loop
    diff_high = m_fwd[-1] - m_bwd[-1]   # positive for a good loop
    return (max(0.0, diff_low) + max(0.0, diff_high)) / 4.0  # normalised to 1


def _parse_hyst_fname(fpath):
    """Parse 'hysteresis_{A}_vs_{B}_T{t}.txt' → (op_a, op_b) or None."""
    inner = fpath.stem[len("hysteresis_"):]
    tidx  = inner.rfind("_T")
    if tidx == -1:
        return None
    pair_str = inner[:tidx]
    vidx = pair_str.find("_vs_")
    if vidx == -1:
        return None
    return pair_str[:vidx], pair_str[vidx + 4:]


def _scan_hysteresis(models):
    """
    Scan all hysteresis directories for `models` and return a DataFrame with
    columns: model, op_a, op_b, loop_open, area, score, fpath
    """
    records = []
    for mdl in models:
        hdir_rel = HYSTERESIS_DIRS.get(mdl["short"])
        if not hdir_rel:
            continue
        hdir = BASE_DATA / hdir_rel
        if not hdir.exists():
            continue
        for fpath in sorted(hdir.glob("hysteresis_*_vs_*_T*.txt")):
            parsed = _parse_hyst_fname(fpath)
            if parsed is None:
                continue
            op_a, op_b = parsed
            field, m_fwd, m_bwd = load_hysteresis_file(fpath)
            if field is None:
                continue
            loop_open = _loop_completeness(field, m_fwd, m_bwd)
            area      = float(np.sum(np.abs(m_fwd - m_bwd))
                              * np.abs(np.diff(field)).mean())
            records.append(dict(model=mdl["short"], label=mdl["label"],
                                color=mdl["color"],
                                op_a=op_a, op_b=op_b,
                                loop_open=loop_open, area=area,
                                score=loop_open * area, fpath=str(fpath)))
    return pd.DataFrame(records)


def _select_pairs(scan_df, n_pairs=3, completeness_thresh=0.5):
    """
    Pick `n_pairs` opinion pairs that best show complete hysteresis loops.

    Priority:
      1. Pairs where 2+ models show a complete loop (loop_open > thresh)
      2. Pairs where 1 model shows a complete loop
    Within each tier, rank by maximum completeness score across models.
    """
    complete = scan_df[scan_df["loop_open"] >= completeness_thresh]
    pair_stats = complete.groupby(["op_a", "op_b"]).agg(
        n_complete=("model", "count"),
        max_score=("score", "max"),
    ).reset_index().sort_values(["n_complete", "max_score"], ascending=False)

    selected = []
    for _, row in pair_stats.iterrows():
        if len(selected) >= n_pairs:
            break
        selected.append((row["op_a"], row["op_b"]))

    # If not enough, fill from single-model loops
    if len(selected) < n_pairs:
        singles = scan_df[scan_df["loop_open"] >= completeness_thresh].sort_values(
            "score", ascending=False)
        for _, row in singles.iterrows():
            pair = (row["op_a"], row["op_b"])
            if pair not in selected:
                selected.append(pair)
            if len(selected) >= n_pairs:
                break

    print(f"  Selected {len(selected)} pairs:")
    for a, b in selected:
        sub = scan_df[(scan_df.op_a == a) & (scan_df.op_b == b)]
        info = ", ".join(
            f"{r.label} (loop={r.loop_open:.2f})"
            for _, r in sub.iterrows()
        )
        print(f"    {a} vs {b}: {info}")
    return selected


def make_hysteresis_examples():
    """
    3-column × 3-row grid of complete hysteresis cycles.

    Columns = 3 opinion pairs selected for having the most complete loops.
    Rows    = 3 models (gemma-3-27b-it, gemma-3-12b-it, Qwen3-32B).

    Pairs are chosen to maximise completeness across models; pairs visible
    in 2+ models are prioritised over single-model examples.
    """
    models_to_show = [
        {"short": "gemma-3-27b-it", "label": "Gemma 3 27B", "color": "#1f77b4"},
        {"short": "gemma-3-12b-it", "label": "Gemma 3 12B", "color": "#aec7e8"},
        {"short": "Qwen3-32B",      "label": "Qwen3 32B",   "color": "#d62728"},
    ]

    scan_df  = _scan_hysteresis(models_to_show)
    pairs    = _select_pairs(scan_df, n_pairs=3, completeness_thresh=0.4)

    n_rows = len(models_to_show)
    n_cols = max(len(pairs), 1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 10), squeeze=False)
    plt.subplots_adjust(hspace=0.42, wspace=0.28)
    letters = "abcdefghijklmnopqrstuvwxyz"

    for ri, mdl in enumerate(models_to_show):
        color = mdl["color"]
        for ci, (op_a, op_b) in enumerate(pairs):
            ax        = axes[ri, ci]
            panel_idx = ri * n_cols + ci

            ax.text(-0.12, 1.05, f"({letters[panel_idx]})",
                    transform=ax.transAxes, fontsize=11,
                    fontweight="bold", va="bottom", ha="left")

            # Look up the file for this (model, pair)
            row = scan_df[(scan_df.model == mdl["short"]) &
                          (scan_df.op_a  == op_a) &
                          (scan_df.op_b  == op_b)]

            if row.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=9)
                ax.set_xlim(-0.4, 0.4); ax.set_ylim(-1.1, 1.1)
                ax.grid(True, alpha=0.18)
            else:
                fpath         = Path(row.iloc[0]["fpath"])
                loop_open     = row.iloc[0]["loop_open"]
                field, m_fwd, m_bwd = load_hysteresis_file(fpath)

                ax.plot(field, m_fwd, color=color, lw=1.8, zorder=3)
                ax.plot(field, m_bwd, color=color, lw=1.8, ls="--", zorder=3)
                ax.fill_between(field, m_fwd, m_bwd, color=color, alpha=0.15)
                ax.axhline(0, color="gray", lw=0.6, ls=":", alpha=0.5)
                ax.axvline(0, color="gray", lw=0.6, ls=":", alpha=0.5)
                ax.set_ylim(-1.1, 1.1)
                ax.grid(True, alpha=0.18)

                # Mark completeness quality
                quality = "complete" if loop_open >= 0.5 else "partial"
                ax.text(0.97, 0.04, quality,
                        transform=ax.transAxes, fontsize=7, ha="right",
                        color="darkgreen" if quality == "complete" else "gray",
                        style="italic")

            if ri == 0:
                title = f"{op_a}\nvs {op_b}"
                ax.set_title(title, fontsize=8, pad=3)
            if ri == n_rows - 1:
                ax.set_xlabel("External field $h_{ext}$", fontsize=9)
            else:
                ax.tick_params(labelbottom=False)
            if ci == 0:
                ax.set_ylabel(r"Magnetisation $m$" + f"\n{mdl['label']}", fontsize=9)
            else:
                ax.tick_params(labelleft=False)
            if ci == 0:
                ax.set_ylabel(r"Magnetisation $m$" + f"\n{mdl['label']}", fontsize=9)
            else:
                ax.tick_params(labelleft=False)

    # Legend (forward / backward)
    handles = [
        Line2D([0], [0], color="gray", lw=2,        label="Forward sweep"),
        Line2D([0], [0], color="gray", lw=2, ls="--", label="Backward sweep"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.07)

    out = OUT_DIR / "SI_hysteresis_examples.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)

# =============================================================================
# FIGURE 8 — log-ratio distributions within model families
# =============================================================================

def make_family_ratios():
    """
    For each within-family pair (large / small model), compute per-opinion:

        log( β_large / β_small )    — both positive, ratio straightforward
        log( |h_large| / |h_small| ) — use magnitudes to stay positive;
                                       opinions with |h_small| < 0.05 are
                                       excluded as noise (denominator too small)

    Layout: 2 rows × 3 cols
      Row 0 : distribution of log( β_large / β_small )
      Row 1 : distribution of log( |h_large| / |h_small| )
      Cols  : Gemma, Qwen3, Qwen2.5
    """
    from scipy.stats import gaussian_kde

    df = get_fits()

    COLOR_BETA = "#1f77b4"
    COLOR_H    = "#d62728"
    H_MIN      = 0.05    # minimum |h_small| to include in h-ratio

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    plt.subplots_adjust(hspace=0.45, wspace=0.30)
    letters = "abcdefghijklmnopqrstuvwxyz"

    row_titles = [
        r"$\log\!\left(\beta_\mathrm{large}\,/\,\beta_\mathrm{small}\right)$",
        r"$\log\!\left(|h_\mathrm{large}|\,/\,|h_\mathrm{small}|\right)$",
    ]

    for ci, (large_short, small_short, large_lbl, small_lbl) in enumerate(FAMILY_PAIRS):

        sub_l = df[df["model"] == large_short][["opinion_a", "h", "beta"]].rename(
            columns={"h": "h_l", "beta": "beta_l"})
        sub_s = df[df["model"] == small_short][["opinion_a", "h", "beta"]].rename(
            columns={"h": "h_s", "beta": "beta_s"})

        merged = pd.merge(sub_l, sub_s, on="opinion_a").dropna()

        # --- β ratio ---
        valid_beta = merged[(merged["beta_l"] > 0) & (merged["beta_s"] > 0)]
        log_beta   = np.log(valid_beta["beta_l"].values / valid_beta["beta_s"].values)

        # --- |h| ratio ---
        valid_h  = merged[np.abs(merged["h_s"]) >= H_MIN]
        log_h    = np.log(np.abs(valid_h["h_l"].values) / np.abs(valid_h["h_s"].values))

        for ri, (log_vals, color, n_label) in enumerate([
            (log_beta, COLOR_BETA, f"$n={len(log_beta)}$"),
            (log_h,    COLOR_H,    f"$n={len(log_h)}$"),
        ]):
            ax        = axes[ri, ci]
            panel_idx = ri * 3 + ci

            ax.text(-0.12, 1.05, f"({letters[panel_idx]})",
                    transform=ax.transAxes, fontsize=11,
                    fontweight="bold", va="bottom", ha="left")

            if len(log_vals) < 3:
                ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                continue

            # Histogram
            ax.hist(log_vals, bins=20, color=color, alpha=0.45,
                    density=True, zorder=2)

            # KDE overlay
            kde_x = np.linspace(log_vals.min() - 0.5, log_vals.max() + 0.5, 400)
            try:
                kde = gaussian_kde(log_vals, bw_method="scott")
                ax.plot(kde_x, kde(kde_x), color=color, lw=2.0, zorder=3)
            except Exception:
                pass

            # Reference line at 0 (ratio = 1, i.e. no change)
            ax.axvline(0, color="black", lw=1.2, ls="--", alpha=0.7,
                       label="ratio $= 1$")

            # Median marker
            med = np.median(log_vals)
            ax.axvline(med, color=color, lw=1.2, ls=":", alpha=0.9)

            # Annotation: median and fraction where large > small
            frac_above = (log_vals > 0).mean()
            ax.text(0.97, 0.95,
                    f"median = {med:+.2f}\n{frac_above*100:.0f}% large > small\n{n_label}",
                    transform=ax.transAxes, fontsize=8, va="top", ha="right",
                    bbox=dict(fc="white", ec="none", alpha=0.75))

            ax.set_xlabel(row_titles[ri], fontsize=10)
            ax.set_ylabel("Density", fontsize=9)
            ax.grid(True, alpha=0.18)

            if ri == 0:
                ax.set_title(f"{large_lbl} / {small_lbl}", fontsize=10, pad=4)

    # Legend
    handles = [
        Line2D([0], [0], color="black", lw=1.5, ls="--", label="log-ratio $= 0$ (equal)"),
        Line2D([0], [0], color="gray",  lw=1.2, ls=":",  label="median"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.10)

    out = OUT_DIR / "SI_family_ratios.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)

# =============================================================================
# FIGURE 8b — normalized difference within model families
# =============================================================================

def make_family_ratios_normdiff():
    """
    Same layout as make_family_ratios() but uses the normalized difference

        (X_large - X_small) / X_large

    instead of log-ratios, for both β and |h|.

    Normalization is by the smaller model's value, equivalent to
    (X_large / X_small - 1), so positive values mean the larger model
    exceeds the smaller and the result is bounded below by -1.

    Filtering:
      β  — both values must be positive (always true for well-fitted curves)
      |h| — |h_small| >= H_MIN  (we divide by the small value here)
    """
    from scipy.stats import gaussian_kde

    df = get_fits()

    COLOR_BETA = "#1f77b4"
    COLOR_H    = "#d62728"
    H_MIN      = 0.05    # minimum |h_large| to include in h comparison

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    plt.subplots_adjust(hspace=0.45, wspace=0.30)
    letters = "abcdefghijklmnopqrstuvwxyz"

    row_titles = [
        r"$(\beta_\mathrm{large} - \beta_\mathrm{small})\;/\;\beta_\mathrm{small}$",
        r"$(|h_\mathrm{large}| - |h_\mathrm{small}|)\;/\;|h_\mathrm{small}|$",
    ]

    for ci, (large_short, small_short, large_lbl, small_lbl) in enumerate(FAMILY_PAIRS):

        sub_l = df[df["model"] == large_short][["opinion_a", "h", "beta"]].rename(
            columns={"h": "h_l", "beta": "beta_l"})
        sub_s = df[df["model"] == small_short][["opinion_a", "h", "beta"]].rename(
            columns={"h": "h_s", "beta": "beta_s"})

        merged = pd.merge(sub_l, sub_s, on="opinion_a").dropna()

        # --- β normalized difference ---
        valid_beta = merged[(merged["beta_l"] > 0) & (merged["beta_s"] > 0)]
        diff_beta  = ((valid_beta["beta_l"].values - valid_beta["beta_s"].values)
                      / valid_beta["beta_s"].values)

        # --- |h| normalized difference (normalize by |h_small|) ---
        valid_h  = merged[np.abs(merged["h_s"]) >= H_MIN]
        diff_h   = ((np.abs(valid_h["h_l"].values) - np.abs(valid_h["h_s"].values))
                    / np.abs(valid_h["h_s"].values))

        for ri, (vals, color, n_label) in enumerate([
            (diff_beta, COLOR_BETA, f"$n={len(diff_beta)}$"),
            (diff_h,    COLOR_H,    f"$n={len(diff_h)}$"),
        ]):
            ax        = axes[ri, ci]
            panel_idx = ri * 3 + ci

            ax.text(-0.12, 1.05, f"({letters[panel_idx]})",
                    transform=ax.transAxes, fontsize=11,
                    fontweight="bold", va="bottom", ha="left")

            if len(vals) < 3:
                ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                continue

            # Histogram
            ax.hist(vals, bins=20, color=color, alpha=0.45, density=True, zorder=2)

            # KDE overlay
            x_min, x_max = vals.min() - 0.05, vals.max() + 0.05
            kde_x = np.linspace(x_min, x_max, 400)
            try:
                kde = gaussian_kde(vals, bw_method="scott")
                ax.plot(kde_x, kde(kde_x), color=color, lw=2.0, zorder=3)
            except Exception:
                pass

            # Reference line at 0 (large == small)
            ax.axvline(0, color="black", lw=1.2, ls="--", alpha=0.7)

            # Median marker
            med = np.median(vals)
            ax.axvline(med, color=color, lw=1.2, ls=":", alpha=0.9)

            # Annotation
            frac_above = (vals > 0).mean()
            ax.text(0.97, 0.95,
                    f"median = {med:+.2f}\n{frac_above*100:.0f}% large > small\n{n_label}",
                    transform=ax.transAxes, fontsize=8, va="top", ha="right",
                    bbox=dict(fc="white", ec="none", alpha=0.75))

            ax.set_xlabel(row_titles[ri], fontsize=10)
            ax.set_ylabel("Density", fontsize=9)
            ax.grid(True, alpha=0.18)

            if ri == 0:
                ax.set_title(f"{large_lbl} / {small_lbl}", fontsize=10, pad=4)

    # Legend
    handles = [
        Line2D([0], [0], color="black", lw=1.5, ls="--", label="no difference"),
        Line2D([0], [0], color="gray",  lw=1.2, ls=":",  label="median"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.10)

    out = OUT_DIR / "SI_family_ratios_normdiff.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Figure 1: temperature robustness …")
    make_temperature_figure()

    print("\nFigure 2: prompt robustness …")
    make_prompt_figure()

    print("\nFigure 3: system-size scaling …")
    make_system_size_figure()

    print("\nFigure 4: opinion pairs LaTeX table …")
    make_opinion_pairs_table()

    print("\nFigure 5: phase diagram — all 9 models (3×3) …")
    make_phase_diagram_all_models()

    print("\nFigure 6: model–model h correlation matrix …")
    make_correlation_matrix()

    print("\nFigure 7: hysteresis examples …")
    make_hysteresis_examples()

    print("\nFigure 8: within-family β and h ratios …")
    make_family_ratios()

    print(f"\nAll outputs saved to: {OUT_DIR}/")

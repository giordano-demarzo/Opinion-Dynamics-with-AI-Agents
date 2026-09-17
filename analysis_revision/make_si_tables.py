#!/usr/bin/env python3
"""Generate LaTeX table fragments for the revised SI from the resubmission data.
Writes .tex files into resubmission/manuscript/SI_tables/ for \\input{}."""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("/home/jovyan/LLMs_opinion_dynamics_bias")
DATA = BASE / "resubmission" / "data"
OUT = BASE / "resubmission" / "manuscript" / "SI_tables"
OUT.mkdir(exist_ok=True)

OFFICIAL = set(pd.read_csv(BASE / "submission" / "opinion_pairs.csv").Opinion_A)

NAMES = {
    "Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "Qwen2.5-14B-Instruct": "Qwen2.5 14B",
    "Qwen2.5-32B-Instruct": "Qwen2.5 32B",
    "Qwen3-14B": "Qwen3 14B",
    "Qwen3-32B": "Qwen3 32B",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "gemma-3-12b-it": "Gemma 3 12B",
    "gemma-3-27b-it": "Gemma 3 27B",
    "gpt-5-mini": "GPT-5 mini",
}
MODEL_ORDER = ["gemma-3-27b-it", "gemma-3-12b-it", "Llama-3.1-8B-Instruct",
               "Qwen3-32B", "Qwen3-14B", "Qwen2.5-32B-Instruct",
               "Qwen2.5-14B-Instruct", "gpt-5-mini", "gemini-2.5-flash-lite"]


def esc(s):
    return (str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            .replace("#", "\\#"))


def write(name, text):
    # every fragment must carry its own \bottomrule: placing it after \input{}
    # in the main file fails with "Misplaced \noalign" at the file boundary
    if not text.rstrip().endswith("\\bottomrule"):
        text = text.rstrip("\n") + "\n\\bottomrule\n"
    (OUT / name).write_text(text)
    print("wrote", name)


# ---------------------------------------------------------------- held-out set
pairs = pd.read_csv(DATA / "grounded_opinion_pairs_globalopinionqa.csv")
fits = pd.read_csv(DATA / "new_pairs_fits.csv")
gg = fits[fits.subset == "grounded_general"]
frac = gg.groupby("Opinion_A").agg(meta=("classification", lambda s: (s == "metastable").mean()),
                                   n=("classification", "size"))
rows = []
for _, r in pairs.iterrows():
    m = frac.loc[r.Opinion_A] if r.Opinion_A in frac.index else None
    mf = f"{100*m.meta:.0f}\\% ({int(m.n)})" if m is not None else "--"
    rows.append(f"{esc(r.full_description_A)} & {esc(r.full_description_B)} & {mf} \\\\")
write("heldout_pairs.tex", "\n".join(rows))

# ------------------------------------------------------------------ safety set
sp = pd.read_csv(DATA / "ai_safety_opinion_pairs.csv")
med = pd.read_csv(DATA / "ai_safety_per_pair_median.csv").set_index("Opinion_A")

def brk(s):
    # allow line breaks after slashes and hyphens in narrow columns
    return esc(s).replace("/", "/\\allowbreak ").replace("-", "-\\allowbreak ")

rows = []
for _, r in sp.iterrows():
    rows.append(f"{esc(r.full_description_A)} & {esc(r.full_description_B)} & "
                f"{brk(r.category)} & {esc(r.grounding_note)} \\\\")
write("safety_pairs.tex", "\n".join(rows))

rows = []
for _, m in med.sort_values("meta", ascending=False).reset_index().iterrows():
    rows.append(f"{esc(m.Opinion_A)} & {m.med_h:.2f} & {100*m.meta:.0f}\\% & {int(m.n)} \\\\")
write("safety_fits.tex", "\n".join(rows))

# ---------------------------------------------------------------- discard rates
d = pd.read_csv(DATA / "discard_rates_all_models.csv")
rows = []
for mod in MODEL_ORDER:
    g = d[d.Model == mod].discard_rate
    rows.append(f"{NAMES[mod]} & {100*g.mean():.2f}\\% & {100*g.median():.1f}\\% & "
                f"{100*g.quantile(0.9):.1f}\\% & {100*g.max():.0f}\\% \\\\")
write("discard_models.tex", "\n".join(rows))

top = (d.groupby(["Model", "Opinion_A", "Opinion_B"]).discard_rate.mean()
       .sort_values(ascending=False).head(15).reset_index())
rows = [f"{NAMES[r.Model]} & {esc(r.Opinion_A)} vs.\\ {esc(r.Opinion_B)} & "
        f"{100*r.discard_rate:.0f}\\% \\\\" for _, r in top.iterrows()]
write("discard_top.tex", "\n".join(rows))

# ---------------------------------------------------------------- bracket test
bt = pd.read_csv(DATA / "bracket_test_all.csv")
rows = []
for (mod, oa), g in bt.groupby(["Model", "Opinion_A"], sort=False):
    w = g[g.condition == "with_brackets"].iloc[0]
    n = g[g.condition == "no_brackets"].iloc[0]
    rows.append(f"{NAMES[mod]} & {esc(oa)} & "
                f"{int(w.n_valid)}/{int(w.n_deflect)}/{int(w.n_other_invalid)} & "
                f"{int(n.n_valid)}/{int(n.n_deflect)}/{int(n.n_other_invalid)} \\\\")
write("bracket.tex", "\n".join(rows))

# ------------------------------------------------------------------- isolation
iso = pd.read_csv(DATA / "isolation_vs_balanced.csv").dropna(subset=["p_A", "p0_fit"])
rows = []
for mod in MODEL_ORDER:
    g = iso[iso.Model == mod]
    flip = ((g.p_A - 0.5) * (g.p0_fit - 0.5) < 0).mean()
    r = np.corrcoef(g.p_A, g.p0_fit)[0, 1]
    rows.append(f"{NAMES[mod]} & {len(g)} & {r:.2f} & {g.abs_diff.mean():.2f} & "
                f"{100*flip:.0f}\\% & {100*g.discard_rate.mean():.1f}\\% \\\\")
g = iso
flip = ((g.p_A - 0.5) * (g.p0_fit - 0.5) < 0).mean()
r = np.corrcoef(g.p_A, g.p0_fit)[0, 1]
rows.append("\\midrule")
rows.append(f"All models & {len(g)} & {r:.2f} & {g.abs_diff.mean():.2f} & "
            f"{100*flip:.1f}\\% & {100*g.discard_rate.mean():.1f}\\% \\\\")
write("isolation_models.tex", "\n".join(rows))

inv = iso[(iso.p_A - 0.5) * (iso.p0_fit - 0.5) < 0].sort_values("abs_diff", ascending=False).head(12)
rows = [f"{NAMES[r.Model]} & {esc(r.Opinion_A)} vs.\\ {esc(r.Opinion_B)} & "
        f"{r.p_A:.2f} & {r.p0_fit:.2f} \\\\" for _, r in inv.iterrows()]
write("isolation_inversions.tex", "\n".join(rows))

# ---------------------------------------------------- bootstrap classification
f = pd.read_csv(DATA / "fits_with_ci.csv")
f = f[f.Opinion_A.isin(OFFICIAL)]
b = pd.read_csv(DATA / "bootstrap_ci.csv")
m = f.merge(b, on=["Model", "Opinion_A"], how="left")
infl = (m.h_boot_sd / m.se_h.replace(0, np.nan)).clip(lower=1.0).fillna(1.0)
m["z"] = m.margin_h / (m.se_margin * infl)
rows = []
for mod in MODEL_ORDER:
    g = m[m.Model == mod]
    meta = g.classification == "metastable"
    rm = meta & (g.z > 2)
    ro = (~meta) & (g.z < -2)
    rows.append(f"{NAMES[mod]} & {len(g)} & {100*meta.mean():.0f}\\% & "
                f"{100*rm.mean():.0f}\\% & {100*ro.mean():.0f}\\% & "
                f"{100*(1-rm.mean()-ro.mean()):.0f}\\% \\\\")
meta = m.classification == "metastable"
rm = meta & (m.z > 2)
ro = (~meta) & (m.z < -2)
rows.append("\\midrule")
rows.append(f"All models & {len(m)} & {100*meta.mean():.1f}\\% & {100*rm.mean():.1f}\\% & "
            f"{100*ro.mean():.1f}\\% & {100*(1-rm.mean()-ro.mean()):.1f}\\% \\\\")
write("bootstrap_models.tex", "\n".join(rows))

# ------------------------------------------------------------ family comparison
fam = pd.read_csv(DATA / "response_family_comparison.csv")
def pair_from_file(fn):
    parts = fn.split("_", 4)
    return parts[4].replace(".txt", "").replace("_", " ").strip() if len(parts) >= 5 else fn
fam["pair"] = fam.file.apply(pair_from_file)
low = {p.lower() for p in OFFICIAL}
fam = fam[fam.pair.str.lower().isin(low)]
rows = []
for mod in MODEL_ORDER:
    g = fam[fam.Model == mod]
    wc = g.winner.value_counts(normalize=True)
    rows.append(f"{NAMES[mod]} & {len(g)} & {100*wc.get('tanh',0):.0f}\\% & "
                f"{100*wc.get('jump',0):.0f}\\% & {100*wc.get('linear',0):.0f}\\% & "
                f"{100*wc.get('step',0):.0f}\\% & {g.dbic_voter_vs_tanh.median():.0f} \\\\")
wc = fam.winner.value_counts(normalize=True)
rows.append("\\midrule")
rows.append(f"All models & {len(fam)} & {100*wc.get('tanh',0):.1f}\\% & "
            f"{100*wc.get('jump',0):.1f}\\% & {100*wc.get('linear',0):.1f}\\% & "
            f"{100*wc.get('step',0):.1f}\\% & {fam.dbic_voter_vs_tanh.median():.0f} \\\\")
write("family_models.tex", "\n".join(rows))

# -------------------------------------------------------- gelastopoulos per model
g = pd.read_csv(DATA / "gelastopoulos_comparison.csv")
g = g[g.Opinion_A.isin(OFFICIAL)]
rows = []
for mod in MODEL_ORDER:
    s = g[g.Model == mod]
    rows.append(f"{NAMES[mod]} & {len(s)} & {100*(s.z_M.abs()>2).mean():.0f}\\% & "
                f"{s.M.abs().median():.4f} & {100*(s.delta_bic>2).mean():.0f}\\% \\\\")
rows.append("\\midrule")
rows.append(f"All models & {len(g)} & {100*(g.z_M.abs()>2).mean():.1f}\\% & "
            f"{g.M.abs().median():.4f} & {100*(g.delta_bic>2).mean():.1f}\\% \\\\")
write("gelastopoulos_models.tex", "\n".join(rows))

# ------------------------------------------------------------ pew battery
pw = pd.read_csv(DATA / "pew_typology_2017_pairs.csv")
psum = pd.read_csv(DATA / "pew_per_pair_summary.csv").set_index("Opinion_A")
rows = []
for _, r in pw.iterrows():
    m = psum.loc[r.Opinion_A]
    rows.append(f"{esc(r.pew_question_id)} & {esc(r.Opinion_A)} & {esc(r.Opinion_B)} & "
                f"{m.medh:.2f} & {100*m.meta:.0f}\\% \\\\")
write("pew_pairs.tex", "\n".join(rows))

print("done")

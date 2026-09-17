#!/usr/bin/env python3
"""Estimator dependence of the metastable classification: paper WLS (variance floor 1e-4)
vs WLS with Agresti-Coull variance vs exact binomial MLE. Cross with bootstrap-inflated robustness."""
import warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit, OptimizeWarning
warnings.filterwarnings("ignore")
BASE = Path("/home/jovyan/LLMs_opinion_dynamics_bias"); DATA = BASE/"resubmission"/"data"
MODELS = ["Llama-3.1-8B-Instruct","gemma-3-12b-it","gemma-3-27b-it","Qwen2.5-14B-Instruct","Qwen2.5-32B-Instruct","Qwen3-14B","Qwen3-32B","gemini-2.5-flash-lite","gpt-5-mini"]
lines=[]
def say(s=""): print(s); lines.append(s)
def parse_txt(fp):
    try:
        d = pd.read_csv(fp)
        if "m0" in d.columns: d = d.rename(columns={"m0":"m","count_A":"cA","count_B":"cB"})
        else:
            d = pd.read_csv(fp, header=None, usecols=[0,1,2,3,4]); d.columns=["m","cA","cB","probability","standard_error"]
    except Exception: return None
    for c in ["m","cA","cB"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=["m","cA","cB"]); d=d[(d.cA+d.cB)>0].copy(); return d.reset_index(drop=True)
def find_file(out_dir,a):
    for c in (out_dir/f"transition_prob_50_0.2_{a}.txt", out_dir/f"transition_prob_50_opinions_{a.replace(' ','_')}.txt"):
        if c.exists(): return c
def f(m,beta,h,dP): return 0.5*np.tanh(beta*(m+h))+0.5+dP
def wls(m,y,sig):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error",OptimizeWarning)
            p,_=curve_fit(f,m,y,p0=[1,0,0],sigma=sig,absolute_sigma=True,maxfev=3000,bounds=([-np.inf,-np.inf,-1e-4],[np.inf,np.inf,1e-4]))
        return p
    except Exception: return None
def hsp(b):
    if b<=1: return np.nan
    ms=np.sqrt(1-1/b); return ms-np.arctanh(ms)/b
def meta(b,h): return bool(b>1 and abs(h)<hsp(b))
off=list(pd.read_csv(BASE/"submission"/"opinion_pairs.csv").Opinion_A)
rows=[]
for model in MODELS:
    od=BASE/model/"results_batched_vllm"/"N=50"
    for a in off:
        fp=find_file(od,a)
        if fp is None: continue
        d=parse_txt(fp)
        if d is None or len(d)<6: continue
        m=d.m.values.astype(float); n=(d.cA+d.cB).values.astype(float); p=(d.cA/n).values
        sig_clip=np.sqrt(np.clip(p*(1-p),1e-4,None)/n)                     # paper
        pt=(d.cA.values+1)/(n+2); sig_ac=np.sqrt(pt*(1-pt)/n)               # Agresti-Coull floor
        sig_unw=np.ones_like(p)                                              # unweighted
        r=dict(Model=model,Opinion_A=a)
        for lab,s in [("clip",sig_clip),("ac",sig_ac),("unw",sig_unw)]:
            q=wls(m,p,s)
            r[f"beta_{lab}"]=q[0] if q is not None else np.nan; r[f"h_{lab}"]=q[1] if q is not None else np.nan
            r[f"meta_{lab}"]=meta(q[0],q[1]) if q is not None else np.nan
        rows.append(r)
df=pd.DataFrame(rows)
mle=pd.read_csv(DATA/"hardening_lrt_ramp.csv")[["Model","Opinion_A","beta","h","tanh_meta"]].rename(columns={"beta":"beta_mle","h":"h_mle","tanh_meta":"meta_mle"})
df=df.merge(mle,on=["Model","Opinion_A"])
fc=pd.read_csv(DATA/"fits_with_ci.csv"); fc=fc[fc.Opinion_A.isin(off)]
b=pd.read_csv(DATA/"bootstrap_ci.csv")
fc=fc.merge(b,on=["Model","Opinion_A"],how="left")
infl=(fc.h_boot_sd/fc.se_h.replace(0,np.nan)).clip(lower=1).fillna(1)
fc["z"]=fc.margin_h/(fc.se_margin*infl)
fc["robust"]=fc.z.abs()>2
df=df.merge(fc[["Model","Opinion_A","classification","z","robust","beta","h"]].rename(columns={"beta":"beta_paper","h":"h_paper"}),on=["Model","Opinion_A"])
df["meta_paper"]=df.classification=="metastable"
df.to_csv(DATA/"hardening_estimator_dependence.csv",index=False)
say(f"n curves = {len(df)}")
say("Metastable fraction by estimator:")
for lab,name in [("paper","WLS, variance floor 1e-4 (paper, fits_with_ci.csv)"),("clip","WLS, variance floor 1e-4 (refit here)"),("ac","WLS, Agresti-Coull variance"),("unw","unweighted least squares"),("mle","exact binomial MLE")]:
    say(f"  {name:48s} {100*df[f'meta_{lab}'].mean():5.1f}%   agreement with paper {100*(df[f'meta_{lab}']==df.meta_paper).mean():5.1f}%")
say(f"Paper robust (bootstrap-inflated |z|>2): {100*df.robust.mean():.1f}% of curves; robust metastable {100*(df.robust&df.meta_paper).mean():.1f}%, robust monostable {100*(df.robust&~df.meta_paper).mean():.1f}%")
for lab in ["ac","unw","mle"]:
    dis=df[df[f"meta_{lab}"]!=df.meta_paper]
    say(f"  {lab}: disagreements n={len(dis)}; of these unresolved in paper's bootstrap: {100*(~dis.robust).mean():.1f}%;  agreement among robust curves: {100*(df[df.robust][f'meta_{lab}']==df[df.robust].meta_paper).mean():.1f}%")
say(f"MLE vs paper: median beta ratio {np.median(df.beta_mle/df.beta_paper):.2f}; median |h| ratio {np.median(df.h_mle.abs()/df.h_paper.abs().clip(lower=1e-3)):.2f}; Spearman h {df[['h_mle','h_paper']].corr(method='spearman').iloc[0,1]:.3f}")
dis=df[(df.meta_mle!=df.meta_paper)]
say(f"MLE/paper disagreements: paper-meta/MLE-mono {int((dis.meta_paper&~dis.meta_mle).sum())}, paper-mono/MLE-meta {int((~dis.meta_paper&dis.meta_mle).sum())}; median paper |z| among them {dis.z.abs().median():.2f} vs {df[df.meta_mle==df.meta_paper].z.abs().median():.2f} among agreements")
say(f"Metastable among curves robust in paper AND classified by all 4 estimators identically: {100*(df[df.robust&(df.meta_mle==df.meta_paper)&(df.meta_ac==df.meta_paper)&(df.meta_unw==df.meta_paper)].meta_paper).mean():.1f}%  (n={int((df.robust&(df.meta_mle==df.meta_paper)&(df.meta_ac==df.meta_paper)&(df.meta_unw==df.meta_paper)).sum())})")
say("Per-model MLE metastable % (paper % in parentheses):")
for mod,g in df.groupby("Model"): say(f"  {mod:26s} {100*g.meta_mle.mean():5.1f}%  ({100*g.meta_paper.mean():.0f}%)")
(DATA/"hardening_checks_4_summary.txt").write_text("\n".join(lines)+"\n")

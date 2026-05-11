"""
Three-panel figure 4:
  Left   — tipping-point dynamics (both opinion pairs)
  Center — hysteresis loop for gender self-identification (3 cycles)
  Right  — theory vs observed z_c (all models)
"""

import json, glob, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, OptimizeWarning

rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
})

BASE = Path(__file__).parent.parent / 'data'
N_REGULAR = 50

# ── colours ────────────────────────────────────────────────────────────────────
C_GENDER   = '#e15759'   # red   – gender self-id
C_ENERGY   = '#4e79a7'   # blue  – renewable energy
C_FWD      = '#1f77b4'
C_BWD      = '#d62728'
C_THEORY   = '#2ca02c'
C_GRAY     = '#7f7f7f'
CYCLE_COLORS = ['#e15759', '#4e79a7', '#59a14f']


# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — tipping point
# ══════════════════════════════════════════════════════════════════════════════

TIPPING_FILES = [
    {
        'path': BASE / 'gemma-3-27b-it/results_tipping_point/tipping_gender self-identification_Ns35_T0.20.json',
        'color': C_GENDER,
        'label': 'Gender self-id.',
    },
    {
        'path': BASE / 'gemma-3-27b-it/results_tipping_point/tipping_renewable energy_Ns225_T0.20.json',
        'color': C_ENERGY,
        'label': 'Renewable energy',
    },
]


def draw_tipping(ax):
    t1 = t2 = total_blocks = None
    for cfg in TIPPING_FILES:
        d = json.load(open(cfg['path']))
        t1 = d['t1'];  t2 = d['t2']
        total_blocks = len(d['trajectories'][0])
        blocks = np.arange(1, total_blocks + 1)
        Ns = d['N_stubborn_B']

        for r, traj in enumerate(d['trajectories']):
            ax.plot(blocks, traj, color=cfg['color'], lw=1.0, alpha=0.35)
        ax.plot(blocks, d['mean'], color=cfg['color'], lw=2.2,
                label=cfg['label'])

    ax.axvline(t1 + 0.5, color='k', lw=1.3, ls='--', alpha=0.7)
    ax.axvline(t2 + 0.5, color='k', lw=1.3, ls='--', alpha=0.7)

    ytop = 1.05
    ax.text((t1 + 0.5) / 2,                 ytop, 'Free',     ha='center', fontsize=10, va='bottom')
    ax.text((t1 + t2) / 2 + 0.5,            ytop, 'Stubborn', ha='center', fontsize=10, va='bottom')
    ax.text((t2 + 0.5 + total_blocks) / 2,  ytop, 'Free',     ha='center', fontsize=10, va='bottom')

    ax.axhline(0, color='k', lw=0.7, alpha=0.3, ls=':')
    ax.set_xlabel('Time $t$', fontsize=13)
    ax.set_ylabel(r'Collective opinion $m$', fontsize=13)
    ax.set_xlim(0.5, total_blocks + 0.5)
    ax.set_ylim(-1.3, 1.15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower left')


# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — hysteresis loop (gender self-id, 3 individual cycles)
# ══════════════════════════════════════════════════════════════════════════════

HYST_FILE = (BASE / 'gemma-3-27b-it/results_hysteresis_vllm/N=50'
             / 'hysteresis_gender self-identification_vs_biological sex classification_T0.20.txt')


def draw_hysteresis(ax):
    df = pd.read_csv(HYST_FILE, comment='#')
    z = df['stubborn_count'].values / N_REGULAR

    n_cycles = sum(1 for c in df.columns if c.startswith('magnetization_forward_c'))

    for c in range(n_cycles):
        col = CYCLE_COLORS[c % len(CYCLE_COLORS)]
        ax.plot(z, df[f'magnetization_forward_c{c+1}'].values,
                color=col, lw=1.2, alpha=0.55, label=f'Cycle {c+1}')
        ax.plot(z, df[f'magnetization_backward_c{c+1}'].values,
                color=col, lw=1.2, alpha=0.55, ls='--')

    ax.plot(z, df['magnetization_forward'].values,
            color='k', lw=2.2, alpha=0.7, label='Mean')
    ax.plot(z, df['magnetization_backward'].values,
            color='k', lw=2.2, alpha=0.7, ls='--')

    ax.axhline(0, color=C_GRAY, lw=0.8, alpha=0.4)
    ax.axvline(0, color=C_GRAY, lw=0.8, alpha=0.4)
    ax.set_xlabel(r'Stubborn ratio $z$', fontsize=13)
    ax.set_ylabel(r'Collective opinion $m$', fontsize=13)
    ax.set_ylim(-1.12, 1.12)
    ax.set_xlim(z.min() * 1.03, z.max() * 1.03)
    ax.grid(True, alpha=0.3)

    handles = [Line2D([0], [0], color=CYCLE_COLORS[c], lw=1.5, label=f'Cycle {c+1}')
               for c in range(n_cycles)]
    handles += [
        Line2D([0], [0], color='k', lw=2.2, alpha=0.7, label='Mean'),
        Line2D([0], [0], color='k', lw=1.5, ls='-',  label='Forward'),
        Line2D([0], [0], color='k', lw=1.5, ls='--', label='Backward'),
    ]
    ax.legend(handles=handles, fontsize=8, loc='lower right')



# ══════════════════════════════════════════════════════════════════════════════
# PANEL C — theory vs observed z_c  (re-used from plot4_standalone)
# ══════════════════════════════════════════════════════════════════════════════

def fit_func(m, beta, h, delta_P):
    return 0.5 * np.tanh(beta * (m + h)) + 0.5 + delta_P

def spinodal_eq(m, beta, h, s):
    arg = beta * (m + h)
    return m - np.tanh(arg) - (beta / np.cosh(arg)**2 - 1.0) * (s - m)

def z_from_m(m, beta, h):
    arg = beta * (m + h)
    return beta / np.cosh(arg)**2 - 1.0

def find_spinodal(beta, h, s, Ngrid=2000):
    a, b = (-0.999, -1e-4) if s == +1 else (1e-4, 0.999)
    ms = np.linspace(a, b, Ngrid)
    Fv = spinodal_eq(ms, beta, h, s)
    sc = np.where(np.sign(Fv[:-1]) * np.sign(Fv[1:]) < 0)[0]
    if len(sc) == 0:
        return None, None
    l, r = ms[sc[0]], ms[sc[0]+1]
    for _ in range(80):
        mid = 0.5*(l+r)
        if spinodal_eq(l,beta,h,s)*spinodal_eq(mid,beta,h,s) <= 0:
            r = mid
        else:
            l = mid
    mc = 0.5*(l+r)
    return mc, z_from_m(mc, beta, h)

def h_spinodal(beta):
    if beta <= 1: return None
    ms = np.sqrt(1 - 1/beta)
    return abs(-ms + np.arctanh(ms)/beta)

def in_spinodal(beta, h):
    if beta <= 1: return False
    hs = h_spinodal(beta)
    return hs is not None and abs(h) <= hs

def load_fits(model_name):
    pairs = pd.read_csv(BASE / 'opinion_pairs.csv')
    batched = BASE / model_name / 'results_batched_vllm'
    pattern = str(batched / f'N={N_REGULAR}/transition_prob_*.txt')
    fits = {}
    valid_A = set(pairs['Opinion_A'].str.lower().str.strip())
    valid_B = set(pairs['Opinion_B'].str.lower().str.strip())
    valid   = valid_A | valid_B
    for fp in glob.glob(pattern):
        bn = os.path.basename(fp).replace('.txt','')
        parts = bn.split('_')
        if len(parts) <= 4: continue
        op = ' '.join(parts[4:])
        opl = op.lower().strip()
        if opl not in valid: continue
        row = pairs[pairs['Opinion_A'].str.lower().str.strip() == opl]
        if row.empty:
            row = pairs[pairs['Opinion_B'].str.lower().str.strip() == opl]
            if row.empty: continue
            opA = row['Opinion_B'].values[0]; opB = row['Opinion_A'].values[0]
        else:
            opA = row['Opinion_A'].values[0]; opB = row['Opinion_B'].values[0]
        try:
            data = pd.read_csv(fp)
            x = data['m0'].astype(float).values if 'm0' in data.columns else data.iloc[:,0].astype(float).values
            y = data['probability'].astype(float).values if 'probability' in data.columns else data.iloc[:,3].astype(float).values
            msk = np.isfinite(x) & np.isfinite(y)
            x, y = x[msk], y[msk]
            if len(x) < 4: continue
            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)
                popt, _ = curve_fit(fit_func, x, y, maxfev=5000,
                                    p0=[2.,0.,0.], bounds=([0.1,-1.,-0.5],[20.,1.,0.5]))
            fits[(opA.lower(), opB.lower())] = {'beta': popt[0], 'h': popt[1]}
        except: pass
    return fits

def load_hyst_transitions(model_name):
    for rdir in ['results_hysteresis_vllm','results_hysteresis_gemini','results_hysteresis_openai']:
        hp = BASE / model_name / rdir
        if hp.exists(): break
    else: return {}
    results = {}
    for fp in glob.glob(str(hp / f'N={N_REGULAR}/hysteresis_*.txt')):
        bn = os.path.basename(fp)
        name = bn.replace('hysteresis_','').replace('_T0.20.txt','').replace('.txt','')
        parts = name.split('_vs_')
        if len(parts) != 2: continue
        opA, opB = parts
        try:
            df = pd.read_csv(fp, comment='#')
            sc = df['stubborn_count'].values
            mf = df['magnetization_forward'].values
            mb = df['magnetization_backward'].values
            fz = bz = None
            for i in range(len(mf)-1):
                if mf[i] < -0.7 <= mf[i+1]:
                    fs = sc[i] + ((-0.7-mf[i])/(mf[i+1]-mf[i]))*(sc[i+1]-sc[i])
                    if fs >= -5: fz = max(0, fs)/N_REGULAR
                    break
            for i in range(len(mb)-1):
                if mb[i] < 0.7 <= mb[i+1]:
                    bs = sc[i] + ((0.7-mb[i])/(mb[i+1]-mb[i]))*(sc[i+1]-sc[i])
                    if bs <= 5: bz = abs(min(0,bs))/N_REGULAR
                    break
            results[(opA.lower(), opB.lower())] = {'fz': fz, 'bz': bz}
        except: pass
    return results

def build_comparison(model_name):
    fits  = load_fits(model_name)
    hysts = load_hyst_transitions(model_name)
    rows  = []
    for key, hd in hysts.items():
        opA, opB = key
        fd = fits.get(key) or fits.get((opB, opA))
        if fd is None: continue
        beta = fd['beta']
        h    = fd['h'] if key in fits else -fd['h']
        if not in_spinodal(beta, h): continue
        _, zcp = find_spinodal(beta, h, s=+1)
        _, zcm = find_spinodal(beta, h, s=-1)
        rows.append({'zcp': zcp, 'zcm': zcm, 'fz': hd['fz'], 'bz': hd['bz']})
    return rows

def detect_models():
    dirs = [
        'gemini-2.5-flash-lite','gemma-3-12b-it','gemma-3-27b-it',
        'gpt-5-mini','gpt-oss-20b',
        'Llama-3.1-70B-Instruct','Llama-3.1-8B-Instruct','Llama-3.3-70B-Instruct',
        'Qwen2.5-14B-Instruct','Qwen2.5-32B-Instruct','Qwen3-14B','Qwen3-32B',
    ]
    return [d for d in dirs if (BASE/d).exists() and
            any((BASE/d/r).exists() for r in
                ['results_hysteresis_vllm','results_hysteresis_gemini','results_hysteresis_openai'])]

def draw_theory_vs_obs(ax):
    z_max = 35.0 / N_REGULAR

    all_data = []
    for m in detect_models():
        all_data.extend(build_comparison(m))

    th_p = np.array([d['zcp'] for d in all_data if d['zcp'] and d['fz'] and d['zcp'] <= z_max])
    ob_p = np.array([d['fz']  for d in all_data if d['zcp'] and d['fz'] and d['zcp'] <= z_max])
    th_m = np.array([d['zcm'] for d in all_data if d['zcm'] and d['bz'] and d['zcm'] <= z_max])
    ob_m = np.array([d['bz']  for d in all_data if d['zcm'] and d['bz'] and d['zcm'] <= z_max])

    ax.scatter(th_p, ob_p, s=25, c=C_FWD, alpha=0.25, edgecolors='none', zorder=2)
    ax.scatter(th_m, ob_m, s=25, c=C_BWD, alpha=0.25, edgecolors='none', zorder=2)

    bins = np.linspace(0, z_max, 12)
    bc   = (bins[:-1] + bins[1:]) / 2
    for th, ob, col, ls, lbl in [
        (th_p, ob_p, C_FWD, '-',  'Forward'),
        (th_m, ob_m, C_BWD, '--', 'Backward'),
    ]:
        bx, bm, bs = [], [], []
        for i in range(len(bc)):
            sel = ob[(th >= bins[i]) & (th < bins[i+1])]
            if len(sel) >= 2:
                bx.append(bc[i]); bm.append(np.mean(sel)); bs.append(np.std(sel))
        if bx:
            bx = np.array(bx); bm = np.array(bm); bs = np.array(bs)
            ax.fill_between(bx, bm-bs, bm+bs, color=col, alpha=0.2, zorder=4)
            ax.plot(bx, bm, color=col, lw=2.5, ls=ls, alpha=0.95, zorder=8, label=lbl)

    ax.plot([0, z_max], [0, z_max], 'k--', lw=1.8, alpha=0.7, label='Theory', zorder=3)
    ax.set_xlabel(r'Theoretical $z_c$', fontsize=13)
    ax.set_ylabel(r'Observed $z_c$', fontsize=13)
    ax.set_xlim(0, z_max); ax.set_ylim(0, z_max)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper left')


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

print('Panel (a): tipping point …')
draw_tipping(axes[0])

print('Panel (b): hysteresis loop …')
draw_hysteresis(axes[1])

print('Panel (c): theory vs observation …')
draw_theory_vs_obs(axes[2])

for ax, lbl in zip(axes, ['(a)', '(b)', '(c)']):
    ax.text(-0.10, 1.04, lbl, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='bottom', ha='left')

plt.tight_layout(rect=[0, 0, 1, 0.97])

out_png = BASE / 'figure4_tipping_hysteresis_theory.png'
out_pdf = BASE / 'figure4_tipping_hysteresis_theory.pdf'
plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f'Saved: {out_png}')
plt.close()

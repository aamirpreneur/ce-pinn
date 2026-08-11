"""Candidate figures for the paper, for the author to choose among.

Every number plotted here comes from the same functions already used to
produce the paper's tables, imported directly rather than re-derived:
  analyze_rlw1.load_run / build_trainer / analyze   -> results.tex Table 2
  ceiling_sweep.PROBLEMS / analyze                  -> results.tex Table 4

Nothing here is fit for a fresh claim; every plotted quantity already has a
table entry in the manuscript. The point of these figures is to show the
same results visually, not to introduce new ones.

Saved as vector PDF only (no PNG/raster), per the CPC Guide for Authors:
line drawings must be EPS or PDF, not TIFF/JPG/PNG, which are for
photographs and halftones. matplotlib's PDF backend embeds text as text
(not outlines), so it stays editable, matching "submit math equations and
artwork as editable text, not as images" in spirit.

Output goes to a scratch directory, not results/ or paper/, since none of
these has been chosen yet.

Run:  py -3 tools/make_candidate_figures.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_rlw1 import load_run, build_trainer, analyze as analyze_rlw1
from ceiling_sweep import PROBLEMS, analyze as analyze_ceiling

torch.set_default_dtype(torch.float64)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
OUT = os.environ.get('FIG_OUT', os.path.join(ROOT, '_candidate_figures'))
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,   # embed as real text, not a bitmap font
})

RLW1_VARIANTS = ['vanilla', 'hard_bc', 'ce_poly', 'ce_helm', 'ce_full']
LABELS = {'vanilla': 'vanilla', 'hard_bc': 'hard_bc', 'ce_poly': 'ce_poly',
         'ce_helm': 'ce_helm', 'ce_full': 'ce_full'}
# plain matplotlib text (not mathtext), so underscores need no escaping;
# an earlier version wrote 'hard\\_bc' out of habit from LaTeX source and
# every legend printed a literal backslash
COLORS = {'vanilla': 'tab:gray', 'hard_bc': 'tab:blue', 'ce_poly': 'tab:orange',
          'ce_helm': 'tab:red', 'ce_full': 'tab:green'}


def latest_run(prefix):
    import glob
    runs = sorted(glob.glob(os.path.join(RES, prefix, '2026*')))
    if not runs:
        raise FileNotFoundError(prefix)
    return runs[-1]


def save(fig, name):
    path = os.path.join(OUT, name + '.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f'  wrote {path}')


# ---------------------------------------------------------------------
def fig_solution_profile():
    """A. Predicted vs. exact solution, RLW1 hard_bc, at t = T, with the
    error panel below. The most standard figure in this literature: what
    the network actually produces, not just a summary statistic of it."""
    run_dir = latest_run('rlw1_hard_bc')
    tmp = os.path.join(ROOT, 'tests', '_figtmp')
    cfg, res, hist, sd = load_run(run_dir)
    tr, arch = build_trainer(cfg, sd, tmp)
    T = cfg['T']
    x = tr.x.cpu().numpy()
    from ce_pinn.equations import rlw_soliton
    _, _, u_t = tr._diagnostics(T)
    u = u_t.cpu().numpy()
    u_ex = rlw_soliton(x, T, eps=cfg['eps'], mu=cfg['mu'], c=0.1, x0=0.0)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.5, 4.2), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1]})
    a1.plot(x, u_ex, 'k-', lw=1.8, label='Exact')
    a1.plot(x, u, 'r--', lw=1.3, label='CE-PINN (hard_bc)')
    a1.set_ylabel('$u(x,T)$')
    a1.legend(frameon=False)
    a2.semilogy(x, np.abs(u - u_ex) + 1e-18, 'b-', lw=0.9)
    a2.set_xlabel('$x$')
    a2.set_ylabel('$|{\\rm error}|$')
    fig.suptitle(f'RLW1, $T={T:g}$, $L_2$ relative error {res["L2_relative"]:.2e}')
    save(fig, 'A_solution_profile_rlw1')


def fig_error_vs_time():
    """B. L2 relative error vs. t, all five RLW1 variants. Currently this
    exists only in results/ANALYSIS_RLW1.md; the paper's Table 2 reports
    only the t = T endpoint."""
    tmp = os.path.join(ROOT, 'tests', '_figtmp')
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for v in RLW1_VARIANTS:
        run_dir = latest_run(f'rlw1_{v}')
        *_, rows = analyze_rlw1(run_dir, tmp)  # returns (cfg, res, hist, arch, rows)
        t = [r['t'] for r in rows]
        e = [r['e_tot'] for r in rows]
        ax.semilogy(t, e, '-o', ms=2.5, lw=1.2, color=COLORS[v], label=LABELS[v])
    ax.set_xlabel('$t$')
    ax.set_ylabel('$L_2$ relative error')
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.set_title('RLW1: error growth over the run')
    save(fig, 'B_error_vs_time_rlw1')


def fig_i2_drift_vs_time():
    """C. I2 (the derivative-dependent invariant) drift vs. t, same five
    variants. Pairs naturally with B: conservation is flat for ce_helm
    while the error above still grows at the same rate as hard_bc's."""
    tmp = os.path.join(ROOT, 'tests', '_figtmp')
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for v in RLW1_VARIANTS:
        run_dir = latest_run(f'rlw1_{v}')
        *_, rows = analyze_rlw1(run_dir, tmp)  # returns (cfg, res, hist, arch, rows)
        t = [r['t'] for r in rows]
        d2 = [max(r['I2'], 1e-18) for r in rows]
        ax.semilogy(t, d2, '-o', ms=2.5, lw=1.2, color=COLORS[v], label=LABELS[v])
    ax.set_xlabel('$t$')
    ax.set_ylabel('$I_2$ relative drift')
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.set_title('RLW1: conservation of the derivative-dependent invariant')
    save(fig, 'C_i2_drift_vs_time_rlw1')


def fig_ceiling_validity_radius():
    """D. Theorem 6 validity radius: ratio (realized / predicted leading
    term) vs. perturbation amplitude, all four ceiling_sweep problems.
    Visualizes Table 4 (tab:ceilsweep) and is probably the strongest
    single figure for the paper's central theoretical claim: the ratio
    sits at 1 below ~1e-2 and departs above it."""
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    markers = {'RLW1  (small soliton)': 'o', 'RLW1T (tall soliton)': 's',
              'CH1   (Camassa-Holm)': '^', 'RLW2  (two solitons)': 'd'}
    for spec in PROBLEMS:
        idem, rows = analyze_ceiling(spec)
        a = [r[0] for r in rows]
        ratio = [r[5] for r in rows]
        ax.semilogx(a, ratio, '-' + markers.get(spec['name'], 'o'), ms=4,
                    lw=1.2, label=spec['name'].split('(')[0].strip())
    ax.axhline(1.0, color='k', lw=0.7, ls=':')
    ax.set_xlabel('perturbation amplitude (relative $L_2$)')
    ax.set_ylabel('realized / predicted improvement')
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('Theorem 6: validity radius of the first-order bound')
    save(fig, 'D_ceiling_validity_radius')


def fig_newton_tolerance():
    """E. Tolerable I2 shortfall before the lam2 clamp binds, vs. grid
    spacing h, the four problems of results.tex Table 5 (tab:deficit).
    Reads the single already-verified table in results/ANALYSIS_NEWTON.md
    ("problem | dx | mu | lam2_min | tolerable I2 deficit") rather than
    re-running the bisection, which takes several minutes."""
    import re
    md = open(os.path.join(RES, 'ANALYSIS_NEWTON.md'), encoding='utf-8').read()
    table = md.split('## How much of an I2 deficit')[1]
    rows = re.findall(
        r'\|\s*([\w.\-() ]+?)\s*\|\s*([\d.]+)\s*\|\s*[\d.]+\s*\|'
        r'\s*-?[\d.eE+-]+\s*\|\s*([\d.]+)%\s*\|', table)
    if not rows:
        print('  SKIPPED newton tolerance figure: table format did not parse '
              'as expected, check results/ANALYSIS_NEWTON.md by hand')
        return
    names = [r[0].strip() for r in rows]
    h = np.array([float(r[1]) for r in rows])
    t = np.array([float(r[2]) for r in rows])
    order = np.argsort(h)
    h, t = h[order], t[order]
    names = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(h, t, 'o-', color='tab:purple')
    for hi, ti, ni in zip(h, t, names):
        ax.annotate(ni.split('(')[0].strip(), (hi, ti),
                    textcoords='offset points', xytext=(5, 5), fontsize=7)
    ax.margins(y=0.18)  # headroom so the top-right point's label isn't clipped
    ax.set_xlabel('grid spacing $h$')
    ax.set_ylabel('tolerable $I_2$ shortfall (%)')
    ax.set_title(r'Newton solve: how much $I_2$ deficit before the '
                r'$\lambda_2$ bound binds')
    save(fig, 'E_newton_tolerance_vs_h')


if __name__ == '__main__':
    print(f'writing candidate figures to {OUT}\n')
    for fn in (fig_solution_profile, fig_error_vs_time, fig_i2_drift_vs_time,
              fig_ceiling_validity_radius, fig_newton_tolerance):
        print(f'{fn.__name__} ...')
        try:
            fn()
        except Exception as e:
            print(f'  FAILED: {type(e).__name__}: {e}')
    print('\ndone')

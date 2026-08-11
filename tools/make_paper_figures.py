"""Generate publication-quality figures and tables from saved results.

Usage: python tools/make_paper_figures.py results/rlw1_ce_helm/20260725-143000/
   or: python tools/make_paper_figures.py results/  (processes all runs)

Loads model checkpoints + solution_final.npz, produces:
  - Solution comparison plot (pred vs exact)
  - Error heatmap
  - Conservation drift over time
  - Loss convergence curve
  - Summary error table (LaTeX)
"""
import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def plot_solution_1d(run_dir):
    """Plot predicted vs exact solution."""
    sol = np.load(os.path.join(run_dir, 'solution_final.npz'))
    x, u_pred, u_exact = sol['x'], sol['u_pred'], sol['u_exact']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(x, u_exact, 'k-', lw=2, label='Exact')
    ax1.plot(x, u_pred, 'r--', lw=1.5, label='CE-PINN')
    ax1.set_ylabel('u(x, T)')
    ax1.legend()
    ax1.set_title(f'Solution at T = {sol["t"]:.1f}')

    error = np.abs(u_pred - u_exact)
    ax2.semilogy(x, error + 1e-16, 'b-', lw=1)
    ax2.set_xlabel('x')
    ax2.set_ylabel('|error|')
    ax2.set_ylim(bottom=1e-16)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'fig_solution.pdf'))
    plt.savefig(os.path.join(run_dir, 'fig_solution.png'))
    plt.close()
    print(f'  Saved fig_solution.pdf')


def plot_solution_2d(run_dir):
    """Plot 2D solution comparison."""
    sol = np.load(os.path.join(run_dir, 'solution_final.npz'))
    x, y = sol['x'], sol['y']
    u_pred, u_exact = sol['u_pred'], sol['u_exact']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    X, Y = np.meshgrid(x, y, indexing='ij')

    vmin = min(u_exact.min(), u_pred.min())
    vmax = max(u_exact.max(), u_pred.max())

    im0 = axes[0].pcolormesh(X, Y, u_exact, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title('Exact')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(X, Y, u_pred, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title('CE-PINN')
    plt.colorbar(im1, ax=axes[1])

    error = np.abs(u_pred - u_exact)
    im2 = axes[2].pcolormesh(X, Y, error, cmap='hot_r')
    axes[2].set_title('|Error|')
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'fig_solution_2d.pdf'))
    plt.savefig(os.path.join(run_dir, 'fig_solution_2d.png'))
    plt.close()
    print(f'  Saved fig_solution_2d.pdf')


def plot_loss_curve(run_dir):
    """Plot training loss convergence."""
    hist_path = os.path.join(run_dir, 'history.json')
    if not os.path.exists(hist_path):
        return

    history = json.load(open(hist_path))
    steps = [h['global_step'] if 'global_step' in h else h.get('step', i)
             for i, h in enumerate(history)]
    losses = [h.get('loss_pde', h.get('loss', h.get('loss_total', None)))
              for h in history]
    losses = [l for l in losses if l is not None]
    steps = steps[:len(losses)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(steps, losses, 'b-', lw=1)
    ax.set_xlabel('Training step')
    ax.set_ylabel('PDE loss')
    ax.set_title('Training convergence')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'fig_loss.pdf'))
    plt.savefig(os.path.join(run_dir, 'fig_loss.png'))
    plt.close()
    print(f'  Saved fig_loss.pdf')


def plot_conservation_drift(run_dir):
    """Plot conservation drift over training from history."""
    hist_path = os.path.join(run_dir, 'history.json')
    if not os.path.exists(hist_path):
        return

    history = json.load(open(hist_path))
    drift_entries = [h for h in history if 'I1_drift' in h]
    if not drift_entries:
        return

    steps = [h['global_step'] for h in drift_entries]
    d1 = [h['I1_drift'] for h in drift_entries]
    d2 = [h['I2_drift'] for h in drift_entries]
    d3 = [h['I3_drift'] for h in drift_entries]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(steps, d1, 'b-o', ms=3, label='I₁ drift')
    ax.semilogy(steps, d2, 'r-s', ms=3, label='I₂ drift')
    ax.semilogy(steps, d3, 'g-^', ms=3, label='I₃ drift')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Relative drift |I(t)−C|/|C|')
    ax.set_title('Conservation drift during training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'fig_conservation.pdf'))
    plt.savefig(os.path.join(run_dir, 'fig_conservation.png'))
    plt.close()
    print(f'  Saved fig_conservation.pdf')


def make_latex_table(results_dirs):
    """Generate LaTeX comparison table across runs."""
    rows = []
    for d in sorted(results_dirs):
        rpath = os.path.join(d, 'results.json')
        if not os.path.exists(rpath):
            continue
        r = json.load(open(rpath))
        name = os.path.basename(os.path.dirname(d)) if os.path.basename(d).startswith('20') else os.path.basename(d)
        rows.append({
            'name': name,
            'variant': r.get('variant', '?'),
            'L2_rel': r.get('L2_relative', float('nan')),
            'Linf': r.get('Linf', float('nan')),
            'C1_drift': r.get('C1_drift', float('nan')),
            'C2_drift': r.get('C2_drift', float('nan')),
            'C3_drift': r.get('C3_drift', float('nan')),
            'time_s': r.get('training_time_s', float('nan')),
        })

    if not rows:
        print('No results.json files found')
        return

    print('\n% LaTeX table:')
    print('\\begin{tabular}{lcccccc}')
    print('\\toprule')
    print('Variant & $L^2_{\\rm rel}$ & $L^\\infty$ & $\\Delta I_1$ & $\\Delta I_2$ & $\\Delta I_3$ & Time (s) \\\\')
    print('\\midrule')
    for r in rows:
        print(f'{r["variant"]} & {r["L2_rel"]:.2e} & {r["Linf"]:.2e} & '
              f'{r["C1_drift"]:.2e} & {r["C2_drift"]:.2e} & {r["C3_drift"]:.2e} & '
              f'{r["time_s"]:.0f} \\\\')
    print('\\bottomrule')
    print('\\end{tabular}')


def process_run(run_dir):
    """Process a single run directory."""
    print(f'\nProcessing: {run_dir}')

    sol_path = os.path.join(run_dir, 'solution_final.npz')
    if os.path.exists(sol_path):
        sol = np.load(sol_path)
        if 'y' in sol.files:
            plot_solution_2d(run_dir)
        else:
            plot_solution_1d(run_dir)

    plot_loss_curve(run_dir)
    plot_conservation_drift(run_dir)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python tools/make_paper_figures.py <results_dir>')
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isfile(os.path.join(target, 'results.json')):
        process_run(target)
    else:
        runs = []
        for root, dirs, files in os.walk(target):
            if 'results.json' in files:
                runs.append(root)
        if runs:
            for r in sorted(runs):
                process_run(r)
            make_latex_table(runs)
        else:
            print(f'No results.json found under {target}')

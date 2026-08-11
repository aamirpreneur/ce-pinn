"""Test the ceiling on every problem, without training anything.

Theorem 6 is a statement about the operator and the problem. Testing it only on
trained networks makes it inherit every training failure, which is why it
appeared to hold on RLW1 and be untestable elsewhere -- three of the four
problems exceed what a plain PINN can propagate and never produce a field
accurate enough to test an asymptotic result.

That confound is removable. Perturb the *exact* solution by a controlled amount
and project it. No network, no training, no problem excluded because it failed
to converge, and the error level is set rather than hoped for -- so the validity
radius of a first-order expansion is measured instead of assumed.

For each problem, at t = T:

    u~ = u_exact + a * p,   p smooth, band-limited, vanishing at both ends
    project with the exact time-dependent targets and boundary data
    compare the realized improvement against 1 - sqrt(1 - V)

`a` runs over eight orders of magnitude. The ratio should approach 1 as a -> 0
and depart as the O(||e||) remainder takes over; where it departs is the number
worth reporting.

Run:  py -3 tools/ceiling_sweep.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from ce_pinn.equations import rlw_soliton_torch, ch_solitary_torch
from ce_pinn import projection as proj

torch.set_default_dtype(torch.float64)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
OUT_MD = os.path.join(RES, 'ANALYSIS_CEILING.md')
REF2 = os.path.join(RES, 'reference', 'rlw2_reference.npz')

AMPS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]


def rlw(c, x0, eps=1.0, mu=1.0):
    def f(x, t):
        return rlw_soliton_torch(x, torch.full_like(x, float(t)), eps=eps,
                                 mu=mu, c=c, x0=x0)
    return f


def ch(c, kappa, x0):
    def f(x, t):
        return ch_solitary_torch(x, torch.full_like(x, float(t)), c=c,
                                 kappa=kappa, x0=x0)
    return f


PROBLEMS = [
    dict(name='RLW1  (small soliton)', xa=-40.0, xb=60.0, N=401, T=20.0,
         mu=1.0, share=1.8, exact=rlw(0.1, 0.0), trains='yes'),
    dict(name='RLW1T (tall soliton)', xa=0.0, xb=100.0, N=601, T=20.0,
         mu=1.0, share=11.3, exact=rlw(5.33 / 3.0, 15.0), trains='no'),
    dict(name='CH1   (Camassa-Holm)', xa=0.0, xb=40.0, N=601, T=6.0,
         mu=1.0, share=28.0, exact=ch(1.0, 0.1, 10.0), trains='poorly'),
    dict(name='RLW2  (two solitons)', ref=REF2, mu=1.0, share=10.8,
         trains='no'),
]


def grid(xa, xb, N, mu):
    dx = (xb - xa) / (N - 1)
    x = torch.linspace(xa, xb, N, dtype=torch.float64)
    q = torch.full((N,), dx, dtype=torch.float64)
    q[0] = q[-1] = dx / 2
    D1 = proj.central_d1_matrix(N, dx)
    A = torch.diag(q) + mu * (D1.T @ torch.diag(q) @ D1)
    return x, dx, q, D1, A


def perturbation(x, ue, rng_seed=0):
    """Smooth, band-limited, zero at both ends, unit relative L2."""
    rng = np.random.default_rng(rng_seed)
    xn = x.numpy()
    s = (xn - xn[0]) / (xn[-1] - xn[0])
    p = np.zeros_like(xn)
    for k in range(1, 9):
        p += rng.normal() * np.sin(k * np.pi * s) / k
    p *= 4.0 * s * (1.0 - s)
    p /= np.sqrt(np.sum(p ** 2) / np.sum(ue ** 2))
    return p


def analyze(spec):
    if 'ref' in spec:
        d = np.load(spec['ref'])
        xn, un = d['x'], d['u'][-1]
        N, xa, xb = len(xn), float(xn[0]), float(xn[-1])
        T = float(d['T'])
        x, dx, q, D1, A = grid(xa, xb, N, spec['mu'])
        ue = torch.tensor(un)
    else:
        xa, xb, N, T = spec['xa'], spec['xb'], spec['N'], spec['T']
        x, dx, q, D1, A = grid(xa, xb, N, spec['mu'])
        ue = spec['exact'](x, T)

    Qn = np.diag(q.numpy())
    An = A.numpy()
    uen = ue.numpy()
    nrm = lambda v: float(np.sqrt(v @ Qn @ v))
    l2 = lambda v: float(np.sqrt(np.sum(v ** 2) / np.sum(uen ** 2)))

    C1 = float(torch.dot(q, ue))
    C2 = float(ue @ A @ ue)
    ga, gb = float(ue[0]), float(ue[-1])

    # Theorem 7 first: if the exact solution is not a fixed point, every number
    # below is measuring the displacement instead of the ceiling.
    up0 = proj.helmholtz_projection(ue.clone(), C1, C2, spec['mu'], dx, q, D1,
                                    ga=ga, gb=gb)[0]
    idem = nrm((up0 - ue).numpy())

    # four-column normal space: the two invariants and the two Dirichlet values
    qn = q.numpy()
    g1 = np.ones(N)
    g2 = 2.0 * (An @ uen) / qn
    ea = np.zeros(N)
    ea[0] = 1.0 / qn[0]
    eb = np.zeros(N)
    eb[-1] = 1.0 / qn[-1]
    B = np.stack([g1, g2, ea, eb], axis=1)
    G = B.T @ Qn @ B

    p = perturbation(x, uen)
    rows = []
    for a in AMPS:
        u = uen + a * p
        u[0], u[-1] = uen[0], uen[-1]
        e = u - uen
        eN = B @ np.linalg.solve(G, B.T @ Qn @ e)
        vis = (nrm(eN) / nrm(e)) ** 2
        ceiling = 1.0 - np.sqrt(max(0.0, 1.0 - vis))
        try:
            upn = proj.helmholtz_projection(
                torch.tensor(u), C1, C2, spec['mu'], dx, q, D1,
                ga=ga, gb=gb)[0].numpy()
        except Exception:
            rows.append((a, l2(e), vis, ceiling, float('nan'), float('nan')))
            continue
        before, after = l2(e), l2(upn - uen)
        realized = 1.0 - after / before
        rows.append((a, before, vis, ceiling, realized,
                     realized / ceiling if ceiling > 0 else float('nan')))
    return idem, rows


def main():
    lines = []

    def out(s=''):
        print(s)
        lines.append(s)

    out('# The ceiling, tested on every problem without training')
    out()
    out('Theorem 6 concerns the operator and the problem, not the optimizer.')
    out('Perturbing the exact solution by a set amount and projecting tests it')
    out('on all four problems -- including the three no plain PINN can')
    out('propagate -- across eight orders of magnitude of error. Nothing is')
    out('excluded for having failed to train.')
    out()
    out('`ratio` is realized / predicted. Theorem 6 is first order, so it')
    out('should sit near 1 for small errors and depart as the remainder takes')
    out('over. Where it departs is the validity radius.')
    out()
    out('> **The ceiling VALUES here are not representative of real PINN')
    out('> error.** The perturbation is smooth and band-limited, and it aligns')
    out('> with the constraint gradients far better than a trained network\'s')
    out('> error does: it gives a visible fraction around 27% on RLW1 where the')
    out('> trained runs give 2.3-3.7%. So the ceilings below are roughly an')
    out('> order of magnitude above what a network actually sees.')
    out('>')
    out('> What this sweep establishes is that Theorem 6 *predicts* the')
    out('> realized improvement, whatever that improvement happens to be. That')
    out('> is a property of the operator and holds for any error shape, which')
    out('> is exactly why a synthetic perturbation is the right instrument for')
    out('> it. Read the ratio column, not the ceiling column.')
    out()

    summary = []
    for spec in PROBLEMS:
        idem, rows = analyze(spec)
        out(f"## {spec['name']}")
        out()
        out(f"derivative share {spec['share']}% · a plain PINN trains this: "
            f"**{spec['trains']}**")
        out()
        out(f'Theorem 7 check, ‖P(u_exact) − u_exact‖ = `{idem:.2e}`'
            + ('' if idem < 1e-12 else
               '  ← **not a fixed point; numbers below are unreliable**'))
        out()
        out('| perturbation | L2 before | visible | predicted ceiling | '
            'realized | ratio |')
        out('|---|---|---|---|---|---|')
        for a, before, vis, ceil, real, ratio in rows:
            out(f'| {a:.0e} | {before:.3e} | {100*vis:.3f}% | '
                f'{100*ceil:.4f}% | {100*real:.4f}% | '
                + (f'{ratio:.4f} |' if np.isfinite(ratio) else 'diverged |'))
        out()
        good = [r[5] for r in rows if r[1] < 1e-2 and np.isfinite(r[5])]
        if good:
            summary.append((spec['name'], spec['share'], min(good), max(good),
                            float(np.median(good)), idem))
            out(f'Over the rows with L2 < 1e-02: ratio ranges '
                f'{min(good):.4f} to {max(good):.4f}, median '
                f'{float(np.median(good)):.4f}.')
        out()

    out('## Summary')
    out()
    out('| problem | derivative share | ratio min | max | median | Thm 7 |')
    out('|---|---|---|---|---|---|')
    for nm, sh, lo, hi, md, idem in summary:
        out(f'| {nm} | {sh}% | {lo:.4f} | {hi:.4f} | **{md:.4f}** | '
            f'{idem:.1e} |')
    out()
    if summary:
        allmed = [s[4] for s in summary]
        out(f'Median ratio across all four problems: '
            f'{min(allmed):.4f} to {max(allmed):.4f}.')
    out()

    os.makedirs(RES, exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nwritten: {os.path.relpath(OUT_MD, ROOT)}')


if __name__ == '__main__':
    main()

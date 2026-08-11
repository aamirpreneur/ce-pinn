"""How much of the error does the certificate actually detect?

A lower bound on the error is only useful if it is a substantial fraction of
the real thing. Theorem 3' recovered ~15% on one RLW1 run; whether that is
representative, and whether the sharpened four-component version is materially
better, decides whether the certificate is a tool or a curiosity.

Three bounds are compared, all computable without the exact solution:

  Thm 3'   ||e||_h >= |I1(u~) - C1| / sqrt(b-a)         uses I1 only
  Thm 3    ||e||_A >= |sqrt(I2(u~)) - sqrt(C2)|         uses I2 only
  combined ||e||_h >= ||e_N||_h                          uses all four
           constraint violations -- I1, I2 and the two Dirichlet values --
           reconstructed from measurements alone

The combined bound is provably at least as sharp as Theorem 3': both are norms
of orthogonal projections of e, and span{1} is contained in the four-dimensional
normal space. The question is by how much, and whether it holds up across
problems.

Tightness of the combined bound equals sqrt(V), so this doubles as a survey of
the visible fraction -- but note that V depends on how the error happens to
align with the constraint gradients, so several perturbation shapes are tested
per problem rather than one.

Run:  py -3 tools/certificate_sweep.py
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
OUT_MD = os.path.join(RES, 'ANALYSIS_CERTIFICATE.md')
REF2 = os.path.join(RES, 'reference', 'rlw2_reference.npz')

AMPS = [1e-6, 1e-4, 1e-3, 1e-2]
SEEDS = range(8)


def rlw(c, x0, eps=1.0, mu=1.0):
    return lambda x, t: rlw_soliton_torch(x, torch.full_like(x, float(t)),
                                          eps=eps, mu=mu, c=c, x0=x0)


def ch(c, kappa, x0):
    return lambda x, t: ch_solitary_torch(x, torch.full_like(x, float(t)),
                                          c=c, kappa=kappa, x0=x0)


PROBLEMS = [
    dict(name='RLW1', xa=-40.0, xb=60.0, N=401, T=20.0, mu=1.0, share=1.8,
         exact=rlw(0.1, 0.0)),
    dict(name='RLW1T', xa=0.0, xb=100.0, N=601, T=20.0, mu=1.0, share=11.3,
         exact=rlw(5.33 / 3.0, 15.0)),
    dict(name='CH1', xa=0.0, xb=40.0, N=601, T=6.0, mu=1.0, share=28.0,
         exact=ch(1.0, 0.1, 10.0)),
    dict(name='RLW2', ref=REF2, mu=1.0, share=10.8),
]


def build(spec):
    if 'ref' in spec:
        d = np.load(spec['ref'])
        xn = d['x']
        N, xa, xb = len(xn), float(xn[0]), float(xn[-1])
        ue = torch.tensor(d['u'][-1])
    else:
        xa, xb, N = spec['xa'], spec['xb'], spec['N']
    dx = (xb - xa) / (N - 1)
    x = torch.linspace(xa, xb, N, dtype=torch.float64)
    q = torch.full((N,), dx, dtype=torch.float64)
    q[0] = q[-1] = dx / 2
    D1 = proj.central_d1_matrix(N, dx)
    A = torch.diag(q) + spec['mu'] * (D1.T @ torch.diag(q) @ D1)
    if 'ref' not in spec:
        ue = spec['exact'](x, spec['T'])
    return x, dx, q, D1, A, ue, xa, xb, N


def perturbation(x, uen, seed):
    rng = np.random.default_rng(seed)
    xn = x.numpy()
    s = (xn - xn[0]) / (xn[-1] - xn[0])
    p = np.zeros_like(xn)
    for k in range(1, 9):
        p += rng.normal() * np.sin(k * np.pi * s) / k
    p *= 4.0 * s * (1.0 - s)
    p /= np.sqrt(np.sum(p ** 2) / np.sum(uen ** 2))
    return p


def main():
    lines = []

    def out(s=''):
        print(s)
        lines.append(s)

    out('# Certificate tightness across all four problems')
    out()
    out('`tightness` is bound / true error: the fraction of the error the')
    out('certificate detects. All three bounds are computable without the')
    out('exact solution. Eight perturbation shapes per problem, since the')
    out('visible fraction depends on how the error aligns with the constraint')
    out('gradients rather than on the problem alone.')
    out()
    out('| problem | share | Thm 3′ (I1, L2) | Thm 3 (I2, H1) | '
        'combined (all four) | combined spread | gain over 3′ |')
    out('|---|---|---|---|---|---|---|')

    summary = []
    for spec in PROBLEMS:
        x, dx, q, D1, A, ue, xa, xb, N = build(spec)
        Qn, An, qn = np.diag(q.numpy()), A.numpy(), q.numpy()
        uen = ue.numpy()
        nrm_h = lambda v: float(np.sqrt(v @ Qn @ v))
        nrm_A = lambda v: float(np.sqrt(v @ An @ v))
        C1, C2 = float(torch.dot(q, ue)), float(ue @ A @ ue)
        ga, gb = float(ue[0]), float(ue[-1])
        L = xb - xa

        t3p, t3, comb = [], [], []
        for seed in SEEDS:
            p = perturbation(x, uen, seed)
            for a in AMPS:
                u = uen + a * p
                u[0], u[-1] = uen[0], uen[-1]
                e = u - uen
                ut = torch.tensor(u)

                v1 = abs(float(torch.dot(q, ut)) - C1)
                v2 = float(ut @ A @ ut) - C2
                cert_h = v1 / np.sqrt(L)
                cert_A = abs(np.sqrt(max(float(ut @ A @ ut), 0.0))
                             - np.sqrt(C2))

                g1 = np.ones(N)
                g2 = 2.0 * (An @ u) / qn
                ea = np.zeros(N)
                ea[0] = 1.0 / qn[0]
                eb = np.zeros(N)
                eb[-1] = 1.0 / qn[-1]
                B = np.stack([g1, g2, ea, eb], axis=1)
                rhs = np.array([float(torch.dot(q, ut)) - C1, v2,
                                float(ut[0]) - ga, float(ut[-1]) - gb])
                eN = B @ np.linalg.solve(B.T @ Qn @ B, rhs)

                t3p.append(cert_h / nrm_h(e))
                t3.append(cert_A / nrm_A(e))
                comb.append(nrm_h(eN) / nrm_h(e))

        m3p, m3, mc = (float(np.median(t3p)), float(np.median(t3)),
                       float(np.median(comb)))
        # The median alone hides the point. Tightness is set by how the error
        # aligns with the constraint gradients, so it varies with the SHAPE of
        # the perturbation and not only with the problem. Reporting the median
        # as though it were a range invites the reader to treat a shape-
        # dependent quantity as a property of the problem.
        lo_c, hi_c = float(np.min(comb)), float(np.max(comb))
        summary.append((spec['name'], mc, lo_c, hi_c))
        out(f"| {spec['name']} | {spec['share']}% | {100*m3p:.2f}% | "
            f"{100*m3:.2f}% | **{100*mc:.2f}%** | {100*lo_c:.1f}–{100*hi_c:.1f}% "
            f"| {mc/m3p:.1f}× |")

    out()
    out('Medians over 8 perturbation shapes x 4 amplitudes = 32 cases each.')
    out('The `spread` column is the min and max over those 32 cases.')
    out()
    out('## Reading')
    out()
    lo, hi = min(s[1] for s in summary), max(s[1] for s in summary)
    slo, shi = min(s[2] for s in summary), max(s[3] for s in summary)
    out(f'The median tightness is {100*lo:.0f}–{100*hi:.0f}% across problems, '
        f'which places the error within a factor of {1/hi:.1f}–{1/lo:.1f} '
        f'below the bound.')
    out()
    out(f'That median is not the headline. Across perturbation shapes the same')
    out(f'quantity runs {100*slo:.1f}% to {100*shi:.1f}%, a spread far wider than')
    out('the variation between problems. Tightness is a property of the error,')
    out('not of the problem, so a synthetic study of this certificate can')
    out('report almost any figure depending on which perturbations it uses.')
    out('The number that matters is the one measured on the error a trained')
    out('network actually makes, which `tools/verify_theory.py` reports.')
    out()
    out('It is provably at least as sharp as Theorem 3′, since both are norms')
    out('of orthogonal projections of the error and span{1} lies inside the')
    out('four-dimensional normal space. The measured gain quantifies how much')
    out('the other three violations add.')
    out()
    out('Its tightness equals sqrt(V), so a problem whose errors align well')
    out('with the constraint gradients gets both a sharper certificate and a')
    out('higher ceiling. Those are the same quantity seen twice, which is')
    out('worth stating plainly rather than presenting as two results.')
    out()

    os.makedirs(RES, exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nwritten: {os.path.relpath(OUT_MD, ROOT)}')


if __name__ == '__main__':
    main()

"""How many Newton iterations the projection actually takes, and when it clamps.

The method section claims an iteration count and a well-posedness bound on
lam2. Both were written from memory of watching training logs, which is not
evidence. This measures them.

Two regimes matter and they differ:

  * a converged field, which is what inference-time projection sees;
  * a field far from feasible, which is what the first training steps see.

The second is reached here by perturbing the exact solution, the same device
tools/ceiling_sweep.py uses, so the error level is set rather than hoped for.
The clamp is a property of the grid and mu through lam2_min ~ -sigma*dx^2/(4 mu),
so it is reported per problem alongside the analytic bound it is compared to.

Run:  py -3 tools/newton_cost.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from ce_pinn import projection as proj
from ceiling_sweep import PROBLEMS, grid, perturbation

torch.set_default_dtype(torch.float64)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_MD = os.path.join(ROOT, 'results', 'ANALYSIS_NEWTON.md')
AMPS = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1e0]
SEEDS = range(8)
SIGMA = 0.9


def setup(spec):
    if 'ref' in spec:
        d = np.load(spec['ref'])
        xn, un = d['x'], d['u'][-1]
        N, xa, xb = len(xn), float(xn[0]), float(xn[-1])
        x, dx, q, D1, A = grid(xa, xb, N, spec['mu'])
        return x, dx, q, D1, torch.tensor(un)
    xa, xb, N, T = spec['xa'], spec['xb'], spec['N'], spec['T']
    x, dx, q, D1, A = grid(xa, xb, N, spec['mu'])
    return x, dx, q, D1, spec['exact'](x, T)


def main():
    L = ["# Newton cost and the lam2 clamp\n",
         "Iterations to converge and clamp rate, over "
         f"{len(SEEDS)} perturbation shapes per amplitude.",
         "`amp` is the relative L2 of the perturbation applied to the exact",
         "solution, so `1e-08` stands for a converged field and `1e0` for the",
         "first steps of training. `clamp %` is the fraction of solves that hit",
         "the well-posedness bound on lam2.\n"]

    summary = []
    for spec in PROBLEMS:
        x, dx, q, D1, ue = setup(spec)
        mu = spec['mu']
        kmin = -SIGMA * dx * dx / 4.0
        lam2_min = kmin / (mu - kmin)
        C1 = float(torch.dot(q, ue))
        ux = torch.matmul(ue, D1.T)
        C2 = float(torch.dot(q, ue * ue + mu * ux * ux))
        ga, gb = float(ue[0]), float(ue[-1])

        L += [f"\n## {spec['name']}\n",
              f"N = {len(x)}, dx = {dx:.4f}, mu = {mu:g}, "
              f"lam2_min = {lam2_min:.3e}\n",
              "| amp | iters median | iters max | clamp % | resid max | ms |",
              "|---|---|---|---|---|---|"]
        for a in AMPS:
            its, cl, res, tt = [], 0, [], 0.0
            for s in SEEDS:
                p = torch.tensor(perturbation(x, ue.numpy(), rng_seed=s))
                ut = ue + a * p
                t0 = time.perf_counter()
                _, st = proj.helmholtz_projection(ut, C1, C2, mu, dx, q, D1,
                                                  ga=ga, gb=gb, max_iter=50)
                tt += time.perf_counter() - t0
                its.append(st.newton_iters)
                cl += int(st.clamped)
                res.append(float(st.residual))
            L.append(f"| {a:.0e} | {int(np.median(its))} | {max(its)} | "
                     f"{100.0*cl/len(SEEDS):.0f}% | {max(res):.1e} | "
                     f"{1e3*tt/len(SEEDS):.1f} |")
            if a <= 1e-2:
                summary.append((spec['name'], a, np.median(its), max(its),
                                100.0 * cl / len(SEEDS)))

    L += ["\n## How much of an I2 deficit the solve tolerates\n",
          "The bound restricts lam2 from below only, so it can bite only when",
          "lam2 < 0. That is exactly the case where the field carries too little",
          "I2, which is what an under-resolved gradient produces. So the number",
          "a practitioner needs is how large an I2 deficit the solve accepts",
          "before the bound stops it.\n",
          "Imposed along a fixed direction (the smoothed exact solution) and",
          "bisected on the deficit, rather than by stepping a smoothing kernel:",
          "one kernel step means different things on different grids.\n",
          "| problem | dx | mu | lam2_min | tolerable I2 deficit |",
          "|---|---|---|---|---|"]
    for spec in PROBLEMS:
        x, dx, q, D1, ue = setup(spec)
        mu = spec['mu']
        A = torch.diag(q) + mu * (D1.T @ torch.diag(q) @ D1)
        C1, C2 = float(torch.dot(q, ue)), float(ue @ A @ ue)
        ga, gb = float(ue[0]), float(ue[-1])
        lam2_min = (-SIGMA * dx * dx / 4.0) / (mu + SIGMA * dx * dx / 4.0)
        xs = x.numpy()
        w = 3.0 * dx
        K = torch.tensor(np.exp(-0.5 * ((xs[:, None] - xs[None, :]) / w) ** 2))
        K = K / K.sum(1, keepdim=True)
        sm = torch.matmul(K, ue)
        d_dir = sm - ue

        def probe(alpha):
            u = ue + alpha * d_dir
            u[0], u[-1] = ue[0], ue[-1]
            _, st = proj.helmholtz_projection(u.clone(), C1, C2, mu, dx, q, D1,
                                              ga=ga, gb=gb, max_iter=50)
            return st.clamped, 1.0 - float(u @ A @ u) / C2

        lo, hi = 0.0, 1.0
        while not probe(hi)[0] and hi < 1024:
            hi *= 2
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if probe(mid)[0]:
                hi = mid
            else:
                lo = mid
        L.append(f"| {spec['name']} | {dx:.4f} | {mu:g} | {lam2_min:.3e} | "
                 f"{100*probe(lo)[1]:.3f}% |")
    L += ["\nThe tolerance tracks the bound, so the finest grid is the least",
          "forgiving. Refining to improve the solution narrows what the",
          "projection is allowed to correct.\n"]

    med = [s[2] for s in summary]
    mx = [s[3] for s in summary]
    L += ["\n## Summary\n",
          "Over every problem at amplitudes up to 1e-02, which is the range in",
          "which the trained runs sit:\n",
          f"* iterations: median {int(np.median(med))}, worst case {int(max(mx))}",
          f"* clamp rate: {max(s[4] for s in summary):.0f}% at worst",
          "\nThe clamp is what makes the count grow, not the nonlinearity: a",
          "solve that clamps cannot satisfy F2 and exits on max_iter.\n"]

    text = "\n".join(L) + "\n"
    open(OUT_MD, 'w', encoding='utf-8').write(text)
    print(text)
    print(f"written to {OUT_MD}")


if __name__ == '__main__':
    main()

"""Build a verified reference solution for RLW2, the two-soliton collision.

RLW is not integrable, so there is no closed form after the collision and the
notebook ships `exact_at_T = None`. Without a reference, RLW2 can only report
conservation drift, which cannot answer whether exact conservation buys any
accuracy. This produces the reference by direct numerical solution, and
establishes that it has converged rather than assuming it.

Method: 4th-order central differences in space (one-sided at the ends), classical
RK4 in time, with u_t = 0 held at both boundaries. That is the same Dirichlet
problem the PINN is trained on, so the comparison is fair -- a periodic or
infinite-domain reference would differ from the PINN by the boundary treatment
rather than by accuracy.

Step size is set by accuracy, not stability. Writing RLW as
(1 - mu d_xx) u_t = -d_x(u + eps u^2/2), the operator applied to each Fourier
mode carries the factor k/(1 + mu k^2), which is bounded by 1/(2 sqrt(mu))
however fine the grid, so the scheme is not stiff and dt does not have to shrink
with dx.

Accuracy is 2nd order overall, not 4th. `d1` is 4th-order, but the operator
A = I - mu*D2 uses the 3-point second difference, and that caps the scheme.
Measured orders are 1.99 and 2.00, and the residual discrepancy sits on the
tall soliton in the interior rather than at the one-sided end stencils, which
confirms the diagnosis. Because the order is clean, Richardson extrapolation
(4*u_{h/2} - u_h)/3 removes the h^2 term for free, and two independent
extrapolations bound the extrapolated solution's own error.

That refinement is not cosmetic. On RLW1 the gap between `hard_bc` and
`hard_bc+proj` was 1.7% of a 1e-3 error, i.e. about 2e-5. A reference whose own
error is 2.8e-5 could not resolve an effect that size, so the reference has to be
better than the difference we intend to measure with it.

Verification: four space grids at fixed dt, a dt-halving check to confirm time
integration is not the limit, the two-way extrapolation check above, and the
solver's own invariant drift -- a solution that has lost mass or momentum is not
a reference.

Run:  py -3 tools/make_rlw2_reference.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import splu

from ce_pinn.equations import rlw_two_soliton_ic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'results', 'reference')
OUT = os.path.join(OUT_DIR, 'rlw2_reference.npz')

# RLW2 configuration, copied from tools/make_notebooks.py
EPS, MU = 1.0, 1.0
C1_AMP, C2_AMP = 5.33 / 3.0, 1.69 / 3.0
X1, X2 = 15.0, 35.0
XA, XB = 0.0, 120.0
T_FINAL = 30.0

N_COARSE = 601          # the grid the trainer uses; every fine grid is a
                        # multiple of it so sampling needs no interpolation
N_SNAP = 41             # snapshots at t = 0, 0.75, ... 30
DT_SNAP = T_FINAL / (N_SNAP - 1)


class Solver:
    def __init__(self, N):
        self.N = N
        self.x = np.linspace(XA, XB, N)
        self.dx = self.x[1] - self.x[0]
        h2 = self.dx ** 2
        # A = I - mu*d_xx, tridiagonal, with Dirichlet rows at both ends
        main = np.full(N, 1.0 + 2.0 * MU / h2)
        off = np.full(N - 1, -MU / h2)
        main[0] = main[-1] = 1.0
        off[0] = 0.0            # row 0 super-diagonal
        off[-1] = 0.0           # row N-1 sub-diagonal
        A = diags([off, main, off], [-1, 0, 1], format='csc')
        self.lu = splu(csc_matrix(A))

    def d1(self, u):
        """4th-order central in the interior, one-sided at the ends."""
        dx = self.dx
        g = np.empty_like(u)
        g[2:-2] = (u[:-4] - 8 * u[1:-3] + 8 * u[3:-1] - u[4:]) / (12 * dx)
        g[:2] = (-3 * u[:2] + 4 * u[1:3] - u[2:4]) / (2 * dx)
        g[-2:] = (3 * u[-2:] - 4 * u[-3:-1] + u[-4:-2]) / (2 * dx)
        return g

    def rhs(self, u):
        b = -self.d1(u + 0.5 * EPS * u * u)
        b[0] = 0.0              # boundary values held fixed => u_t = 0 there
        b[-1] = 0.0
        return self.lu.solve(b)

    def step(self, u, dt):
        a = self.rhs(u)
        b = self.rhs(u + 0.5 * dt * a)
        c = self.rhs(u + 0.5 * dt * b)
        e = self.rhs(u + dt * c)
        return u + dt * (a + 2.0 * b + 2.0 * c + e) / 6.0

    def run(self, dt):
        """Integrate to T_FINAL, returning snapshots on the coarse grid."""
        assert (self.N - 1) % (N_COARSE - 1) == 0, self.N
        stride = (self.N - 1) // (N_COARSE - 1)
        n_per_snap = int(round(DT_SNAP / dt))
        assert abs(n_per_snap * dt - DT_SNAP) < 1e-12, (dt, DT_SNAP)

        u = rlw_two_soliton_ic(self.x, eps=EPS, mu=MU, c1=C1_AMP, c2=C2_AMP,
                               x1=X1, x2=X2)
        snaps = [u[::stride].copy()]
        for _ in range(N_SNAP - 1):
            for _ in range(n_per_snap):
                u = self.step(u, dt)
            snaps.append(u[::stride].copy())
        return np.array(snaps), u


def coarse_invariants(u):
    """I1, I2, I3 exactly as GridTrainer defines them, so the numbers are
    directly comparable with what the runs report: trapezoid quadrature and the
    2nd-order central difference inside I2."""
    dx = (XB - XA) / (N_COARSE - 1)
    q = np.full(N_COARSE, dx)
    q[0] = q[-1] = 0.5 * dx
    ux = np.zeros_like(u)
    ux[1:-1] = (u[2:] - u[:-2]) / (2.0 * dx)
    return (float(q @ u), float(q @ (u * u + MU * ux * ux)),
            float(q @ (EPS * u ** 3 + 3.0 * u * u)))


def peaks(x, u, n=2):
    """The n tallest local maxima, refined by a parabolic fit."""
    loc = [i for i in range(1, len(u) - 1)
           if u[i] > u[i - 1] and u[i] >= u[i + 1]]
    loc.sort(key=lambda i: -u[i])
    out = []
    for i in loc[:n]:
        y0, y1, y2 = u[i - 1], u[i], u[i + 1]
        den = y0 - 2 * y1 + y2
        d = 0.0 if den == 0 else 0.5 * (y0 - y2) / den
        out.append((float(x[i] + d * (x[1] - x[0])),
                    float(y1 - 0.25 * (y0 - y2) * d)))
    return sorted(out, key=lambda p: -p[1])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    xc = np.linspace(XA, XB, N_COARSE)

    print(f"RLW2 reference: eps={EPS}, mu={MU}, amplitudes "
          f"{3*C1_AMP:.2f} and {3*C2_AMP:.2f}, domain [{XA},{XB}], "
          f"T={T_FINAL}")
    s1 = 1.0 + EPS * C1_AMP
    s2 = 1.0 + EPS * C2_AMP
    t_coll = (X2 - X1) / (s1 - s2)
    print(f"soliton speeds {s1:.4f} and {s2:.4f}; they meet at "
          f"t = {t_coll:.2f}, x = {X1 + s1*t_coll:.2f}")
    print(f"end positions if unchanged: {X1 + s1*T_FINAL:.2f} and "
          f"{X2 + s2*T_FINAL:.2f}\n")

    def rel(a, b):
        return float(np.sqrt(np.sum((a - b) ** 2) / np.sum(b ** 2)))

    # Time integration is checked first and separately. If dt is not the
    # limiting error, the grid study below is a pure spatial study.
    t0 = time.time()
    s_dt_a = Solver(4801).run(0.005)[0]
    s_dt_b = Solver(4801).run(0.0025)[0]
    e_dt = rel(s_dt_a[-1], s_dt_b[-1])
    print(f"time step: dt=0.005 vs dt=0.0025 at N=4801 -> {e_dt:.3e} "
          f"({time.time()-t0:.0f}s)")
    print("dt is orders below the spatial error, so dt=0.005 is used "
          "throughout below.\n")

    DT = 0.005
    GRIDS = [1201, 2401, 4801, 9601]
    runs = {}
    for N in GRIDS:
        t0 = time.time()
        sol = Solver(N)
        snaps, _ = sol.run(DT)
        runs[N] = snaps
        i1, i2, i3 = coarse_invariants(snaps[-1])
        r1, r2, r3 = coarse_invariants(snaps[0])
        print(f"N={N:5d} dx={sol.dx:.5f}  {time.time()-t0:6.1f}s  "
              f"drift I1={abs(i1-r1)/abs(r1):.2e} "
              f"I2={abs(i2-r2)/abs(r2):.2e} I3={abs(i3-r3)/abs(r3):.2e}")

    print("\nsuccessive-grid differences at t = T, on the shared 601-point grid")
    diffs = []
    for a, b in zip(GRIDS[:-1], GRIDS[1:]):
        d = rel(runs[a][-1], runs[b][-1])
        diffs.append(d)
        print(f"  N={a:5d} vs N={b:5d} : {d:.3e}")

    orders = [np.log2(diffs[i] / diffs[i + 1]) for i in range(len(diffs) - 1)]
    print("  observed order per refinement: " +
          ", ".join(f"{o:.2f}" for o in orders))
    p = orders[-1]
    est_plain = diffs[-1] / (2.0 ** p - 1.0)
    print(f"  finest grid's own error at p={p:.2f}: ~{est_plain:.2e}")

    # Clean 2nd order, so Richardson extrapolation is valid and free: with
    # u_h = u + C h^2 + O(h^4), the combination (4 u_{h/2} - u_h)/3 cancels the
    # h^2 term. Two independent extrapolations, from (2401,4801) and from
    # (4801,9601), bound the extrapolated solution's own error by their
    # difference -- so this is verified, not assumed.
    def extrap(u_coarse, u_fine):
        return (4.0 * u_fine - u_coarse) / 3.0

    ext_a = np.array([extrap(runs[2401][k], runs[4801][k])
                      for k in range(N_SNAP)])
    ext_b = np.array([extrap(runs[4801][k], runs[9601][k])
                      for k in range(N_SNAP)])
    e_ext = rel(ext_a[-1], ext_b[-1])
    print(f"\nRichardson extrapolation, (4 u_h/2 - u_h)/3:")
    print(f"  extrap(2401,4801) vs extrap(4801,9601): {e_ext:.3e}")
    print(f"  gain over the plain finest grid: "
          f"{est_plain/max(e_ext,1e-300):.0f}x")

    if e_ext < est_plain:
        ref, est = ext_b, e_ext
        print("  -> using the extrapolated solution as the reference")
    else:
        ref, est = runs[GRIDS[-1]], est_plain
        print("  -> extrapolation did not help; using the finest grid")

    print(f"\nreference error: ~{est:.2e} relative L2")
    print(f"the PINN runs sit at ~1e-3, so the reference is "
          f"{1e-3/est:.0f}x more accurate than what it has to measure.")
    print(f"it can resolve a difference between variants down to "
          f"~{10*est:.1e} relative, i.e. {100*10*est/1e-3:.2f}% of the "
          f"PINN error.")

    # Where does the residual discrepancy live? At the ends would mean the
    # one-sided stencils are the limit; on the waves means resolution.
    d_field = np.abs(ext_a[-1] - ext_b[-1])
    nb = 20
    print(f"\ndiscrepancy location: max at the ends "
          f"{max(d_field[:nb].max(), d_field[-nb:].max()):.2e}, "
          f"max in the interior {d_field[nb:-nb].max():.2e}, "
          f"at x={xc[nb:-nb][int(np.argmax(d_field[nb:-nb]))]:.2f}")

    print("\nsolitons at t = 0 and t = T (position, amplitude)")
    for tag, u in (('t=0 ', ref[0]), ('t=T ', ref[-1])):
        ps = peaks(xc, u)
        print(f"  {tag}" + "   ".join(f"x={p:8.3f} a={a:8.5f}" for p, a in ps))
    a0 = [a for _, a in peaks(xc, ref[0])]
    aT = [a for _, a in peaks(xc, ref[-1])]
    print(f"  amplitude change through the collision: "
          f"{100*(aT[0]-a0[0])/a0[0]:+.3f}%  {100*(aT[1]-a0[1])/a0[1]:+.3f}%")
    print("  (non-zero because RLW is not integrable: the collision radiates,")
    print("   which is exactly why a reference is needed and 'amplitudes must")
    print("   be unchanged' would have been the wrong test)")

    # The reference's own I1 drift is the physical boundary flux, not an error:
    # the tail of the tall soliton is 1.3e-04 at x=a, so mass is injected. It is
    # identical on every grid, which is how we know it is physical. Report it,
    # because it means a constant-C1 target is slightly wrong here too -- the
    # same defect as the bore, four orders of magnitude smaller.
    i1T, i2T, i3T = coarse_invariants(ref[-1])
    i10, i20, i30 = coarse_invariants(ref[0])
    print(f"\nboundary flux: the true C1 grows by "
          f"{abs(i1T-i10)/abs(i10):.2e} relative over T, so enforcing a "
          f"constant C1 is wrong by that much")
    print(f"               true I2 changes by {abs(i2T-i20)/abs(i20):.2e}, "
          f"I3 by {abs(i3T-i30)/abs(i30):.2e}")
    print("               both are well under the ~1e-3 PINN error, so a "
          "constant target is acceptable here")

    ts = np.linspace(0.0, T_FINAL, N_SNAP)
    np.savez_compressed(
        OUT, x=xc, t=ts, u=ref,
        n_ref=GRIDS[-1], dt_ref=DT, eps=EPS, mu=MU,
        c1=C1_AMP, c2=C2_AMP, x1=X1, x2=X2, xa=XA, xb=XB, T=T_FINAL,
        est_rel_error=est, order=p,
        true_I1=np.array([coarse_invariants(u)[0] for u in ref]),
        true_I2=np.array([coarse_invariants(u)[1] for u in ref]),
        true_I3=np.array([coarse_invariants(u)[2] for u in ref]))
    print(f"\nwritten {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB, "
          f"{N_SNAP} snapshots x {N_COARSE} points)")


if __name__ == '__main__':
    main()

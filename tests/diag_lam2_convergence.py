"""Does the projection become marginally conditioned exactly at convergence?

Hypothesis. As the network approaches the exact soliton, I2[u~] -> C2, so

    lam2 = sqrt(I2[u~]/C2) - 1 -> 0,

which is the boundary between the well-posed branch (lam2 > 0) and the
ill-posed one (lam2 < 0, Prop. 4). Near convergence, optimizer noise then
decides which side the solve lands on. That would explain why the RLW1 run was
stable for 4000 steps and destabilized only after reaching a loss of 1e-8, with
the clamp count moving at the same moment.

Test. Take the exact solution, perturb it by shrinking amounts of two kinds
(rough and smooth), and record lam2, whether the clamp binds, and how far the
analytic adjoint drifts from finite differences.

Prediction if true:
  * lam2 -> 0 as the perturbation shrinks,
  * its sign is set by whether the perturbation adds or removes u_x content,
  * the clamp binds on the negative side,
  * the adjoint degrades where it binds.

Run:  py -3 tests/diag_lam2_convergence.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from ce_pinn.equations import rlw_soliton_torch
from ce_pinn.projection import (central_d1_matrix, invariant_i2,
                                helmholtz_projection, lam2_lower_bound)
from ce_pinn.operators import trapezoid_weights

torch.set_default_dtype(torch.float64)

# RLW1 configuration
EPS, MU, C_AMP, X0 = 1.0, 1.0, 0.1, 0.0
XA, XB, T = -40.0, 60.0, 20.0
N = 401

dx = (XB - XA) / (N - 1)
x = torch.linspace(XA, XB, N)
q = trapezoid_weights(N, dx)
D1c = central_d1_matrix(N, dx)

exact = lambda t: rlw_soliton_torch(x, torch.full_like(x, t), eps=EPS, mu=MU,
                                    c=C_AMP, x0=X0)

u0 = exact(0.0)
C1 = torch.dot(q, u0)
C2 = invariant_i2(u0, q, D1c, MU)
lam2_min = lam2_lower_bound(MU, dx)

print(f"N={N}  dx={dx:.4f}   C1={float(C1):.8f}  C2={float(C2):.8f}")
print(f"lam2 lower bound (clamp) = {lam2_min:.6e}\n")

# ---------------------------------------------------------------------------
print("1. Does I2 of the exact solution stay at C2 as it propagates?")
# ---------------------------------------------------------------------------
print(f"{'t':>6} {'I2[exact]':>14} {'I2/C2':>12} {'lam2 implied':>14}")
for t in (0.0, 5.0, 10.0, 15.0, 20.0):
    ue = exact(t)
    i2 = float(invariant_i2(ue, q, D1c, MU))
    r = i2 / float(C2)
    print(f"{t:6.1f} {i2:14.10f} {r:12.9f} {np.sqrt(r)-1:14.3e}")

# ---------------------------------------------------------------------------
print("\n2. lam2 as a perturbation of the converged state shrinks")
# ---------------------------------------------------------------------------
torch.manual_seed(0)
t_mid = T / 2.0
ue = exact(t_mid)
ga, gb = float(ue[0]), float(ue[-1])

# rough perturbation adds u_x content, smooth removes it relative to the exact
# profile; both shrink to zero
rough = torch.randn(N) * torch.exp(-((x - 11.0) / 15.0) ** 2)
smooth = torch.exp(-((x - 11.0) / 15.0) ** 2)

print(f"{'kind':>7} {'delta':>9} {'I2/C2':>13} {'lam2':>12} {'clamped':>8} "
      f"{'iters':>6} {'resid':>10}")
rows = []
for kind, pert in (('rough', rough), ('smooth', smooth)):
    for delta in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        ut = ue + delta * pert
        ut[0], ut[-1] = ga, gb
        r = float(invariant_i2(ut, q, D1c, MU) / C2)
        _, st = helmholtz_projection(ut, C1, C2, MU, dx, q, D1c, ga=ga, gb=gb)
        rows.append((kind, delta, r, st.lam[1], st.clamped, st.residual))
        print(f"{kind:>7} {delta:9.0e} {r:13.9f} {st.lam[1]:12.3e} "
              f"{str(st.clamped):>8} {st.newton_iters:6d} {st.residual:10.1e}")

# ---------------------------------------------------------------------------
print("\n3. Adjoint accuracy against finite differences, as lam2 -> 0")
# ---------------------------------------------------------------------------
Ns = 81
dxs = (XB - XA) / (Ns - 1)
xs = torch.linspace(XA, XB, Ns)
qs = trapezoid_weights(Ns, dxs)
D1s = central_d1_matrix(Ns, dxs)
es = lambda t: rlw_soliton_torch(xs, torch.full_like(xs, t), eps=EPS, mu=MU,
                                 c=C_AMP, x0=X0)
u0s = es(0.0)
C1s = torch.dot(qs, u0s)
C2s = invariant_i2(u0s, qs, D1s, MU)
lam2_min_s = lam2_lower_bound(MU, dxs)
ues = es(t_mid)
gas, gbs = float(ues[0]), float(ues[-1])
ps = torch.randn(Ns) * torch.exp(-((xs - 11.0) / 15.0) ** 2)
w = torch.randn(Ns)

def loss_of(uin):
    us, _ = helmholtz_projection(uin, C1s, C2s, MU, dxs, qs, D1s,
                                 ga=gas, gb=gbs)
    return torch.dot(w, us)

print(f"(coarse grid N={Ns} for finite differencing; clamp at {lam2_min_s:.2e})")
print(f"{'delta':>9} {'lam2':>12} {'clamped':>8} {'grad rel err':>14}")
for delta in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
    base = ues + delta * ps
    base[0], base[-1] = gas, gbs
    _, st = helmholtz_projection(base, C1s, C2s, MU, dxs, qs, D1s,
                                 ga=gas, gb=gbs)
    ua = base.clone().requires_grad_(True)
    loss_of(ua).backward()
    ga_an = ua.grad.clone()

    h = 1e-7
    gfd = torch.zeros(Ns)
    for i in range(1, Ns - 1):
        up = base.clone(); up[i] += h
        um = base.clone(); um[i] -= h
        gfd[i] = (loss_of(up) - loss_of(um)).item() / (2 * h)
    rel = float((ga_an - gfd).abs().max() / (gfd.abs().max() + 1e-300))
    print(f"{delta:9.0e} {st.lam[1]:12.3e} {str(st.clamped):>8} {rel:14.3e}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
neg = [r for r in rows if r[3] < 0]
clamped = [r for r in rows if r[4]]
print(f"perturbations giving lam2 < 0 : {len(neg)} of {len(rows)}")
print(f"perturbations hitting the clamp: {len(clamped)} of {len(rows)}")
small = [abs(r[3]) for r in rows if r[1] <= 1e-4]
print(f"|lam2| for perturbations <= 1e-4: max {max(small):.2e}")
print("\nlam2 -> 0 as the state converges: "
      f"{'CONFIRMED' if max(small) < 1e-3 else 'NOT CONFIRMED'}")

"""Correctness tests for the projection layers.

Run:  py -3 tests/test_projection.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from ce_pinn.projection import (
    central_d1_matrix, invariant_i2, helmholtz_projection,
    polynomial_projection, project_tangent,
)
from ce_pinn.operators import trapezoid_weights

torch.set_default_dtype(torch.float64)

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ''))


# ---------------------------------------------------------------------------
# Fixture: RLW single soliton, the paper's Example 1 parameters
# ---------------------------------------------------------------------------
EPS, MU, C_AMP, X0 = 1.0, 1.0, 0.1, 0.0
XA, XB, N = -40.0, 60.0, 401
dx = (XB - XA) / (N - 1)
x = torch.linspace(XA, XB, N)
q = trapezoid_weights(N, dx)
D1c = central_d1_matrix(N, dx)


def rlw_soliton(x, t):
    speed = 1.0 + EPS * C_AMP
    k = 0.5 * np.sqrt(EPS * C_AMP / (MU * speed))
    return 3.0 * C_AMP / torch.cosh(k * (x - speed * t - X0)) ** 2


u_exact = rlw_soliton(x, 0.0)
C1 = torch.dot(q, u_exact)
C2 = invariant_i2(u_exact, q, D1c, MU)

print(f"\nGrid N={N}, dx={dx:.4f}")
print(f"C1 = {C1:.10f}   (paper: 3.979927)")
print(f"C2 = {C2:.10f}   (paper: 0.8104625)")

# fraction of I2 carried by the derivative term
ux = torch.matmul(u_exact, D1c.T)
frac = float(MU * torch.dot(q, ux * ux) / C2)
print(f"derivative term is {100*frac:.2f}% of I2\n")


# ---------------------------------------------------------------------------
print("1. Helmholtz projection enforces both invariants")
# ---------------------------------------------------------------------------
torch.manual_seed(0)
GA0, GB0 = float(u_exact[0]), float(u_exact[-1])   # true BC values, not zero
print(f"boundary values: u(a)={GA0:.3e}, u(b)={GB0:.3e}\n")

u_pert = u_exact + 0.02 * torch.randn(N) * torch.exp(-((x - 10) / 20) ** 2)
u_pert[0] = GA0
u_pert[-1] = GB0

u_star, st = helmholtz_projection(u_pert, C1, C2, MU, dx, q, D1c,
                                  ga=GA0, gb=GB0)
e1 = float((torch.dot(q, u_star) - C1).abs())
e2 = float((invariant_i2(u_star, q, D1c, MU) - C2).abs())
check("C1 enforced", e1 < 1e-10, f"|err|={e1:.2e}")
check("C2 enforced", e2 < 1e-10, f"|err|={e2:.2e}")
check("Newton converged", st.residual < 1e-12,
      f"iters={st.newton_iters}, resid={st.residual:.2e}")


# ---------------------------------------------------------------------------
print("\n2. Projection is the identity on an already-conserving field")
# ---------------------------------------------------------------------------
u_id, _ = helmholtz_projection(u_exact, C1, C2, MU, dx, q, D1c, ga=GA0, gb=GB0)
d = float((u_id - u_exact).abs().max())
check("u* == u~ when constraints already hold", d < 1e-9, f"max|diff|={d:.2e}")


# ---------------------------------------------------------------------------
print("\n3. Boundary conditions preserved (Helmholtz) / broken (polynomial)")
# ---------------------------------------------------------------------------
GA, GB = 0.1, 0.0            # non-homogeneous, as in the undular bore
u_bore = u_pert + GA * (1.0 - (x - XA) / (XB - XA))
C1b = torch.dot(q, u_bore)
C2b = invariant_i2(u_bore, q, D1c, MU) * 1.05      # ask for 5% more

uh, _ = helmholtz_projection(u_bore, C1b, C2b, MU, dx, q, D1c, ga=GA, gb=GB)
bc_h = max(abs(float(uh[0]) - GA), abs(float(uh[-1]) - GB))
check("Helmholtz preserves Dirichlet data", bc_h < 1e-12, f"bc err={bc_h:.2e}")

up, _ = polynomial_projection(u_bore, C1b, C2b, q)
bc_p = max(abs(float(up[0]) - GA), abs(float(up[-1]) - GB))
check("polynomial breaks Dirichlet data (expected)", bc_p > 1e-6,
      f"bc err={bc_p:.2e}")


# ---------------------------------------------------------------------------
print("\n4. Polynomial projection cannot hold the derivative-dependent I2")
# ---------------------------------------------------------------------------
# Give the polynomial projection the polynomial invariants it was designed for.
C2_poly = torch.dot(q, u_exact * u_exact)
up2, _ = polynomial_projection(u_pert, C1, C2_poly, q)
drift_poly = float((invariant_i2(up2, q, D1c, MU) - C2).abs() / C2)
us2, _ = helmholtz_projection(u_pert, C1, C2, MU, dx, q, D1c, ga=GA0, gb=GB0)
drift_helm = float((invariant_i2(us2, q, D1c, MU) - C2).abs() / C2)
check("polynomial leaves I2 drift", drift_poly > 1e-6, f"drift={drift_poly:.3e}")
check("Helmholtz drives I2 drift to zero", drift_helm < 1e-12,
      f"drift={drift_helm:.3e}")
print(f"       ratio = {drift_poly/max(drift_helm,1e-300):.2e}x")


# ---------------------------------------------------------------------------
print("\n5. Backward pass matches finite differences (the IFT gradient)")
# ---------------------------------------------------------------------------
torch.manual_seed(1)
Ns = 81                                  # small grid so FD checking is cheap
dxs = (XB - XA) / (Ns - 1)
xs = torch.linspace(XA, XB, Ns)
qs = trapezoid_weights(Ns, dxs)
D1s = central_d1_matrix(Ns, dxs)
us_ex = rlw_soliton(xs, 0.0)
C1s = torch.dot(qs, us_ex)
C2s = invariant_i2(us_ex, qs, D1s, MU)

GAs, GBs = float(us_ex[0]), float(us_ex[-1])
u0 = (us_ex + 0.05 * torch.randn(Ns) * torch.exp(-((xs - 10) / 20) ** 2))
u0[0] = GAs
u0[-1] = GBs
w = torch.randn(Ns)                      # arbitrary loss weighting


def loss_of(uin):
    us, _ = helmholtz_projection(uin, C1s, C2s, MU, dxs, qs, D1s, ga=GAs, gb=GBs)
    return torch.dot(w, us)


ua = u0.clone().requires_grad_(True)
loss_of(ua).backward()
g_analytic = ua.grad.clone()

g_fd = torch.zeros(Ns)
h = 1e-6
for i in range(1, Ns - 1):               # boundary rows have zero gradient
    up_ = u0.clone(); up_[i] += h
    um_ = u0.clone(); um_[i] -= h
    g_fd[i] = (loss_of(up_) - loss_of(um_)).item() / (2 * h)

num = float((g_analytic - g_fd).abs().max())
den = float(g_fd.abs().max())
check("analytic gradient == finite differences", num / den < 1e-5,
      f"rel err={num/den:.3e}")

# and show what the naive (frozen-multiplier) gradient would have given
Ev = torch.zeros(Ns); Ev[1:-1] = w[1:-1]
naive = torch.linalg.solve(st.A[:Ns, :Ns] * 0 + torch.eye(Ns), Ev)  # placeholder
us_, st_ = helmholtz_projection(u0, C1s, C2s, MU, dxs, qs, D1s, ga=GAs, gb=GBs)
Ew = w.clone(); Ew[0] = 0; Ew[-1] = 0
naive = torch.linalg.solve(st_.A.T, Ew) / st_.coeff
naive[0] = 0; naive[-1] = 0
rel_naive = float((naive - g_fd).abs().max() / den)
print(f"       (frozen-multiplier gradient would give rel err = {rel_naive:.3e})")


# ---------------------------------------------------------------------------
print("\n6. project_tangent matches d/dt by finite differences")
# ---------------------------------------------------------------------------
def u_of_t(tval):
    return rlw_soliton(x, tval) * (1.0 + 0.1 * tval)      # some t-dependence


t0, ht = 1.0, 1e-6
u_t0 = u_of_t(t0)
# analytic tangent of u~
ut_tilde = (u_of_t(t0 + ht) - u_of_t(t0 - ht)) / (2 * ht)

GAt, GBt = float(u_t0[0]), float(u_t0[-1])
us_t0, st_t = helmholtz_projection(u_t0, C1, C2, MU, dx, q, D1c, ga=GAt, gb=GBt)
tan_analytic = project_tangent(ut_tilde, st_t)

usp, _ = helmholtz_projection(u_of_t(t0 + ht), C1, C2, MU, dx, q, D1c, ga=GAt, gb=GBt)
usm, _ = helmholtz_projection(u_of_t(t0 - ht), C1, C2, MU, dx, q, D1c, ga=GAt, gb=GBt)
tan_fd = (usp - usm) / (2 * ht)

num = float((tan_analytic - tan_fd).abs().max())
den = float(tan_fd.abs().max())
check("tangent == finite differences", num / den < 1e-5,
      f"rel err={num/den:.3e}")


# ---------------------------------------------------------------------------
print("\n7. polynomial tangent matches finite differences")
# ---------------------------------------------------------------------------
upp, stp = polynomial_projection(u_t0, C1, C2_poly, q)
tanp = project_tangent(ut_tilde, stp)
a1, _ = polynomial_projection(u_of_t(t0 + ht), C1, C2_poly, q)
a2, _ = polynomial_projection(u_of_t(t0 - ht), C1, C2_poly, q)
tanp_fd = (a1 - a2) / (2 * ht)
num = float((tanp - tanp_fd).abs().max())
den = float(tanp_fd.abs().max())
check("polynomial tangent == FD", num / den < 1e-5, f"rel err={num/den:.3e}")


# ---------------------------------------------------------------------------
print(f"\n{'='*58}")
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print(f"   FAILED: {f}")
    sys.exit(1)
print("all good")

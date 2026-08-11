"""Verify the RLW boundary-flux coefficients for the undular bore.

The claim to be checked, for a steady left state u0 and a zero right state:

    dC1/dt = u0 + (eps/2) u0^2
    dC2/dt = u0^2 + (2 eps/3) u0^3

These were derived by hand from

    d/dt int u dx        = -[ u + (eps/2)u^2 - mu u_xt ]_a^b
    d/dt int(u^2+mu u_x^2) dx = -[ u^2 + (2eps/3)u^3 - 2 mu u u_xt ]_a^b

by dropping the u_xt terms on the assumption that the boundary state is
steady. Both the flux identities and that assumption are tested here against a
direct integration of the equation.

Method: write RLW as (1 - mu d_xx) u_t = -d_x f(u), f = u + (eps/2)u^2, and
solve for u_t at each stage with u_t = 0 imposed at both ends (the boundary
values are held fixed in time). Integrate with RK4 and difference the measured
invariants.

Run:  py -3 tests/verify_bore_flux.py
"""
import numpy as np

try:
    from scipy.linalg import lu_factor, lu_solve
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# undular-bore parameters
EPS, MU = 1.5, 1.0 / 6.0
U0, XC, D = 0.1, 0.0, 5.0
XA, XB = -36.0, 100.0        # right end far enough that it stays at 0
N = 1201
T_END, DT = 4.0, 0.005

x = np.linspace(XA, XB, N)
dx = x[1] - x[0]

# (1 - mu d_xx) with u_t = 0 at both ends
D2 = np.zeros((N, N))
i = np.arange(1, N - 1)
D2[i, i - 1] = 1.0 / dx**2
D2[i, i] = -2.0 / dx**2
D2[i, i + 1] = 1.0 / dx**2
A = np.eye(N) - MU * D2
A[0, :] = 0.0; A[0, 0] = 1.0
A[-1, :] = 0.0; A[-1, -1] = 1.0
if HAVE_SCIPY:
    lu = lu_factor(A)
    solve = lambda b: lu_solve(lu, b)
else:
    Ainv = np.linalg.inv(A)
    solve = lambda b: Ainv @ b


def d1(u):
    """4th-order interior, one-sided ends."""
    g = np.empty_like(u)
    g[2:-2] = (u[:-4] - 8 * u[1:-3] + 8 * u[3:-1] - u[4:]) / (12 * dx)
    g[:2] = (-3 * u[:2] + 4 * u[1:3] - u[2:4]) / (2 * dx)
    g[-2:] = (3 * u[-2:] - 4 * u[-3:-1] + u[-4:-2]) / (2 * dx)
    return g


def rhs(u):
    f = u + 0.5 * EPS * u * u
    b = -d1(f)
    b[0] = 0.0            # boundary values held fixed => u_t = 0 there
    b[-1] = 0.0
    return solve(b)


def rk4(u, dt):
    a = rhs(u); b = rhs(u + 0.5 * dt * a)
    c = rhs(u + 0.5 * dt * b); e = rhs(u + dt * c)
    return u + dt * (a + 2 * b + 2 * c + e) / 6.0


def C1(u):
    return np.trapezoid(u, dx=dx)


def C2(u):
    ux = d1(u)
    return np.trapezoid(u * u + MU * ux * ux, dx=dx)


u = 0.5 * U0 * (1.0 - np.tanh((x - XC) / D))
print(f"eps={EPS}  mu={MU:.6f}  u0={U0}  domain [{XA},{XB}]  N={N}  dx={dx:.4f}")
print(f"scipy: {HAVE_SCIPY}")
print(f"left state u(a)={u[0]:.8f}   right state u(b)={u[-1]:.3e}\n")

pred_dC1 = U0 + 0.5 * EPS * U0**2
pred_dC2 = U0**2 + (2 * EPS / 3) * U0**3
print(f"hand-derived  dC1/dt = u0 + (eps/2)u0^2      = {pred_dC1:.10f}")
print(f"hand-derived  dC2/dt = u0^2 + (2eps/3)u0^3   = {pred_dC2:.10f}\n")

c1_0, c2_0 = C1(u), C2(u)
print(f"C1(0) = {c1_0:.8f}     C2(0) = {c2_0:.8f}\n")
print(f"{'t':>6} {'C1':>12} {'dC1/dt':>12} {'err':>10} "
      f"{'C2':>12} {'dC2/dt':>12} {'err':>10}")

nsteps = int(round(T_END / DT))
report = max(1, nsteps // 8)
t = 0.0
for k in range(1, nsteps + 1):
    u = rk4(u, DT)
    t = k * DT
    if k % report == 0:
        m1 = (C1(u) - c1_0) / t
        m2 = (C2(u) - c2_0) / t
        print(f"{t:6.2f} {C1(u):12.6f} {m1:12.8f} {abs(m1-pred_dC1)/pred_dC1:10.2e} "
              f"{C2(u):12.6f} {m2:12.8f} {abs(m2-pred_dC2)/pred_dC2:10.2e}")

# is dropping the u_xt boundary terms justified?
ut = rhs(u)
ut_x = d1(ut)
print(f"\nboundary u_xt:  left {ut_x[0]:.3e}   right {ut_x[-1]:.3e}")
print(f"  mu*u_xt(a)                = {MU*ut_x[0]:.3e}  "
      f"(vs u0 + (eps/2)u0^2 = {pred_dC1:.4f})")
print(f"  2*mu*u(a)*u_xt(a)         = {2*MU*u[0]*ut_x[0]:.3e}  "
      f"(vs u0^2 + (2eps/3)u0^3 = {pred_dC2:.4f})")

m1 = (C1(u) - c1_0) / t
m2 = (C2(u) - c2_0) / t
e1, e2 = abs(m1 - pred_dC1) / pred_dC1, abs(m2 - pred_dC2) / pred_dC2
print(f"\nfinal relative error:  dC1/dt {e1:.3e}   dC2/dt {e2:.3e}")
ok = e1 < 2e-3 and e2 < 2e-3
print("VERIFIED" if ok else "MISMATCH - do not use these coefficients")

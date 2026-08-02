"""Gate for the Phase 0 fixes. Run before tagging exp-v4.

Three things are checked, none of which any existing test covers.

  1. CLAMP ADJOINT (fix 0.3). When the lam2 clamp binds, lam2 is held at its
     bound rather than determined by F2 = 0, so dlam2/du~ = 0 and only lam1
     responds. The projector must drop to rank 1 on those steps. No existing
     test reaches a clamped state at all -- test_projection.py case 8 asserts
     the fixture is on the *unclamped* branch -- so the fix is unverified until
     here. Checked against finite differences, with the previous rank-2
     behaviour computed alongside to confirm the test can tell them apart.

  2. IDEMPOTENCE (fixes 0.1 + 0.2). MATH.md Theorem 7: if the exact solution
     satisfies the imposed boundary data and targets, projecting it must return
     it unchanged. With data frozen at t=0 this failed at 5.6e-06 and produced
     an additive error floor. With ga_fn/gb_fn and C1_fn/C2_fn supplied it must
     reach machine precision.

  3. NO WARNINGS (fix 0.4). float() on a tensor carrying grad history emits a
     UserWarning. Training is run with warnings promoted to errors.

Run:  py -3 tests/test_phase0.py
"""
import os
import sys
import io
import contextlib
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from ce_pinn.equations import rlw_soliton, rlw_soliton_torch
from ce_pinn.network import MLP, HardBC1D
from ce_pinn.train import GridTrainer
from ce_pinn import projection as proj

torch.set_default_dtype(torch.float64)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'tests', '_phase0')
os.makedirs(OUT, exist_ok=True)

EPS = MU = 1.0
C_AMP, X0 = 0.1, 0.0
XA, XB, T = -40.0, 60.0, 20.0
N = 201

passed, failed = [], []


def check(ok, name, detail=''):
    (passed if ok else failed).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ''))


def grid():
    dx = (XB - XA) / (N - 1)
    x = torch.linspace(XA, XB, N, dtype=torch.float64)
    q = torch.full((N,), dx, dtype=torch.float64)
    q[0] = q[-1] = dx / 2
    D1c = proj.central_d1_matrix(N, dx)
    return x, dx, q, D1c


# ---------------------------------------------------------------------------
print('=' * 62)
print('1. Clamped adjoint is rank 1 (fix 0.3)')
print('=' * 62)

x, dx, q, D1c = grid()
u_t = torch.tensor(rlw_soliton(x.numpy(), 0.0, eps=EPS, mu=MU, c=C_AMP, x0=X0))
C1 = float(q @ u_t)
I2_0 = float(q @ (u_t * u_t + MU * (D1c @ u_t) ** 2))

# Ask for a C2 far above what the field carries. The solver must push lam2
# negative to inflate the field, and lam2 runs into its well-posedness bound.
C2_big = 40.0 * I2_0
_, st = proj.helmholtz_projection(u_t.clone(), C1, C2_big, MU, dx, q, D1c,
                                  ga=float(u_t[0]), gb=float(u_t[-1]))
check(st.clamped, 'constructed a genuinely clamped state',
      f'lam2={st.lam[1]:.3e} clamped={st.clamped}')

if st.clamped:
    check(st.P_eff.shape[1] == 1 and st.R_eff.shape[1] == 1,
          'projector dropped to rank 1 when clamped',
          f'P_eff {tuple(st.P_eff.shape)}, R_eff {tuple(st.R_eff.shape)}')

    # directional derivative of the projection, analytic vs finite difference
    torch.manual_seed(0)
    d = torch.randn(N, dtype=torch.float64)
    d[0] = d[-1] = 0.0
    d = d / d.norm()

    ut = u_t.clone().requires_grad_(True)
    us, _ = proj.helmholtz_projection(ut, C1, C2_big, MU, dx, q, D1c,
                                      ga=float(u_t[0]), gb=float(u_t[-1]))
    w = torch.randn(N, dtype=torch.float64)
    (us * w).sum().backward()
    analytic = float(ut.grad @ d)

    h = 1e-7
    fp = proj.helmholtz_projection(u_t + h * d, C1, C2_big, MU, dx, q, D1c,
                                   ga=float(u_t[0]), gb=float(u_t[-1]))[0]
    fm = proj.helmholtz_projection(u_t - h * d, C1, C2_big, MU, dx, q, D1c,
                                   ga=float(u_t[0]), gb=float(u_t[-1]))[0]
    fd = float(((fp - fm) / (2 * h) * w).sum())
    rel = abs(analytic - fd) / max(abs(fd), 1e-30)
    check(rel < 1e-5, 'clamped adjoint matches finite differences',
          f'rel err={rel:.3e}')

    # what the old rank-2 code would have produced, for contrast
    s = st.P.T @ w
    z = torch.linalg.solve(st.J.T, s)
    g_old = w - st.R @ z
    g_old = torch.linalg.solve(st.A.T, g_old)
    g_old = (g_old / st.coeff).clone()
    g_old[0] = g_old[-1] = 0.0
    rel_old = abs(float(g_old @ d) - fd) / max(abs(fd), 1e-30)
    check(rel_old > 10 * rel, 'rank-2 projector is detectably wrong here',
          f'rank-2 rel err={rel_old:.3e} vs rank-1 {rel:.3e}')

# ---------------------------------------------------------------------------
print()
print('=' * 62)
print('2. Idempotence with consistent data (fixes 0.1 + 0.2)')
print('=' * 62)


def ic_np(z):
    return rlw_soliton(z, 0.0, eps=EPS, mu=MU, c=C_AMP, x0=X0)


def ic_t(z):
    return rlw_soliton_torch(z, torch.zeros_like(z), eps=EPS, mu=MU,
                             c=C_AMP, x0=X0)


def exact(t):
    return rlw_soliton(x.numpy(), t, eps=EPS, mu=MU, c=C_AMP, x0=X0)


A = torch.diag(q) + MU * (D1c.T @ torch.diag(q) @ D1c)


def C1_fn(t):
    return float(q @ torch.tensor(exact(t)))


def C2_fn(t):
    v = torch.tensor(exact(t))
    return float(v @ A @ v)


base = dict(equation='rlw', eps=EPS, mu=MU, xa=XA, xb=XB, T=T, n_grid=N,
            n_t=4, projection='helmholtz', soft_ibc=False, lam_ibc=10.0,
            ic_fn=ic_np, diag_every=10 ** 9, C2_poly_ref=1.0)

for label, extra in [
    ('frozen data (old behaviour)', {}),
    ('consistent data (fixed)',
     dict(ga_fn=lambda t: float(exact(t)[0]),
          gb_fn=lambda t: float(exact(t)[-1]),
          C1_fn=C1_fn, C2_fn=C2_fn)),
]:
    torch.manual_seed(0)
    mlp = MLP(d_in=2, d_out=1, hidden_layers=2, hidden_width=16)
    model = HardBC1D(mlp, ic_t, XA, XB, T=T)
    with contextlib.redirect_stdout(io.StringIO()):
        tr = GridTrainer(model, dict(base, **extra), device='cpu', out_dir=OUT)
    ue = torch.tensor(exact(T))
    up = proj.helmholtz_projection(
        ue.clone(), tr._C1(T), tr._C2(T), MU, dx, q, D1c,
        ga=tr._ga(T), gb=tr._gb(T))[0]
    dev = float(torch.sqrt((up - ue) @ torch.diag(q) @ (up - ue)))
    print(f'  {label:>30}: ||P(u_exact) - u_exact|| = {dev:.3e}')
    if 'fixed' in label:
        check(dev < 1e-13, 'Theorem 7 holds with consistent data',
              f'{dev:.3e}')

# ---------------------------------------------------------------------------
print()
print('=' * 62)
print('3. Training emits no warnings (fix 0.4)')
print('=' * 62)

torch.manual_seed(0)
mlp = MLP(d_in=2, d_out=1, hidden_layers=2, hidden_width=16)
model = HardBC1D(mlp, ic_t, XA, XB, T=T)
with contextlib.redirect_stdout(io.StringIO()):
    tr = GridTrainer(model, dict(base, projection=None), device='cpu',
                     out_dir=OUT)
try:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with contextlib.redirect_stdout(io.StringIO()):
            tr.train_adam(3, lr=1e-3, log_every=10 ** 9)
            tr.train_lbfgs(2, log_every=10 ** 9)
    check(True, 'Adam and L-BFGS run with warnings promoted to errors')
except Warning as w:
    check(False, 'Adam and L-BFGS run with warnings promoted to errors',
          f'{type(w).__name__}: {w}')

print()
print('=' * 62)
print(f'{len(passed)} passed, {len(failed)} failed')
if failed:
    for f in failed:
        print(f'  FAILED: {f}')
print('=' * 62)
sys.exit(1 if failed else 0)

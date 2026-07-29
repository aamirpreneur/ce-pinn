"""Exact solutions, initial conditions, and conservation law evaluators.

All functions operate on numpy arrays (float64) for reference/validation
and on torch tensors (float64) for training.
"""
import numpy as np
import torch

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# ---------------------------------------------------------------------------
# KdV: u_t + 6uu_x + u_xxx = 0
# ---------------------------------------------------------------------------

def kdv_one_soliton(x, t, k=1.0):
    """u = (k^2/2) sech^2(k/2 (x - k^2 t))."""
    arg = (k / 2.0) * (x - k**2 * t)
    return (k**2 / 2.0) / np.cosh(arg)**2


def kdv_one_soliton_torch(x, t, k=1.0):
    arg = (k / 2.0) * (x - k**2 * t)
    return (k**2 / 2.0) / torch.cosh(arg)**2


def kdv_two_soliton(x, t, k1=1.0, k2=2.0):
    """Hirota bilinear two-soliton, u = 2 d^2/dx^2 ln f.

    Differentiated analytically rather than by finite differences, so the
    result is exact and the function works for an input of any length. The
    previous finite-difference version needed at least two grid points and
    failed when the hard-BC layer evaluated the initial condition at a single
    boundary node.

    All four exponentials are factored by their common maximum before
    exponentiating, which keeps the intermediate values in range.
    """
    x = np.asarray(x, dtype=float)
    th1 = k1 * x - k1 ** 3 * t
    th2 = k2 * x - k2 ** 3 * t
    a12 = ((k1 - k2) / (k1 + k2)) ** 2

    # exponents of the four terms of f, with their x-derivative coefficients
    e = np.stack([np.zeros_like(x), th1, th2, th1 + th2 + np.log(a12)])
    c1 = np.array([0.0, k1, k2, k1 + k2]).reshape(-1, *([1] * x.ndim))

    m = e.max(axis=0)
    w = np.exp(e - m)                 # weights relative to the largest term

    f = w.sum(axis=0)
    fx = (c1 * w).sum(axis=0)
    fxx = (c1 ** 2 * w).sum(axis=0)

    # the common factor exp(m) cancels in (f_xx f - f_x^2)/f^2
    return 2.0 * (fxx * f - fx ** 2) / f ** 2


def kdv_two_soliton_torch(x, t, k1=1.0, k2=2.0):
    """Torch version of the analytic two-soliton solution."""
    th1 = k1 * x - k1 ** 3 * t
    th2 = k2 * x - k2 ** 3 * t
    a12 = ((k1 - k2) / (k1 + k2)) ** 2

    e = torch.stack([torch.zeros_like(x), th1, th2,
                     th1 + th2 + float(np.log(a12))])
    c1 = torch.tensor([0.0, k1, k2, k1 + k2], dtype=x.dtype,
                      device=x.device).reshape(-1, *([1] * x.dim()))

    m = e.max(dim=0).values
    w = torch.exp(e - m)

    f = w.sum(dim=0)
    fx = (c1 * w).sum(dim=0)
    fxx = (c1 ** 2 * w).sum(dim=0)
    return 2.0 * (fxx * f - fx ** 2) / f ** 2


def kdv_exact_invariants(k=1.0):
    """Analytical invariants for one-soliton on infinite domain."""
    return 2.0 * k, 2.0 * k**3 / 3.0, k**5 / 5.0


# ---------------------------------------------------------------------------
# RLW: u_t + u_x + eps*u*u_x - mu*u_xxt = 0
# ---------------------------------------------------------------------------

def rlw_soliton(x, t, eps=1.0, mu=1.0, c=0.1, x0=0.0):
    """u = 3c sech^2(k(x - (1+eps*c)t - x0)), k = 0.5*sqrt(eps*c/(mu*(1+eps*c)))."""
    speed = 1.0 + eps * c
    k = 0.5 * np.sqrt(eps * c / (mu * speed))
    arg = k * (x - speed * t - x0)
    return 3.0 * c / np.cosh(arg)**2


def rlw_soliton_torch(x, t, eps=1.0, mu=1.0, c=0.1, x0=0.0):
    speed = 1.0 + eps * c
    k = 0.5 * np.sqrt(eps * c / (mu * speed))
    arg = k * (x - speed * t - x0)
    return 3.0 * c / torch.cosh(arg)**2


def rlw_two_soliton_ic(x, eps=1.0, mu=1.0, c1=5.33/3, c2=1.69/3, x1=15.0, x2=35.0):
    """Superposition IC for two well-separated RLW solitons."""
    u1 = rlw_soliton(x, 0.0, eps, mu, c1, x1)
    u2 = rlw_soliton(x, 0.0, eps, mu, c2, x2)
    return u1 + u2


def rlw_undular_bore_ic(x, u0=0.1, xc=0.0, d=5.0):
    """u(x,0) = 0.5*u0*(1 - tanh((x - xc)/d))."""
    return 0.5 * u0 * (1.0 - np.tanh((x - xc) / d))


def rlw_invariants_numpy(u, ux, dx, eps=1.0, mu=1.0):
    """Compute I1, I2, I3 for RLW from discrete arrays.

    I3 = int(eps*u^3 + 3u^2) dx. The eps factor matters whenever eps != 1
    (the undular bore uses eps = 1.5); dropping it makes I3 drift.
    """
    I1 = _trapz(u, dx=dx)
    I2 = _trapz(u**2 + mu * ux**2, dx=dx)
    I3 = _trapz(eps * u**3 + 3.0 * u**2, dx=dx)
    return I1, I2, I3


# ---------------------------------------------------------------------------
# KP: (u_t + 6uu_x + u_xxx)_x + sigma*u_yy = 0,  sigma = -3
# ---------------------------------------------------------------------------

def kp_lump(x, y, t, sigma=-3.0):
    """Lump solution for KP-I: u = 4(-xi^2 + y^2 + b)/(xi^2 + y^2 + b)^2,
    with b = -3/sigma, speed c = -sigma, xi = x - c*t."""
    b = -3.0 / sigma
    c = -sigma
    xi = x - c * t
    denom = xi**2 + y**2 + b
    return 4.0 * (-xi**2 + y**2 + b) / denom**2


def kp_lump_torch(x, y, t, sigma=-3.0):
    b = -3.0 / sigma
    c = -sigma
    xi = x - c * t
    denom = xi**2 + y**2 + b
    return 4.0 * (-xi**2 + y**2 + b) / denom**2


def kp_line_soliton(x, y, t, k1=1.0, m1=0.0, sigma=-3.0):
    """Line soliton: u = (k1^2/2) sech^2(0.5*(k1*x + m1*y - omega*t))."""
    omega = (k1**4 + sigma * m1**2) / k1
    arg = 0.5 * (k1 * x + m1 * y - omega * t)
    return (k1**2 / 2.0) / np.cosh(arg)**2


def kp_line_soliton_torch(x, y, t, k1=1.0, m1=0.0, sigma=-3.0):
    omega = (k1**4 + sigma * m1**2) / k1
    arg = 0.5 * (k1 * x + m1 * y - omega * t)
    return (k1**2 / 2.0) / torch.cosh(arg)**2


# ---------------------------------------------------------------------------
# Conservation law computation (torch, differentiable)
# ---------------------------------------------------------------------------

def compute_invariants_1d(u, x, dx, mu=None, eps=1.0):
    """Compute conservation integrals for 1D solutions.
    If mu is None: KdV-style (C1=int u, C2=int u^2, C3=int(u^3 - 0.5*ux^2))
    If mu is given: RLW-style (I1=int u, I2=int(u^2+mu*ux^2),
                               I3=int(eps*u^3 + 3u^2))
    """
    ux = torch.gradient(u, spacing=(dx,), dim=-1)[0]
    C1 = torch.trapezoid(u, dx=dx)
    if mu is None:
        C2 = torch.trapezoid(u**2, dx=dx)
        C3 = torch.trapezoid(u**3 - 0.5 * ux**2, dx=dx)
    else:
        C2 = torch.trapezoid(u**2 + mu * ux**2, dx=dx)
        C3 = torch.trapezoid(eps * u**3 + 3.0 * u**2, dx=dx)
    return C1, C2, C3

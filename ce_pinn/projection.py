"""Conservation projection layers.

Two projections are provided.

`polynomial_projection` reproduces the affine standardisation of PINN-Proj and
SDIFP. It enforces invariants of the form int(u) and int(u^2).

`helmholtz_projection` is the contribution of this work. It enforces
int(u) = C1 together with a derivative-dependent invariant
int(u^2 + mu*u_x^2) = C2 by solving the Euler-Lagrange equation

    (1 + lam2) u* - lam2*mu*u*_xx = u~ - lam1/2,   u*(a)=ga, u*(b)=gb,

which is a linear two-point boundary value problem for fixed multipliers. The
multipliers are found by Newton's method on a 2x2 system. All nonlinearity is
confined to those two scalars; the N-dimensional part stays linear, which is
what keeps the cost at one banded solve per Newton iteration.

Each projection returns `(u_star, state)`. The state carries everything needed
to apply the projection's Jacobian to a tangent vector, which is how the time
derivative of the projected field is obtained (see `project_tangent`).

Discretisation note: the Helmholtz operator uses the three-point second-order
stencil, and the invariant uses the matching second-order central difference
`D1c`. The same `D1c` must be used to compute the reference value C2, so that
the discrete constraint is exactly attainable. The higher-order operators in
`operators.py` are for the PDE residual, not for the invariant.
"""
import torch


# ---------------------------------------------------------------------------
# Second-order central difference used inside the invariant
# ---------------------------------------------------------------------------

def central_d1_matrix(N, dx, device='cpu', dtype=torch.float64):
    """Second-order first-derivative matrix, consistent with the 3-point
    Helmholtz operator. Used for the u_x term inside the invariant."""
    D = torch.zeros((N, N), device=device, dtype=dtype)
    idx = torch.arange(1, N - 1, device=device)
    D[idx, idx - 1] = -1.0 / (2.0 * dx)
    D[idx, idx + 1] = 1.0 / (2.0 * dx)
    D[0, 0] = -3.0 / (2.0 * dx)
    D[0, 1] = 4.0 / (2.0 * dx)
    D[0, 2] = -1.0 / (2.0 * dx)
    D[N - 1, N - 1] = 3.0 / (2.0 * dx)
    D[N - 1, N - 2] = -4.0 / (2.0 * dx)
    D[N - 1, N - 3] = 1.0 / (2.0 * dx)
    return D


def central_d2_matrix(N, dx, device='cpu', dtype=torch.float64):
    """Three-point second-derivative matrix.

    This must be exactly the operator that appears in the Helmholtz matrix,
    otherwise the sensitivity p2 is inconsistent with the system and Newton
    fails to converge. Boundary rows are left at zero: p1 and p2 vanish there.
    """
    D = torch.zeros((N, N), device=device, dtype=dtype)
    idx = torch.arange(1, N - 1, device=device)
    D[idx, idx - 1] = 1.0 / (dx * dx)
    D[idx, idx] = -2.0 / (dx * dx)
    D[idx, idx + 1] = 1.0 / (dx * dx)
    return D


def invariant_i2(u, q, D1c, mu):
    """Discrete int(u^2 + mu*u_x^2) dx."""
    ux = torch.matmul(u, D1c.T)
    return torch.dot(q, u * u + mu * ux * ux)


# ---------------------------------------------------------------------------
# Helmholtz operator assembly
# ---------------------------------------------------------------------------

def build_helmholtz_matrix(N, kappa, dx, device, dtype):
    """Dense tridiagonal Helmholtz matrix with Dirichlet boundary rows.

    Row 0 and row N-1 are replaced by the identity so that u*(a) and u*(b) are
    imposed directly. Only the super-diagonal entry of row 0 and the
    sub-diagonal entry of row N-1 are removed. The sub-diagonal of row 1 and
    the super-diagonal of row N-2 are retained: they carry the boundary data
    into the interior. Zeroing them would decouple the first and last interior
    nodes from the boundary, which is harmless only when ga = gb = 0.
    """
    c = kappa / (dx * dx)
    A = torch.zeros((N, N), device=device, dtype=dtype)
    idx = torch.arange(1, N - 1, device=device)
    A[idx, idx] = 1.0 + 2.0 * c
    A[idx, idx - 1] = -c
    A[idx, idx + 1] = -c
    A[0, 0] = 1.0
    A[N - 1, N - 1] = 1.0
    return A


class _State:
    """Everything needed to apply the projection Jacobian to a tangent."""
    __slots__ = ('kind', 'A', 'P', 'R', 'J', 'coeff', 'u_star', 'u_tilde',
                 'q', 'D1c', 'mu', 'alpha', 'm_tilde', 'sigma_tilde',
                 'sigma0', 'vol', 'newton_iters', 'residual', 'lam', 'clamped')


# ---------------------------------------------------------------------------
# Polynomial (affine) projection - PRIOR WORK, used as baseline
# ---------------------------------------------------------------------------

def polynomial_projection(u_tilde, C1, C2, q):
    """Affine rescaling matching int(u) = C1 and int(u^2) = C2.

    u* = (sigma0/sigma_tilde) * (u_tilde - m_tilde) + m0

    This is the closed-form solution of the constrained least-squares problem
    when both invariants are polynomial. Reproduces SDIFP / PINN-Proj.
    """
    vol = q.sum()
    m_tilde = torch.dot(q, u_tilde) / vol
    s_tilde = torch.dot(q, u_tilde * u_tilde) / vol
    var_tilde = torch.clamp(s_tilde - m_tilde * m_tilde, min=1e-30)
    sigma_tilde = torch.sqrt(var_tilde)

    m0 = C1 / vol
    var0 = torch.clamp(C2 / vol - m0 * m0, min=1e-30) if torch.is_tensor(C2) \
        else max(C2 / vol.item() - (C1 / vol.item()) ** 2, 1e-30)
    sigma0 = torch.as_tensor(var0, device=u_tilde.device,
                             dtype=u_tilde.dtype).sqrt()

    alpha = sigma0 / sigma_tilde
    u_star = alpha * (u_tilde - m_tilde) + m0

    st = _State()
    st.kind = 'polynomial'
    st.u_tilde = u_tilde.detach()
    st.q = q
    st.vol = vol.detach()
    st.alpha = alpha.detach()
    st.m_tilde = m_tilde.detach()
    st.sigma_tilde = sigma_tilde.detach()
    st.sigma0 = sigma0.detach()
    st.newton_iters = 0
    st.residual = 0.0
    st.lam = (0.0, 0.0)
    st.clamped = False
    return u_star, st


# ---------------------------------------------------------------------------
# Helmholtz projection - THIS WORK
# ---------------------------------------------------------------------------

def _initial_guess(u_tilde, C1, C2, mu, q, D1c):
    """Analytic starting point for the Newton iteration.

    Dropping the smoothing term, the Euler-Lagrange solution is the affine map
    u* = (u~ - lam1/2)/(1 + lam2), for which I2[u*] ~ I2[u~]/(1+lam2)^2. That
    inverts to

        lam2 ~ sqrt(I2[u~]/C2) - 1,
        lam1 ~ 2*(int u~ - (1+lam2)*C1)/|Omega|.

    Starting from zero instead means Newton has to travel to lam2 of order 30
    at late times, which needs many damped steps and can stray into lam2 < 0
    where the operator is indefinite. This guess is cheap and adapts to each
    time level, unlike a single cached multiplier.
    """
    vol = float(q.sum())
    i2 = float(invariant_i2(u_tilde, q, D1c, mu))
    c2 = float(C2)
    ratio = i2 / c2 if c2 > 0 else 1.0
    lam2 = (ratio ** 0.5) - 1.0 if ratio > 0 else 0.0
    lam1 = 2.0 * (float(torch.dot(q, u_tilde)) - (1.0 + lam2) * float(C1)) / vol
    return lam1, lam2


def lam2_lower_bound(mu, dx, safety=0.9):
    """Smallest lam2 for which the discrete operator stays positive definite.

    For lam2 < 0 the coefficient kappa = lam2*mu/(1+lam2) is negative and
    A = I - kappa*D2 becomes I + |kappa|*D2. The discrete Laplacian D2 has
    eigenvalues down to -4/dx^2, so A loses positive definiteness once
    |kappa| exceeds dx^2/4. Inverting kappa(lam2) at kappa = -safety*dx^2/4
    gives the bound below.

    Negative lam2 is the legitimate case where the field carries too little
    I2 and has to be amplified, but the amplification an elliptic operator can
    supply on a given grid is limited. Clamping here keeps a rare bad step
    from producing a near-singular solve.
    """
    kappa_min = -safety * dx * dx / 4.0
    return kappa_min / (mu - kappa_min)


def _newton_solve(u_tilde, C1, C2, mu, dx, q, D1c, ga, gb, tol, max_iter,
                  lam_init=None):
    """Run Newton on (lam1, lam2). Returns the converged state.

    Runs without autograd tracking; the gradient is supplied analytically by
    the custom Function below. `lam_init` warm-starts from the multipliers of
    the previous training step, which typically cuts the iteration count to
    one or two because the network changes slowly between steps.
    """
    N = u_tilde.shape[0]
    device, dtype = u_tilde.device, u_tilde.dtype
    lam2_min = lam2_lower_bound(mu, dx)
    if lam_init is None:
        lam1, lam2 = _initial_guess(u_tilde, C1, C2, mu, q, D1c)
    else:
        lam1, lam2 = float(lam_init[0]), float(lam_init[1])
    lam1 = torch.as_tensor(lam1, device=device, dtype=dtype)
    lam2 = torch.as_tensor(max(lam2, lam2_min), device=device, dtype=dtype)
    D2c = central_d2_matrix(N, dx, device, dtype)

    A = None
    u_star = None
    n_iter = 0
    resid = float('inf')

    for n_iter in range(1, max_iter + 1):
        coeff = 1.0 + lam2
        kappa = lam2 * mu / coeff
        A = build_helmholtz_matrix(N, kappa, dx, device, dtype)

        h = (u_tilde - lam1 / 2.0) / coeff
        h = h.clone()
        h[0] = ga
        h[-1] = gb

        u_star = torch.linalg.solve(A, h)

        F1 = torch.dot(q, u_star) - C1
        F2 = invariant_i2(u_star, q, D1c, mu) - C2
        resid = float(F1.abs() + F2.abs())
        if resid < tol:
            break

        # Sensitivities: same operator, two extra right-hand sides.
        rhs = torch.zeros((N, 2), device=device, dtype=dtype)
        rhs[1:-1, 0] = -0.5 / coeff
        u_star_xx = torch.matmul(u_star, D2c.T)
        rhs[1:-1, 1] = ((mu * u_star_xx - u_star) / coeff)[1:-1]
        P = torch.linalg.solve(A, rhs)

        R = _constraint_gradients(u_star, q, D1c, mu)
        J = R.T @ P

        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if float(det.abs()) < 1e-300:
            break
        F = torch.stack([F1, F2])
        dlam = -torch.linalg.solve(J, F)

        # Backtracking line search. A full Newton step from a cold start can
        # overshoot badly when the incoming field is far from the constraint
        # set, which happens early in training. Halve the step until the
        # residual decreases, and keep lam2 > -1 so the operator stays
        # non-singular.
        step = 1.0
        for _ in range(30):
            l1 = lam1 + step * dlam[0]
            l2 = lam2 + step * dlam[1]
            if float(l2) < lam2_min:
                l2 = torch.as_tensor(lam2_min, device=device, dtype=dtype)
            cf = 1.0 + l2
            kp = l2 * mu / cf
            At = build_helmholtz_matrix(N, kp, dx, device, dtype)
            ht = (u_tilde - l1 / 2.0) / cf
            ht = ht.clone()
            ht[0] = ga
            ht[-1] = gb
            ut = torch.linalg.solve(At, ht)
            r_new = float((torch.dot(q, ut) - C1).abs()
                          + (invariant_i2(ut, q, D1c, mu) - C2).abs())
            if r_new < resid or step < 1e-6:
                break
            step *= 0.5
        lam1 = lam1 + step * dlam[0]
        lam2 = torch.clamp(lam2 + step * dlam[1], min=lam2_min)

    # Recompute sensitivities AT the converged multipliers. Inside the loop
    # they are produced before the next convergence test, so on exit they
    # correspond to the previous iterate.
    coeff = 1.0 + lam2
    kappa = lam2 * mu / coeff
    A = build_helmholtz_matrix(N, kappa, dx, device, dtype)
    h = (u_tilde - lam1 / 2.0) / coeff
    h = h.clone()
    h[0] = ga
    h[-1] = gb
    u_star = torch.linalg.solve(A, h)

    rhs = torch.zeros((N, 2), device=device, dtype=dtype)
    rhs[1:-1, 0] = -0.5 / coeff
    u_star_xx = torch.matmul(u_star, D2c.T)
    rhs[1:-1, 1] = ((mu * u_star_xx - u_star) / coeff)[1:-1]
    P = torch.linalg.solve(A, rhs)
    R = _constraint_gradients(u_star, q, D1c, mu)
    J = R.T @ P

    st = _State()
    st.kind = 'helmholtz'
    st.lam = (float(lam1), float(lam2))
    # lam2 sitting on its lower bound means the target C2 exceeds what an
    # elliptic projection can supply on this grid. That is a well-posedness
    # limit, not a solver failure: for lam2 < 0 the second variation
    # 2*int[(1+lam2)v^2 + lam2*mu*v_x^2] is negative for high-frequency v, so
    # the continuous problem has no minimiser.
    st.clamped = bool(float(lam2) <= lam2_min + 1e-12)
    st.A = A
    st.P = P
    st.R = R
    st.J = J
    st.coeff = coeff
    st.u_star = u_star
    st.u_tilde = u_tilde
    st.q = q
    st.D1c = D1c
    st.mu = mu
    st.newton_iters = n_iter
    st.residual = resid
    return st


def _constraint_gradients(u_star, q, D1c, mu):
    """R = [dF1/du*, dF2/du*] as an (N, 2) matrix."""
    r1 = q
    ux = torch.matmul(u_star, D1c.T)
    r2 = 2.0 * q * u_star + 2.0 * mu * torch.matmul(q * ux, D1c)
    return torch.stack([r1, r2], dim=1)


def helmholtz_projection(u_tilde, C1, C2, mu, dx, q, D1c,
                         ga=0.0, gb=0.0, tol=1e-12, max_iter=20,
                         lam_init=None):
    """Project u_tilde onto {int u = C1, int(u^2 + mu u_x^2) = C2}.

    Returns (u_star, state). The state is needed by `project_tangent`, and
    `state.lam` can be fed back as `lam_init` on the next step.
    """
    with torch.no_grad():
        st = _newton_solve(u_tilde.detach(), C1, C2, mu, dx, q, D1c,
                           ga, gb, tol, max_iter, lam_init)
    u_star = _HelmholtzApply.apply(u_tilde, st)
    return u_star, st


class _HelmholtzApply(torch.autograd.Function):
    """Attaches the analytic gradient to an already-computed projection."""

    @staticmethod
    def forward(ctx, u_tilde, st):
        ctx.state = st
        return st.u_star.clone()

    @staticmethod
    def backward(ctx, grad_out):
        st = ctx.state
        s = st.P.T @ grad_out
        z = torch.linalg.solve(st.J.T, s)
        g_eff = grad_out - st.R @ z
        w = torch.linalg.solve(st.A.T, g_eff)
        g_tilde = (w / st.coeff).clone()
        g_tilde[0] = 0.0
        g_tilde[-1] = 0.0
        return g_tilde, None


# ---------------------------------------------------------------------------
# Tangent propagation: d(u*)/dt from d(u~)/dt
# ---------------------------------------------------------------------------

def project_tangent(v, st):
    """Apply the projection Jacobian to a tangent vector.

    Because u*(t) = P(u~(t)), the chain rule gives
        du*/dt = (dP/du~) du~/dt = (I - Pi) M v,
    with M = A^{-1} E / (1 + lam2) and Pi = P J^{-1} R^T. This is exact for the
    tangent. The state is held fixed, so second-order dependence of the
    Jacobian on u~ is not differentiated through during backpropagation.
    """
    if st.kind == 'polynomial':
        vol = st.vol
        dm = torch.dot(st.q, v) / vol
        ds2 = torch.dot(st.q, 2.0 * st.u_tilde * v) / vol - 2.0 * st.m_tilde * dm
        dsig = ds2 / (2.0 * st.sigma_tilde)
        return (-(st.sigma0 / st.sigma_tilde ** 2) * dsig
                * (st.u_tilde - st.m_tilde)
                + st.alpha * (v - dm))

    Ev = v.clone()
    Ev[0] = 0.0
    Ev[-1] = 0.0
    Mv = torch.linalg.solve(st.A, Ev) / st.coeff
    s = st.R.T @ Mv
    z = torch.linalg.solve(st.J, s)
    return Mv - st.P @ z


# ---------------------------------------------------------------------------
# 2D polynomial projection (KP)
# ---------------------------------------------------------------------------

def polynomial_projection_2d(u_tilde, C1, C2, dx, dy):
    """Affine rescaling on a 2D field. Same formula, 2D integrals."""
    Nx, Ny = u_tilde.shape
    area = dx * (Nx - 1) * dy * (Ny - 1)
    int_u = torch.trapezoid(torch.trapezoid(u_tilde, dx=dy, dim=1), dx=dx, dim=0)
    int_u2 = torch.trapezoid(torch.trapezoid(u_tilde ** 2, dx=dy, dim=1),
                             dx=dx, dim=0)
    m_tilde = int_u / area
    var_tilde = torch.clamp(int_u2 / area - m_tilde ** 2, min=1e-30)
    m0 = C1 / area
    var0 = max(C2 / area - m0 ** 2, 1e-30)
    alpha = torch.sqrt(torch.as_tensor(var0, device=u_tilde.device,
                                       dtype=u_tilde.dtype) / var_tilde)
    return alpha * (u_tilde - m_tilde) + m0

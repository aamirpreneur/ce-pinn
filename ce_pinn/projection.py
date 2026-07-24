"""Conservation projection layers.

Polynomial projection (KdV/KP): affine rescaling to match C1, C2.
Helmholtz projection (RLW): Newton on 2x2 system with tridiagonal solves.
"""
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Polynomial (affine) projection — reproduces SDIFP baseline
# ---------------------------------------------------------------------------

def polynomial_projection(u_tilde, C1, C2, dx):
    """Affine rescale u_tilde to satisfy int u dx = C1 and int u^2 dx = C2.

    u* = (sigma0 / sigma_tilde) * (u_tilde - mu_tilde) + mu0

    Args:
        u_tilde: (N,) tensor, BC-satisfying network output at one time level
        C1, C2: target invariant values (scalars)
        dx: grid spacing (float)
    Returns:
        u_star: (N,) tensor satisfying both invariants
    """
    N = u_tilde.shape[0]
    L = dx * (N - 1)

    mu_tilde = torch.trapezoid(u_tilde, dx=dx) / L
    sigma2_tilde = torch.trapezoid(u_tilde**2, dx=dx) / L - mu_tilde**2

    mu0 = C1 / L
    sigma2_0 = C2 / L - mu0**2

    sigma_ratio = torch.sqrt(torch.clamp(sigma2_0, min=1e-30) /
                             torch.clamp(sigma2_tilde, min=1e-30))
    u_star = sigma_ratio * (u_tilde - mu_tilde) + mu0
    return u_star


# ---------------------------------------------------------------------------
# Thomas algorithm for tridiagonal systems (pure torch, O(N))
# ---------------------------------------------------------------------------

def thomas_solve(lower, diag, upper, rhs):
    """Solve tridiagonal system Ax = rhs via Thomas algorithm.

    Args:
        lower: (N-1,) sub-diagonal
        diag:  (N,) main diagonal
        upper: (N-1,) super-diagonal
        rhs:   (N,) right-hand side
    Returns:
        x: (N,) solution
    """
    N = diag.shape[0]
    c = torch.zeros(N, dtype=diag.dtype, device=diag.device)
    d = torch.zeros(N, dtype=diag.dtype, device=diag.device)

    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, N):
        denom = diag[i] - lower[i - 1] * c[i - 1]
        if i < N - 1:
            c[i] = upper[i] / denom
        d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / denom

    x = torch.zeros(N, dtype=diag.dtype, device=diag.device)
    x[N - 1] = d[N - 1]
    for i in range(N - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


# ---------------------------------------------------------------------------
# Helmholtz projection — the novel contribution
# ---------------------------------------------------------------------------

def _build_helmholtz_tridiag(N, kappa, dx):
    """Build tridiagonal Helmholtz operator H = (1+lam2)I - lam2*mu*D_xx.

    In practice we factor out and work with H such that Hu* = rhs.
    The operator is: diag = 1 + 2*kappa/dx^2, off-diag = -kappa/dx^2.
    Boundary rows are identity (BCs imposed directly).
    """
    c = kappa / dx**2
    diag = torch.full((N,), 1.0 + 2.0 * c, dtype=torch.float64)
    off = torch.full((N - 1,), -c, dtype=torch.float64)

    diag[0] = 1.0
    diag[-1] = 1.0
    off[0] = 0.0
    off[-1] = 0.0

    return off, diag, off.clone()


class HelmholtzProjection(torch.autograd.Function):
    """Custom autograd function for Helmholtz projection with IFT backward."""

    @staticmethod
    def forward(ctx, u_tilde, C1, C2, mu, dx, ga, gb, tol, max_iter):
        """Newton solve for lambda1, lambda2, then Helmholtz solve for u*.

        Args:
            u_tilde: (N,) tensor
            C1, C2: target invariants (float)
            mu: dispersion parameter (float)
            dx: grid spacing (float)
            ga, gb: boundary values (float)
            tol: Newton convergence tolerance
            max_iter: max Newton iterations
        Returns:
            u_star: (N,) projected solution
        """
        N = u_tilde.shape[0]
        lam1 = torch.tensor(0.0, dtype=torch.float64)
        lam2 = torch.tensor(0.0, dtype=torch.float64)

        for newton_iter in range(max_iter):
            coeff = 1.0 + lam2
            kappa = lam2 * mu / coeff
            rhs_helm = (u_tilde - lam1 / 2.0) / coeff
            rhs_helm[0] = ga
            rhs_helm[-1] = gb

            lower, diag, upper = _build_helmholtz_tridiag(N, kappa, dx)
            u_star = thomas_solve(lower, diag, upper, rhs_helm)

            u_star_x = torch.zeros_like(u_star)
            u_star_x[1:-1] = (u_star[2:] - u_star[:-2]) / (2.0 * dx)
            u_star_x[0] = (u_star[1] - u_star[0]) / dx
            u_star_x[-1] = (u_star[-1] - u_star[-2]) / dx

            F1 = torch.trapezoid(u_star, dx=dx) - C1
            F2 = torch.trapezoid(u_star**2 + mu * u_star_x**2, dx=dx) - C2

            if abs(F1.item()) + abs(F2.item()) < tol:
                break

            # Sensitivity p1: H p1 = -1/2, p1(a)=p1(b)=0
            rhs_p1 = torch.full((N,), -0.5 / coeff, dtype=torch.float64)
            rhs_p1[0] = 0.0
            rhs_p1[-1] = 0.0
            p1 = thomas_solve(lower, diag, upper, rhs_p1)

            # Sensitivity p2: H p2 = (-u* + mu*u*_xx) / coeff, p2(a)=p2(b)=0
            u_star_xx = torch.zeros_like(u_star)
            u_star_xx[1:-1] = (u_star[2:] - 2.0 * u_star[1:-1] + u_star[:-2]) / dx**2
            rhs_p2 = (-u_star + mu * u_star_xx) / coeff
            rhs_p2[0] = 0.0
            rhs_p2[-1] = 0.0
            p2 = thomas_solve(lower, diag, upper, rhs_p2)

            p1_x = torch.zeros_like(p1)
            p1_x[1:-1] = (p1[2:] - p1[:-2]) / (2.0 * dx)
            p2_x = torch.zeros_like(p2)
            p2_x[1:-1] = (p2[2:] - p2[:-2]) / (2.0 * dx)

            J11 = torch.trapezoid(p1, dx=dx)
            J12 = torch.trapezoid(p2, dx=dx)
            J21 = torch.trapezoid(2.0 * u_star * p1 + 2.0 * mu * u_star_x * p1_x, dx=dx)
            J22 = torch.trapezoid(2.0 * u_star * p2 + 2.0 * mu * u_star_x * p2_x, dx=dx)

            det = J11 * J22 - J12 * J21
            if abs(det.item()) < 1e-30:
                break
            dlam1 = -(J22 * F1 - J12 * F2) / det
            dlam2 = -(-J21 * F1 + J11 * F2) / det

            lam1 = lam1 + dlam1
            lam2 = lam2 + dlam2

        ctx.save_for_backward(u_tilde, u_star, lam1, lam2)
        ctx.mu = mu
        ctx.dx = dx
        ctx.ga = ga
        ctx.gb = gb
        return u_star

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass via implicit function theorem (adjoint method)."""
        u_tilde, u_star, lam1, lam2 = ctx.saved_tensors
        mu = ctx.mu
        dx = ctx.dx
        N = u_tilde.shape[0]

        coeff = 1.0 + lam2
        kappa = lam2 * mu / coeff
        lower, diag, upper = _build_helmholtz_tridiag(N, kappa, dx)

        # Adjoint solve: H^T v = grad_output (H is symmetric so H^T = H)
        rhs_adj = grad_output.clone()
        rhs_adj[0] = 0.0
        rhs_adj[-1] = 0.0
        v = thomas_solve(lower, diag, upper, rhs_adj)

        # grad w.r.t. u_tilde: du*/du_tilde = H^{-1} / (1+lam2) for interior
        grad_u_tilde = v / coeff
        grad_u_tilde[0] = 0.0
        grad_u_tilde[-1] = 0.0

        return grad_u_tilde, None, None, None, None, None, None, None, None


def helmholtz_projection(u_tilde, C1, C2, mu, dx, ga=0.0, gb=0.0,
                          tol=1e-12, max_iter=20):
    """Apply Helmholtz projection to enforce I1=C1, I2=int(u^2+mu*ux^2)=C2.

    Args:
        u_tilde: (N,) tensor, BC-satisfying network output at one time level
        C1: target mass invariant
        C2: target momentum invariant (derivative-dependent)
        mu: dispersion parameter
        dx: grid spacing
        ga, gb: boundary values for u*
        tol: Newton tolerance
        max_iter: max Newton iterations
    Returns:
        u_star: (N,) projected solution
    """
    return HelmholtzProjection.apply(u_tilde, C1, C2, mu, dx, ga, gb, tol, max_iter)


# ---------------------------------------------------------------------------
# 2D polynomial projection for KP
# ---------------------------------------------------------------------------

def polynomial_projection_2d(u_tilde, C1, C2, dx, dy):
    """Affine rescale 2D field to match C1=int u dA and C2=int u^2 dA.

    Args:
        u_tilde: (Nx, Ny) tensor
        C1, C2: target invariant values
        dx, dy: grid spacings
    Returns:
        u_star: (Nx, Ny) tensor
    """
    Nx, Ny = u_tilde.shape
    area = dx * (Nx - 1) * dy * (Ny - 1)

    int_u = torch.trapezoid(torch.trapezoid(u_tilde, dx=dy, dim=1), dx=dx, dim=0)
    int_u2 = torch.trapezoid(torch.trapezoid(u_tilde**2, dx=dy, dim=1), dx=dx, dim=0)

    mu_tilde = int_u / area
    sigma2_tilde = int_u2 / area - mu_tilde**2

    mu0 = C1 / area
    sigma2_0 = C2 / area - mu0**2

    sigma_ratio = torch.sqrt(torch.clamp(sigma2_0, min=1e-30) /
                             torch.clamp(sigma2_tilde, min=1e-30))
    return sigma_ratio * (u_tilde - mu_tilde) + mu0

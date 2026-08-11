"""Finite-difference operators and quadrature on a uniform grid.

The conservation projection is a grid operation, so the PDE residual is
evaluated on the same grid. Spatial derivatives therefore come from finite
differences rather than automatic differentiation. Interior stencils are
fourth-order; the two nodes at each end fall back to one-sided stencils.

All matrices are dense (N <= ~1001 in practice) so they can be applied with a
single matmul and differentiated through by autograd without special handling.
"""
import torch


def trapezoid_weights(N, dx, device='cpu', dtype=torch.float64):
    """Weights q such that q @ u approximates the integral of u."""
    q = torch.full((N,), dx, device=device, dtype=dtype)
    q[0] = dx / 2.0
    q[-1] = dx / 2.0
    return q


def _empty(N, device, dtype):
    return torch.zeros((N, N), device=device, dtype=dtype)


def fd_matrix_d1(N, dx, device='cpu', dtype=torch.float64):
    """First derivative. Fourth-order interior, second-order one-sided ends."""
    D = _empty(N, device, dtype)
    for i in range(2, N - 2):
        D[i, i - 2] = 1.0 / (12.0 * dx)
        D[i, i - 1] = -8.0 / (12.0 * dx)
        D[i, i + 1] = 8.0 / (12.0 * dx)
        D[i, i + 2] = -1.0 / (12.0 * dx)
    # one-sided, second order
    for i in (0, 1):
        D[i, i] = -3.0 / (2.0 * dx)
        D[i, i + 1] = 4.0 / (2.0 * dx)
        D[i, i + 2] = -1.0 / (2.0 * dx)
    for i in (N - 2, N - 1):
        D[i, i] = 3.0 / (2.0 * dx)
        D[i, i - 1] = -4.0 / (2.0 * dx)
        D[i, i - 2] = 1.0 / (2.0 * dx)
    return D


def fd_matrix_d2(N, dx, device='cpu', dtype=torch.float64):
    """Second derivative. Fourth-order interior, second-order one-sided ends."""
    D = _empty(N, device, dtype)
    h2 = dx * dx
    for i in range(2, N - 2):
        D[i, i - 2] = -1.0 / (12.0 * h2)
        D[i, i - 1] = 16.0 / (12.0 * h2)
        D[i, i] = -30.0 / (12.0 * h2)
        D[i, i + 1] = 16.0 / (12.0 * h2)
        D[i, i + 2] = -1.0 / (12.0 * h2)
    for i in (0, 1):
        D[i, i] = 2.0 / h2
        D[i, i + 1] = -5.0 / h2
        D[i, i + 2] = 4.0 / h2
        D[i, i + 3] = -1.0 / h2
    for i in (N - 2, N - 1):
        D[i, i] = 2.0 / h2
        D[i, i - 1] = -5.0 / h2
        D[i, i - 2] = 4.0 / h2
        D[i, i - 3] = -1.0 / h2
    return D


def fd_matrix_d3(N, dx, device='cpu', dtype=torch.float64):
    """Third derivative. Fourth-order interior, second-order one-sided ends."""
    D = _empty(N, device, dtype)
    h3 = dx ** 3
    for i in range(3, N - 3):
        D[i, i - 3] = 1.0 / (8.0 * h3)
        D[i, i - 2] = -8.0 / (8.0 * h3)
        D[i, i - 1] = 13.0 / (8.0 * h3)
        D[i, i + 1] = -13.0 / (8.0 * h3)
        D[i, i + 2] = 8.0 / (8.0 * h3)
        D[i, i + 3] = -1.0 / (8.0 * h3)
    for i in (0, 1, 2):
        D[i, i] = -1.0 / h3
        D[i, i + 1] = 3.0 / h3
        D[i, i + 2] = -3.0 / h3
        D[i, i + 3] = 1.0 / h3
    for i in (N - 3, N - 2, N - 1):
        D[i, i] = 1.0 / h3
        D[i, i - 1] = -3.0 / h3
        D[i, i - 2] = 3.0 / h3
        D[i, i - 3] = -1.0 / h3
    return D


class GridOps:
    """Bundle of grid operators for one spatial discretisation."""

    def __init__(self, xa, xb, N, device='cpu', dtype=torch.float64,
                 need_d3=False):
        self.N = N
        self.dx = (xb - xa) / (N - 1)
        self.x = torch.linspace(xa, xb, N, device=device, dtype=dtype)
        self.q = trapezoid_weights(N, self.dx, device, dtype)
        self.D1 = fd_matrix_d1(N, self.dx, device, dtype)
        self.D2 = fd_matrix_d2(N, self.dx, device, dtype)
        self.D3 = fd_matrix_d3(N, self.dx, device, dtype) if need_d3 else None
        self.device = device
        self.dtype = dtype

    def integral(self, u):
        """Trapezoidal integral along the last axis."""
        return torch.matmul(u, self.q)

    def d1(self, u):
        return torch.matmul(u, self.D1.T)

    def d2(self, u):
        return torch.matmul(u, self.D2.T)

    def d3(self, u):
        if self.D3 is None:
            raise RuntimeError("GridOps built without need_d3=True")
        return torch.matmul(u, self.D3.T)

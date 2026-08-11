"""MLP with hard boundary condition embedding (Lagaris-type)."""
import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden_layers, hidden_width, activation=nn.SiLU):
        super().__init__()
        layers = []
        prev = d_in
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev, hidden_width))
            layers.append(activation())
            prev = hidden_width
        layers.append(nn.Linear(prev, d_out))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class VanillaNet(nn.Module):
    """Plain MLP with the same (x, t) call signature as the hard-BC layers.

    Used for the `vanilla` baseline, which imposes the initial and boundary
    conditions through loss penalties instead of construction.
    """
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, x, t):
        inp = torch.stack([x.reshape(-1), t.reshape(-1)], dim=-1)
        return self.mlp(inp).squeeze(-1).reshape(x.shape)


class HardBC1D(nn.Module):
    """Hard boundary + initial condition embedding for 1D evolution PDEs.

    û(x,t) = A(x,t) + B(x,t) * N(x,t; theta)

    where A enforces IC and BCs exactly, B vanishes on boundaries and at t=0.
    """
    def __init__(self, mlp, ic_fn, xa, xb, T=1.0, ga_fn=None, gb_fn=None):
        """
        Args:
            mlp: raw MLP network
            ic_fn: callable(x_tensor) -> u(x,0), initial condition
            xa, xb: domain boundaries (float)
            T: final time, used to normalise the mask B to O(1)
            ga_fn: callable(t_tensor) -> u(xa,t), left BC (default: constant ic_fn(xa))
            gb_fn: callable(t_tensor) -> u(xb,t), right BC (default: constant ic_fn(xb))
        """
        super().__init__()
        self.mlp = mlp
        self.ic_fn = ic_fn
        self.xa = float(xa)
        self.xb = float(xb)
        self.T = float(T)
        self._ga_fn = ga_fn
        self._gb_fn = gb_fn

        xa_t = torch.tensor([self.xa], dtype=torch.float64)
        xb_t = torch.tensor([self.xb], dtype=torch.float64)
        self._ga0 = float(ic_fn(xa_t).item())
        self._gb0 = float(ic_fn(xb_t).item())

    def _ga(self, t):
        if self._ga_fn is not None:
            return self._ga_fn(t)
        return torch.full_like(t, self._ga0)

    def _gb(self, t):
        if self._gb_fn is not None:
            return self._gb_fn(t)
        return torch.full_like(t, self._gb0)

    def forward(self, x, t):
        """x, t: tensors of shape (N,) or (N,1). Returns û of same shape."""
        x_flat = x.reshape(-1)
        t_flat = t.reshape(-1)
        inp = torch.stack([x_flat, t_flat], dim=-1)

        N_raw = self.mlp(inp).squeeze(-1)

        s = (x_flat - self.xa) / (self.xb - self.xa)
        f_x = self.ic_fn(x_flat)
        ga_t = self._ga(t_flat)
        gb_t = self._gb(t_flat)

        A = f_x + (1.0 - s) * (ga_t - self._ga0) + s * (gb_t - self._gb0)
        # Normalised mask: peaks at 1 in the middle of the domain at t = T.
        # Without the normalisation B reaches T*((b-a)/2)^2, which is ~5e4 on
        # the RLW domain and amplifies the untrained network by four orders of
        # magnitude.
        B = (t_flat / self.T) * 4.0 * s * (1.0 - s)

        u_hat = A + B * N_raw
        return u_hat.reshape(x.shape)


class HardBC2D(nn.Module):
    """Hard BC embedding for 2D evolution PDEs (e.g., KP).

    û(x,y,t) = f(x,y) + t*(x-a1)*(b1-x)*(y-a2)*(b2-y)*N(x,y,t; theta)
    """
    def __init__(self, mlp, ic_fn, xa, xb, ya, yb, T=1.0):
        super().__init__()
        self.mlp = mlp
        self.ic_fn = ic_fn
        self.xa, self.xb = float(xa), float(xb)
        self.ya, self.yb = float(ya), float(yb)
        self.T = float(T)

    def forward(self, x, y, t):
        """x, y, t: tensors of shape (N,). Returns û of shape (N,)."""
        inp = torch.stack([x, y, t], dim=-1)
        N_raw = self.mlp(inp).squeeze(-1)

        f_xy = self.ic_fn(x, y)
        # Normalised mask, peaking at 1 at the domain centre at t = T. The
        # unnormalised product reaches ~1e6 on the KP domain.
        sx = (x - self.xa) / (self.xb - self.xa)
        sy = (y - self.ya) / (self.yb - self.ya)
        B = (t / self.T) * 16.0 * sx * (1.0 - sx) * sy * (1.0 - sy)

        return f_xy + B * N_raw

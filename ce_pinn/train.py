"""Training loop for CE-PINN: Adam -> L-BFGS with optional conservation projection."""
import os
import json
import time
import torch
import numpy as np
from . import equations as eq
from . import projection as proj


def latin_hypercube_2d(n, bounds):
    """LHS in 2D: returns (n, 2) array in [bounds[0], bounds[1]] x [bounds[2], bounds[3]]."""
    rng = np.random.default_rng()
    d = 2
    samples = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        samples[:, j] = (perm + rng.uniform(size=n)) / n
    lo = np.array([bounds[0], bounds[2]])
    hi = np.array([bounds[1], bounds[3]])
    return samples * (hi - lo) + lo


def latin_hypercube_3d(n, bounds):
    """LHS in 3D for KP: (x, y, t) in given bounds."""
    rng = np.random.default_rng()
    d = 3
    samples = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        samples[:, j] = (perm + rng.uniform(size=n)) / n
    lo = np.array([bounds[0], bounds[2], bounds[4]])
    hi = np.array([bounds[1], bounds[3], bounds[5]])
    return samples * (hi - lo) + lo


def pde_residual_kdv(model, x, t):
    """KdV residual: u_t + 6uu_x + u_xxx."""
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)

    grads = torch.autograd.grad(u, [x, t], grad_outputs=torch.ones_like(u),
                                create_graph=True)
    u_x, u_t = grads[0], grads[1]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u_xx),
                                create_graph=True)[0]

    return u_t + 6.0 * u * u_x + u_xxx


def pde_residual_rlw(model, x, t, eps=1.0, mu=1.0):
    """RLW residual: u_t + u_x + eps*u*u_x - mu*u_xxt."""
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    u_xt = torch.autograd.grad(u_x, t, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    u_xxt = torch.autograd.grad(u_xt, x, grad_outputs=torch.ones_like(u_xt),
                                create_graph=True)[0]

    return u_t + u_x + eps * u * u_x - mu * u_xxt


def pde_residual_kp(model, x, y, t, sigma=-3.0):
    """KP residual: (u_t + 6uu_x + u_xxx)_x + sigma*u_yy."""
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, y, t)

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u_xx), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    inner = u_t + 6.0 * u * u_x + u_xxx
    inner_x = torch.autograd.grad(inner, x, grad_outputs=torch.ones_like(inner),
                                  create_graph=True)[0]

    return inner_x + sigma * u_yy


class Trainer:
    """Generic CE-PINN trainer for 1D problems (KdV, RLW)."""

    def __init__(self, model, config, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.out_dir = os.environ.get("OUT_DIR", ".")
        os.makedirs(self.out_dir, exist_ok=True)

        self._setup_data()
        self._compute_reference_invariants()

    def _setup_data(self):
        cfg = self.config
        bounds = [cfg['xa'], cfg['xb'], 0.0, cfg['T']]
        pts_pde = latin_hypercube_2d(cfg['n_pde'], bounds)
        self.x_pde = torch.tensor(pts_pde[:, 0], dtype=torch.float64, device=self.device)
        self.t_pde = torch.tensor(pts_pde[:, 1], dtype=torch.float64, device=self.device)

        x_ic = np.linspace(cfg['xa'], cfg['xb'], cfg['n_ic'])
        self.x_ic = torch.tensor(x_ic, dtype=torch.float64, device=self.device)
        self.t_ic = torch.zeros_like(self.x_ic)

        N_cons = cfg.get('n_conservation_grid', 256)
        self.x_cons = torch.linspace(cfg['xa'], cfg['xb'], N_cons,
                                     dtype=torch.float64, device=self.device)
        self.dx_cons = (cfg['xb'] - cfg['xa']) / (N_cons - 1)

    def _compute_reference_invariants(self):
        cfg = self.config
        x_np = self.x_cons.cpu().numpy()
        dx = self.dx_cons

        ic_vals = cfg['ic_fn'](x_np)
        ic_x = np.gradient(ic_vals, dx)

        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

        if cfg['equation'] == 'rlw':
            mu = cfg['mu']
            self.C1_ref = _trapz(ic_vals, dx=dx)
            self.C2_ref = _trapz(ic_vals**2 + mu * ic_x**2, dx=dx)
            self.C3_ref = _trapz(ic_vals**3 + 3.0 * ic_vals**2, dx=dx)
        else:
            self.C1_ref = _trapz(ic_vals, dx=dx)
            self.C2_ref = _trapz(ic_vals**2, dx=dx)
            self.C3_ref = _trapz(ic_vals**3 - 0.5 * ic_x**2, dx=dx)

        print(f"Reference invariants: C1={self.C1_ref:.8f}, C2={self.C2_ref:.8f}, "
              f"C3={self.C3_ref:.8f}")

    def _conservation_loss(self, t_samples):
        """Evaluate conservation at random time samples."""
        cfg = self.config
        loss = torch.tensor(0.0, dtype=torch.float64, device=self.device)

        for t_val in t_samples:
            t_vec = torch.full_like(self.x_cons, t_val.item())
            u_hat = self.model(self.x_cons, t_vec)

            if cfg.get('projection') == 'helmholtz':
                u_proj = proj.helmholtz_projection(
                    u_hat, self.C1_ref, self.C2_ref,
                    cfg['mu'], self.dx_cons,
                    ga=cfg.get('ga', 0.0), gb=cfg.get('gb', 0.0))
            elif cfg.get('projection') == 'polynomial':
                u_proj = proj.polynomial_projection(u_hat, self.C1_ref, self.C2_ref,
                                                    self.dx_cons)
            else:
                u_proj = u_hat

            if cfg.get('soft_I3', False):
                ux = torch.gradient(u_proj, spacing=(self.dx_cons,), dim=-1)[0]
                if cfg['equation'] == 'rlw':
                    I3 = torch.trapezoid(u_proj**3 + 3.0 * u_proj**2, dx=self.dx_cons)
                else:
                    I3 = torch.trapezoid(u_proj**3 - 0.5 * ux**2, dx=self.dx_cons)
                loss = loss + (I3 - self.C3_ref)**2

        return loss / len(t_samples)

    def _pde_loss(self):
        if self.config['equation'] == 'kdv':
            r = pde_residual_kdv(self.model, self.x_pde, self.t_pde)
        elif self.config['equation'] == 'rlw':
            r = pde_residual_rlw(self.model, self.x_pde, self.t_pde,
                                  eps=self.config['eps'], mu=self.config['mu'])
        return (r**2).mean()

    def train_adam(self, n_iter, lr=1e-3, lam_cons=0.0, log_every=500):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        cfg = self.config
        history = []

        for step in range(1, n_iter + 1):
            optimizer.zero_grad()
            loss_pde = self._pde_loss()
            loss = loss_pde

            if lam_cons > 0 and cfg.get('projection') is not None:
                n_t = cfg.get('n_t_conservation', 10)
                t_samples = torch.rand(n_t, dtype=torch.float64, device=self.device) * cfg['T']
                loss_cons = self._conservation_loss(t_samples)
                loss = loss + lam_cons * loss_cons

            loss.backward()
            optimizer.step()

            if step % log_every == 0 or step == 1:
                msg = f"[Adam {step}/{n_iter}] PDE={loss_pde.item():.6e} Total={loss.item():.6e}"
                print(msg, flush=True)
                history.append({'step': step, 'pde': loss_pde.item(), 'total': loss.item()})

            if step % (n_iter // 5) == 0 and step > 0:
                torch.save(self.model.state_dict(),
                           os.path.join(self.out_dir, f"model_adam_{step}.pth"))

        return history

    def train_lbfgs(self, n_iter, lr=1.0, lam_cons=0.0, log_every=100):
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr,
                                       max_iter=1, history_size=50,
                                       line_search_fn='strong_wolfe')
        cfg = self.config
        history = []
        step_counter = [0]

        for step in range(1, n_iter + 1):
            loss_val = [None]

            def closure():
                optimizer.zero_grad()
                loss_pde = self._pde_loss()
                loss = loss_pde

                if lam_cons > 0 and cfg.get('projection') is not None:
                    n_t = cfg.get('n_t_conservation', 10)
                    t_samples = torch.rand(n_t, dtype=torch.float64,
                                           device=self.device) * cfg['T']
                    loss_cons = self._conservation_loss(t_samples)
                    loss = loss + lam_cons * loss_cons

                loss.backward()
                loss_val[0] = loss.item()
                return loss

            optimizer.step(closure)
            step_counter[0] += 1

            if step % log_every == 0 or step == 1:
                msg = f"[L-BFGS {step}/{n_iter}] Loss={loss_val[0]:.6e}"
                print(msg, flush=True)
                history.append({'step': step, 'loss': loss_val[0]})

        torch.save(self.model.state_dict(),
                   os.path.join(self.out_dir, "model_final.pth"))
        return history

    def evaluate(self, exact_fn, n_eval=1000):
        """Compute L2 and Linf errors + conservation drift at final time."""
        cfg = self.config
        x_eval = torch.linspace(cfg['xa'], cfg['xb'], n_eval,
                                dtype=torch.float64, device=self.device)
        t_eval = torch.full_like(x_eval, cfg['T'])

        with torch.no_grad():
            u_pred = self.model(x_eval, t_eval)

        x_np = x_eval.cpu().numpy()
        u_pred_np = u_pred.cpu().numpy()
        u_exact_np = exact_fn(x_np)

        l2_err = np.sqrt(np.sum((u_pred_np - u_exact_np)**2) /
                         np.sum(u_exact_np**2 + 1e-30))
        linf_err = np.max(np.abs(u_pred_np - u_exact_np))

        dx_eval = (cfg['xb'] - cfg['xa']) / (n_eval - 1)
        ux_pred = np.gradient(u_pred_np, dx_eval)
        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

        if cfg['equation'] == 'rlw':
            mu = cfg['mu']
            I1 = _trapz(u_pred_np, dx=dx_eval)
            I2 = _trapz(u_pred_np**2 + mu * ux_pred**2, dx=dx_eval)
            I3 = _trapz(u_pred_np**3 + 3.0 * u_pred_np**2, dx=dx_eval)
        else:
            I1 = _trapz(u_pred_np, dx=dx_eval)
            I2 = _trapz(u_pred_np**2, dx=dx_eval)
            I3 = _trapz(u_pred_np**3 - 0.5 * ux_pred**2, dx=dx_eval)

        results = {
            'L2_relative': float(l2_err),
            'Linf': float(linf_err),
            'C1_pred': float(I1), 'C1_ref': float(self.C1_ref),
            'C2_pred': float(I2), 'C2_ref': float(self.C2_ref),
            'C3_pred': float(I3), 'C3_ref': float(self.C3_ref),
            'C1_drift': float(abs(I1 - self.C1_ref) / (abs(self.C1_ref) + 1e-30)),
            'C2_drift': float(abs(I2 - self.C2_ref) / (abs(self.C2_ref) + 1e-30)),
            'C3_drift': float(abs(I3 - self.C3_ref) / (abs(self.C3_ref) + 1e-30)),
        }

        print(f"L2_rel={l2_err:.6e}, Linf={linf_err:.6e}")
        print(f"C1 drift={results['C1_drift']:.6e}, "
              f"C2 drift={results['C2_drift']:.6e}, "
              f"C3 drift={results['C3_drift']:.6e}")

        json.dump(results, open(os.path.join(self.out_dir, "results.json"), "w"), indent=2)
        return results

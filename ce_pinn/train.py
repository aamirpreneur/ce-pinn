"""Training loop for CE-PINN with W&B logging and proper checkpointing."""
import os
import json
import time
import math
import torch
import numpy as np
from . import projection as proj

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def latin_hypercube_2d(n, bounds):
    rng = np.random.default_rng()
    samples = np.zeros((n, 2))
    for j in range(2):
        perm = rng.permutation(n)
        samples[:, j] = (perm + rng.uniform(size=n)) / n
    lo = np.array([bounds[0], bounds[2]])
    hi = np.array([bounds[1], bounds[3]])
    return samples * (hi - lo) + lo


def latin_hypercube_3d(n, bounds):
    rng = np.random.default_rng()
    samples = np.zeros((n, 3))
    for j in range(3):
        perm = rng.permutation(n)
        samples[:, j] = (perm + rng.uniform(size=n)) / n
    lo = np.array([bounds[0], bounds[2], bounds[4]])
    hi = np.array([bounds[1], bounds[3], bounds[5]])
    return samples * (hi - lo) + lo


def pde_residual_kdv(model, x, t):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)
    u_x, u_t = torch.autograd.grad(u, [x, t], grad_outputs=torch.ones_like(u),
                                    create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u_xx),
                                create_graph=True)[0]
    return u_t + 6.0 * u * u_x + u_xxx


def pde_residual_rlw(model, x, t, eps=1.0, mu=1.0):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xt = torch.autograd.grad(u_x, t, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_xxt = torch.autograd.grad(u_xt, x, grad_outputs=torch.ones_like(u_xt), create_graph=True)[0]
    return u_t + u_x + eps * u * u_x - mu * u_xxt


def pde_residual_kp(model, x, y, t, sigma=-3.0):
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
    inner_x = torch.autograd.grad(inner, x, grad_outputs=torch.ones_like(inner), create_graph=True)[0]
    return inner_x + sigma * u_yy


class Trainer:
    """CE-PINN trainer with W&B logging and proper checkpointing."""

    def __init__(self, model, config, device='cpu', out_dir=None):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.out_dir = out_dir or os.environ.get("OUT_DIR", "results")
        os.makedirs(self.out_dir, exist_ok=True)

        self.history = []
        self.best_loss = float('inf')
        self.global_step = 0
        self.use_wandb = HAS_WANDB and os.environ.get("WANDB_API_KEY")

        self._setup_data()
        self._compute_reference_invariants()
        self._save_config()

    def _save_config(self):
        cfg = {k: v for k, v in self.config.items() if not callable(v)}
        cfg['C1_ref'] = float(self.C1_ref)
        cfg['C2_ref'] = float(self.C2_ref)
        cfg['C3_ref'] = float(self.C3_ref)
        cfg['device'] = str(self.device)
        json.dump(cfg, open(os.path.join(self.out_dir, "config.json"), "w"), indent=2)

    def _setup_data(self):
        cfg = self.config
        bounds = [cfg['xa'], cfg['xb'], 0.0, cfg['T']]
        pts_pde = latin_hypercube_2d(cfg['n_pde'], bounds)
        self.x_pde = torch.tensor(pts_pde[:, 0], dtype=torch.float64, device=self.device)
        self.t_pde = torch.tensor(pts_pde[:, 1], dtype=torch.float64, device=self.device)

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

        if cfg['equation'] == 'rlw':
            mu = cfg['mu']
            self.C1_ref = float(_trapz(ic_vals, dx=dx))
            self.C2_ref = float(_trapz(ic_vals**2 + mu * ic_x**2, dx=dx))
            self.C3_ref = float(_trapz(ic_vals**3 + 3.0 * ic_vals**2, dx=dx))
        else:
            self.C1_ref = float(_trapz(ic_vals, dx=dx))
            self.C2_ref = float(_trapz(ic_vals**2, dx=dx))
            self.C3_ref = float(_trapz(ic_vals**3 - 0.5 * ic_x**2, dx=dx))

        print(f"Reference invariants: C1={self.C1_ref:.8f}, C2={self.C2_ref:.8f}, "
              f"C3={self.C3_ref:.8f}")

    def _compute_conservation_at_t(self, t_val):
        """Compute I1, I2, I3 at a given time."""
        t_vec = torch.full_like(self.x_cons, t_val)
        with torch.no_grad():
            u = self.model(self.x_cons, t_vec)
        u_np = u.cpu().numpy()
        ux_np = np.gradient(u_np, self.dx_cons)
        dx = self.dx_cons

        cfg = self.config
        if cfg['equation'] == 'rlw':
            mu = cfg['mu']
            I1 = float(_trapz(u_np, dx=dx))
            I2 = float(_trapz(u_np**2 + mu * ux_np**2, dx=dx))
            I3 = float(_trapz(u_np**3 + 3.0 * u_np**2, dx=dx))
        else:
            I1 = float(_trapz(u_np, dx=dx))
            I2 = float(_trapz(u_np**2, dx=dx))
            I3 = float(_trapz(u_np**3 - 0.5 * ux_np**2, dx=dx))
        return I1, I2, I3

    def _conservation_loss(self, t_samples):
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

    def _log_step(self, phase, step, total, loss_pde, loss_total):
        self.global_step += 1
        entry = {
            'global_step': self.global_step,
            'phase': phase, 'step': step,
            'loss_pde': loss_pde, 'loss_total': loss_total,
        }

        if self.global_step % 500 == 0:
            I1, I2, I3 = self._compute_conservation_at_t(self.config['T'] / 2.0)
            entry['I1_mid'] = I1
            entry['I2_mid'] = I2
            entry['I3_mid'] = I3
            entry['I1_drift'] = abs(I1 - self.C1_ref) / (abs(self.C1_ref) + 1e-30)
            entry['I2_drift'] = abs(I2 - self.C2_ref) / (abs(self.C2_ref) + 1e-30)
            entry['I3_drift'] = abs(I3 - self.C3_ref) / (abs(self.C3_ref) + 1e-30)

        self.history.append(entry)

        if self.use_wandb:
            wandb.log({k: v for k, v in entry.items() if k != 'phase'})

        if loss_pde < self.best_loss and not math.isnan(loss_pde):
            self.best_loss = loss_pde
            torch.save(self.model.state_dict(),
                       os.path.join(self.out_dir, "model_best.pth"))

        print(f"[{phase} {step}/{total}] PDE={loss_pde:.6e} Total={loss_total:.6e}", flush=True)

    def _save_history(self):
        json.dump(self.history, open(os.path.join(self.out_dir, "history.json"), "w"))

    def _save_checkpoint(self, tag):
        path = os.path.join(self.out_dir, f"model_{tag}.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
        }, path)

    def train_adam(self, n_iter, lr=1e-3, lam_cons=0.0, log_every=500):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        cfg = self.config

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
                self._log_step('Adam', step, n_iter, loss_pde.item(), loss.item())

            if n_iter >= 5 and step % (n_iter // 5) == 0:
                self._save_checkpoint(f"adam_{step}")

        self._save_history()

    def train_lbfgs(self, n_iter, lr=1.0, lam_cons=0.0, log_every=100):
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr,
                                       max_iter=1, history_size=50,
                                       line_search_fn='strong_wolfe')
        cfg = self.config

        for step in range(1, n_iter + 1):
            loss_val = [None, None]

            def closure():
                optimizer.zero_grad()
                loss_pde = self._pde_loss()
                loss = loss_pde
                if lam_cons > 0 and cfg.get('projection') is not None:
                    n_t = cfg.get('n_t_conservation', 10)
                    t_samples = torch.rand(n_t, dtype=torch.float64,
                                           device=self.device) * cfg['T']
                    loss = loss + lam_cons * self._conservation_loss(t_samples)
                loss.backward()
                loss_val[0] = loss_pde.item()
                loss_val[1] = loss.item()
                return loss

            optimizer.step(closure)

            if step % log_every == 0 or step == 1:
                self._log_step('LBFGS', step, n_iter, loss_val[0], loss_val[1])

        self._save_checkpoint("final")
        self._save_history()

    def evaluate(self, exact_fn, n_eval=1000):
        cfg = self.config
        x_eval = torch.linspace(cfg['xa'], cfg['xb'], n_eval,
                                dtype=torch.float64, device=self.device)
        t_eval = torch.full_like(x_eval, cfg['T'])

        with torch.no_grad():
            u_pred = self.model(x_eval, t_eval)

        x_np = x_eval.cpu().numpy()
        u_pred_np = u_pred.cpu().numpy()
        u_exact_np = exact_fn(x_np)

        diff = u_pred_np - u_exact_np
        l2_abs = float(np.sqrt(np.mean(diff**2)))
        l2_rel = float(np.sqrt(np.sum(diff**2) / (np.sum(u_exact_np**2) + 1e-30)))
        linf = float(np.max(np.abs(diff)))

        # Conservation at final time
        I1_T, I2_T, I3_T = self._compute_conservation_at_t(cfg['T'])

        results = {
            'L2_absolute': l2_abs,
            'L2_relative': l2_rel,
            'Linf': linf,
            'C1_pred': I1_T, 'C1_ref': self.C1_ref,
            'C2_pred': I2_T, 'C2_ref': self.C2_ref,
            'C3_pred': I3_T, 'C3_ref': self.C3_ref,
            'C1_drift': abs(I1_T - self.C1_ref) / (abs(self.C1_ref) + 1e-30),
            'C2_drift': abs(I2_T - self.C2_ref) / (abs(self.C2_ref) + 1e-30),
            'C3_drift': abs(I3_T - self.C3_ref) / (abs(self.C3_ref) + 1e-30),
            'best_pde_loss': self.best_loss,
            'total_steps': self.global_step,
        }

        print(f"L2_abs={l2_abs:.6e}, L2_rel={l2_rel:.6e}, Linf={linf:.6e}")
        print(f"C1 drift={results['C1_drift']:.6e}, C2 drift={results['C2_drift']:.6e}, "
              f"C3 drift={results['C3_drift']:.6e}")

        json.dump(results, open(os.path.join(self.out_dir, "results.json"), "w"), indent=2)

        if self.use_wandb:
            wandb.log({f"final/{k}": v for k, v in results.items()})

        # Save predicted solution for later plotting
        np.savez(os.path.join(self.out_dir, "solution_final.npz"),
                 x=x_np, u_pred=u_pred_np, u_exact=u_exact_np, t=cfg['T'])

        return results

    def is_run_valid(self):
        """Check if the run produced valid results (for push gating)."""
        if math.isnan(self.best_loss) or math.isinf(self.best_loss):
            return False
        if self.best_loss > 1e10:
            return False
        return True

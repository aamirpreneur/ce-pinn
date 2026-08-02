"""Grid-based trainer for CE-PINN.

The conservation projection is a global operation on a spatial grid, so the PDE
residual is evaluated on that same grid rather than at scattered points. Each
training step samples `n_t` time levels; at every level the network is
evaluated on the full spatial grid, the hard boundary layer is applied, the
projection is applied, and the residual is formed **on the projected field**.
That is what makes conservation part of the model rather than a post-processing
step.

Derivatives:
  * spatial derivatives come from the fourth-order finite-difference operators
    in `operators.py`;
  * the time derivative of the raw field is ordinary automatic differentiation
    (the network is pointwise in t, so one backward pass gives the whole
    vector);
  * the time derivative of the *projected* field follows from the chain rule,
    d(u*)/dt = (dP/du~) du~/dt, applied by `project_tangent`.
"""
import os
import json
import math
import torch
import numpy as np

from . import projection as proj
from .operators import GridOps

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


class GridTrainer:
    """CE-PINN trainer with grid collocation and hard conservation."""

    def __init__(self, model, config, device='cpu', out_dir='results'):
        self.model = model.to(device)
        self.device = device
        self.cfg = config
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.history = []
        self.best_loss = float('inf')
        self.global_step = 0
        self.use_wandb = HAS_WANDB and bool(os.environ.get("WANDB_API_KEY"))
        self._lam_cache = None          # warm start for Newton
        self._newton_failures = 0   # diverged
        self._newton_clamped = 0    # lam2 on its well-posedness bound

        eq = config['equation']
        self.ops = GridOps(config['xa'], config['xb'], config['n_grid'],
                           device=device, need_d3=(eq == 'kdv'))
        self.x = self.ops.x
        self.dx = self.ops.dx
        self.q = self.ops.q
        # second-order operator used *inside* the invariant, matched to the
        # Helmholtz stencil so the discrete constraint is exactly attainable
        self.D1c = proj.central_d1_matrix(config['n_grid'], self.dx,
                                          device=device)

        self._setup_bc()
        self._compute_reference_invariants()
        self._check_boundary_flux()
        self._save_config()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def _setup_bc(self):
        cfg = self.cfg
        ic_np = cfg['ic_fn'](self.x.cpu().numpy())
        self.ic_vals = torch.tensor(ic_np, dtype=torch.float64,
                                    device=self.device)
        # Boundary values default to the initial condition evaluated at the
        # ends. Hardcoding zero is wrong whenever the solution does not decay
        # exactly to zero at the domain edge (and badly wrong for the bore).
        self.ga_fn = cfg.get('ga_fn')
        self.gb_fn = cfg.get('gb_fn')
        self.ga0 = float(self.ic_vals[0])
        self.gb0 = float(self.ic_vals[-1])

    def _ga(self, t):
        return float(self.ga_fn(t)) if self.ga_fn else self.ga0

    def _gb(self, t):
        return float(self.gb_fn(t)) if self.gb_fn else self.gb0

    def _dga(self, t, h=1e-6):
        """g_a'(t), needed by the tangent when the boundary data moves."""
        if self.ga_fn is None:
            return 0.0
        return (float(self.ga_fn(t + h)) - float(self.ga_fn(t - h))) / (2 * h)

    def _dgb(self, t, h=1e-6):
        if self.gb_fn is None:
            return 0.0
        return (float(self.gb_fn(t + h)) - float(self.gb_fn(t - h))) / (2 * h)

    def _invariants(self, u):
        """Discrete (I1, I2, I3) using the same quadrature as the constraints."""
        cfg = self.cfg
        ux = torch.matmul(u, self.D1c.T)
        I1 = torch.dot(self.q, u)
        if cfg['equation'] == 'rlw':
            I2 = torch.dot(self.q, u * u + cfg['mu'] * ux * ux)
            # I3 = int(eps*u^3 + 3u^2). The eps factor is required for
            # eps != 1: writing the equation as (1-mu*d_xx)u_t = -d_x f with
            # f = u + (eps/2)u^2 gives d/dt int(u^2/2 + eps*u^3/6) = 0.
            # Verified numerically against a pseudospectral solve at eps=1.5,
            # where int(u^3+3u^2) drifts by 5e-4 over t=5 and this does not.
            I3 = torch.dot(self.q, cfg['eps'] * u ** 3 + 3.0 * u * u)
        else:
            I2 = torch.dot(self.q, u * u)
            I3 = torch.dot(self.q, u ** 3 - 0.5 * ux * ux)
        return I1, I2, I3

    def _compute_reference_invariants(self):
        with torch.no_grad():
            C1, C2, C3 = self._invariants(self.ic_vals)
        self.C1_ref, self.C2_ref, self.C3_ref = (float(C1), float(C2), float(C3))
        # Targets may be time-dependent. On a finite interval the invariants
        # are only conserved when the boundary flux vanishes; where it does
        # not, projecting onto the *initial* values enforces the wrong target
        # and the exact solution is no longer a fixed point of the projection,
        # which shows up as an additive error floor rather than one that
        # vanishes with the network error. Supplying C1_fn, C2_fn, C3_fn
        # removes it. Absent them the behaviour is unchanged: constants.
        self.C1_fn = self.cfg.get('C1_fn')
        self.C2_fn = self.cfg.get('C2_fn')
        self.C3_fn = self.cfg.get('C3_fn')
        self.C2_poly_fn = self.cfg.get('C2_poly_fn')
        self.time_dependent_targets = any(
            f is not None for f in (self.C1_fn, self.C2_fn, self.C3_fn))
        print(f"Reference invariants: C1={self.C1_ref:.8f}, "
              f"C2={self.C2_ref:.8f}, C3={self.C3_ref:.8f}")
        if self.time_dependent_targets:
            names = [n for n, f in (('C1', self.C1_fn), ('C2', self.C2_fn),
                                    ('C3', self.C3_fn)) if f is not None]
            print(f"Time-dependent targets supplied for: {', '.join(names)}")
        print(f"Boundary values: ga={self.ga0:.6e}, gb={self.gb0:.6e}")

    def _C1(self, t):
        return float(self.C1_fn(t)) if self.C1_fn else self.C1_ref

    def _C2(self, t):
        return float(self.C2_fn(t)) if self.C2_fn else self.C2_ref

    def _C3(self, t):
        return float(self.C3_fn(t)) if self.C3_fn else self.C3_ref

    def _C2poly(self, t):
        return (float(self.C2_poly_fn(t)) if self.C2_poly_fn
                else self.cfg['C2_poly_ref'])

    def _check_boundary_flux(self):
        """Warn if the invariants are not actually conserved on this domain.

        On a finite interval the conservation laws carry boundary flux terms.
        For RLW, integrating the equation gives

            d/dt int_a^b u dx = -[ u + (eps/2) u^2 - mu u_xt ]_a^b,

        and for KdV, -[ 3u^2 + u_xx ]_a^b. These vanish when the solution
        decays to zero at both ends, which is the soliton case. They do not
        vanish for the undular bore, whose left boundary is held at u0 = 0.1:
        mass is injected steadily and C1 is not constant.

        Enforcing a constant C1 in that situation projects onto the wrong
        target. Such a case needs time-dependent targets C1(t), C2(t) built
        from the flux, which is not implemented here, so this only reports
        the problem rather than hiding it.
        """
        cfg = self.cfg
        ga, gb = self.ga0, self.gb0
        if cfg['equation'] == 'rlw':
            # u_xt is taken as zero, valid for a steady boundary state
            fa = ga + 0.5 * cfg['eps'] * ga * ga
            fb = gb + 0.5 * cfg['eps'] * gb * gb
        else:
            fa, fb = 3.0 * ga * ga, 3.0 * gb * gb
        dC1dt = -(fb - fa)
        rel = abs(dC1dt) * cfg['T'] / (abs(self.C1_ref) + 1e-30)
        self.flux_dC1dt = dC1dt
        # 0.1% over the whole run: below this the soliton tails are simply
        # not exactly zero at the truncated domain edge, which is harmless.
        if rel > 1e-3 and not self.time_dependent_targets:
            print(f"WARNING: boundary flux is non-zero. dC1/dt = {dC1dt:.4e}, "
                  f"which moves C1 by {100*rel:.1f}% over T={cfg['T']}.")
            print("         The invariants are NOT conserved on this domain, so "
                  "projecting onto\n         the initial C1 and C2 enforces the "
                  "wrong target. Supply C1_fn,\n         C2_fn in the config to "
                  "track them.")
        elif rel > 1e-3:
            print(f"Boundary flux is non-zero (dC1/dt = {dC1dt:.2e}) but "
                  f"time-dependent targets\nare supplied, so the moving value "
                  f"is tracked rather than assumed constant.")
        else:
            print(f"Boundary flux negligible (dC1/dt = {dC1dt:.2e}); "
                  f"invariants are conserved.")

    def _save_config(self):
        c = {k: v for k, v in self.cfg.items() if not callable(v)}
        # Callables are dropped from the record, so note which were supplied --
        # otherwise a stored config cannot be told apart from one that used
        # constant targets or frozen boundary data.
        c.update(C1_ref=self.C1_ref, C2_ref=self.C2_ref, C3_ref=self.C3_ref,
                 ga0=self.ga0, gb0=self.gb0, device=str(self.device),
                 time_dependent_targets=self.time_dependent_targets,
                 supplied_fns=sorted(k for k in ('ga_fn', 'gb_fn', 'C1_fn',
                                                 'C2_fn', 'C3_fn',
                                                 'C2_poly_fn')
                                     if self.cfg.get(k) is not None))
        with open(os.path.join(self.out_dir, "config.json"), "w") as f:
            json.dump(c, f, indent=2)

    # ------------------------------------------------------------------
    # forward: one time level
    # ------------------------------------------------------------------

    def _fields_at(self, t_val, project=True):
        """Return (u_star, u_star_t, state) at a single time level."""
        N = self.ops.N
        t_vec = torch.full((N,), float(t_val), dtype=torch.float64,
                           device=self.device, requires_grad=True)
        u_tilde = self.model(self.x, t_vec)
        # the network is pointwise in t, so one backward pass gives the whole
        # vector d(u~)/dt
        u_tilde_t = torch.autograd.grad(
            u_tilde, t_vec, grad_outputs=torch.ones_like(u_tilde),
            create_graph=True)[0]

        kind = self.cfg.get('projection')
        if not project or kind is None:
            return u_tilde, u_tilde_t, None

        if kind == 'helmholtz':
            u_star, st = proj.helmholtz_projection(
                u_tilde, self._C1(t_val), self._C2(t_val), self.cfg['mu'],
                self.dx, self.q, self.D1c,
                ga=self._ga(t_val), gb=self._gb(t_val),
                tol=self.cfg.get('newton_tol', 1e-12),
                max_iter=self.cfg.get('newton_max_iter', 50))
            # No cross-step multiplier cache: lam2 varies strongly with t
            # (roughly 0 at t=0 rising to tens at late times), so a value
            # cached from another time level is a worse start than the
            # analytic guess computed inside the solver.
            if st.residual >= max(self.cfg.get('newton_tol', 1e-12), 1e-10):
                if st.clamped:
                    self._newton_clamped += 1
                else:
                    self._newton_failures += 1
        elif kind == 'polynomial':
            u_star, st = proj.polynomial_projection(
                u_tilde, self._C1(t_val), self._C2poly(t_val), self.q)
        else:
            raise ValueError(f"unknown projection {kind!r}")

        u_star_t = proj.project_tangent(u_tilde_t, st,
                                        dga=self._dga(t_val),
                                        dgb=self._dgb(t_val))
        return u_star, u_star_t, st

    def _residual(self, u, u_t):
        cfg = self.cfg
        ux = self.ops.d1(u)
        if cfg['equation'] == 'rlw':
            uxxt = self.ops.d2(u_t)
            return u_t + ux + cfg['eps'] * u * ux - cfg['mu'] * uxxt
        if cfg['equation'] == 'kdv':
            return u_t + 6.0 * u * ux + self.ops.d3(u)
        raise ValueError(cfg['equation'])

    # ------------------------------------------------------------------
    # loss
    # ------------------------------------------------------------------

    def _loss(self, lam_soft_i3=0.0):
        cfg = self.cfg
        n_t = cfg.get('n_t', 10)
        skip = cfg.get('edge_skip', 4)
        T = cfg['T']

        t_samples = (torch.rand(n_t, dtype=torch.float64) * T).tolist()

        loss_pde = torch.zeros((), dtype=torch.float64, device=self.device)
        loss_i3 = torch.zeros((), dtype=torch.float64, device=self.device)

        for t_val in t_samples:
            u, u_t, _ = self._fields_at(t_val)
            r = self._residual(u, u_t)
            loss_pde = loss_pde + (r[skip:-skip] ** 2).mean()
            if lam_soft_i3 > 0:
                _, _, I3 = self._invariants(u)
                # Relative penalty so the weight does not have to absorb the
                # scale of C3, guarded because C3 can legitimately vanish for
                # some initial data (it is 0.2, 2.58 and 6.6 in the present
                # test problems, but nothing forces it to be non-zero).
                C3_t = self._C3(t_val)
                scale = abs(C3_t) + 1e-12
                loss_i3 = loss_i3 + ((I3 - C3_t) / scale) ** 2

        loss_pde = loss_pde / n_t
        total = loss_pde + lam_soft_i3 * loss_i3 / n_t

        # Vanilla has no hard boundary layer, so the initial and boundary
        # conditions must be imposed softly. Without them the PDE residual
        # alone is minimised by u == 0, which solves every equation here.
        loss_ibc = torch.zeros((), dtype=torch.float64, device=self.device)
        if cfg.get('soft_ibc', False):
            zeros = torch.zeros_like(self.x)
            u0 = self.model(self.x, zeros)
            loss_ibc = ((u0 - self.ic_vals) ** 2).mean()
            tb = torch.tensor(t_samples, dtype=torch.float64,
                              device=self.device)
            xa = torch.full_like(tb, cfg['xa'])
            xb = torch.full_like(tb, cfg['xb'])
            ua, ub = self.model(xa, tb), self.model(xb, tb)
            loss_ibc = loss_ibc + ((ua - self.ga0) ** 2).mean() \
                                + ((ub - self.gb0) ** 2).mean()
            total = total + cfg.get('lam_ibc', 1.0) * loss_ibc

        return total, loss_pde.detach(), loss_ibc.detach()

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _diagnostics(self, t_val):
        """Invariant drift and boundary error at one time, on the output field."""
        N = self.ops.N
        t_vec = torch.full((N,), float(t_val), dtype=torch.float64,
                           device=self.device)
        u_tilde = self.model(self.x, t_vec)
        kind = self.cfg.get('projection')
        if kind == 'helmholtz':
            u = proj.helmholtz_projection(
                u_tilde, self._C1(t_val), self._C2(t_val), self.cfg['mu'],
                self.dx, self.q, self.D1c,
                ga=self._ga(t_val), gb=self._gb(t_val))[0]
        elif kind == 'polynomial':
            u = proj.polynomial_projection(
                u_tilde, self._C1(t_val), self._C2poly(t_val), self.q)[0]
        else:
            u = u_tilde
        I1, I2, I3 = self._invariants(u)
        # Drift is measured against the target *at this time*, so with
        # time-dependent targets it reports failure to track a moving value
        # rather than failure to stay constant.
        refs = (self._C1(t_val), self._C2(t_val), self._C3(t_val))
        drift = [abs(float(I) - R) / (abs(R) + 1e-30)
                 for I, R in zip((I1, I2, I3), refs)]
        bc_err = max(abs(float(u[0]) - self._ga(t_val)),
                     abs(float(u[-1]) - self._gb(t_val)))
        return drift, bc_err, u

    def _track_best(self, loss_pde):
        """Record a new minimum and checkpoint it. Called every step."""
        if loss_pde < self.best_loss and not math.isnan(loss_pde):
            self.best_loss = loss_pde
            torch.save(self.model.state_dict(),
                       os.path.join(self.out_dir, "model_best.pth"))

    def _log(self, phase, step, total, loss_total, loss_pde, loss_ibc, lr=None):
        self.global_step += 1
        e = {'global_step': self.global_step, 'phase': phase, 'step': step,
             'loss_total': loss_total, 'loss_pde': loss_pde}
        if loss_ibc:
            e['loss_ibc'] = loss_ibc
        if lr is not None:
            e['lr'] = lr
        if self.global_step % self.cfg.get('diag_every', 20) == 0:
            drift, bc_err, _ = self._diagnostics(self.cfg['T'] / 2.0)
            e.update(I1_drift=drift[0], I2_drift=drift[1], I3_drift=drift[2],
                     bc_err=bc_err)
        self.history.append(e)
        if self.use_wandb:
            wandb.log({k: v for k, v in e.items() if k != 'phase'})
        self._track_best(loss_pde)
        msg = f"[{phase} {step}/{total}] total={loss_total:.6e} pde={loss_pde:.6e}"
        if lr is not None:
            msg += f" lr={lr:.2e}"
        if self._newton_failures:
            msg += f" newton_fail={self._newton_failures}"
        if self._newton_clamped:
            msg += f" clamped={self._newton_clamped}"
        if 'I2_drift' in e:
            msg += f" I2drift={e['I2_drift']:.3e} bc={e['bc_err']:.2e}"
        print(msg, flush=True)

    # ------------------------------------------------------------------
    # optimisation
    # ------------------------------------------------------------------

    def train_adam(self, n_iter, lr=1e-3, lam_soft_i3=0.0, log_every=100,
                   lr_final=None):
        """Adam with a cosine schedule from `lr` down to `lr_final`.

        A constant rate does not settle. Adam normalises by gradient RMS, so it
        keeps moving about `lr` per step in parameter space no matter how small
        the gradients have become, and the residual then oscillates instead of
        converging. Measured on a reduced RLW1 (tests/diag_lr_stability.py),
        over 2500 steps from an identical initialisation:

            constant 1e-3        best 2.49e-08   final 4.01e-07   2 clamp events
            cosine 1e-3 -> 1e-5  best 1.12e-08   final 3.11e-08   0 clamp events

        The constant run spikes to 2.8e-06 and never settles; the cosine run
        descends monotonically and ends 13x lower. The full-scale run showed the
        same failure, reaching 1.68e-08 at step 3600 and then degrading to
        2.79e-06.

        The clamp counts are the tell. Newton hits its well-posedness bound only
        when the optimiser has thrown the network somewhere bad, so a schedule
        that stops throwing removes those events entirely. The clamp was a
        symptom, not the cause.

        Pass `lr_final=lr` to recover the old constant behaviour.
        """
        if lr_final is None:
            lr_final = lr * 1e-2
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        for step in range(1, n_iter + 1):
            frac = (step - 1) / max(n_iter - 1, 1)
            cur = lr_final + 0.5 * (lr - lr_final) * (1.0 + math.cos(math.pi * frac))
            for g in opt.param_groups:
                g['lr'] = cur
            opt.zero_grad()
            total, lp, li = self._loss(lam_soft_i3)
            total.backward()
            opt.step()
            # Track the minimum on every step, not only on logging steps.
            # Otherwise `model_best.pth` is the best of every log_every-th
            # step, which with log_every=200 can miss the real minimum by a
            # wide margin on a run that oscillates.
            self._track_best(float(lp.detach()))
            if step % log_every == 0 or step == 1:
                self._log('Adam', step, n_iter, float(total.detach()),
                          float(lp.detach()), float(li.detach()), lr=cur)
        self._save_history()

    def train_lbfgs(self, n_iter, lr=1.0, lam_soft_i3=0.0, log_every=20):
        opt = torch.optim.LBFGS(self.model.parameters(), lr=lr, max_iter=1,
                                history_size=50, line_search_fn='strong_wolfe')
        box = [0.0, 0.0, 0.0]

        def closure():
            opt.zero_grad()
            total, lp, li = self._loss(lam_soft_i3)
            total.backward()
            # detach before float(): these still carry grad history at this
            # point, and float() on such a tensor raises a UserWarning
            box[0], box[1], box[2] = (float(total.detach()), float(lp.detach()),
                                      float(li.detach()))
            return total

        for step in range(1, n_iter + 1):
            opt.step(closure)
            if step % log_every == 0 or step == 1:
                self._log('LBFGS', step, n_iter, box[0], box[1], box[2])
        self._save_checkpoint('final')
        self._save_history()

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------

    def _save_history(self):
        with open(os.path.join(self.out_dir, "history.json"), "w") as f:
            json.dump(self.history, f)

    def _save_checkpoint(self, tag):
        torch.save({'model_state_dict': self.model.state_dict(),
                    'global_step': self.global_step,
                    'best_loss': self.best_loss},
                   os.path.join(self.out_dir, f"model_{tag}.pth"))

    def evaluate(self, exact_fn=None):
        T = self.cfg['T']
        drift, bc_err, u = self._diagnostics(T)
        u_np = u.cpu().numpy()
        x_np = self.x.cpu().numpy()

        res = {'C1_drift': drift[0], 'C2_drift': drift[1], 'C3_drift': drift[2],
               'bc_error': bc_err, 'best_pde_loss': self.best_loss,
               'total_steps': self.global_step,
               'newton_failures': self._newton_failures,
               'newton_clamped': self._newton_clamped,
               'C1_ref': self.C1_ref, 'C2_ref': self.C2_ref,
               'C3_ref': self.C3_ref}

        if exact_fn is not None:
            u_ex = exact_fn(x_np)
            d = u_np - u_ex
            res['L2_absolute'] = float(np.sqrt(np.mean(d ** 2)))
            res['L2_relative'] = float(
                np.sqrt(np.sum(d ** 2) / (np.sum(u_ex ** 2) + 1e-30)))
            res['Linf'] = float(np.max(np.abs(d)))
            np.savez(os.path.join(self.out_dir, "solution_final.npz"),
                     x=x_np, u_pred=u_np, u_exact=u_ex, t=T)
            print(f"L2_abs={res['L2_absolute']:.6e}  "
                  f"L2_rel={res['L2_relative']:.6e}  Linf={res['Linf']:.6e}")
        else:
            np.savez(os.path.join(self.out_dir, "solution_final.npz"),
                     x=x_np, u_pred=u_np, t=T)

        print(f"drift: I1={drift[0]:.3e} I2={drift[1]:.3e} I3={drift[2]:.3e}"
              f"   bc_err={bc_err:.3e}")
        json.dump(res, open(os.path.join(self.out_dir, "results.json"), "w"),
                  indent=2)
        if self.use_wandb:
            wandb.log({f"final/{k}": v for k, v in res.items()})
        return res

    def is_run_valid(self):
        """Only rejects NaN/Inf. High-loss runs are still useful for ablation."""
        return not (math.isnan(self.best_loss) or math.isinf(self.best_loss))

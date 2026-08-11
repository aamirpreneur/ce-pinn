"""Does a constant Adam learning rate destabilize the run late in training?

The RLW1 sweep reached 1.68e-08 at step 3600, then blew up to 2.79e-06 by step
4400 and only partially recovered. The clamp was ruled out as the cause
(diag_lam2_convergence.py: near convergence lam2 ~ 1e-05 against a clamp at
-1.4e-02, and the adjoint holds at 1e-09 there).

The remaining suspect is the optimizer. Adam normalises by gradient RMS, so
once the residual is tiny it still moves roughly `lr` per step in parameter
space. With lr fixed at 1e-3 for 20000 steps that is a large step relative to
how converged the network is.

This runs the same reduced configuration three times, changing only the
schedule, and records the best loss reached, the loss at the end, and how far
the run drifted from its own best. A constant schedule that peaks and then
degrades, next to decayed schedules that do not, implicates the schedule.

Run:  py -3 tests/diag_lr_stability.py
"""
import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from ce_pinn.equations import rlw_soliton, rlw_soliton_torch
from ce_pinn.network import MLP, HardBC1D
from ce_pinn.train import GridTrainer

torch.set_default_dtype(torch.float64)

EPS, MU, C_AMP, X0 = 1.0, 1.0, 0.1, 0.0
XA, XB, T = -40.0, 60.0, 20.0

# reduced so three runs finish on CPU; the question is the shape of the loss
# curve late in training, not the absolute level
NGRID, NT, STEPS = 161, 4, 2500
LAYERS, WIDTH = 5, 30

ic_np = lambda x: rlw_soliton(x, 0.0, eps=EPS, mu=MU, c=C_AMP, x0=X0)
ic_t = lambda x: rlw_soliton_torch(x, torch.zeros_like(x), eps=EPS, mu=MU,
                                   c=C_AMP, x0=X0)


def build(out_dir):
    torch.manual_seed(0)
    np.random.seed(0)
    mlp = MLP(d_in=2, d_out=1, hidden_layers=LAYERS, hidden_width=WIDTH)
    model = HardBC1D(mlp, ic_t, XA, XB, T=T)
    cfg = dict(equation='rlw', eps=EPS, mu=MU, xa=XA, xb=XB, T=T,
               ic_fn=ic_np, n_grid=NGRID, n_t=NT, projection='helmholtz',
               diag_every=10 ** 9)
    return GridTrainer(model, cfg, device='cpu', out_dir=out_dir)


def run(name, lr_at):
    """lr_at(step, total) -> learning rate. Returns the loss history."""
    tr = build(os.path.join(os.path.dirname(__file__), '_lr', name))
    opt = torch.optim.Adam(tr.model.parameters(), lr=lr_at(0, STEPS))
    hist = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        for g in opt.param_groups:
            g['lr'] = lr_at(step, STEPS)
        opt.zero_grad()
        total, lp, _ = tr._loss(0.0)
        total.backward()
        opt.step()
        hist.append(float(lp))
    return hist, time.time() - t0, tr


SCHEDULES = {
    'constant 1e-3': lambda s, n: 1e-3,
    'cosine 1e-3->1e-5': lambda s, n: 1e-5 + 0.5 * (1e-3 - 1e-5)
                                       * (1 + math.cos(math.pi * s / n)),
    'step decay /10 at 50%,75%': lambda s, n: 1e-3 * (0.1 if s > 0.5 * n else 1.0)
                                              * (0.1 if s > 0.75 * n else 1.0),
}

print(f"grid={NGRID}  n_t={NT}  steps={STEPS}  net={LAYERS}x{WIDTH}  seed=0")
print("identical initialization and data in all three; only the schedule differs\n")

results, traces = {}, {}
for _i, (name, sched) in enumerate(SCHEDULES.items()):
    hist, secs, tr = run(f'sched{_i}', sched)
    traces[name] = hist
    best = min(hist)
    best_at = hist.index(best) + 1
    final = float(np.mean(hist[-50:]))          # average out step-to-step noise
    tail_max = max(hist[best_at:]) if best_at < len(hist) else best
    results[name] = dict(best=best, best_at=best_at, final=final,
                         tail_max=tail_max, blowup=tail_max / best,
                         clamped=tr._newton_clamped, secs=secs)
    print(f"{name:28} best {best:.3e} @{best_at:5d}   final {final:.3e}   "
          f"worst-after-best {tail_max:.3e}   ratio {tail_max/best:8.1f}x   "
          f"clamped {tr._newton_clamped:3d}   {secs:.0f}s")

print("\nloss trace (mean over the preceding 25 steps)")
print(f"{'step':>6}" + "".join(f"{n[:17]:>19}" for n in SCHEDULES))
for k in range(249, STEPS, 250):
    row = f"{k+1:6d}"
    for name in SCHEDULES:
        h = traces[name]
        row += f"{np.mean(h[max(0, k - 24):k + 1]):19.3e}"
    print(row)

print("\n" + "=" * 74)
c = results['constant 1e-3']
others = [v for k, v in results.items() if k != 'constant 1e-3']
print(f"constant schedule degraded by {c['blowup']:.1f}x after its best")
for k, v in results.items():
    if k == 'constant 1e-3':
        continue
    print(f"  {k:28} degraded by {v['blowup']:.1f}x, best {v['best']:.3e}")

worst_other = max(v['blowup'] for v in others)
best_other = min(v['best'] for v in others)
print()
if c['blowup'] > 5 * worst_other:
    print("CONFIRMED: the constant schedule destabilises and the decayed ones do not.")
elif c['blowup'] > 3.0 and worst_other > 3.0:
    print("NOT CONFIRMED: every schedule degrades, so the cause is not the schedule.")
else:
    print("INCONCLUSIVE at this scale; see the numbers above.")
print(f"lowest loss reached: constant {c['best']:.3e}  vs  decayed {best_other:.3e}")

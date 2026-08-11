"""End-to-end check of the grid trainer for every variant.

Runs a handful of Adam steps on CPU. The point is not accuracy, it is that
the pipeline runs and that each variant behaves the way the paper claims:

  vanilla   soft IC/BC, no projection      -> must not collapse to u == 0
  hard_bc   hard IC/BC, no projection      -> I2 drifts freely
  ce_poly   hard IC/BC, affine projection  -> I2 still drifts (wrong invariant)
  ce_helm   hard IC/BC, Helmholtz          -> I2 drift at machine precision

Run:  py -3 tests/test_trainer.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from ce_pinn.equations import rlw_soliton, rlw_soliton_torch
from ce_pinn.network import MLP, HardBC1D, VanillaNet
from ce_pinn.train import GridTrainer
from ce_pinn.projection import central_d1_matrix
from ce_pinn.operators import trapezoid_weights

torch.set_default_dtype(torch.float64)

EPS, MU, C_AMP, X0 = 1.0, 1.0, 0.1, 0.0
XA, XB, T = -40.0, 60.0, 20.0
NGRID, NT, STEPS = 201, 4, 30

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ''))


def ic_np(x):
    return rlw_soliton(x, 0.0, eps=EPS, mu=MU, c=C_AMP, x0=X0)


def ic_torch(x):
    return rlw_soliton_torch(x, torch.zeros_like(x), eps=EPS, mu=MU,
                             c=C_AMP, x0=X0)


def exact_T(x):
    return rlw_soliton(x, T, eps=EPS, mu=MU, c=C_AMP, x0=X0)


# reference value of the *polynomial* invariant, which is what the affine
# projection is designed for
_dx = (XB - XA) / (NGRID - 1)
_q = trapezoid_weights(NGRID, _dx)
_x = torch.linspace(XA, XB, NGRID)
_ic = ic_torch(_x)
C2_POLY = float(torch.dot(_q, _ic * _ic))


def build(variant, out_dir):
    torch.manual_seed(0)
    mlp = MLP(d_in=2, d_out=1, hidden_layers=4, hidden_width=30)
    if variant == 'vanilla':
        model, projection, soft_ibc = VanillaNet(mlp), None, True
    elif variant == 'hard_bc':
        model, projection, soft_ibc = HardBC1D(mlp, ic_torch, XA, XB, T=T), None, False
    elif variant == 'ce_poly':
        model, projection, soft_ibc = HardBC1D(mlp, ic_torch, XA, XB, T=T), 'polynomial', False
    elif variant == 'ce_helm':
        model, projection, soft_ibc = HardBC1D(mlp, ic_torch, XA, XB, T=T), 'helmholtz', False
    else:
        raise ValueError(variant)

    cfg = dict(equation='rlw', eps=EPS, mu=MU, xa=XA, xb=XB, T=T,
               ic_fn=ic_np, n_grid=NGRID, n_t=NT, projection=projection,
               soft_ibc=soft_ibc, lam_ibc=10.0, C2_poly_ref=C2_POLY,
               diag_every=10 ** 9)      # silence per-step diagnostics
    return GridTrainer(model, cfg, device='cpu', out_dir=out_dir)


TMP = os.path.join(os.path.dirname(__file__), '_tmp')
results = {}

for variant in ('vanilla', 'hard_bc', 'ce_poly', 'ce_helm'):
    print(f"\n--- {variant} ---")
    t0 = time.time()
    tr = build(variant, os.path.join(TMP, variant))
    tr.train_adam(STEPS, lr=1e-3, log_every=STEPS)
    dt = time.time() - t0
    drift, bc_err, u = tr._diagnostics(T / 2.0)
    amp = float(u.abs().max())
    results[variant] = dict(drift=drift, bc=bc_err, amp=amp, secs=dt,
                            valid=tr.is_run_valid())
    print(f"    I1={drift[0]:.3e}  I2={drift[1]:.3e}  I3={drift[2]:.3e}  "
          f"bc={bc_err:.2e}  max|u|={amp:.4f}  {dt:.1f}s")

print("\n" + "=" * 62)

check("all variants ran without error", len(results) == 4)
check("all runs valid (no NaN/Inf)", all(r['valid'] for r in results.values()))

check("vanilla did not collapse to zero",
      results['vanilla']['amp'] > 0.05,
      f"max|u|={results['vanilla']['amp']:.4f} (soliton amplitude 0.3)")

check("hard_bc leaves I2 drifting",
      results['hard_bc']['drift'][1] > 1e-6,
      f"drift={results['hard_bc']['drift'][1]:.3e}")

check("ce_poly leaves I2 drifting (enforces the wrong invariant)",
      results['ce_poly']['drift'][1] > 1e-6,
      f"drift={results['ce_poly']['drift'][1]:.3e}")

check("ce_helm holds I2 at machine precision",
      results['ce_helm']['drift'][1] < 1e-11,
      f"drift={results['ce_helm']['drift'][1]:.3e}")

check("ce_helm holds I1 at machine precision",
      results['ce_helm']['drift'][0] < 1e-11,
      f"drift={results['ce_helm']['drift'][0]:.3e}")

check("ce_helm preserves boundary data",
      results['ce_helm']['bc'] < 1e-12,
      f"bc err={results['ce_helm']['bc']:.2e}")

ratio = results['ce_poly']['drift'][1] / max(results['ce_helm']['drift'][1], 1e-300)
print(f"\n  I2 drift ratio ce_poly / ce_helm = {ratio:.2e}")

over = results['ce_helm']['secs'] / results['hard_bc']['secs']
print(f"  Helmholtz overhead vs hard_bc     = {over:.2f}x  "
      f"({results['ce_helm']['secs']:.1f}s vs {results['hard_bc']['secs']:.1f}s)")

print("=" * 62)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print(f"   FAILED: {f}")
    sys.exit(1)
print("all good")

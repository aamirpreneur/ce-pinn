"""RLW-1: Single soliton, eps=1, mu=1, c=0.1, domain [-40,60], T=20.

Tests: Vanilla, Hard-BC, CE-poly, CE-Helm, CE-Full.
Variant controlled via VARIANT env var.
CE-Helm is the MAIN CONTRIBUTION (Helmholtz projection for I2 = int(u^2 + mu*ux^2)).
"""
import os
import sys
import json
import time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
torch.set_default_dtype(torch.float64)

from ce_pinn.equations import rlw_soliton, rlw_soliton_torch
from ce_pinn.network import MLP, HardBC1D
from ce_pinn.train import Trainer

VARIANT = os.environ.get("VARIANT", "ce_helm")
OUT = os.environ.get("OUT_DIR", ".")
SMOKE = os.environ.get("SMOKE", "0") == "1"
os.makedirs(OUT, exist_ok=True)

EPS, MU, C_AMP = 1.0, 1.0, 0.1
X0 = 0.0
XA, XB = -40.0, 60.0
T_FINAL = 20.0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}, Variant: {VARIANT}, Smoke: {SMOKE}", flush=True)


def ic_fn_np(x):
    return rlw_soliton(x, 0.0, eps=EPS, mu=MU, c=C_AMP, x0=X0)


def ic_fn_torch(x):
    return rlw_soliton_torch(x, torch.zeros_like(x), eps=EPS, mu=MU, c=C_AMP, x0=X0)


def exact_at_T(x):
    return rlw_soliton(x, T_FINAL, eps=EPS, mu=MU, c=C_AMP, x0=X0)


hidden_layers = 8
hidden_width = 50
n_pde = 2000 if SMOKE else 20000
n_ic = 500 if SMOKE else 5000
adam_iter = 500 if SMOKE else 30000
lbfgs_iter = 100 if SMOKE else 5000

mlp = MLP(d_in=2, d_out=1, hidden_layers=hidden_layers, hidden_width=hidden_width)

if VARIANT == 'vanilla':
    model = mlp
    projection = None
elif VARIANT in ('hard_bc', 'ce_poly', 'ce_helm', 'ce_full'):
    model = HardBC1D(mlp, ic_fn_torch, XA, XB)
    if VARIANT == 'hard_bc':
        projection = None
    elif VARIANT == 'ce_poly':
        projection = 'polynomial'
    elif VARIANT in ('ce_helm', 'ce_full'):
        projection = 'helmholtz'
else:
    raise ValueError(f"Unknown variant: {VARIANT}")

config = {
    'equation': 'rlw',
    'eps': EPS, 'mu': MU,
    'xa': XA, 'xb': XB, 'T': T_FINAL,
    'ic_fn': ic_fn_np,
    'n_pde': n_pde, 'n_ic': n_ic,
    'n_conservation_grid': 256,
    'n_t_conservation': 10,
    'projection': projection,
    'soft_I3': VARIANT == 'ce_full',
    'ga': 0.0, 'gb': 0.0,
}

trainer = Trainer(model, config, device=device)

t0 = time.time()

lam_cons = 1e-4 if VARIANT in ('ce_poly', 'ce_helm', 'ce_full') else 0.0

print(f"\n=== Adam phase ({adam_iter} iters) ===", flush=True)
hist_adam = trainer.train_adam(adam_iter, lr=1e-3, lam_cons=lam_cons, log_every=500)

print(f"\n=== L-BFGS phase ({lbfgs_iter} iters) ===", flush=True)
hist_lbfgs = trainer.train_lbfgs(lbfgs_iter, lr=1.0, lam_cons=0.0, log_every=100)

elapsed = time.time() - t0
print(f"\nTraining time: {elapsed:.1f}s", flush=True)

print("\n=== Evaluation ===", flush=True)
results = trainer.evaluate(exact_at_T)
results['variant'] = VARIANT
results['training_time_s'] = elapsed
results['smoke'] = SMOKE

json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print(f"\nDone. Results saved to {OUT}/results.json", flush=True)

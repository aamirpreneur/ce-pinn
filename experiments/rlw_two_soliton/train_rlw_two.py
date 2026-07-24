"""RLW-2: Two-soliton collision, eps=1, mu=1, domain [0,120], T=30.

Two-stage curriculum from the RLW paper:
  Stage 1: 50k Adam + 10k L-BFGS (lr=1e-3, no conservation)
  Stage 2: 15k Adam + 5k L-BFGS  (lr=1e-4, with conservation lam=1e-5)
"""
import os
import sys
import json
import time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
torch.set_default_dtype(torch.float64)

from ce_pinn.equations import rlw_two_soliton_ic, rlw_soliton_torch
from ce_pinn.network import MLP, HardBC1D
from ce_pinn.train import Trainer

VARIANT = os.environ.get("VARIANT", "ce_helm")
OUT = os.environ.get("OUT_DIR", ".")
SMOKE = os.environ.get("SMOKE", "0") == "1"
os.makedirs(OUT, exist_ok=True)

EPS, MU = 1.0, 1.0
C1_AMP, C2_AMP = 5.33 / 3.0, 1.69 / 3.0
X1, X2 = 15.0, 35.0
XA, XB = 0.0, 120.0
T_FINAL = 30.0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}, Variant: {VARIANT}, Smoke: {SMOKE}", flush=True)


def ic_fn_np(x):
    return rlw_two_soliton_ic(x, eps=EPS, mu=MU, c1=C1_AMP, c2=C2_AMP, x1=X1, x2=X2)


def ic_fn_torch(x):
    u1 = rlw_soliton_torch(x, torch.zeros_like(x), eps=EPS, mu=MU, c=C1_AMP, x0=X1)
    u2 = rlw_soliton_torch(x, torch.zeros_like(x), eps=EPS, mu=MU, c=C2_AMP, x0=X2)
    return u1 + u2


hidden_layers = 8
hidden_width = 100

if SMOKE:
    s1_adam, s1_lbfgs = 500, 100
    s2_adam, s2_lbfgs = 200, 50
    n_pde, n_ic = 2000, 500
else:
    s1_adam, s1_lbfgs = 50000, 10000
    s2_adam, s2_lbfgs = 15000, 5000
    n_pde, n_ic = 40000, 10000

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
    else:
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

# Stage 1: no conservation
print(f"\n=== Stage 1: Adam ({s1_adam}) ===", flush=True)
trainer.train_adam(s1_adam, lr=1e-3, lam_cons=0.0, log_every=1000)
print(f"\n=== Stage 1: L-BFGS ({s1_lbfgs}) ===", flush=True)
trainer.train_lbfgs(s1_lbfgs, lr=1.0, lam_cons=0.0, log_every=500)

# Stage 2: with conservation
lam_cons = 1e-5 if VARIANT in ('ce_poly', 'ce_helm', 'ce_full') else 0.0
print(f"\n=== Stage 2: Adam ({s2_adam}, lam={lam_cons}) ===", flush=True)
trainer.train_adam(s2_adam, lr=1e-4, lam_cons=lam_cons, log_every=500)
print(f"\n=== Stage 2: L-BFGS ({s2_lbfgs}) ===", flush=True)
trainer.train_lbfgs(s2_lbfgs, lr=1.0, lam_cons=0.0, log_every=200)

elapsed = time.time() - t0
print(f"\nTraining time: {elapsed:.1f}s", flush=True)

# No closed-form at T=30, report conservation only
results = {
    'variant': VARIANT,
    'training_time_s': elapsed,
    'smoke': SMOKE,
    'C1_ref': trainer.C1_ref,
    'C2_ref': trainer.C2_ref,
    'C3_ref': trainer.C3_ref,
}

json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)
torch.save(model.state_dict(), os.path.join(OUT, "model_final.pth"))
print(f"\nDone. Results saved to {OUT}/results.json", flush=True)

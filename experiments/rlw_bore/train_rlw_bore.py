"""RLW-3: Undular bore, eps=1.5, mu=1/6, domain [-36,300], T=250.

Causal time-windowing: N=5 windows of 50 time units each.
Non-homogeneous left BC: u(-36,t) = 0.1.
"""
import os
import sys
import json
import time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
torch.set_default_dtype(torch.float64)

from ce_pinn.equations import rlw_undular_bore_ic, rlw_soliton_torch
from ce_pinn.network import MLP, HardBC1D
from ce_pinn.train import Trainer

VARIANT = os.environ.get("VARIANT", "ce_helm")
OUT = os.environ.get("OUT_DIR", ".")
SMOKE = os.environ.get("SMOKE", "0") == "1"
GENTLE = os.environ.get("GENTLE", "1") == "1"
os.makedirs(OUT, exist_ok=True)

EPS, MU = 1.5, 1.0 / 6.0
U0 = 0.1
XC = 0.0
D = 5.0 if GENTLE else 2.0
XA, XB = -36.0, 300.0
T_FINAL = 250.0
N_WINDOWS = 5
WINDOW_SIZE = T_FINAL / N_WINDOWS

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}, Variant: {VARIANT}, Smoke: {SMOKE}, Gentle: {GENTLE}", flush=True)


def ic_fn_np(x):
    return rlw_undular_bore_ic(x, u0=U0, xc=XC, d=D)


def ic_fn_torch(x):
    return 0.5 * U0 * (1.0 - torch.tanh((x - XC) / D))


def ga_fn(t):
    return torch.full_like(t, U0)


hidden_layers = 8
hidden_width = 100

if SMOKE:
    adam_per_win, lbfgs_per_win = 500, 100
    n_pde, n_ic = 2000, 500
else:
    adam_per_win, lbfgs_per_win = 20000, 5000
    n_pde, n_ic = 40000, 10000

mlp = MLP(d_in=2, d_out=1, hidden_layers=hidden_layers, hidden_width=hidden_width)

if VARIANT == 'vanilla':
    model = mlp
    projection = None
elif VARIANT in ('hard_bc', 'ce_poly', 'ce_helm', 'ce_full'):
    model = HardBC1D(mlp, ic_fn_torch, XA, XB, ga_fn=ga_fn)
    if VARIANT == 'hard_bc':
        projection = None
    elif VARIANT == 'ce_poly':
        projection = 'polynomial'
    else:
        projection = 'helmholtz'
else:
    raise ValueError(f"Unknown variant: {VARIANT}")

t0 = time.time()
all_history = []

for win in range(N_WINDOWS):
    t_start = win * WINDOW_SIZE
    t_end = (win + 1) * WINDOW_SIZE
    print(f"\n{'='*60}", flush=True)
    print(f"Window {win+1}/{N_WINDOWS}: t in [{t_start}, {t_end}]", flush=True)
    print(f"{'='*60}", flush=True)

    config = {
        'equation': 'rlw',
        'eps': EPS, 'mu': MU,
        'xa': XA, 'xb': XB, 'T': t_end,
        'ic_fn': ic_fn_np,
        'n_pde': n_pde, 'n_ic': n_ic,
        'n_conservation_grid': 256,
        'n_t_conservation': 10,
        'projection': projection,
        'soft_I3': VARIANT == 'ce_full',
        'ga': U0, 'gb': 0.0,
    }

    trainer = Trainer(model, config, device=device)
    lam_cons = 1e-5 if VARIANT in ('ce_poly', 'ce_helm', 'ce_full') else 0.0

    trainer.train_adam(adam_per_win, lr=1e-3, lam_cons=lam_cons, log_every=1000)
    trainer.train_lbfgs(lbfgs_per_win, lr=1.0, lam_cons=0.0, log_every=500)

    torch.save(model.state_dict(), os.path.join(OUT, f"model_window_{win+1}.pth"))

elapsed = time.time() - t0
print(f"\nTotal training time: {elapsed:.1f}s", flush=True)

results = {
    'variant': VARIANT,
    'training_time_s': elapsed,
    'smoke': SMOKE,
    'gentle': GENTLE,
    'n_windows': N_WINDOWS,
}

json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)
print(f"\nDone. Results saved to {OUT}/results.json", flush=True)

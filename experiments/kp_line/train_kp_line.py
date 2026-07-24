"""KP-1: Line soliton (KP-II), c=1, alpha=0, domain [-30,50]x[-10,10], T=10."""
import os
import sys
import json
import time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
torch.set_default_dtype(torch.float64)

from ce_pinn.equations import kp_line_soliton, kp_line_soliton_torch
from ce_pinn.network import MLP, HardBC2D
from ce_pinn.projection import polynomial_projection_2d
from ce_pinn.train import latin_hypercube_3d, pde_residual_kp

VARIANT = os.environ.get("VARIANT", "ce_poly")
OUT = os.environ.get("OUT_DIR", ".")
SMOKE = os.environ.get("SMOKE", "0") == "1"
os.makedirs(OUT, exist_ok=True)

K1, M1 = 1.0, 0.0
SIGMA = -3.0
XA, XB = -30.0, 50.0
YA, YB = -10.0, 10.0
T_FINAL = 10.0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}, Variant: {VARIANT}, Smoke: {SMOKE}", flush=True)


def ic_fn_np(x, y):
    return kp_line_soliton(x, y, 0.0, k1=K1, m1=M1, sigma=SIGMA)


def ic_fn_torch(x, y):
    return kp_line_soliton_torch(x, y, torch.zeros_like(x), k1=K1, m1=M1, sigma=SIGMA)


def exact_at_T(x, y):
    return kp_line_soliton(x, y, T_FINAL, k1=K1, m1=M1, sigma=SIGMA)


n_pde = 3000 if SMOKE else 40000
adam_iter = 500 if SMOKE else 30000
lbfgs_iter = 100 if SMOKE else 5000

mlp = MLP(d_in=3, d_out=1, hidden_layers=8, hidden_width=100)

if VARIANT == 'vanilla':
    model = mlp
elif VARIANT in ('hard_bc', 'ce_poly'):
    model = HardBC2D(mlp, ic_fn_torch, XA, XB, YA, YB)

# Collocation points
bounds = [XA, XB, YA, YB, 0.0, T_FINAL]
pts = latin_hypercube_3d(n_pde, bounds)
x_pde = torch.tensor(pts[:, 0], dtype=torch.float64, device=device)
y_pde = torch.tensor(pts[:, 1], dtype=torch.float64, device=device)
t_pde = torch.tensor(pts[:, 2], dtype=torch.float64, device=device)

# Reference invariants from IC
Nx_c, Ny_c = 64, 64
x_c = np.linspace(XA, XB, Nx_c)
y_c = np.linspace(YA, YB, Ny_c)
X_c, Y_c = np.meshgrid(x_c, y_c, indexing='ij')
ic_vals = ic_fn_np(X_c, Y_c)
dx_c = (XB - XA) / (Nx_c - 1)
dy_c = (YB - YA) / (Ny_c - 1)

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
C1_ref = _trapz(_trapz(ic_vals, dx=dy_c, axis=1), dx=dx_c, axis=0)
C2_ref = _trapz(_trapz(ic_vals**2, dx=dy_c, axis=1), dx=dx_c, axis=0)
print(f"Reference invariants: C1={C1_ref:.8f}, C2={C2_ref:.8f}")

model = model.to(device)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
t0 = time.time()

for step in range(1, adam_iter + 1):
    optimizer.zero_grad()
    r = pde_residual_kp(model, x_pde, y_pde, t_pde, sigma=SIGMA)
    loss = (r**2).mean()
    loss.backward()
    optimizer.step()

    if step % 1000 == 0 or step == 1:
        print(f"[Adam {step}/{adam_iter}] PDE={loss.item():.6e}", flush=True)

    if step % (adam_iter // 3) == 0:
        torch.save(model.state_dict(), os.path.join(OUT, f"model_adam_{step}.pth"))

elapsed = time.time() - t0
print(f"\nTraining time: {elapsed:.1f}s", flush=True)

# Evaluation
Nx_e, Ny_e = 128, 64
x_e = np.linspace(XA, XB, Nx_e)
y_e = np.linspace(YA, YB, Ny_e)
X_e, Y_e = np.meshgrid(x_e, y_e, indexing='ij')
u_exact = exact_at_T(X_e, Y_e)

x_et = torch.tensor(X_e.ravel(), dtype=torch.float64, device=device)
y_et = torch.tensor(Y_e.ravel(), dtype=torch.float64, device=device)
t_et = torch.full_like(x_et, T_FINAL)

with torch.no_grad():
    u_pred = model(x_et, y_et, t_et).cpu().numpy().reshape(Nx_e, Ny_e)

l2_err = np.sqrt(np.sum((u_pred - u_exact)**2) / (np.sum(u_exact**2) + 1e-30))
linf_err = np.max(np.abs(u_pred - u_exact))

results = {
    'variant': VARIANT,
    'L2_relative': float(l2_err),
    'Linf': float(linf_err),
    'C1_ref': float(C1_ref),
    'C2_ref': float(C2_ref),
    'training_time_s': elapsed,
    'smoke': SMOKE,
}
print(f"L2_rel={l2_err:.6e}, Linf={linf_err:.6e}")

json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)
torch.save(model.state_dict(), os.path.join(OUT, "model_final.pth"))
print(f"\nDone. Results saved to {OUT}/results.json", flush=True)

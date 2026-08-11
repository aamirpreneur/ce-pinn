"""Where does Newton fail, and where does the time go?"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from ce_pinn.equations import rlw_soliton, rlw_soliton_torch
from ce_pinn.network import MLP, HardBC1D
from ce_pinn.train import GridTrainer
from ce_pinn import projection as proj

torch.set_default_dtype(torch.float64)
EPS, MU, C, X0 = 1.0, 1.0, 0.1, 0.0
XA, XB, T, N = -40.0, 60.0, 20.0, 201

ic_np = lambda x: rlw_soliton(x, 0.0, eps=EPS, mu=MU, c=C, x0=X0)
ic_t = lambda x: rlw_soliton_torch(x, torch.zeros_like(x), eps=EPS, mu=MU, c=C, x0=X0)

torch.manual_seed(0)
mlp = MLP(2, 1, 4, 30)
model = HardBC1D(mlp, ic_t, XA, XB, T=T)
cfg = dict(equation='rlw', eps=EPS, mu=MU, xa=XA, xb=XB, T=T, ic_fn=ic_np,
           n_grid=N, n_t=4, projection='helmholtz', diag_every=10**9)
tr = GridTrainer(model, cfg, out_dir=os.path.join(os.path.dirname(__file__), '_tmp/diag'))

print("\nNewton behavior across a spread of time levels (untrained net):")
print(f"{'t':>7} {'iters':>6} {'residual':>12}  {'lam1':>11} {'lam2':>11}")
for t in [0.0, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0]:
    tv = torch.full((N,), t, dtype=torch.float64)
    with torch.no_grad():
        ut = model(tr.x, tv)
    _, st = proj.helmholtz_projection(ut, tr.C1_ref, tr.C2_ref, MU, tr.dx,
                                      tr.q, tr.D1c, ga=tr.ga0, gb=tr.gb0)
    flag = '' if st.residual < 1e-10 else '   <-- FAIL'
    print(f"{t:7.3f} {st.newton_iters:6d} {st.residual:12.3e}  "
          f"{st.lam[0]:11.3e} {st.lam[1]:11.3e}{flag}")

# how far is the incoming field from the constraint set at t ~ 0?
print("\nIncoming I2 vs target at small t (hard BC pins u to the IC at t=0):")
for t in [0.0, 1e-4, 1e-2, 1.0]:
    tv = torch.full((N,), t, dtype=torch.float64)
    with torch.no_grad():
        ut = model(tr.x, tv)
    i2 = float(proj.invariant_i2(ut, tr.q, tr.D1c, MU))
    print(f"  t={t:<8g} I2[u~]={i2:.8f}  target={tr.C2_ref:.8f}  "
          f"ratio={i2/tr.C2_ref:.6f}")

# timing split
print("\nTiming (200 projections, N=201):")
tv = torch.full((N,), 5.0, dtype=torch.float64)
with torch.no_grad():
    ut = model(tr.x, tv)
t0 = time.time()
for _ in range(200):
    proj.helmholtz_projection(ut, tr.C1_ref, tr.C2_ref, MU, tr.dx, tr.q,
                              tr.D1c, ga=tr.ga0, gb=tr.gb0)
cold = time.time() - t0
_, st = proj.helmholtz_projection(ut, tr.C1_ref, tr.C2_ref, MU, tr.dx, tr.q,
                                  tr.D1c, ga=tr.ga0, gb=tr.gb0)
t0 = time.time()
for _ in range(200):
    proj.helmholtz_projection(ut, tr.C1_ref, tr.C2_ref, MU, tr.dx, tr.q,
                              tr.D1c, ga=tr.ga0, gb=tr.gb0, lam_init=st.lam)
warm = time.time() - t0
print(f"  cold start : {1000*cold/200:.2f} ms/projection ({st.newton_iters} iters)")
print(f"  warm start : {1000*warm/200:.2f} ms/projection")
print(f"  speedup    : {cold/warm:.2f}x")

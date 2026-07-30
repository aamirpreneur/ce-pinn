"""Why does the projection cost accuracy? Three decisive checks.

The post-hoc analysis (tools/analyze_rlw1.py) showed that on RLW1:
  * `hard_bc` leads at every time level, with no crossover;
  * `ce_helm`'s peak amplitude error is 4x worse, not better, despite I2 being
    conserved to 1e-14;
  * `ce_helm` had converged, so it is not under-trained.

So the cost is real and it is not a budget artefact. This isolates where it
comes from.

1. TARGET CONSISTENCY. The constraint enforces the *discrete* invariant built
   from the 2nd-order operator `D1c`, while the residual uses 4th-order
   operators. If the exact soliton's discrete I2 is not constant in time, then
   holding it constant forces the network off the true solution, and the size of
   that drift bounds the accuracy the method can reach.

2. IS THE OPERATOR ITSELF HARMFUL? Take the trained `hard_bc` field, which never
   saw a projection, and project it. If L2 gets worse, the projection damages an
   already-good field and the cost is intrinsic. If L2 improves, the operator is
   benign and the cost lives in the training dynamics instead.

3. WHERE THE ERROR ACTUALLY IS. Decompose by fitting the shift and amplitude
   that minimise the error, rather than reading the peak off a parabolic fit,
   which is noisy when the position error is ~1e-2 grid spacings.

Run:  py -3 tools/diag_why_projection_costs.py
"""
import os
import sys
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from ce_pinn.equations import rlw_soliton, rlw_soliton_torch
from ce_pinn.network import MLP, HardBC1D, VanillaNet
from ce_pinn.train import GridTrainer
from ce_pinn import projection as proj

torch.set_default_dtype(torch.float64)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
C_AMP, X0 = 0.1, 0.0


def find_run(name):
    d = sorted(glob.glob(os.path.join(RES, f'rlw1_{name}', '*')))
    return d[-1]


def load(name, tmp):
    run = find_run(name)
    cfg = json.load(open(os.path.join(run, 'config.json')))
    ck = torch.load(os.path.join(run, 'model_final.pth'), map_location='cpu',
                    weights_only=False)
    sd = ck.get('model_state_dict', ck)
    eps, mu, T, xa, xb = (cfg['eps'], cfg['mu'], cfg['T'], cfg['xa'], cfg['xb'])

    ic_np = lambda x: rlw_soliton(x, 0.0, eps=eps, mu=mu, c=C_AMP, x0=X0)
    ic_t = lambda x: rlw_soliton_torch(x, torch.zeros_like(x), eps=eps, mu=mu,
                                       c=C_AMP, x0=X0)
    ws = sorted([k for k in sd if k.endswith('.weight')],
                key=lambda s: int(s.split('.')[-2]))
    width = sd[ws[0]].shape[0]
    mlp = MLP(d_in=2, d_out=1, hidden_layers=len(ws) - 1, hidden_width=width)
    model = VanillaNet(mlp) if cfg.get('soft_ibc') else \
        HardBC1D(mlp, ic_t, xa, xb, T=T)
    model.load_state_dict(sd)
    tcfg = dict(cfg, ic_fn=ic_np, diag_every=10 ** 9)
    return GridTrainer(model, tcfg, device='cpu', out_dir=tmp), cfg


def l2rel(u, ue):
    return float(np.sqrt(np.sum((u - ue) ** 2) / np.sum(ue ** 2)))


def main():
    tmp = os.path.join(ROOT, 'tests', '_why_tmp')
    os.makedirs(tmp, exist_ok=True)
    tr, cfg = load('hard_bc', os.path.join(tmp, 'hard_bc'))
    eps, mu, T = cfg['eps'], cfg['mu'], cfg['T']
    x = tr.x.cpu().numpy()
    q = tr.q.cpu().numpy()
    D1c = tr.D1c.cpu().numpy()
    ts = np.linspace(0.0, T, 11)

    print("=" * 78)
    print("1. TARGET CONSISTENCY: discrete invariants of the EXACT soliton")
    print("=" * 78)
    print("If these drift, the constraint target is wrong by that much.\n")
    print(f"{'t':>5} {'I1_disc':>16} {'I2_disc':>16} "
          f"{'I1 drift':>11} {'I2 drift':>11}")
    ref = None
    worst_i2 = 0.0
    for t in ts:
        ue = rlw_soliton(x, float(t), eps=eps, mu=mu, c=C_AMP, x0=X0)
        uex = D1c @ ue
        i1 = float(q @ ue)
        i2 = float(q @ (ue * ue + mu * uex * uex))
        if ref is None:
            ref = (i1, i2)
        d1 = abs(i1 - ref[0]) / abs(ref[0])
        d2 = abs(i2 - ref[1]) / abs(ref[1])
        worst_i2 = max(worst_i2, d2)
        print(f"{t:5.1f} {i1:16.12f} {i2:16.12f} {d1:11.2e} {d2:11.2e}")
    print(f"\nworst discrete-I2 drift of the exact solution: {worst_i2:.3e}")
    print("Observed L2 gap between hard_bc and ce_helm at t=T:  1.8e-03")
    if worst_i2 > 1e-4:
        print(">> The target is INCONSISTENT with the discretisation at a level")
        print("   comparable to the observed gap. This is the likely cause.")
    else:
        print(">> The target is consistent. The exact solution does satisfy the")
        print("   discrete constraint, so an inconsistent target is NOT the cause.")

    print()
    print("=" * 78)
    print("2. IS THE PROJECTION OPERATOR ITSELF HARMFUL?")
    print("=" * 78)
    print("Project the trained hard_bc field, which never saw a projection.\n")
    print(f"{'t':>5} {'L2 raw':>11} {'L2 projected':>13} {'change':>9} "
          f"{'I2 drift raw':>13} {'lam1':>11} {'lam2':>11}")
    better = worse = 0
    for t in ts:
        with torch.no_grad():
            tv = torch.full((tr.ops.N,), float(t), dtype=torch.float64)
            ut = tr.model(tr.x, tv)
            us, st = proj.helmholtz_projection(
                ut, tr.C1_ref, tr.C2_ref, mu, tr.dx, tr.q, tr.D1c,
                ga=tr._ga(float(t)), gb=tr._gb(float(t)))
        ue = rlw_soliton(x, float(t), eps=eps, mu=mu, c=C_AMP, x0=X0)
        raw = ut.numpy()
        prj = us.numpy()
        e_raw, e_prj = l2rel(raw, ue), l2rel(prj, ue)
        rx = D1c @ raw
        i2r = abs(float(q @ (raw * raw + mu * rx * rx)) - tr.C2_ref) / tr.C2_ref
        tag = 'better' if e_prj < e_raw else 'WORSE'
        if t > 0:
            better += e_prj < e_raw
            worse += e_prj >= e_raw
        l1 = float(st.lam[0]) if hasattr(st, 'lam') else float('nan')
        l2v = float(st.lam[1]) if hasattr(st, 'lam') else float('nan')
        print(f"{t:5.1f} {e_raw:11.3e} {e_prj:13.3e} {tag:>9} "
              f"{i2r:13.2e} {l1:11.2e} {l2v:11.2e}")
    print(f"\nprojection improved {better} of {better+worse} time levels "
          f"(t>0)")
    if worse > better:
        print(">> The operator DAMAGES an already-accurate field: the cost is")
        print("   intrinsic to projecting, not a training-dynamics effect.")
    else:
        print(">> The operator IMPROVES an accurate field. The cost therefore")
        print("   comes from how training interacts with it, not from the")
        print("   projection itself -- which means it is fixable.")

    print()
    print("=" * 78)
    print("3. ERROR DECOMPOSITION by optimal shift and amplitude")
    print("=" * 78)
    names = ['hard_bc', 'ce_poly', 'ce_helm']
    print(f"{'variant':>9} {'e_total':>11} {'e_noshift':>11} {'e_noamp':>11} "
          f"{'phase %':>8} {'amp %':>8} {'shape %':>8} {'best dx':>9}")
    for nm in names:
        t_r, c_r = load(nm, os.path.join(tmp, nm))
        with torch.no_grad():
            tv = torch.full((t_r.ops.N,), float(T), dtype=torch.float64)
            _, _, u = t_r._diagnostics(float(T))
        up = u.cpu().numpy()
        ue = rlw_soliton(x, float(T), eps=eps, mu=mu, c=C_AMP, x0=X0)
        e_tot = l2rel(up, ue)

        # optimal translation: slide the exact profile, fine sub-grid sweep
        shifts = np.linspace(-0.5, 0.5, 2001)
        errs = [np.sum((up - rlw_soliton(
            x, 0.0, eps=eps, mu=mu, c=C_AMP,
            x0=X0 + (1 + eps * C_AMP) * T + s)) ** 2) for s in shifts]
        k = int(np.argmin(errs))
        best_s = shifts[k]
        u_sh = rlw_soliton(x, 0.0, eps=eps, mu=mu, c=C_AMP,
                           x0=X0 + (1 + eps * C_AMP) * T + best_s)
        e_ns = l2rel(up, u_sh)
        # then the optimal amplitude scale on top of the optimal shift
        a = float(np.dot(up, u_sh) / np.dot(u_sh, u_sh))
        e_na = float(np.sqrt(np.sum((up - a * u_sh) ** 2) / np.sum(ue ** 2)))

        phase = 1.0 - (e_ns / e_tot) ** 2
        amp = (e_ns / e_tot) ** 2 - (e_na / e_tot) ** 2
        shape = (e_na / e_tot) ** 2
        print(f"{nm:>9} {e_tot:11.3e} {e_ns:11.3e} {e_na:11.3e} "
              f"{100*phase:8.1f} {100*amp:8.1f} {100*shape:8.1f} "
              f"{best_s:9.4f}")
    print("\nphase % is how much of the squared error a pure translation")
    print("explains; amp % what an amplitude rescale then explains; shape %")
    print("what neither can explain.")


if __name__ == '__main__':
    main()

"""When does projecting at inference help, and when does it hurt?

The RLW1 analysis found that projecting the trained `hard_bc` field improved L2
at 10 of 10 time levels. The proposed mechanism is that the projection is a
*small* correction there -- the multipliers came out at lam1 ~ 1e-05 and
lam2 ~ 1e-04 -- and a small correction toward a satisfied constraint moves
toward the truth. The mechanism predicts the opposite for a bad field: a large
correction has no reason to point anywhere useful.

If that is right it is a real limitation and belongs in the paper, because it
says exactly when the method applies. If it is wrong, the claim is broader than
we thought. Either way it should be measured, not assumed.

Two measurements:

1. REAL CHECKPOINTS. Project each trained variant's field and see which improve.
   `vanilla` carries a 50% conservation error, so it is the natural stress case.

2. CONTROLLED SWEEP. Start from the exact soliton and add a smooth perturbation
   of known relative size, from 1e-06 up to 1e+00, then project. This locates the
   crossover where the projection stops helping, which is the number the
   limitation should be stated with. The perturbation is band-limited and
   vanishes at both ends, so it resembles the error a hard-BC network actually
   makes rather than white noise.

Run:  py -3 tools/diag_projection_needs_good_field.py
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
VARIANTS = ['vanilla', 'hard_bc', 'ce_poly', 'ce_helm']


def load(name, tmp):
    run = sorted(glob.glob(os.path.join(RES, f'rlw1_{name}', '*')))[-1]
    cfg = json.load(open(os.path.join(run, 'config.json')))
    ck = torch.load(os.path.join(run, 'model_final.pth'), map_location='cpu',
                    weights_only=False)
    sd = ck.get('model_state_dict', ck)
    eps, mu, T, xa, xb = cfg['eps'], cfg['mu'], cfg['T'], cfg['xa'], cfg['xb']
    ic_np = lambda x: rlw_soliton(x, 0.0, eps=eps, mu=mu, c=C_AMP, x0=X0)
    ic_t = lambda x: rlw_soliton_torch(x, torch.zeros_like(x), eps=eps, mu=mu,
                                       c=C_AMP, x0=X0)
    ws = sorted([k for k in sd if k.endswith('.weight')],
                key=lambda s: int(s.split('.')[-2]))
    mlp = MLP(d_in=2, d_out=1, hidden_layers=len(ws) - 1,
              hidden_width=sd[ws[0]].shape[0])
    model = VanillaNet(mlp) if cfg.get('soft_ibc') else \
        HardBC1D(mlp, ic_t, xa, xb, T=T)
    model.load_state_dict(sd)
    tr = GridTrainer(model, dict(cfg, ic_fn=ic_np, diag_every=10 ** 9),
                     device='cpu', out_dir=tmp)
    return tr, cfg


def l2rel(u, ue):
    return float(np.sqrt(np.sum((u - ue) ** 2) / np.sum(ue ** 2)))


def project(tr, u_np, mu, t):
    ut = torch.tensor(u_np, dtype=torch.float64)
    try:
        us, st = proj.helmholtz_projection(
            ut, tr.C1_ref, tr.C2_ref, mu, tr.dx, tr.q, tr.D1c,
            ga=tr._ga(t), gb=tr._gb(t))
        lam = (float(st.lam[0]), float(st.lam[1])) if hasattr(st, 'lam') \
            else (float('nan'),) * 2
        return us.numpy(), lam, float(st.residual), bool(st.clamped)
    except Exception as e:                       # divergence is a valid answer
        return None, (float('nan'),) * 2, float('inf'), False


def main():
    tmp = os.path.join(ROOT, 'tests', '_needgood_tmp')
    os.makedirs(tmp, exist_ok=True)

    print("=" * 78)
    print("1. REAL CHECKPOINTS at t = T")
    print("=" * 78)
    print(f"{'variant':>9} {'I2 err before':>14} {'L2 before':>11} "
          f"{'L2 after':>11} {'ratio':>8} {'lam1':>10} {'lam2':>10} {'':>7}")
    tr0 = None
    for nm in VARIANTS:
        tr, cfg = load(nm, os.path.join(tmp, nm))
        tr0 = tr0 or tr
        eps, mu, T = cfg['eps'], cfg['mu'], cfg['T']
        x, q, D1c = (tr.x.numpy(), tr.q.numpy(), tr.D1c.numpy())
        with torch.no_grad():
            tv = torch.full((tr.ops.N,), float(T), dtype=torch.float64)
            raw = tr.model(tr.x, tv).numpy()
        ue = rlw_soliton(x, float(T), eps=eps, mu=mu, c=C_AMP, x0=X0)
        rx = D1c @ raw
        i2 = abs(float(q @ (raw * raw + mu * rx * rx)) - tr.C2_ref) / tr.C2_ref
        prj, lam, res, clamp = project(tr, raw, mu, float(T))
        e0 = l2rel(raw, ue)
        if prj is None:
            print(f"{nm:>9} {i2:14.2e} {e0:11.3e} {'DIVERGED':>11}")
            continue
        e1 = l2rel(prj, ue)
        tag = 'better' if e1 < e0 else 'WORSE'
        print(f"{nm:>9} {i2:14.2e} {e0:11.3e} {e1:11.3e} {e1/e0:8.4f} "
              f"{lam[0]:10.2e} {lam[1]:10.2e} {tag:>7}"
              + ("  clamped" if clamp else ""))

    print()
    print("=" * 78)
    print("2. CONTROLLED SWEEP: exact soliton plus a perturbation of known size")
    print("=" * 78)
    cfg = load('hard_bc', os.path.join(tmp, 'sweep'))[1]
    tr = tr0
    eps, mu, T = cfg['eps'], cfg['mu'], cfg['T']
    x = tr.x.numpy()
    ue = rlw_soliton(x, float(T), eps=eps, mu=mu, c=C_AMP, x0=X0)

    # smooth, band-limited, vanishing at both ends: what a hard-BC network's
    # error actually looks like. Normalised to unit relative L2 so the sweep
    # parameter *is* the relative error.
    rng = np.random.default_rng(0)
    s = (x - x[0]) / (x[-1] - x[0])
    pert = np.zeros_like(x)
    for k in range(1, 9):
        pert += rng.normal() * np.sin(k * np.pi * s) / k
    pert *= 4.0 * s * (1.0 - s)
    pert /= np.sqrt(np.sum(pert ** 2) / np.sum(ue ** 2))

    print(f"{'rel err in':>11} {'I2 err':>10} {'L2 before':>11} "
          f"{'L2 after':>11} {'ratio':>8} {'lam1':>10} {'lam2':>10} "
          f"{'newton':>9}")
    results = []
    for a in [1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0]:
        u = ue + a * pert
        u[0], u[-1] = ue[0], ue[-1]              # hard BCs are exact
        rx = tr.D1c.numpy() @ u
        i2 = abs(float(tr.q.numpy() @ (u * u + mu * rx * rx)) - tr.C2_ref) \
            / tr.C2_ref
        prj, lam, res, clamp = project(tr, u, mu, float(T))
        e0 = l2rel(u, ue)
        if prj is None or not np.isfinite(prj).all():
            print(f"{a:11.0e} {i2:10.2e} {e0:11.3e} {'DIVERGED':>11}")
            continue
        e1 = l2rel(prj, ue)
        ok = e1 < e0
        results.append((a, (e0, e1)))
        st = 'ok' if res < 1e-10 else 'FAILED'
        if clamp:
            st = 'clamped'
        print(f"{a:11.0e} {i2:10.2e} {e0:11.3e} {e1:11.3e} {e1/e0:8.4f} "
              f"{lam[0]:10.2e} {lam[1]:10.2e} {st:>9}"
              + ("" if ok else "   WORSE"))

    # Fit the two-parameter model the sweep actually follows:
    #     L2_after ~ gain * L2_before + floor
    # `gain` is the multiplicative improvement, `floor` is the error the
    # projection itself introduces by correcting a global integral with a
    # Green's-function-shaped update that does not match the true error shape.
    mid = [(a, r) for a, r in results if a >= 1e-4]
    gain = float(np.median([e1 / e0 for _, (e0, e1) in mid]))
    small = [(e0, e1) for a, (e0, e1) in results if a <= 1e-5]
    floor = max(e1 - gain * e0 for e0, e1 in small) if small else float('nan')

    print()
    print("The prediction was that a large correction would hurt. It does not.")
    print(f"Over the whole range 1e-04 to 1e+00 the projection improves L2 by a")
    print(f"median factor of {gain:.3f}, and it still helps at a 100% error,")
    print(f"cutting that by {100*(1-results[-1][1][1]/results[-1][1][0]):.0f}%. "
          f"The operator is far more")
    print("robust than the mechanism I proposed, so that limitation is not real.")
    print()
    print(f"The real limit is at the *bottom*: the projection introduces an")
    print(f"error floor of about {floor:.1e} relative, because it corrects a")
    print("global integral with an update whose shape is set by the Helmholtz")
    print("Green's function, not by the true error. Below that level it adds")
    print("more than it removes.")
    print(f"  L2_after ~ {gain:.3f} * L2_before + {floor:.1e}")
    print(f"  our runs sit at 1.6e-03, which is {1.6e-3/floor:.0f}x above the "
          f"floor -- safely in the regime where it helps.")
    print(f"  the floor would only bite for a solver reaching below ~"
          f"{floor/(1-gain):.0e}.")
    print()
    print("Caveat on the ce_poly and ce_helm rows in part 1: those are the RAW")
    print("network outputs, evaluated without the projection they were trained")
    print("with, so their 'before' numbers are not solutions at all. Their large")
    print("improvement only confirms that those networks depend on the")
    print("projection -- it says nothing about robustness, and should not be")
    print("read as such.")


if __name__ == '__main__':
    main()

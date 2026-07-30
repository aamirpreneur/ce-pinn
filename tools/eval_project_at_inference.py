"""Score `hard_bc` + projection applied at inference, against the whole sweep.

Motivated by tools/diag_why_projection_costs.py, which found that projecting the
trained `hard_bc` field improved L2 at 10 of 10 time levels while driving I2 to
machine precision. That combination was never in the sweep: `hard_bc` trains
without the projection, and `ce_helm` trains with it in the loss. This measures
the third option.

Reports, over the whole time interval rather than at t=T alone:
  L2 (mean and at T), C1/C2/C3 drift (worst over time), boundary error.

Run:  py -3 tools/eval_project_at_inference.py
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
OUT_MD = os.path.join(RES, 'ANALYSIS_RLW1_INFERENCE.md')
C_AMP, X0 = 0.1, 0.0
N_T = 41
ORDER = ['vanilla', 'hard_bc', 'ce_poly', 'ce_helm', 'ce_full']


def load(name, tmp):
    run = sorted(glob.glob(os.path.join(RES, f'rlw1_{name}', '*')))[-1]
    cfg = json.load(open(os.path.join(run, 'config.json')))
    res = json.load(open(os.path.join(run, 'results.json')))
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
    return tr, cfg, res


def score(tr, cfg, override_projection=None):
    """Metrics over the whole time interval for one model.

    `override_projection` replaces what the run trained with, so 'helmholtz'
    here means: apply the projection at inference regardless of training.
    """
    eps, mu, T = cfg['eps'], cfg['mu'], cfg['T']
    x = tr.x.cpu().numpy()
    q, D1c = tr.q.cpu().numpy(), tr.D1c.cpu().numpy()
    kind = cfg.get('projection') if override_projection is None \
        else override_projection

    l2s, d1s, d2s, d3s, bcs = [], [], [], [], []
    for t in np.linspace(0.0, T, N_T):
        with torch.no_grad():
            tv = torch.full((tr.ops.N,), float(t), dtype=torch.float64)
            ut = tr.model(tr.x, tv)
            if kind == 'helmholtz':
                u = proj.helmholtz_projection(
                    ut, tr.C1_ref, tr.C2_ref, mu, tr.dx, tr.q, tr.D1c,
                    ga=tr._ga(float(t)), gb=tr._gb(float(t)))[0]
            elif kind == 'polynomial':
                u = proj.polynomial_projection(
                    ut, tr.C1_ref, cfg['C2_poly_ref'], tr.q)[0]
            else:
                u = ut
        un = u.numpy()
        ue = rlw_soliton(x, float(t), eps=eps, mu=mu, c=C_AMP, x0=X0)
        l2s.append(np.sqrt(np.sum((un - ue) ** 2) / np.sum(ue ** 2)))
        ux = D1c @ un
        i1 = float(q @ un)
        i2 = float(q @ (un * un + mu * ux * ux))
        i3 = float(q @ (eps * un ** 3 + 3.0 * un * un))
        d1s.append(abs(i1 - tr.C1_ref) / abs(tr.C1_ref))
        d2s.append(abs(i2 - tr.C2_ref) / abs(tr.C2_ref))
        d3s.append(abs(i3 - tr.C3_ref) / abs(tr.C3_ref))
        bcs.append(max(abs(un[0] - tr._ga(float(t))),
                       abs(un[-1] - tr._gb(float(t)))))
    return dict(l2_mean=float(np.mean(l2s)), l2_T=float(l2s[-1]),
                l2_max=float(np.max(l2s)), C1=max(d1s), C2=max(d2s),
                C3=max(d3s), bc=max(bcs))


def main():
    tmp = os.path.join(ROOT, 'tests', '_inf_tmp')
    os.makedirs(tmp, exist_ok=True)

    rows = []
    for nm in ORDER:
        print(f"scoring {nm} ...", flush=True)
        tr, cfg, res = load(nm, os.path.join(tmp, nm))
        rows.append((nm, score(tr, cfg)))
        if nm == 'hard_bc':
            print("scoring hard_bc + projection at inference ...", flush=True)
            rows.append(('hard_bc+proj', score(tr, cfg, 'helmholtz')))

    hdr = (f"| variant | L2 mean | L2 at T | L2 max | C1 drift | C2 drift | "
           f"C3 drift | bc error |")
    L = ["# RLW1: projection at inference vs projection in training\n",
         "All quantities are worst-case over 41 time levels on [0, 20], except",
         "the L2 columns which are the mean, the value at t = T, and the worst.",
         "`hard_bc+proj` is the `hard_bc` network with the Helmholtz projection",
         "applied at evaluation time only; it was never trained with it.\n",
         hdr, "|" + "---|" * 8]
    for nm, s in rows:
        L.append(f"| {nm} | {s['l2_mean']:.3e} | {s['l2_T']:.3e} | "
                 f"{s['l2_max']:.3e} | {s['C1']:.2e} | {s['C2']:.2e} | "
                 f"{s['C3']:.2e} | {s['bc']:.2e} |")
    text = "\n".join(L) + "\n"
    open(OUT_MD, 'w', encoding='utf-8').write(text)
    print("\n" + text)

    d = dict(rows)
    hb, hp, ch = d['hard_bc'], d['hard_bc+proj'], d['ce_helm']
    print(f"hard_bc+proj vs hard_bc :  L2 mean {hp['l2_mean']/hb['l2_mean']:.4f}x, "
          f"C2 drift {hb['C2']/max(hp['C2'],1e-300):.2e}x better")
    print(f"hard_bc+proj vs ce_helm :  L2 mean {hp['l2_mean']/ch['l2_mean']:.4f}x, "
          f"C2 drift {ch['C2']/max(hp['C2'],1e-300):.2e}x")
    print(f"written to {OUT_MD}")


if __name__ == '__main__':
    main()

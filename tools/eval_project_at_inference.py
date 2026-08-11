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
import time

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


def target_fns(cfg):
    """Rebuild the time-dependent targets the notebook supplied at training.

    config.json records `time_dependent_targets` but cannot store the callables
    themselves, so a trainer reconstructed from it alone silently falls back to
    the constants C1_ref, C2_ref frozen at t = 0. Projecting against those puts
    the exact solution outside F and adds the error floor of docs/THEORY.md section
    5b to every number in this table. The quadrature and the difference matrix
    must be the trainer's own, which is what makes the sampled exact solution
    satisfy the discrete constraints rather than merely the continuous ones.
    """
    from ce_pinn.operators import trapezoid_weights
    eps, mu, xa, xb, N = (cfg['eps'], cfg['mu'], cfg['xa'], cfg['xb'],
                          cfg['n_grid'])
    dx = (xb - xa) / (N - 1)
    xg = torch.linspace(xa, xb, N, dtype=torch.float64)
    q = trapezoid_weights(N, dx)
    # the trainer's own D1c (train.py:64), not GridOps.D1: the invariant and
    # the projection must be measured with one and the same matrix
    D1 = proj.central_d1_matrix(N, dx)

    def ex(t):
        return rlw_soliton_torch(xg, torch.full_like(xg, float(t)),
                                 eps=eps, mu=mu, c=C_AMP, x0=X0)

    def edge(t, xe):
        tt = torch.as_tensor(float(t), dtype=torch.float64)
        return rlw_soliton_torch(torch.full_like(tt, xe), tt,
                                 eps=eps, mu=mu, c=C_AMP, x0=X0)

    def C2_fn(t):
        u = ex(t)
        ux = torch.matmul(u, D1.T)
        return float(torch.dot(q, u * u + mu * ux * ux))

    return dict(
        ga_fn=lambda t: edge(t, xa), gb_fn=lambda t: edge(t, xb),
        C1_fn=lambda t: float(torch.dot(q, ex(t))),
        C2_fn=C2_fn,
        C3_fn=lambda t: float(torch.dot(q, eps * ex(t) ** 3 + 3.0 * ex(t) ** 2)),
        C2_poly_fn=lambda t: float(torch.dot(q, ex(t) ** 2)))


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
    tr = GridTrainer(model, dict(cfg, ic_fn=ic_np, diag_every=10 ** 9,
                                 **target_fns(cfg)),
                     device='cpu', out_dir=tmp)
    assert tr.C1_fn is not None and tr.C2_fn is not None
    return tr, cfg, res, os.path.basename(run)


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
    t_net = t_proj = 0.0
    for t in np.linspace(0.0, T, N_T):
        t = float(t)
        # every target is read at the current time, never at t = 0
        C1, C2, C3 = tr._C1(t), tr._C2(t), tr._C3(t)
        with torch.no_grad():
            tv = torch.full((tr.ops.N,), t, dtype=torch.float64)
            t0 = time.perf_counter()
            ut = tr.model(tr.x, tv)
            t1 = time.perf_counter()
            if kind == 'helmholtz':
                u = proj.helmholtz_projection(
                    ut, C1, C2, mu, tr.dx, tr.q, tr.D1c,
                    ga=tr._ga(t), gb=tr._gb(t))[0]
            elif kind == 'polynomial':
                u = proj.polynomial_projection(
                    ut, C1, tr._C2poly(t), tr.q)[0]
            else:
                u = ut
            t2 = time.perf_counter()
        t_net += t1 - t0
        t_proj += t2 - t1
        un = u.numpy()
        ue = rlw_soliton(x, t, eps=eps, mu=mu, c=C_AMP, x0=X0)
        l2s.append(np.sqrt(np.sum((un - ue) ** 2) / np.sum(ue ** 2)))
        ux = D1c @ un
        i1 = float(q @ un)
        i2 = float(q @ (un * un + mu * ux * ux))
        i3 = float(q @ (eps * un ** 3 + 3.0 * un * un))
        d1s.append(abs(i1 - C1) / abs(C1))
        d2s.append(abs(i2 - C2) / abs(C2))
        d3s.append(abs(i3 - C3) / abs(C3))
        bcs.append(max(abs(un[0] - tr._ga(t)), abs(un[-1] - tr._gb(t))))
    return dict(l2_mean=float(np.mean(l2s)), l2_T=float(l2s[-1]),
                l2_max=float(np.max(l2s)), C1=max(d1s), C2=max(d2s),
                C3=max(d3s), bc=max(bcs),
                t_net=t_net / N_T, t_proj=t_proj / N_T)


def main():
    tmp = os.path.join(ROOT, 'tests', '_inf_tmp')
    os.makedirs(tmp, exist_ok=True)

    rows, prov = [], []
    for nm in ORDER:
        print(f"scoring {nm} ...", flush=True)
        tr, cfg, res, run = load(nm, os.path.join(tmp, nm))
        prov.append((nm, run, res.get('code_version'), res.get('training_time_s')))
        rows.append((nm, score(tr, cfg)))
        if nm == 'hard_bc':
            print("scoring hard_bc + projection at inference ...", flush=True)
            rows.append(('hard_bc+proj', score(tr, cfg, 'helmholtz')))
            prov.append(('hard_bc+proj', run, res.get('code_version'),
                         res.get('training_time_s')))

    hdr = (f"| variant | L2 mean | L2 at T | L2 max | C1 drift | C2 drift | "
           f"C3 drift | bc error |")
    L = ["# RLW1: projection at inference vs projection in training\n",
         "All quantities are worst-case over 41 time levels on [0, 20], except",
         "the L2 columns which are the mean, the value at t = T, and the worst.",
         "`hard_bc+proj` is the `hard_bc` network with the Helmholtz projection",
         "applied at evaluation time only; it was never trained with it.\n",
         "Targets and boundary data are evaluated at the current time, matching",
         "what the runs trained with. An earlier version of this script projected",
         "against C1, C2 frozen at t = 0, which adds the error floor of docs/THEORY.md",
         "section 5b to every projected row.\n",
         hdr, "|" + "---|" * 8]
    for nm, s in rows:
        L.append(f"| {nm} | {s['l2_mean']:.3e} | {s['l2_T']:.3e} | "
                 f"{s['l2_max']:.3e} | {s['C1']:.2e} | {s['C2']:.2e} | "
                 f"{s['C3']:.2e} | {s['bc']:.2e} |")
    L += ["\n## Cost\n",
          "Per evaluated time level on CPU, mean over 41 levels.\n",
          "| variant | network (ms) | projection (ms) | training (s) |",
          "|---|---|---|---|"]
    tt = dict((p[0], p[3]) for p in prov)
    for nm, s in rows:
        w = tt.get(nm)
        L.append(f"| {nm} | {1e3*s['t_net']:.2f} | {1e3*s['t_proj']:.2f} | "
                 f"{w:.0f} |" if w else
                 f"| {nm} | {1e3*s['t_net']:.2f} | {1e3*s['t_proj']:.2f} | - |")
    L += ["\n## Provenance\n", "| variant | run | code_version | training_time_s |",
          "|---|---|---|---|"]
    for nm, run, ver, w in prov:
        L.append(f"| {nm} | `{run}` | `{ver}` | {w} |")
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

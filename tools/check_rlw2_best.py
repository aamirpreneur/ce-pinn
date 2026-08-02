"""Was the RLW2 run salvageable before L-BFGS blew up?

Run 1 of the RLW2 sweep finished with L2 = 0.99, i.e. no solution at all. The
history shows why: L-BFGS was descending normally until step 260 (loss 6.1e-04)
and then jumped to 3.9e+28 at step 280, after which it never recovered. The
reported numbers come from `model_final.pth`, which is that wreckage.

`model_best.pth` is the checkpoint at the lowest PDE loss seen, 1.63e-04, which
was reached before the blow-up. If that checkpoint scores a sane L2 then the
Adam phase did its job and the only thing to change is the L-BFGS budget -- a
config edit, no numerical change, no CODE_VERSION move, and the completed RLW1
sweep stays valid. If it does not, the problem is deeper than the optimiser and
the sweep design needs revisiting.

This reads stored checkpoints only. No training, no GPU, no numerics touched.

Run:  py -3 tools/check_rlw2_best.py
"""
import os
import sys
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from ce_pinn.equations import rlw_soliton_torch, rlw_two_soliton_ic
from ce_pinn.network import MLP, HardBC1D

torch.set_default_dtype(torch.float64)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
REF = os.path.join(RES, 'reference', 'rlw2_reference.npz')

# RLW2 problem constants, as set in the notebook config cell. Only these are
# absent from the stored config; everything else is read from config.json.
C1_AMP, C2_AMP = 5.33 / 3.0, 1.69 / 3.0
X1, X2 = 15.0, 35.0


def infer_arch(sd):
    ws = sorted([k for k in sd if k.endswith('.weight')],
                key=lambda s: int(s.split('.')[-2]))
    return len(ws) - 1, sd[ws[0]].shape[0]


def build(cfg, sd):
    eps, mu, T, xa, xb = (cfg['eps'], cfg['mu'], cfg['T'],
                          cfg['xa'], cfg['xb'])

    def ic_t(x):
        z = torch.zeros_like(x)
        return (rlw_soliton_torch(x, z, eps=eps, mu=mu, c=C1_AMP, x0=X1)
                + rlw_soliton_torch(x, z, eps=eps, mu=mu, c=C2_AMP, x0=X2))

    layers, width = infer_arch({k.replace('mlp.', ''): v
                                for k, v in sd.items() if k.startswith('mlp.')})
    mlp = MLP(d_in=2, d_out=1, hidden_layers=layers, hidden_width=width)
    model = HardBC1D(mlp, ic_t, xa, xb, T=T)
    model.load_state_dict(sd)
    model.eval()
    return model


def l2rel(u, ue):
    return float(np.sqrt(np.sum((u - ue) ** 2) / np.sum(ue ** 2)))


def main():
    run = sorted(glob.glob(os.path.join(RES, 'rlw2_hard_bc', '*')))[-1]
    cfg = json.load(open(os.path.join(run, 'config.json')))
    res = json.load(open(os.path.join(run, 'results.json')))
    hist = json.load(open(os.path.join(run, 'history.json')))

    ref = np.load(REF)
    x_ref, t_ref, u_ref = ref['x'], ref['t'], ref['u']
    assert cfg['n_grid'] == len(x_ref), 'grid mismatch with reference'

    print('=' * 74)
    print(f'run: {os.path.relpath(run, ROOT)}')
    print(f'reference own error: {float(ref["est_rel_error"]):.2e} relative')
    print('=' * 74)

    # where the blow-up happened, from the stored history
    lb = [h for h in hist if h['phase'] == 'LBFGS']
    last_ok = max((h for h in lb if h['loss_total'] < 1.0),
                  key=lambda h: h['step'], default=None)
    first_bad = min((h for h in lb if h['loss_total'] > 1e6),
                    key=lambda h: h['step'], default=None)
    print('\nL-BFGS trajectory')
    if last_ok:
        print(f'  last sane step   : {last_ok["step"]:4d}  '
              f'loss {last_ok["loss_total"]:.3e}')
    if first_bad:
        print(f'  first divergence : {first_bad["step"]:4d}  '
              f'loss {first_bad["loss_total"]:.3e}')
    adam = [h for h in hist if h['phase'] == 'Adam']
    print(f'  Adam final       : {adam[-1]["step"]:4d}  '
          f'loss {adam[-1]["loss_total"]:.3e}')
    print(f'  best PDE loss    : {res["best_pde_loss"]:.3e}  '
          '(the checkpoint in model_best.pth)')

    x = torch.linspace(cfg['xa'], cfg['xb'], cfg['n_grid'])
    print(f'\n{"checkpoint":>12} {"L2 rel at T":>12} {"Linf":>10} '
          f'{"verdict":>26}')
    scores = {}
    for tag in ['model_best', 'model_final']:
        p = os.path.join(run, tag + '.pth')
        if not os.path.exists(p):
            continue
        ck = torch.load(p, map_location='cpu', weights_only=False)
        sd = ck.get('model_state_dict', ck)
        model = build(cfg, sd)
        with torch.no_grad():
            tv = torch.full((cfg['n_grid'],), float(cfg['T']))
            u = model(x, tv).numpy()
        e = l2rel(u, u_ref[-1])
        li = float(np.max(np.abs(u - u_ref[-1])))
        scores[tag] = e
        if not np.isfinite(e):
            v = 'diverged'
        elif e > 0.5:
            v = 'no solution'
        elif e > 0.05:
            v = 'poor'
        elif e > 5e-3:
            v = 'usable, not competitive'
        else:
            v = 'good'
        print(f'{tag:>12} {e:12.3e} {li:10.2e} {v:>26}')

    print()
    print('=' * 74)
    b, f = scores.get('model_best'), scores.get('model_final')
    if b is not None and f is not None and np.isfinite(b) and b < f / 10:
        print('The Adam phase produced a real solution; L-BFGS destroyed it.')
        print(f'model_best is {f / b:.0f}x better than the reported final state.')
        print('Fix is a config edit (shorter or no L-BFGS), not a code change.')
    else:
        print('model_best is not meaningfully better. The problem is not the')
        print('L-BFGS tail alone -- the Adam phase did not solve RLW2 either.')
    print('=' * 74)
    print('\nRLW1 hard_bc reached L2 = 1.0e-03 for scale.')


if __name__ == '__main__':
    main()

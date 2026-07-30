"""Post-hoc analysis of the finished RLW1 sweep. No training, no GPU.

The sweep table reports one number per run, and every one of them is taken at
`t = T` only (`GridTrainer.evaluate` calls `_diagnostics(T)`). That leaves three
questions open, and all three can be answered from the checkpoints that are
already in `results/`:

  A. How does the error grow in time? If `hard_bc` leads at every time level
     then exact conservation buys nothing on this horizon. If `ce_helm` closes
     the gap as t grows, the crossover is the result.

  B. Is the L2 gap amplitude or phase? I2 = int(u^2 + mu*u_x^2) pins the
     soliton's amplitude, not its position. Aligning the predicted peak with
     the exact one and re-measuring separates the two: whatever alignment
     removes was phase error, and conservation was never meant to control it.

  C. Had `ce_helm` converged at step 5000? Its final residual was 5.65e-09
     against `hard_bc`'s 1.53e-09. If its loss curve was still descending, the
     L2 gap is a training-budget artefact rather than a property of the method.

Run:  py -3 tools/analyze_rlw1.py
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

torch.set_default_dtype(torch.float64)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
OUT_MD = os.path.join(RES, 'ANALYSIS_RLW1.md')

# RLW1 problem constants. Only `c` is absent from the stored config, so it is
# recovered by checking the analytic C1 = 6c/k against the recorded C1_ref
# rather than trusting a hardcoded value.
C_AMP, X0 = 0.1, 0.0
N_TIMES = 21                      # t = 0, 1, ... 20
ORDER = ['vanilla', 'hard_bc', 'ce_poly', 'ce_helm', 'ce_full']


def infer_arch(sd):
    """Recover (hidden_layers, hidden_width) from a state dict."""
    ws = [k for k in sd if k.endswith('.weight')]
    widths = [sd[k].shape[0] for k in sorted(
        ws, key=lambda s: int(s.split('.')[-2]))]
    return len(widths) - 1, widths[0]


def load_run(run_dir):
    cfg = json.load(open(os.path.join(run_dir, 'config.json')))
    res = json.load(open(os.path.join(run_dir, 'results.json')))
    hist = json.load(open(os.path.join(run_dir, 'history.json')))
    ck = torch.load(os.path.join(run_dir, 'model_final.pth'),
                    map_location='cpu', weights_only=False)
    sd = ck['model_state_dict'] if 'model_state_dict' in ck else ck
    return cfg, res, hist, sd


def build_trainer(cfg, sd, out_dir):
    eps, mu, T = cfg['eps'], cfg['mu'], cfg['T']
    xa, xb = cfg['xa'], cfg['xb']

    def ic_np(x):
        return rlw_soliton(x, 0.0, eps=eps, mu=mu, c=C_AMP, x0=X0)

    def ic_t(x):
        return rlw_soliton_torch(x, torch.zeros_like(x), eps=eps, mu=mu,
                                 c=C_AMP, x0=X0)

    layers, width = infer_arch({k.replace('mlp.', ''): v
                                for k, v in sd.items()})
    mlp = MLP(d_in=2, d_out=1, hidden_layers=layers, hidden_width=width)
    model = VanillaNet(mlp) if cfg.get('soft_ibc') else \
        HardBC1D(mlp, ic_t, xa, xb, T=T)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    tcfg = dict(cfg)
    tcfg['ic_fn'] = ic_np
    tcfg['diag_every'] = 10 ** 9
    tr = GridTrainer(model, tcfg, device='cpu', out_dir=out_dir)
    # the stored reference invariants must reproduce, otherwise C_AMP is wrong
    assert abs(tr.C1_ref - cfg['C1_ref']) < 1e-10, \
        f"C1 mismatch: rebuilt {tr.C1_ref} vs stored {cfg['C1_ref']}"
    return tr, (layers, width)


def peak_fit(x, u):
    """Sub-grid peak (position, amplitude) by parabolic fit on three points."""
    i = int(np.argmax(u))
    if i == 0 or i == len(u) - 1:
        return float(x[i]), float(u[i])
    y0, y1, y2 = u[i - 1], u[i], u[i + 1]
    denom = y0 - 2.0 * y1 + y2
    d = 0.0 if denom == 0 else 0.5 * (y0 - y2) / denom
    return float(x[i] + d * (x[1] - x[0])), float(y1 - 0.25 * (y0 - y2) * d)


def l2_rel(u, u_ex):
    return float(np.sqrt(np.sum((u - u_ex) ** 2) /
                         (np.sum(u_ex ** 2) + 1e-30)))


def analyse(run_dir, tmp):
    cfg, res, hist, sd = load_run(run_dir)
    tr, arch = build_trainer(cfg, sd, tmp)
    eps, mu, T = cfg['eps'], cfg['mu'], cfg['T']
    x = tr.x.cpu().numpy()
    speed = 1.0 + eps * C_AMP
    amp_ex = 3.0 * C_AMP

    rows = []
    for t in np.linspace(0.0, T, N_TIMES):
        drift, bc_err, u_t = tr._diagnostics(float(t))
        u = u_t.cpu().numpy()
        u_ex = rlw_soliton(x, float(t), eps=eps, mu=mu, c=C_AMP, x0=X0)

        e_tot = l2_rel(u, u_ex)
        xp, ap = peak_fit(x, u)
        x_ex = X0 + speed * float(t)

        # exact soliton translated onto the predicted peak: what remains after
        # this is not attributable to a phase shift
        u_shift = rlw_soliton(x, 0.0, eps=eps, mu=mu, c=C_AMP, x0=xp)
        e_align = l2_rel(u, u_shift)

        rows.append(dict(t=float(t), e_tot=e_tot, e_align=e_align,
                         amp=ap, amp_err=(ap - amp_ex) / amp_ex,
                         pos=xp, pos_err=xp - x_ex,
                         I1=drift[0], I2=drift[1], I3=drift[2], bc=bc_err))
    return cfg, res, hist, arch, rows


def convergence(hist):
    """Loss level at four points in Adam, plus the L-BFGS end state."""
    ad = [(h['step'], h['loss_pde']) for h in hist if h['phase'] == 'Adam']
    lb = [(h['step'], h['loss_pde']) for h in hist if h['phase'] == 'LBFGS']
    if not ad:
        return {}
    n = max(s for s, _ in ad)

    def at(frac):
        target = frac * n
        return min(ad, key=lambda p: abs(p[0] - target))[1]

    out = dict(adam_steps=n, l_50=at(0.50), l_75=at(0.75), l_90=at(0.90),
               l_100=at(1.0))
    # how much of the total descent happened in the final tenth: a run that is
    # still falling fast at the end has not converged
    out['last_decile_drop'] = out['l_90'] / max(out['l_100'], 1e-300)
    if lb:
        out['lbfgs_steps'] = max(s for s, _ in lb)
        out['lbfgs_final'] = lb[-1][1]
        out['lbfgs_gain'] = out['l_100'] / max(lb[-1][1], 1e-300)
    return out


def main():
    tmp = os.path.join(ROOT, 'tests', '_analysis_tmp')
    os.makedirs(tmp, exist_ok=True)

    runs = {}
    for d in sorted(glob.glob(os.path.join(RES, 'rlw1_*'))):
        name = os.path.basename(d).replace('rlw1_', '')
        if 'smoke' in name:
            continue
        sub = sorted(glob.glob(os.path.join(d, '*')))
        if not sub:
            continue
        print(f"loading {name} ...", flush=True)
        runs[name] = analyse(sub[-1], os.path.join(tmp, name))

    names = [n for n in ORDER if n in runs] + \
            [n for n in runs if n not in ORDER]

    L = []
    w = L.append
    w("# RLW1 post-hoc analysis\n")
    w("Computed from the stored `model_final.pth` of each run; no retraining.")
    w("`e_total` is the relative L2 error against the exact soliton at that")
    w("time. `e_aligned` is the same error after translating the exact soliton")
    w("onto the predicted peak, so it is the part of the error that a pure")
    w("phase shift cannot explain.\n")
    cv = runs[names[0]][0]
    w(f"Problem: RLW, eps={cv['eps']}, mu={cv['mu']}, c={C_AMP} "
      f"(amplitude {3*C_AMP}), domain [{cv['xa']}, {cv['xb']}], "
      f"T={cv['T']}, n_grid={cv['n_grid']}.\n")

    # ---------------- A: error growth in time ----------------
    w("## A. Error growth in time\n")
    w("Relative L2 error at each time level.\n")
    w("| t | " + " | ".join(f"{n}" for n in names) + " |")
    w("|" + "---|" * (len(names) + 1))
    ts = [r['t'] for r in runs[names[0]][4]]
    for i, t in enumerate(ts):
        cells = [f"{runs[n][4][i]['e_tot']:.3e}" for n in names]
        w(f"| {t:.0f} | " + " | ".join(cells) + " |")
    w("")

    # ---------------- B: amplitude vs phase ----------------
    w("## B. Amplitude and phase, and what alignment removes\n")
    for n in names:
        rows = runs[n][4]
        w(f"### {n}\n")
        w("| t | e_total | e_aligned | amp | amp_err | pos | pos_err |")
        w("|---|---|---|---|---|---|---|")
        for r in rows[::4]:
            w(f"| {r['t']:.0f} | {r['e_tot']:.3e} | {r['e_align']:.3e} | "
              f"{r['amp']:.6f} | {r['amp_err']:+.3e} | {r['pos']:.4f} | "
              f"{r['pos_err']:+.4f} |")
        w("")

    w("### Summary at t = T\n")
    w("| variant | e_total | e_aligned | phase share | amp_err | pos_err |")
    w("|---|---|---|---|---|---|")
    for n in names:
        r = runs[n][4][-1]
        share = 1.0 - (r['e_align'] / r['e_tot']) ** 2 if r['e_tot'] > 0 else 0.0
        w(f"| {n} | {r['e_tot']:.3e} | {r['e_align']:.3e} | "
          f"{100*share:.1f}% | {r['amp_err']:+.3e} | {r['pos_err']:+.4f} |")
    w("")

    # ---------------- invariant drift in time ----------------
    w("## Invariant drift in time (I2 = the derivative-dependent one)\n")
    w("| t | " + " | ".join(names) + " |")
    w("|" + "---|" * (len(names) + 1))
    for i, t in enumerate(ts[::4]):
        j = i * 4
        cells = [f"{runs[n][4][j]['I2']:.2e}" for n in names]
        w(f"| {t:.0f} | " + " | ".join(cells) + " |")
    w("")

    # ---------------- C: convergence ----------------
    w("## C. Had the runs converged?\n")
    w("`loss_pde` at fractions of the Adam budget, then L-BFGS.\n")
    w("| variant | arch | 50% | 75% | 90% | 100% | 90%/100% | "
      "L-BFGS final | Adam/L-BFGS |")
    w("|---|---|---|---|---|---|---|---|---|")
    for n in names:
        cfg, res, hist, arch, rows = runs[n]
        c = convergence(hist)
        if not c:
            continue
        w(f"| {n} | {arch[0]}x{arch[1]} | {c['l_50']:.2e} | {c['l_75']:.2e} | "
          f"{c['l_90']:.2e} | {c['l_100']:.2e} | "
          f"{c['last_decile_drop']:.2f}x | "
          f"{c.get('lbfgs_final', float('nan')):.2e} | "
          f"{c.get('lbfgs_gain', float('nan')):.2f}x |")
    w("")
    w("A `90%/100%` ratio near 1 means the loss had flattened. A large ratio")
    w("means it was still descending when the budget ran out.\n")

    # ---------------- provenance ----------------
    w("## Provenance\n")
    w("| variant | code_version | reported L2 (t=T) | recomputed L2 (t=T) |")
    w("|---|---|---|---|")
    for n in names:
        cfg, res, hist, arch, rows = runs[n]
        w(f"| {n} | {res.get('code_version', 'n/a')} | "
          f"{res.get('L2_relative', float('nan')):.3e} | "
          f"{rows[-1]['e_tot']:.3e} |")
    w("")

    text = "\n".join(L)
    open(OUT_MD, 'w', encoding='utf-8').write(text)
    print("\n" + text)
    print(f"\nwritten to {OUT_MD}")


if __name__ == '__main__':
    main()

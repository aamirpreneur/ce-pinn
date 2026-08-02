"""Numerical verification of the two theory results, against the RLW1 sweep.

Two statements are being checked. Both are claims about the discrete operators
the code actually uses, not about their continuous idealisations, so they are
checked with the trainer's own `q` and `D1c` rather than reimplementations.

  THEOREM 2 (a posteriori certificate). Write the discrete momentum as a
  quadratic form, I2(u) = u^T A u with A = Q + mu D^T Q D, so that
  I2(u) = ||u||_A^2 and ||.||_A is the discrete H1_mu norm. The exact solution
  satisfies ||u||_A^2 = C2 by definition of the invariant. Then for any network
  output u~, the reverse triangle inequality in the A-norm gives

      || u~ - u ||_A  >=  | sqrt(I2(u~)) - sqrt(C2) |.

  Every quantity on the right is computable without knowing u. This is a
  certificate: a rigorous lower bound on a black-box network's H1 error,
  carrying no unknown constants and no fitted parameters.

  The verification asks two things. Does the bound hold (it must -- if it fails
  the algebra is wrong)? And is it tight enough to be useful, or so loose that
  it certifies nothing?

  PROPOSITION (codimension 2). M = {u : I1(u) = C1, I2(u) = C2} is the
  intersection of a hyperplane and an ellipsoid, so it has codimension 2. Its
  normal space at u is span{grad I1, grad I2}, a 2-dimensional subspace. To
  first order the invariant violations depend only on the component of the
  error in that subspace; everything orthogonal to it is invisible to any
  method built on these two invariants, and the projection cannot touch it.

  If the visible fraction ||e_N||^2 / ||e||^2 comes out small, that is the
  reason conservation does not control accuracy -- and it predicts the
  separately-measured result that 96-99% of the error is shape error.

Run:  py -3 tools/verify_theory.py
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
OUT_MD = os.path.join(RES, 'ANALYSIS_THEORY.md')

C_AMP, X0 = 0.1, 0.0
ORDER = ['vanilla', 'hard_bc', 'ce_poly', 'ce_helm', 'ce_full']


def load(name, tmp):
    run = sorted(glob.glob(os.path.join(RES, f'rlw1_{name}', '*')))[-1]
    cfg = json.load(open(os.path.join(run, 'config.json')))
    ck = torch.load(os.path.join(run, 'model_final.pth'), map_location='cpu',
                    weights_only=False)
    sd = ck.get('model_state_dict', ck)
    eps, mu, T, xa, xb = cfg['eps'], cfg['mu'], cfg['T'], cfg['xa'], cfg['xb']

    def ic_np(x):
        return rlw_soliton(x, 0.0, eps=eps, mu=mu, c=C_AMP, x0=X0)

    def ic_t(x):
        return rlw_soliton_torch(x, torch.zeros_like(x), eps=eps, mu=mu,
                                 c=C_AMP, x0=X0)

    ws = sorted([k for k in sd if k.startswith('mlp.') and k.endswith('.weight')],
                key=lambda s: int(s.split('.')[-2]))
    mlp = MLP(d_in=2, d_out=1, hidden_layers=len(ws) - 1,
              hidden_width=sd[ws[0]].shape[0])
    model = VanillaNet(mlp) if cfg.get('soft_ibc') else \
        HardBC1D(mlp, ic_t, xa, xb, T=T)
    model.load_state_dict(sd)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        tr = GridTrainer(model, dict(cfg, ic_fn=ic_np, diag_every=10 ** 9),
                         device='cpu', out_dir=tmp)
    return tr, cfg


def main():
    tmp = os.path.join(ROOT, 'tests', '_theory_tmp')
    os.makedirs(tmp, exist_ok=True)
    lines = []

    def out(s=''):
        print(s)
        lines.append(s)

    rows = []
    for name in ORDER:
        try:
            tr, cfg = load(name, os.path.join(tmp, name))
        except (IndexError, FileNotFoundError):
            continue
        eps, mu, T = cfg['eps'], cfg['mu'], cfg['T']
        x = tr.x.numpy()
        q = tr.q.numpy()
        D = tr.D1c.numpy()
        N = len(x)

        # A = Q + mu D^T Q D, so that I2(u) = u^T A u = ||u||_A^2
        Q = np.diag(q)
        A = Q + mu * (D.T @ Q @ D)

        with torch.no_grad():
            tv = torch.full((N,), float(T), dtype=torch.float64)
            ut = tr.model(tr.x, tv).numpy()
        ue = rlw_soliton(x, float(T), eps=eps, mu=mu, c=C_AMP, x0=X0)
        e = ut - ue

        C1, C2 = tr.C1_ref, tr.C2_ref
        I1t = float(q @ ut)
        I2t = float(ut @ A @ ut)

        nrm_h = lambda v: float(np.sqrt(v @ Q @ v))
        nrm_A = lambda v: float(np.sqrt(v @ A @ v))

        # ---- Theorem 2: certificate ----------------------------------
        cert_A = abs(np.sqrt(max(I2t, 0.0)) - np.sqrt(C2))
        true_A = nrm_A(e)
        holds = cert_A <= true_A * (1 + 1e-12)
        tight = cert_A / true_A if true_A > 0 else float('nan')

        cert_h = abs(I1t - C1) / np.sqrt(float(q.sum()))
        true_h = nrm_h(e)
        holds1 = cert_h <= true_h * (1 + 1e-12)

        # ---- Proposition: codimension-2 decomposition ----------------
        # normal space at the EXACT solution, in the discrete L2 metric:
        #   grad I1 = 1,  grad I2 = 2 Q^{-1} A u
        g1 = np.ones(N)
        g2 = 2.0 * (A @ ue) / q
        # Gram-Schmidt in <.,.>_h, then project e onto span{g1,g2}
        B = np.stack([g1, g2], axis=1)
        G = B.T @ Q @ B
        rhs = B.T @ Q @ e
        coef = np.linalg.solve(G, rhs)
        eN = B @ coef
        eT = e - eN
        vis = (nrm_h(eN) / nrm_h(e)) ** 2 if nrm_h(e) > 0 else float('nan')

        # ---- Corollary: a ceiling on how much projection can help ----
        # The projection can only remove the visible component e_N. Whatever
        # it does with that, the tangential part survives untouched, so
        #     ||e_after|| >= ||e_T|| = ||e|| * sqrt(1 - vis),
        # giving a *rigorous upper bound* on the achievable improvement that
        # depends on no property of the projection beyond which invariants it
        # enforces. It therefore bounds every method built on I1 and I2, not
        # just this one. Measured against what the projection actually
        # achieves, from the same checkpoint.
        ceiling = 1.0 - np.sqrt(max(0.0, 1.0 - vis))
        realised = float('nan')
        try:
            from ce_pinn import projection as proj
            us, st = proj.helmholtz_projection(
                torch.tensor(ut), tr.C1_ref, tr.C2_ref, mu, tr.dx, tr.q,
                tr.D1c, ga=tr._ga(float(T)), gb=tr._gb(float(T)))
            up = us.numpy()
            e_after = up - ue
            l2_before = np.sqrt(np.sum(e ** 2) / np.sum(ue ** 2))
            l2_after = np.sqrt(np.sum(e_after ** 2) / np.sum(ue ** 2))
            realised = 1.0 - l2_after / l2_before
        except Exception:
            pass

        rows.append(dict(name=name, N=N, C2=C2, I2t=I2t,
                         cert_A=cert_A, true_A=true_A, holds=holds,
                         tight=tight, cert_h=cert_h, true_h=true_h,
                         holds1=holds1, vis=vis, ceiling=ceiling,
                         realised=realised,
                         l2rel=float(np.sqrt(np.sum(e ** 2) / np.sum(ue ** 2)))))

    out('# Numerical verification of the theory')
    out()
    out('Checked against the stored RLW1 checkpoints at `t = T`, using the')
    out("trainer's own `q` and `D1c`, so these test the discrete operators the")
    out('code uses rather than their continuous idealisations.')
    out()
    out('## Theorem 2 — a posteriori certificate')
    out()
    out('    || u~ - u ||_A  >=  | sqrt(I2(u~)) - sqrt(C2) |')
    out()
    out('The right-hand side is computable without knowing the exact solution.')
    out('`holds` must be true for every row; a false entry means the algebra is')
    out('wrong. `tightness` is certificate / true error — how much of the real')
    out('error the certificate actually detects.')
    out()
    out('| variant | L2 rel | true ‖e‖_A | certificate | holds | tightness |')
    out('|---|---|---|---|---|---|')
    for r in rows:
        out(f"| `{r['name']}` | {r['l2rel']:.3e} | {r['true_A']:.3e} | "
            f"{r['cert_A']:.3e} | {'yes' if r['holds'] else '**NO**'} | "
            f"{r['tight']:.2e} |")
    out()
    out('The same construction on the linear invariant, in the L2 norm:')
    out()
    out('| variant | true ‖e‖_h | certificate from I1 | holds |')
    out('|---|---|---|---|')
    for r in rows:
        out(f"| `{r['name']}` | {r['true_h']:.3e} | {r['cert_h']:.3e} | "
            f"{'yes' if r['holds1'] else '**NO**'} |")
    out()
    out('## Proposition — codimension 2')
    out()
    out('`visible` is ‖e_N‖²/‖e‖², the fraction of error energy lying in')
    out('span{grad I1, grad I2}. Everything else is invisible to these two')
    out('invariants and cannot be removed by any projection built on them.')
    out('`2/N` is the value expected for an isotropic error in N dimensions.')
    out()
    out('| variant | N | visible | invisible | 2/N |')
    out('|---|---|---|---|---|')
    for r in rows:
        out(f"| `{r['name']}` | {r['N']} | {100*r['vis']:.3f}% | "
            f"{100*(1-r['vis']):.3f}% | {2/r['N']*100:.3f}% |")
    out()

    out('## Corollary — a ceiling on the benefit of conservation')
    out()
    out('Only `e_N` can be removed, so any projection enforcing I1 and I2 has')
    out('its improvement bounded by `1 - sqrt(1 - visible)`. This bounds every')
    out('method built on these two invariants, not only this one. `realised` is')
    out('what the Helmholtz projection actually achieves on the same field.')
    out()
    out('| variant | visible | predicted ceiling | realised | ratio |')
    out('|---|---|---|---|---|')
    for r in rows:
        rr = (r['realised'] / r['ceiling']) if r['ceiling'] > 0 else float('nan')
        out(f"| `{r['name']}` | {100*r['vis']:.3f}% | {100*r['ceiling']:.3f}% | "
            f"{100*r['realised']:.3f}% | {rr:.3f} |")
    out()

    ok2 = all(r['holds'] for r in rows)
    ok1 = all(r['holds1'] for r in rows)
    out('## Verdict')
    out()
    out(f"* Theorem 2 (H1 certificate): "
        f"{'holds on every run' if ok2 else '**FAILS — algebra is wrong**'}")
    out(f"* I1 certificate: "
        f"{'holds on every run' if ok1 else '**FAILS — algebra is wrong**'}")
    hb = [r for r in rows if r['name'] == 'hard_bc']
    if hb:
        out(f"* visible fraction for `hard_bc`: {100*hb[0]['vis']:.3f}%, so "
            f"{100*(1-hb[0]['vis']):.3f}% of the error is")
        out('  beyond the reach of these invariants by construction.')
    out()

    os.makedirs(RES, exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nwritten: {os.path.relpath(OUT_MD, ROOT)}')


if __name__ == '__main__':
    main()

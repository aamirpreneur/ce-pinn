"""Generate standalone Kaggle notebooks from the ce_pinn library.

Each generated notebook carries the full library source inline, so it runs on
Kaggle with no pip install and no git clone. The library remains the single
source of truth: fix a bug there, rerun this script, and every notebook picks
it up. Editing a generated notebook directly is a mistake, because the next
regeneration overwrites it.

Run:  py -3 tools/make_notebooks.py
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, 'ce_pinn')
OUT = os.path.join(ROOT, 'notebooks')
REPO = 'https://github.com/aamirpreneur/ce-pinn.git'

# order matters: later modules use names from earlier ones
MODULES = ['equations.py', 'operators.py', 'network.py', 'projection.py',
           'train.py']

DROP = re.compile(r'^\s*(from\s+\.|from\s+ce_pinn|import\s+wandb\b)')


def strip_module(path):
    """Remove intra-package imports and the module docstring."""
    src = open(path, encoding='utf-8').read()
    # drop the leading module docstring
    m = re.match(r'\s*(?:"""|\'\'\')', src)
    if m:
        q = src.strip()[:3]
        end = src.index(q, src.index(q) + 3) + 3
        src = src[end:]
    keep = [ln for ln in src.split('\n') if not DROP.match(ln)]
    return '\n'.join(keep).strip('\n')


def library_cells():
    """One cell per module, plus the shim that makes `proj.x` resolve."""
    cells = []
    for mod in MODULES:
        body = strip_module(os.path.join(LIB, mod))
        header = f"# ===== {mod} " + "=" * (52 - len(mod)) + "\n"
        if mod == 'train.py':
            header += ("# `proj` resolves to this notebook's own globals, so the\n"
                       "# library code below works unchanged.\n"
                       "import sys as _sys\n"
                       "proj = _sys.modules[__name__]\n"
                       "HAS_WANDB = True\n\n")
        cells.append(header + body)
    return cells


def code(src):
    return {"cell_type": "code", "metadata": {}, "source": src,
            "execution_count": None, "outputs": []}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


SECRETS = """import os, subprocess
try:
    from kaggle_secrets import UserSecretsClient
    _s = UserSecretsClient()
    os.environ['WANDB_API_KEY'] = _s.get_secret('WANDB_API_KEY')
    GITHUB_TOKEN = _s.get_secret('GITHUB_TOKEN')
    ON_KAGGLE = True
except Exception:
    ON_KAGGLE = False
    print('Not on Kaggle')"""

PRELUDE = """import torch, torch.nn as nn, numpy as np, json, math, time, os
import wandb

torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name())

RUN_NAME = f'{TAG}_{VARIANT}' + ('_smoke' if SMOKE else '')
OUT_DIR = f'results/{RUN_NAME}'
os.makedirs(OUT_DIR, exist_ok=True)

wandb.init(project='ce-pinn', name=RUN_NAME, config=dict(
    variant=VARIANT, smoke=SMOKE, equation=EQUATION, tag=TAG,
    xa=XA, xb=XB, T=T_FINAL, n_grid=N_GRID, n_t=N_T,
    layers=HIDDEN_LAYERS, width=HIDDEN_WIDTH,
    adam=ADAM_ITER, lbfgs=LBFGS_ITER, lam_i3=LAM_I3))
print('Run:', RUN_NAME)"""


def push_cell(tag):
    return f"""# Push results to GitHub, only if the run produced finite numbers.
# A crashed notebook never reaches this cell, so bad runs cannot be stored.
#
# The push is a convenience, not the safety net: everything is already under
# /kaggle/working/{{OUT_DIR}}, which Kaggle keeps in the notebook's Output tab.
# A failed push therefore never costs a run.

def _push():
    import shutil, datetime, urllib.request, json as _json
    if not ON_KAGGLE:
        print('Not on Kaggle, skipping push.')
        return
    if not trainer.is_run_valid():
        print('Run produced NaN/Inf, skipping push.')
        return

    token = GITHUB_TOKEN.strip()

    # Check the credentials before touching git, so a bad token gives a clear
    # message instead of "Authentication failed" from deep inside a clone.
    try:
        req = urllib.request.Request(
            'https://api.github.com/repos/aamirpreneur/ce-pinn',
            headers={{'Authorization': 'Bearer ' + token,
                     'Accept': 'application/vnd.github+json'}})
        with urllib.request.urlopen(req) as r:
            info = _json.load(r)
        if not info.get('permissions', {{}}).get('push'):
            print('Token is valid but has no push permission on this repo.')
            return
        print('Token OK, repo', info['full_name'])
    except Exception as e:
        print('GITHUB_TOKEN was rejected:', e)
        print('  - a 401 means the token is expired or mistyped')
        print('  - a 404 on a private repo means it lacks "repo" scope,')
        print('    or is a fine-grained token without access to ce-pinn')
        print(f'Results remain in /kaggle/working/{{OUT_DIR}} (Output tab).')
        return

    url = f'https://x-access-token:{{token}}@github.com/aamirpreneur/ce-pinn.git'
    subprocess.run(['git', 'config', '--global', 'user.email',
                    'aamir@auto.local'], check=True)
    subprocess.run(['git', 'config', '--global', 'user.name',
                    'aamir-kaggle'], check=True)
    os.chdir('/kaggle/working')
    subprocess.run(['rm', '-rf', '/kaggle/working/repo'])
    # capture output so the URL, which carries the token, is never echoed
    c = subprocess.run(['git', 'clone', '--depth', '1', url,
                        '/kaggle/working/repo'], capture_output=True)
    if c.returncode != 0:
        print('Clone failed (details suppressed to avoid leaking the token).')
        print(f'Results remain in /kaggle/working/{{OUT_DIR}} (Output tab).')
        return

    os.chdir('/kaggle/working/repo')
    ts = datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    dest = f'results/{{RUN_NAME}}/{{ts}}'
    os.makedirs(dest, exist_ok=True)
    for fl in os.listdir(f'/kaggle/working/{{OUT_DIR}}'):
        shutil.copy2(f'/kaggle/working/{{OUT_DIR}}/{{fl}}', dest)
    subprocess.run(['git', 'add', '-A', 'results'])
    subprocess.run(['git', 'commit', '-m', f'results({{RUN_NAME}}/{{ts}})'])
    p = subprocess.run(['git', 'push', 'origin', 'main'], capture_output=True)
    if p.returncode == 0:
        print('Pushed to', dest)
    else:
        print('Push failed (details suppressed).')
        print(f'Results remain in /kaggle/working/{{OUT_DIR}} (Output tab).')

_push()"""


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = {}

EXPERIMENTS['RLW1_single_soliton'] = dict(
    title="RLW-1: Single Solitary Wave",
    blurb=("The central experiment. RLW's momentum invariant\n"
           "I2 = int(u^2 + mu*u_x^2) dx carries a derivative, so the affine\n"
           "projection of PINN-Proj/SDIFP cannot enforce it. Comparing\n"
           "`ce_poly` against `ce_helm` here isolates that difference.\n\n"
           "Parameters follow the published benchmark: eps=1, mu=1, c=0.1,\n"
           "domain [-40,60], T=20. Reference invariants I1=3.979927,\n"
           "I2=0.8104625, I3=2.5790075."),
    config="""TAG      = 'rlw1'
EQUATION = 'rlw'
VARIANT  = 'ce_helm'   # vanilla | hard_bc | ce_poly | ce_helm | ce_full
SMOKE    = True        # True for a quick check, False for the real run

EPS, MU  = 1.0, 1.0
C_AMP    = 0.1         # amplitude parameter; wave amplitude is 3*c
X0       = 0.0
XA, XB   = -40.0, 60.0
T_FINAL  = 20.0

HIDDEN_LAYERS, HIDDEN_WIDTH = 8, 50
N_GRID   = 201 if SMOKE else 401     # projection / residual grid
N_T      = 4   if SMOKE else 10      # time levels per step
ADAM_ITER  = 200 if SMOKE else 20000
LBFGS_ITER = 20  if SMOKE else 2000
LAM_I3   = 1e-4 if VARIANT == 'ce_full' else 0.0
LOG_EVERY = 20 if SMOKE else 200""",
    build="""def ic_np(x):
    return rlw_soliton(x, 0.0, eps=EPS, mu=MU, c=C_AMP, x0=X0)

def ic_torch(x):
    return rlw_soliton_torch(x, torch.zeros_like(x), eps=EPS, mu=MU,
                             c=C_AMP, x0=X0)

def exact_at_T(x):
    return rlw_soliton(x, T_FINAL, eps=EPS, mu=MU, c=C_AMP, x0=X0)

# The affine projection is designed for int(u^2); give it that target so the
# baseline is the method its authors actually published, not a strawman.
_q = trapezoid_weights(N_GRID, (XB - XA) / (N_GRID - 1))
_ic = ic_torch(torch.linspace(XA, XB, N_GRID))
C2_POLY = float(torch.dot(_q, _ic * _ic))

torch.manual_seed(SEED)
mlp = MLP(d_in=2, d_out=1, hidden_layers=HIDDEN_LAYERS,
          hidden_width=HIDDEN_WIDTH)

if VARIANT == 'vanilla':
    model, projection, soft_ibc = VanillaNet(mlp), None, True
elif VARIANT == 'hard_bc':
    model, projection, soft_ibc = HardBC1D(mlp, ic_torch, XA, XB, T=T_FINAL), None, False
elif VARIANT == 'ce_poly':
    model, projection, soft_ibc = HardBC1D(mlp, ic_torch, XA, XB, T=T_FINAL), 'polynomial', False
elif VARIANT in ('ce_helm', 'ce_full'):
    model, projection, soft_ibc = HardBC1D(mlp, ic_torch, XA, XB, T=T_FINAL), 'helmholtz', False
else:
    raise ValueError(VARIANT)

config = dict(equation=EQUATION, eps=EPS, mu=MU, xa=XA, xb=XB, T=T_FINAL,
              ic_fn=ic_np, n_grid=N_GRID, n_t=N_T, projection=projection,
              soft_ibc=soft_ibc, lam_ibc=10.0, C2_poly_ref=C2_POLY,
              diag_every=LOG_EVERY)

print(f'{sum(p.numel() for p in model.parameters()):,} parameters, '
      f'projection={projection}')""",
    exact='exact_at_T',
)

EXPERIMENTS['KdV1_single_soliton'] = dict(
    title="KdV-1: Single Soliton (control)",
    blurb=("A control, not a headline. KdV's I2 = int(u^2) dx is polynomial,\n"
           "so the affine projection already handles it and `ce_poly` and\n"
           "`ce_helm` should tie. That tie is what shows the RLW advantage\n"
           "comes from the derivative term specifically, rather than from\n"
           "some incidental smoothing.\n\n"
           "k=1, domain [-30,50], T=10. Exact invariants C1=2, C2=2/3,\n"
           "C3=1/5 on the infinite line."),
    config="""TAG      = 'kdv1'
EQUATION = 'kdv'
VARIANT  = 'ce_poly'   # vanilla | hard_bc | ce_poly | ce_helm
SMOKE    = True

K_SOL    = 1.0
MU       = 0.0         # KdV has no dispersion parameter; kept for the config
XA, XB   = -30.0, 50.0
T_FINAL  = 10.0

HIDDEN_LAYERS, HIDDEN_WIDTH = 8, 50
N_GRID   = 201 if SMOKE else 401
N_T      = 4   if SMOKE else 10
ADAM_ITER  = 200 if SMOKE else 20000
LBFGS_ITER = 20  if SMOKE else 2000
LAM_I3   = 0.0
LOG_EVERY = 20 if SMOKE else 200""",
    build="""def ic_np(x):
    return kdv_one_soliton(x, 0.0, k=K_SOL)

def ic_torch(x):
    return kdv_one_soliton_torch(x, torch.zeros_like(x), k=K_SOL)

def exact_at_T(x):
    return kdv_one_soliton(x, T_FINAL, k=K_SOL)

_q = trapezoid_weights(N_GRID, (XB - XA) / (N_GRID - 1))
_ic = ic_torch(torch.linspace(XA, XB, N_GRID))
C2_POLY = float(torch.dot(_q, _ic * _ic))

torch.manual_seed(SEED)
mlp = MLP(d_in=2, d_out=1, hidden_layers=HIDDEN_LAYERS,
          hidden_width=HIDDEN_WIDTH)

if VARIANT == 'vanilla':
    model, projection, soft_ibc = VanillaNet(mlp), None, True
elif VARIANT == 'hard_bc':
    model, projection, soft_ibc = HardBC1D(mlp, ic_torch, XA, XB, T=T_FINAL), None, False
elif VARIANT == 'ce_poly':
    model, projection, soft_ibc = HardBC1D(mlp, ic_torch, XA, XB, T=T_FINAL), 'polynomial', False
elif VARIANT in ('ce_helm', 'ce_full'):
    model, projection, soft_ibc = HardBC1D(mlp, ic_torch, XA, XB, T=T_FINAL), 'helmholtz', False
else:
    raise ValueError(VARIANT)

config = dict(equation=EQUATION, eps=0.0, mu=MU, xa=XA, xb=XB, T=T_FINAL,
              ic_fn=ic_np, n_grid=N_GRID, n_t=N_T, projection=projection,
              soft_ibc=soft_ibc, lam_ibc=10.0, C2_poly_ref=C2_POLY,
              diag_every=LOG_EVERY)

print(f'{sum(p.numel() for p in model.parameters()):,} parameters, '
      f'projection={projection}')
print('exact invariants: C1=%.6f C2=%.6f C3=%.6f' % kdv_exact_invariants(K_SOL))""",
    exact='exact_at_T',
)

EXPERIMENTS['KdV2_two_soliton'] = dict(
    title="KdV-2: Two-Soliton Collision (control)",
    blurb=("The second control. Conservation during a nonlinear interaction,\n"
           "but with polynomial invariants, so `ce_poly` should again match\n"
           "`ce_helm`. Exact solution from the Hirota bilinear form,\n"
           "evaluated with log-sum-exp for stability.\n\n"
           "k1=1, k2=2, domain [-20,40], T=5. Well-separated initial solitons\n"
           "give C1=6, C2=6, C3=6.6 asymptotically."),
    config="""TAG      = 'kdv2'
EQUATION = 'kdv'
VARIANT  = 'ce_poly'
SMOKE    = True

K1, K2   = 1.0, 2.0
MU       = 0.0
XA, XB   = -20.0, 40.0
T_FINAL  = 5.0

HIDDEN_LAYERS, HIDDEN_WIDTH = 8, 100
N_GRID   = 201 if SMOKE else 601
N_T      = 4   if SMOKE else 10
ADAM_ITER  = 200 if SMOKE else 30000
LBFGS_ITER = 20  if SMOKE else 3000
LAM_I3   = 0.0
LOG_EVERY = 20 if SMOKE else 200""",
    build="""def ic_np(x):
    return kdv_two_soliton(x, 0.0, k1=K1, k2=K2)

def ic_torch(x):
    # constant in theta, so evaluating through numpy is fine
    v = kdv_two_soliton(x.detach().cpu().numpy(), 0.0, k1=K1, k2=K2)
    return torch.tensor(v, dtype=x.dtype, device=x.device)

def exact_at_T(x):
    return kdv_two_soliton(x, T_FINAL, k1=K1, k2=K2)

_q = trapezoid_weights(N_GRID, (XB - XA) / (N_GRID - 1))
_ic = ic_torch(torch.linspace(XA, XB, N_GRID))
C2_POLY = float(torch.dot(_q, _ic * _ic))

torch.manual_seed(SEED)
mlp = MLP(d_in=2, d_out=1, hidden_layers=HIDDEN_LAYERS,
          hidden_width=HIDDEN_WIDTH)

if VARIANT == 'vanilla':
    model, projection, soft_ibc = VanillaNet(mlp), None, True
elif VARIANT == 'hard_bc':
    model, projection, soft_ibc = HardBC1D(mlp, ic_torch, XA, XB, T=T_FINAL), None, False
elif VARIANT == 'ce_poly':
    model, projection, soft_ibc = HardBC1D(mlp, ic_torch, XA, XB, T=T_FINAL), 'polynomial', False
elif VARIANT in ('ce_helm', 'ce_full'):
    model, projection, soft_ibc = HardBC1D(mlp, ic_torch, XA, XB, T=T_FINAL), 'helmholtz', False
else:
    raise ValueError(VARIANT)

config = dict(equation=EQUATION, eps=0.0, mu=MU, xa=XA, xb=XB, T=T_FINAL,
              ic_fn=ic_np, n_grid=N_GRID, n_t=N_T, projection=projection,
              soft_ibc=soft_ibc, lam_ibc=10.0, C2_poly_ref=C2_POLY,
              diag_every=LOG_EVERY)

print(f'{sum(p.numel() for p in model.parameters()):,} parameters, '
      f'projection={projection}')""",
    exact='exact_at_T',
)

EXPERIMENTS['RLW2_two_soliton'] = dict(
    title="RLW-2: Two Solitary Waves",
    blurb=("Nonlinear interaction. The derivative term is a much larger share\n"
           "of I2 here than for the single wave (roughly 11% against 1.8%),\n"
           "so this discriminates between the two projections more strongly.\n\n"
           "No closed-form solution at T, so conservation drift and the wave\n"
           "amplitudes after collision are the metrics. Amplitudes 5.33 and\n"
           "1.69, domain [0,120], T=30."),
    config="""TAG      = 'rlw2'
EQUATION = 'rlw'
VARIANT  = 'ce_helm'
SMOKE    = True

EPS, MU  = 1.0, 1.0
C1_AMP, C2_AMP = 5.33 / 3.0, 1.69 / 3.0
X1, X2   = 15.0, 35.0
XA, XB   = 0.0, 120.0
T_FINAL  = 30.0

HIDDEN_LAYERS, HIDDEN_WIDTH = 8, 100
N_GRID   = 201 if SMOKE else 601
N_T      = 4   if SMOKE else 10
ADAM_ITER  = 200 if SMOKE else 30000
LBFGS_ITER = 20  if SMOKE else 3000
LAM_I3   = 1e-5 if VARIANT == 'ce_full' else 0.0
LOG_EVERY = 20 if SMOKE else 200""",
    build="""def ic_np(x):
    return rlw_two_soliton_ic(x, eps=EPS, mu=MU, c1=C1_AMP, c2=C2_AMP,
                              x1=X1, x2=X2)

def ic_torch(x):
    z = torch.zeros_like(x)
    return (rlw_soliton_torch(x, z, eps=EPS, mu=MU, c=C1_AMP, x0=X1)
            + rlw_soliton_torch(x, z, eps=EPS, mu=MU, c=C2_AMP, x0=X2))

exact_at_T = None      # no closed form after the collision

_q = trapezoid_weights(N_GRID, (XB - XA) / (N_GRID - 1))
_ic = ic_torch(torch.linspace(XA, XB, N_GRID))
C2_POLY = float(torch.dot(_q, _ic * _ic))

torch.manual_seed(SEED)
mlp = MLP(d_in=2, d_out=1, hidden_layers=HIDDEN_LAYERS,
          hidden_width=HIDDEN_WIDTH)

if VARIANT == 'vanilla':
    model, projection, soft_ibc = VanillaNet(mlp), None, True
elif VARIANT == 'hard_bc':
    model, projection, soft_ibc = HardBC1D(mlp, ic_torch, XA, XB, T=T_FINAL), None, False
elif VARIANT == 'ce_poly':
    model, projection, soft_ibc = HardBC1D(mlp, ic_torch, XA, XB, T=T_FINAL), 'polynomial', False
elif VARIANT in ('ce_helm', 'ce_full'):
    model, projection, soft_ibc = HardBC1D(mlp, ic_torch, XA, XB, T=T_FINAL), 'helmholtz', False
else:
    raise ValueError(VARIANT)

config = dict(equation=EQUATION, eps=EPS, mu=MU, xa=XA, xb=XB, T=T_FINAL,
              ic_fn=ic_np, n_grid=N_GRID, n_t=N_T, projection=projection,
              soft_ibc=soft_ibc, lam_ibc=10.0, C2_poly_ref=C2_POLY,
              diag_every=LOG_EVERY)

print(f'{sum(p.numel() for p in model.parameters()):,} parameters, '
      f'projection={projection}')""",
    exact='exact_at_T',
)

EXPERIMENTS['RLW3_undular_bore'] = dict(
    title="RLW-3: Undular Bore",
    blurb=("The second discriminating case. The left boundary sits at\n"
           "u0 = 0.1, not zero, and the affine projection provably shifts\n"
           "boundary values (u*(a) = alpha*g_a + beta), so `ce_poly` should\n"
           "show a measurable boundary violation while `ce_helm` carries the\n"
           "Dirichlet data inside the elliptic solve and stays exact.\n\n"
           "eps=1.5, mu=1/6, u0=0.1, d=5, domain [-36,300]. T is shortened\n"
           "from the published 250; long-time runs need causal windowing,\n"
           "which is not wired into this notebook yet."),
    config="""TAG      = 'rlw3'
EQUATION = 'rlw'
VARIANT  = 'ce_helm'
SMOKE    = True

EPS, MU  = 1.5, 1.0 / 6.0
U0, XC, DSTEP = 0.1, 0.0, 5.0
XA, XB   = -36.0, 300.0
T_FINAL  = 50.0        # one window; the published benchmark runs to 250

HIDDEN_LAYERS, HIDDEN_WIDTH = 8, 100
N_GRID   = 301 if SMOKE else 801
N_T      = 4   if SMOKE else 10
ADAM_ITER  = 200 if SMOKE else 20000
LBFGS_ITER = 20  if SMOKE else 2000
LAM_I3   = 1e-5 if VARIANT == 'ce_full' else 0.0
LOG_EVERY = 20 if SMOKE else 200""",
    build="""def ic_np(x):
    return rlw_undular_bore_ic(x, u0=U0, xc=XC, d=DSTEP)

def ic_torch(x):
    return 0.5 * U0 * (1.0 - torch.tanh((x - XC) / DSTEP))

exact_at_T = None      # no closed form

# Non-homogeneous Dirichlet data: this is the case that breaks the affine
# projection. Both are constant in time here.
GA_FN = lambda t: U0
GB_FN = lambda t: 0.0

_q = trapezoid_weights(N_GRID, (XB - XA) / (N_GRID - 1))
_ic = ic_torch(torch.linspace(XA, XB, N_GRID))
C2_POLY = float(torch.dot(_q, _ic * _ic))

torch.manual_seed(SEED)
mlp = MLP(d_in=2, d_out=1, hidden_layers=HIDDEN_LAYERS,
          hidden_width=HIDDEN_WIDTH)

if VARIANT == 'vanilla':
    model, projection, soft_ibc = VanillaNet(mlp), None, True
elif VARIANT == 'hard_bc':
    model, projection, soft_ibc = HardBC1D(mlp, ic_torch, XA, XB, T=T_FINAL), None, False
elif VARIANT == 'ce_poly':
    model, projection, soft_ibc = HardBC1D(mlp, ic_torch, XA, XB, T=T_FINAL), 'polynomial', False
elif VARIANT in ('ce_helm', 'ce_full'):
    model, projection, soft_ibc = HardBC1D(mlp, ic_torch, XA, XB, T=T_FINAL), 'helmholtz', False
else:
    raise ValueError(VARIANT)

config = dict(equation=EQUATION, eps=EPS, mu=MU, xa=XA, xb=XB, T=T_FINAL,
              ic_fn=ic_np, n_grid=N_GRID, n_t=N_T, projection=projection,
              soft_ibc=soft_ibc, lam_ibc=10.0, C2_poly_ref=C2_POLY,
              ga_fn=GA_FN, gb_fn=GB_FN, diag_every=LOG_EVERY)

print(f'{sum(p.numel() for p in model.parameters()):,} parameters, '
      f'projection={projection}')""",
    exact='exact_at_T',
)


TRAIN_CELL = """trainer = GridTrainer(model, config, device=device, out_dir=OUT_DIR)
t0 = time.time()

print(f'\\n--- Adam: {ADAM_ITER} iterations ---')
trainer.train_adam(ADAM_ITER, lr=1e-3, lam_soft_i3=LAM_I3, log_every=LOG_EVERY)

if LBFGS_ITER > 0:
    print(f'\\n--- L-BFGS: {LBFGS_ITER} iterations ---')
    trainer.train_lbfgs(LBFGS_ITER, lr=1.0, lam_soft_i3=0.0,
                        log_every=max(1, LOG_EVERY // 10))

elapsed = time.time() - t0
print(f'\\nTraining time: {elapsed:.1f}s ({elapsed/60:.1f} min)')
if trainer._newton_failures:
    print(f'WARNING: {trainer._newton_failures} Newton solves did not converge')"""

EVAL_CELL = """results = trainer.evaluate(EXACT_FN)
results.update(variant=VARIANT, seed=SEED, smoke=SMOKE,
               training_time_s=elapsed,
               newton_failures=trainer._newton_failures)
json.dump(results, open(os.path.join(OUT_DIR, 'results.json'), 'w'), indent=2)
wandb.finish()

print('\\nrun valid:', trainer.is_run_valid())
for f in sorted(os.listdir(OUT_DIR)):
    print(f'  {f}')"""


def build_notebook(name, spec, libcells):
    cells = [md(f"# {spec['title']}\n\n{spec['blurb']}\n\n"
                "Standalone: no pip install, no clone. The library source is "
                "inlined below by `tools/make_notebooks.py`. Edit the library "
                "and regenerate rather than editing this file.")]
    cells.append(code(SECRETS))
    cells.append(md("## Configuration\n\nEverything you would normally change "
                    "lives in this cell."))
    cells.append(code(spec['config'] + "\nSEED     = 0"))
    cells.append(code(PRELUDE))
    cells.append(md("## Library (generated, do not edit here)"))
    for c in libcells:
        cells.append(code(c))
    cells.append(md("## Model"))
    cells.append(code(spec['build']))
    cells.append(md("## Train"))
    cells.append(code(TRAIN_CELL))
    cells.append(md("## Evaluate"))
    cells.append(code(f"EXACT_FN = {spec['exact']}\n" + EVAL_CELL))
    cells.append(md("## Push results"))
    cells.append(code(push_cell(spec.get('tag', name))))

    for i, c in enumerate(cells):
        c['id'] = f'cell-{i}'
        c['source'] = c['source'].split('\n')
        c['source'] = [ln + '\n' for ln in c['source'][:-1]] + [c['source'][-1]]

    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python"}}}


def main():
    libcells = library_cells()
    os.makedirs(OUT, exist_ok=True)
    for name, spec in EXPERIMENTS.items():
        nb = build_notebook(name, spec, libcells)
        path = os.path.join(OUT, name + '.ipynb')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        n = sum(len(c['source']) for c in nb['cells'])
        print(f'  wrote {name}.ipynb  ({len(nb["cells"])} cells, {n} lines)')


if __name__ == '__main__':
    main()

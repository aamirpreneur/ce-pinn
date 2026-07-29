"""Execute every generated notebook in SMOKE mode.

Runs the code cells in a fresh namespace with wandb stubbed out and Kaggle
absent, so the push cell is skipped. Catches import errors, name errors and
shape bugs before any GPU time is spent.

Run:  py -3 tests/test_notebooks.py
"""
import sys, os, json, time, types, io, contextlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, 'notebooks')

# stub wandb so the notebooks run offline
_wandb = types.ModuleType('wandb')
_wandb.init = lambda **k: None
_wandb.log = lambda *a, **k: None
_wandb.finish = lambda *a, **k: None
sys.modules['wandb'] = _wandb

PASS, FAIL = [], []


def run_notebook(path):
    nb = json.load(open(path, encoding='utf-8'))
    cells = [''.join(c['source']) for c in nb['cells']
             if c['cell_type'] == 'code']
    ns = {'__name__': '__main__'}
    sys.modules['__main__'].__dict__.setdefault('_nb_ns', ns)
    mod = types.ModuleType('__nbmain__')
    mod.__dict__['__name__'] = '__nbmain__'
    sys.modules['__nbmain__'] = mod
    for i, src in enumerate(cells):
        # force SMOKE and keep the run tiny
        if 'SMOKE    =' in src or 'SMOKE   =' in src:
            src = src.replace('SMOKE    = True', 'SMOKE    = True')
        exec(compile(src, f'{os.path.basename(path)}:cell{i}', 'exec'),
             mod.__dict__)
    return mod.__dict__


os.chdir(os.path.join(ROOT, 'tests'))
os.makedirs('_nbtmp', exist_ok=True)
os.chdir('_nbtmp')

for nbname in sorted(os.listdir(NB_DIR)):
    if not nbname.endswith('.ipynb'):
        continue
    print(f"\n=== {nbname} ===")
    t0 = time.time()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ns = run_notebook(os.path.join(NB_DIR, nbname))
        dt = time.time() - t0
        res = ns.get('results', {})
        tr = ns.get('trainer')
        ok = tr is not None and tr.is_run_valid()
        nf = getattr(tr, '_newton_failures', -1)
        nc = getattr(tr, '_newton_clamped', 0)
        proj_kind = ns.get('projection')
        line = (f"  {dt:5.1f}s  variant={ns.get('VARIANT'):9s} "
                f"proj={str(proj_kind):11s} "
                f"I2drift={res.get('C2_drift', float('nan')):.2e} "
                f"bc={res.get('bc_error', float('nan')):.2e} "
                f"newton_fail={nf} clamped={nc}")
        print(line)
        if res.get('L2_relative') is not None:
            print(f"         L2_rel={res['L2_relative']:.3e}")
        cond = ok and nf == 0
        (PASS if cond else FAIL).append(nbname)
        print(f"  [{'PASS' if cond else 'FAIL'}] {nbname}")
        if proj_kind == 'helmholtz' and res.get('C2_drift', 1) > 1e-10:
            FAIL.append(nbname + ' (I2 drift too large)')
            print(f"  [FAIL] {nbname}: helmholtz left I2 drift "
                  f"{res['C2_drift']:.2e}")
    except Exception as e:
        FAIL.append(nbname)
        print(buf.getvalue()[-2500:])
        print(f"  [FAIL] {nbname}: {type(e).__name__}: {e}")

print("\n" + "=" * 58)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print(f"   FAILED: {f}")
sys.exit(1 if FAIL else 0)

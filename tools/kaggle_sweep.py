"""Push and run the sweep on Kaggle without uploading notebooks by hand.

Fifteen runs uploaded through the browser is fifteen chances to mis-set VARIANT
or SEED. This generates one kernel per (experiment, variant, seed), each with
those two values already substituted, pushes them, and pulls the outputs back.

Credentials are read by the Kaggle client from ~/.kaggle; nothing here reads,
prints or copies the token.

Usage
  py -3 tools/kaggle_sweep.py list                    what would be created
  py -3 tools/kaggle_sweep.py push <slug>             push one kernel
  py -3 tools/kaggle_sweep.py push-all                push every kernel
  py -3 tools/kaggle_sweep.py status                  status of all kernels
  py -3 tools/kaggle_sweep.py fetch <slug>            download one kernel's output
  py -3 tools/kaggle_sweep.py fetch-all               download everything finished

Kaggle secrets (WANDB_API_KEY, GITHUB_TOKEN) cannot be attached through the
API. The notebooks fall back gracefully without them -- wandb logging is
skipped and the git push cell is a no-op -- and `fetch` pulls the results
directly, so secrets are optional rather than required.
"""
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, 'notebooks')
STAGE = os.path.join(ROOT, 'tests', '_kaggle')
RESULTS = os.path.join(ROOT, 'results')

# (notebook stem, tag) -> the sweep for that problem.
# hard_bc gets three seeds because it is the baseline the ceiling is measured
# against; the other two variants only need to establish where they sit.
EXPERIMENTS = [
    ('RLW1_single_soliton', 'rlw1'),
    ('RLW1T_tall_soliton', 'rlw1t'),
    ('CH1_solitary_wave', 'ch1'),
]
SWEEP = [('hard_bc', 0), ('hard_bc', 1), ('hard_bc', 2),
         ('ce_poly', 0), ('ce_helm', 0)]


def jobs():
    for stem, tag in EXPERIMENTS:
        for variant, seed in SWEEP:
            slug = f"ce-pinn-{tag}-{variant.replace('_','-')}-s{seed}"
            yield dict(stem=stem, tag=tag, variant=variant, seed=seed,
                       slug=slug)


def api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi()
    a.authenticate()
    return a


def username(a):
    return a.config_values.get('username')


def build(job, user):
    """Write a kernel folder with VARIANT and SEED already substituted."""
    src = os.path.join(NB_DIR, job['stem'] + '.ipynb')
    nb = json.load(open(src, encoding='utf-8'))

    hits = {'variant': 0, 'seed': 0}
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        for i, line in enumerate(cell['source']):
            if re.match(r"^VARIANT\s*=", line):
                cell['source'][i] = re.sub(
                    r"^(VARIANT\s*=\s*)'[^']*'", r"\1'%s'" % job['variant'],
                    line)
                hits['variant'] += 1
            elif re.match(r"^SEED\s*=", line):
                cell['source'][i] = re.sub(
                    r"^(SEED\s*=\s*)\d+", r"\g<1>%d" % job['seed'], line)
                hits['seed'] += 1
    if hits['variant'] != 1 or hits['seed'] != 1:
        raise SystemExit(
            f"{job['stem']}: expected exactly one VARIANT and one SEED "
            f"assignment, found {hits}. Refusing to push a kernel whose "
            f"configuration was not set as intended.")

    folder = os.path.join(STAGE, job['slug'])
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'kernel.ipynb'), 'w', encoding='utf-8') as f:
        json.dump(nb, f)
    meta = {
        'id': f"{user}/{job['slug']}",
        'title': f"CE-PINN {job['tag']} {job['variant']} s{job['seed']}",
        'code_file': 'kernel.ipynb',
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': True,
        'enable_internet': True,
        'dataset_sources': [],
        'competition_sources': [],
        'kernel_sources': [],
    }
    with open(os.path.join(folder, 'kernel-metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return folder


def cmd_list():
    print(f"{'slug':>40} {'notebook':>22} {'variant':>9} {'seed':>5}")
    for j in jobs():
        print(f"{j['slug']:>40} {j['stem']:>22} {j['variant']:>9} "
              f"{j['seed']:>5}")
    print(f'\n{len(list(jobs()))} kernels')


def cmd_push(which=None):
    a = api()
    user = username(a)
    for j in jobs():
        if which and j['slug'] != which:
            continue
        folder = build(j, user)
        print(f"pushing {j['slug']} ... ", end='', flush=True)
        try:
            r = a.kernels_push(folder)
            ref = getattr(r, 'ref', None) or getattr(r, 'versionNumber', '')
            err = getattr(r, 'error', None)
            print('ERROR: ' + str(err) if err else f'ok  ({ref})')
        except Exception as e:
            print(f'FAILED: {type(e).__name__}: {e}')


def cmd_status():
    a = api()
    user = username(a)
    print(f"{'slug':>40} {'status':>14}")
    for j in jobs():
        try:
            s = a.kernels_status(f"{user}/{j['slug']}")
            st = s.get('status') if isinstance(s, dict) else getattr(
                s, 'status', '?')
            fail = s.get('failureMessage') if isinstance(s, dict) else None
            print(f"{j['slug']:>40} {str(st):>14}"
                  + (f'  {fail}' if fail else ''))
        except Exception as e:
            print(f"{j['slug']:>40} {'not pushed':>14}  ({type(e).__name__})")


def cmd_fetch(which=None):
    a = api()
    user = username(a)
    for j in jobs():
        if which and j['slug'] != which:
            continue
        dest = os.path.join(STAGE, 'out', j['slug'])
        os.makedirs(dest, exist_ok=True)
        try:
            a.kernels_output(f"{user}/{j['slug']}", path=dest, quiet=True)
        except Exception as e:
            print(f"{j['slug']:>40}  no output ({type(e).__name__})")
            continue
        # the notebook writes results/<run>/... inside the working directory
        moved = 0
        for dirpath, _, files in os.walk(dest):
            if 'results.json' not in files:
                continue
            rel = os.path.relpath(dirpath, dest)
            parts = rel.replace('\\', '/').split('/')
            if 'results' in parts:
                rel = '/'.join(parts[parts.index('results') + 1:])
            target = os.path.join(RESULTS, rel)
            os.makedirs(os.path.dirname(target) or target, exist_ok=True)
            if os.path.exists(target):
                shutil.rmtree(target)
            shutil.copytree(dirpath, target)
            moved += 1
        print(f"{j['slug']:>40}  {moved} run(s) -> results/")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    c = sys.argv[1]
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if c == 'list':
        cmd_list()
    elif c == 'push':
        if not arg:
            raise SystemExit('push needs a slug; use push-all for everything')
        cmd_push(arg)
    elif c == 'push-all':
        cmd_push()
    elif c == 'status':
        cmd_status()
    elif c == 'fetch':
        cmd_fetch(arg)
    elif c == 'fetch-all':
        cmd_fetch()
    else:
        print(__doc__)
        raise SystemExit(1)

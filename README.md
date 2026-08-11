# CE-PINN

Conservation-embedded physics-informed neural networks: a differentiable
projection that enforces a derivative-dependent integral invariant,
$\int(u^2+\mu u_x^2)\,dx$, exactly, and a bound on what that delivers.

This repository holds the code, notebooks, analysis scripts and archived
run outputs behind a paper submitted to *Computer Physics Communications*.
The manuscript source is not included here; see "Citing this work" below.

## Repository layout

```
ce_pinn/     Core library (pip-installable)
  equations.py    Exact solutions, initial conditions, invariant evaluators
  network.py      MLP + hard boundary-condition ansatz (Lagaris-type)
  operators.py    Quadrature weights, difference matrices, GridOps
  projection.py   Affine (ce_poly) and Helmholtz (ce_helm) projection layers
  train.py        Trainer: Adam + L-BFGS, W&B logging, checkpointing

notebooks/   One self-contained Kaggle notebook per experiment. Each embeds
             the ce_pinn/ source inline, so it runs with no pip install and
             no git clone. Regenerate after any library change with
             `python tools/make_notebooks.py`; do not edit a notebook
             directly, the next regeneration overwrites it.

tools/       Analysis and verification scripts, run against results/ after
             training. Each writes a results/ANALYSIS_*.md file with the
             table it produced, so a reported number can be traced back to
             the script that computed it.

tests/       Unit tests (test_*.py) and standalone verification scripts
             (diag_*.py, verify_*.py).

results/     One directory per (problem, variant, seed), timestamped.
             See "What's archived" below.
```

## The six problems

| Notebook | Paper's name (Table 1) | Derivative share |
|---|---|---|
| `RLW1_single_soliton.ipynb` | RLW1 | 1.8% |
| `RLWA_moderate_share.ipynb` | RLW-A | 6.3% |
| `RLWB_high_share.ipynb` | RLW-B | 11.3% |
| `RLW1T_tall_soliton.ipynb` | RLW1T | 11.3% |
| `RLW2_two_soliton.ipynb` | RLW2 | 10.8% |
| `CH1_solitary_wave.ipynb` | CH1 | 28.0% |

`KdV1_single_soliton.ipynb`, `KdV2_two_soliton.ipynb` and
`RLW3_undular_bore.ipynb` also exist and run, but are not part of this
paper.

## Variants

| Label | Boundary/IC | $I_1$ | $I_2$ | Notes |
|---|---|---|---|---|
| `vanilla` | soft (loss penalty) | soft | soft | baseline, not tuned |
| `hard_bc` | hard | soft | soft | no projection |
| `ce_poly` | hard | hard | hard (affine) | prior-work-style projection |
| `ce_helm` | hard | hard | hard (Helmholtz) | this paper's operator |
| `ce_full` | hard | hard | hard (Helmholtz) | + soft penalty on $I_3$ |

## Running a notebook

1. Create a GitHub PAT with `repo` scope, and a free W&B account and API
   key.
2. On Kaggle: Add-ons -> Secrets -> add `GITHUB_TOKEN` and `WANDB_API_KEY`.
3. Accelerator = GPU T4 x2, Internet = ON.
4. Import the notebook from GitHub, set `VARIANT` and `SMOKE` at the top,
   Save & Run All.
5. On success (no NaN/Inf) the run pushes its output to `results/` on
   GitHub. Training is visible live on wandb.ai.

## What's archived under `results/<run>/<timestamp>/`

```
config.json          every parameter, including time_dependent_targets
                      and the CODE_VERSION digest the run was produced under
history.json          loss and conservation drift at every logged step
results.json           final errors and invariant drift
model_best.pth          lowest-loss checkpoint
model_final.pth          final checkpoint
solution_final.npz       predicted vs. exact solution, for plotting
```

`config.json`'s `code_version` is a hash of the five numerical modules
(`equations.py` through `train.py`, in that order) at the time of the run.
Any table in the paper can be traced to the exact code that produced it.
`tools/make_notebooks.py` computes and embeds it.

The `.pth` and `.npz` files are training output. A reader checking the
paper's claims does not need them; `config.json`, `results.json` and
`history.json` are the provenance record, and are what the analysis tools
in `tools/` read. This repository may be cited as the code-availability
source for a journal submission. If so, and the checkpoints are needed for
exact reproduction, check the paper's data-availability statement for an
archived copy. They may have moved to a DOI-bearing archive (Zenodo)
separate from the GitHub history, and would then not be present here at
every commit.

## Reproducing a table

```bash
git pull
python tools/verify_theory.py          # Theorems 3, 3', 6 against the sweep
python tools/ceiling_sweep.py          # Theorem 6 under controlled perturbation
python tools/certificate_sweep.py      # certificate tightness
python tools/newton_cost.py            # multiplier-solve cost and the lam2 clamp
python tools/eval_project_at_inference.py
python tools/make_paper_figures.py results/
```

Each writes `results/ANALYSIS_<NAME>.md`.

## Requirements

```bash
pip install -e .
```

All computation uses `torch.float64`.

## Citing this work

See [`CITATION.cff`](CITATION.cff) for the software. The paper itself is
not archived in this repository; a full citation will be added here once
it is published.

## License

MIT. See [`LICENSE`](LICENSE).

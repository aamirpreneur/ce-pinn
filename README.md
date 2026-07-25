# CE-PINN

Conservation-Embedded Physics-Informed Neural Networks with Helmholtz projection for derivative-dependent conservation laws.

Target: Journal of Computational Physics.

## Structure

```
ce_pinn/              Core library (pip-installable)
  equations.py        Exact solutions, ICs, conservation evaluators
  network.py          MLP + hard BC embedding (Lagaris-type)
  projection.py       Polynomial + Helmholtz projection layers
  train.py            Trainer with W&B logging + checkpointing

notebooks/            One notebook per experiment (run on Kaggle)
  KdV1_single_soliton.ipynb
  KdV2_two_soliton.ipynb
  RLW1_single_soliton.ipynb   ← main contribution
  RLW2_two_soliton.ipynb
  RLW3_undular_bore.ipynb
  KP1_line_soliton.ipynb
  KP2_lump.ipynb

tools/
  make_paper_figures.py       Load checkpoints → PDF figures + LaTeX tables

results/              Auto-populated by successful runs
```

## Running on Kaggle

1. Create a GitHub PAT with `repo` scope
2. Create a W&B account (free) and get your API key
3. In Kaggle notebook: Add-ons → Secrets → add `GITHUB_TOKEN` and `WANDB_API_KEY`
4. Set Accelerator = **GPU T4 x2**, Internet = ON
5. Import notebook from GitHub, set VARIANT and SMOKE, Save & Run All
6. Results push to GitHub only if the run succeeds (no NaN/Inf)
7. Monitor training live on wandb.ai

## Each notebook saves

```
results/<run_name>/<timestamp>/
  config.json           All parameters (reproducibility)
  history.json          Loss + conservation at every log step
  model_best.pth        Lowest-loss weights
  model_final.pth       Final weights + global_step + best_loss
  results.json          Final errors (L2 abs, L2 rel, Linf) + conservation drift
  solution_final.npz    Predicted + exact solution arrays (for plotting)
```

## Making paper figures

```bash
git pull
python tools/make_paper_figures.py results/
```

Produces PDF figures + LaTeX error table from all completed runs.

## Variants

| Label | BCs | I1 | I2 | I3 |
|-------|-----|----|----|-----|
| vanilla | soft | - | - | - |
| hard_bc | hard | - | - | - |
| ce_poly | hard | hard | hard(poly) | - |
| ce_helm | hard | hard | hard(deriv) | - |
| ce_full | hard | hard | hard | soft |

## Float64

All computation uses `torch.float64` throughout.

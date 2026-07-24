# CE-PINN

Conservation-Embedded Physics-Informed Neural Networks with Helmholtz projection for derivative-dependent conservation laws.

Target: Journal of Computational Physics.

## Structure

```
ce_pinn/              Core library
  equations.py        Exact solutions, ICs, conservation evaluators
  network.py          MLP + hard BC embedding (Lagaris-type)
  projection.py       Polynomial + Helmholtz projection layers
  train.py            Training loop (Adam -> L-BFGS)

experiments/          One folder per benchmark problem
  kdv_single/         KdV-1: single soliton
  kdv_two_soliton/    KdV-2: two-soliton collision
  rlw_single/         RLW-1: single soliton (main contribution)
  rlw_two_soliton/    RLW-2: two-soliton collision
  rlw_bore/           RLW-3: undular bore
  kp_line/            KP-1: line soliton (2D)
  kp_lump/            KP-2: lump solution (2D)

kaggle/               Kaggle GPU runner
  run.py              Target dispatcher + auto-push results
  notebook_cell.py    Paste into one Kaggle cell

results/              Auto-populated by Kaggle runs
```

## Running on Kaggle

1. Create a GitHub PAT with `repo` scope
2. Add it as Kaggle Secret named `GITHUB_TOKEN`
3. Set notebook Accelerator to **GPU T4 x2**, Internet ON
4. Paste `kaggle/notebook_cell.py` into one cell, set TARGET, Save & Run All
5. Results push back to `results/<target>/<timestamp>/`

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

# RLW1: projection at inference vs projection in training

All quantities are worst-case over 41 time levels on [0, 20], except
the L2 columns which are the mean, the value at t = T, and the worst.
`hard_bc+proj` is the `hard_bc` network with the Helmholtz projection
applied at evaluation time only; it was never trained with it.

| variant | L2 mean | L2 at T | L2 max | C1 drift | C2 drift | C3 drift | bc error |
|---|---|---|---|---|---|---|---|
| vanilla | 3.393e-01 | 5.283e-01 | 5.283e-01 | 2.74e-01 | 5.04e-01 | 5.16e-01 | 8.56e-04 |
| hard_bc | 1.011e-03 | 1.603e-03 | 1.603e-03 | 8.91e-04 | 5.74e-04 | 5.96e-04 | 0.00e+00 |
| hard_bc+proj | 9.944e-04 | 1.548e-03 | 1.548e-03 | 3.35e-16 | 3.42e-15 | 1.47e-06 | 0.00e+00 |
| ce_poly | 1.668e-03 | 2.596e-03 | 2.662e-03 | 4.46e-16 | 1.69e-05 | 1.77e-05 | 2.25e-04 |
| ce_helm | 2.277e-03 | 3.433e-03 | 3.433e-03 | 1.45e-15 | 1.22e-14 | 4.11e-06 | 6.07e-19 |
| ce_full | 2.420e-03 | 3.437e-03 | 3.483e-03 | 8.81e-15 | 2.64e-13 | 4.30e-06 | 1.04e-18 |

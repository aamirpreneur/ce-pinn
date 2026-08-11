# RLW1: projection at inference vs projection in training

All quantities are worst-case over 41 time levels on [0, 20], except
the L2 columns which are the mean, the value at t = T, and the worst.
`hard_bc+proj` is the `hard_bc` network with the Helmholtz projection
applied at evaluation time only; it was never trained with it.

Targets and boundary data are evaluated at the current time, matching
what the runs trained with. An earlier version of this script projected
against C1, C2 frozen at t = 0, which adds the error floor of THEORY.md
section 5b to every projected row.

| variant | L2 mean | L2 at T | L2 max | C1 drift | C2 drift | C3 drift | bc error |
|---|---|---|---|---|---|---|---|
| vanilla | 3.393e-01 | 5.283e-01 | 5.283e-01 | 2.74e-01 | 5.04e-01 | 5.16e-01 | 8.56e-04 |
| hard_bc | 1.002e-03 | 1.532e-03 | 1.535e-03 | 5.82e-04 | 5.84e-04 | 6.07e-04 | 1.27e-05 |
| hard_bc+proj | 9.922e-04 | 1.497e-03 | 1.508e-03 | 3.35e-16 | 8.22e-16 | 1.29e-06 | 0.00e+00 |
| ce_poly | 3.953e-03 | 7.873e-03 | 7.873e-03 | 4.46e-16 | 1.68e-04 | 1.61e-04 | 2.92e-04 |
| ce_helm | 9.659e-03 | 1.573e-02 | 1.573e-02 | 2.68e-15 | 4.98e-13 | 4.00e-05 | 2.72e-18 |
| ce_full | 2.420e-03 | 3.437e-03 | 3.483e-03 | 8.81e-15 | 2.63e-13 | 4.31e-06 | 1.59e-18 |

## Cost

Per evaluated time level on CPU, mean over 41 levels.

| variant | network (ms) | projection (ms) | training (s) |
|---|---|---|---|
| vanilla | 1.32 | 0.00 | 397 |
| hard_bc | 1.23 | 0.00 | 438 |
| hard_bc+proj | 1.31 | 11.41 | 438 |
| ce_poly | 1.19 | 0.25 | 572 |
| ce_helm | 1.30 | 13.19 | 3995 |
| ce_full | 1.32 | 13.17 | 2937 |

## Provenance

| variant | run | code_version | training_time_s |
|---|---|---|---|
| vanilla | `20260729-125019` | `e299b122b9be` | 397.3224275112152 |
| hard_bc | `20260803-071602` | `f226607bf810` | 438.4557275772095 |
| hard_bc+proj | `20260803-071602` | `f226607bf810` | 438.4557275772095 |
| ce_poly | `20260803-073544` | `f226607bf810` | 572.3306457996368 |
| ce_helm | `20260803-092617` | `f226607bf810` | 3995.303579568863 |
| ce_full | `20260729-134835` | `e299b122b9be` | 2937.374910593033 |

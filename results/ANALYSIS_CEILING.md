# The ceiling, tested on every problem without training

Theorem 6 concerns the operator and the problem, not the optimizer.
Perturbing the exact solution by a set amount and projecting tests it
on all four problems -- including the three no plain PINN can
propagate -- across eight orders of magnitude of error. Nothing is
excluded for having failed to train.

`ratio` is realized / predicted. Theorem 6 is first order, so it
should sit near 1 for small errors and depart as the remainder takes
over. Where it departs is the validity radius.

> **The ceiling VALUES here are not representative of real PINN
> error.** The perturbation is smooth and band-limited, and it aligns
> with the constraint gradients far better than a trained network's
> error does: it gives a visible fraction around 27% on RLW1 where the
> trained runs give 2.3-3.7%. So the ceilings below are roughly an
> order of magnitude above what a network actually sees.
>
> What this sweep establishes is that Theorem 6 *predicts* the
> realized improvement, whatever that improvement happens to be. That
> is a property of the operator and holds for any error shape, which
> is exactly why a synthetic perturbation is the right instrument for
> it. Read the ratio column, not the ceiling column.

## RLW1  (small soliton)

derivative share 1.8% · a plain PINN trains this: **yes**

Theorem 7 check, ‖P(u_exact) − u_exact‖ = `0.00e+00`

| perturbation | L2 before | visible | predicted ceiling | realized | ratio |
|---|---|---|---|---|---|
| 1e-08 | 1.000e-08 | 17.272% | 9.0451% | 9.0452% | 1.0000 |
| 1e-07 | 1.000e-07 | 17.272% | 9.0451% | 9.0452% | 1.0000 |
| 1e-06 | 1.000e-06 | 17.272% | 9.0451% | 9.0452% | 1.0000 |
| 1e-05 | 1.000e-05 | 17.272% | 9.0451% | 9.0454% | 1.0000 |
| 1e-04 | 1.000e-04 | 17.272% | 9.0451% | 9.0475% | 1.0003 |
| 1e-03 | 1.000e-03 | 17.272% | 9.0451% | 9.0684% | 1.0026 |
| 1e-02 | 1.000e-02 | 17.272% | 9.0451% | 9.2800% | 1.0260 |
| 1e-01 | 1.000e-01 | 17.272% | 9.0451% | 11.6299% | 1.2858 |
| 1e+00 | 1.000e+00 | 17.272% | 9.0451% | 40.1817% | 4.4424 |

Over the rows with L2 < 1e-02: ratio ranges 1.0000 to 1.0026, median 1.0000.

## RLW1T (tall soliton)

derivative share 11.3% · a plain PINN trains this: **no**

Theorem 7 check, ‖P(u_exact) − u_exact‖ = `2.75e-15`

| perturbation | L2 before | visible | predicted ceiling | realized | ratio |
|---|---|---|---|---|---|
| 1e-08 | 1.000e-08 | 18.714% | 9.8414% | 9.8369% | 0.9995 |
| 1e-07 | 1.000e-07 | 18.714% | 9.8414% | 9.8369% | 0.9995 |
| 1e-06 | 1.000e-06 | 18.714% | 9.8414% | 9.8369% | 0.9995 |
| 1e-05 | 1.000e-05 | 18.714% | 9.8414% | 9.8371% | 0.9996 |
| 1e-04 | 1.000e-04 | 18.714% | 9.8414% | 9.8391% | 0.9998 |
| 1e-03 | 1.000e-03 | 18.714% | 9.8414% | 9.8590% | 1.0018 |
| 1e-02 | 1.000e-02 | 18.714% | 9.8414% | 10.0593% | 1.0221 |
| 1e-01 | 1.000e-01 | 18.714% | 9.8414% | 12.2279% | 1.2425 |
| 1e+00 | 1.000e+00 | 18.714% | 9.8414% | 38.8641% | 3.9490 |

Over the rows with L2 < 1e-02: ratio ranges 0.9995 to 1.0221, median 0.9996.

## CH1   (Camassa-Holm)

derivative share 28.0% · a plain PINN trains this: **poorly**

Theorem 7 check, ‖P(u_exact) − u_exact‖ = `1.87e-15`

| perturbation | L2 before | visible | predicted ceiling | realized | ratio |
|---|---|---|---|---|---|
| 1e-08 | 1.000e-08 | 12.451% | 6.4324% | 6.4311% | 0.9998 |
| 1e-07 | 1.000e-07 | 12.451% | 6.4324% | 6.4311% | 0.9998 |
| 1e-06 | 1.000e-06 | 12.451% | 6.4324% | 6.4311% | 0.9998 |
| 1e-05 | 1.000e-05 | 12.451% | 6.4324% | 6.4310% | 0.9998 |
| 1e-04 | 1.000e-04 | 12.451% | 6.4324% | 6.4307% | 0.9997 |
| 1e-03 | 1.000e-03 | 12.451% | 6.4324% | 6.4271% | 0.9992 |
| 1e-02 | 1.000e-02 | 12.451% | 6.4324% | 6.3927% | 0.9938 |
| 1e-01 | 1.000e-01 | 12.451% | 6.4324% | 6.2465% | 0.9711 |
| 1e+00 | 1.000e+00 | 12.451% | 6.4324% | 19.9797% | 3.1061 |

Over the rows with L2 < 1e-02: ratio ranges 0.9992 to 0.9998, median 0.9998.

## RLW2  (two solitons)

derivative share 10.8% · a plain PINN trains this: **no**

Theorem 7 check, ‖P(u_exact) − u_exact‖ = `4.34e-15`

| perturbation | L2 before | visible | predicted ceiling | realized | ratio |
|---|---|---|---|---|---|
| 1e-08 | 1.000e-08 | 15.230% | 7.9294% | 7.9267% | 0.9996 |
| 1e-07 | 1.000e-07 | 15.230% | 7.9294% | 7.9267% | 0.9996 |
| 1e-06 | 1.000e-06 | 15.230% | 7.9294% | 7.9267% | 0.9997 |
| 1e-05 | 1.000e-05 | 15.230% | 7.9294% | 7.9268% | 0.9997 |
| 1e-04 | 1.000e-04 | 15.230% | 7.9294% | 7.9282% | 0.9998 |
| 1e-03 | 1.000e-03 | 15.230% | 7.9294% | 7.9420% | 1.0016 |
| 1e-02 | 1.000e-02 | 15.230% | 7.9294% | 8.0827% | 1.0193 |
| 1e-01 | 1.000e-01 | 15.230% | 7.9294% | 9.6942% | 1.2226 |
| 1e+00 | 1.000e+00 | 15.230% | 7.9294% | 34.7243% | 4.3792 |

Over the rows with L2 < 1e-02: ratio ranges 0.9996 to 1.0016, median 0.9997.

## Summary

| problem | derivative share | ratio min | max | median | Thm 7 |
|---|---|---|---|---|---|
| RLW1  (small soliton) | 1.8% | 1.0000 | 1.0026 | **1.0000** | 0.0e+00 |
| RLW1T (tall soliton) | 11.3% | 0.9995 | 1.0221 | **0.9996** | 2.8e-15 |
| CH1   (Camassa-Holm) | 28.0% | 0.9992 | 0.9998 | **0.9998** | 1.9e-15 |
| RLW2  (two solitons) | 10.8% | 0.9996 | 1.0016 | **0.9997** | 4.3e-15 |

Median ratio across all four problems: 0.9996 to 1.0000.


# Newton cost and the lam2 clamp

Iterations to converge and clamp rate, over 8 perturbation shapes per amplitude.
`amp` is the relative L2 of the perturbation applied to the exact
solution, so `1e-08` stands for a converged field and `1e0` for the
first steps of training. `clamp %` is the fraction of solves that hit
the well-posedness bound on lam2.


## RLW1  (small soliton)

N = 401, dx = 0.2500, mu = 1, lam2_min = -1.387e-02

| amp | iters median | iters max | clamp % | resid max | ms |
|---|---|---|---|---|---|
| 1e-08 | 2 | 2 | 0% | 2.0e-15 | 7.9 |
| 1e-06 | 2 | 2 | 0% | 2.3e-13 | 6.8 |
| 1e-04 | 3 | 3 | 0% | 1.3e-15 | 10.5 |
| 1e-02 | 4 | 4 | 0% | 2.7e-13 | 16.2 |
| 1e-01 | 27 | 50 | 50% | 6.2e-02 | 619.0 |
| 1e+00 | 5 | 50 | 12% | 1.5e-01 | 168.3 |

## RLW1T (tall soliton)

N = 601, dx = 0.1667, mu = 1, lam2_min = -6.211e-03

| amp | iters median | iters max | clamp % | resid max | ms |
|---|---|---|---|---|---|
| 1e-08 | 2 | 2 | 0% | 5.0e-14 | 16.5 |
| 1e-06 | 3 | 3 | 0% | 7.3e-13 | 25.3 |
| 1e-04 | 3 | 3 | 0% | 3.6e-14 | 23.3 |
| 1e-02 | 4 | 4 | 0% | 3.6e-14 | 31.0 |
| 1e-01 | 27 | 50 | 50% | 6.8e+00 | 1376.0 |
| 1e+00 | 5 | 8 | 0% | 9.7e-13 | 44.1 |

## CH1   (Camassa-Holm)

N = 601, dx = 0.0667, mu = 1, lam2_min = -9.990e-04

| amp | iters median | iters max | clamp % | resid max | ms |
|---|---|---|---|---|---|
| 1e-08 | 2 | 2 | 0% | 2.4e-15 | 16.1 |
| 1e-06 | 3 | 3 | 0% | 1.9e-13 | 22.1 |
| 1e-04 | 3 | 3 | 0% | 2.7e-15 | 24.9 |
| 1e-02 | 4 | 50 | 25% | 4.1e-03 | 641.4 |
| 1e-01 | 5 | 50 | 25% | 7.9e-02 | 611.8 |
| 1e+00 | 6 | 9 | 0% | 6.3e-14 | 51.5 |

## RLW2  (two solitons)

N = 601, dx = 0.2000, mu = 1, lam2_min = -8.920e-03

| amp | iters median | iters max | clamp % | resid max | ms |
|---|---|---|---|---|---|
| 1e-08 | 2 | 2 | 0% | 9.9e-14 | 16.8 |
| 1e-06 | 3 | 3 | 0% | 1.4e-13 | 24.2 |
| 1e-04 | 3 | 3 | 0% | 6.4e-14 | 23.9 |
| 1e-02 | 4 | 4 | 0% | 7.1e-14 | 31.3 |
| 1e-01 | 5 | 50 | 25% | 2.4e+00 | 712.7 |
| 1e+00 | 5 | 7 | 0% | 7.2e-13 | 43.1 |

## How much of an I2 deficit the solve tolerates

The bound restricts lam2 from below only, so it can bite only when
lam2 < 0. That is exactly the case where the field carries too little
I2, which is what an under-resolved gradient produces. So the number
a practitioner needs is how large an I2 deficit the solve accepts
before the bound stops it.

Imposed along a fixed direction (the smoothed exact solution) and
bisected on the deficit, rather than by stepping a smoothing kernel:
one kernel step means different things on different grids.

| problem | dx | mu | lam2_min | tolerable I2 deficit |
|---|---|---|---|---|
| RLW1  (small soliton) | 0.2500 | 1 | -1.387e-02 | 2.266% |
| RLW1T (tall soliton) | 0.1667 | 1 | -6.211e-03 | 1.353% |
| CH1   (Camassa-Holm) | 0.0667 | 1 | -9.990e-04 | 0.373% |
| RLW2  (two solitons) | 0.2000 | 1 | -8.920e-03 | 1.863% |

The tolerance tracks the bound, so the finest grid is the least
forgiving. Refining to improve the solution narrows what the
projection is allowed to correct.


## Summary

Over every problem at amplitudes up to 1e-02, which is the range in
which the trained runs sit:

* iterations: median 3, worst case 50
* clamp rate: 25% at worst

The clamp is what makes the count grow, not the nonlinearity: a
solve that clamps cannot satisfy F2 and exits on max_iter.


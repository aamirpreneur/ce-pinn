# Numerical verification of the theory

Checked against the stored RLW1 checkpoints at `t = T`, using the
trainer's own `q` and `D1c`, so these test the discrete operators the
code uses rather than their continuous idealisations.

## Theorem 2 — a posteriori certificate

    || u~ - u ||_A  >=  | sqrt(I2(u~)) - sqrt(C2) |

The right-hand side is computable without knowing the exact solution.
`holds` must be true for every row; a false entry means the algebra is
wrong. `tightness` is certificate / true error — how much of the real
error the certificate actually detects.

| variant | L2 rel | true ‖e‖_A | certificate | holds | tightness |
|---|---|---|---|---|---|
| `vanilla` | 5.283e-01 | 4.827e-01 | 2.647e-01 | yes | 5.48e-01 |
| `hard_bc` | 1.603e-03 | 1.783e-03 | 2.583e-04 | yes | 1.45e-01 |
| `ce_poly` | 7.339e+00 | 6.607e+00 | 6.607e+00 | yes | 1.00e+00 |
| `ce_helm` | 1.270e+00 | 1.144e+00 | 1.144e+00 | yes | 1.00e+00 |
| `ce_full` | 2.220e+00 | 2.000e+00 | 2.000e+00 | yes | 1.00e+00 |

The same construction on the linear invariant, in the L2 norm:

| variant | true ‖e‖_h | certificate from I1 | holds |
|---|---|---|---|
| `vanilla` | 4.714e-01 | 1.071e-01 | yes |
| `hard_bc` | 1.430e-03 | 3.544e-04 | yes |
| `ce_poly` | 6.548e+00 | 2.906e+00 | yes |
| `ce_helm` | 1.133e+00 | 5.013e-01 | yes |
| `ce_full` | 1.981e+00 | 8.856e-01 | yes |

## Proposition — codimension 2

`visible` is ‖e_N‖²/‖e‖², the fraction of error energy lying in
span{grad I1, grad I2}. Everything else is invisible to these two
invariants and cannot be removed by any projection built on them.
`2/N` is the value expected for an isotropic error in N dimensions.

| variant | N | visible | invisible | 2/N |
|---|---|---|---|---|
| `vanilla` | 401 | 56.953% | 43.047% | 0.499% |
| `hard_bc` | 401 | 6.722% | 93.278% | 0.499% |
| `ce_poly` | 401 | 99.925% | 0.075% | 0.499% |
| `ce_helm` | 401 | 99.994% | 0.006% | 0.499% |
| `ce_full` | 401 | 99.996% | 0.004% | 0.499% |

## Corollary — a ceiling on the benefit of conservation

Only `e_N` can be removed, so any projection enforcing I1 and I2 has
its improvement bounded by `1 - sqrt(1 - visible)`. This bounds every
method built on these two invariants, not only this one. `realised` is
what the Helmholtz projection actually achieves on the same field.

| variant | visible | predicted ceiling | realised | ratio |
|---|---|---|---|---|
| `vanilla` | 56.953% | 34.390% | 0.041% | 0.001 |
| `hard_bc` | 6.722% | 3.420% | 3.402% | 0.995 |
| `ce_poly` | 99.925% | 97.253% | 99.696% | 1.025 |
| `ce_helm` | 99.994% | 99.257% | 99.730% | 1.005 |
| `ce_full` | 99.996% | 99.378% | 99.845% | 1.005 |

## Verdict

* Theorem 2 (H1 certificate): holds on every run
* I1 certificate: holds on every run
* visible fraction for `hard_bc`: 6.722%, so 93.278% of the error is
  beyond the reach of these invariants by construction.


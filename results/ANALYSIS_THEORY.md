# Verification of MATH.md against the sweep

> **WARNING — mixed code versions: ['e299b122b9be', 'f226607bf810'].** Results from different versions are not comparable; rerun the stragglers before drawing any conclusion from the table below.

## Theorems 3 and 3′ — certificates

Computable from the network output alone. `holds` must be true everywhere.

| run | ‖e‖_A | certificate | holds | ‖e‖_h | cert (I1) | holds |
|---|---|---|---|---|---|---|
| `rlw1_vanilla/20260729-125019` | 4.827e-01 | 2.647e-01 | yes | 4.714e-01 | 1.071e-01 | yes |
| `rlw1_hard_bc/20260729-125832` | 1.783e-03 | 2.352e-04 | yes | 1.430e-03 | 3.238e-04 | yes |
| `rlw1_hard_bc/20260803-071602` | 1.698e-03 | 2.399e-04 | yes | 1.362e-03 | 2.031e-04 | yes |
| `rlw1_hard_bc_s1/20260803-071851` | 3.723e-03 | 3.283e-04 | yes | 3.257e-03 | 1.763e-04 | yes |
| `rlw1_hard_bc_s2/20260803-073456` | 1.200e-03 | 1.163e-06 | yes | 9.752e-04 | 1.478e-04 | yes |
| `rlw1_ce_poly/20260729-123957` | 2.629e-03 | 5.366e-06 | yes | 2.312e-03 | 0.000e+00 | yes |
| `rlw1_ce_poly/20260803-073544` | 7.974e-03 | 3.684e-06 | yes | 7.023e-03 | 8.882e-17 | yes |
| `rlw1_ce_helm/20260729-131851` | 3.563e-03 | 6.550e-15 | yes | 3.061e-03 | 2.665e-16 | yes |
| `rlw1_ce_helm/20260803-092617` | 1.458e-02 | 2.249e-13 | yes | 1.404e-02 | 4.885e-16 | yes |
| `rlw1_ce_full/20260729-134835` | 3.518e-03 | 8.282e-14 | yes | 3.068e-03 | 6.217e-16 | yes |
| `rlw1t_hard_bc/20260803-075359` | 8.714e+00 | 1.332e+00 | yes | 7.650e+00 | 5.759e-01 | yes |
| `rlwa_hard_bc/20260808-041608` | 5.991e-03 | 7.126e-04 | yes | 4.662e-03 | 5.698e-05 | yes |
| `rlwa_ce_poly/20260808-041824` | 1.406e-02 | 1.362e-04 | yes | 1.054e-02 | 3.003e-16 | yes |
| `rlwb_hard_bc/20260807-112911` | 3.693e-02 | 2.821e-03 | yes | 2.753e-02 | 2.457e-03 | yes |
| `rlwb_hard_bc_s1/20260808-002637` | 5.525e-02 | 3.261e-03 | yes | 4.201e-02 | 3.773e-04 | yes |
| `rlwb_hard_bc_s2/20260808-002633` | 5.518e-02 | 3.624e-03 | yes | 4.393e-02 | 2.722e-03 | yes |
| `rlwb_ce_poly/20260807-114816` | 7.654e-02 | 4.138e-03 | yes | 5.945e-02 | 0.000e+00 | yes |
| `rlwb_ce_helm/20260807-152938` | 2.907e+00 | 1.066e-14 | yes | 2.446e+00 | 1.180e-14 | yes |
| `ch1_hard_bc/20260803-083812` | 3.145e+00 | 2.239e+00 | yes | 3.056e+00 | 2.331e+00 | yes |
| `ch1_hard_bc/20260803-105750` | 2.551e-01 | 2.299e-02 | yes | 1.508e-01 | 2.872e-02 | yes |
| `ch1_hard_bc/20260803-115245` | 8.499e-02 | 1.694e-03 | yes | 4.548e-02 | 9.530e-03 | yes |

## Theorem 6 — the ceiling. This is the paper’s main figure.

Theorem 6 bounds the improvement by `1 - sqrt(1 - V) + C||e||`. The
`ratio` column compares the realised improvement against the *leading*
term only. Rows marked † have a ceiling smaller than twice the
relative error, so the `C||e||` remainder is comparable to the quantity
being predicted and the ratio should not be read as a test of the
theorem. A ratio away from 1 there is a statement about the validity
radius of a first-order expansion, not about whether Theorem 6 holds.

Rows marked ‡ are NOT direct tests of Theorem 6. Those variants
project during training, so their error is measured after their own
projection and the ceiling asks how much a further Helmholtz
projection could remove from an already-projected field. For ce_helm
Theorem 7 says that is ~0, and it is. The unmarked rows are the
direct verifications.

| problem | variant | seed | version | L2 | visible | predicted ceiling | realised | ratio |
|---|---|---|---|---|---|---|---|---|
| RLW1 (small soliton) | `vanilla` | 0 | `e299b122b9be` ⚠ | 5.283e-01 | 56.957% | **34.393%** | **2.955%** | 0.086 † |
| RLW1 (small soliton) | `hard_bc` | 0 | `e299b122b9be` ⚠ | 1.603e-03 | 5.657% | **2.869%** | **2.850%** | 0.993 |
| RLW1 (small soliton) | `hard_bc` | 0 | `f226607bf810` | 1.526e-03 | 3.733% | **1.884%** | **1.852%** | 0.983 |
| RLW1 (small soliton) | `hard_bc` | 1 | `f226607bf810` | 3.651e-03 | 2.257% | **1.135%** | **1.063%** | 0.936 |
| RLW1 (small soliton) | `hard_bc` | 2 | `f226607bf810` | 1.093e-03 | 2.820% | **1.420%** | **1.433%** | 1.009 |
| RLW1 (small soliton) | `ce_poly` | 0 | `e299b122b9be` ⚠ | 2.594e-03 | 0.225% | **0.113%** | **0.224%** | 1.983 † ‡ |
| RLW1 (small soliton) | `ce_poly` | 0 | `f226607bf810` | 7.873e-03 | 0.034% | **0.017%** | **0.031%** | 1.805 † ‡ |
| RLW1 (small soliton) | `ce_helm` | 0 | `e299b122b9be` ⚠ | 3.431e-03 | 0.001% | **0.000%** | **0.000%** | 0.000 † ‡ |
| RLW1 (small soliton) | `ce_helm` | 0 | `f226607bf810` | 1.573e-02 | 0.009% | **0.004%** | **0.000%** | 0.000 † ‡ |
| RLW1 (small soliton) | `ce_full` | 0 | `e299b122b9be` ⚠ | 3.438e-03 | 0.001% | **0.000%** | **0.000%** | 0.000 † ‡ |
| RLW1T (tall soliton) | `hard_bc` | 0 | `f226607bf810` | 7.861e-01 | 40.582% | **22.917%** | **0.031%** | 0.001 † |
| RLW-A (share 6.2%) | `hard_bc` | 0 | `f226607bf810` | 8.350e-04 | 2.491% | **1.253%** | **1.275%** | 1.017 |
| RLW-A (share 6.2%) | `ce_poly` | 0 | `f226607bf810` | 1.887e-03 | 0.072% | **0.036%** | **0.069%** | 1.919 † ‡ |
| RLW-B (share 11.3%) | `hard_bc` | 0 | `f226607bf810` | 1.414e-03 | 1.234% | **0.619%** | **0.604%** | 0.976 |
| RLW-B (share 11.3%) | `hard_bc` | 1 | `f226607bf810` | 2.158e-03 | 0.714% | **0.358%** | **0.328%** | 0.918 † |
| RLW-B (share 11.3%) | `hard_bc` | 2 | `f226607bf810` | 2.257e-03 | 0.723% | **0.362%** | **0.344%** | 0.949 † |
| RLW-B (share 11.3%) | `ce_poly` | 0 | `f226607bf810` | 3.054e-03 | 0.476% | **0.238%** | **0.279%** | 1.170 † ‡ |
| RLW-B (share 11.3%) | `ce_helm` | 0 | `f226607bf810` | 1.257e-01 | 0.712% | **0.356%** | **-0.000%** | -0.000 † ‡ |
| CH1 (Camassa-Holm) | `hard_bc` | 0 | `f226607bf810` | 2.891e+00 | 59.273% | **36.182%** | **61.483%** | 1.699 † |
| CH1 (Camassa-Holm) | `hard_bc` | 0 | `f226607bf810` | 1.427e-01 | 11.413% | **5.879%** | **1.859%** | 0.316 † |
| CH1 (Camassa-Holm) | `hard_bc` | 0 | `f226607bf810` | 4.303e-02 | 5.835% | **2.962%** | **2.275%** | 0.768 † |

## Error in three norms

| run | L2 rel | H1 rel | L-inf |
|---|---|---|---|
| `rlw1_vanilla/20260729-125019` | 5.283e-01 | 5.362e-01 | 1.616e-01 |
| `rlw1_hard_bc/20260729-125832` | 1.603e-03 | 1.981e-03 | 4.282e-04 |
| `rlw1_hard_bc/20260803-071602` | 1.526e-03 | 1.886e-03 | 4.545e-04 |
| `rlw1_hard_bc_s1/20260803-071851` | 3.651e-03 | 4.136e-03 | 8.415e-04 |
| `rlw1_hard_bc_s2/20260803-073456` | 1.093e-03 | 1.333e-03 | 3.755e-04 |
| `rlw1_ce_poly/20260729-123957` | 2.594e-03 | 2.920e-03 | 8.323e-04 |
| `rlw1_ce_poly/20260803-073544` | 7.873e-03 | 8.857e-03 | 1.935e-03 |
| `rlw1_ce_helm/20260729-131851` | 3.431e-03 | 3.958e-03 | 8.190e-04 |
| `rlw1_ce_helm/20260803-092617` | 1.573e-02 | 1.620e-02 | 3.669e-03 |
| `rlw1_ce_full/20260729-134835` | 3.438e-03 | 3.908e-03 | 8.516e-04 |
| `rlw1t_hard_bc/20260803-075359` | 7.861e-01 | 8.433e-01 | 3.562e+00 |
| `rlwa_hard_bc/20260808-041608` | 8.350e-04 | 1.039e-03 | 9.335e-04 |
| `rlwa_ce_poly/20260808-041824` | 1.887e-03 | 2.439e-03 | 1.851e-03 |
| `rlwb_hard_bc/20260807-112911` | 1.414e-03 | 1.786e-03 | 4.292e-03 |
| `rlwb_hard_bc_s1/20260808-002637` | 2.158e-03 | 2.673e-03 | 8.056e-03 |
| `rlwb_hard_bc_s2/20260808-002633` | 2.257e-03 | 2.670e-03 | 6.912e-03 |
| `rlwb_ce_poly/20260807-114816` | 3.054e-03 | 3.703e-03 | 1.596e-02 |
| `rlwb_ce_helm/20260807-152938` | 1.257e-01 | 1.406e-01 | 4.218e-01 |
| `ch1_hard_bc/20260803-083812` | 2.891e+00 | 2.525e+00 | 8.339e-01 |
| `ch1_hard_bc/20260803-105750` | 1.427e-01 | 2.048e-01 | 9.856e-02 |
| `ch1_hard_bc/20260803-115245` | 4.303e-02 | 6.825e-02 | 3.455e-02 |

## Verdict

* 21 runs analysed
* Theorem 3 (H1 certificate): holds everywhere
* Theorem 3′ (L2 certificate): holds everywhere
* realised/predicted ceiling over 5 current-version runs: min 0.936, max 1.017, median 0.983
* 5 run(s) at a superseded CODE_VERSION are shown for reference but excluded from the summary; they used frozen boundary data, so their ceilings are not comparable


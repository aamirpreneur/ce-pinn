# Certificate tightness across all four problems

`tightness` is bound / true error: the fraction of the error the
certificate detects. All three bounds are computable without the
exact solution. Eight perturbation shapes per problem, since the
visible fraction depends on how the error aligns with the constraint
gradients rather than on the problem alone.

| problem | share | Thm 3′ (I1, L2) | Thm 3 (I2, H1) | combined (all four) | combined spread | gain over 3′ |
|---|---|---|---|---|---|---|
| RLW1 | 1.8% | 47.34% | 29.00% | **52.13%** | 24.8–85.3% | 1.1× |
| RLW1T | 11.3% | 47.34% | 18.50% | **54.18%** | 28.5–78.2% | 1.1× |
| CH1 | 28.0% | 47.34% | 49.53% | **56.46%** | 11.7–74.7% | 1.2× |
| RLW2 | 10.8% | 47.34% | 7.72% | **49.78%** | 6.5–75.5% | 1.1× |

Medians over 8 perturbation shapes x 4 amplitudes = 32 cases each.
The `spread` column is the min and max over those 32 cases.

## Reading

The median tightness is 50–56% across problems, which places the error within a factor of 1.8–2.0 below the bound.

That median is not the headline. Across perturbation shapes the same
quantity runs 6.5% to 85.3%, a spread far wider than
the variation between problems. Tightness is a property of the error,
not of the problem, so a synthetic study of this certificate can
report almost any figure depending on which perturbations it uses.
The number that matters is the one measured on the error a trained
network actually makes, which `tools/verify_theory.py` reports.

It is provably at least as sharp as Theorem 3′, since both are norms
of orthogonal projections of the error and span{1} lies inside the
four-dimensional normal space. The measured gain quantifies how much
the other three violations add.

Its tightness equals sqrt(V), so a problem whose errors align well
with the constraint gradients gets both a sharper certificate and a
higher ceiling. Those are the same quantity seen twice, which is
worth stating plainly rather than presenting as two results.


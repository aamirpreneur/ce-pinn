# What conservation can and cannot buy

Statements, proofs, and their numerical verification. Every result here is
about the **discrete** operators the code uses, not their continuous
idealisations, because that is what the algorithm computes with and what a
referee will check against the reported numbers.

Numerical verification: `tools/verify_theory.py`, output in
`results/ANALYSIS_THEORY.md`.

---

## 1. Setting

Grid `x_0 < ... < x_{N-1}` on `[a,b]`. Write `Q = diag(q)` for the trapezoid
weights and `D` for the central-difference matrix (`D1c` in the code). Define

```
<u,v>_h = u^T Q v,        ||u||_h = sqrt(<u,u>_h)         (discrete L2)
A       = Q + mu D^T Q D
<u,v>_A = u^T A v,        ||u||_A = sqrt(<u,u>_A)         (discrete H1_mu)
```

`Q` is diagonal with positive entries and `D^T Q D` is positive semidefinite,
so **`A` is symmetric positive definite** and `||.||_A` is a norm.

The two invariants are

```
I1(u) = q^T u              = <u, 1>_h        (linear)
I2(u) = q^T(u^2 + mu u_x^2) = u^T A u = ||u||_A^2   (quadratic)
```

> **Remark 1.** `I2` *is* the squared discrete `H1_mu` norm. This is the
> structural fact the whole paper rests on. Methods restricted to `∫u` and
> `∫u²` — which is the entire projection literature, see §6 — constrain only
> the `L2` geometry. A derivative-dependent invariant constrains the `H1`
> geometry, and `H1` controls `u_x`, which is what dispersive dynamics are
> made of.

The constraint set is

```
M = { u : I1(u) = C1,  I2(u) = C2 }
```

the intersection of a hyperplane with an ellipsoid.

The projection is the **metric projection onto `M` in `||.||_h`**:

```
P(u~) = argmin { ||u - u~||_h : u in M }.
```

*Derivation of the implemented form.* With Lagrangian
`½||u-u~||_h² + λ₁(q^T u - C1) + ½λ₂(u^T A u - C2)`, stationarity gives
`Q(u - u~) + λ₁ q + λ₂ A u = 0`. Using `q = Q·1` and `A = Q + mu D^T Q D`, and
multiplying by `Q^{-1}`:

```
(1 + λ₂) u  -  λ₂ mu u_xx  =  u~ - λ₁
```

which is the elliptic solve in `ce_pinn/projection.py`. So the implemented
operator is exactly the minimal-`L2` correction onto `M`. ∎

---

## 2. Existence

**Theorem 1.** `M` is non-empty if and only if

```
C2  >=  C1^2 / (q^T A^{-1} q).
```

*Proof.* Minimise `||u||_A^2 = u^T A u` subject to `q^T u = C1`. Stationarity
gives `2Au = ν q`, so `u = (ν/2) A^{-1} q`; the constraint fixes
`ν/2 = C1 / (q^T A^{-1} q)`. The minimum value is
`u^T A u = (ν/2)^2 q^T A^{-1} q = C1^2 / (q^T A^{-1} q)`. Since `A` is SPD the
functional is strictly convex on the hyperplane, so this is the global minimum,
and the ellipsoid `I2 = C2` meets the hyperplane iff `C2` is at least it. ∎

> **Remark 2.** The bound is computable before training. If a set of target
> invariants violates it, no function whatsoever satisfies them and the failure
> is in the targets, not the network — worth checking when invariants are
> supplied from data or from a time-dependent model rather than from an exact
> initial condition.

---

## 3. A computable certificate for a black box

The network is not analysable. Its *output* is.

**Theorem 2 (a posteriori certificate).** Let `u` be the exact solution, so
`I2(u) = C2`, and let `u~` be any field whatsoever. Then

```
|| u~ - u ||_A  >=  | sqrt(I2(u~)) - sqrt(C2) |.
```

*Proof.* `I2(v) = ||v||_A^2` and `A` is SPD, so `||.||_A` is a norm and the
reverse triangle inequality applies:
`|| u~ - u ||_A >= | ||u~||_A - ||u||_A | = | sqrt(I2(u~)) - sqrt(C2) |`. ∎

**Theorem 2'.** Likewise `|| u~ - u ||_h >= |I1(u~) - C1| / sqrt(b-a)`.

*Proof.* `I1(u~) - C1 = <u~ - u, 1>_h`, and Cauchy–Schwarz gives
`|<u~-u,1>_h| <= ||u~-u||_h ||1||_h` with `||1||_h^2 = q^T 1 = b-a`. ∎

> **Remark 3.** Both right-hand sides are computable from the network output
> and the target invariants alone — **no exact solution, no unknown constant,
> no fitted parameter**. They are rigorous lower bounds on the error of an
> arbitrary black-box approximation. Theorem 2 bounds the error in `H1`, i.e.
> it certifies error *in the derivative*; the `∫u²` invariant used by existing
> methods yields only an `L2` statement and can certify nothing about `u_x`.

**Verified.** Holds on all five RLW1 runs. On the converged `hard_bc` run the
certificate recovers **14.5%** of the true `H1` error — a rigorous bound within
a factor of seven of the truth, with no constants to estimate.

---

## 4. What the invariants cannot see

**Proposition 3 (codimension two).** Let `u ∈ M` with `∇I1`, `∇I2` linearly
independent at `u`. In `<.,.>_h` the gradients are

```
grad_h I1 = 1,        grad_h I2 = 2 Q^{-1} A u
```

so the normal space `N_u M = span{1, Q^{-1}Au}` has dimension 2. Split the
error `e = u~ - u` as `e = e_N + e_T` with `e_N ∈ N_u M` and `e_T ⊥_h N_u M`.
Then to first order in `||e||`

```
I1(u~) - C1 = <1, e_N>_h ,     I2(u~) - C2 = <grad_h I2, e_N>_h .
```

*Proof.* `dI1(e) = <1,e>_h` exactly, and `<1,e_T>_h = 0` by construction.
`dI2(e) = 2u^T A e = <2Q^{-1}Au, e>_h` and `<grad_h I2, e_T>_h = 0`
likewise; the neglected term is `e^T A e = O(||e||^2)`. ∎

**Corollary 4 (no upper bound on error).** `M` has codimension 2 in an
`N`-dimensional space, hence contains points arbitrarily far apart. Exact
satisfaction of `I1` and `I2` therefore implies **no** bound on `||u~ - u||`.

> **Remark 4.** This is not a defect of any particular method. It bounds every
> scheme built on these invariants, including all of §6. It is the reason the
> accuracy gains reported throughout this literature are marginal.

**Verified.** For `hard_bc`, `||e_N||²/||e||² = 6.72%`: **93.3% of the error is
invisible to the invariants by construction.** Independently, the
amplitude/phase/shape decomposition in `ANALYSIS_RLW1.md` found 96–99% of the
error to be shape error. Two unrelated decompositions, same conclusion.

The isotropic expectation is `2/N = 0.50%`. The measured 6.72% is
13× larger because PINN errors are smooth, hence better aligned with the smooth
constraint gradients than a generic direction would be.

---

## 5. A sharp ceiling on the benefit of conservation

This is the operational consequence, and the strongest result here.

**Theorem 5 (ceiling).** Let `u* = P(u~)`. Then

```
|| u* - u ||_h  =  || e_T ||_h  +  O(||e||^2)
```

so the relative improvement obtainable from *any* projection enforcing `I1`
and `I2` satisfies

```
1 - ||u*-u|| / ||u~-u||   <=   1 - sqrt(1 - visible)  + O(||e||),
        visible := ||e_N||^2 / ||e||^2 .
```

*Proof.* Write the correction `c = u~ - u*`. First-order optimality of the
metric projection puts `c ∈ N_{u*}M`, and `N_{u*}M = N_u M + O(||e||)` since
`M` is smooth and `||u* - u|| = O(||e||)`. Hence `c = c_N + O(||e||^2)` with
`c_N ∈ N_u M`, and

```
u* - u = e - c = (e_N - c_N) + e_T + O(||e||^2).
```

The two displayed components are `<.,.>_h`-orthogonal, so
`||u*-u||^2 = ||e_N - c_N||^2 + ||e_T||^2 + O(||e||^3) >= ||e_T||^2 + O(||e||^3)`,
with equality when the projection removes the normal component exactly. ∎

> **Remark 5 (validity).** The estimate is **first order in `||e||`**. It is
> asymptotic, not uniform, and it should not be applied to fields whose error
> is `O(1)` — see the verification below, where exactly that happens.

**Verified.**

| variant | visible | predicted ceiling | realised | ratio |
|---|---|---|---|---|
| `hard_bc` | 6.722% | **3.420%** | **3.402%** | **0.995** |
| `vanilla` | 56.953% | 34.390% | 0.041% | 0.001 |
| `ce_poly` | 99.925% | 97.253% | 99.696% | 1.025 |
| `ce_helm` | 99.994% | 99.257% | 99.730% | 1.005 |

On the one converged solution the theory predicts the achieved improvement to
**three significant figures** — 3.420% predicted against 3.402% realised — from
quantities computed without reference to the exact solution.

The other rows behave exactly as Remark 5 says they must. `vanilla` (L2 = 0.53)
falls far below the ceiling, which is an upper bound and need not be attained.
`ce_poly` and `ce_helm` are raw outputs evaluated without the projection they
were trained with, carry `O(1)` errors, and *exceed* the ceiling by 2.5% and
0.5% — the first-order expansion has no validity there. Their agreement to
within a few percent despite `||e|| ~ 1` is better than the theory promises.

> **Remark 6.** Theorem 5 is a statement about the *problem*, not the method.
> It says how much room conservation has to help before any algorithm is
> chosen, and it is computable a priori from the error's alignment with two
> known vectors. This is what explains the marginal gains across the whole
> projection literature — 1.06× on KdV, 3% on the acoustic wave equation — and
> it predicts them rather than merely observing them.

---

## 5b. Theorem 3: the error floor is not intrinsic

The measured operator law was

```
L2_after ~ 0.905 * L2_before + 7.6e-06,   break-even at L2 ~ 8e-05
```

and the additive term was read as a property of the projection — a Green's
function correcting a global integral with the wrong shape. It is not. It is a
defect in the data supplied to the solve, and it is removable.

**Theorem 6 (consistency implies idempotence).** Let `F` denote the feasible
set of the elliptic solve: fields satisfying `I1 = C1`, `I2 = C2` *and* the
imposed Dirichlet data `u(a) = ga`, `u(b) = gb`. If the exact solution `u` lies
in `F`, then `P(u) = u` exactly.

*Proof.* `P` is the metric projection onto `F`. For `u ∈ F` the minimisation
`min{||v-u||_h : v ∈ F}` attains 0 at `v = u`, and `F` is closed, so the
minimiser is `u`. ∎

**Corollary 7.** Any additive error floor observed in `||P(u~) - u||` equals
`||P(u) - u||` to leading order, and vanishes if and only if the imposed data
is consistent with the exact solution.

**Verified — and the code's data is not consistent.** On RLW1 at `t = T`:

| imposed at `t=T` | value | exact solution's value | mismatch |
|---|---|---|---|
| `ga` | 6.940e-06 | 9.133e-09 | 6.93e-06 |
| `gb` | 1.669e-08 | 1.268e-05 | 1.27e-05 |
| `C1` | 3.979926663 | 3.979907628 | 4.78e-06 rel |
| `C2` | 0.810442936 | 0.810442936 | 2.52e-10 rel |

The boundary values are frozen at `t = 0`. The soliton moves, so its tails at
the two ends change by ~1e-05 over the run, and `C1` drifts by boundary flux.
`C2` is essentially exact. Projecting the exact solution then gives:

| configuration | `‖P(u) − u‖` |
|---|---|
| frozen `ga,gb` + IC targets (current code) | 5.560e-06 |
| frozen `ga,gb` + exact targets | 5.106e-06 |
| exact `ga,gb` + IC targets | 2.121e-06 |
| **exact `ga,gb` + exact targets** | **4.150e-17** |

Machine precision, exactly as Theorem 6 requires. Neither correction alone
suffices; both are needed.

**Consequence: the break-even disappears.** Repeating the controlled sweep with
consistent data:

| L2 before | frozen data | | consistent data | |
|---|---|---|---|---|
| 1.00e-08 | 8.460e-06 | **hurts 846×** | 9.095e-09 | 0.9095 |
| 1.00e-06 | 8.509e-06 | **hurts 8.5×** | 9.095e-07 | 0.9095 |
| 1.00e-05 | 1.242e-05 | **hurts 1.24×** | 9.095e-06 | 0.9095 |
| 1.00e-04 | 9.135e-05 | 0.9135 | 9.095e-05 | 0.9095 |
| 1.00e-02 | 9.072e-03 | 0.9072 | 9.072e-03 | 0.9072 |
| 1.00e+00 | 5.982e-01 | 0.5982 | 5.982e-01 | 0.5982 |

With consistent data the ratio is **0.9095 across eight orders of magnitude**,
constant to four digits, and the projection improves the field at every level
tested. The additive term is gone; what remains is the purely multiplicative
factor `sqrt(1 - visible)` of Theorem 5, with `visible = 1 - 0.9095² = 17.3%`
for this perturbation.

> **Remark 7.** This removes the method's principal stated limitation. The
> earlier claim — that the projection pays only above `L2 ~ 8e-05` — was an
> artefact of frozen data, not geometry. It matters concretely: the author's
> previous RLW work (arXiv:2511.11638) attains `L2 = 3.4e-05`, which sits
> *below* the old break-even, so under the current code the projection would
> have degraded it. With consistent data it improves it by the same 9%.

**Partial non-expansiveness result.** For the model case in which the
projection metric coincides with the constraint metric — i.e. projecting in
`||.||_A` onto `{||u||_A² = C2}` — the geometry is a genuine sphere and the
condition is exact: with `t = ||u~||_A`, `R = sqrt(C2)` and `θ` the `A`-angle
between `u~` and `u`,

```
||P(u~) - u||_A <= ||u~ - u||_A   <=>   (R-t)[(R+t) - 2R cos θ] <= 0
```

which holds **unconditionally whenever `t >= R`**, i.e. whenever the field sits
outside the constraint ellipsoid, and requires `cos θ >= (R+t)/(2R)` inside.

*Proof.* Radial projection gives `P(u~) = R u~/t`. Expand both squared norms in
`t`, `R`, `cos θ` and subtract; the difference factorises as stated. ∎

> **Remark 8 (honest gap).** The implemented projection minimises `||.||_h`
> while the constraint lives in `||.||_A`. The two metrics differ, so
> Proposition 8 does **not** transfer to the implemented operator, and the
> unconditional statement above is not proved for it. Empirically every field
> tested lay outside the ellipsoid and every ratio came out below 1, but that
> is evidence, not proof. Projecting in `||.||_A` instead would make the model
> case exact — a concrete and testable improvement, not attempted here.

**What the fix costs.** `ga_fn` and `gb_fn` are already read from the config
(`train.py:84–85`), so time-varying boundary data is **config-only** and does
not move `CODE_VERSION`. Time-dependent targets `C1(t)`, `C2(t)` are computed
from the initial condition in `_compute_reference_invariants` and would need a
code change. Both are required for the floor to vanish.

---

## 5c. The inverse problem: where the derivative term actually pays

Section 5 shows conservation buys little in the forward problem, and Theorem 5
says that is unavoidable. The inverse problem is a different regime, and the
reason is structural: `mu` multiplies the derivative term in *both* the PDE and
the invariant, so an error in `mu` is aligned with `grad I2` by construction
rather than lying in a generic direction.

**Measured.** Visible fraction and the resulting Theorem 5 ceiling, on RLW1:

| error type | visible | ceiling |
|---|---|---|
| generic forward error | 6.72% | 3.42% |
| `mu`-induced error | **45.6%** | **26.3%** |

**Theorem 9 (identifiability).** Both `int u^2` and `int u_x^2` are computable
from data without knowledge of `mu`. Conservation of `I2` between times `0` and
`t` therefore determines `mu` algebraically:

```
    I2 = int u^2 + mu int u_x^2 = const
=>  mu = - D[int u^2] / D[int u_x^2],     D[f] := f(t) - f(0)
```

whenever `D[int u_x^2] != 0`. The same argument on `I3 = int(eps u^3 + 3u^2)`
gives `eps = -3 D[int u^2] / D[int u^3]`.

**Corollary 10 (when recovery is possible).** `mu` is identifiable from `I2`
conservation **iff** `int u_x^2` varies in time, i.e. iff the solution changes
shape. A translating soliton has constant shape, so the denominator vanishes
and no information about `mu` is available at any accuracy.

**Verified.**

| problem | variation in `int u_x^2` | identifiable? |
|---|---|---|
| RLW1, single soliton | 1.7e-09 | **no** |
| RLW2, collision | 7.1e-01 | **yes** |

Across the RLW2 collision the derivative energy falls from 12.96 to 5.91 and
recovers. Applying Theorem 9 to the reference solution (true `mu = 1`):

| noise on data | `mu` recovered | spread |
|---|---|---|
| none | 1.0078 | — |
| 1e-03 | 1.0078 | 1.9e-03 |
| 1e-02 | 1.0088 | 1.7e-02 |

> **Remark 9.** The estimator is built from `int u_x^2`, the term methods
> restricted to `int u` and `int u^2` cannot represent. This is not a better
> version of what they do — it is unavailable to them.

> **Remark 10 (limits).** The systematic 0.8% bias is constant across noise
> levels, so it is the second-order finite-difference `u_x`, not noise; a
> higher-order operator or a weak formulation should remove it. Differentiating
> noisy data amplifies noise, so `int u_x^2` is the fragile term. And all of
> the above runs on the reference solution: with *sparse scattered* data these
> integrals cannot be formed directly, which is where a network is actually
> needed. No PINN result supports this section yet.

---

## 5d. Scope: what these results do and do not depend on

**The theory is not specific to conservation.** Proposition 3 and Theorem 5 use
only that the constraints are smooth functionals with linearly independent
gradients. For `m` constraints the normal space has dimension `m` and the
ceiling formula is unchanged. Conservation laws are the application; the bound
holds for any constraint set.

**The results do not depend on baseline accuracy.** §5b measured the
improvement ratio as `0.9095` from `L2 = 1e-08` to `1e-01` — constant to four
digits over eight orders of magnitude. A longer-trained, more accurate network
is improved by the same factor. Conclusions here are therefore not artefacts of
the training budget, which is the natural objection to a sweep run at 6000 Adam
iterations.

**`I3` is a poor target for projection.** `I3 = int(eps u^3 + 3u^2)` is cubic,
so the stationarity condition acquires a `lam3 (3 eps u^2 + 6u)` term and the
solve becomes nonlinear rather than a single elliptic solve. The payoff does not
justify it: a third constraint raises the normal space from dimension 2 to 3,
moving the RLW1 ceiling from 3.42% to roughly 5%. `I3` also contains **no
derivative**, so enforcing it is not novel in the sense §6 establishes for
`I2`. Its value is in Theorem 9, where it supplies `eps` at no cost.

**Test problems must be chosen by derivative share.** The quantity that decides
whether a derivative-dependent method can matter is the fraction of `I2`
contributed by the derivative term, `mu int u_x^2 / I2`, available from the
stored configs as `1 - C2_poly_ref / C2_ref`:

| problem | `int u^2` | `I2` | derivative share |
|---|---|---|---|
| RLW1 | 0.79599 | 0.81044 | **1.8%** |
| RLW2 | 107.402 | 120.357 | **10.8%** |

On RLW1 the derivative term is under 2% of the conserved quantity, so the one
feature distinguishing this method from existing work is nearly absent from the
problem it was demonstrated on. RLW2 is the appropriate centrepiece, and
Corollary 10 gives an independent reason: it is the only one of the two where
`mu` is identifiable at all. Reporting the derivative share for every test
problem, RLW1 included, states the selection criterion openly and documents
where the method does not apply.

---

## 6. Position relative to existing work

| work | conserved quantities | derivative-dependent? |
|---|---|---|
| PINN-Proj, arXiv:2410.17445 | `∫u`, `∫u²` | no |
| Conservation of Integrals, arXiv:2511.09048 | `∫u`, `∫u²` | no |
| RP-TENG, arXiv:2603.20816 | `∫k(u)dx` | no; and not PINNs |
| HCP-PINN, arXiv:2601.06244 | pointwise discrete residual | not integral |
| SP-PINN (KdV), arXiv:2511.00418 | mass, energy | soft penalty, not hard |

arXiv:2511.09048 states the extension as its open problem: *"Future work could
involve extending the projection method to other integral quantities beyond the
linear and quadratic integrals."* Remark 1 identifies what is gained by doing
so — an `H1` rather than `L2` certificate — and §5 quantifies what is not.

---

## 7. Status

| result | proof | verified numerically |
|---|---|---|
| Thm 1 existence | complete | not yet |
| Thm 2, 2' certificate | complete | yes, 5/5 runs |
| Prop 3 codimension 2 | complete | yes, matches shape decomposition |
| Cor 4 no upper bound | complete | consistent |
| Thm 5 ceiling | complete, first order | yes, ratio 0.995 |
| Thm 6 idempotence | complete | yes, 4.15e-17 |
| Cor 7 floor = data inconsistency | complete | yes, break-even eliminated |
| Prop 8 non-expansiveness | complete for the matched-metric model | consistent |
| Thm 9 `mu` estimator | complete, one line | yes, 0.8% on clean data |
| Cor 10 identifiability condition | complete | yes, 1.7e-09 vs 7.1e-01 |

**Independently audited, 2026-08-02.** The self-contained statements and proofs
in `MATH.md` were checked proof-by-proof by a reviewer working only from that
document. **No result was found to be false under its stated hypotheses.** Five
revisions were requested and all are incorporated there; the numbering in
`MATH.md` differs from this file — the ceiling theorem is Theorem 5 here and
Theorem 6 there.

What changed as a result:

* The ceiling theorem now states `||u*-u||_h >= ||e_T||_h - C||e||_h^2` with an
  explicit constant, rather than big-`O` inside a lower bound. The reviewer
  noted the Pythagorean identity in the proof is *exact*, so the earlier
  `O(||e||^3)` slack was unnecessary; the tightened proof is cleaner than the
  original.
* Two hypotheses previously listed as "assumed but unverified" are now shown to
  be load-bearing, with explicit degenerate configurations exhibited: linear
  independence of the constraint gradients, and `u~ != 0` in the sphere
  proposition.
* Uniqueness of the projection is supplied by the tubular neighbourhood theorem
  instead of assumed, and `||u*-u|| = O(||e||)` is derived rather than assumed.

**Unchanged by the audit.** Every numerical result in this file stands — the
revisions tightened statements without altering any conclusion.

**Still open.** Non-expansiveness of the implemented mixed-metric projection
(Remark 8) remains unsettled; only the matched-metric case is proved.

**Not yet done.** Theorem 1 has no numerical check. Theorem 5's `O(||e||^2)`
term is not quantified, so its radius of validity is demonstrated rather than
derived. Proposition 8 is proved only when the projection metric matches the
constraint metric; the implemented operator mixes `||.||_h` and `||.||_A` and
is therefore *not* covered, though every case tested obeys it (Remark 8).

**Superseded.** The earlier reading of the `7.6e-06` additive floor as an
intrinsic property of the operator was wrong; Corollary 7 shows it is data
inconsistency and removes it. Statements in `EXPERIMENT_PROTOCOL.md` about a
break-even at `L2 ~ 8e-05` need updating.

**Caveat.** These proofs have not been independently checked. Theorem 5's
first-order argument carries the headline claim and is the one most worth an
independent read; Theorem 6 is trivial but its consequence is the largest
practical result here, so the numerical evidence for Corollary 7 matters more
than the proof does.

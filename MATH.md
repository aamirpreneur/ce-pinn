# Projection onto derivative-dependent invariants: statements and proofs

Self-contained. No external references, no code, no numerical results — this
document is meant to be checked purely as mathematics.

Everything is finite-dimensional and discrete: these are statements about the
matrices an implementation actually uses, not about their continuum limits.

---

## 1. Setting

Let `[a,b]` be partitioned by `x_j = a + jh`, `j = 0,...,N-1`, `h = (b-a)/(N-1)`,
`N >= 3`.

**Weights.** `q in R^N` are the trapezoid weights: `q_0 = q_{N-1} = h/2` and
`q_j = h` otherwise. All entries are strictly positive. Write `Q = diag(q)`,
which is symmetric positive definite, and `1 = (1,...,1)^T`, so that `q = Q 1`.

**Differentiation.** `D in R^{N x N}` is any fixed differentiation matrix
(central differences in the interior, one-sided at the ends). No property of
`D` is used below beyond its being a fixed real matrix.

**Constant.** `mu > 0`.

**Inner products.**

```
<u,v>_h := u^T Q v ,        ||u||_h := sqrt(<u,u>_h)
A       := Q + mu D^T Q D
<u,v>_A := u^T A v ,        ||u||_A := sqrt(<u,u>_A)
```

**Lemma 0.** `A` is symmetric positive definite, so `||.||_A` is a norm.

*Proof.* `Q` is diagonal with strictly positive entries, hence SPD. `D^T Q D`
is symmetric, and `u^T D^T Q D u = (Du)^T Q (Du) = ||Du||_h^2 >= 0`, hence
positive semidefinite. A sum of an SPD and a PSD matrix is SPD. ∎

**Invariants.**

```
I1(u) := q^T u                              = <u, 1>_h
I2(u) := q^T( u∘u + mu (Du)∘(Du) )          = u^T A u = ||u||_A^2
```

where `∘` is the entrywise product. The identity `I2(u) = ||u||_A^2` is the
structural fact everything below relies on: **the derivative-dependent
invariant is a squared norm.**

**Constraint set.** For constants `C1 in R`, `C2 > 0`,

```
M := { u in R^N : I1(u) = C1,  I2(u) = C2 }.
```

**Projection.**

```
P(u~) := argmin { ||u - u~||_h  :  u in M }.
```

Throughout, `u` denotes a reference ("exact") vector with `u in M`, and
`e := u~ - u`.

---

## 2. Existence

**Theorem 1.** `M` is non-empty if and only if

```
C2  >=  C1^2 / (q^T A^{-1} q).
```

*Proof.* Consider `min { u^T A u : q^T u = C1 }`. The constraint set is a
non-empty affine hyperplane (`q != 0`), and `u -> u^T A u` is strictly convex
and coercive by Lemma 0, so a unique minimiser exists. Stationarity of
`u^T A u - nu (q^T u - C1)` gives `2Au = nu q`, i.e. `u = (nu/2) A^{-1} q`.
The constraint fixes `(nu/2) q^T A^{-1} q = C1`, and `q^T A^{-1} q > 0` since
`A^{-1}` is SPD and `q != 0`; hence `nu/2 = C1 / (q^T A^{-1} q)`. The minimum
value is

```
u^T A u = (nu/2)^2 q^T A^{-1} q = C1^2 / (q^T A^{-1} q).
```

On the hyperplane, `u -> u^T A u` is continuous, attains this minimum, and is
unbounded above (the hyperplane is unbounded and `A` is SPD). The hyperplane is
connected, so by the intermediate value theorem the map attains every value
`>= ` the minimum and no value below it. ∎

**Remark 1.** `M` is the intersection of a hyperplane with the ellipsoid
`{u^T A u = C2}`. Since `A` is SPD that ellipsoid is **compact**, hence `M` is
compact. This is used in §5 and §7.

**Remark 2 (the degenerate case).** At equality,
`C2 = C1^2 / (q^T A^{-1} q)`, the set `M` is the single point

```
u_0 = ( C1 / (q^T A^{-1} q) ) A^{-1} q ,
```

and there `A u_0 = ( C1 / (q^T A^{-1} q) ) q`, so `Q^{-1} A u_0` is parallel to
`1`. The two constraint gradients are then dependent, and the hypothesis of
Propositions 2 and 4 fails. Every result from §3 onward implicitly excludes
this configuration.

---

## 3. Characterisation of the projection

**Proposition 2.** Let `u*` be a local minimiser of the projection problem at
which `grad I1` and `grad I2` are linearly independent. Then there exist
`lam1, lam2 in R` with

```
(1 + lam2) u*  +  lam2 mu Q^{-1} D^T Q D u*  =  u~ - lam1 · 1 ,
```

together with `I1(u*) = C1`, `I2(u*) = C2`.

*Proof.* Linear independence of the constraint gradients is the LICQ condition,
so a KKT multiplier exists. With

```
L(u, lam) = (1/2)(u - u~)^T Q (u - u~) + lam1 (q^T u - C1)
                                       + (lam2/2)(u^T A u - C2),
```

`grad_u L = Q(u - u~) + lam1 q + lam2 A u = 0`. Substituting `q = Q 1` and
`A = Q + mu D^T Q D`, then multiplying on the left by `Q^{-1}`:

```
(u - u~) + lam1 · 1 + lam2 u + lam2 mu Q^{-1} D^T Q D u = 0,
```

which rearranges to the stated identity. ∎

**Remark 1.** `Q^{-1} D^T Q D` is the discrete negative second derivative, so
the displayed equation is the linear elliptic problem
`(1+lam2) u* - lam2 mu (u*)_xx = u~ - lam1`.

**Remark 2 (LICQ is not decorative).** The Euclidean constraint gradients are
`q` and `2Au`; these are independent exactly when `1` and `Q^{-1}Au` are, since
`Q` is invertible. Independence can genuinely fail. At the degenerate point
`u_0` of §2 Remark 2, `q` and `A u_0` are parallel, so the stationarity
condition `Q(u_0 - u~) + lam1 q + lam2 A u_0 = 0` requires
`Q(u_0 - u~) in span{q}` — false for generic `u~`, even though `u_0` *is* the
projection (it is the only feasible point). Multipliers therefore need not
exist without the hypothesis.

---

## 4. A posteriori certificates

These bound the error of an *arbitrary* vector `u~` — no regularity, no
smallness, no relation to any algorithm — using only computable quantities.

**Theorem 3 (H1 certificate).** For `u in M` and any `u~ in R^N`,

```
|| u~ - u ||_A  >=  | sqrt(I2(u~)) - sqrt(C2) | .
```

*Proof.* By Lemma 0, `||.||_A` is a norm, so the reverse triangle inequality
gives `|| u~ - u ||_A >= | ||u~||_A - ||u||_A |`. Now
`||u~||_A = sqrt(I2(u~))` and `||u||_A = sqrt(I2(u)) = sqrt(C2)`. ∎

**Theorem 3' (L2 certificate).** For `u in M` and any `u~ in R^N`,

```
|| u~ - u ||_h  >=  | I1(u~) - C1 | / sqrt(b - a) .
```

*Proof.* `I1(u~) - C1 = I1(u~) - I1(u) = <u~ - u, 1>_h`. Cauchy–Schwarz in
`<.,.>_h` gives `|<u~-u, 1>_h| <= ||u~-u||_h ||1||_h`, and
`||1||_h^2 = 1^T Q 1 = sum_j q_j = b - a`. ∎

**Remark.** Both right-hand sides involve only `u~` and the target constants.
Theorem 3 bounds the error in `||.||_A`, which controls `Du`; an invariant of
the form `int u^2` yields only the analogue of Theorem 3', which controls `u`
alone.

---

## 5. What the invariants cannot determine

**Proposition 4 (gradients).** With respect to `<.,.>_h`,

```
grad_h I1(u) = 1 ,        grad_h I2(u) = 2 Q^{-1} A u .
```

If these are linearly independent at `u in M`, then near `u` the set `M` is a
`C^infinity` submanifold of dimension `N - 2`, with normal space
`N_u M = span{ 1, Q^{-1} A u }`.

*Proof.* For any `v`, `dI1(u)[v] = q^T v = <1, v>_h`, giving the first gradient.
`dI2(u)[v] = 2 u^T A v = (2 Q^{-1} A u)^T Q v = <2 Q^{-1} A u, v>_h`, giving the
second. If the gradients are linearly independent then
`Phi := (I1, I2) : R^N -> R^2` is a submersion near `u`, so `(C1,C2)` is a
regular value locally and `M = Phi^{-1}(C1,C2)` is a submanifold of codimension
2 with `T_u M = ker dPhi(u)` and normal space the span of the gradients. ∎

**Corollary 5 (no error bound is derivable from the invariants).** Assume the
hypothesis of Proposition 4 and `N >= 3`. Then there is **no** function
`f : [0,inf)^2 -> [0,inf)` with `f(0,0) = 0` such that

```
|| u~ - u ||_h  <=  f( |I1(u~) - C1| , |I2(u~) - C2| )
```

for all `u~`.

*Proof.* Suppose such an `f` existed. Every `v in M` has
`|I1(v) - C1| = |I2(v) - C2| = 0`, so the bound would force
`||v - u||_h <= f(0,0) = 0`, i.e. `v = u`, i.e. `M = {u}`. But by Proposition 4
`M` is a submanifold of dimension `N - 2 >= 1` near `u`, so it is not a single
point. Contradiction. ∎

**Remark (correcting a natural but false argument).** One is tempted to argue
that `M` contains points arbitrarily far apart. That is **false**: `M` is
compact, being the intersection of a hyperplane with the compact ellipsoid
`{u^T A u = C2}` (Lemma 0). The correct argument is the dimension count above,
which needs no unboundedness.

---

## 6. A ceiling on what the projection can achieve

**Theorem 6.** Assume the hypothesis of Proposition 4. Then `M` is a
`C^infinity` embedded submanifold near `u`, and for `u~` in a sufficiently
small neighbourhood of `u` the metric projection `u* := P(u~)` is unique
(tubular neighbourhood theorem). Decompose

```
e = e_N + e_T ,    e_N in N_u M ,    e_T perp_h N_u M .
```

Then there is a constant `C > 0`, depending only on `M` near `u`, with

```
|| u* - u ||_h  >=  || e_T ||_h  -  C ||e||_h^2 ,
```

and consequently the relative improvement satisfies

```
1 - ||u* - u||_h / ||e||_h   <=   1 - sqrt(1 - V)  +  C ||e||_h ,
V := ||e_N||_h^2 / ||e||_h^2 .
```

*Proof.* First, no growth assumption on `||u* - u||_h` is needed: `u in M` is
feasible, so `||u~ - u*||_h <= ||u~ - u||_h = ||e||_h` by optimality, whence

```
||u* - u||_h <= ||u* - u~||_h + ||u~ - u||_h <= 2 ||e||_h .            (*)
```

Put `c := u~ - u*`, so `||c||_h <= ||e||_h`. First-order optimality of the
metric projection onto a smooth submanifold gives `c in N_{u*} M`. Let `c_N` be
the `<.,.>_h`-orthogonal projection of `c` onto `N_u M` and set `r := c - c_N`,
so that `r perp_h N_u M` by construction. Because `M` is `C^infinity` and
embedded, its normal bundle is locally Lipschitz: there is `L > 0` with the gap
between `N_{u*}M` and `N_u M` at most `L ||u* - u||_h`. Since `c in N_{u*}M`,

```
||r||_h  <=  L ||u* - u||_h ||c||_h  <=  2L ||e||_h^2 ,
```

using (*). Now

```
u* - u = e - c = (e_N - c_N) + (e_T - r) ,
```

where the first bracket lies in `N_u M` and the second is `<.,.>_h`-orthogonal
to it. The Pythagorean identity is therefore **exact**:

```
||u* - u||_h^2 = ||e_N - c_N||_h^2 + ||e_T - r||_h^2  >=  ||e_T - r||_h^2 .
```

Hence, by the triangle inequality,

```
||u* - u||_h >= ||e_T - r||_h >= ||e_T||_h - ||r||_h >= ||e_T||_h - C ||e||_h^2
```

with `C = 2L`. Dividing by `||e||_h` and using `||e_T||_h = ||e||_h sqrt(1-V)`,
which is orthogonality of the splitting, gives the second claim. ∎

**Remarks.**

1. The bound is genuinely asymptotic: when `V` is close to 1, `||e_T||_h` is
   small and the `C ||e||_h` term can dominate. That is a limitation of the
   statement, not an error in it.
2. `C^2` regularity need not be assumed. `I1` and `I2` are polynomial, so under
   Proposition 4's independence hypothesis `M` is automatically `C^infinity`.
3. Local uniqueness of `u*` is genuinely needed and is supplied by the tubular
   neighbourhood theorem, not by assumption.

---

## 7. Consistency and the origin of an error floor

Let `F` denote the feasible set of the implemented solve: vectors satisfying
`I1 = C1`, `I2 = C2` **and** prescribed endpoint values `u_0 = g_a`,
`u_{N-1} = g_b`. Let `P_F` be the `||.||_h`-metric projection onto `F`.

**Theorem 7 (idempotence).** If `u in F` then `P_F(u) = u`.

*Proof.* `F` is closed and `u in F`, so `min{ ||v - u||_h : v in F } = 0`,
attained at `v = u`. Any minimiser has `||v - u||_h = 0`, hence `v = u`. ∎

**Corollary 8.** Assume `F != empty`. If the reference vector `u` does **not**
lie in `F` — for instance because `g_a, g_b` or `C1, C2` are held at values `u`
does not satisfy — then `d := dist_h(u, F) > 0`, and for every selection
`v_e in argmin{ ||v - (u+e)||_h : v in F }`,

```
|| v_e - u ||_h  ->  d > 0    as   ||e||_h -> 0 .
```

The displacement therefore appears as an **additive** term in
`||P_F(u~) - u||_h`, not as one vanishing with `||e||_h`.

*Proof.* `F` is closed, and compact: it is contained in the ellipsoid
`{v^T A v = C2}`, which is compact by Lemma 0. Hence the distance to `F` is
attained, and `d > 0` because `u not in F`. For the limit, note
`| dist_h(u+e, F) - dist_h(u, F) | <= ||e||_h` since the distance function is
1-Lipschitz, so `||v_e - (u+e)||_h -> d`, and therefore
`| ||v_e - u||_h - d | <= ||e||_h + | ||v_e-(u+e)||_h - d | -> 0`. No
single-valuedness or continuity of `P_F` is required. ∎

**Remark.** The hypothesis `F != empty` is needed only so that `P_F` is
defined; it holds by construction whenever the targets are taken from an
admissible field.

---

## 8. Non-expansiveness in the matched-metric case

Theorem 6 bounds the improvement but does not exclude the projection making
things worse. Here is a case where non-expansiveness can be settled exactly.
Note the metric is `||.||_A`, matching the constraint, rather than `||.||_h`.

**Proposition 9.** Let `S := { v : ||v||_A = R }` with `R > 0`, and let
`P_S(u~) := R u~ / ||u~||_A` for `u~ != 0` be the metric projection onto `S` in
`||.||_A`. Let `u in S`, `t := ||u~||_A`, and `cos(theta) := <u~,u>_A / (tR)`.
Then

```
|| P_S(u~) - u ||_A  <=  || u~ - u ||_A
  <=>  (R - t) [ (R + t) - 2 R cos(theta) ]  <=  0 .
```

In particular the inequality holds for every `u~` with `t >= R`.

*Proof.* `<P_S(u~), u>_A = (R/t) <u~,u>_A = R^2 cos(theta)`, so

```
||P_S(u~) - u||_A^2 = R^2 - 2R^2 cos(theta) + R^2 = 2R^2 (1 - cos(theta)),
||u~ - u||_A^2      = t^2 - 2 t R cos(theta) + R^2 .
```

Their difference is

```
2R^2 - 2R^2 cos(th) - t^2 + 2 t R cos(th) - R^2
   = (R^2 - t^2) - 2R cos(th)(R - t)
   = (R - t)[(R + t) - 2R cos(th)] ,
```

giving the equivalence. If `t >= R` then `R - t <= 0`, while
`(R + t) - 2R cos(th) >= (R + t) - 2R = t - R >= 0` since `cos(th) <= 1`; the
product is `<= 0`. ∎

**Remark 1.** The exclusion `u~ != 0` is necessary, not cosmetic. At `u~ = 0`
every point of `S` is nearest, and the selection `P_S(0) = -u` gives
`||P_S(0) - u||_A = 2R` against `||0 - u||_A = R`, so non-expansiveness fails
for that selection.

**Remark 2.** This does **not** apply to the projection of §3, which minimises
`||.||_h` while the constraint is stated in `||.||_A`. Whether that mixed-metric
projection is non-expansive is open here.

---

## 9. Parameter identification

Now let `u(·,t)` depend on time and suppose `I2` is conserved, where `mu` is an
**unknown** constant. Write

```
S0(t) := q^T (u∘u)(t) ,     S1(t) := q^T ((Du)∘(Du))(t) ,
```

both computable from `u(·,t)` without knowledge of `mu`, and
`Delta f := f(t) - f(0)`.

**Theorem 10.** If `I2(u(·,t)) = S0(t) + mu S1(t)` is constant in `t`, then for
any `t` with `Delta S1 != 0`,

```
mu  =  - Delta S0 / Delta S1 .
```

*Proof.* `S0(t) + mu S1(t) = S0(0) + mu S1(0)` gives
`Delta S0 + mu Delta S1 = 0`; divide by `Delta S1 != 0`. ∎

**Corollary 11 (identifiability).** `mu` is determined by conservation of `I2`
if and only if `S1` is non-constant in `t`. If `S1` is constant then for every
`m in R` the quantity `S0 + m S1` is conserved, so no value of `mu` is
distinguished.

*Proof.* The "if" direction is Theorem 10. For the converse, if `S1(t) = S1(0)`
for all `t` then `Delta S0 = -mu Delta S1 = 0`, so
`Delta(S0 + m S1) = 0` for every `m`. ∎

**Remark.** The same argument applied to a conserved `I3 = eps T0 + 3 S0` with
`T0 := q^T (u∘u∘u)` yields `eps = -3 Delta S0 / Delta T0` whenever
`Delta T0 != 0`. That estimator involves no differentiation matrix.

---

## 10. Review status

**Independently audited.** Every statement above was checked proof-by-proof by
a reviewer working only from this document. No result was found to be false
under its stated hypotheses. Five revisions were requested and all are
incorporated:

1. **Theorem 6** — big-`O` inside a lower bound replaced by an explicit
   constant, `||u*-u||_h >= ||e_T||_h - C||e||_h^2`, and the square-root step
   made rigorous. The reviewer also observed that the Pythagorean identity in
   the proof is *exact* once `r` is defined as the component of `c` orthogonal
   to `N_u M`, so the `O(||e||^3)` slack in the earlier version was unnecessary.
   The proof above is the tightened form.
2. **Theorem 6** — the tubular neighbourhood theorem now supplies uniqueness of
   the projection instead of it being assumed, and `C^2` regularity is noted to
   be automatic because `I1`, `I2` are polynomial.
3. **Theorem 6** — `||u* - u||_h = O(||e||_h)` is no longer assumed; it follows
   from feasibility of `u`, as step (*) of the proof.
4. **Corollary 8** — `F != empty` added, and the conclusion strengthened to
   hold for any selection from a set-valued projection, using compactness of
   `F` rather than continuity of `P_F`.
5. **Corollary 5** — the compactness correction retained and made explicit.

Two hypotheses were confirmed as load-bearing rather than technical, with
explicit degenerate configurations exhibited: linear independence of the
constraint gradients (§2 Remark 2, §3 Remark 2, §5), and `u~ != 0` in
Proposition 9 (§8 Remark 1).

**Remaining caveat.** Theorem 6 is asymptotic in `||e||_h` and has no uniform
validity; when `V` is near 1 the remainder can dominate. This is a property of
the statement, not a defect in the proof.

**Still open.** Non-expansiveness of the mixed-metric projection of §3 — the
one actually implemented — is not settled. Proposition 9 covers only the
matched-metric case.

# Plan to submission

Written 2026-08-02. The paper is a forward-problem paper; the inverse problem
(Theorems 10–11 of `MATH.md`) is deliberately held back for a second paper.

---

## The claim the paper makes

> A projection that enforces derivative-dependent invariants exactly — which no
> existing method can do — together with a proof of how much conservation
> enforcement can help *any* PINN, and a demonstration that the operator
> attains that predicted bound across different equations.

Not "our method is more accurate." That claim is false and provably so
(`MATH.md` Corollary 5), and every attempt to make it has failed.

**The paper's single most important figure** is predicted ceiling against
realised improvement, one point per test problem. The claim it supports is
*predictive accuracy across equations* — the theory forecasts the achieved
improvement on a problem it was never tuned to. Everything else is supporting
material for that figure.

> **Correction, 2026-08-02.** An earlier version of this plan claimed the
> ceiling rises with the derivative share, and promised a trend "spanning an
> order of magnitude". **That is not established and the measurement
> contradicts it.** Camassa–Holm has 28× RLW1's derivative share (50% against
> 1.8%) but only 2× the ceiling (6.89% against 3.42%). The visible fraction
> depends on how the *error* aligns with the constraint gradients, not only on
> how much derivative content the invariant carries, so it has to be measured
> per problem rather than inferred from the share. The paper must not claim
> otherwise.
>
> What survives is stronger anyway: the theory predicted 3.42% and 6.89% on two
> different equations, and the projection realised 3.40% and 6.28% — 99.5% and
> 91% of the bound. Predictive accuracy is the claim; the spread of the axis is
> incidental.

---

## Phase 0 — Fixes, in one pass. *Blocks everything else.*

Two of these change numerics, so `CODE_VERSION` moves and **every existing
result becomes non-comparable**. That is why they must land together, before
any new run, and be tagged. Running experiments first and fixing after is how
the last three weeks were lost.

| # | change | numerical? | source |
|---|---|---|---|
| 0.1 | Time-varying Dirichlet data: pass `ga_fn`, `gb_fn` | no — config only | `THEORY.md` §5b |
| 0.2 | Time-dependent invariant targets `C1(t)`, `C2(t)` | **yes** | `THEORY.md` §5b; also fixes RLW3 |
| 0.3 | Rank-1 projector when the `lam2` clamp binds | **yes** | queued in protocol |
| 0.4 | `requires_grad` warning in `train.py` | cosmetic, but touches a hashed file | queued in protocol |

**Gate:** all tests pass; `tools/verify_theory.py` reproduces the ceiling result
on a fresh run; projecting the exact solution returns it to ~1e-16 (Theorem 7).
Then `git tag exp-v4`.

**Estimate: 1–2 days.**

---

## Phase 1 — Test problems

Problems are chosen to span **different equations**, not to maximise any single
scalar. The derivative share is reported for each — openly, RLW1's 1.8%
included — but it is a description, not a selection criterion, and not a
predictor of the ceiling (see the correction above).

| problem | share | measured ceiling | new code | risk |
|---|---|---|---|---|
| RLW1, `c = 0.1` | 1.8% | 3.42% | none — already runs | none |
| RLW tall soliton, `c = 1.777` | **11.30%** | to be measured | **done** — `RLW1T_tall_soliton.ipynb` | none |
| Camassa–Holm, `kappa = 0.1` | **27.97%** | to be measured | **done** — `CH1_solitary_wave.ipynb` | none |
| *Rosenau* (optional) | H² invariant | — | new residual **and** 4th-order solve | higher |

### 1.0 — Camassa–Holm compatibility. **DONE, 2026-08-02. Passed.**

CH conserves `E1(u) = int(u^2 + u_x^2)` (Constantin–Escher; peakon-stability
literature), which is RLW's `I2` at `mu = 1`. Verified against the existing
code:

* `helmholtz_projection(..., mu=1.0, ...)` is idempotent on an exact CH
  solution: `||P(u) - u||_h = 2.98e-14`, `I1` error 2.2e-16, `I2` error 1.4e-13
* `E1` conserved on the grid to 1.7e-15 as the wave translates
* On a perturbed field the projection realises 6.28% against a predicted
  ceiling of 6.89%

**No new projection code is needed for CH.** Only the PDE residual.

*(Idempotence reaches 3e-14 rather than RLW's 4e-17. Far below any PINN error,
but it indicates the Newton solve is not converging to machine precision on a
field with a corner. Recheck once smooth waves replace the peakon.)*

### 1.1 — CH solution: smooth waves. **DECIDED — peakons are ruled out.**

Measured: the peakon's corner makes `u_x` jump, so a centred difference across
it is `O(1)` wrong and `int u_x^2` converges only at **first order**:

| N | dx | error in `int u_x^2` |
|---|---|---|
| 301 | 0.20 | 17.6% |
| 1201 | 0.05 | 4.8% |
| 4801 | 0.0125 | 1.2% |

At the finest grid the invariant is still 1% wrong. Enforcing it to 1e-14 would
be enforcing a target that is 1% wrong — the same failure mode as the coarse-grid
`C2` problem recorded in `EXPERIMENT_PROTOCOL.md`, an order of magnitude worse.

**Consequence: a reference for the smooth `kappa > 0` solitary wave is now a
blocking item.** It has no simple closed form, so a source is required before
Phase 1 can proceed. This is the one literature dependency in the plan.

### 1.1b — Tall soliton. **DONE, 2026-08-02.**

`RLW1T_tall_soliton.ipynb`, `c = 5.33/3` so the amplitude is 5.33, on
`[0,100]` with `T = 20`. Analytic solution, so no reference solver. Verified at
the real `N_GRID = 601`:

| check | result |
|---|---|
| derivative share | **11.30%** measured against 11.35% predicted by `eps*c/(5+6*eps*c)` |
| resolution | 15 points across the wave; travels 55.5 of 100 units |
| Theorem 7, exact time-dependent data | `3.43e-16` |
| Theorem 7, frozen at `t=0` | `4.20e-05` |

The last two rows are the point. Frozen boundary data would have carried a
floor of `4.20e-05` here — an order of magnitude worse than RLW1's `5.6e-06`,
because this wave starts 15 units from the left edge and ends 70 away, so its
tail there falls from `1.31e-04` to `6.7e-24`. The Phase 0 fix cuts that floor
by a factor of 122,000. Had this problem been run before Phase 0, the floor
would have sat only ~20x below the expected solution error.

This is the first experiment to use `ga_fn`, `gb_fn`, `C1_fn`, `C2_fn`,
`C3_fn` and `C2_poly_fn` together.

### 1.1c — Camassa–Holm. **DONE, 2026-08-02.**

`CH1_solitary_wave.ipynb`, `c = 1`, `kappa = 0.1` (amplitude 0.8), on `[0,40]`
with `T = 20`, `N = 601`. Verified at the real grid:

| check | result |
|---|---|
| derivative share | **27.97%** measured against 28.04% exact |
| CH residual on the exact wave | rms `8.5e-05` against `|u_t|` rms `1.1e-01`; converges to `6.0e-07` at N=2401, observed order 3.5–5 |
| Theorem 7, exact time-dependent data | `1.37e-16` |
| Theorem 7, frozen at `t=0` | `4.08e-05` |
| resolution | 35 points across the wave |

The residual check is the one that matters: a sign error in the CH equation
would still have produced a plausible training curve while making every
downstream number meaningless. It converges cleanly to zero, so the
implementation is right.

**Grid: `N = 601`, not the 2401 first estimated.** The earlier figure came from
comparing the grid-computed `C2` against its *continuum* value. That is the
wrong criterion — the trainer computes `C1_fn`/`C2_fn` with the same quadrature
and `D1c` it uses for the constraint, so the sampled exact solution satisfies
the discrete constraint by construction at any resolution. Confirmed: Theorem 7
holds at `2.7e-15` from N=401 to N=2401, *including* with the crest placed
halfway between grid points. The binding constraint is resolving the wave for
the PDE residual, which is far cheaper. Same argument applies to the
coarse-grid "confound" recorded earlier for RLW1 — it was measured the same
way and overstated.

**Peakons stay ruled out**, now with the convergence rate measured: order
0.94/0.97/0.99 against 1.93/1.98/2.00 for the smooth wave, and 35x the error at
equal resolution.

### 1.2 — Reference solutions

Where no closed form exists, built and verified the way
`make_rlw2_reference.py` was: Richardson extrapolation at the *observed* order,
checked against a second independent extrapolation, error reported.

**Estimate: 3 days** for RLW-tall and CH, once the CH solitary-wave reference
is in hand. Rosenau adds ~3 more.

---

## Phase 2 — Runs

Per problem: `hard_bc` × 3 seeds, `ce_poly` × 1, `ce_helm` × 1 = 5 runs.
`hard_bc+proj` is scored from the `hard_bc` checkpoints and costs nothing.

Three problems → **15 runs**, plus **one long run per problem** at a
competitive budget to answer "your baseline is weak" directly.

**Do not fix RLW2.** The two-soliton training failure is a research problem
with no guaranteed answer. It is not needed for the forward paper, and it *is*
needed for the inverse paper — where the collision is the only configuration in
which `mu` is identifiable at all. Leave it there.

**Estimate: 2 days**, assuming the Kaggle API is set up. Longer if runs are
uploaded by hand.

---

## Phase 3 — Analysis

| # | output |
|---|---|
| 3.1 | **The main figure.** Predicted ceiling vs realised improvement, one point per problem, with the diagonal drawn. Points should sit on it. This is the paper. |
| 3.2 | Certificate table (Theorems 3, 3′): bound vs true error, all problems |
| 3.3 | Visible fraction and derivative share per problem, reported side by side — **as two separate measurements**, with the observation that they do not track each other |
| 3.4 | Idempotence check post-fix (Theorem 7) — should now be ~1e-16 |
| 3.5 | Numerical check of Theorem 1, the one result never verified |
| 3.6 | L², H¹, L∞ reported for every run — as measurements, **not** as a claimed H¹ advantage, which was tested and does not hold |

**Estimate: 1 day.** All of it runs on stored checkpoints; no GPU.

---

## Phase 4 — Writing

Sections, in the order they should be drafted:

1. Method — the operator, `MATH.md` §3
2. Theory — `MATH.md` §§4–8, already audited
3. Experiments — Phase 3 outputs
4. Related work — the five-paper comparison in `THEORY.md` §6, with the
   competitor's own "future work" sentence quoted
5. Limitations — the ceiling, the L∞ degradation, indefinite invariants
   (KdV/NLS) explicitly out of scope, the mixed-metric non-expansiveness gap
6. Intro and abstract, written last

**Estimate: 2 weeks.** This is the phase most likely to overrun, and the one
that should be allowed to.

---

## Timeline

| phase | days |
|---|---|
| 0 — fixes and tag | 1–2 |
| 1 — problems | 3 |
| 2 — runs | 2 |
| 3 — analysis | 1 |
| 4 — writing | 10–14 |
| **total** | **~3 weeks** |

Against a target of "before October 2026", that leaves real slack. Use it on
Phase 4, not on adding a fourth test problem.

---

## Open decisions

| decision | status |
|---|---|
| CH peakon or smooth solitary wave? | **decided — smooth.** Peakons ruled out by measurement (§1.1) |
| Does CH need new projection code? | **decided — no.** Verified (§1.0) |
| Include Rosenau? | open; only if Phases 0–2 finish early |
| Venue | JCP as stated, but see the risk below |

**Blocking:** a source for the smooth CH solitary wave. Nothing in Phase 1 can
start for CH until that is in hand.

---

## Risks, and what to do about each

**~~The paper is thin on problems.~~** *Retired.* The CH check passed, so three
problems are reachable with no new projection code. The residual risk is the
smooth-solitary-wave reference, which is a literature dependency rather than a
research one.

**The main figure has a narrow axis.** With ceilings of 3.42% and 6.89%, the
horizontal spread is a factor of 2, not the order of magnitude originally
planned. A referee may find two or three clustered points unconvincing as a
"trend". *Mitigation:* frame the figure as predictive accuracy — how close each
point sits to the diagonal — rather than as a trend across a range. Adding
Rosenau would widen the axis and is the strongest argument for including it.

**Venue mismatch.** JCP may treat this as machine learning rather than
computational physics and desk-reject on scope. *Mitigation:* lead with the
theory and the discrete numerical analysis, not the network. Have a second
venue chosen in advance — Journal of Scientific Computing or CMAME — so a desk
rejection costs days, not months.

**Theorem 6 is asymptotic.** A referee may object that the ceiling has no
uniform validity. *Mitigation:* it is stated as asymptotic in `MATH.md`, the
numerics show where it breaks (the `O(1)`-error rows), and that limit is
reported rather than hidden.

**The proofs have one external audit, no human review.** *Mitigation:* have the
supervisor read `MATH.md` §§4–8 before submission. The audit found no false
results, but a machine audit is not a referee.

**Writing overruns.** The most likely failure. *Mitigation:* the timeline has
slack; use it here and do not add scope.

---

## What is already done

- Operator: implemented and working
- Theory: `MATH.md`, independently audited, no false results
- Verification tooling: `tools/verify_theory.py`
- Novelty check: five competing papers, `THEORY.md` §6
- **Phase 1.0 complete**: CH verified to need no new projection code; peakons
  ruled out; the share-to-ceiling assumption disproved and recorded
- RLW1 sweep: complete, but **superseded** by the Phase 0 fixes
- Inverse-problem groundwork: held for paper 2

## Literature still to obtain

| for | what is needed | blocking? |
|---|---|---|
| Camassa–Holm | the smooth `kappa > 0` solitary wave — form and parameters | **yes**, blocks Phase 1 |
| Camassa–Holm | a citable source for `E1 = int(u^2 + u_x^2)` being conserved | no, but needed for the write-up |
| Rosenau | equation, invariants, and a conservative-scheme paper for reference values | only if Rosenau is included |
| RLW | published error tables for context in §Experiments | no |

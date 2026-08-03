# Experiment protocol

The problem this solves: code kept changing while experiments were running, so
it was never clear whether a result was still valid. This is the rule for
deciding that, and the process for avoiding the question in the first place.

## What invalidates a result

Only changes to the five numerical modules can change a number:

```
ce_pinn/equations.py    ce_pinn/network.py    ce_pinn/train.py
ce_pinn/operators.py    ce_pinn/projection.py
```

Everything else is inert as far as results are concerned: the derivations
document, the README, tests, `tools/`, comments, notebook prose. A fix to any
of those never requires a rerun.

`tools/make_notebooks.py` hashes exactly those five files into `CODE_VERSION`,
which is printed at the top of every run and written into every `results.json`.
Two results are comparable if and only if their `CODE_VERSION` matches. To find
what a stored result was produced by:

```bash
python -c "import json;d=json.load(open('results/<run>/<ts>/results.json'));print(d['code_version'], d['code_commit'])"
```

## What does not invalidate a result, even when the numerics change

A numerical change only invalidates the experiments whose code path it
actually touches. Record that reach explicitly rather than rerunning
everything.

## The cycle

1. **Freeze.** Tag the current numerics: `git tag -a exp-vN -m "..."`.
2. **Generate** notebooks from the tagged state and upload them to Kaggle. Each
   notebook carries its own inlined copy of the library, so later edits to
   `ce_pinn/` cannot reach a run already in flight.
3. **Run** the batch. During this window no numerical edits land on `main`.
   Audit findings are recorded below and applied afterwards, not immediately.
4. **Apply** all queued findings in one pass, re-verify, tag `exp-v(N+1)`.
5. **Rerun** only the experiments the change log marks as invalidated.

The failure mode to avoid is fix, audit, fix, audit, which never terminates and
leaves every run of uncertain provenance.

## Change log

Each numerical change records which experiments it invalidates. Anything not
listed here changed no numbers.

| Date | Change | Numerical? | Invalidates |
|---|---|---|---|
| 2026-07-29 | Grid-based residual; projection moved into the forward pass | yes | everything before it (the projection had no effect on training at all) |
| 2026-07-29 | Hard-BC mask normalised to O(1) | yes | every `hard_bc`, `ce_poly`, `ce_helm`, `ce_full` run |
| 2026-07-29 | `vanilla` given IC/BC penalties | yes | `vanilla` only |
| 2026-07-29 | `I3 = int(eps*u^3 + 3u^2)`, was missing `eps` | yes | **RLW3 only** (eps=1.5). RLW1, RLW2, KdV1, KdV2 all use eps=1 and are unaffected |
| 2026-07-29 | Analytic Newton start, damped line search, lam2 clamp | yes | nothing already valid: removes non-convergence, does not move a converged answer |
| 2026-07-29 | KdV two-soliton IC differentiated analytically | yes | KdV2 only |
| 2026-07-29 | Boundary tangent `g_a'(t)`, `g_b'(t)` | yes | nothing — no experiment uses time-varying boundary data |
| 2026-07-29 | Three audit passes on the derivations document | no | nothing |
| 2026-07-29 | Push-cell token check, `.gitignore` fix, provenance stamp | no | nothing |
| 2026-07-29 | Soft-I3 denominator guarded against `C3 = 0` | yes | nothing — only reachable for `ce_full`, and `C3` is 0.2 / 2.58 / 6.6 in all three test problems, so no run could have divided by zero |
| 2026-07-29 | Cosine LR schedule in `train_adam`; `best_loss` tracked every step | yes | **everything** — this is exp-v3. No full run had completed, so nothing real is lost |
| 2026-07-29 | Adam budget 20000 -> 5000 (6000 for the two-soliton cases), L-BFGS 2000 -> 500 | config only | all runs must use one budget to be comparable |
| 2026-07-30 | Post-hoc analysis tooling (`analyze_rlw1.py`, `diag_why_projection_costs.py`, `eval_project_at_inference.py`, `diag_projection_needs_good_field.py`) | no | nothing — reads stored checkpoints, writes no numerics |
| 2026-07-30 | RLW2 reference solution (`make_rlw2_reference.py`, `results/reference/`) and its inlining into the RLW2 notebook | no | nothing — a new measuring stick, not a change to what is measured |
| 2026-08-02 | Theory (`THEORY.md`), verification (`tools/verify_theory.py`), `check_rlw2_best.py` | no | nothing — reads stored checkpoints, proves statements about the existing operator |
| 2026-08-02 | **exp-v4, Phase 0.** Time-dependent targets `C1_fn`/`C2_fn`/`C3_fn`; clamped-state solve made well-posed and its adjoint dropped to rank 1; `float()` detached; file handles closed | yes | **everything.** `CODE_VERSION` `e299b122b9be` -> `0c25c545396d`. See the reach note below |

| 2026-08-02 | **exp-v5.** Camassa-Holm support: `ch_solitary`/`ch_solitary_torch` in `equations.py`, `ch` branches in `_residual` and `_invariants`, `need_d3` for `ch`, boundary-flux check no longer falls through to the KdV formula for unknown equations | yes | **nothing beyond exp-v4** — no run was ever produced under exp-v4, so `exp-v5` is the first version any experiment will use. `CODE_VERSION` `0c25c545396d` -> `1275f618af65` |

### Reach of the exp-v5 changes

Additive: new `ch` branches and two new functions. The `rlw` and `kdv` code
paths are untouched, so the exp-v4 reach note below still describes everything
that changed relative to the completed RLW1 sweep. The one non-additive edit is
the boundary-flux check, which previously applied the KdV flux expression to
*any* equation that was not `rlw` — now it says so instead of printing a
reassuring number derived from the wrong conservation law.

`exp-v4` was tagged but produced no runs, so nothing is lost by superseding it.
**Every experiment should run under `exp-v5` or later.**

### Reach of the exp-v4 changes

`CODE_VERSION` moves, so nothing before it is formally comparable. What
actually changed, per experiment:

* **Time-dependent targets.** Inert unless `C1_fn`/`C2_fn`/`C3_fn` are supplied.
  Every completed run used constants and would reproduce bit-for-bit. The
  *point* of the change is that new runs can supply them, which removes the
  additive error floor (`THEORY.md` §5b) — verified here at exactly `0.0`
  against `7.574e-06` with frozen data.
* **Clamped-state solve.** Reaches only steps where the `lam2` clamp binds. The
  cosine schedule fires it 0 times in 2500 steps, so the completed RLW1 sweep
  is almost certainly untouched — `newton_clamped` is recorded per run and is
  auditable.
* **`float()` detach, file handles.** Cosmetic; no numerical reach.

Practically: the completed RLW1 sweep is still *believable*, but it must be
rerun under `exp-v4` before publication so that every reported number carries
one code version and the boundary/target fix is actually in effect.

`CODE_VERSION` is still `e299b122b9be` after all of the above, so the RLW2 runs
will be directly comparable with the completed RLW1 sweep. That is the point of
hashing only the five numerical modules.

## Queued for a later version

*(Both entries below were resolved in exp-v4 on 2026-08-02. Kept for the
record, with what the fix turned out to require.)*

**~~Adjoint is inconsistent when the lam2 clamp binds.~~ FIXED — and the
problem was larger than this entry described.** The entry assumed `lam1` was
still determined by `F1 = 0` when the clamp binds, so that a rank-1 projector
would be the correct adjoint. It is not. The Newton step is computed from the
full 2x2 system and only `lam2` is clamped afterwards, so `lam1` takes a step
predicated on `lam2` moving, which it then does not; the loop exits on
`max_iter` and `lam1` is left wherever that put it. Measured on a clamped RLW
fixture: `F1` came out between `-1.0` and `-2.6`, not `0`. **Neither constraint
was satisfied and the forward map was the output of a truncated iteration**, so
no implicit-function adjoint existed at any rank — a rank-1 projector alone
fails a finite-difference check at 14% relative error.

*Fix as applied:* pin `lam2` at its bound and solve the remaining scalar
equation `F1(lam1) = 0` exactly. `F1` is affine in `lam1` (the solve is linear
in its right-hand side), so one extra linear solve gives the root. `I1` is then
enforced exactly, `I2` is not — it cannot be at the bound — and `lam1` becomes
a genuine implicit function of `u_tilde`, which makes the rank-1 adjoint
correct. Verified against finite differences at `2.1e-10` relative, with the
old rank-2 projector failing the same check at `1.5e-02`. `tests/test_phase0.py`
constructs a clamped state deliberately; no previous test reached one.

**~~Cosmetic `requires_grad` warning.~~ FIXED.** `float()` now detaches in
`train_adam` and `train_lbfgs`. `tests/test_phase0.py` runs both with warnings
promoted to errors. Unclosed file handles in `_save_history`/`_save_config`
were fixed in the same pass.

---

*Original entry, retained:*

**Adjoint is inconsistent when the lam2 clamp binds.** Measured and downgraded:
with the cosine schedule the clamp does not fire at all (0 events against 2 for
the constant schedule over 2500 steps), and near convergence `lam2` reaches
only about 1e-05 against a bound at -1.4e-02. The clamp turned out to be a
symptom of optimiser instability rather than a cause of it. When `lam2` sits on its
lower bound it is held there rather than determined by `F2 = 0`, so
`dlam2/du~ = 0`. The backward pass nonetheless applies the full rank-2
projector `Pi = P J^-1 R^T`, which assumes both multipliers respond. The
correct treatment on a clamped step is a rank-1 projector using `lam1` alone.

*Reach:* numerical, but does **not** invalidate a completed run. The clamp
binds on roughly 0.25% of projections, all in early training, and the effect is
a slightly wrong descent direction on those steps rather than an error in any
reported quantity. Everything reported comes from the forward projection, which
is exact whether or not the clamp was active, and the run records
`newton_clamped` so the frequency is auditable per run.

*Fix:* in `_HelmholtzApply.backward` and `project_tangent`, drop the `lam2`
column from `P` and `R` when `st.clamped` is set.

**Cosmetic `requires_grad` warning.** `train.py` calls `float()` on a tensor
that still carries a grad history, which prints a UserWarning. It is harmless —
`float()` detaches — but the fix would touch a numerical module and move
`CODE_VERSION`, invalidating the completed RLW1 sweep for the sake of a warning.
It waits for the next version bump.

## What RLW1 established, and what it did not

Recorded here because the next experiments were designed around it.

* The central claim holds: `ce_helm` drives I2 drift to 1.2e-14 where `ce_poly`,
  the affine projection existing methods use, reaches only 1.7e-05.
* The projection **costs accuracy when it sits inside training**: `ce_helm`
  plateaus at a PDE residual of 8.5e-09 against `hard_bc`'s 1.9e-09, a 4.4x
  floor, and it had converged — its loss at 90% and 100% of budget are equal, so
  this is not an under-training artefact.
* The projection **helps when applied at inference**: projecting the trained
  `hard_bc` field improved L2 at 10 of 10 time levels and gave exact
  conservation, exact boundary data, and the lowest I3 drift in the sweep, at
  1/7.9 the training cost. `hard_bc+proj` is the best configuration measured on
  every axis.
* ~~A controlled sweep puts numbers on the operator: `L2_after ~ 0.905 *
  L2_before + 7.6e-06` ... break-even is at L2 ~ 8e-05.~~ **Superseded, see
  `THEORY.md` §5b.** The additive term was not a property of the operator. It
  was caused by the elliptic solve being given Dirichlet data frozen at `t = 0`
  while the soliton's tails move, together with `C1` targets that ignore
  boundary flux. With consistent data the projection is idempotent on the exact
  solution to `4.15e-17`, the additive term vanishes, and the ratio is `0.9095`
  from `L2 = 1e-08` to `1e-01` — constant to four digits over eight orders of
  magnitude. **There is no break-even.**
* Not established: any accuracy benefit over time. `hard_bc` leads at every one
  of 20 time levels with a roughly constant ratio, so there is no crossover on
  this horizon. 96–99.9% of the error is shape, which neither a translation nor
  an amplitude rescale explains, so conserving two scalars cannot reach it.
* RLW1 is probably too easy to discriminate: a single wave of amplitude 0.3 is
  nearly trivial for a PINN. RLW2 was chosen next for that reason.

## What the theory changed (2026-08-02)

Full statements and proofs in `THEORY.md`; numbers regenerated by
`tools/verify_theory.py`. The results that alter how experiments should be run:

* **There is a computable ceiling on what conservation can achieve.** Only the
  component of the error inside `span{grad I1, grad I2}` — a 2-dimensional
  subspace — is reachable. For `hard_bc` on RLW1 that is 6.72% of the error, so
  the maximum possible improvement is 3.42%. The projection realises 3.402%.
  Chasing a larger forward-accuracy gain on this problem is chasing something
  that provably does not exist.

* **Two data defects must be fixed before any further projection runs.** The
  elliptic solve is given `ga`, `gb` frozen at `t = 0` (mismatch 6.9e-06 and
  1.3e-05 at `t = T` on RLW1) and `C1` targets that ignore boundary flux
  (4.8e-06 relative). Both are needed; neither alone suffices. `ga_fn`/`gb_fn`
  are already read from config (`train.py:84`), so the boundary half is
  **config-only and does not move `CODE_VERSION`**. Time-dependent `C1(t)`,
  `C2(t)` require a numerical change — the same one already queued for RLW3,
  now known to matter on every problem rather than only the bore.

* **Baseline accuracy does not affect the conclusions.** The improvement ratio
  is constant over eight orders of magnitude of baseline error, so a short
  training budget does not bias the comparison. One long run is still worth
  doing to answer the objection directly, but a full sweep at a larger budget
  is not needed.

* **Test problems should be selected by derivative share**, `1 - C2_poly_ref /
  C2_ref`: RLW1 is 1.8%, RLW2 is 10.8%. RLW1 barely exercises the feature that
  distinguishes this method. RLW2 should be the centrepiece; RLW1 stays as a
  baseline with its share reported openly.

* **RLW2's failure is less costly than it looked.** The collision is the only
  configuration tested in which `mu` is identifiable from conservation at all
  (`int u_x^2` varies by 0.71 there against 1.7e-09 for a single soliton), so
  it is the right problem for the inverse direction regardless of the forward
  training failure.

* **`I3` should not get a projection.** Cubic, so the solve becomes nonlinear;
  it contains no derivative, so enforcing it is not novel; and it would move
  the ceiling only from 3.42% to about 5%. Use it for `eps` recovery instead.

## Known gaps, carried deliberately

**RLW3 conservation targets.** The bore injects mass at the left boundary at
`dC1/dt = 0.108`, which moves `C1` by about 150% over `T=50`. The invariants are
not constant, so projecting onto their initial values enforces the wrong
target. The trainer prints this at set-up. RLW3 conservation numbers are not
publishable until time-dependent `C1(t)`, `C2(t)` are implemented; its boundary
error and solution accuracy are still meaningful.

**RLW2 boundary flux.** The tall soliton's tail is 1.3e-04 at `x = a`, so mass
is injected and the true `C1` grows by 1.06e-04 relative over `T=30`. Measured
on the reference solution, where it is identical on every grid, which is how we
know it is physical rather than discretisation error. Enforcing a constant `C1`
is therefore slightly wrong here too — the same defect as the bore, four orders
of magnitude smaller. It sits well under the ~1e-03 PINN error, so a constant
target is acceptable, but it belongs in the paper's limitations rather than
being left unsaid. `I2` and `I3` move by 4.5e-07 and 7.5e-09 and need no
qualification.

**`vanilla` baseline.** L2 = 0.53 on RLW1, which is not a defensible baseline
for a paper. Dropped from the RLW2 sweep rather than reported as-is. It needs
fixing as its own task before any submission, because a reviewer will ask what a
standard PINN achieves.

**KP.** No 2D trainer. The stale notebooks were removed rather than left
broken.

**End-to-end gradient.** Exact for `du*/du~` and for the tangent at fixed
state; the projector state is frozen when differentiating with respect to the
network parameters, so second-order dependence is dropped. Standard practice,
and stated as such in the derivations document rather than claimed as fully
exact.

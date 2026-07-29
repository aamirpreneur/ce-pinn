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

## Queued for a later version

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

## Known gaps, carried deliberately

**RLW3 conservation targets.** The bore injects mass at the left boundary at
`dC1/dt = 0.108`, which moves `C1` by about 150% over `T=50`. The invariants are
not constant, so projecting onto their initial values enforces the wrong
target. The trainer prints this at set-up. RLW3 conservation numbers are not
publishable until time-dependent `C1(t)`, `C2(t)` are implemented; its boundary
error and solution accuracy are still meaningful.

**KP.** No 2D trainer. The stale notebooks were removed rather than left
broken.

**End-to-end gradient.** Exact for `du*/du~` and for the tangent at fixed
state; the projector state is frozen when differentiating with respect to the
network parameters, so second-order dependence is dropped. Standard practice,
and stated as such in the derivations document rather than claimed as fully
exact.

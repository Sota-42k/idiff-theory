# How the measurement is set up

The generative model, the split and oracle boundaries, the perturbation, the metric
conventions, and what was frozen before any number was reported. Read alongside
[`results.md`](results.md), which follows the same order as the manuscript's Section 4.

---

## The generative model

Bounded exponential-family latents, the compatible example of the manuscript:

    z_i | u  ~  exp( a_i(u) z_i + b_i(u) z_i^2 ) / Z_i(u)   on  Z_i = (-1, 1)
    T_i(z)   = (z, z^2)                                      so  k = 2
    x        = g(z)

`artanh` carries the bounded box onto `R^n`, which is what makes the statistic bounds and the
finite overlaps of that example available while the observation space is all of `R^d`.
Boundedness is a sufficient design choice, not a necessary property of an identifiable model.

`u` is uniform on `nk + 1 = 2n + 1` values, exactly the reference points of (G3). Natural
parameters are drawn i.i.d. on `[-s, s] x [-s, -0.1]` with `s = 1`; `b_i < 0` keeps the
density well behaved. Draws whose `L` is numerically rank deficient are rejected at the
predeclared cutoff `sigma_min(L)/sigma_max(L) > 1e-6`, and the rejected count is recorded.
This is a conditioned benchmark, not a claim about all draws.

Two observation maps are measured, and they answer different questions:

| case | `g` | role |
|---|---|---|
| `artanh` | `artanh` component-wise | analytic control. `Dg` is **diagonal**, so the coordinates never mix, and every derivative, inverse and log-determinant is closed form. Any error is the solver's. |
| `mixing` | `h o artanh`, three layers of `x -> R_j chi(x)` with `R_j` seeded orthogonal and `chi(y) = tanh(y) + 0.1 y` | the nonlinear benchmark. `chi' in [0.1, 1.1]` makes `h` a bi-Lipschitz `C^inf` diffeomorphism. `g^{-1}` is found by bisection, so this case carries a numerical error source the control does not. |

The control is degenerate on purpose: because its Jacobian is diagonal it is a special case of
the diffeomorphism the theory addresses, and it exists to separate solver error from mixing
error rather than to stand in for the general case.

Jacobians, inverses and log-determinants are computed analytically in NumPy, never by
autograd, and are checked against central differences in `tests/`.

Core: `n = 2` (`q = 4`, five conditions), seeds 0-9. `n = 3` (`q = 6`, seven conditions)
replicates it and is reported as a replication, not as a second main result.

Both readout routes run on every case, consuming the same score array and the same splits.

---

## The perturbation

One saved reproducible smooth function of `(x, u)`, identical across every split and
amplitude; drawn from a different function class than the solver's basis, so that
representability is not built into the experiment; in a conservative and a nonconservative
variant, so that the projection's behaviour is visible.

The solver basis is linear and quadratic monomials plus seeded Gaussian-Fourier cosines, so
the perturbation is a seeded two-layer `tanh` network. With `x_tilde = (x - m)/s`, the
coordinate mean and standard deviation taken once on the perturbation-calibration split and
saved,

    h(x)  = tanh( W2 tanh( W1 x_tilde + c1 ) + c2 )    in R^32, hidden width 32
    Dh(x) = its analytic Jacobian

- **conservative** (`gradient`): `delta(x,u) = alpha * Dh(x)^T omega_u`, which is exactly
  `alpha * grad_x <omega_u, h(x)>`. A gradient field, so the projection of `eq:ls-potential`
  loses nothing in population.
- **nonconservative** (`field`): `delta(x,u) = alpha * V_u h(x)` with `V_u` seeded. Generically
  nonzero curl, so the projection discards a component. That is a property of the estimator,
  not an error; the realized curl energy is recorded so the discarded part stays visible.

`W1, c1, W2, c2, omega_u, V_u, m, s` come from one seed stream, are saved to the run record,
and are reused verbatim. A perturbation is one function, not fresh noise per evaluation.
Neither variant is claimed to be the score of a normalized density; what is measured is the
induced field error, which is what `J_Delta` is defined on.

**Amplitude.** The realized size is `rho(delta)^2 = E_{p_0} E_pi ||delta(x,u)||^2`. The base
function is normalized once, on the perturbation-calibration split, so that `rho = 1` at
`alpha = 1`; that scalar is frozen and saved. `rho` is then recomputed on test and reported.
`alpha` is the nominal knob; the realized `rho` is the number on any axis. The grid is
`alpha = 0` and nine values from `1e-4` to `1` at half-decade spacing, chosen so that the
floor is visible in more than one point and the injection dominates in the upper half.

**The directional construction.** For the conditioning section, with `G = L^T = U diag(s) V^T`
and `v = grad psi_v`, inject `delta(x, u^0) = 0` and `delta(x, u^l) = alpha U[l, j] v(x)`. The
field error is then exactly rank one, so the least-squares potential acts row-wise and the
readout error is `alpha psi_v_hat(x) U[:, j]`: the raw error is identical for every mode `j`
while the aligned error is `alpha psi_v_hat V[:, j] / s_j`. Selecting `U` from the true `L` is
an oracle control, labelled as one in every record, and is not an operation an estimator could
perform.

---

## Splits and the oracle boundary

Five mutually independent splits per seed, drawn from the same generated model through
`stream_seed(seed, k)`; parameters use stream 0 and the mixing stream 1.

| stream | split | per condition | used for | never used for |
|---|---|---|---|---|
| 3 | `fitting` | 10000 | the potential coefficients | any reported number |
| 4 | `validation` | 4000 | the solver setting, on seeds held out from production | anything after the freeze |
| 5 | `pert_calibration` | 4000 | the perturbation normalization and standardization | any reported error |
| 6 | `calibration` | 4000 | the affine map and the additive centring offset | evaluation |
| 7 | `test` | 10000 | every reported number | any fit or selection |

Counts are per condition, so `(q+1)` times those totals. Splits are stored condition by
condition, so anything compared against a `p_0` expectation is evaluated on a sample drawn
from every condition, never on a head slice.

Rules enforced in code rather than by discipline:

- `fit_affine` only ever sees calibration; `affine_residual` evaluates a map it was handed and
  structurally cannot refit.
- The centring offset is `mean_calibration(Lambda_hat - Lambda)`, with its standard error
  reported. Test data never determines a constant.
- `V_T`, the normalizing denominator, is one common held-out variance, saved and reused for
  both routes.
- Uncertainty on test means is stratified by condition. Seed spread is reported as mean and SD
  and labelled as spread, not as a confidence interval.

**Oracle truth** --- `L`, `kappa`, `lambda`, `Upsilon = T o g^{-1}`, `DUpsilon`, the analytic
clean scores --- is used for exactly three things, each labelled in the output: building the
controlled fields, evaluating error, and selecting directions in the conditioning section. It
never enters a fitting routine.

---

## Metrics and units

All continuous expectations are under the clean marginal `p_0`.

| metric | definition | role |
|---|---|---|
| field error | `E ||w_hat - grad Lambda||^2`, summed over `l` | the input the bound is stated on |
| solver residual | held-out `E ||grad Lambda_hat - w_hat||^2` | the solver's own objective, on data it did not fit |
| raw readout error | `E ||e||^2` with `e` centred on calibration | the readout's own coordinates |
| oracle-aligned error | `E ||G^{-1} e||^2`, `G = L^T` or `C_phi` | the left-hand side of `eq:quantitative` / `eq:quantitative-pi` |
| fitted-affine MSE, NMSE | `A, b` fitted on calibration, evaluated on test against `V_T` | secondary |

Squared and unsquared, normalized and unnormalized, raw and aligned are kept distinct
throughout and named in the record.

The fitted-affine metric is secondary for a structural reason, not a stylistic one: it is
invariant to any invertible reparameterization of the readout, because the fit absorbs it. It
therefore cannot distinguish an alignment that recovers the factorization from one that
destroys it, which the factorization section measures directly.

A point is marked injection-dominated when its raw readout MSE reaches ten times that run's
own `alpha = 0` readout error. The comparison is against the readout baseline and not against
a field residual: the fields are built from analytic scores, so the `alpha = 0` field error is
at the float64 floor and sets no usable scale. The numerical baseline lives entirely in the
potential solver. MSEs are never subtracted --- the injected and baseline errors are not
orthogonal and the cross term is not assumed to vanish --- and baseline-dominated points are
reported, not dropped.

---

## What was frozen, and what the freeze does not claim

The potential solver has two numerical parameters, the basis width and the ridge. They are
chosen **per case**: a single shared setting is measurably worse for one of the two, and a
control that is handicapped relative to the benchmark is not serving as a control.

Selection ran on seeds disjoint from the production seeds, on the validation split, under a
criterion fixed in `scripts/select_potential.py` before it ran: the minimum geometric mean of
the oracle-aligned MSE per case, ties within 5% going to the smaller basis. It is an **oracle
validation selection**, labelled as one, and frozen before any test number was computed. The
grid is a complete cross of three widths with three ridges, geometrically spaced; a partial
grid hides exactly the cell that matters. The record is in `results/selection/`.

The configurations state what the freeze does not claim. In particular one case is selected at
the edge of the ridge axis, where lowering the ridge further keeps improving the metric while
the Gram condition number degrades by the same factor, so the axis has no interior optimum.
That is an optimisation question about the numerical solver rather than a property of the
recovery map, it is out of scope by decision, and that setting is not claimed to be optimal.

The production sensitivity axis reports the same complete cross on the production seeds. It is
a report, not a re-selection: nothing chosen before the freeze is revisited against it.

---

## Outputs

`results/recovery/<config_id>/seed<k>.json`, one per seed, with a manifest and a summary CSV.
Each record carries the full configuration, the generated parameters (`natural`, `L`, `kappa`,
singular values, rejected draws, rank cutoff), the mixing rotations, the saved perturbation
arrays and their frozen normalization, the solver diagnostics (`gram_condition`,
normal-equation residual, status, fit seconds), every split seed and size, the environment and
source hashes, the wall clock, and a `status`.

A failed seed is written and stays in the manifest with its reason; the analysis excludes it
from every number and says so rather than quietly analysing the survivors. Numerical
exclusions require a written reproducible rule.

---

## Module layout

`src/recovery_core/`, NumPy and SciPy only:

| module | contents |
|---|---|
| `generator.py` | bounded exponential-family sources, both mixings, the oracle |
| `readout.py` | both field constructions, the finite-basis potential and its batched form |
| `perturbation.py` | the saved perturbations, their calibration, the curl diagnostics |
| `recovery.py` | splits, both routes, the evaluation chain |
| `metrics.py` | raw, aligned and fitted errors, Fisher diagnostics, the block-permutation defect |
| `directions.py` | the singular-direction accounting |
| `bounds.py` | the analytic comparison constants of the plug-in panel |
| `runs.py` | configuration validation, provenance hashing, result files |

`tests/` holds the exact fixtures and the metric controls, including known-nonzero cases: a
zero control alone cannot detect an implementation that always returns zero.

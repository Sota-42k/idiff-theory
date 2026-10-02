# The recovery map, measured

From `data/recovery/2ab29ae68a24d88b`, configuration `2ab29ae68a24d88b`, frozen before the run. 10/10 seeds completed with no failures, and
the source did not change between the run and this analysis. Wall clock 7.15 h. Regenerate with `scripts/analyse_recovery.py`.

Every number is computed on the **test** split. The calibration split fixes the affine map and
the centring offset and nothing else. Spreads are across the 10 seeds, each drawing its own
generated model, so a seed SD is variation over models and samples together and not a
confidence interval for a fixed model.

Two observation maps are measured throughout. `artanh` applies `artanh` component-wise, so its
Jacobian is diagonal, every derivative is closed form, and any error is the solver's. `mixing`
composes that with three orthogonal/leaky-tanh layers and is the nonlinear benchmark. The
potential solver is configured per case --- `artanh` at 2048 basis functions with ridge 1e-08, `mixing` at 4096 basis functions with ridge 1e-09 --- because a shared setting is known to handicap one of them, and a control that
is handicapped against the benchmark is not serving as a control.

**What this is not.** No theorem is established or contradicted. These are finite-sample,
finite-basis estimates of population statements whose hypotheses ((H), (P), (R1),
(G3)/(G3-phi)) are assumed throughout. The bound panel is a plug-in diagnostic with a
deliberately loose constant, not a certificate. Ground truth is used to build the controlled
fields, to evaluate, and to choose directions in the conditioning section; each use is
labelled where it occurs and none of it reaches a fitting routine.

## The readout identity

`prop:readout` states that the readout is `Lambda(x) = L^T T(g^{-1}(x)) - kappa`, with no
learnable component. That identity, its gradient against the contrast field, and the
invariance of both field constructions to a condition-independent score field
(`rem:guidance`) are exact statements, so they are checked as exact fixtures rather than
measured: `tests/test_generator.py` and `tests/test_readout.py`, in float64, to tolerances
of `2e-13` on the identity and `2e-12` on the fields, for `n` in `{2,3}` and both maps.

Nothing below re-establishes them. Everything below measures what happens once the readout is
computed from an estimated score through a finite-basis potential.

## The least-squares potential

`eq:ls-potential` defines the estimate as the least-squares potential of the readout field
over `H^1(p_0)`. The implementation replaces that population projection with a ridge solve
over a finite basis, and this section measures what the replacement costs.

The fields here are built from analytic clean conditional scores, so they are exact: the
measured field error at zero injected error is `1.3e-31` for `artanh` and `3.1e-30` for `mixing`, the float64 floor. The entire numerical baseline therefore sits in the solver, which
is why the injection threshold later in this document is measured against this readout error
and never against a field residual.

| case | route | metric | mean | SD | min | max | spread |
|---|---|---|---|---|---|---|---|
| `artanh` | reference | oracle-aligned MSE | 6.011e-06 | 4.2e-06 | 6.06e-07 | 1.29e-05 | 21.3x |
| `artanh` | reference | raw readout MSE | 6.861e-06 | 7.9e-06 | 1.68e-06 | 2.84e-05 | 16.9x |
| `artanh` | reference | affine NMSE | 7.978e-06 | 5.6e-06 | 8.03e-07 | 1.74e-05 | 21.7x |
| `artanh` | reference | solver residual fraction | 1.074e-05 | 5.4e-06 | 3.56e-06 | 1.85e-05 | 5.2x |
| `artanh` | regression | oracle-aligned MSE | 6.011e-06 | 4.2e-06 | 6.06e-07 | 1.29e-05 | 21.3x |
| `artanh` | regression | raw readout MSE | 8.099e-07 | 8.8e-07 | 1.67e-07 | 2.98e-06 | 17.9x |
| `artanh` | regression | affine NMSE | 7.978e-06 | 5.6e-06 | 8.03e-07 | 1.74e-05 | 21.7x |
| `artanh` | regression | solver residual fraction | 1.075e-05 | 4.8e-06 | 4.08e-06 | 1.83e-05 | 4.5x |
| `mixing` | reference | oracle-aligned MSE | 1.283e-05 | 3.8e-06 | 8.02e-06 | 1.73e-05 | 2.2x |
| `mixing` | reference | raw readout MSE | 9.328e-06 | 3.6e-06 | 3.00e-06 | 1.63e-05 | 5.4x |
| `mixing` | reference | affine NMSE | 1.703e-05 | 5.2e-06 | 1.07e-05 | 2.32e-05 | 2.2x |
| `mixing` | reference | solver residual fraction | 1.801e-03 | 4.7e-04 | 1.05e-03 | 2.49e-03 | 2.4x |
| `mixing` | regression | oracle-aligned MSE | 1.283e-05 | 3.8e-06 | 8.02e-06 | 1.73e-05 | 2.2x |
| `mixing` | regression | raw readout MSE | 1.045e-06 | 4.2e-07 | 2.80e-07 | 1.82e-06 | 6.5x |
| `mixing` | regression | affine NMSE | 1.703e-05 | 5.2e-06 | 1.07e-05 | 2.32e-05 | 2.2x |
| `mixing` | regression | solver residual fraction | 1.825e-03 | 4.2e-04 | 1.25e-03 | 2.54e-03 | 2.0x |

**The baseline is far more seed-dependent for `artanh` than for `mixing`**: a spread of
21.3x against
2.2x, with coefficients of variation
0.70 and
0.30. A single-seed baseline is not
"the floor", and the injection threshold is therefore taken per seed against that run's own
zero-amplitude value rather than against a global constant.

**The solver residual separates the two maps by a factor of
168** (`1.07e-05` against `1.80e-03`). The nonlinear mixing makes `Upsilon = T o g^-1` materially harder for the basis to
represent, which is what the `artanh` control exists to separate out.

### Sensitivity of the frozen setting

The grid is a complete cross, reported on all 10 production seeds as the geometric mean of
the oracle-aligned MSE. A partial grid hides exactly the cell that matters, so every width is
run at every ridge.

| case | `n_features` | `ridge` | geo. mean | vs the frozen setting |
|---|---|---|---|---|
| `artanh` | 1024 | 1e-09 | 1.609e-05 | 3.76x |
| `artanh` | 1024 | 1e-08 | 8.341e-06 | 1.95x |
| `artanh` | 1024 | 1e-07 | 5.468e-06 | 1.28x |
| `artanh` | 2048 | 1e-09 | 7.850e-06 | 1.83x |
| `artanh` | 2048 | 1e-07 | 5.270e-06 | 1.23x |
| `artanh` | 4096 | 1e-09 | 5.477e-06 | 1.28x |
| `artanh` | 4096 | 1e-08 | 5.998e-06 | 1.40x |
| `artanh` | 4096 | 1e-07 | 7.976e-06 | 1.86x |
| `artanh` | **2048** | **1e-08** | **4.281e-06** | frozen |
| `mixing` | 1024 | 1e-09 | 3.106e-05 | 2.52x |
| `mixing` | 1024 | 1e-08 | 4.268e-05 | 3.47x |
| `mixing` | 1024 | 1e-07 | 6.286e-05 | 5.10x |
| `mixing` | 2048 | 1e-09 | 1.820e-05 | 1.48x |
| `mixing` | 2048 | 1e-08 | 2.632e-05 | 2.14x |
| `mixing` | 2048 | 1e-07 | 3.712e-05 | 3.01x |
| `mixing` | 4096 | 1e-08 | 1.921e-05 | 1.56x |
| `mixing` | 4096 | 1e-07 | 2.831e-05 | 2.30x |
| `mixing` | **4096** | **1e-09** | **1.232e-05** | frozen |

The setting was chosen on seeds disjoint from these, on the validation split, under a
criterion fixed before that selection ran; `configs/` records it, including what it does not
claim to be optimal. The rows above are a report, not a re-selection.

## Quantitative recovery

`thm:recovery` (2) bounds the recovery error of the reference-point readout by the conditional
Fisher divergence of the field it consumes (`eq:quantitative`); part (3) does the same for the
observable-feature readout through `C_phi` (`eq:quantitative-pi`). This section injects a
saved smooth score error of controlled size and follows it through the whole chain.

The perturbation is one reproducible function of `(x,u)`, saved as arrays and reused across
every split and amplitude, built from a tanh network so that it lies outside the solver's own
basis. It comes in a conservative variant, exactly `grad_x psi(x,u)`, and a nonconservative
one. Neither is claimed to be the score of a normalized density; what is measured is the
induced field error, which is what `J_Delta` is defined on.

Means over the 10 seeds, reference route. `vs base` is against each run's own zero-amplitude
readout error, and a point is marked dominated when that ratio reaches ten.


### `artanh` / gradient

| `alpha` | realized `rho` | field MSE | raw MSE | oracle-aligned MSE | affine NMSE | vs base | dominated | aligned / `alpha^2` |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 1.26e-31 | 6.86e-06 | 6.01e-06 | 7.98e-06 | 1.00 | baseline | - |
| 1.00e-04 | 9.99e-05 | 7.20e-08 | 6.88e-06 | 7.14e-06 | 8.20e-06 | 1.01 | 0/10 | 7.137e+02 |
| 3.16e-04 | 3.16e-04 | 7.20e-07 | 7.08e-06 | 1.74e-05 | 1.04e-05 | 1.06 | 0/10 | 1.742e+02 |
| 1.00e-03 | 9.99e-04 | 7.20e-06 | 9.13e-06 | 1.21e-04 | 3.18e-05 | 1.61 | 0/10 | 1.206e+02 |
| 3.16e-03 | 3.16e-03 | 7.20e-05 | 2.97e-05 | 1.15e-03 | 2.36e-04 | 7.10 | 2/10 | 1.153e+02 |
| 1.00e-02 | 9.99e-03 | 7.20e-04 | 2.35e-04 | 1.15e-02 | 2.02e-03 | 61.97 | 9/10 | 1.148e+02 |
| 3.16e-02 | 3.16e-02 | 7.20e-03 | 2.29e-03 | 1.15e-01 | 1.57e-02 | 610.63 | 10/10 | 1.148e+02 |
| 1.00e-01 | 9.99e-02 | 7.20e-02 | 2.29e-02 | 1.15e+00 | 5.88e-02 | 6097.02 | 10/10 | 1.148e+02 |
| 3.16e-01 | 3.16e-01 | 7.20e-01 | 2.29e-01 | 1.15e+01 | 1.42e-01 | 60960.47 | 10/10 | 1.148e+02 |
| 1.00e+00 | 9.99e-01 | 7.20e+00 | 2.29e+00 | 1.15e+02 | 2.45e-01 | 609593.29 | 10/10 | 1.148e+02 |

### `artanh` / field

| `alpha` | realized `rho` | field MSE | raw MSE | oracle-aligned MSE | affine NMSE | vs base | dominated | aligned / `alpha^2` |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 1.26e-31 | 6.86e-06 | 6.01e-06 | 7.98e-06 | 1.00 | baseline | - |
| 1.00e-04 | 1.00e-04 | 6.18e-08 | 6.91e-06 | 7.03e-06 | 8.19e-06 | 1.01 | 0/10 | 7.033e+02 |
| 3.16e-04 | 3.17e-04 | 6.18e-07 | 7.24e-06 | 1.64e-05 | 1.03e-05 | 1.10 | 0/10 | 1.643e+02 |
| 1.00e-03 | 1.00e-03 | 6.18e-06 | 1.04e-05 | 1.11e-04 | 3.23e-05 | 1.93 | 0/10 | 1.108e+02 |
| 3.16e-03 | 3.17e-03 | 6.18e-05 | 4.18e-05 | 1.06e-03 | 2.81e-04 | 10.24 | 4/10 | 1.056e+02 |
| 1.00e-02 | 1.00e-02 | 6.18e-04 | 3.54e-04 | 1.05e-02 | 4.05e-03 | 93.01 | 10/10 | 1.051e+02 |
| 3.16e-02 | 3.17e-02 | 6.18e-03 | 3.47e-03 | 1.05e-01 | 3.00e-02 | 920.03 | 10/10 | 1.051e+02 |
| 1.00e-01 | 1.00e-01 | 6.18e-02 | 3.46e-02 | 1.05e+00 | 6.86e-02 | 9187.87 | 10/10 | 1.051e+02 |
| 3.16e-01 | 3.17e-01 | 6.18e-01 | 3.46e-01 | 1.05e+01 | 1.34e-01 | 91858.97 | 10/10 | 1.051e+02 |
| 1.00e+00 | 1.00e+00 | 6.18e+00 | 3.46e+00 | 1.05e+02 | 2.05e-01 | 918546.69 | 10/10 | 1.051e+02 |

### `mixing` / gradient

| `alpha` | realized `rho` | field MSE | raw MSE | oracle-aligned MSE | affine NMSE | vs base | dominated | aligned / `alpha^2` |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 3.09e-30 | 9.33e-06 | 1.28e-05 | 1.70e-05 | 1.00 | baseline | - |
| 1.00e-04 | 9.99e-05 | 7.69e-08 | 9.34e-06 | 1.34e-05 | 1.73e-05 | 1.00 | 0/10 | 1.337e+03 |
| 3.16e-04 | 3.16e-04 | 7.69e-07 | 9.43e-06 | 1.83e-05 | 1.92e-05 | 1.01 | 0/10 | 1.827e+02 |
| 1.00e-03 | 9.99e-04 | 7.69e-06 | 1.05e-05 | 6.74e-05 | 3.88e-05 | 1.16 | 0/10 | 6.742e+01 |
| 3.16e-03 | 3.16e-03 | 7.69e-05 | 2.10e-05 | 5.59e-04 | 2.35e-04 | 2.58 | 0/10 | 5.595e+01 |
| 1.00e-02 | 9.99e-03 | 7.69e-04 | 1.27e-04 | 5.48e-03 | 2.10e-03 | 16.86 | 9/10 | 5.482e+01 |
| 3.16e-02 | 3.16e-02 | 7.69e-03 | 1.19e-03 | 5.47e-02 | 1.44e-02 | 159.88 | 10/10 | 5.471e+01 |
| 1.00e-01 | 9.99e-02 | 7.69e-02 | 1.18e-02 | 5.47e-01 | 5.72e-02 | 1590.63 | 10/10 | 5.470e+01 |
| 3.16e-01 | 3.16e-01 | 7.69e-01 | 1.18e-01 | 5.47e+00 | 1.23e-01 | 15899.82 | 10/10 | 5.470e+01 |
| 1.00e+00 | 9.99e-01 | 7.69e+00 | 1.18e+00 | 5.47e+01 | 2.30e-01 | 158997.26 | 10/10 | 5.470e+01 |

### `mixing` / field

| `alpha` | realized `rho` | field MSE | raw MSE | oracle-aligned MSE | affine NMSE | vs base | dominated | aligned / `alpha^2` |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 3.09e-30 | 9.33e-06 | 1.28e-05 | 1.70e-05 | 1.00 | baseline | - |
| 1.00e-04 | 1.00e-04 | 6.24e-08 | 9.34e-06 | 1.32e-05 | 1.71e-05 | 1.00 | 0/10 | 1.321e+03 |
| 3.16e-04 | 3.16e-04 | 6.24e-07 | 9.44e-06 | 1.67e-05 | 1.81e-05 | 1.01 | 0/10 | 1.672e+02 |
| 1.00e-03 | 1.00e-03 | 6.24e-06 | 1.04e-05 | 5.19e-05 | 2.82e-05 | 1.15 | 0/10 | 5.193e+01 |
| 3.16e-03 | 3.16e-03 | 6.24e-05 | 2.04e-05 | 4.05e-04 | 1.29e-04 | 2.46 | 0/10 | 4.047e+01 |
| 1.00e-02 | 1.00e-02 | 6.24e-04 | 1.20e-04 | 3.93e-03 | 1.10e-03 | 15.64 | 8/10 | 3.934e+01 |
| 3.16e-02 | 3.16e-02 | 6.24e-03 | 1.12e-03 | 3.92e-02 | 7.77e-03 | 147.48 | 10/10 | 3.923e+01 |
| 1.00e-01 | 1.00e-01 | 6.24e-02 | 1.11e-02 | 3.92e-01 | 3.66e-02 | 1465.92 | 10/10 | 3.922e+01 |
| 3.16e-01 | 3.16e-01 | 6.24e-01 | 1.11e-01 | 3.92e+00 | 8.88e-02 | 14650.63 | 10/10 | 3.922e+01 |
| 1.00e+00 | 1.00e+00 | 6.24e+00 | 1.11e+00 | 3.92e+01 | 1.32e-01 | 146498.55 | 10/10 | 3.922e+01 |

### Slope of the aligned error in the amplitude

| case | kind | seeds | slope mean | SD | min | max |
|---|---|---|---|---|---|---|
| `artanh` | gradient | 10 | 1.9991 | 0.0014 | 1.9954 | 2.0000 |
| `artanh` | field | 10 | 1.9992 | 0.0009 | 1.9975 | 2.0001 |
| `mixing` | gradient | 10 | 1.9979 | 0.0031 | 1.9899 | 2.0000 |
| `mixing` | field | 10 | 1.9964 | 0.0047 | 1.9861 | 2.0000 |

The slope is `1.9982 +- 0.0030` over all 40
case/kind/seed fits, and `aligned / alpha^2` settles onto a constant once the injection
dominates.

**This is not evidence for the rate in `eq:quantitative`, nor for its sharpness.** The
injected field enters a linear ridge solve and the error is measured as a square; a slope of
two is what that arithmetic produces. What it does show is that the implemented chain hides no
extra amplitude dependence --- no regime where the readout error grows faster than the field
error it comes from --- which is a check on the implementation, not on the theorem.

### Where the injection overtakes the baseline

| case | kind | first dominated `alpha` | seeds dominated there | fully dominated from |
|---|---|---|---|---|
| `artanh` | gradient | 3.16e-03 | 2/10 | 3.16e-02 |
| `artanh` | field | 3.16e-03 | 4/10 | 1.00e-02 |
| `mixing` | gradient | 1.00e-02 | 9/10 | 3.16e-02 |
| `mixing` | field | 1.00e-02 | 8/10 | 3.16e-02 |

The 3 smallest amplitudes are baseline-dominated in every seed of every
configuration, which is what makes the floor visible. No point was dropped for being
baseline-dominated. The baseline and the total are both reported and **MSEs are never
subtracted**: the injected and baseline errors are not orthogonal and the cross term is not
assumed to vanish.

### The fitted-affine metric saturates; the aligned error does not

| case | kind | oracle-aligned MSE at the top amplitude | affine NMSE there |
|---|---|---|---|
| `artanh` | gradient | 1.148e+02 | 0.2449 +- 0.0678 |
| `artanh` | field | 1.051e+02 | 0.2053 +- 0.0670 |
| `mixing` | gradient | 5.470e+01 | 0.2302 +- 0.0956 |
| `mixing` | field | 3.922e+01 | 0.1324 +- 0.0486 |

The aligned error reaches `4e+01` to `1e+02` while the held-out affine
NMSE stays between `0.07` and `0.45`. The fitted affine map absorbs
and discards much of the error. That is why the fitted metric is secondary here and the
oracle-aligned error is the one that diagnoses the theorem's left-hand side, and it is a
caution against reading a normalized recovery error as if it measured the readout.

### The two readout routes

`C_phi = Sigma_phi^(1/2) L^T`, and both routes consume the same scores and splits through one
linear ridge projection, so their oracle-aligned and fitted-affine metrics must coincide.

Over all 380 paired rows the largest relative gap in either is `1.64e-08`. The *raw* readout errors do not agree and should not: reference over regression runs from 5.6 to 14.0, the same error expressed in different coordinates, and the two are not interchangeable.

This is an implementation control. It says the two constructions were implemented consistently. It is not independent evidence for either, and it is not a comparison one route can win.

### Conservative against nonconservative

| case | aligned MSE ratio, nonconservative / conservative, at the top amplitude | seeds above 1 |
|---|---|---|
| `artanh` | 1.292 +- 0.503 | 6/10 |
| `mixing` | 0.768 +- 0.329 | 2/10 |

Nothing is concluded from this. The projection of `eq:ls-potential` discards the curl
component, whose energy is recorded in each seed file, and how much of a perturbation of fixed
size lands in that component varies with the draw.

## Conditioning of the alignment

The bounds carry the alignment inverse: `1/sigma_min(L)^2` in `eq:quantitative` and
`1/sigma_min(C_phi)^2` in `eq:quantitative-pi`. Injecting `delta(x,u^0) = 0` and
`delta(x,u^l) = alpha U[l,j] v(x)`, with `v` a gradient field and `U[:,j]` a left singular
vector of `G = L^T`, makes the field error exactly rank one. The raw readout error is then the
same for every mode `j` while the oracle-aligned error should be `raw / s_j^2`. Direction
selection uses the true `L`: an oracle control, available to no estimator.

| quantity | over all 240 rows |
|---|---|
| oracle-aligned MSE over predicted `raw / s_j^2` | max deviation from 1: `1.54e-14` |
| raw MSE across modes, max/min within a group | `1.0000000000` |
| rank-one residual of the readout error | median `6.04e-12`, max `1.54e-09` |
| full pipeline over injection-only aligned MSE | `0.999` to `1.010` |
| `sigma_min(L)` over the 10 generated models | `0.0230` to `0.4694` |
| `cond(L)` over the 10 generated models | `2.3` to `93.2` |

At a fixed raw readout error the aligned error moves by the square of the singular-value ratio:
`5` to `8695` between the strongest and weakest
mode across these models, `1373` on average. The full pipeline agrees
with the injection-only construction to within `1.0%`, these
amplitudes all being injection-dominated.

This isolates the alignment factor at a **fixed** generated model: the law, the overlaps and
the score difficulty are untouched and only the direction of the error moves. A separation
sweep cannot do this, changing all of them at once. What is demonstrated is the amplification
in the construction, at these models. It says nothing about the tightness of `eq:quantitative`,
and the direction selection is not an operation an estimator could perform.

### `varkappa`

`rem:overlap-tension` observes that the reference points enter `eq:quantitative-fisher` only
through `varkappa = M / sigma_min(L)^2`, that choosing them means minimising it --- enough
diversity for a well-conditioned `L`, not so much that the conditionals stop overlapping ---
and suggests reporting it alongside any empirical result.

`M = max_l sup_x p_0(x)/p(x|u^l)` is given two ways, bounding it from opposite sides. `M_hat`
is the maximum over the test sample, a **lower** approximation to the supremum. `M_analytic`
is the bounded-density comparison formula of `bounds.py`, a deliberately loose **upper** bound.
Both depend on the latent model and the reference points alone, so they are identical for the
two observation maps, which share a generated model per seed.

| quantity | mean | SD | min | max |
|---|---|---|---|---|
| `M_hat` | 6.264e+00 | 3.4e+00 | 2.56e+00 | 1.38e+01 |
| `M_analytic` | 8.568e+02 | 1.1e+03 | 4.40e+01 | 3.49e+03 |
| `sigma_min` | 1.611e-01 | 1.4e-01 | 2.30e-02 | 4.69e-01 |
| `kappa_hat` | 1.770e+03 | 3.1e+03 | 1.57e+01 | 1.01e+04 |
| `kappa_analytic` | 1.887e+05 | 2.5e+05 | 7.52e+02 | 6.14e+05 |

The two estimates of `M` differ by a factor of 106 on average, so `varkappa` is pinned
here only to within about 2 orders of magnitude. Anyone using it to choose
reference points is optimising a quantity of that uncertainty, which is worth saying alongside
the suggestion to report it.

Within that caveat, `varkappa_hat` runs from 1.57e+01 to 1.01e+04, a spread of
644x driven by `sigma_min(L)` (20x) rather than by `M`
(5.4x). The diversity term dominates the overlap term throughout, so the
tension the remark describes is not visible in this design: these reference points sit where
more diversity would still help. That is a property of how the natural parameters are drawn
here, not a general finding.

## An observable diagnostic

`rem:curl` proposes `C^(l)(x) = ||d_x w^(l) - (d_x w^(l))^T||_F`, which vanishes when the score
is exact, as a measure of how much estimation error has entered the readout. It is the only
diagnostic in the manuscript that needs no ground truth, and so the only quantity in this
document a practitioner could compute; everything else here is an oracle quantity. It is worth
knowing what it detects.

`C^(l)` below is root-mean-squared over the test sample and maximised over `l`, against the
oracle-aligned readout error at the same amplitude. It is exactly linear in the amplitude, the
true contrast being a gradient field.

| case | kind | `alpha` | `max_l C^(l)` | oracle-aligned MSE |
|---|---|---|---|---|
| `artanh` | gradient | 1.00e-04 | 2.541e-14 | 7.137e-06 |
| `artanh` | gradient | 1.00e-02 | 2.543e-12 | 1.148e-02 |
| `artanh` | gradient | 1.00e+00 | 2.542e-10 | 1.148e+02 |
| `artanh` | field | 1.00e-04 | 1.406e-04 | 7.033e-06 |
| `artanh` | field | 1.00e-02 | 1.406e-02 | 1.051e-02 |
| `artanh` | field | 1.00e+00 | 1.406e+00 | 1.051e+02 |
| `mixing` | gradient | 1.00e-04 | 9.862e-14 | 1.337e-05 |
| `mixing` | gradient | 1.00e-02 | 9.863e-12 | 5.482e-03 |
| `mixing` | gradient | 1.00e+00 | 9.864e-10 | 5.470e+01 |
| `mixing` | field | 1.00e-04 | 1.812e-04 | 1.321e-05 |
| `mixing` | field | 1.00e-02 | 1.812e-02 | 3.934e-03 |
| `mixing` | field | 1.00e+00 | 1.812e+00 | 3.922e+01 |

**The diagnostic is blind to a conservative score error.** The gradient-kind perturbation is
exactly `grad_x psi(x,u)`, so its contrast is a gradient field and `C^(l)` stays at the
central-difference floor at every amplitude --- at most `1.1e-09` over all rows --- while
the readout error it causes runs from `7e-07` to `7e+02`. That is not
an exotic class: `eq:ls-potential` is itself a projection onto gradient fields, so the
component of the error that survives the projection and reaches the readout is exactly the
component `C^(l)` cannot see.

**On the non-gradient part it is informative.** For the field kind `C^(l)` is at least
`1.4e-04`, linear in the amplitude while the readout error is quadratic, so within a kind
it orders the runs correctly.

`rem:curl` states that `C^(l)` vanishes when the score is exact. That is true, and the
gradient-kind rows sit at the floor as it says. The converse does not hold and the remark does
not assert it, but a reader could take "its magnitude measures how much estimation error has
entered the readout" that way. On this evidence it measures how much **non-gradient**
estimation error has entered. A one-line qualification would foreclose the stronger reading.
This is an observation about the diagnostic's reach, not a counterexample to anything the
manuscript proves.

## From affine recovery to the factorization

`cor:factorization` lifts `~_A` to `~_P`. The Discussion distinguishes affine readout
error from component recovery. This section measures how
little the first constrains the second.

It is **not** a factorization search. No witness is constructed and nothing here bears on
whether the search of Section 4.4 is tractable, which the manuscript explicitly declines to
claim.

A candidate alignment is written `M A_0`, with `A_0` the true `(L^T)^{-1}`. What `~_P` permits
is `M` in the block-permutation group: a permutation of the `n` blocks of size `k` with free
invertible maps inside each block. `block defect` is the fraction of squared Frobenius energy
`M` places off the best block pattern, zero exactly when `M` is admissible. `single-factor
R^2` regresses each recovered coordinate on a quadratic in each single latent and keeps the
best, so a block-separated readout scores 1.

### With the exact readout, no estimation error

*no fitting at all, so any blindness here is structural.*

| candidate alignment | affine NMSE | block defect | single-factor R^2 |
|---|---|---|---|
| identity | 1.171e-29 | 0.0000 | 1.000000 |
| random invertible 1 | 6.198e-30 | 0.2170 | 0.799080 |
| random invertible 2 | 1.236e-29 | 0.3797 | 0.627132 |
| random invertible 3 | 5.895e-30 | 0.4356 | 0.734042 |
| block permutation | 2.708e-30 | 0.0000 | 1.000000 |
| block-mixing shear | 7.873e-30 | 0.4194 | 0.931255 |

### With this run's estimated readouts

*the same question carried by the run's own errors.*

| candidate alignment | affine NMSE | block defect | single-factor R^2 |
|---|---|---|---|
| identity | 1.251e-05 | 0.0000 | 0.999978 +- 0.0000 |
| random invertible 1 | 1.251e-05 | 0.4099 | 0.767143 +- 0.0948 |
| random invertible 2 | 1.251e-05 | 0.3628 | 0.797869 +- 0.1053 |
| random invertible 3 | 1.251e-05 | 0.3587 | 0.751710 +- 0.0654 |
| block permutation | 1.251e-05 | 0.0000 | 0.999978 +- 0.0000 |
| block-mixing shear | 1.251e-05 | 0.4194 | 0.934945 +- 0.0032 |

**The affine metric does not distinguish them.** Across all 120 estimated
rows the fitted-affine NMSE differs from the identity alignment by at most `9.33e-13` in
relative terms, and with the exact readout it is at the float64 floor for every candidate. It
is invariant to any invertible reparameterization of the readout, because the fit absorbs it.

**The block structure does distinguish them.** Only the 2 candidates `~_P` permits
reach zero defect and `R^2` of one; the others lose component recovery while leaving the affine
metric untouched.

`~_A` asks whether some invertible `A` exists, which is always true once recovery holds. `~_P`
asks whether the `A` one has is a block permutation away from `A_0`, which no affine criterion
can test. That gap is what the search of Section 4.4 exists to close, under hypotheses it must
establish separately.

One caution the numbers make concrete: a candidate that mixes blocks can still score high on
`single-factor R^2`. A high value is not evidence of factorization.

## Plug-in bound panel

**A plug-in population-bound diagnostic, not a certificate.** `C_P` is the deliberately loose
analytic comparison constant of `bounds.py` evaluated in float64; the potential is a
finite-basis approximation of the population projection; and every expectation is a finite
sample. A bound computed this way certifies nothing about the computed output, and an apparent
crossing would be a diagnostic to investigate rather than a theorem violation.

With `RHS = C_P / sigma_min(L)^2 * sum_l J_Delta(u^l)` and `LHS` the oracle-aligned MSE, over
the injection-dominated points of the reference route:

| quantity | value |
|---|---|
| points compared | 202 |
| `LHS / RHS` | `1.71e-06` to `7.17e-04` |
| points with `LHS > RHS` | 0 |

The inequality holds at every point, with 3 to 6 orders of slack. That slack measures the looseness of the constant, not the tightness of the bound.

## The same measurement at n = 3

Configuration `8dc577d359c32f4d`, 10 seeds, everything unchanged but the
dimension: `n = 3` gives `q = 6` and 7 conditions against
`n = 2`'s `q = 4` and 5. The manuscript's statements do not
depend on the dimension, so this is a check that the implemented chain does not either. It is
a replication, not a second main result, and the sections above report `n = 2` alone.

| quantity | `n = 2` | `n = 3` |
|---|---|---|
| `artanh` baseline oracle-aligned MSE | 6.011e-06 | 1.911e-04 |
| `artanh` baseline seed spread | 21.3x | 20.6x |
| `mixing` baseline oracle-aligned MSE | 1.283e-05 | 7.393e-05 |
| `mixing` baseline seed spread | 2.2x | 2.0x |
| aligned-error slope in the amplitude | 1.9982 +- 0.0030 | 1.9979 +- 0.0041 |
| conditioning: max deviation of aligned from `raw / s_j^2` | 1.5e-14 | 8.5e-15 |
| conditioning: raw MSE across modes, max/min | 1.0000000000 | 1.0000000000 |
| conditioning: amplification, weakest over strongest mode | 5 to 8695 | 114 to 11059203 |
| two routes: max relative gap | 1.6e-08 | 9.3e-08 |
| `artanh` affine NMSE at the top amplitude | 0.2449 | 0.2223 |
| `mixing` affine NMSE at the top amplitude | 0.2302 | 0.1768 |

The structural results carry over unchanged: the conditioning identity holds to the same
float64 tolerance, the raw error stays matched across modes to ten decimals, the two routes
agree to the same order, and the slope is the same to within its seed spread. The dominance
crossing sits at the same amplitude in 2 of the 4 case/kind combinations.

What differs is level, not behaviour. A larger `q` means more outputs from one basis and more
conditions in the marginal, so the baselines move; they are reported side by side above rather
than reconciled, because the two dimensions are different measurement problems and nothing
here requires them to agree. The conclusions the `n = 2` sections draw --- that the
numerical floor sits in the solver, that the affine metric saturates while the aligned error
does not, that the amplification follows the singular values --- hold at `n = 3` too.

## Limits

Supported, as implementation evidence: the readout chain runs end to end from conditional
scores to statistics recovered up to an affine map; the two routes are consistent; the
finite-basis solver's contribution is measured rather than assumed; an injected field error
propagates with no extra amplitude dependence; the alignment factor appears exactly as
constructed at a fixed model; and the affine metric is shown not to constrain the
factorization.

Not supported: any claim about the rate, sharpness or constants of `eq:quantitative` or
`eq:quantitative-pi`; any claim about nonatomic conditioning, which this finite categorical
design does not exercise; any claim about the tractability or success of a factorization
search, which is not implemented; and any claim that one readout route is better than the
other, the two being exactly equivalent after alignment by construction.

Synthetic data throughout. No score model is trained here, so nothing is said about what a
trained conditional diffusion model's error looks like; the injected error is smooth, of
controlled size, and of a shape chosen by the experimenter.

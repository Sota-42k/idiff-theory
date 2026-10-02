# Which estimator reaches the construction

Configuration `52ed7c149f7cd299`; 10/10 completed seeds; n=2, `mixing`.


## 1. From an estimate to the readout

Models use observed (x,u) for training. The independent calibration split fixes the affine evaluation map; the test split measures its error. Ground truth is used for evaluation only. Direct log-odds estimate the readout; the potential route integrates estimated contrast fields.

Tables report mean ± sample SD across seeds, with the contributing seed count. Each seed changes both the generated model and samples; SD is not a confidence interval. `1 - affine NMSE` is variance explained after affine calibration and does not establish component recovery.

The analytic-score baseline measures the numerical potential approximation, not a required accuracy threshold for a learned estimator: 1.703e-05 ± 5.17e-06 (n=10).

| estimator | width | updates | route | affine NMSE | 1 - NMSE |
|---|---|---|---|---|---|
| logodds | 64 | 40000 | direct | 0.08813 ± 0.05014 (n=10) | 0.9119 ± 0.05014 (n=10) |
| logodds | 64 | 40000 | through_the_potential | 0.08813 ± 0.05014 (n=10) | 0.9119 ± 0.05014 (n=10) |
| dsm | 64 | 40000 | through_the_potential | 0.3292 ± 0.1278 (n=10) | 0.6708 ± 0.1278 (n=10) |
| logodds | 512 | 40000 | direct | 0.1659 ± 0.06112 (n=10) | 0.8341 ± 0.06112 (n=10) |
| logodds | 512 | 40000 | through_the_potential | 0.1655 ± 0.06129 (n=10) | 0.8345 ± 0.06129 (n=10) |
| dsm | 512 | 40000 | through_the_potential | 0.4833 ± 0.2657 (n=10) | 0.5167 ± 0.2657 (n=10) |

Agreement of direct and potential metrics is a numerical consistency diagnostic. It does not prove that the potential adds zero error.

## 2. Recovery along DSM training

| width | updates | 1 - NMSE |
|---|---|---|
| 64 | 5000 | 0.8329 ± 0.08988 (n=10) |
| 64 | 10000 | 0.7622 ± 0.0938 (n=10) |
| 64 | 20000 | 0.6861 ± 0.07389 (n=10) |
| 64 | 40000 | 0.6708 ± 0.1278 (n=10) |
| 512 | 5000 | 0.8301 ± 0.05311 (n=10) |
| 512 | 10000 | 0.7741 ± 0.09626 (n=10) |
| 512 | 20000 | 0.5505 ± 0.2623 (n=10) |
| 512 | 40000 | 0.5167 ± 0.2657 (n=10) |

These are measurements at the listed iterates, not a guarantee of monotonicity or convergence. The raw denoising objective is not the score-error metric. A lack of improvement over measured budgets does not exclude optimization effects or establish that further training could never help.

## 3. Recovery along classifier training

| width | updates | 1 - NMSE |
|---|---|---|
| 64 | 5000 | 0.9598 ± 0.02281 (n=10) |
| 64 | 10000 | 0.9583 ± 0.02152 (n=10) |
| 64 | 20000 | 0.9362 ± 0.03587 (n=10) |
| 64 | 40000 | 0.9119 ± 0.05014 (n=10) |
| 512 | 5000 | 0.9138 ± 0.04278 (n=10) |
| 512 | 10000 | 0.9119 ± 0.04233 (n=10) |
| 512 | 20000 | 0.8869 ± 0.05302 (n=10) |
| 512 | 40000 | 0.8345 ± 0.06129 (n=10) |

Measurements at the listed iterates from one classifier training run per seed/width, kept at the same budgets as the DSM curve above and evaluated through the same potential route for direct comparison. The direct-route evaluation (no potential solver) is recorded per seed in the underlying data, not tabulated here. Not a guarantee of monotonicity or convergence.

## 4. A convex classifier reference

| features | affine NMSE | 1 - NMSE | converged |
|---|---|---|---|
| 128 | 0.1265 ± 0.05636 (n=10) | 0.8735 ± 0.05636 (n=10) | 10/10 |
| 512 | 0.1552 ± 0.06327 (n=10) | 0.8448 ± 0.06327 (n=10) | 10/10 |
| 2048 | 0.1761 ± 0.07039 (n=10) | 0.8239 ± 0.07039 (n=10) | 10/10 |

Convergence refers to this fixed-basis, regularized problem. Its performance does not exclude local optimization or capacity limitations in the neural models.

## Scope

This fixed synthetic design illustrates affine recovery and its limitations. It is not a general comparison of objectives, a validation of component recovery, or evidence for a unique failure mechanism. Controlling contrast error efficiently and separating statistical from optimization effects remain future work.

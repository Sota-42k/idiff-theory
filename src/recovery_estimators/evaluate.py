"""Put any estimate on the same axes as the controlled-error experiment.

Two routes are reported for every estimator, because they answer different questions.

`through_the_potential` runs the estimate through the construction the theorem describes:
contrast field, least-squares potential, readout. It is comparable with the controlled-error
curve point for point, and it carries that curve's numerical floor.

`direct` applies only where the estimate is of `Lambda` itself. It skips the potential solver,
so it is not on the curve, but it isolates the estimator from the solver's floor.
"""

from typing import Any

import numpy as np

from recovery_core.metrics import (affine_residual, centered_errors, fit_affine, mean_square,
                                   oracle_alignment_error, raw_readout_error, variance)
from recovery_core.readout import PotentialBasis, fit_potential
from recovery_core.recovery import Case, Split, evaluate_route, observed_field, true_field
from recovery_core.types import Array


def through_the_potential(case: Case, fitting: Split, calibration: Split, test: Split,
                          scores_fitting: Array, scores_test: Array, *, n_features: int,
                          ridge: float, basis_seed: int, chunk_size: int = 512,
                          route: str = "reference") -> dict[str, Any]:
    basis = PotentialBasis.from_data(fitting.x, n_features, basis_seed)
    potential = fit_potential(fitting.x, observed_field(case, scores_fitting, route),
                              n_features=n_features, seed=basis_seed, ridge=ridge,
                              chunk_size=chunk_size, basis=basis)
    report = evaluate_route(case, route, potential, calibration, test, scores_test)
    return {"route_kind": "through_the_potential", **report}


def direct(case: Case, calibration: Split, test: Split, readout_calibration: Array,
           readout_test: Array, route: str = "reference") -> dict[str, Any]:
    """Evaluate an estimate of Lambda without fitting a potential."""
    G = case.alignment(route)
    truth = lambda s: np.asarray(case.oracle.statistics(s.x) @ G.T, dtype=float)
    error, centering = centered_errors(readout_calibration, truth(calibration),
                                       readout_test, truth(test))
    statistics_cal = case.oracle.statistics(calibration.x)
    statistics_test = case.oracle.statistics(test.x)
    fitted = fit_affine(readout_calibration, statistics_cal)
    return {"route_kind": "direct", "route": route,
            **raw_readout_error(error),
            "oracle_aligned_mse": oracle_alignment_error(error, G),
            **affine_residual(fitted, readout_test, statistics_test,
                              denominator=variance(statistics_test)),
            "evaluation_split": test.name, "calibration_split": calibration.name,
            "solver_used": False,
            **{f"centering_{k.removeprefix('centering_')}": v for k, v in centering.items()}}


def field_error(case: Case, x: Array, scores: Array, route: str = "reference") -> float:
    """sum_l J_Delta(u^l): the quantity the bound is stated on."""
    return mean_square(observed_field(case, scores, route)-true_field(case, x, route))


def error_decomposition(case: Case, x: Array, scores: Array) -> dict[str, float]:
    """Split the score error into the part the readout cancels and the part it keeps.

    The readout only sees differences, so a condition-independent error contributes nothing.
    Reporting both says how much of an objective's remaining error is spent where the
    construction cannot be hurt by it.
    """
    error = scores-case.oracle.scores(x)
    common = error.mean(axis=1, keepdims=True)
    return {"common_score_mse": mean_square(np.broadcast_to(common, error.shape).copy()),
            "centered_score_mse": mean_square(error-common)}

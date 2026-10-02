"""the recovery-map experiments pipeline: splits, both readout routes, and the evaluation chain.

Implements what `docs/experiment-design.md` specifies and nothing more. No configuration is
frozen here, no path under `results/` is written, and no split is chosen for the caller:
sizes, amplitudes, basis settings and which splits play the calibration and evaluation
roles are all supplied. A production run additionally requires the freeze of design
Section 8, which this module neither performs nor assumes.

Oracle quantities (`L`, `kappa`, `lambda`, `Upsilon`, `DUpsilon`, the analytic clean
scores) enter only where the design allows: constructing controlled fields, evaluating
error, and selecting directions in alignment-conditioning. They never reach a fitting routine, which sees
`(x, u)` and the observable feature alone.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from recovery_core.generator import (Mixing, Oracle, Params, draw_parameters,
                                     mixing, sample_dataset, stream_seed)
from recovery_core.metrics import (affine_residual, centered_errors, fit_affine, mean_square,
                                   oracle_alignment_error, raw_readout_error,
                                   stratified_mean_se, variance)
from recovery_core.perturbation import Perturbation
from recovery_core.readout import (Features, Potential, PotentialBasis, fit_potential,
                                   reference_field, regression_field)
from recovery_core.types import Array, IntArray

ROUTES = ("reference", "regression")

# Design Section 6. Distinct per split so that no two splits can silently coincide.
STREAMS = {"parameters": 0, "mixing": 1, "fitting": 3, "validation": 4,
           "pert_calibration": 5, "calibration": 6, "test": 7}


@dataclass(frozen=True)
class Split:
    name: str
    x: Array
    u: IntArray
    n_per_u: int
    seed: int


@dataclass(frozen=True)
class Case:
    """One generated model: parameters, mixing, oracle and the observable feature."""

    params: Params
    mix: Mixing
    oracle: Oracle
    features: Features
    seed: int
    identity: bool

    def alignment(self, route: str) -> Array:
        """The prescribed G of the design: L^T for differences, C_phi for regression."""
        if route == "reference":
            return np.asarray(self.params.L.T, dtype=float)
        if route == "regression":
            return self.features.oracle_alignment(self.params.lambdas)
        raise ValueError(f"route must be one of {ROUTES}")

    def split(self, name: str, n_per_u: int) -> Split:
        if name not in STREAMS or name in ("parameters", "mixing"):
            raise ValueError(f"Unknown data split {name!r}")
        seed = stream_seed(self.seed, STREAMS[name])
        data = sample_dataset(self.params, self.mix, n_per_u, seed)
        return Split(name, data.observed.x, data.observed.u, n_per_u, seed)


def stratified_sample(split: Split, count: int) -> Array:
    """An equal share of `count` points from each condition.

    `sample_dataset` stores the conditions one after another, so a head slice of a split is
    one condition's data and not a sample from the marginal `p_0`. Anything compared against
    a `p_0` expectation has to be evaluated on a marginal sample; this builds one.
    """
    if count < 1:
        raise ValueError("count must be positive")
    labels = np.unique(split.u)
    per = max(1, count//len(labels))
    index = np.concatenate([np.flatnonzero(split.u == u)[:per] for u in labels])
    return np.asarray(split.x[index], dtype=float)


def build_case(n: int, seed: int, separation: float, *, identity: bool) -> Case:
    params = draw_parameters(n, stream_seed(seed, STREAMS["parameters"]), separation)
    mix = mixing(n, stream_seed(seed, STREAMS["mixing"]), identity=identity)
    return Case(params, mix, Oracle(params, mix), Features.categorical(params.q), seed, identity)


def perturbed_scores(case: Case, x: Array, perturbation: Perturbation | None,
                     amplitude: float) -> Array:
    """Analytic clean conditional scores, plus the saved perturbation at this amplitude."""
    scores = case.oracle.scores(x)
    if perturbation is None or amplitude == 0:
        return scores
    if perturbation.n_conditions != scores.shape[1]:
        raise ValueError("Perturbation condition count does not match the model")
    return scores + perturbation(x, amplitude)


def observed_field(case: Case, scores: Array, route: str) -> Array:
    """The field the construction consumes; uses no oracle quantity."""
    if route == "reference":
        return reference_field(scores)
    if route == "regression":
        return regression_field(scores, case.features)
    raise ValueError(f"route must be one of {ROUTES}")


def true_field(case: Case, x: Array, route: str) -> Array:
    """G DUpsilon(x); oracle, for evaluation only."""
    return np.einsum("qr,nrd->nqd", case.alignment(route), case.oracle.statistic_jacobian(x))


def true_readout(case: Case, x: Array, route: str) -> Array:
    """G Upsilon(x); oracle. Additive constants are absorbed by centering."""
    return np.asarray(case.oracle.statistics(x) @ case.alignment(route).T, dtype=float)


def fit_route(case: Case, route: str, fitting: Split, scores: Array, *, n_features: int,
              ridge: float, basis_seed: int,
              basis: PotentialBasis | None = None) -> Potential:
    """The finite-basis surrogate for eq:ls-potential, fitted on the fitting split alone."""
    return fit_potential(fitting.x, observed_field(case, scores, route), n_features=n_features,
                         seed=basis_seed, ridge=ridge, basis=basis)


def gradient_residual(potential: Potential, x: Array, field: Array,
                      chunk_size: int = 512) -> float:
    """E ||grad Lambda_hat - field||^2, chunked: the basis gradient is (N, width, d)."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    total = sum(float(np.sum((potential.gradient(x[i:i+chunk_size])-field[i:i+chunk_size])**2))
                for i in range(0, len(x), chunk_size))
    return total/len(x)


def evaluate_route(case: Case, route: str, potential: Potential, calibration: Split,
                   evaluation: Split, scores_evaluation: Array) -> dict[str, Any]:
    """Every number here is computed on `evaluation`; `calibration` fixes A, b and centering.

    No calibration-split score is needed: the potential is already fitted, so `A`, `b` and the
    centering offset are fixed from its values on `calibration.x` alone.
    """
    G = case.alignment(route)
    field = observed_field(case, scores_evaluation, route)
    truth = true_field(case, evaluation.x, route)
    # Evaluated once each: the potential is not cheap and the splits do not change here.
    hat_cal, hat_eval = potential(calibration.x), potential(evaluation.x)

    error, centering = centered_errors(hat_cal, true_readout(case, calibration.x, route),
                                       hat_eval, true_readout(case, evaluation.x, route))
    statistics_cal = case.oracle.statistics(calibration.x)
    statistics_eval = case.oracle.statistics(evaluation.x)
    denominator = variance(statistics_eval)
    fitted = fit_affine(hat_cal, statistics_cal)
    per_sample = np.sum((statistics_eval-fitted(hat_eval))**2, axis=1)

    report: dict[str, Any] = {
        "route": route,
        "field_error_mse": mean_square(field-truth),
        "solver_gradient_residual": gradient_residual(potential, evaluation.x, field),
        **raw_readout_error(error),
        "oracle_aligned_mse": oracle_alignment_error(error, G),
        **affine_residual(fitted, hat_eval, statistics_eval,
                          denominator=denominator),
        "affine_mse_stratified_se": float(stratified_mean_se(per_sample[:, None],
                                                             evaluation.u)[0]),
        "alignment_singular_values": np.linalg.svd(G, compute_uv=False),
        "fitted_affine_rank": fitted.rank,
        "calibration_split": calibration.name, "evaluation_split": evaluation.name,
        "evaluation_samples": len(evaluation.x),
        "oracle_use": "evaluation only: Upsilon, DUpsilon and G",
        **{f"centering_{k.removeprefix('centering_')}": v for k, v in centering.items()},
    }
    report["solver_residual_fraction"] = (report["solver_gradient_residual"]
                                          / max(mean_square(truth), 1e-300))
    return report

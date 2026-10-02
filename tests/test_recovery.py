"""The the recovery-map experiments pipeline: split separation, the oracle boundary, and the two-route control."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from recovery_core.perturbation import draw_perturbation
from recovery_core.recovery import (ROUTES, STREAMS, build_case, evaluate_route, fit_route,
                                  observed_field, perturbed_scores, true_field, true_readout)
from recovery_core.readout import reference_field


def _case(identity=True):
    return build_case(2, 0, 1., identity=identity)


def test_split_streams_are_distinct_and_produce_distinct_data():
    seeds = [STREAMS[k] for k in ("fitting", "validation", "pert_calibration",
                                  "calibration", "test")]
    assert len(set(seeds)) == len(seeds)
    case = _case()
    splits = {name: case.split(name, 40) for name in
              ("fitting", "validation", "pert_calibration", "calibration", "test")}
    assert len({s.seed for s in splits.values()}) == 5
    assert not np.allclose(splits["fitting"].x, splits["test"].x)
    with pytest.raises(ValueError, match="Unknown data split"):
        case.split("mixing", 10)


def test_zero_amplitude_is_the_exact_analytic_score():
    case = _case()
    split = case.split("validation", 30)
    perturbation = draw_perturbation(split.x, case.params.q+1, kind="gradient", seed=2)
    assert_allclose(perturbed_scores(case, split.x, perturbation, 0.),
                    case.oracle.scores(split.x), atol=0)
    assert not np.allclose(perturbed_scores(case, split.x, perturbation, 1.),
                           case.oracle.scores(split.x))


def test_observed_field_consumes_scores_only():
    """The construction may not touch an oracle quantity; differencing is all it does."""
    case = _case()
    split = case.split("validation", 30)
    scores = case.oracle.scores(split.x)
    assert_allclose(observed_field(case, scores, "reference"), reference_field(scores), atol=0)
    with pytest.raises(ValueError, match="route must be"):
        observed_field(case, scores, "guess")


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("identity", [True, False])
def test_true_readout_has_the_true_field_as_its_gradient(route, identity):
    case = _case(identity)
    x = case.split("validation", 20).x
    step = 1e-5
    numerical = np.stack([(true_readout(case, x+step*np.eye(2)[i], route)
                           - true_readout(case, x-step*np.eye(2)[i], route))/(2*step)
                          for i in range(2)], axis=-1)
    assert_allclose(true_field(case, x, route), numerical, rtol=2e-6, atol=2e-6)


def test_regression_route_is_exactly_the_reference_route_after_alignment():
    """C_phi = Sigma_phi^{1/2} L^T, so the aligned errors coincide: a control, not evidence."""
    case = _case()
    fitting, calibration, evaluation = (case.split(n, 300) for n in
                                        ("fitting", "calibration", "validation"))
    perturbation = draw_perturbation(case.split("pert_calibration", 300).x,
                                     case.params.q+1, kind="field", seed=4)
    scores = {s.name: perturbed_scores(case, s.x, perturbation, .2)
              for s in (fitting, calibration, evaluation)}
    reports = {}
    for route in ROUTES:
        potential = fit_route(case, route, fitting, scores["fitting"],
                              n_features=16, ridge=1e-8, basis_seed=3)
        reports[route] = evaluate_route(case, route, potential, calibration, evaluation,
                                        scores["validation"])
    for key in ("oracle_aligned_mse", "affine_mse", "affine_nmse", "V_T"):
        assert_allclose(reports["reference"][key], reports["regression"][key], rtol=1e-7)
    # The raw errors live in different coordinates and are not interchangeable.
    assert not np.isclose(reports["reference"]["raw_readout_mse"],
                          reports["regression"]["raw_readout_mse"], rtol=1e-3)


def test_injected_error_moves_every_stage_of_the_chain():
    case = _case()
    fitting, calibration, evaluation = (case.split(n, 300) for n in
                                        ("fitting", "calibration", "validation"))
    perturbation = draw_perturbation(case.split("pert_calibration", 300).x,
                                     case.params.q+1, kind="gradient", seed=4)
    previous = None
    for amplitude in (0., .3, 3.):
        scores = {s.name: perturbed_scores(case, s.x, perturbation, amplitude)
                  for s in (fitting, calibration, evaluation)}
        potential = fit_route(case, "reference", fitting, scores["fitting"],
                              n_features=16, ridge=1e-8, basis_seed=3)
        report = evaluate_route(case, "reference", potential, calibration, evaluation,
                                scores["validation"])
        if previous is not None:
            assert report["field_error_mse"] > previous["field_error_mse"]
            assert report["raw_readout_mse"] > previous["raw_readout_mse"]
        previous = report
    assert previous is not None and previous["evaluation_split"] == "validation"
    assert previous["centering_source"] == "calibration"


def test_stratified_sample_covers_every_condition():
    """A head slice of a split is one condition; p_0 expectations need a marginal sample."""
    from recovery_core.recovery import stratified_sample
    case = _case()
    split = case.split("test", 50)
    conditions = case.params.q+1
    assert np.array_equal(split.u[:50], np.zeros(50, dtype=split.u.dtype)), \
        "the split really is stored condition by condition"
    x = stratified_sample(split, 5*conditions)
    assert len(x) == 5*conditions
    # Every condition contributes, which a head slice does not.
    matched = [int(np.any(np.all(np.isclose(split.x[split.u == u][:, None], x[None]), axis=2)))
               for u in range(conditions)]
    assert sum(matched) == conditions
    with pytest.raises(ValueError, match="count must be positive"):
        stratified_sample(split, 0)

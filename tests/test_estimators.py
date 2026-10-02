"""Both estimators run, and the log-odds identity holds on a model we can compute exactly."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from recovery_core.recovery import build_case
from recovery_estimators.schedule import VPSchedule

from recovery_estimators import dsm, logodds
from recovery_estimators.evaluate import error_decomposition, field_error
from recovery_estimators.networks import ConditionMLP, NetworkConfig


def small():
    return NetworkConfig(width=32, hidden_layers=2, embedding_dim=8)


def test_log_odds_of_the_true_conditionals_is_the_readout():
    """rem:guidance, checked against the generator rather than against a trained model."""
    case = build_case(2, 0, 1., identity=True)
    test = case.split("test", 40)
    logs = case.oracle.log_density(test.x)             # log p(x|u), every condition
    # Uniform pi, so the class prior cancels and the log-odds are the readout exactly.
    assert_allclose(logs[:, 1:]-logs[:, :1], case.oracle.readout(test.x), atol=2e-13)


def test_dsm_trains_and_produces_scores_of_the_right_shape():
    case = build_case(2, 0, 1., identity=True)
    train = case.split("fitting", 40)
    config = dsm.DSMConfig(updates=8, batch_size=16, network=small())
    model, snapshots, record = dsm.train(train, case.params.q+1, config, VPSchedule(), seed=0)
    assert snapshots == {}, "no snapshot was requested"
    s = dsm.scores(model, train.x[:20], config.t_min, VPSchedule(), config)
    assert s.shape == (20, case.params.q+1, 2)
    assert record["updates"] == 8 and len(record["final_state_sha256"]) == 64
    assert "irreducible" in record["raw_loss_caveat"]


def test_log_odds_trains_and_its_pivot_is_zero_in_the_score_array():
    case = build_case(2, 0, 1., identity=True)
    train = case.split("fitting", 40)
    config = logodds.LogOddsConfig(updates=8, batch_size=16, network=small())
    model, record = logodds.train(train, case.params.q+1, config, seed=0)
    log_pi = np.full(case.params.q+1, -np.log(case.params.q+1))
    lam = logodds.readout(model, train.x[:20], log_pi, config)
    assert lam.shape == (20, case.params.q)
    s = logodds.contrast_scores(model, train.x[:20], config)
    assert s.shape == (20, case.params.q+1, 2)
    # The pivot row is zero by construction; the readout only sees differences.
    assert_allclose(s[:, 0], 0., atol=0)
    assert "cross-entropy" in record["objective"]


def test_the_decomposition_separates_what_the_readout_cancels():
    case = build_case(2, 0, 1., identity=True)
    split = case.split("test", 40)
    exact = case.oracle.scores(split.x)
    # A purely condition-independent error: the readout cancels it, so `centered` must vanish.
    common = np.random.default_rng(0).normal(size=(len(split.x), 1, 2))
    report = error_decomposition(case, split.x, exact+common)
    assert report["centered_score_mse"] < 1e-24
    assert report["common_score_mse"] > 1e-3
    assert field_error(case, split.x, exact+common) < 1e-22
    # And a condition-dependent error survives both.
    varied = np.random.default_rng(1).normal(size=exact.shape)
    assert error_decomposition(case, split.x, exact+varied)["centered_score_mse"] > 1e-3


def test_dsm_keeps_weights_at_the_requested_budgets():
    """The budget curve has to come from one run, or it is a different experiment per point."""
    case = build_case(2, 0, 1., identity=True)
    train = case.split("fitting", 40)
    config = dsm.DSMConfig(updates=8, batch_size=16, network=small(), evaluate_at=(2, 5))
    model, snapshots, record = dsm.train(train, case.params.q+1, config, VPSchedule(), seed=0)
    assert sorted(snapshots) == [2, 5] and record["evaluated_at"] == [2, 5]
    assert not np.allclose(snapshots[2]["network.0.weight"], snapshots[5]["network.0.weight"])
    with pytest.raises(ValueError, match="evaluate_at"):
        dsm.DSMConfig(updates=8, network=small(), evaluate_at=(5, 2))
    with pytest.raises(ValueError, match="evaluate_at"):
        dsm.DSMConfig(updates=8, network=small(), evaluate_at=(99,))


def test_convex_reference_solves_and_gives_analytic_gradients():
    """Its value is that it has no optimiser to blame, so convergence must be recorded."""
    from recovery_estimators import convex
    case = build_case(2, 0, 1., identity=True)
    train = case.split("fitting", 300)
    fit = convex.fit(train, case.params.q+1, n_features=8, basis_seed=3, ridge=1e-4,
                     max_iterations=20000)
    assert fit.diagnostics["converged"], fit.diagnostics["status"]
    log_pi = np.full(case.params.q+1, -np.log(case.params.q+1))
    assert fit.readout(train.x[:20], log_pi).shape == (20, case.params.q)
    scores = fit.contrast_scores(train.x[:20])
    assert scores.shape == (20, case.params.q+1, 2)
    assert_allclose(scores[:, 0], 0., atol=0)
    # The analytic gradient must agree with central differences of the log-odds.
    step, x = 1e-5, train.x[:20]
    numerical = np.stack([(fit.readout(x+step*np.eye(2)[i], log_pi)
                           - fit.readout(x-step*np.eye(2)[i], log_pi))/(2*step)
                          for i in range(2)], axis=-1)
    assert_allclose(scores[:, 1:], numerical, rtol=1e-5, atol=1e-6)

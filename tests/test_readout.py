"""Both field constructions and the finite-basis surrogate for eq:ls-potential."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from recovery_core.generator import Oracle, draw_parameters, mixing
from recovery_core.readout import Features, fit_potential, reference_field, regression_field


@pytest.mark.parametrize("identity", [True, False])
def test_exact_fields_and_common_field_cancellation(identity):
    n = 2
    params = draw_parameters(n, 41, 1.)
    oracle = Oracle(params, mixing(n, 22, identity=identity))
    x = oracle.mix.g(np.random.default_rng(5).uniform(-.8, .8, (16, n)))
    scores = oracle.scores(x)
    features = Features.categorical(params.q)
    C_phi = features.oracle_alignment(params.lambdas)
    assert_allclose(features.pi @ features.xi, 0, atol=2e-15)
    assert_allclose(features.xi.T @ (features.pi[:, None]*features.xi), np.eye(params.q), atol=3e-15)

    assert_allclose(reference_field(scores), oracle.contrast(x), atol=2e-12)
    true_regression = np.einsum("qr,nrd->nqd", C_phi, oracle.statistic_jacobian(x))
    assert_allclose(regression_field(scores, features), true_regression, atol=2e-12)

    # A condition-independent field is removed by both constructions (Remark: guidance).
    common = np.random.default_rng(9).normal(size=(len(x), 1, n))
    assert_allclose(reference_field(scores+common), oracle.contrast(x), atol=2e-12)
    assert_allclose(regression_field(scores+common, features), true_regression, atol=2e-12)


def test_known_conservative_potential_and_nonzero_control():
    rng = np.random.default_rng(55)
    x, test = rng.normal(size=(200, 2)), rng.normal(size=(70, 2))
    gradient = lambda a: np.stack((2*a[:, 0]+3*a[:, 1], 3*a[:, 0]-a[:, 1]+2), axis=1)[:, None]
    truth = lambda a: a[:, 0]**2+3*a[:, 0]*a[:, 1]-.5*a[:, 1]**2+2*a[:, 1]
    potential = fit_potential(x, gradient(x), n_features=0, ridge=1e-12)
    assert_allclose(potential.gradient(test), gradient(test), atol=2e-10)
    assert_allclose(potential(test)[:, 0], truth(test)-truth(x).mean(), atol=2e-10)
    assert_allclose(potential(x).mean(axis=0), 0, atol=1e-14)
    # A solver that always returned zero would pass the line above and fail this one.
    assert np.mean((potential.gradient(test)-np.zeros((70, 1, 2)))**2) > 1
    assert potential.diagnostics["status"] == "solved"
    assert potential.diagnostics["training_gradient_mse"] < 1e-18


def test_nonconservative_target_leaves_a_residual():
    """The projection of eq:ls-potential discards a curl component; that is the estimator."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 2))
    rotational = np.stack((-x[:, 1], x[:, 0]), axis=1)[:, None]
    potential = fit_potential(x, rotational, n_features=0, ridge=1e-12)
    assert potential.diagnostics["training_gradient_mse"] > .5*np.mean(rotational**2)


def test_two_routes_share_one_linear_projection():
    """Expected agreement up to Sigma_phi^{1/2}: a control, not independent evidence."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(100, 2))
    scores = rng.normal(size=(100, 5, 2))
    features = Features.categorical(4)
    reference = fit_potential(x, reference_field(scores), n_features=8, seed=9)
    regression = fit_potential(x, regression_field(scores, features), n_features=8, seed=9)
    eig, vec = np.linalg.eigh(features.covariance)
    root = (vec*np.sqrt(eig)) @ vec.T
    assert_allclose(regression(x), reference(x) @ root.T, atol=1e-10)


def test_singular_feature_covariance_is_reported_not_regularized():
    with pytest.raises(ValueError, match="singular or numerically rank deficient"):
        Features.from_values(np.array([[0., 0.], [1., 1.], [2., 2.]]), np.full(3, 1/3))


def test_batched_fit_is_identical_to_fitting_each_field_alone():
    """The production sweep shares one Gram; it must not change a single number."""
    from recovery_core.readout import PotentialBasis, fit_potentials
    rng = np.random.default_rng(21)
    x = rng.normal(size=(300, 2))
    fields = {"a": rng.normal(size=(300, 4, 2)), "b": rng.normal(size=(300, 3, 2))}
    basis = PotentialBasis.from_data(x, 24, 5)
    batched = fit_potentials(x, fields, ridge=1e-8, basis=basis)
    for name, field in fields.items():
        alone = fit_potential(x, field, ridge=1e-8, basis=basis)
        assert_allclose(batched[name].coefficients, alone.coefficients, atol=0, rtol=1e-12)
        assert_allclose(batched[name](x), alone(x), atol=0, rtol=1e-12)
        assert_allclose(batched[name].gradient(x), alone.gradient(x), atol=0, rtol=1e-12)
    assert batched["a"].diagnostics["output_slice"] == [0, 4]
    assert batched["b"].diagnostics["output_slice"] == [4, 7]
    # The stacked training MSE is not any one field's and must not be served as if it were.
    assert "training_gradient_mse" not in batched["a"].diagnostics
    with pytest.raises(ValueError, match="Nothing to fit"):
        fit_potentials(x, {}, basis=basis)

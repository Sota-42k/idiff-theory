"""Generator, oracle and the exact eq:main-identity fixture. Float64; no experiment runs."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad, trapezoid

from recovery_core.generator import (Oracle, draw_parameters, generate_dataset, mixing,
                                     quadrature_diagnostics, sample_latents, stream_seed)


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("identity", [True, False])
def test_inverse_jacobian_and_logdet(n, identity):
    mix = mixing(n, 22, identity=identity)
    z = np.random.default_rng(5).uniform(-.8, .8, (12, n))
    x = mix.g(z)
    assert_allclose(mix.g_inv(x), z, atol=2e-13)
    jac = mix.jac(z)
    assert_allclose(mix.jac_inv(x) @ jac, np.broadcast_to(np.eye(n), jac.shape), atol=2e-12)
    assert_allclose(mix.logdet_jac_inv(x), -np.linalg.slogdet(jac)[1], atol=2e-12)
    step = 1e-5
    numerical = np.stack([(mix.g(z+step*np.eye(n)[i])-mix.g(z-step*np.eye(n)[i]))/(2*step)
                          for i in range(n)], axis=-1)
    assert_allclose(jac, numerical, rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize("identity", [True, False])
def test_analytic_scores_match_differences_of_log_density(identity):
    n = 2
    oracle = Oracle(draw_parameters(n, 41, 1.), mixing(n, 22, identity=identity))
    x = oracle.mix.g(np.random.default_rng(5).uniform(-.8, .8, (12, n)))
    step = 1e-5
    numerical = np.stack([(oracle.log_density(x+step*np.eye(n)[i])
                           - oracle.log_density(x-step*np.eye(n)[i]))/(2*step)
                          for i in range(n)], axis=-1)
    assert_allclose(oracle.scores(x), numerical, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("identity", [True, False])
def test_main_identity_and_its_gradient(n, identity):
    """eq:main-identity: Lambda = L^T Upsilon - kappa, and grad Lambda is the contrast field."""
    params = draw_parameters(n, 41, 1.)
    oracle = Oracle(params, mixing(n, 22, identity=identity))
    x = oracle.mix.g(np.random.default_rng(5).uniform(-.8, .8, (12, n)))
    logs = oracle.log_density(x)
    assert_allclose(logs[:, 1:]-logs[:, :1], oracle.readout(x), atol=2e-13)
    assert_allclose(oracle.readout(x), oracle.statistics(x) @ params.L - params.kappa, atol=2e-13)
    step = 1e-5
    numerical = np.stack([(oracle.readout(x+step*np.eye(n)[i])
                           - oracle.readout(x-step*np.eye(n)[i]))/(2*step)
                          for i in range(n)], axis=-1)
    assert_allclose(oracle.contrast(x), numerical, rtol=2e-6, atol=2e-6)


def test_density_normalization_and_sampler_moments():
    params = draw_parameters(2, 12, 2.)
    for i in range(params.n):
        a, b = params.a[0, i], params.b[0, i]
        normalizer = quad(lambda z: np.exp(a*z+b*z*z), -1, 1, epsabs=1e-12)[0]
        assert_allclose(params.log_z[0, i], np.log(normalizer), atol=2e-7)
    z = sample_latents(params, 0, 60000, np.random.default_rng(33))
    assert np.all(np.abs(z) < 1)
    for j in (1, 2):
        expected = trapezoid(params.density[0]*params.grid**j, params.grid, axis=-1)
        # Fixed-seed Monte Carlo control, separate from deterministic quadrature accuracy.
        assert_allclose(np.mean(z**j, axis=0), expected, atol=.008)


def test_grid_doubling_is_recorded_and_small():
    params = draw_parameters(2, 12, 2.)
    report = quadrature_diagnostics(params, seed=3, n_samples=256)
    assert report["grid_sizes"] == [4096, 8192]
    assert report["max_log_normalizer_change"] < 1e-6
    assert report["max_moment_change"] < 1e-6


def test_splits_from_distinct_streams_are_independent():
    """the recovery-map experiments relies on stream separation; equal streams would silently reuse data."""
    assert stream_seed(0, 3) != stream_seed(0, 7)
    a = generate_dataset(2, 50, 0, 1.)
    b = generate_dataset(2, 50, 1, 1.)
    assert not np.allclose(a.observed.x, b.observed.x)

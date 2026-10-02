"""The saved perturbation family: reproducibility, conservativeness and the alignment-conditioning injection."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from recovery_core.directions import singular_directions
from recovery_core.generator import Oracle, draw_parameters, mixing
from recovery_core.perturbation import (TanhFeatures, curl_energy, directional_perturbation,
                                        draw_perturbation)
from recovery_core.readout import reference_field


def _sample(n=2, seed=5, count=200):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(count, n))


def test_feature_jacobian_matches_central_differences():
    x = _sample()
    features = TanhFeatures.from_data(x, seed=3)
    step = 1e-6
    numerical = np.stack([(features.values(x+step*np.eye(2)[i])
                           - features.values(x-step*np.eye(2)[i]))/(2*step)
                          for i in range(2)], axis=-1)
    assert_allclose(features.jacobian(x), numerical, rtol=1e-6, atol=1e-8)
    # A degenerate map would make the whole study vacuous.
    assert np.mean(np.abs(features.jacobian(x))) > 1e-3


@pytest.mark.parametrize("kind", ["gradient", "field"])
def test_one_function_across_splits_and_amplitudes(kind):
    calibration = _sample(seed=1)
    perturbation = draw_perturbation(calibration, 5, kind=kind, seed=7)
    a, b = _sample(seed=2, count=30), _sample(seed=3, count=40)
    shared = np.concatenate((a, b))
    assert_allclose(perturbation(shared, .5)[:len(a)], perturbation(a, .5), atol=0)
    assert_allclose(perturbation(shared, .5)[len(a):], perturbation(b, .5), atol=0)
    # Amplitude is a scalar knob on one frozen function, not a redraw.
    assert_allclose(perturbation(a, 2.), 2*perturbation(a, 1.), atol=1e-14)
    assert_allclose(perturbation(a, 0.), 0., atol=0)
    # Rebuilding from the same seed reproduces it exactly.
    again = draw_perturbation(calibration, 5, kind=kind, seed=7)
    assert_allclose(again(a, 1.), perturbation(a, 1.), atol=0)
    assert again.normalization == perturbation.normalization


@pytest.mark.parametrize("kind", ["gradient", "field"])
def test_amplitude_calibration_is_frozen_and_realized_size_reported(kind):
    calibration = _sample(seed=1, count=4000)
    perturbation = draw_perturbation(calibration, 5, kind=kind, seed=7)
    assert_allclose(perturbation.realized_size(calibration, 1.), 1., rtol=1e-12)
    # On an independent split the realized size is close but not equal: report, do not assume.
    other = perturbation.realized_size(_sample(seed=9, count=4000), 1.)
    assert .5 < other < 2. and other != 1.
    assert perturbation.diagnostics["calibration_samples"] == 4000
    assert perturbation.diagnostics["uncalibrated_size"] > 0


def test_conservative_variant_is_exactly_a_gradient():
    x = _sample(count=60)
    perturbation = draw_perturbation(_sample(seed=1), 5, kind="gradient", seed=7)
    step = 1e-6
    numerical = np.stack([(perturbation.potential(x+step*np.eye(2)[i], .7)
                           - perturbation.potential(x-step*np.eye(2)[i], .7))/(2*step)
                          for i in range(2)], axis=-1)
    assert_allclose(perturbation(x, .7), numerical, rtol=1e-6, atol=1e-9)


def test_nonconservative_variant_has_curl_and_conservative_one_is_at_the_floor():
    x = _sample(count=300)
    conservative = draw_perturbation(_sample(seed=1), 5, kind="gradient", seed=7)
    rotational = draw_perturbation(_sample(seed=1), 5, kind="field", seed=7)
    floor = curl_energy(conservative, x, 1.)
    injected = curl_energy(rotational, x, 1.)
    assert injected["curl_energy"] > 1e4*floor["curl_energy"]
    assert floor["finite_difference_step"] == 1e-5
    with pytest.raises(ValueError, match="conservative kind"):
        rotational.potential(x, 1.)


def test_directional_injection_is_rank_one_with_matched_raw_norms():
    """alignment-conditioning: same injected field norm per mode; only the oracle-aligned error should move."""
    n = 2
    params = draw_parameters(n, 41, 1.)
    oracle = Oracle(params, mixing(n, 22, identity=True))
    x = oracle.mix.g(np.random.default_rng(5).uniform(-.8, .8, (300, n)))
    U, s, _ = singular_directions(params.L.T)
    scores = oracle.scores(x)
    sizes = []
    for j in range(params.q):
        weights = np.concatenate(([0.], U[:, j]))
        perturbation = directional_perturbation(x, weights, seed=11)
        assert perturbation.diagnostics["oracle_direction_selection"] is True
        error = reference_field(scores+perturbation(x, .3))-reference_field(scores)
        # The pivot u^0 is untouched, so the field error is exactly c[1:] tensor v(x).
        v = error[:, int(np.argmax(np.abs(U[:, j])))]/U[np.argmax(np.abs(U[:, j])), j]
        assert_allclose(error, np.einsum("l,nd->nld", U[:, j], v), atol=1e-12)
        sizes.append(np.mean(np.sum(error**2, axis=(1, 2))))
    assert_allclose(sizes, sizes[0], rtol=1e-9)
    assert s[0] > s[-1]


def test_invalid_inputs_are_rejected():
    x = _sample()
    with pytest.raises(ValueError, match="kind must be"):
        draw_perturbation(x, 5, kind="wiggle", seed=1)
    with pytest.raises(ValueError, match="amplitude"):
        draw_perturbation(x, 5, kind="gradient", seed=1)(x, -1.)
    with pytest.raises(ValueError, match="condition_weights"):
        directional_perturbation(x, np.zeros(5), seed=1)


def test_rank_one_injection_propagates_to_a_rank_one_readout_error():
    """The design's alignment-conditioning prediction, verified rather than asserted.

    A rank-one field error c tensor v gives readout error c * psi_v, because the ridge
    solve shares one Gram across output rows. Hence the raw error is matched across modes
    and only the oracle-aligned error moves, by exactly 1/s_j.
    """
    from recovery_core.metrics import oracle_alignment_error, raw_readout_error
    from recovery_core.readout import PotentialBasis, fit_potential

    n = 2
    params = draw_parameters(n, 41, 1.)
    oracle = Oracle(params, mixing(n, 22, identity=True))
    x = oracle.mix.g(np.random.default_rng(5).uniform(-.8, .8, (400, n)))
    G = params.L.T
    U, s, _ = singular_directions(G)
    basis = PotentialBasis.from_data(x, 24, 4)

    raw, aligned = [], []
    for j in range(params.q):
        perturbation = directional_perturbation(x, np.concatenate(([0.], U[:, j])), seed=11)
        error_field = reference_field(perturbation(x, .3))
        error = fit_potential(x, error_field, ridge=1e-10, basis=basis)(x)
        # Rank one in the readout index, along exactly the injected singular direction.
        assert_allclose(error, np.outer(error @ U[:, j], U[:, j]), atol=1e-10)
        raw.append(raw_readout_error(error)["raw_readout_mse"])
        aligned.append(oracle_alignment_error(error, G))

    assert_allclose(raw, raw[0], rtol=1e-8)
    assert_allclose(np.array(aligned), np.array(raw)/s**2, rtol=1e-7)
    assert aligned[-1]/aligned[0] > 1.5


def test_contrast_curl_is_linear_and_blind_to_a_gradient_perturbation():
    """rem:curl: C^(l) sees only the non-gradient part of the injected field error."""
    from recovery_core.perturbation import contrast_curl
    x = _sample(count=300)
    calibration = _sample(seed=1)
    conservative = draw_perturbation(calibration, 5, kind="gradient", seed=7)
    rotational = draw_perturbation(calibration, 5, kind="field", seed=7)

    floor = contrast_curl(conservative, x, 1.)
    injected = contrast_curl(rotational, x, 1.)
    assert floor.shape == injected.shape == (4,)
    # A conservative score error is invisible to this diagnostic, whatever its size.
    assert np.max(floor) < 1e-6*np.min(injected)
    assert np.max(contrast_curl(conservative, x, 100.)) < 1e-4*np.min(injected)
    # Exactly linear in the amplitude, so one evaluation extends to the whole grid.
    assert_allclose(contrast_curl(rotational, x, 0.25), .25*injected, rtol=1e-9)
    assert_allclose(contrast_curl(rotational, x, 0.), 0., atol=1e-12)
    with pytest.raises(ValueError, match="step must be positive"):
        contrast_curl(rotational, x, 1., step=0.)

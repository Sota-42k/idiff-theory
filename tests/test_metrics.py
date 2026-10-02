"""Calibration/test separation, nonzero controls and explicit alignment units."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from recovery_core.directions import direction_energies, singular_directions
from recovery_core.metrics import (affine_residual, centered_errors, fisher_contrast,
                                   fisher_individual, fisher_regression, fit_affine,
                                   oracle_alignment_error, raw_readout_error,
                                   stratified_mean_se, variance)


def test_affine_map_is_calibrated_once_and_errors_are_not_trivially_zero():
    rng = np.random.default_rng(4)
    calibration, test = rng.normal(size=(100, 3)), rng.normal(size=(70, 3))
    A = np.diag([2., -1., 3.])
    fit = fit_affine(calibration, calibration @ A+2)
    assert affine_residual(fit, test, test @ A+2)["affine_mse"] < 1e-25
    bad = affine_residual(fit, test, test @ A+3)
    assert_allclose(bad["affine_mse"], 3.)
    assert bad["affine_nmse"] > 0


def test_common_denominator_is_honoured():
    rng = np.random.default_rng(11)
    cal, test = rng.normal(size=(80, 2)), rng.normal(size=(60, 2))
    fit = fit_affine(cal, cal+1)
    shared = variance(test)
    reported = affine_residual(fit, test, test+1.5, denominator=shared)
    assert_allclose(reported["V_T"], shared)
    assert_allclose(reported["affine_nmse"], reported["affine_mse"]/shared)


def test_centering_comes_from_calibration_and_reports_uncertainty():
    rng = np.random.default_rng(4)
    calibration, test = rng.normal(size=(100, 3)), rng.normal(size=(70, 3))
    error, info = centered_errors(calibration+2, calibration, test+3, test)
    assert_allclose(error, np.ones_like(test))
    assert info["centering_source"] == "calibration"
    assert info["centering_samples"] == 100
    assert np.all(info["centering_standard_error"] >= 0)
    assert_allclose(raw_readout_error(error)["raw_readout_mse"], 3.)
    assert_allclose(oracle_alignment_error(error, 2*np.eye(3)), .75)


def test_oracle_alignment_amplifies_by_the_inverse_singular_value():
    """The 1/sigma_min factor alignment-conditioning isolates, on an exactly known alignment."""
    G = np.diag([4., 2., .5]) @ np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    U, s, V = singular_directions(G)
    assert_allclose(np.sort(s)[::-1], s)
    for j in range(3):
        e = .3*np.tile(U[:, j], (50, 1))
        assert_allclose(oracle_alignment_error(e, G), .09/s[j]**2, rtol=1e-12)
        # Raw error is matched across modes by construction; only the aligned error moves.
        assert_allclose(raw_readout_error(e)["raw_readout_mse"], .09, rtol=1e-12)
    energies = direction_energies(.3*np.tile(U[:, 0], (50, 1)), np.zeros((50, 3, 2)),
                                  np.zeros((50, 3)), U, s, V)
    assert_allclose(energies["oracle_aligned_mse"].mean(axis=0), [.09/s[0]**2, 0., 0.], atol=1e-12)


def test_fisher_diagnostics_use_the_right_law_and_bessel_ordering():
    field = np.zeros((10, 2, 3))
    field[:, 0] = 1.
    assert_allclose(fisher_contrast(field, np.zeros_like(field)), [3., 0.])
    scores = np.array([[[1.], [100.]], [[100.], [2.]]])
    # J(u) is under p(.|u): only the matching rows enter each entry.
    assert_allclose(fisher_individual(scores, np.zeros_like(scores), np.array([0, 1])), [1., 4.])
    pi = np.full(2, .5)
    xi = np.array([[-1.], [1.]])
    report = fisher_regression(scores, np.zeros_like(scores), pi, xi)
    assert report["regression_field_mse"] <= report["J_Delta_pi"]+1e-12


def test_stratified_uncertainty_needs_every_stratum():
    values = np.random.default_rng(1).normal(size=(20, 2))
    labels = np.repeat([0, 1], 10)
    assert np.all(stratified_mean_se(values, labels) > 0)
    with pytest.raises(ValueError, match="two samples per stratum"):
        stratified_mean_se(values[:11], np.append(np.zeros(10, int), 1))


def test_block_permutation_defect_separates_legal_from_illegal_maps():
    """`~_P` constrains the block pattern only; within-block content is free."""
    from recovery_core.metrics import block_permutation_defect
    rng = np.random.default_rng(3)
    blocks = [rng.normal(size=(2, 2)) for _ in range(2)]
    diagonal = np.block([[blocks[0], np.zeros((2, 2))], [np.zeros((2, 2)), blocks[1]]])
    assert_allclose(block_permutation_defect(diagonal, 2, 2)["defect"], 0., atol=1e-15)
    swapped = np.block([[np.zeros((2, 2)), blocks[0]], [blocks[1], np.zeros((2, 2))]])
    report = block_permutation_defect(swapped, 2, 2)
    assert_allclose(report["defect"], 0., atol=1e-15)
    assert report["matched_blocks"] == [1, 0]
    # An arbitrary within-block map is legal and must not register as a defect.
    scaled = diagonal.copy()
    scaled[:2, :2] = np.array([[0., 7.], [-3., .2]])
    assert_allclose(block_permutation_defect(scaled, 2, 2)["defect"], 0., atol=1e-15)
    # Mixing across blocks is not legal and must.
    shear = diagonal.copy()
    shear[0, 2] = 5.
    assert block_permutation_defect(shear, 2, 2)["defect"] > .3
    # A map with all its energy off the diagonal blocks, in both pattern choices.
    worst = np.ones((4, 4))
    assert_allclose(block_permutation_defect(worst, 2, 2)["defect"], .5, atol=1e-12)
    assert block_permutation_defect(np.eye(6), 3, 2)["defect"] == 0.
    with pytest.raises(ValueError, match="square of size"):
        block_permutation_defect(np.eye(5), 2, 2)

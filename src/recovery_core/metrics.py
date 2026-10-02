# Extracted from identifiability-research-1, src/idiff/metrics.py at commit
# b4cd012dc31a5305227ddd2c4f385e88fc651af3 (sha256 ac08a1a2e5c7b02e2f53e197765cec4b2bacfea570b256f2d12c4438b45cd84e).
# Modified in this repository: import path idiff -> recovery_core; disc_threshold removed, the recovery-map experiments having no
# discrete experiment.
# No upstream TCL ancestor; this is a project addition, not a TCL-derived file.
# See docs/provenance.md.
"""Held-out affine metrics and raw/oracle/Fisher diagnostics in explicit units."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from recovery_core.types import Array, IntArray


def weights_for(n: int, weights: Array | None) -> Array:
    if n < 1:
        raise ValueError("Empty evaluation set")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Invalid sample weights")
    return w/w.sum()


def mean_square(error: Array, weights: Array | None = None) -> float:
    return float(weights_for(len(error), weights) @ np.sum(error.reshape(len(error), -1)**2, axis=1))


def variance(target: Array, weights: Array | None = None) -> float:
    w = weights_for(len(target), weights)
    return mean_square(target-w @ target, w)


def stratified_mean_se(values: Array, labels: IntArray) -> Array:
    """SE of an equally weighted mean of independent conditional sample means."""
    groups = [values[labels == label] for label in np.unique(labels)]
    if any(len(group) < 2 for group in groups):
        raise ValueError("Need at least two samples per stratum")
    return np.sqrt(sum(group.var(axis=0, ddof=1)/len(group) for group in groups))/len(groups)


@dataclass
class AffineMap:
    A: Array
    b: Array
    rank: int
    singular_values: Array

    def __call__(self, readout: Array) -> Array:
        return readout @ self.A.T + self.b


def fit_affine(readout_calibration: Array, target_calibration: Array,
               weights: Array | None = None) -> AffineMap:
    if len(readout_calibration) != len(target_calibration):
        raise ValueError("Calibration lengths differ")
    root = np.sqrt(weights_for(len(readout_calibration), weights))[:, None]
    design = np.column_stack((readout_calibration, np.ones(len(readout_calibration))))
    solution, _, rank, singular = np.linalg.lstsq(root*design, root*target_calibration, rcond=None)
    return AffineMap(solution[:-1].T, solution[-1], int(rank), singular)


def affine_residual(alignment: AffineMap, readout_test: Array, target_test: Array, *,
                    weights: Array | None = None, denominator: float | None = None) -> dict[str, float]:
    """Evaluate a previously fitted map; this function cannot refit on test data."""
    denom = variance(target_test, weights) if denominator is None else denominator
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("Target variance must be positive")
    raw = mean_square(target_test-alignment(readout_test), weights)
    return {"affine_mse": raw, "affine_nmse": raw/denom, "V_T": denom}


def raw_readout_error(error: Array, weights: Array | None = None) -> dict[str, float]:
    squared = mean_square(error, weights)
    return {"raw_readout_mse": squared, "raw_readout_l2": float(np.sqrt(squared))}


def oracle_alignment_error(error: Array, G: Array, weights: Array | None = None) -> float:
    return mean_square(np.linalg.solve(G, error.T).T, weights)


def centered_errors(calibration_hat: Array, calibration_true: Array,
                    test_hat: Array, test_true: Array) -> tuple[Array, dict[str, Any]]:
    difference = calibration_hat-calibration_true
    shift = difference.mean(axis=0)
    return test_hat-test_true-shift, {
        "centering_offset": shift,
        "centering_standard_error": difference.std(axis=0, ddof=1)/np.sqrt(len(difference)),
        "centering_samples": len(difference), "centering_source": "calibration"}


def block_permutation_defect(B: Array, n: int, k: int) -> dict[str, Any]:
    """How far a q x q map is from the block-permutation group of `~_P`, q = nk.

    `~_P` allows a permutation of the n blocks of size k composed with invertible maps inside
    each block, so only the block PATTERN is constrained: the content of a diagonal block is
    free. The defect is the fraction of squared Frobenius energy sitting off the best pattern,

        defect(B) = min_pi sum_{j != pi(i)} ||B_ij||_F^2 / ||B||_F^2,

    which is 0 exactly when B is block-permutation structured and lies in [0, 1). The
    assignment runs over squared block Frobenius norms, so this is a Frobenius statement; an
    operator-norm claim would need the corresponding dimension factor.
    """
    if n < 1 or k < 1 or B.shape != (n*k, n*k):
        raise ValueError("B must be square of size n*k")
    total = float(np.sum(B*B))
    if total <= 0:
        raise ValueError("B must be nonzero")
    energy = np.array([[float(np.sum(B[i*k:(i+1)*k, j*k:(j+1)*k]**2)) for j in range(n)]
                       for i in range(n)])
    # Maximise the on-pattern energy; scipy's assignment minimises, hence the negation.
    rows, columns = linear_sum_assignment(-energy)
    kept = float(energy[rows, columns].sum())
    return {"defect": (total-kept)/total, "matched_blocks": columns.tolist(),
            "on_pattern_energy": kept, "total_energy": total, "norm": "squared Frobenius"}


def fisher_contrast(field: Array, true_field: Array) -> Array:
    return np.mean(np.sum((field-true_field)**2, axis=-1), axis=0)


def fisher_individual(scores: Array, clean_scores: Array, labels: IntArray) -> Array:
    """Stratify by u: J(u) is under p(x|u), not the clean marginal."""
    if not all(np.any(labels == u) for u in range(scores.shape[1])):
        raise ValueError("Missing conditional evaluation samples")
    return np.array([mean_square(scores[labels == u, u]-clean_scores[labels == u, u])
                     for u in range(scores.shape[1])])


def fisher_regression(scores: Array, clean_scores: Array, pi: Array, xi: Array) -> dict[str, float]:
    error = scores-clean_scores
    centered = error-np.einsum("u,nud->nd", pi, error)[:, None]
    j = float(np.mean(np.einsum("u,nud,nud->n", pi, centered, centered)))
    field_error = np.einsum("u,uq,nud->nqd", pi, xi, error)
    return {"J_Delta_pi": j, "regression_field_mse": mean_square(field_error)}

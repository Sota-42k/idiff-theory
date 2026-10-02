# Extracted from identifiability-research-1, src/idiff/directions.py at commit
# b4cd012dc31a5305227ddd2c4f385e88fc651af3 (sha256 be388bf71ea97759286b2b43fc4dbb407c66f5a250f5abe2f28a3fde4bcb8ef2).
# Modified in this repository: import path idiff -> recovery_core; the CCA diagnostic linear_recoverability removed,
# alignment-conditioning needing only the singular-direction accounting.
# No upstream TCL ancestor; this is a project addition, not a TCL-derived file.
# See docs/provenance.md.
"""Oracle direction accounting and calibration-fitted linear-recovery diagnostics."""

import numpy as np

from recovery_core.types import Array


def singular_directions(G: Array) -> tuple[Array, Array, Array]:
    """G=U diag(s) V.T, descending s; fix signs by the largest left loading."""
    U, singular, Vt = np.linalg.svd(G, full_matrices=False)
    if G.shape[0] != G.shape[1] or singular[-1] <= 1e-12*singular[0]:
        raise ValueError("Direction accounting requires a numerically invertible square alignment")
    for j in range(len(singular)):
        if U[np.argmax(np.abs(U[:, j])), j] < 0:
            U[:, j] *= -1
            Vt[j] *= -1
    return U, singular, Vt.T


def direction_energies(raw_error: Array, field_error: Array, fitted_error: Array,
                       U: Array, singular: Array, V: Array) -> dict[str, Array]:
    """Per-sample energies; field units differ from readout/statistic units."""
    raw = (raw_error @ U)**2
    field = np.einsum("qj,nqd->njd", U, field_error)
    return {"raw_readout_mse": raw, "oracle_aligned_mse": raw/singular**2,
            "field_error_mse": np.sum(field**2, axis=2),
            "fitted_affine_mse": (fitted_error @ V)**2}

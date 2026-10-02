# Extracted from identifiability-research-1, src/idiff/readout.py at commit
# b4cd012dc31a5305227ddd2c4f385e88fc651af3 (sha256 f7fa9826a4b0f9433c0289948acb234d1110f9b6352a322017e688d61cca3c19).
# Modified in this repository: import path idiff -> recovery_core; the positive-time helper rescaled_scores and its
# VP-schedule dependency removed, the recovery-map experiments evaluating clean scores at t = 0 only.
# No upstream TCL ancestor; this is a project addition, not a TCL-derived file.
# See docs/provenance.md.
"""Two observable field constructions and a fitted linear-basis potential.

The finite categorical routes are exactly related by Sigma_phi**(1/2).
A shared linear ridge projection preserves this relation; equal affine recovery
curves are therefore an expected control, not evidence of independent methods.
The field arithmetic has no learned parameters, but potential coefficients are
fitted and their cost/approximation error must be reported.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from recovery_core.types import Array


@dataclass
class Features:
    phi: Array
    pi: Array
    mean: Array
    covariance: Array
    xi: Array

    @classmethod
    def categorical(cls, q: int) -> "Features":
        return cls.from_values(np.vstack((np.zeros((1, q)), np.eye(q))), np.full(q+1, 1/(q+1)))

    @classmethod
    def from_values(cls, phi: Array, pi: Array) -> "Features":
        if len(phi) != len(pi) or np.any(pi <= 0) or not np.isclose(pi.sum(), 1):
            raise ValueError("pi must be a strictly positive probability vector")
        mean = pi @ phi
        centered = phi-mean
        covariance = centered.T @ (pi[:, None]*centered)
        eigenvalues, vectors = np.linalg.eigh(covariance)
        if eigenvalues[0] <= 1e-12*eigenvalues[-1] or eigenvalues[-1] <= 0:
            raise ValueError("Feature covariance is singular or numerically rank deficient")
        whitening = (vectors/np.sqrt(eigenvalues)) @ vectors.T
        return cls(phi, pi, mean, covariance, centered @ whitening)

    def oracle_alignment(self, lambdas: Array) -> Array:
        return self.xi.T @ (self.pi[:, None]*(lambdas-self.pi @ lambdas))


def reference_field(scores: Array) -> Array:
    """scores (N,U,d) -> field (N,U-1,d)."""
    return scores[:, 1:]-scores[:, :1]


def regression_field(scores: Array, features: Features) -> Array:
    return np.einsum("u,uq,nud->nqd", features.pi, features.xi, scores)


@dataclass
class PotentialBasis:
    """Linear/quadratic polynomials plus seeded Gaussian Fourier cosines.

    Coordinates use fitting-split mean/std only. Frequencies are normal with
    log-uniform scales in [0.3,3] in those coordinates; phases are uniform.
    No oracle parameters or labels enter this basis.
    """
    mean: Array
    scale: Array
    frequencies: Array
    phases: Array

    @classmethod
    def from_data(cls, x: Array, n_features: int, seed: int) -> "PotentialBasis":
        if n_features < 0:
            raise ValueError("n_features must be nonnegative")
        # Separate streams make capacity doubling a nested basis comparison.
        generators = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
        scales = np.exp(generators[0].uniform(np.log(.3), np.log(3), (n_features, 1)))
        return cls(x.mean(axis=0), np.maximum(x.std(axis=0), 1e-8),
                   generators[1].normal(size=(n_features, x.shape[1]))*scales,
                   generators[2].uniform(0, 2*np.pi, n_features))

    def values_gradients(self, x: Array) -> tuple[Array, Array]:
        y = (x-self.mean)/self.scale
        n, d = y.shape
        values = [y]
        gradients = [np.broadcast_to(np.diag(1/self.scale), (n, d, d))]
        for i in range(d):
            for j in range(i, d):
                values.append((y[:, i]*y[:, j])[:, None])
                grad = np.zeros((n, 1, d))
                grad[:, 0, i] += y[:, j]/self.scale[i]
                grad[:, 0, j] += y[:, i]/self.scale[j]
                gradients.append(grad)
        angle = y @ self.frequencies.T+self.phases
        normalization = np.sqrt(2/max(1, len(self.phases)))
        values.append(normalization*np.cos(angle))
        gradients.append(-normalization*np.sin(angle)[:, :, None]
                         * (self.frequencies/self.scale)[None])
        return np.concatenate(values, axis=1), np.concatenate(gradients, axis=1)


@dataclass
class Potential:
    basis: PotentialBasis
    coefficients: Array  # (basis, outputs)
    offset: Array
    diagnostics: dict[str, Any]

    def __call__(self, x: Array) -> Array:
        return np.concatenate([self.basis.values_gradients(x[i:i+1024])[0] @ self.coefficients-self.offset
                               for i in range(0, len(x), 1024)])

    def gradient(self, x: Array) -> Array:
        return np.einsum("nbd,bq->nqd", self.basis.values_gradients(x)[1], self.coefficients)

    def save_arrays(self) -> dict[str, Array]:
        return {"mean": self.basis.mean, "scale": self.basis.scale,
                "frequencies": self.basis.frequencies, "phases": self.basis.phases,
                "coefficients": self.coefficients, "offset": self.offset}


def fit_potentials(x: Array, fields: dict[str, Array], *, n_features: int = 128, seed: int = 0,
                   ridge: float = 1e-6, chunk_size: int = 512,
                   basis: PotentialBasis | None = None) -> dict[str, Potential]:
    """Fit many fields that share one basis and one design matrix in a single pass.

    The Gram matrix depends on the basis and x alone, so a production sweep over amplitudes
    and routes repeats it once per fit for no reason. Stacking the targets along the output
    axis reuses it. This is the same computation as calling `fit_potential` per field with the
    same basis and ridge, not an approximation of it; `tests/test_readout.py` pins the exact
    agreement.
    """
    if not fields:
        raise ValueError("Nothing to fit")
    widths = [f.shape[1] for f in fields.values()]
    if any(f.shape[0] != len(x) or f.shape[2] != x.shape[1] for f in fields.values()):
        raise ValueError("Every field must have shape (len(x), q, d)")
    stacked = fit_potential(x, np.concatenate(list(fields.values()), axis=1),
                            n_features=n_features, seed=seed, ridge=ridge,
                            chunk_size=chunk_size, basis=basis)
    out, start = {}, 0
    for name, width in zip(fields, widths):
        diagnostics = {**stacked.diagnostics, "stacked_fields": list(fields),
                       "stacked_outputs": sum(widths), "output_slice": [start, start+width]}
        # The stacked training MSE covers every field at once and is not this field's.
        diagnostics.pop("training_gradient_mse", None)
        diagnostics.pop("objective", None)
        out[name] = Potential(stacked.basis, stacked.coefficients[:, start:start+width],
                              stacked.offset[start:start+width], diagnostics)
        start += width
    return out


def fit_potential(x: Array, field: Array, *, n_features: int = 128, seed: int = 0,
                  ridge: float = 1e-6, chunk_size: int = 512,
                  basis: PotentialBasis | None = None) -> Potential:
    """Minimize empirical mean ||grad f-field||_F^2 + ridge ||coeff||_F^2.

    A direct float64 solve has no iterative stopping criterion. Its normal-
    equation residual and conditioning are recorded; no population optimality
    or global theorem certificate is inferred from these diagnostics.
    """
    if ridge <= 0 or chunk_size < 1 or field.shape[0] != len(x) or field.shape[2] != x.shape[1]:
        raise ValueError("Invalid ridge, chunk size or field shape")
    started = perf_counter()
    basis = basis or PotentialBasis.from_data(x, n_features, seed)
    width = basis.values_gradients(x[:1])[0].shape[1]
    gram = np.zeros((width, width))
    rhs = np.zeros((width, field.shape[1]))
    mean = np.zeros(width)
    for start in range(0, len(x), chunk_size):
        values, gradients = basis.values_gradients(x[start:start+chunk_size])
        design = gradients.transpose(0, 2, 1).reshape(-1, width)
        target = field[start:start+chunk_size].transpose(0, 2, 1).reshape(-1, field.shape[1])
        gram += design.T @ design / len(x)
        rhs += design.T @ target / len(x)
        mean += values.sum(axis=0)/len(x)
    regularized = gram + ridge*np.eye(width)
    coefficients = np.linalg.solve(regularized, rhs)
    diagnostics: dict[str, Any] = {
        "method": "linear_quadratic_fourier_ridge", "seed": seed,
        "n_fourier_features": len(basis.phases), "basis_width": width, "ridge": ridge,
        "frequency_scale_range": [.3, 3.], "dtype": "float64",
        "gram_condition": float(np.linalg.cond(regularized)),
        "unregularized_rank": int(np.linalg.matrix_rank(gram)),
        "normal_equation_residual": float(np.linalg.norm(regularized @ coefficients-rhs)),
        "status": "solved", "normalization": "zero fitting-split mean",
    }
    potential = Potential(basis, coefficients, mean @ coefficients, diagnostics)
    squared = sum(float(np.sum((potential.gradient(x[i:i+chunk_size])-field[i:i+chunk_size])**2))
                  for i in range(0, len(x), chunk_size))/len(x)
    diagnostics["training_gradient_mse"] = squared
    diagnostics["objective"] = squared+ridge*float(np.sum(coefficients**2))
    diagnostics["fit_seconds"] = perf_counter()-started
    return potential

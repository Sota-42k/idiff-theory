"""A convex reference for the classifier route. No upstream ancestor.

Multinomial logistic regression on a fixed feature basis. The objective is convex, so there
is no optimisation pathology to confound the answer, and the basis supplies analytic values
*and* gradients, so both `Lambda_hat` and `grad Lambda_hat` come out in closed form with no
autograd and no numerical differentiation.

This is **not** a ceiling on what any estimator could achieve; it is a well-conditioned
reference with the pathologies removed. Its purpose is to separate "the objective or the
optimiser is the obstacle" from "this is what the data supports".
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax

from recovery_core.generator import Observations
from recovery_core.readout import PotentialBasis
from recovery_core.types import Array, IntArray


@dataclass
class ConvexLogOdds:
    basis: PotentialBasis
    coefficients: Array       # (width, K)
    ridge: float
    diagnostics: dict[str, Any]

    def log_posterior(self, x: Array, chunk: int = 4096) -> Array:
        out = []
        for start in range(0, len(x), chunk):
            values, _ = self.basis.values_gradients(x[start:start+chunk])
            out.append(log_softmax(values @ self.coefficients, axis=1))
        return np.concatenate(out)

    def readout(self, x: Array, log_pi: Array) -> Array:
        """Lambda_hat(x), shape (N, K-1). Uniform pi leaves the prior term at zero."""
        logp = self.log_posterior(x)
        return (logp[:, 1:]-logp[:, :1])-(log_pi[1:]-log_pi[0])

    def contrast_scores(self, x: Array, chunk: int = 1024) -> Array:
        """grad_x of the log-odds as a score array (N, K, d), pivot row zero.

        Closed form: d/dx log_softmax(B(x) c)_k = (c_k - sum_j p_j c_j)^T dB/dx.
        """
        out = []
        for start in range(0, len(x), chunk):
            values, gradients = self.basis.values_gradients(x[start:start+chunk])
            p = np.exp(log_softmax(values @ self.coefficients, axis=1))       # (n, K)
            effective = self.coefficients[None]-np.einsum(
                "nk,wk->nw", p, self.coefficients)[:, :, None]                # (n, W, K)
            scores = np.einsum("nwk,nwd->nkd", effective, gradients)
            out.append(scores-scores[:, :1])
        return np.concatenate(out)


def fit(train: Observations, n_conditions: int, *, n_features: int, basis_seed: int,
        ridge: float = 1e-6, max_iterations: int = 500) -> ConvexLogOdds:
    basis = PotentialBasis.from_data(train.x, n_features, basis_seed)
    values, _ = basis.values_gradients(train.x)
    width = values.shape[1]
    onehot = np.eye(n_conditions)[train.u]
    started = perf_counter()

    def objective(flat: Array) -> tuple[float, Array]:
        c = flat.reshape(width, n_conditions)
        logp = log_softmax(values @ c, axis=1)
        loss = -float(np.sum(logp*onehot))/len(values) + ridge*float(np.sum(c*c))
        grad = values.T @ (np.exp(logp)-onehot)/len(values) + 2*ridge*c
        return loss, grad.ravel()

    result = minimize(objective, np.zeros(width*n_conditions), jac=True, method="L-BFGS-B",
                      options={"maxiter": max_iterations})
    return ConvexLogOdds(basis, result.x.reshape(width, n_conditions), ridge, {
        "objective": "multinomial cross-entropy, convex, L-BFGS-B",
        "basis_width": width, "n_features": n_features, "ridge": ridge,
        "converged": bool(result.success), "status": str(result.message),
        "iterations": int(result.nit), "final_objective": float(result.fun),
        "gradient_norm": float(np.linalg.norm(result.jac)),
        "fit_seconds": perf_counter()-started})

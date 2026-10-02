"""Saved smooth score perturbations for error-propagation and alignment-conditioning. No upstream ancestor.

The potential solver of `readout.fit_potential` spans linear and quadratic monomials
plus seeded Gaussian-Fourier cosines. A perturbation drawn from that same family would
build representability into the main experiment, so the injected field is instead a
seeded two-layer tanh network: smooth, cheap, analytically differentiable and in a
different function class from the solver basis.

A perturbation is one reproducible function of (x,u), saved as explicit arrays and
reused verbatim across every split, amplitude and readout route. It is never redrawn
at evaluation time. Its `kind` records whether it is exactly a gradient field:

  "gradient" (conservative control)  delta(x,u) = grad_x <omega_u, h(x)> = Dh(x)^T omega_u
  "field"    (nonconservative)       delta(x,u) = V_u h(x), generically nonzero curl

Neither variant is claimed to be the score of a normalized density; a perturbed field
need not be one. What the experiment measures is the induced field error, which is what
J_Delta and J_Delta_pi are defined on.

Amplitude is calibrated once, on a dedicated split, so that the realized size
rho(delta)^2 = E_{p_0} E_pi ||delta||^2 equals 1 at amplitude 1. That scalar is frozen and
saved. The realized rho is recomputed on test data and reported; the nominal amplitude is
never substituted for it.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from recovery_core.types import Array


@dataclass(frozen=True)
class TanhFeatures:
    """h(x) = tanh(W2 tanh(W1 (x-mean)/scale + c1) + c2), with its analytic Jacobian."""

    mean: Array      # (d,)
    scale: Array     # (d,)
    W1: Array        # (hidden, d)
    c1: Array        # (hidden,)
    W2: Array        # (n_features, hidden)
    c2: Array        # (n_features,)

    @classmethod
    def from_data(cls, x: Array, *, seed: int, n_features: int = 32,
                  hidden: int = 32) -> "TanhFeatures":
        """Standardization uses the calibration split only; no oracle quantity enters."""
        if x.ndim != 2 or n_features < 1 or hidden < 1:
            raise ValueError("Require x of shape (N,d) and positive layer widths")
        d = x.shape[1]
        first, second = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
        # 1/sqrt(fan_in) keeps pre-activations O(1), so the tanh is neither flat nor linear.
        return cls(x.mean(axis=0), np.maximum(x.std(axis=0), 1e-8),
                   first.normal(size=(hidden, d))/np.sqrt(d), first.uniform(-1, 1, hidden),
                   second.normal(size=(n_features, hidden))/np.sqrt(hidden),
                   second.uniform(-1, 1, n_features))

    @property
    def n_features(self) -> int:
        return len(self.c2)

    def _layers(self, x: Array) -> tuple[Array, Array]:
        if x.ndim != 2 or x.shape[1] != len(self.mean):
            raise ValueError("x must have shape (N,d) matching the saved standardization")
        hidden = np.tanh((x-self.mean)/self.scale @ self.W1.T + self.c1)
        return hidden, np.tanh(hidden @ self.W2.T + self.c2)

    def values(self, x: Array) -> Array:
        """(N,d) -> (N,n_features)."""
        return self._layers(x)[1]

    def jacobian(self, x: Array) -> Array:
        """(N,d) -> (N,n_features,d), the exact Dh."""
        hidden, out = self._layers(x)
        inner = (1-hidden*hidden)[:, :, None]*(self.W1/self.scale)
        return (1-out*out)[:, :, None]*np.einsum("rp,npi->nri", self.W2, inner)

    def save_arrays(self) -> dict[str, Array]:
        return {"mean": self.mean, "scale": self.scale, "W1": self.W1, "c1": self.c1,
                "W2": self.W2, "c2": self.c2}


@dataclass
class Perturbation:
    """A frozen delta(x,u); `normalization` is calibrated once and never refitted."""

    features: TanhFeatures
    kind: str
    coefficients: Array   # "gradient": (U,n_features);  "field": (U,d,n_features)
    normalization: float
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        if self.kind not in ("gradient", "field"):
            raise ValueError("kind must be 'gradient' or 'field'")
        expected = ((len(self.coefficients), self.features.n_features) if self.kind == "gradient"
                    else (len(self.coefficients), len(self.features.mean), self.features.n_features))
        if self.coefficients.shape != expected:
            raise ValueError(f"coefficients must have shape {expected}")
        if not np.isfinite(self.normalization) or self.normalization <= 0:
            raise ValueError("normalization must be a positive finite scalar")

    @property
    def n_conditions(self) -> int:
        return len(self.coefficients)

    def __call__(self, x: Array, amplitude: float) -> Array:
        """(N,d) -> (N,U,d), the score perturbation at this amplitude."""
        if not np.isfinite(amplitude) or amplitude < 0:
            raise ValueError("amplitude must be finite and nonnegative")
        weight = amplitude*self.normalization
        if self.kind == "gradient":
            return weight*np.einsum("ur,nri->nui", self.coefficients, self.features.jacobian(x))
        return weight*np.einsum("udr,nr->nud", self.coefficients, self.features.values(x))

    def potential(self, x: Array, amplitude: float) -> Array:
        """(N,U) scalar psi with delta = grad_x psi; conservative kind only."""
        if self.kind != "gradient":
            raise ValueError("Only the conservative kind has a scalar potential")
        return amplitude*self.normalization*(self.features.values(x) @ self.coefficients.T)

    def realized_size(self, x: Array, amplitude: float) -> float:
        """rho = sqrt(E_{p_0} E_pi ||delta||^2) on the supplied sample; pi is uniform here."""
        delta = self(x, amplitude)
        return float(np.sqrt(np.mean(np.sum(delta*delta, axis=2))))

    def save_arrays(self) -> dict[str, Array]:
        return {**self.features.save_arrays(), "coefficients": self.coefficients}


def _calibrated(features: TanhFeatures, kind: str, coefficients: Array,
                x_calibration: Array, diagnostics: dict[str, Any]) -> Perturbation:
    raw = Perturbation(features, kind, coefficients, 1., diagnostics)
    size = raw.realized_size(x_calibration, 1.)
    if size <= 0 or not np.isfinite(size):
        raise ValueError("Perturbation vanished on the calibration split")
    return Perturbation(features, kind, coefficients, 1/size,
                        {**diagnostics, "calibration_samples": len(x_calibration),
                         "uncalibrated_size": size,
                         "size_convention": "rho^2 = E_{p_0} E_pi ||delta||^2, uniform pi"})


def draw_perturbation(x_calibration: Array, n_conditions: int, *, kind: str, seed: int,
                      n_features: int = 32, hidden: int = 32) -> Perturbation:
    """error-propagation: all conditions perturbed, calibrated to rho = 1 at amplitude 1."""
    if n_conditions < 2:
        raise ValueError("Require at least two conditions")
    features = TanhFeatures.from_data(x_calibration, seed=seed, n_features=n_features, hidden=hidden)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    d = x_calibration.shape[1]
    shape = (n_conditions, n_features) if kind == "gradient" else (n_conditions, d, n_features)
    return _calibrated(features, kind, rng.normal(size=shape), x_calibration,
                       {"seed": seed, "kind": kind, "n_features": n_features, "hidden": hidden,
                        "construction": "draw_perturbation"})


def directional_perturbation(x_calibration: Array, condition_weights: Array, *, seed: int,
                             n_features: int = 32, hidden: int = 32) -> Perturbation:
    """alignment-conditioning: delta(x,u) = c_u v(x) with v = grad psi_v, exactly rank one in the condition index.

    `condition_weights` is c, with c[0] = 0 for the reference-point pivot u^0 and the
    remaining entries a left singular vector of the prescribed alignment. Choosing it from
    the true L is an oracle control and must be labelled as one by the caller.
    """
    c = np.asarray(condition_weights, dtype=float)
    if c.ndim != 1 or len(c) < 2 or not np.all(np.isfinite(c)) or not np.any(c):
        raise ValueError("condition_weights must be a nonzero finite vector of length U")
    features = TanhFeatures.from_data(x_calibration, seed=seed, n_features=n_features, hidden=hidden)
    omega = np.random.default_rng(np.random.SeedSequence([seed, 1])).normal(size=n_features)
    return _calibrated(features, "gradient", np.outer(c, omega), x_calibration,
                       {"seed": seed, "kind": "gradient", "n_features": n_features,
                        "hidden": hidden, "construction": "directional_perturbation",
                        "condition_weights": c, "oracle_direction_selection": True})


def contrast_curl(perturbation: Perturbation, x: Array, amplitude: float,
                  step: float = 1e-5) -> Array:
    """The manuscript's readout diagnostic, per reference point l.

    `rem:curl` proposes C^(l)(x) = ||d_x w^(l) - (d_x w^(l))^T||_F, which vanishes when the
    score is exact, as a measure of how much estimation error has entered the readout. It is
    observable: it needs the estimated field and nothing else.

    Here w^(l) = (true contrast) + amplitude * (delta(.,u^l) - delta(.,u^0)). The true
    contrast is exactly a gradient, so only the injected difference contributes, and the
    result is exactly linear in the amplitude. Returned as the root-mean-square of C^(l)
    over the sample, one entry per l.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    d = x.shape[1]
    basis = step*np.eye(d)

    def contrast(points: Array) -> Array:
        delta = perturbation(points, amplitude)
        return delta[:, 1:]-delta[:, :1]

    jac = np.stack([(contrast(x+basis[i])-contrast(x-basis[i]))/(2*step) for i in range(d)],
                   axis=-1)
    antisymmetric = jac-jac.transpose(0, 1, 3, 2)
    return np.sqrt(np.mean(np.sum(antisymmetric**2, axis=(2, 3)), axis=0))


def curl_energy(perturbation: Perturbation, x: Array, amplitude: float,
                step: float = 1e-5) -> dict[str, Any]:
    """E ||antisymmetric part of D delta||_F^2 by central differences.

    The conservative kind returns a value at the differentiation floor, not an exact zero;
    the step and the floor are reported rather than compared against zero.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    d = x.shape[1]
    basis = step*np.eye(d)
    jac = np.stack([(perturbation(x+basis[i], amplitude)-perturbation(x-basis[i], amplitude))/(2*step)
                    for i in range(d)], axis=-1)   # (N,U,d,i) = d delta_d / d x_i
    antisymmetric = jac-jac.transpose(0, 1, 3, 2)
    return {"curl_energy": float(np.mean(np.sum(antisymmetric**2, axis=(2, 3))/2)),
            "finite_difference_step": step, "samples": len(x),
            "method": "central differences; a floor value is not an exact zero"}

# Derived from hmorioka/TCL (Apache-2.0), subfunc/generate_artificial_data.py,
# pinned revision e813b79399c6d9c5550822c740872f9df0c91871
# (sha256 c2ca46cb57ef4bf53c0d7795ad4f511547b61c2db4940bec87370a919799cbc3).
# MODIFIED with respect to that upstream file: sources truncated to (-1,1); the Laplace family
# with modulated scale (sufficient statistic |z|, k=1) replaced by the exponential family with
# T_i(z) = (z, z^2), k=2; mixing matrices changed from condition-number-rejected random uniform
# to fixed random orthogonal; leaky-tanh chi(y) = tanh(y) + 0.1 y in place of leaky-ReLU.
# These are research modifications and are not the original TCL algorithm.
#
# EXTRACTED from identifiability-research-1, src/idiff/data.py at commit
# b4cd012dc31a5305227ddd2c4f385e88fc651af3
# (sha256 9c73e9bc7538464ee5cad76bb6798efb67f3b3cbe23c4322b003659392a7ff7a).
# Modified in this repository: renamed data.py -> generator.py; import path idiff -> recovery_core.
# No numerical change; equivalence with the immediate source is recorded in docs/provenance.md.
# See docs/provenance.md and NOTICE.
"""Bounded sources and independent model/split randomness; k=2 throughout.

The identity control is g=artanh. The benchmark composes it with three
orthogonal/leaky-tanh layers. Their singular values lie in [0.1,1.1]. All
derivatives of tanh are bounded and chi' >= .1; differentiating the inverse
identity gives bounded inverse derivatives of each finite order, and hence
bounded derivatives of log det Dh. Together with the bounded derivatives of
tanh in g^{-1}, this gives bounded first/second derivatives of the clean
log-density. No numerical check here certifies smoothing-overlap hypotheses
or the geometric witness hypotheses of optional E4.

Normalizers/CDFs use composite trapezoidal quadrature. Endpoints are allowed
in quadrature only; inverse-CDF samples are clipped to the adjacent float
inside (-1,1), with counts recorded. Linear CDF interpolation has no iterative
tolerance; grid-doubling and paired sampling checks quantify its approximation.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import logsumexp

from recovery_core.types import Array, IntArray


def stream_seed(seed: int, stream: int) -> int:
    return int(np.random.SeedSequence([seed, stream]).generate_state(1)[0])


def orthogonal(n: int, rng: np.random.Generator) -> Array:
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    return np.asarray(q * np.where(np.diag(r) < 0, -1, 1), dtype=float)


@dataclass
class Params:
    natural: Array  # (U,n,2), interleaved (z_i,z_i^2)
    seed: int
    separation: float
    rejected_draws: int
    rank_cutoff: float = 1e-6
    grid_size: int = 4096

    def __post_init__(self) -> None:
        if self.grid_size < 32:
            raise ValueError("grid_size must be at least 32")
        self.grid = np.linspace(-1, 1, self.grid_size)
        exponent = (self.natural[..., 0, None] * self.grid
                    + self.natural[..., 1, None] * self.grid**2)
        shift = exponent.max(axis=-1)
        values = np.exp(exponent - shift[..., None])
        norm = trapezoid(values, self.grid, axis=-1)
        self.log_z = np.log(norm) + shift
        self.density = values / norm[..., None]
        self.cdf = cumulative_trapezoid(self.density, self.grid, axis=-1, initial=0)
        self.cdf /= self.cdf[..., -1, None]

    @property
    def n(self) -> int:
        return self.natural.shape[1]

    @property
    def q(self) -> int:
        return 2 * self.n

    @property
    def a(self) -> Array:
        return self.natural[..., 0]

    @property
    def b(self) -> Array:
        return self.natural[..., 1]

    @property
    def lambdas(self) -> Array:
        return self.natural.reshape(self.q + 1, self.q)

    @property
    def L(self) -> Array:
        return (self.lambdas[1:] - self.lambdas[0]).T

    @property
    def kappa(self) -> Array:
        normalizers = self.log_z.sum(axis=1)
        return normalizers[1:] - normalizers[0]

    def T_of_z(self, z: Array) -> Array:
        return np.stack((z, z**2), axis=-1).reshape(len(z), self.q)

    def DT(self, z: Array) -> Array:
        out = np.zeros((len(z), self.q, self.n))
        for i in range(self.n):
            out[:, 2*i, i] = 1
            out[:, 2*i+1, i] = 2*z[:, i]
        return out

    def log_latent(self, z: Array) -> Array:
        return self.T_of_z(z) @ self.lambdas.T - self.log_z.sum(axis=1)

    def metadata(self) -> dict[str, Any]:
        return {"natural": self.natural, "L": self.L, "kappa": self.kappa,
                "seed": self.seed, "separation": self.separation,
                "singular_values": np.linalg.svd(self.L, compute_uv=False),
                "rank_cutoff": self.rank_cutoff, "rejected_draws": self.rejected_draws,
                "grid_size": self.grid_size, "quadrature": "composite trapezoidal",
                "inverse_cdf": "linear interpolation", "k": 2}


def draw_parameters(n: int, seed: int, separation: float, *,
                    grid_size: int = 4096, rank_cutoff: float = 1e-6) -> Params:
    if n < 1 or separation <= .1 or not 0 < rank_cutoff < 1:
        raise ValueError("Require n>=1, separation>0.1 and 0<rank_cutoff<1")
    rng = np.random.default_rng(seed)
    for rejected in range(10000):
        natural = np.stack((rng.uniform(-separation, separation, (2*n+1, n)),
                            rng.uniform(-separation, -.1, (2*n+1, n))), axis=-1)
        flat = natural.reshape(2*n+1, 2*n)
        singular = np.linalg.svd((flat[1:]-flat[0]).T, compute_uv=False)
        if singular[-1]/singular[0] > rank_cutoff:
            return Params(natural, seed, separation, rejected, rank_cutoff, grid_size)
    raise RuntimeError("Parameter rejection budget exhausted")


def sample_latents(params: Params, u: int, n_samples: int,
                   rng: np.random.Generator, *, diagnostics: dict[str, Any] | None = None) -> Array:
    if not 0 <= u <= params.q or n_samples < 1:
        raise ValueError("Invalid condition or sample count")
    uniform = rng.random((n_samples, params.n))
    z = np.column_stack([np.interp(uniform[:, i], params.cdf[u, i], params.grid)
                         for i in range(params.n)])
    count = int(np.count_nonzero(np.abs(z) >= 1))
    if diagnostics is not None:
        diagnostics["endpoint_clips"] = diagnostics.get("endpoint_clips", 0) + count
    return np.clip(z, np.nextafter(-1., 0.), np.nextafter(1., 0.))


@dataclass
class Mixing:
    rotations: Array
    n: int
    seed: int
    inverse_steps: int = 70

    def _forward(self, z: Array) -> tuple[Array, Array, Array, Array]:
        if z.ndim != 2 or z.shape[1] != self.n or np.any(np.abs(z) >= 1):
            raise ValueError("z must have shape (N,n) and lie strictly in (-1,1)")
        value = np.arctanh(z)
        jac = np.eye(self.n)[None] / (1-z[:, None, :]**2)
        logdet = -np.log1p(-z*z).sum(axis=1)
        grad_logdet = 2*z/(1-z*z)
        for rotation in self.rotations:
            tanh = np.tanh(value)
            first = 1-tanh*tanh+.1
            second = -2*tanh*(1-tanh*tanh)
            grad_logdet += np.einsum("ni,nij->nj", second/first, jac)
            logdet += np.log(first).sum(axis=1)
            jac = np.einsum("ij,njk->nik", rotation, first[:, :, None]*jac)
            value = (tanh + .1*value) @ rotation.T
        return value, jac, logdet, grad_logdet

    def g(self, z: Array) -> Array:
        return self._forward(z)[0]

    def g_inv(self, x: Array) -> Array:
        value = np.asarray(x, dtype=float).copy()
        for rotation in self.rotations[::-1]:
            target = value @ rotation
            lo, hi = (target-1)/.1, (target+1)/.1
            for _ in range(self.inverse_steps):
                mid = (lo+hi)/2
                smaller = np.tanh(mid)+.1*mid < target
                lo, hi = np.where(smaller, mid, lo), np.where(smaller, hi, mid)
            value = (lo+hi)/2
        z = np.tanh(value)
        if np.any(np.abs(z) >= 1):
            raise FloatingPointError("Inverse saturated at a latent endpoint; increase precision")
        return z

    def jac(self, z: Array) -> Array:
        return self._forward(z)[1]

    def jac_inv(self, x: Array) -> Array:
        return np.linalg.inv(self.jac(self.g_inv(x)))

    def logdet_jac_inv(self, x: Array) -> Array:
        return -self._forward(self.g_inv(x))[2]

    def grad_logdet_jac_inv(self, x: Array) -> Array:
        _, jac, _, grad = self._forward(self.g_inv(x))
        return -np.einsum("ni,nij->nj", grad, np.linalg.inv(jac))


def mixing(n: int, seed: int, *, identity: bool = False) -> Mixing:
    rng = np.random.default_rng(seed)
    rotations = np.empty((0, n, n)) if identity else np.stack([orthogonal(n, rng) for _ in range(3)])
    return Mixing(rotations, n, seed)


@dataclass
class Observations:
    x: Array
    u: IntArray


@dataclass
class Dataset:
    observed: Observations
    z: Array  # oracle only; never passed to a fitting routine
    diagnostics: dict[str, Any]


def sample_dataset(params: Params, mix: Mixing, n_per_u: int, seed: int) -> Dataset:
    diagnostics: dict[str, Any] = {"seed": seed, "endpoint_clips": 0}
    rng = np.random.default_rng(seed)
    z = np.concatenate([sample_latents(params, u, n_per_u, rng, diagnostics=diagnostics)
                        for u in range(params.q+1)])
    u = np.repeat(np.arange(params.q+1), n_per_u)
    return Dataset(Observations(mix.g(z), u), z, diagnostics)


def generate_dataset(n: int, n_per_u: int, seed: int, separation: float) -> Dataset:
    params = draw_parameters(n, stream_seed(seed, 0), separation)
    mix = mixing(n, stream_seed(seed, 1))
    return sample_dataset(params, mix, n_per_u, stream_seed(seed, 2))


@dataclass
class Oracle:
    params: Params
    mix: Mixing

    def statistics(self, x: Array) -> Array:
        return self.params.T_of_z(self.mix.g_inv(x))

    def statistic_jacobian(self, x: Array) -> Array:
        z = self.mix.g_inv(x)
        return self.params.DT(z) @ np.linalg.inv(self.mix.jac(z))

    def log_density(self, x: Array) -> Array:
        return self.params.log_latent(self.mix.g_inv(x)) + self.mix.logdet_jac_inv(x)[:, None]

    def scores(self, x: Array) -> Array:
        z = self.mix.g_inv(x)
        latent = self.params.a[None] + 2*self.params.b[None]*z[:, None]
        return (np.einsum("nui,nij->nuj", latent, np.linalg.inv(self.mix.jac(z)))
                + self.mix.grad_logdet_jac_inv(x)[:, None])

    def readout(self, x: Array) -> Array:
        return self.statistics(x) @ self.params.L - self.params.kappa

    def contrast(self, x: Array) -> Array:
        return np.einsum("qr,nrd->nqd", self.params.L.T, self.statistic_jacobian(x))

    def overlap_samples(self, x: Array) -> Array:
        logs = self.params.log_latent(self.mix.g_inv(x))
        return np.exp(logsumexp(logs, axis=1)[:, None]-np.log(self.params.q+1)-logs)


def quadrature_diagnostics(params: Params, *, seed: int, n_samples: int = 512) -> dict[str, Any]:
    fine = Params(params.natural, params.seed, params.separation, params.rejected_draws,
                  params.rank_cutoff, 2*params.grid_size)
    moments = lambda p: np.stack([trapezoid(p.density*p.grid**j, p.grid, axis=-1)
                                  for j in (1, 2)])
    coarse_samples, fine_samples = [], []
    clips: dict[str, Any] = {}
    for u in range(params.q+1):
        coarse_samples.append(sample_latents(params, u, n_samples, np.random.default_rng(seed+u), diagnostics=clips))
        fine_samples.append(sample_latents(fine, u, n_samples, np.random.default_rng(seed+u)))
    return {"grid_sizes": [params.grid_size, fine.grid_size],
            "max_log_normalizer_change": float(np.max(np.abs(params.log_z-fine.log_z))),
            "max_moment_change": float(np.max(np.abs(moments(params)-moments(fine)))),
            "max_paired_sample_change": float(np.max(np.abs(np.array(coarse_samples)-fine_samples))),
            "paired_samples_per_u": n_samples, "endpoint_clips": clips["endpoint_clips"]}

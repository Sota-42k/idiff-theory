"""Continuous VP rates. Lives here, not in `recovery_core`, which is torch-free by design.

Derived from identifiability-research-1, `src/idiff/schedule.py`. Unmodified apart from the
import path. The recovery map is stated at `t = 0` and needs no schedule; only a denoising
objective does, so the dependency belongs to the estimator and not to the construction.
"""

from dataclasses import dataclass

import numpy as np
import torch

from recovery_core.types import Array


@dataclass(frozen=True)
class VPSchedule:
    beta_min: float = 1e-4
    beta_max: float = .02

    def __post_init__(self) -> None:
        if not 0 < self.beta_min <= self.beta_max:
            raise ValueError("Require 0 < beta_min <= beta_max")

    def coefficients(self, t: Array | float) -> tuple[Array, Array, Array]:
        t = np.asarray(t, dtype=float)
        if np.any((t < 0) | (t > 1)):
            raise ValueError("t must lie in [0,1]")
        integrated = self.beta_min*t + .5*(self.beta_max-self.beta_min)*t*t
        c = np.exp(-integrated/2)
        sigma = np.sqrt(-np.expm1(-integrated))
        return c, sigma, sigma/c

    def torch_coefficients(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        integrated = self.beta_min*t + .5*(self.beta_max-self.beta_min)*t.square()
        return torch.exp(-integrated/2), torch.sqrt(-torch.expm1(-integrated))

"""Conditional denoising score matching, reduced to what the comparison needs.

Derived from identifiability-research-1, `src/idiff/score.py`. Modified: checkpoint reuse,
validation-selected best states are removed. Training returns the final iterate and
independent copies at declared intermediate budgets. No iterate is selected on test error.

The raw objective carries an irreducible target-variance term that grows as `1/sigma_t^2`, so
its value is not a Fisher error and is not comparable across times. It is recorded for the
record and never reported as an error.
"""

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch

from recovery_core.generator import Observations
from recovery_estimators.schedule import VPSchedule
from recovery_core.types import Array

from recovery_estimators.networks import ConditionMLP, NetworkConfig, as_tensor, batches, state_hash


@dataclass(frozen=True)
class DSMConfig:
    learning_rate: float = 1e-3
    batch_size: int = 256
    updates: int = 20000
    t_min: float = 1e-3
    network: NetworkConfig = NetworkConfig()
    # Update counts at which to keep a copy of the weights, so the readout can be measured
    # against the training budget from one run. Whether more training helps is a question
    # about the objective, and it should be answerable from the record rather than asserted.
    evaluate_at: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if (self.learning_rate <= 0 or min(self.batch_size, self.updates) < 1
                or not 0 < self.t_min < 1):
            raise ValueError("Invalid DSM configuration")
        if any(not 0 < step <= self.updates for step in self.evaluate_at) or \
                list(self.evaluate_at) != sorted(set(self.evaluate_at)):
            raise ValueError("evaluate_at must be distinct, increasing and within the budget")


def train(train_set: Observations, n_conditions: int, config: DSMConfig,
          schedule: VPSchedule, seed: int
          ) -> tuple[ConditionMLP, dict[int, dict[str, Any]], dict[str, Any]]:
    """Returns the final model, weight copies at `evaluate_at`, and the training record."""
    started = perf_counter()
    d = train_set.x.shape[1]
    model = ConditionMLP(d, n_conditions, d, config.network, seed,
                         timed=True, conditioned=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    x = as_tensor(train_set.x, config.network)
    u = torch.as_tensor(train_set.u, dtype=torch.long, device=config.network.device)
    generator = torch.Generator().manual_seed(seed+1)
    history: list[dict[str, Any]] = []
    snapshots: dict[int, dict[str, Any]] = {}
    model.train()
    for step, index in enumerate(batches(len(x), config.batch_size, config.updates, seed+2)):
        xb, ub = x[index], u[index]
        t = config.t_min+(1-config.t_min)*torch.rand(len(xb), generator=generator,
                                                     dtype=xb.dtype)
        noise = torch.randn(xb.shape, generator=generator, dtype=xb.dtype)
        c, sigma = schedule.torch_coefficients(t)
        predicted = model(c[:, None]*xb+sigma[:, None]*noise, ub, t)
        loss = ((predicted+noise/sigma[:, None])**2).sum(dim=1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % max(1, config.updates//20) == 0 or step == config.updates-1:
            history.append({"update": step, "raw_dsm": float(loss.detach())})
        if step+1 in config.evaluate_at:
            snapshots[step+1] = deepcopy(model.state_dict())
    return model, snapshots, {"objective": "denoising score matching",
                   "updates": config.updates, "final_state_sha256": state_hash(model),
                   "raw_loss_history": history,
                   "evaluated_at": list(config.evaluate_at),
                   "raw_loss_caveat": "carries an irreducible target-variance term; not a "
                                      "Fisher error and not comparable across times",
                   "train_seconds": perf_counter()-started}


def scores(model: ConditionMLP, x: Array, t: float, schedule: VPSchedule,
           config: DSMConfig, batch_size: int = 4096) -> Array:
    """s_hat(x,u) = c_t * s_theta(c_t x, u, t), for every condition; shape (N, U, d).

    At positive `t` the target of this rescaling is the smoothed clean score, not the clean
    score, so a residual smoothing error remains however good the estimate is.
    """
    c, _, _ = schedule.coefficients(t)
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = as_tensor(x[start:start+batch_size], config.network)*float(c)
            tb = torch.full((len(xb),), t, dtype=xb.dtype, device=xb.device)
            per = [model(xb, torch.full((len(xb),), k, dtype=torch.long, device=xb.device), tb)
                   for k in range(model.n_conditions)]
            out.append(float(c)*torch.stack(per, dim=1).cpu().numpy())
    return np.concatenate(out).astype(float)

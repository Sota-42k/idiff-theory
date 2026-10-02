"""The readout as the log-odds of a classifier on the condition label. No upstream ancestor.

`rem:guidance` observes that, by Bayes,

    [Lambda(x)]_l = log p(u^l | x) - log p(u^0 | x) - log{ p(u^l) / p(u^0) },

so a classifier trained on the condition label estimates the readout directly. With the
uniform experimental law used here the last term vanishes.

Two properties matter for the comparison. The estimate is of `Lambda` itself, so the
least-squares potential is not needed and its numerical floor does not enter. And the
condition-independent part of the model cancels in the log-odds by construction, which is
exactly the component that dominates a denoising objective's error here.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch

from recovery_core.generator import Observations
from recovery_core.types import Array

from recovery_estimators.networks import ConditionMLP, NetworkConfig, as_tensor, batches, state_hash


@dataclass(frozen=True)
class LogOddsConfig:
    learning_rate: float = 1e-3
    batch_size: int = 256
    updates: int = 20000
    network: NetworkConfig = NetworkConfig()

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or min(self.batch_size, self.updates) < 1:
            raise ValueError("Invalid log-odds configuration")


def train(train_set: Observations, n_conditions: int, config: LogOddsConfig,
          seed: int) -> tuple[ConditionMLP, dict[str, Any]]:
    """Multiclass cross-entropy over the conditions. One network, all log-odds at once."""
    started = perf_counter()
    model = ConditionMLP(train_set.x.shape[1], n_conditions, n_conditions, config.network,
                         seed, timed=False, conditioned=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    x = as_tensor(train_set.x, config.network)
    u = torch.as_tensor(train_set.u, dtype=torch.long, device=config.network.device)
    objective = torch.nn.CrossEntropyLoss()
    history = []
    model.train()
    for step, index in enumerate(batches(len(x), config.batch_size, config.updates, seed+2)):
        loss = objective(model(x[index]), u[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % max(1, config.updates//20) == 0 or step == config.updates-1:
            history.append({"update": step, "cross_entropy": float(loss.detach())})
    return model, {"objective": "multiclass cross-entropy on the condition label",
                   "updates": config.updates, "final_state_sha256": state_hash(model),
                   "loss_history": history,
                   "loss_note": "a proper scoring rule for p(u|x); its minimiser gives the "
                                "log-odds, hence Lambda, with no irreducible variance term",
                   "train_seconds": perf_counter()-started}


def readout(model: ConditionMLP, x: Array, log_pi: Array, config: LogOddsConfig,
            batch_size: int = 4096) -> Array:
    """Lambda_hat(x) directly, shape (N, U-1). No potential solver is involved."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits = model(as_tensor(x[start:start+batch_size], config.network))
            centred = torch.log_softmax(logits, dim=1).cpu().numpy().astype(float)
            out.append(centred[:, 1:]-centred[:, :1])
    return np.concatenate(out)-(log_pi[1:]-log_pi[0])


def contrast_scores(model: ConditionMLP, x: Array, config: LogOddsConfig,
                    batch_size: int = 1024) -> Array:
    """grad_x of the log-odds, as a score array (N, U, d) with the pivot set to zero.

    The readout only ever sees differences, so putting the pivot at zero loses nothing and
    makes this comparable with a score estimate on the same pipeline.
    """
    model.eval()
    out = []
    for start in range(0, len(x), batch_size):
        xb = as_tensor(x[start:start+batch_size], config.network).requires_grad_(True)
        logp = torch.log_softmax(model(xb), dim=1)
        rows = []
        for k in range(model.n_conditions):
            grad, = torch.autograd.grad(logp[:, k].sum(), xb, retain_graph=True)
            rows.append(grad.detach())
        stacked = torch.stack(rows, dim=1)
        out.append((stacked-stacked[:, :1]).cpu().numpy().astype(float))
    return np.concatenate(out)

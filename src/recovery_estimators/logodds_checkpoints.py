"""Classifier training with intermediate-weight checkpoints.

New supplementary module for `CLASSIFIER-CHECKPOINT-PLAN.md`: the original classifier
(`logodds.py`) run never saved intermediate weights, so the only way to measure its
recovery at 5k/10k/20k updates is to retrain from the same seed and keep copies along the
way, exactly as `dsm.train` already does for the score model via `evaluate_at`. This module
does not modify `logodds.py`; it reuses `ConditionMLP`, `batches` and `as_tensor` from
`networks.py` and reproduces `logodds.train`'s loop verbatim, so the trained trajectory
(and therefore the final state hash) is unchanged by adding snapshot capture.
"""

from copy import deepcopy
from time import perf_counter
from typing import Any

import torch

from recovery_core.generator import Observations
from recovery_estimators.logodds import LogOddsConfig
from recovery_estimators.networks import ConditionMLP, as_tensor, batches, state_hash


def train_with_checkpoints(train_set: Observations, n_conditions: int, config: LogOddsConfig,
                            seed: int, evaluate_at: tuple[int, ...]
                            ) -> tuple[ConditionMLP, dict[int, dict[str, Any]], dict[str, Any]]:
    """Returns the final model, weight copies at `evaluate_at`, and the training record.

    Identical to `logodds.train` except for the snapshot capture, so a run with an empty
    `evaluate_at` reproduces `logodds.train`'s final state bit for bit.
    """
    if any(not 0 < step <= config.updates for step in evaluate_at) or \
            list(evaluate_at) != sorted(set(evaluate_at)):
        raise ValueError("evaluate_at must be distinct, increasing and within the budget")
    started = perf_counter()
    model = ConditionMLP(train_set.x.shape[1], n_conditions, n_conditions, config.network,
                         seed, timed=False, conditioned=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    x = as_tensor(train_set.x, config.network)
    u = torch.as_tensor(train_set.u, dtype=torch.long, device=config.network.device)
    objective = torch.nn.CrossEntropyLoss()
    history: list[dict[str, Any]] = []
    snapshots: dict[int, dict[str, Any]] = {}
    model.train()
    for step, index in enumerate(batches(len(x), config.batch_size, config.updates, seed+2)):
        loss = objective(model(x[index]), u[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % max(1, config.updates//20) == 0 or step == config.updates-1:
            history.append({"update": step, "cross_entropy": float(loss.detach())})
        if step+1 in evaluate_at:
            snapshots[step+1] = deepcopy(model.state_dict())
    return model, snapshots, {"objective": "multiclass cross-entropy on the condition label",
                   "updates": config.updates, "final_state_sha256": state_hash(model),
                   "loss_history": history,
                   "evaluated_at": list(evaluate_at),
                   "loss_note": "a proper scoring rule for p(u|x); its minimiser gives the "
                                "log-odds, hence Lambda, with no irreducible variance term",
                   "train_seconds": perf_counter()-started}

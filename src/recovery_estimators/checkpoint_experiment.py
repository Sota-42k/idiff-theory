"""Classifier rows at intermediate training checkpoints, on top of the frozen pipeline.

New supplementary module for `CLASSIFIER-CHECKPOINT-PLAN.md`. Mirrors
`experiment.dsm_rows`'s replay-and-verify structure for the DSM checkpoint curve, applied to
the classifier (`logodds`) estimator via `logodds_checkpoints.train_with_checkpoints`. Reuses
`experiment.Evaluation`, `logodds.contrast_scores`/`logodds.readout` and
`networks.state_hash` unmodified; does not touch `experiment.py` or `logodds.py`.
"""

from copy import deepcopy
from typing import Any

import numpy as np

from recovery_estimators import logodds
from recovery_estimators.experiment import Evaluation
from recovery_estimators.logodds_checkpoints import train_with_checkpoints
from recovery_estimators.networks import NetworkConfig, state_hash


def classifier_checkpoint_rows(evaluation: Evaluation, config: dict[str, Any], seed: int,
                                width: int, evaluate_at: tuple[int, ...],
                                expected_final_hash: str) -> list[dict[str, Any]]:
    """Retrain the classifier from `seed`, keep weights at `evaluate_at`, evaluate both routes.

    Raises if the reproduced final state does not match `expected_final_hash`, i.e. the final
    logodds/through_the_potential hash already recorded for this seed/width in the base
    (published) estimator record: only a replay that reaches the same final weights may be
    presented as the same training trajectory as the published 40k result.
    """
    network = NetworkConfig(**{**config["network"], "width": width})
    cfg = logodds.LogOddsConfig(**config["training"], network=network)
    conditions = evaluation.case.params.q+1
    model, snapshots, training = train_with_checkpoints(evaluation.observations, conditions,
                                                         cfg, seed, evaluate_at)
    final_hash = state_hash(model)
    if final_hash != expected_final_hash:
        raise ValueError("Classifier replay does not match the original final model hash")
    # Keep the final iterate before loading any earlier iterate into the model.
    snapshots[cfg.updates] = deepcopy(model.state_dict())
    log_pi = np.full(conditions, -np.log(conditions))
    rows = []
    for update in sorted(snapshots):
        model.load_state_dict(snapshots[update])
        common = {"estimator": "logodds", "width": width, "update": update,
                  "evaluated_update": update, "evaluated_state_sha256": state_hash(model)}
        rows.append({**common, "route_evaluated": "through_the_potential",
                     **evaluation.scores(
                         logodds.contrast_scores(model, evaluation.fitting.x, cfg),
                         logodds.contrast_scores(model, evaluation.test.x, cfg)),
                     **{k: v for k, v in training.items() if k != "loss_history"}})
        rows.append({**common, "route_evaluated": "direct",
                     **evaluation.readout(
                         logodds.readout(model, evaluation.calibration.x, log_pi, cfg),
                         logodds.readout(model, evaluation.test.x, log_pi, cfg))})
    return rows

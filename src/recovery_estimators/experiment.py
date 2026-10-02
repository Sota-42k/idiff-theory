"""Generated observations -> trained estimate -> readout -> held-out recovery metrics."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from recovery_core.generator import Observations
from recovery_core.recovery import Case, Split, build_case
from recovery_core.types import Array
from recovery_estimators import convex, dsm, logodds
from recovery_estimators.evaluate import direct, error_decomposition, field_error, through_the_potential
from recovery_estimators.networks import NetworkConfig, state_hash
from recovery_estimators.schedule import VPSchedule
from recovery_estimators.records import expected_final_hashes


@dataclass
class Evaluation:
    case: Case
    fitting: Split
    calibration: Split
    test: Split
    potential: dict[str, Any]

    @classmethod
    def generate(cls, config: dict[str, Any], seed: int) -> "Evaluation":
        case = build_case(config["n"], seed, config["separation"],
                          identity=config["case"] == "artanh")
        splits = [case.split(k, config["sizes"][k]) for k in ("fitting", "calibration", "test")]
        return cls(case, splits[0], splits[1], splits[2], config["potential"])

    @property
    def observations(self) -> Observations:
        return Observations(self.fitting.x, self.fitting.u)

    def scores(self, fitting: Array, test: Array) -> dict[str, Any]:
        return {"field_error": field_error(self.case, self.test.x, test),
                **error_decomposition(self.case, self.test.x, test),
                **through_the_potential(self.case, self.fitting, self.calibration, self.test,
                                        fitting, test, **self.potential)}

    def readout(self, calibration: Array, test: Array) -> dict[str, Any]:
        return direct(self.case, self.calibration, self.test, calibration, test)


def dsm_rows(evaluation: Evaluation, config: dict[str, Any], seed: int, width: int,
             *, final_only: bool = False, expected_final_hash: str | None = None
             ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    network = NetworkConfig(**{**config["network"], "width": width})
    cfg = dsm.DSMConfig(**config["training"], network=network, t_min=config["dsm"]["t_min"],
                        evaluate_at=() if final_only else tuple(config["dsm"]["evaluate_at"]))
    schedule = VPSchedule(config["dsm"]["beta_min"], config["dsm"]["beta_max"])
    model, snapshots, training = dsm.train(evaluation.observations, evaluation.case.params.q+1,
                                          cfg, schedule, seed)
    final_hash = state_hash(model)
    if expected_final_hash is not None and final_hash != expected_final_hash:
        raise ValueError("DSM replay does not match the original final model hash")
    # Keep the final iterate before loading any earlier iterate into the model.
    snapshots[cfg.updates] = deepcopy(model.state_dict())
    rows = []
    for update in sorted(snapshots):
        model.load_state_dict(snapshots[update])
        metrics = evaluation.scores(
            dsm.scores(model, evaluation.fitting.x, cfg.t_min, schedule, cfg),
            dsm.scores(model, evaluation.test.x, cfg.t_min, schedule, cfg))
        rows.append({"estimator": "dsm", "width": width, "update": update,
                     "evaluated_update": update, "evaluated_state_sha256": state_hash(model),
                     "route_evaluated": "through_the_potential", **metrics,
                     **{k: v for k, v in training.items() if k != "raw_loss_history"}})
    return rows, training["raw_loss_history"]


def classifier_rows(evaluation: Evaluation, config: dict[str, Any], seed: int,
                    width: int) -> list[dict[str, Any]]:
    network = NetworkConfig(**{**config["network"], "width": width})
    cfg = logodds.LogOddsConfig(**config["training"], network=network)
    conditions = evaluation.case.params.q+1
    model, training = logodds.train(evaluation.observations, conditions, cfg, seed)
    common = {"estimator": "logodds", "width": width, "update": cfg.updates,
              "evaluated_update": cfg.updates, "evaluated_state_sha256": state_hash(model)}
    log_pi = np.full(conditions, -np.log(conditions))
    return [
        {**common, "route_evaluated": "through_the_potential",
         **evaluation.scores(logodds.contrast_scores(model, evaluation.fitting.x, cfg),
                             logodds.contrast_scores(model, evaluation.test.x, cfg)),
         **{k: v for k, v in training.items() if k != "loss_history"}},
        {**common, "route_evaluated": "direct",
         **evaluation.readout(logodds.readout(model, evaluation.calibration.x, log_pi, cfg),
                              logodds.readout(model, evaluation.test.x, log_pi, cfg))},
    ]


def convex_rows(evaluation: Evaluation, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    conditions = evaluation.case.params.q+1
    log_pi = np.full(conditions, -np.log(conditions))
    for features in config["convex"]["n_features"]:
        fit = convex.fit(evaluation.observations, conditions, n_features=features,
                         basis_seed=config["potential"]["basis_seed"],
                         ridge=config["convex"]["ridge"],
                         max_iterations=config["convex"]["max_iterations"])
        common = {"estimator": "convex_logodds", "n_features": features, **fit.diagnostics}
        rows.extend([
            {**common, "route_evaluated": "through_the_potential",
             **evaluation.scores(fit.contrast_scores(evaluation.fitting.x),
                                 fit.contrast_scores(evaluation.test.x))},
            {**common, "route_evaluated": "direct",
             **evaluation.readout(fit.readout(evaluation.calibration.x, log_pi),
                                  fit.readout(evaluation.test.x, log_pi))},
        ])
    return rows


def run_seed(config: dict[str, Any], seed: int, *, base: dict[str, Any] | None = None
             ) -> dict[str, Any]:
    expected_hashes = expected_final_hashes(base, config) if base is not None else {}
    evaluation = Evaluation.generate(config, seed)
    record: dict[str, Any] = {"seed": seed, "case": config["case"], "rows": [],
                              "conditions": evaluation.case.params.q+1, "dsm_loss_history": {}}
    if base is None:
        record["analytic_reference"] = evaluation.scores(
            evaluation.case.oracle.scores(evaluation.fitting.x),
            evaluation.case.oracle.scores(evaluation.test.x))
    for width in config["widths"]:
        rows, history = dsm_rows(evaluation, config, seed, width, final_only=base is not None,
                                 expected_final_hash=expected_hashes.get(width))
        record["rows"].extend(rows)
        record["dsm_loss_history"][str(width)] = history
        if base is None:
            record["rows"].extend(classifier_rows(evaluation, config, seed, width))
    if base is None:
        record["rows"].extend(convex_rows(evaluation, config))
    return record

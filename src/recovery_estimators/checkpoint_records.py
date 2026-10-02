"""Validation and loading for the classifier-checkpoint supplement.

New supplementary module for `CLASSIFIER-CHECKPOINT-PLAN.md`: the classifier ("logodds")
estimator was only ever evaluated at the final training budget in the published run. This
module validates and loads the additive supplement produced by
`scripts/run_classifier_checkpoints.py`, which retrains the classifier from the same seeds
and adds rows at intermediate checkpoints. It mirrors the "dsm-final" supplement machinery in
`records.py` (imported, not modified) for a different tag, `mode == "logodds-checkpoints"`,
and never removes or edits an existing row: loading a supplement only appends checkpoint rows
to the base seed record it was replayed from.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from recovery_core.runs import environment
from recovery_estimators.records import base_records, digest, identifier, read

CHECKPOINT_SCHEMA_VERSION = 1
LOGODDS_ROUTES = ("through_the_potential", "direct")
THROUGH_THE_POTENTIAL_METRICS = ("affine_nmse", "field_error", "raw_readout_mse", "oracle_aligned_mse")
DIRECT_METRICS = ("affine_nmse", "raw_readout_mse", "oracle_aligned_mse")


def route_metrics(route: str) -> tuple[str, ...]:
    return THROUGH_THE_POTENTIAL_METRICS if route == "through_the_potential" else DIRECT_METRICS


def logodds_final_rows(record: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    final = config["training"]["updates"]
    return [r for r in record["rows"] if r.get("estimator") == "logodds" and r.get("update") == final]


def expected_final_logodds_hashes(record: dict[str, Any], config: dict[str, Any]) -> dict[int, str]:
    """The already-published final classifier state hash for every width, from a base record."""
    rows = [r for r in logodds_final_rows(record, config)
            if r.get("route_evaluated") == "through_the_potential"]
    if sorted(r.get("width", -1) for r in rows) != config["widths"]:
        raise ValueError("Base needs exactly one final logodds/through_the_potential row per width")
    hashes = {}
    for row in rows:
        value = row.get("evaluated_state_sha256")
        if (not isinstance(value, str) or len(value) != 64 or
                any(c not in "0123456789abcdef" for c in value)):
            raise ValueError("Base final classifier hash is missing or invalid")
        hashes[row["width"]] = value
    return hashes


def provenance(config: dict[str, Any]) -> dict[str, Any]:
    """Provenance for this supplementary pipeline only; distinct from `records.provenance`."""
    root = Path(__file__).resolve().parents[2]
    paths = [root/"src/recovery_estimators/logodds_checkpoints.py",
             root/"src/recovery_estimators/checkpoint_experiment.py",
             root/"src/recovery_estimators/checkpoint_records.py",
             root/"scripts/run_classifier_checkpoints.py",
             root/"src/recovery_estimators/networks.py",
             root/"src/recovery_estimators/logodds.py",
             root/"src/recovery_estimators/experiment.py",
             root/"src/recovery_estimators/evaluate.py"]
    return {"schema_version": CHECKPOINT_SCHEMA_VERSION, "config": config,
            "config_id": identifier(config),
            "source_hashes": {str(p.relative_to(root)): digest(p) for p in paths},
            "environment": {**environment(), "torch": torch.__version__,
                            "training_dtype": config["network"]["dtype"],
                            "device": config["network"]["device"],
                            "torch_threads": torch.get_num_threads(),
                            "torch_interop_threads": torch.get_num_interop_threads()}}


def validate_checkpoint_supplement(record: dict[str, Any], base: dict[str, Any],
                                    config: dict[str, Any], base_file: Path,
                                    evaluate_at: tuple[int, ...]) -> None:
    """Structural and numerical checks for one classifier-checkpoint seed record."""
    final = config["training"]["updates"]
    expected_updates = sorted(set(evaluate_at) | {final})
    expected_keys = {(w, u, route) for w in config["widths"] for u in expected_updates
                     for route in LOGODDS_ROUTES}
    if (record.get("mode") != "logodds-checkpoints" or
            record.get("schema_version") != CHECKPOINT_SCHEMA_VERSION or
            record.get("status") != "completed" or
            record.get("seed") != base.get("seed") or
            record.get("config_id") != identifier(config) or
            identifier(record.get("config", {})) != identifier(config) or
            record.get("base_seed_sha256") != digest(base_file)):
        raise ValueError("Checkpoint supplement does not match its base seed record")
    rows = record.get("rows", [])
    keys = [(r.get("width"), r.get("update"), r.get("route_evaluated")) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ValueError("Checkpoint supplement rows do not cover the expected grid exactly")
    for row in rows:
        if row.get("estimator") != "logodds" or row.get("evaluated_update") != row.get("update"):
            raise ValueError("Checkpoint row is not a matching logodds evaluation")
        state = row.get("evaluated_state_sha256")
        if (not isinstance(state, str) or len(state) != 64 or
                any(c not in "0123456789abcdef" for c in state)):
            raise ValueError("Checkpoint row is missing a valid state hash")
        for metric in route_metrics(row["route_evaluated"]):
            value = row.get(metric)
            if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                    not np.isfinite(value) or value < 0):
                raise ValueError(f"Invalid checkpoint metric: {metric}")
    by_checkpoint: dict[tuple[int, int], set[str]] = {}
    for row in rows:
        by_checkpoint.setdefault((row["width"], row["update"]), set()).add(row["evaluated_state_sha256"])
    if any(len(states) != 1 for states in by_checkpoint.values()):
        raise ValueError("Both routes at one checkpoint must evaluate identical weights")
    expected_hashes = expected_final_logodds_hashes(base, config)
    base_final = {(r["width"], r["route_evaluated"]): r for r in logodds_final_rows(base, config)}
    for row in rows:
        if row["update"] != final:
            continue
        if row["evaluated_state_sha256"] != expected_hashes.get(row["width"]):
            raise ValueError("Final classifier checkpoint does not reproduce the original training state")
        original = base_final.get((row["width"], row["route_evaluated"]))
        if original is None:
            raise ValueError("Base record is missing the matching final classifier row")
        for metric in route_metrics(row["route_evaluated"]):
            if not np.isclose(row[metric], original[metric], rtol=1e-6, atol=1e-9):
                raise ValueError(f"Final classifier metric {metric} does not match the published value")


def load_checkpoint_results(base_directory: Path, supplement_directory: Path,
                            evaluate_at: tuple[int, ...]
                            ) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Base seed records, each with classifier-checkpoint rows appended where available.

    Purely additive: a base record's existing rows are never dropped or altered, and the
    supplement's own final-update rows are not re-added (they are numerically verified to
    match the base record's published final classifier rows and then discarded).
    """
    base_manifest = read(base_directory/"manifest.json")
    config = base_manifest["config"]
    records_by_seed = base_records(base_directory, config)
    if not records_by_seed:
        raise ValueError("No completed base seeds")
    supplement_manifest = read(supplement_directory/"manifest.json")
    if (supplement_manifest.get("mode") != "logodds-checkpoints" or
            supplement_manifest.get("base_manifest_sha256") != digest(base_directory/"manifest.json") or
            supplement_manifest.get("config_id") != identifier(config)):
        raise ValueError("Checkpoint supplement manifest does not match its base")
    final = config["training"]["updates"]
    notes: list[str] = []
    missing = sorted(set(config["seeds"])-records_by_seed.keys())
    if missing:
        notes.append(f"Incomplete/failed base seeds: {missing}.")
    out = []
    incomplete_checkpoints = []
    for seed, base_record in records_by_seed.items():
        if supplement_manifest["runs"].get(str(seed), {}).get("status") != "completed":
            incomplete_checkpoints.append(seed)
            out.append(base_record)
            continue
        addition = read(supplement_directory/f"seed{seed}.json")
        validate_checkpoint_supplement(addition, base_record, config,
                                       base_directory/f"seed{seed}.json", evaluate_at)
        merged = {**base_record, "rows": [*base_record["rows"],
                  *[r for r in addition["rows"] if r["update"] != final]]}
        out.append(merged)
    if incomplete_checkpoints:
        notes.append(f"Missing classifier checkpoints for seeds: {sorted(incomplete_checkpoints)}.")
    return out, base_manifest, notes

"""Immutable seed records and explicit replacement of invalid legacy final DSM evaluations."""

from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml

from recovery_core.runs import content_hash, environment, jsonable
from recovery_estimators.networks import NetworkConfig
from recovery_estimators.dsm import DSMConfig

SCHEMA_VERSION = 2


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def identifier(config: dict[str, Any]) -> str:
    # Preserve the IDs of the original estimator records.
    return content_hash({k: v for k, v in config.items() if k not in ("name", "output")})


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("frozen") is not True:
        raise ValueError("A frozen estimator configuration is required")
    for key in ("seeds", "widths"):
        values = config.get(key)
        if (not isinstance(values, list) or not values or
                any(type(v) is not int or v < (1 if key == "widths" else 0) for v in values)
                or values != sorted(set(values))):
            raise ValueError(f"{key} must contain distinct, sorted integers")
    if config.get("n", 0) < 2 or config.get("case") not in ("mixing", "artanh"):
        raise ValueError("Require n >= 2 and case mixing or artanh")
    if any(type(config["sizes"].get(k)) is not int or config["sizes"][k] < 2
           for k in ("fitting", "calibration", "test")):
        raise ValueError("Invalid sample sizes")
    DSMConfig(**config["training"], network=NetworkConfig(**config["network"]),
              t_min=config["dsm"]["t_min"], evaluate_at=tuple(config["dsm"]["evaluate_at"]))
    return config


def provenance(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root/"src/recovery_core").glob("*.py"))
    paths += sorted((root/"src/recovery_estimators").glob("*.py"))
    paths += [root/"scripts/run_estimators.py", root/"scripts/analyse_estimators.py"]
    return {"schema_version": SCHEMA_VERSION, "config": config, "config_id": identifier(config),
            "source_hashes": {str(p.relative_to(root)): digest(p) for p in paths},
            "environment": {**environment(), "torch": torch.__version__,
                            "training_dtype": config["network"]["dtype"],
                            "device": config["network"]["device"],
                            "torch_threads": torch.get_num_threads(),
                            "torch_interop_threads": torch.get_num_interop_threads()}}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping: {path}")
    return value


def write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=1, allow_nan=False)+"\n")
    temporary.replace(path)


def base_records(directory: Path, config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    manifest = read(directory/"manifest.json")
    if manifest.get("config_id") != identifier(config) or identifier(manifest["config"]) != identifier(config):
        raise ValueError("Base configuration does not match")
    result = {}
    for seed in config["seeds"]:
        if manifest["runs"].get(str(seed), {}).get("status") != "completed":
            continue
        record = read(directory/f"seed{seed}.json")
        if (record.get("seed") != seed or record.get("status") != "completed" or
                record.get("config_id") != identifier(config) or
                identifier(record["config"]) != identifier(config)):
            raise ValueError(f"Invalid base record for seed {seed}")
        expected_final_hashes(record, config)
        result[seed] = record
    return result


def final_rows(record: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in record["rows"] if r.get("estimator") == "dsm"
            and r.get("update") == config["training"]["updates"]]


def expected_final_hashes(record: dict[str, Any], config: dict[str, Any]) -> dict[int, str]:
    rows = final_rows(record, config)
    if sorted(r.get("width", -1) for r in rows) != config["widths"]:
        raise ValueError("Base needs exactly one final DSM row per width")
    hashes = {}
    for row in rows:
        value = row.get("final_state_sha256")
        if (not isinstance(value, str) or len(value) != 64 or
                any(c not in "0123456789abcdef" for c in value)):
            raise ValueError("Base final model hash is missing or invalid")
        hashes[row["width"]] = value
    return hashes


def validate_final(record: dict[str, Any], config: dict[str, Any],
                   base: dict[str, Any] | None = None) -> None:
    rows = final_rows(record, config)
    if sorted(r["width"] for r in rows) != config["widths"]:
        raise ValueError("Need exactly one final DSM evaluation for every width")
    for row in rows:
        if (row.get("evaluated_update") != config["training"]["updates"] or
                row.get("route_evaluated") != "through_the_potential" or
                not isinstance(row.get("evaluated_state_sha256"), str) or
                len(row["evaluated_state_sha256"]) != 64 or
                row["evaluated_state_sha256"] != row.get("final_state_sha256")):
            raise ValueError("Final DSM evaluation lacks matching state/update evidence")
        for metric in ("affine_nmse", "field_error", "raw_readout_mse", "oracle_aligned_mse"):
            if not np.isfinite(row[metric]) or row[metric] < 0:
                raise ValueError(f"Invalid final metric: {metric}")
        if base is not None:
            old = [r for r in final_rows(base, config) if r["width"] == row["width"]]
            if len(old) != 1 or old[0]["final_state_sha256"] != row["evaluated_state_sha256"]:
                raise ValueError("Final DSM model does not reproduce the original training state")


def validate_supplement(record: dict[str, Any], base: dict[str, Any],
                        config: dict[str, Any], base_file: Path) -> None:
    if (record.get("mode") != "dsm-final" or record.get("schema_version") != SCHEMA_VERSION or
            record.get("status") != "completed" or record.get("seed") != base["seed"] or
            record.get("config_id") != identifier(config) or
            identifier(record["config"]) != identifier(config) or
            record.get("base_seed_sha256") != digest(base_file) or
            len(record["rows"]) != len(config["widths"])):
        raise ValueError("Supplement does not match its base seed record")
    validate_final(record, config, base)


def load_results(directory: Path, supplement: Path | None = None
                 ) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    manifest = read(directory/"manifest.json")
    config = manifest["config"]
    records = base_records(directory, config)
    if not records:
        raise ValueError("No completed base seeds")
    notes = []
    missing = sorted(set(config["seeds"])-records.keys())
    if missing:
        notes.append(f"Incomplete/failed base seeds: {missing}.")
    supplements: dict[str, Any] = {}
    if supplement is not None:
        supplements = read(supplement/"manifest.json")
        if (supplements.get("mode") != "dsm-final" or
                supplements.get("base_manifest_sha256") != digest(directory/"manifest.json") or
                supplements.get("config_id") != identifier(config)):
            raise ValueError("Supplement manifest does not match base")
        if any(supplements["runs"].get(str(s), {}).get("status") != "completed" for s in records):
            raise ValueError("Supplement incomplete: all completed base seeds must be corrected")
    out = []
    excluded = 0
    for seed, original in records.items():
        record = deepcopy(original)
        if supplement is not None:
            addition = read(supplement/f"seed{seed}.json")
            validate_supplement(addition, original, config, directory/f"seed{seed}.json")
            if addition.get("base_manifest_sha256") != supplements["base_manifest_sha256"]:
                raise ValueError("Seed and manifest base digests disagree")
            record["rows"] = [r for r in record["rows"] if r not in final_rows(record, config)]
            record["rows"].extend(addition["rows"])
        elif record.get("schema_version", 1) < SCHEMA_VERSION:
            excluded += len(final_rows(record, config))
            record["rows"] = [r for r in record["rows"] if r not in final_rows(record, config)]
        else:
            validate_final(record, config)
        out.append(record)
    if excluded:
        notes.append(f"Excluded {excluded} invalid legacy final DSM rows: the runner evaluated "
                     "the last intermediate weights under the final-update label. Final recovery "
                     "is unmeasured until the supplement is complete; no plateau can be inferred.")
    if supplement is not None:
        notes.append("Final DSM rows come from a separate supplement, linked by base-file digests "
                     "and matched against the original final-model hashes.")
    return out, manifest, notes


@contextmanager
def run_directory(directory: Path, metadata: dict[str, Any]) -> Iterator[dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory/".run.lock"
    with lock.open("x"):
        pass
    try:
        path = directory/"manifest.json"
        if path.exists():
            manifest = read(path)
            for key, value in metadata.items():
                if manifest.get(key) != value:
                    raise ValueError(f"Existing run differs in {key}; use a new output directory")
        else:
            manifest = {**metadata, "runs": {}}
        yield manifest
    finally:
        lock.unlink()

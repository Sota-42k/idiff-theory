"""The publication dataset: immutable records, explicit coverage, and provenance."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from recovery_core.runs import config_id, load_config
from recovery_estimators.checkpoint_records import load_checkpoint_results
from recovery_estimators.records import identifier, validate_final


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_keys(rows: list[dict[str, Any]], keys: tuple[str, ...], expected: set[tuple[Any, ...]],
               label: str) -> None:
    actual = [tuple(row[k] for k in keys) for row in rows]
    require(len(actual) == len(set(actual)) and set(actual) == expected,
            f"Incomplete or duplicate {label}")


def check_finite(value: Any) -> None:
    if isinstance(value, float):
        require(bool(np.isfinite(value)), "Nonfinite measurement in publication data")
    elif isinstance(value, dict):
        for item in value.values():
            check_finite(item)
    elif isinstance(value, list):
        for item in value:
            check_finite(item)


def load_bundle(root: Path, directory: Path | None = None) -> dict[str, Any]:
    """Reject changed, incomplete or legacy data before producing any publication output."""
    directory = directory or root / "data"
    index = read(directory / "manifest.json")
    files = index["files"]
    actual_files = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()
                    and p != directory / "manifest.json"}
    require(actual_files == set(files), "Publication data file set differs from manifest")
    for name, expected in files.items():
        require(digest(directory / name) == expected, f"Data digest mismatch: {name}")
    for name, expected in index["execution_source_hashes"].items():
        require(digest(root / name) == expected, f"Execution source differs: {name}")

    bundle: dict[str, Any] = {"index": index, "recovery": {}, "gaps": {}}
    expected_files: set[str] = set()
    for n in (2, 3):
        config = load_config(root / f"configs/recovery-n{n}.yaml")
        cid = config_id(config)
        require(index["config_ids"][f"n{n}"] == cid, "Frozen recovery config differs")
        prefix = f"recovery/{cid}"
        manifest = read(directory / prefix / "manifest.json")
        require(config_id(manifest["config"]) == cid, "Recovery manifest config mismatch")
        seeds = config["seeds"]
        require(set(manifest["runs"]) == set(map(str, seeds)), "Incomplete recovery manifest")
        records = []
        for seed in seeds:
            entry = manifest["runs"][str(seed)]
            require(entry["status"] == "completed" and entry["failure"] is None
                    and entry["cases_completed"] == len(config["cases"]), "Failed recovery seed")
            record = read(directory / prefix / f"seed{seed}.json")
            require(record["seed"] == seed and record["config_id"] == cid
                    and record["status"] == "completed" and record["failure"] is None,
                    "Recovery seed identity/status mismatch")
            require(record["source_hashes"] == manifest["source_hashes"], "Seed source mismatch")
            check_keys(record["cases"], ("case",), {(c,) for c in config["cases"]}, "cases")
            for case in record["cases"]:
                require(case["status"] == "completed", "Failed recovery case")
                expected = {(route, None, 0.0) for route in config["routes"]}
                expected |= {(route, kind, a) for route in config["routes"]
                             for kind in config["perturbation"]["kinds"]
                             for a in config["perturbation"]["amplitudes"] if a > 0}
                check_keys(case["propagation"], ("route", "kind", "amplitude"), expected,
                           "propagation grid")
                grid = config["potential_sensitivity"]
                check_keys(case["sensitivity"], ("n_features", "ridge"),
                           {(w, ridge) for w in grid["n_features"] for ridge in grid["ridges"]},
                           "potential sensitivity")
                check_keys(case["conditioning"], ("mode", "amplitude"),
                           {(mode, a) for mode in range(n * config["k"])
                            for a in config["directional"]["amplitudes"]}, "conditioning grid")
            check_finite(record)
            records.append(record)
        bundle["recovery"][n] = records
        expected_files |= {f"{prefix}/{name}" for name in
                           ["manifest.json", "summary.csv", *[f"seed{s}.json" for s in seeds]]}
        gap_name = f"factorization_gap/{cid}.json"
        gap = read(directory / gap_name)
        require(gap["config_id"] == cid and gap["n"] == n, "Factorization config mismatch")
        names = {v["alignment"] for v in gap["structural"]}
        require(len(names) == 6 and len(gap["structural"]) == 6, "Missing structural alignments")
        check_keys(gap["empirical"], ("seed", "case", "alignment"),
                   {(s, c, a) for s in seeds for c in config["cases"] for a in names}, "gap rows")
        check_finite(gap)
        bundle["gaps"][n] = gap
        expected_files.add(gap_name)

    import yaml
    config = yaml.safe_load((root / "configs/estimators.yaml").read_text())
    cid = identifier(config)
    require(index["config_ids"]["estimators"] == cid, "Frozen estimator config differs")
    evaluate_at = tuple(config["dsm"]["evaluate_at"])
    records, manifest, notes = load_checkpoint_results(
        directory / "estimators" / cid, directory / "estimators-checkpoints" / cid, evaluate_at)
    require(identifier(manifest["config"]) == cid, "Estimator manifest config mismatch")
    require(not notes and len(records) == len(config["seeds"]), "Incomplete or legacy estimators")
    expected = {("dsm", w, u, "through_the_potential", None) for w in config["widths"]
                for u in [*evaluate_at, config["training"]["updates"]]}
    expected |= {("logodds", w, u, route, None) for w in config["widths"]
                 for u in [*evaluate_at, config["training"]["updates"]]
                 for route in ("direct", "through_the_potential")}
    expected |= {("convex_logodds", None, None, route, w) for w in config["convex"]["n_features"]
                 for route in ("direct", "through_the_potential")}
    keys = ("estimator", "width", "update", "route_kind", "n_features")
    for record in records:
        validate_final(record, config)
        rows = [{k: row.get(k) for k in keys} for row in record["rows"]]
        check_keys(rows, keys, expected, "estimator checkpoints/routes")
        for row in record["rows"]:
            if row["estimator"] == "dsm":
                require(row["evaluated_update"] == row["update"], "Wrong checkpoint evaluated")
        check_finite(record)
    bundle["estimators"] = records
    bundle["estimator_manifest"] = manifest
    expected_files |= {f"estimators/{cid}/{name}" for name in
                       ["manifest.json", *[f"seed{s}.json" for s in config["seeds"]]]}
    expected_files |= {f"estimators-checkpoints/{cid}/{name}" for name in
                       ["manifest.json", *[f"seed{s}.json" for s in config["seeds"]]]}

    name = f"selection/{index['config_ids']['n2']}/selection.json"
    selection = read(directory / name)
    core = load_config(root / "configs/recovery-n2.yaml")
    require(selection["config_id"] == index["config_ids"]["n2"]
            and config_id(selection["config"]) == index["config_ids"]["n2"],
            "Selection config mismatch")
    require(selection["selection_seeds"] == list(range(100, 110))
            and selection["production_seeds"] == core["seeds"]
            and selection["evaluation_split"] == "validation", "Selection split/seeds differ")
    grid = core["potential_sensitivity"]
    require({tuple(cell) for cell in selection["candidate_cells"]}
            == {(w, r) for w in grid["n_features"] for r in grid["ridges"]}
            and selection["reference_only_cells"] == [[8192, 1e-8]], "Selection candidates differ")
    for case, setting in core["potential_by_case"].items():
        require(all(selection["selection"][case][k] == v for k, v in setting.items()),
                "Selection differs from frozen setting")
    cells = selection["candidate_cells"] + selection["reference_only_cells"]
    check_keys(selection["rows"], ("seed", "case", "n_features", "ridge"),
               {(s, c, w, ridge) for s in selection["selection_seeds"]
                for c in ("artanh", "mixing") for w, ridge in cells}, "selection grid")
    require(len(selection["selection_seeds"]) == 10 and len(cells) == 10, "Incomplete selection")
    check_finite(selection)
    bundle["selection"] = selection
    expected_files.add(name)
    require(set(files) == expected_files, "Unexpected dataset members")
    return bundle

"""Config validation, provenance identity, and one miniature end-to-end production run."""

import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from recovery_core.runs import config_id, load_config, provenance, validate

ROOT = Path(__file__).resolve().parents[1]


def frozen() -> dict:
    return load_config(ROOT/"configs/recovery-n2.yaml")


def miniature(tmp_path: Path) -> Path:
    """The frozen config shrunk until it runs in seconds. Structure is unchanged."""
    config = frozen()
    config.update(name="recovery-miniature", stage="pilot", frozen=False, seeds=[0],
                  sizes={k: 60 for k in config["sizes"]})
    config["potential_by_case"] = {name: {"n_features": 8, "ridge": 1.0e-6}
                                   for name in config["cases"]}
    config["potential_sensitivity"] = {"n_features": [4], "ridges": [1.0e-6]}
    config["potential"] = {**config["potential"], "chunk_size": 64}
    config["perturbation"] = {**config["perturbation"], "amplitudes": [0.0, 0.1, 1.0]}
    config["directional"] = {**config["directional"], "amplitudes": [1.0]}
    path = tmp_path/"miniature.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_the_committed_config_is_valid_and_frozen():
    config = frozen()
    assert config["stage"] == "production" and config["frozen"] is True
    assert config["seeds"] == list(range(10))
    # The solver setting is per case, and the sensitivity axis is a complete cross: the
    # earlier partial axis omitted exactly the cell that mattered.
    assert set(config["potential_by_case"]) == set(config["cases"])
    assert config["potential_by_case"] != {}
    axis = config["potential_sensitivity"]
    assert len(axis["n_features"])*len(axis["ridges"]) == 9
    assert config["perturbation"]["amplitudes"][0] == 0.0
    assert set(config["directional"]["amplitudes"]) <= set(config["perturbation"]["amplitudes"])


def test_config_id_tracks_settings_but_not_identity():
    config = frozen()
    base = config_id(config)
    assert config_id({**config, "name": "renamed", "output": "elsewhere"}) == base
    assert config_id({**config, "separation": 2.0}) != base
    changed = deepcopy(config)
    changed["potential_by_case"]["mixing"]["ridge"] = 1e-7
    assert config_id(changed) != base


def test_validation_refuses_what_it_should():
    config = frozen()
    for update, message in (
            ({"frozen": False}, "frozen: true"),
            ({"stage": "guess"}, "stage must be"),
            ({"seeds": [1, 0]}, "distinct, sorted"),
            ({"k": 1}, "k = 2"),
            ({"routes": ["reference", "nonsense"]}, "routes must be drawn"),
            ({"separation": 0.05}, "separation > 0.1"),
    ):
        with pytest.raises(ValueError, match=message):
            validate({**config, **update})
    broken = deepcopy(config)
    broken["sizes"]["test"] = 0
    with pytest.raises(ValueError, match="sizes must give"):
        validate(broken)
    broken = deepcopy(config)
    broken["perturbation"]["amplitudes"] = [0.1, 0.0]
    with pytest.raises(ValueError, match="amplitudes must start at 0"):
        validate(broken)
    broken = deepcopy(config)
    broken["directional"]["amplitudes"] = [0.7]
    with pytest.raises(ValueError, match="drawn from the error-propagation grid"):
        validate(broken)


def test_provenance_records_the_sources_a_result_depends_on():
    record = provenance(frozen())
    for name in ("generator.py", "readout.py", "perturbation.py", "recovery.py", "runs.py"):
        assert len(record["source_hashes"][name]) == 64
    assert record["environment"]["dtype"] == "float64"


def test_miniature_end_to_end_run(tmp_path):
    config_path = miniature(tmp_path)
    result = subprocess.run(
        [sys.executable, str(ROOT/"scripts/run_recovery.py"), "--config", str(config_path),
         "--output", str(tmp_path/"results")],
        cwd=ROOT, capture_output=True, text=True,
        env={"PYTHONPATH": str(ROOT/"src"), "PATH": "/usr/bin:/bin"})
    assert result.returncode == 0, result.stderr
    directory = next((tmp_path/"results"/"recovery").iterdir())

    manifest = json.loads((directory/"manifest.json").read_text())
    assert manifest["runs"]["0"]["status"] == "completed"
    assert manifest["config_id"] == directory.name

    record = json.loads((directory/"seed0.json").read_text())
    assert record["status"] == "completed" and len(record["cases"]) == 2
    for case in record["cases"]:
        # 1 baseline + 2 kinds x 2 amplitudes, each on both routes.
        assert len(case["propagation"]) == (1+2*2)*2
        assert len(case["sensitivity"]) == 1
        assert case["potential_setting"] == {"n_features": 8, "ridge": 1.0e-6}
        assert len(case["conditioning"]) == case["params"]["k"]*record["config"]["n"]
        assert {e["experiment"] for e in case["propagation"]} == {"potential-baseline", "error-propagation"}
        baseline = [e for e in case["propagation"]
                    if e["experiment"] == "potential-baseline" and e["route"] == "reference"]
        assert len(baseline) == 1 and baseline[0]["ratio_to_readout_baseline"] == 1.0
        assert all(e["evaluation_split"] == "test" for e in case["propagation"])
        assert all(e["centering_source"] == "calibration" for e in case["propagation"])
        assert all(e["oracle_direction_selection"] for e in case["conditioning"])
        assert "test" not in [s for s in case["splits"] if s == "unused"]

    rows = list(csv.DictReader((directory/"summary.csv").read_text().splitlines())) \
        if (directory/"summary.csv").exists() else []
    assert len(rows) == sum(len(c["propagation"])+len(c["sensitivity"])+len(c["conditioning"])
                            for c in record["cases"])


def test_a_config_without_per_case_settings_still_resolves():
    """One shared setting in `potential` still resolves, so a record written under that form
    stays readable. Built here rather than read from a file, so the test depends on the
    compatibility path and not on any particular configuration being present."""
    from recovery_core.runs import solver_settings
    config = deepcopy(frozen())
    config.pop("potential_by_case")
    config["potential"] = {**config["potential"], "n_features": 2048, "ridge": 1e-8}
    validate(config)
    resolved = solver_settings(config)
    assert set(resolved) == set(config["cases"])
    assert all(s["n_features"] == 2048 and s["ridge"] == 1e-8 for s in resolved.values())


def test_per_case_settings_must_cover_every_case():
    from recovery_core.runs import solver_settings
    config = frozen()
    config["potential_by_case"] = {"artanh": {"n_features": 2048, "ridge": 1e-8}}
    with pytest.raises(ValueError, match="cover exactly the configured cases"):
        solver_settings(config)
    broken = deepcopy(frozen())
    broken["potential_by_case"]["mixing"]["ridge"] = 0.0
    with pytest.raises(ValueError, match="needs n_features >= 1, ridge > 0"):
        validate(broken)

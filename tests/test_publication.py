"""A publishable dataset must reject corruption, incompleteness and wrong final weights."""

import json
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from recovery_reporting.data import digest, load_bundle

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def data_copy(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(ROOT / "data", data)
    return data


def rewrite(data, name, change):
    path = data / name
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))
    index_path = data / "manifest.json"
    index = json.loads(index_path.read_text())
    index["files"][name] = digest(path)
    index_path.write_text(json.dumps(index))


def test_published_data_have_all_checkpoints_and_cases():
    bundle = load_bundle(ROOT)
    assert len(bundle["estimators"]) == 10
    assert all(len(records) == 10 for records in bundle["recovery"].values())
    for record in bundle["estimators"]:
        for width in (64, 512):
            dsm = [r for r in record["rows"] if r["estimator"] == "dsm" and r["width"] == width]
            assert len({r["evaluated_state_sha256"] for r in dsm}) == 4
            logodds = [r for r in record["rows"] if r["estimator"] == "logodds" and r["width"] == width]
            assert sorted({r["update"] for r in logodds}) == [5000, 10000, 20000, 40000]
            assert {r["route_evaluated"] for r in logodds} == {"direct", "through_the_potential"}
            assert len(logodds) == 8
            assert len({r["evaluated_state_sha256"] for r in logodds}) == 4


def test_changed_checkpoint_supplement_bytes_fail_before_plotting(data_copy):
    path = data_copy/"estimators-checkpoints/52ed7c149f7cd299/seed0.json"
    path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_bundle(ROOT, data_copy)


def test_checkpoint_supplement_missing_a_row_fails_the_grid_check(data_copy):
    def drop_a_checkpoint(record):
        victim = next(r for r in record["rows"] if r["update"] == 5000
                      and r["route_evaluated"] == "through_the_potential")
        record["rows"].remove(victim)
    rewrite(data_copy, "estimators-checkpoints/52ed7c149f7cd299/seed0.json", drop_a_checkpoint)
    with pytest.raises(ValueError):
        load_bundle(ROOT, data_copy)


def test_changed_raw_bytes_fail_before_plotting(data_copy):
    path = data_copy / "estimators/52ed7c149f7cd299/seed0.json"
    path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_bundle(ROOT, data_copy)


def test_missing_file_cannot_pass_with_edited_manifest(data_copy):
    name = "recovery/2ab29ae68a24d88b/seed9.json"
    (data_copy / name).unlink()
    with pytest.raises(ValueError, match="file set"):
        load_bundle(ROOT, data_copy)


def test_missing_controlled_condition_fails_even_with_new_digest(data_copy):
    rewrite(data_copy, "recovery/2ab29ae68a24d88b/seed0.json", lambda d: d["cases"][0]["propagation"].pop())
    with pytest.raises(ValueError, match="propagation grid"):
        load_bundle(ROOT, data_copy)


def test_legacy_final_state_cannot_be_published(data_copy):
    def change(record):
        final = next(r for r in record["rows"] if r["estimator"] == "dsm" and r["update"] == 40000)
        final["evaluated_state_sha256"] = "0" * 64
    rewrite(data_copy, "estimators/52ed7c149f7cd299/seed0.json", change)
    # The checkpoint supplement links to the base file by digest; keep that link consistent so
    # this exercises validate_final's own DSM check rather than the (also correct) linkage check.
    base_path = data_copy/"estimators/52ed7c149f7cd299/seed0.json"
    rewrite(data_copy, "estimators-checkpoints/52ed7c149f7cd299/seed0.json",
           lambda d: d.update(base_seed_sha256=digest(base_path)))
    with pytest.raises(ValueError, match="Final DSM evaluation"):
        load_bundle(ROOT, data_copy)


def test_factorization_requires_every_seed_and_alignment(data_copy):
    rewrite(data_copy, "factorization_gap/8dc577d359c32f4d.json", lambda d: d["empirical"].pop())
    with pytest.raises(ValueError, match="gap rows"):
        load_bundle(ROOT, data_copy)


@pytest.mark.parametrize("change, message", [
    (lambda d: d.update(evaluation_split="test"), "Selection split/seeds"),
    (lambda d: d["selection"]["mixing"].update(ridge=1e-8), "frozen setting"),
    (lambda d: d["candidate_cells"].pop(), "Selection candidates"),
])
def test_selection_keeps_its_frozen_boundary(data_copy, change, message):
    rewrite(data_copy, "selection/2ab29ae68a24d88b/selection.json", change)
    with pytest.raises(ValueError, match=message):
        load_bundle(ROOT, data_copy)


def test_publication_file_rules_are_exact():
    spec = importlib.util.spec_from_file_location("mirror", ROOT / "scripts/make_publication_mirror.py")
    mirror = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mirror)
    published, unclaimed = mirror.classify(["README.md", "README.md.bak", "data/manifest.json",
                                           mirror.WITHHELD[0], "results/private.json"])
    assert published == ["README.md", "data/manifest.json"]
    assert unclaimed == ["README.md.bak", "results/private.json"]
    assert not mirror.dangling(["scripts/make_publication_mirror.py"])


def test_historical_equivalence_requires_an_explicit_external_source():
    result = subprocess.run([sys.executable, str(ROOT / "docs/equivalence_check.py"), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "--source-src" in result.stdout

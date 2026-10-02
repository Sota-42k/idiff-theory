"""Copy the selected completed rerun into data/, retaining raw bytes and provenance."""

import argparse
from pathlib import Path
import shutil
import tempfile
from typing import Any
import json

from recovery_core.runs import config_id, load_config
from recovery_estimators.records import identifier, load_config as estimator_config
from recovery_reporting.data import digest, read, require, load_bundle

ROOT = Path(__file__).resolve().parents[1]


def prepare(run: Path) -> None:
    target = ROOT / "data"
    require(not target.exists(), "data/ already exists; do not replace the published dataset in place")
    status, snapshot = read(run / "status.json"), read(run / "source-manifest.json")
    require(status["status"] == "completed" and len(status["stages"]) == 8
            and all(s["status"] == "completed" and s["returncode"] == 0 for s in status["stages"]),
            "The selected rerun is not complete")
    for name, value in snapshot["source_hashes"].items():
        require(digest(run / "source" / name) == value, f"Snapshot changed: {name}")
    ids = {f"n{n}": config_id(load_config(ROOT / f"configs/recovery-n{n}.yaml")) for n in (2, 3)}
    ids["estimators"] = identifier(estimator_config(ROOT / "configs/estimators.yaml"))
    files: list[str] = []
    for n in (2, 3):
        prefix = f"recovery/{ids[f'n{n}']}"
        files.extend(f"{prefix}/{name}" for name in ["manifest.json", "summary.csv", *[f"seed{s}.json" for s in range(10)]])
        files.append(f"factorization_gap/{ids[f'n{n}']}.json")
    files.append(f"selection/{ids['n2']}/selection.json")
    files.extend(f"estimators/{ids['estimators']}/{name}" for name in ["manifest.json", *[f"seed{s}.json" for s in range(10)]])
    source = {name: value for name, value in snapshot["source_hashes"].items()
              if name.startswith("src/") or name in
              ("configs/estimators.yaml", "configs/recovery-n2.yaml", "configs/recovery-n3.yaml",
               "scripts/run_recovery.py", "scripts/run_estimators.py", "scripts/select_potential.py",
               "scripts/factorization_gap.py", "uv.lock")}
    index: dict[str, Any] = {"schema_version": 1, "run_id": snapshot["run_id"],
                             "execution_git_head": snapshot["head"],
                             "execution_source_hashes": source, "config_ids": ids,
                             "started_at": status["started_at"], "finished_at": status["finished_at"],
                             "stages": [{k: s[k] for k in ("name", "started_at", "finished_at", "status")} for s in status["stages"]],
                             "files": {name: digest(run / "results" / name) for name in sorted(files)}}
    with tempfile.TemporaryDirectory(prefix=".publication-data-", dir=ROOT) as temp:
        data = Path(temp) / "data"
        for name in files:
            dest = data / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run / "results" / name, dest)
        (data / "manifest.json").write_text(json.dumps(index, indent=2)+"\n")
        load_bundle(ROOT, data)
        data.rename(target)
    print(f"Imported {len(files)} unchanged result files from {snapshot['run_id']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    prepare(parser.parse_args().run)

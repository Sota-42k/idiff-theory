"""Regenerate the published reports and five figures, or check them without overwriting."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

# A writable cache also permits use in read-only/isolated exports.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "recovery-matplotlib"))

import matplotlib
import numpy
import scipy
import torch

import analyse_recovery as recovery
from recovery_estimators.report import render
from recovery_reporting.data import digest, load_bundle, require
from recovery_reporting.figures import generate

ROOT = Path(__file__).resolve().parents[1]


def rendering_hashes(root: Path) -> dict[str, str]:
    paths = sorted((root / "src/recovery_reporting").glob("*.py"))
    paths += [root / name for name in ("scripts/reproduce.py", "scripts/analyse_recovery.py",
                                     "src/recovery_estimators/report.py", "pyproject.toml")]
    return {str(p.relative_to(root)): digest(p) for p in paths}


def controlled_report(bundle: dict[str, Any]) -> str:
    records, replication = bundle["recovery"][2], bundle["recovery"][3]
    index = bundle["index"]
    manifest = json.loads((ROOT / "data/recovery" / index["config_ids"]["n2"] / "manifest.json").read_text())
    rep_manifest = json.loads((ROOT / "data/recovery" / index["config_ids"]["n3"] / "manifest.json").read_text())
    amps = sorted({r["amplitude"] for r in records[0]["cases"][0]["propagation"] if r["amplitude"] > 0})
    lines: list[str] = []
    w = lines.append
    recovery.preamble(w, records, manifest, Path("data/recovery") / index["config_ids"]["n2"], {})
    recovery.readout_identity(w)
    recovery.least_squares_potential(w, records)
    recovery.quantitative_recovery(w, records, amps)
    recovery.conditioning(w, records)
    recovery.observable_diagnostic(w, records, amps)
    recovery.factorization(w, bundle["gaps"][2])
    recovery.plug_in_bound(w, records)
    recovery.replication(w, records, replication, rep_manifest, {}, amps, amps)
    recovery.limits(w)
    return "\n".join(lines)+"\n"


def reproduce(output: Path) -> None:
    bundle = load_bundle(ROOT)
    (output / "docs").mkdir(parents=True, exist_ok=True)
    (output / "docs/results.md").write_text(controlled_report(bundle))
    (output / "docs/estimators.md").write_text(render(bundle["estimators"], bundle["estimator_manifest"], []))
    generate(bundle, output / "figures")
    paths = [output / "docs/results.md", output / "docs/estimators.md"]
    paths += sorted(p for p in (output / "figures").iterdir() if p.name != "manifest.json")
    manifest = {"run_id": bundle["index"]["run_id"], "data_manifest_sha256": digest(ROOT / "data/manifest.json"),
                "rendering_source_hashes": rendering_hashes(ROOT),
                "environment": {"python": sys.version.split()[0], "numpy": numpy.__version__,
                                "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                                "torch": torch.__version__},
                "artifacts": {str(p.relative_to(output)): digest(p) for p in paths}}
    (output / "figures/manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


def verify() -> None:
    """Regenerate in a clean directory: data, reports and rendered files must all match."""
    with tempfile.TemporaryDirectory(prefix="recovery-reproduction-") as temporary:
        output = Path(temporary)
        reproduce(output)
        manifest = json.loads((output / "figures/manifest.json").read_text())
        names = [*manifest["artifacts"], "figures/manifest.json"]
        changed = [name for name in names if not (ROOT / name).is_file()
                   or digest(ROOT / name) != digest(output / name)]
        require(not changed, "Regenerated outputs differ: " + ", ".join(changed))
    print(f"Verified {len(names)} regenerated artifacts against {manifest['run_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="output root; default: this repository")
    parser.add_argument("--check", action="store_true", help="regenerate in a temporary directory and compare bytes")
    args = parser.parse_args()
    if args.check and args.out is not None:
        parser.error("--check and --out are mutually exclusive")
    if args.check:
        verify()
    else:
        reproduce(args.out or ROOT)
        print(f"Regenerated reports and five figures in {args.out or ROOT}")


if __name__ == "__main__":
    main()

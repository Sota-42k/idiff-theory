"""Run the frozen estimator comparison, or complete only its final DSM evaluations."""

import argparse
from pathlib import Path
from time import perf_counter

from recovery_estimators.experiment import run_seed
from recovery_estimators.records import (base_records, digest, identifier, load_config,
                                         provenance, run_directory, validate_final,
                                         validate_supplement, write)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--complete-final-from", type=Path,
                        help="retrain only DSM and evaluate its final iterate; keep base files intact")
    parser.add_argument("--output", type=Path, help="exact result directory (must differ from base)")
    parser.add_argument("--dry-run", action="store_true", help="validate and print scope; train/write nothing")
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else config["seeds"]
    if len(set(seeds)) != len(seeds) or not set(seeds) <= set(config["seeds"]):
        parser.error("--seeds must be a distinct subset of the frozen seeds")
    base = args.complete_final_from
    bases = base_records(base, config) if base is not None else {}
    if base is not None and not set(seeds) <= bases.keys():
        parser.error("Every requested seed needs a completed base record")
    mode = "dsm-final" if base is not None else "full"
    directory = args.output or Path(config["output"])/("estimators-final" if base else "estimators")/identifier(config)
    if base is not None and directory.resolve() == base.resolve():
        parser.error("Supplement output must differ from the base directory")
    for seed in seeds:
        if (directory/f"seed{seed}.json").exists():
            parser.error(f"seed {seed} already exists; choose missing seeds or a new output directory")
    print(f"{mode}: seeds={seeds}, widths={config['widths']}, updates={config['training']['updates']}")
    print(f"output: {directory}")
    if base is not None:
        print("DSM retraining from initialization only; no classifier, convex fit, or analytic baseline.")
    if args.dry_run:
        return
    metadata = {**provenance(config), "mode": mode}
    if base is not None:
        metadata["base_manifest_sha256"] = digest(base/"manifest.json")
    failures = []
    with run_directory(directory, metadata) as manifest:
        for seed in seeds:
            if (directory/f"seed{seed}.json").exists():
                raise FileExistsError(f"Seed {seed} already exists; refusing overwrite")
            started = perf_counter()
            extra = {"base_seed_sha256": digest(base/f"seed{seed}.json")} if base is not None else {}
            previous = manifest["runs"].get(str(seed))
            if previous is not None:
                manifest.setdefault("previous_attempts", {}).setdefault(str(seed), []).append(previous)
            manifest["runs"][str(seed)] = {"status": "running"}
            write(directory/"manifest.json", manifest)
            try:
                record = {**metadata, **extra, **run_seed(config, seed, base=bases.get(seed)),
                          "status": "completed", "wall_clock_seconds": perf_counter()-started}
                if base is not None:
                    validate_supplement(record, bases[seed], config, base/f"seed{seed}.json")
                else:
                    validate_final(record, config)
                write(directory/f"seed{seed}.json", record)
                manifest["runs"][str(seed)] = {"status": "completed", "failure": None,
                                              "wall_clock_seconds": record["wall_clock_seconds"]}
            except Exception as error:
                failures.append(seed)
                manifest["runs"][str(seed)] = {"status": "failed", "failure": repr(error),
                                              "wall_clock_seconds": perf_counter()-started}
            write(directory/"manifest.json", manifest)
            print(f"seed {seed}: {manifest['runs'][str(seed)]}", flush=True)
    if failures:
        raise SystemExit(f"Failed seeds (retained in manifest): {failures}")


if __name__ == "__main__":
    main()

"""Retrain the classifier from a completed base run and add intermediate checkpoints.

New supplementary runner for `CLASSIFIER-CHECKPOINT-PLAN.md`. It never writes into the base
directory: it replays the classifier for the requested seeds from the same frozen
`configs/estimators.yaml`, keeps weight copies at the DSM `evaluate_at` schedule (5k/10k/20k
by default), evaluates both routes at every checkpoint plus the final update, and verifies
that the replayed final state matches the hash already published for that seed/width in the
base record before writing anything.
"""

import argparse
from pathlib import Path
from time import perf_counter

from recovery_estimators.checkpoint_experiment import classifier_checkpoint_rows
from recovery_estimators.checkpoint_records import (expected_final_logodds_hashes, provenance,
                                                     validate_checkpoint_supplement)
from recovery_estimators.experiment import Evaluation
from recovery_estimators.records import base_records, digest, load_config, run_directory, write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True,
                        help="directory with completed base seed records (e.g. data/estimators/<id>)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="validate and print scope; train/write nothing")
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else config["seeds"]
    if len(set(seeds)) != len(seeds) or not set(seeds) <= set(config["seeds"]):
        parser.error("--seeds must be a distinct subset of the frozen seeds")
    bases = base_records(args.base, config)
    if not set(seeds) <= bases.keys():
        parser.error("Every requested seed needs a completed base record")
    evaluate_at = tuple(config["dsm"]["evaluate_at"])
    if args.output.resolve() == args.base.resolve():
        parser.error("Supplement output must differ from the base directory")
    for seed in seeds:
        if (args.output/f"seed{seed}.json").exists():
            parser.error(f"seed {seed} already exists; choose missing seeds or a new output directory")
    print(f"logodds-checkpoints: seeds={seeds}, widths={config['widths']}, "
         f"evaluate_at={list(evaluate_at)}, final={config['training']['updates']}")
    print(f"base: {args.base}")
    print(f"output: {args.output}")
    if args.dry_run:
        return
    metadata = {**provenance(config), "mode": "logodds-checkpoints",
               "base_manifest_sha256": digest(args.base/"manifest.json")}
    failures = []
    with run_directory(args.output, metadata) as manifest:
        for seed in seeds:
            if (args.output/f"seed{seed}.json").exists():
                raise FileExistsError(f"Seed {seed} already exists; refusing overwrite")
            started = perf_counter()
            base_file = args.base/f"seed{seed}.json"
            base = bases[seed]
            previous = manifest["runs"].get(str(seed))
            if previous is not None:
                manifest.setdefault("previous_attempts", {}).setdefault(str(seed), []).append(previous)
            manifest["runs"][str(seed)] = {"status": "running"}
            write(args.output/"manifest.json", manifest)
            try:
                expected_hashes = expected_final_logodds_hashes(base, config)
                evaluation = Evaluation.generate(config, seed)
                rows = []
                for width in config["widths"]:
                    rows.extend(classifier_checkpoint_rows(evaluation, config, seed, width,
                                                           evaluate_at, expected_hashes[width]))
                record = {**metadata, "seed": seed, "status": "completed",
                         "base_seed_sha256": digest(base_file), "rows": rows,
                         "wall_clock_seconds": perf_counter()-started}
                validate_checkpoint_supplement(record, base, config, base_file, evaluate_at)
                write(args.output/f"seed{seed}.json", record)
                manifest["runs"][str(seed)] = {"status": "completed", "failure": None,
                                              "wall_clock_seconds": record["wall_clock_seconds"]}
            except Exception as error:
                failures.append(seed)
                manifest["runs"][str(seed)] = {"status": "failed", "failure": repr(error),
                                              "wall_clock_seconds": perf_counter()-started}
            write(args.output/"manifest.json", manifest)
            print(f"seed {seed}: {manifest['runs'][str(seed)]}", flush=True)
    if failures:
        raise SystemExit(f"Failed seeds (retained in manifest): {failures}")


if __name__ == "__main__":
    main()

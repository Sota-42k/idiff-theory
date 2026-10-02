"""Select the potential-solver setting per case, on seeds held out from production.

**The criterion is declared here, before the run, and is not revisited afterwards.** For each
case independently, choose the `(n_features, ridge)` minimising the geometric mean of the
oracle-aligned MSE over the selection seeds, evaluated on the **validation** split. The
geometric mean is used because the quantity spans orders of magnitude across seeds and no
seed should dominate. Ties within 5% are broken towards the smaller `n_features`, which is
cheaper to fit and better conditioned.

This is an **oracle validation selection**: the aligned error is measured against the true
statistics through the true alignment. The design permits that provided it is labelled, the
choice is frozen before any test number is reported, and the selection seeds are disjoint
from the production seeds. All three hold here.

Settings are chosen per case rather than jointly. A single setting is known to handicap one
case --- 2048 costs `mixing` about 1.37x and 4096 costs `artanh` about 1.4x --- and a control
that is known to be handicapped relative to the benchmark is not serving as a control.

  uv run python scripts/select_potential.py --config configs/recovery-n2.yaml \
      --seeds 100 101 102 103 104 105 106 107 108 109

Writes results/selection/<config_id>/selection.json.
"""

import argparse
import gc
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from recovery_core.recovery import build_case, evaluate_route, fit_route, perturbed_scores
from recovery_core.readout import PotentialBasis
from recovery_core.runs import config_id, jsonable, load_config, provenance, solver_record, write_json

TIE_TOLERANCE = .05


def choose_setting(rows: list[dict[str, Any]], case_name: str,
                   grid: list[tuple[int, float]], reference_only: list[tuple[int, float]]
                   ) -> dict[str, Any]:
    scores = {}
    for width, ridge in grid + reference_only:
        values = [r["oracle_aligned_mse"] for r in rows
                  if r["case"] == case_name and r["n_features"] == width and r["ridge"] == ridge]
        if not values or not np.all(np.isfinite(values)) or min(values) <= 0:
            raise ValueError("Selection requires finite positive errors in every declared cell")
        scores[(width, ridge)] = float(np.exp(np.mean(np.log(values))))
    candidates = {cell: scores[cell] for cell in grid}
    best = min(candidates.values())
    tied = sorted(cell for cell, value in candidates.items() if value <= best*(1+TIE_TOLERANCE))
    chosen = tied[0]
    return {"n_features": chosen[0], "ridge": chosen[1],
            "geometric_mean_aligned_mse": candidates[chosen], "best_cell_value": best,
            "tied_within_tolerance": [list(cell) for cell in tied],
            "grid": {f"{w}:{r:g}": value for (w, r), value in sorted(candidates.items())},
            "reference_only_not_candidates": {f"{w}:{r:g}": scores[(w, r)]
                                               for w, r in reference_only}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--ridges", type=float, nargs="+", default=[1e-9, 1e-8, 1e-7])
    parser.add_argument("--extra-cells", nargs="*", default=["8192:1e-8"],
                        help="measured for context only; never eligible to be selected")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", type=Path, default=Path("results/selection"))
    args = parser.parse_args()

    config = load_config(args.config)
    if set(args.seeds) & set(config["seeds"]):
        raise ValueError("Selection seeds must be disjoint from the production seeds")
    grid = list(dict.fromkeys((w, r) for w in args.widths for r in args.ridges))
    reference_only = list(dict.fromkeys((int(c.split(":")[0]), float(c.split(":")[1]))
                                       for c in args.extra_cells))
    reference_only = [cell for cell in reference_only if cell not in grid]
    cells = grid + reference_only
    basis_seed = config["potential"]["basis_seed"]
    chunk = config["potential"]["chunk_size"]
    started = perf_counter()
    rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        for case_name, identity in config["cases"].items():
            case = build_case(config["n"], seed, config["separation"], identity=identity)
            fitting = case.split("fitting", config["sizes"]["fitting"])
            calibration = case.split("calibration", config["sizes"]["calibration"])
            evaluation = case.split(args.split, config["sizes"][args.split])
            clean_fit = perturbed_scores(case, fitting.x, None, 0.)
            clean_eval = perturbed_scores(case, evaluation.x, None, 0.)
            for width, ridge in cells:
                basis = PotentialBasis.from_data(fitting.x, width, basis_seed)
                potential = fit_route(case, "reference", fitting, clean_fit,
                                      n_features=width, ridge=ridge,
                                      basis_seed=basis_seed, basis=basis)
                rows.append(jsonable({
                    "seed": seed, "case": case_name, "n_features": width, "ridge": ridge,
                    **evaluate_route(case, "reference", potential, calibration, evaluation,
                                     clean_eval),
                    "solver": solver_record(potential)}))
                del potential, basis
                gc.collect()
            print(f"seed {seed} {case_name} done at {perf_counter()-started:.0f}s", flush=True)

    selection = {name: choose_setting(rows, name, grid, reference_only)
                 for name in config["cases"]}
    for name, chosen in selection.items():
        print(f"{name}: chose n_features={chosen['n_features']}, ridge={chosen['ridge']:g}")

    write_json(args.output/config_id(config)/"selection.json",
               {**provenance(config), "role": "oracle validation selection, per case",
                "criterion": "min geometric mean of oracle-aligned MSE over the selection "
                             "seeds, on the validation split; ties within 5% go to the "
                             "smaller n_features",
                "criterion_declared": "before the run, in the script docstring",
                "selection_seeds": args.seeds, "production_seeds": config["seeds"],
                "evaluation_split": args.split,
                "candidate_cells": [list(c) for c in grid],
                "reference_only_cells": [list(c) for c in reference_only],
                "selection": selection, "rows": rows,
                "wall_clock_seconds": perf_counter()-started})
    print(f"\ndone in {perf_counter()-started:.0f}s")


if __name__ == "__main__":
    main()

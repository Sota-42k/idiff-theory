"""What the affine readout metric can and cannot see about the factorization.

The Discussion distinguishes affine readout error from component recovery.
This script measures how little the first says about the second.

It is NOT a factorization search. No witness is constructed, no candidate is returned, and
nothing here bears on whether the search of Section 4.4 is tractable, which the manuscript
explicitly declines to claim. the factorization search remains deferred.

  uv run python scripts/factorization_gap.py --results results/recovery/<config_id>

Writes results/factorization_gap/<config_id>.json.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from recovery_core.metrics import (affine_residual, block_permutation_defect, centered_errors,
                                   fit_affine, oracle_alignment_error, variance)
from recovery_core.recovery import build_case, fit_route, perturbed_scores
from recovery_core.readout import PotentialBasis
from recovery_core.runs import jsonable, solver_settings, write_json


def invertible(rng: np.random.Generator, q: int) -> np.ndarray:
    while True:
        M = rng.normal(size=(q, q))
        if abs(np.linalg.det(M)) > .1:
            return M


def single_factor_r2(psi: np.ndarray, z: np.ndarray) -> float:
    """Component recovery: how much of each recovered coordinate one latent factor explains.

    For each coordinate of `psi`, regress it on a quadratic in each single `z_i` and keep the
    best R^2; average over coordinates. The true statistics are `(z_i, z_i^2)` per factor, so
    a block-separated `psi` scores 1 exactly, and a coordinate that genuinely mixes two
    factors cannot. Oracle evaluation: it uses the true latents.
    """
    best = []
    for a in range(psi.shape[1]):
        target = psi[:, a]-psi[:, a].mean()
        denominator = float(target @ target)
        scores = []
        for i in range(z.shape[1]):
            design = np.column_stack((np.ones(len(z)), z[:, i], z[:, i]**2))
            residual = target-design @ np.linalg.lstsq(design, target, rcond=None)[0]
            scores.append(1-float(residual @ residual)/denominator)
        best.append(max(scores))
    return float(np.mean(best))


def reparameterizations(rng: np.random.Generator, n: int, k: int) -> dict[str, np.ndarray]:
    """Candidate alignments, written as `M A_0`. `M` in the block-permutation group is what
    `~_P` allows; everything else is an alignment that satisfies `~_A` just as well."""
    q = n*k
    out = {"identity": np.eye(q)}
    for i in range(3):
        out[f"random invertible {i+1}"] = invertible(rng, q)
    order = np.concatenate([np.arange(k)+k*j for j in list(range(n))[::-1]])
    out["block permutation"] = np.eye(q)[order]
    shear = np.eye(q)
    shear[0, k] = 1.7
    out["block-mixing shear"] = shear
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/factorization_gap"))
    args = parser.parse_args()
    manifest = json.loads((args.results/"manifest.json").read_text())
    config = manifest["config"]
    n, k = config["n"], config["k"]
    started = perf_counter()
    record: dict[str, Any] = {"config_id": manifest["config_id"], "n": n, "k": k,
                              "role": "the ~_A / ~_P distinction; not a factorization search",
                              "structural": [], "empirical": []}

    # ---- Part A: the exact readout, no estimation error and no fitting at all. ----
    case = build_case(n, config["seeds"][0], config["separation"], identity=True)
    cal, test = (case.split(s, config["sizes"][s]) for s in ("calibration", "test"))
    G = case.alignment("reference")
    exact = lambda s: case.oracle.statistics(s.x) @ G.T - case.params.kappa
    upsilon_test, upsilon_cal = case.oracle.statistics(test.x), case.oracle.statistics(cal.x)
    denominator = variance(upsilon_test)
    A0 = np.linalg.inv(G)
    z_test = case.mix.g_inv(test.x)
    for name, M in reparameterizations(np.random.default_rng(0), n, k).items():
        # Psi = M A_0 (Lambda + kappa): the coordinates this candidate alignment recovers.
        psi_cal = (exact(cal)+case.params.kappa) @ (M @ A0).T
        psi_test = (exact(test)+case.params.kappa) @ (M @ A0).T
        fitted = fit_affine(psi_cal, upsilon_cal)
        record["structural"].append(jsonable({
            "alignment": name,
            **affine_residual(fitted, psi_test, upsilon_test, denominator=denominator),
            "single_factor_r2": single_factor_r2(psi_test, z_test),
            **block_permutation_defect(M, n, k)}))

    # ---- Part B: the same question on the run's actual estimated readouts. ----
    for seed in config["seeds"]:
        for case_name, identity in config["cases"].items():
            primary = solver_settings(config)[case_name]
            built = build_case(n, seed, config["separation"], identity=identity)
            fitting, cal, test = (built.split(s, config["sizes"][s])
                                  for s in ("fitting", "calibration", "test"))
            basis = PotentialBasis.from_data(fitting.x, primary["n_features"],
                                             primary["basis_seed"])
            potential = fit_route(built, "reference", fitting,
                                  perturbed_scores(built, fitting.x, None, 0.),
                                  n_features=primary["n_features"], ridge=primary["ridge"],
                                  basis_seed=primary["basis_seed"], basis=basis)
            G = built.alignment("reference")
            ups_cal, ups_test = (built.oracle.statistics(s.x) for s in (cal, test))
            denominator = variance(ups_test)
            A0 = np.linalg.inv(G)
            z_test = built.mix.g_inv(test.x)
            hat_cal, hat_test = potential(cal.x), potential(test.x)
            # The potential is zero-mean on the fitting split while the true readout carries
            # its own constant, so the offset is removed on calibration exactly as the main
            # evaluation does. Using hat - true directly would charge this diagnostic an
            # additive error the rest of the document removes.
            centred, _ = centered_errors(hat_cal, ups_cal @ G.T - built.params.kappa,
                                         hat_test, ups_test @ G.T - built.params.kappa)
            for name, M in reparameterizations(np.random.default_rng(seed), n, k).items():
                psi_cal = (hat_cal+built.params.kappa) @ (M @ A0).T
                psi_test = (hat_test+built.params.kappa) @ (M @ A0).T
                fitted = fit_affine(psi_cal, ups_cal)
                record["empirical"].append(jsonable({
                    "seed": seed, "case": case_name, "alignment": name,
                    **affine_residual(fitted, psi_test, ups_test, denominator=denominator),
                    "single_factor_r2": single_factor_r2(psi_test, z_test),
                    "oracle_aligned_mse": oracle_alignment_error(centred, G),
                    **block_permutation_defect(M, n, k)}))
            print(f"seed {seed} {case_name} done at {perf_counter()-started:.0f}s", flush=True)

    record["wall_clock_seconds"] = perf_counter()-started
    write_json(args.out/f"{manifest['config_id']}.json", record)
    print(f"wrote {args.out/(manifest['config_id']+'.json')} in {record['wall_clock_seconds']:.0f}s")


if __name__ == "__main__":
    main()

"""the recovery-map experiments production runner. Design Section 8, step 5.

Runs potential-baseline, error-propagation and alignment-conditioning for the seeds of a frozen config and writes one JSON per seed under
results/recovery/<config_id>/, plus a manifest and a summary CSV. Partial and failed seeds stay
in the manifest with their reason.

  uv run python scripts/run_recovery.py --config configs/recovery-n2.yaml
  uv run python scripts/run_recovery.py --config configs/recovery-n2.yaml --seeds 0 1

Selection is already frozen in the config; nothing here chooses a setting, and no number is
computed on anything but the test split. The calibration split fixes A, b and the centering
offset. The alignment-conditioning direction selection uses the true L and is labelled an oracle control.
"""

import argparse
import csv
import gc
import json
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from recovery_core.directions import singular_directions
from recovery_core.metrics import oracle_alignment_error, raw_readout_error
from recovery_core.perturbation import (Perturbation, curl_energy, directional_perturbation,
                                        draw_perturbation)
from recovery_core.recovery import (Case, Split, build_case, evaluate_route, observed_field,
                                  perturbed_scores, stratified_sample)
from recovery_core.readout import PotentialBasis, fit_potential, fit_potentials
from recovery_core.runs import (config_id, jsonable, load_config, perturbation_record,
                                provenance, solver_record, solver_settings, write_json)

SUMMARY_COLUMNS = ("seed", "case", "experiment", "route", "kind", "amplitude", "mode",
                   "n_features", "ridge", "rho_test", "field_error_mse",
                   "solver_gradient_residual", "raw_readout_mse", "oracle_aligned_mse",
                   "affine_mse", "affine_nmse", "V_T", "ratio_to_readout_baseline",
                   "singular_value", "predicted_aligned_mse")


def _amplitude_keys(config: dict[str, Any]) -> list[tuple[str, float]]:
    """(kind, amplitude) pairs; amplitude 0 is shared, being the same field for every kind."""
    kinds = config["perturbation"]["kinds"]
    pairs: list[tuple[str, float]] = [(kinds[0], 0.)]
    for kind in kinds:
        pairs += [(kind, a) for a in config["perturbation"]["amplitudes"] if a != 0]
    return pairs


def run_case(config: dict[str, Any], seed: int, case_name: str, identity: bool) -> dict[str, Any]:
    started = perf_counter()
    case: Case = build_case(config["n"], seed, config["separation"], identity=identity)
    sizes = config["sizes"]
    splits: dict[str, Split] = {name: case.split(name, sizes[name]) for name in sizes}
    fitting, calibration, test, pcal = (splits["fitting"], splits["calibration"],
                                        splits["test"], splits["pert_calibration"])
    routes = config["routes"]
    # basis_seed and chunk_size are shared; n_features and ridge are chosen per case.
    primary = solver_settings(config)[case_name]
    basis = PotentialBasis.from_data(fitting.x, primary["n_features"], primary["basis_seed"])

    settings = config["perturbation"]
    perturbations: dict[str, Perturbation] = {
        kind: draw_perturbation(pcal.x, case.params.q+1, kind=kind, seed=settings["seed"],
                                n_features=settings["n_features"], hidden=settings["hidden"])
        for kind in settings["kinds"]}
    record: dict[str, Any] = {
        "seed": seed, "case": case_name, "identity_mixing": identity,
        "potential_setting": {"n_features": primary["n_features"], "ridge": primary["ridge"]},
        "params": jsonable(case.params.metadata()),
        "rotations": jsonable(np.asarray(case.mix.rotations)),
        "splits": {name: {"n_per_u": s.n_per_u, "seed": s.seed, "samples": len(s.x)}
                   for name, s in splits.items()},
        "oracle_use": "controlled fields, evaluation and alignment-conditioning direction selection only",
        "perturbations": [perturbation_record(p, name) for name, p in perturbations.items()],
        # A marginal sample: the split is stored condition by condition.
        "curl": {name: jsonable(curl_energy(p, stratified_sample(test, 400), 1.))
                 for name, p in perturbations.items()},
        "rho_test": {f"{k}/{a:g}": perturbations[k].realized_size(test.x, a)
                     for k, a in _amplitude_keys(config)},
        "propagation": [], "sensitivity": [], "conditioning": [], "status": "running",
    }

    # ---- potential-baseline and error-propagation share one basis pass: the Gram depends on (basis, x) alone. ----
    pairs = _amplitude_keys(config)
    fields: dict[str, np.ndarray] = {}
    for kind, amplitude in pairs:
        scores = perturbed_scores(case, fitting.x, perturbations[kind], amplitude)
        for route in routes:
            fields[f"{kind}|{amplitude!r}|{route}"] = observed_field(case, scores, route)
        del scores
    potentials = fit_potentials(fitting.x, fields, ridge=primary["ridge"],
                                chunk_size=primary["chunk_size"], basis=basis)
    del fields
    gc.collect()

    baselines: dict[str, float] = {}
    for kind, amplitude in pairs:
        scores = perturbed_scores(case, test.x, perturbations[kind], amplitude)
        for route in routes:
            key = f"{kind}|{amplitude!r}|{route}"
            report = evaluate_route(case, route, potentials[key], calibration, test, scores)
            if amplitude == 0:
                baselines[route] = report["raw_readout_mse"]
            record["propagation"].append(jsonable({
                "experiment": "potential-baseline" if amplitude == 0 else "error-propagation",
                "case": case_name, "kind": None if amplitude == 0 else kind,
                "amplitude": amplitude, **primary,
                "rho_test": perturbations[kind].realized_size(test.x, amplitude),
                # Per seed and per route, against this run's own alpha = 0 readout error.
                "ratio_to_readout_baseline": report["raw_readout_mse"]/baselines[route],
                "injection_dominated": report["raw_readout_mse"] >= 10*baselines[route],
                **report, "solver": solver_record(potentials[key])}))
        del scores
    del potentials
    gc.collect()

    # ---- potential-baseline sensitivity: reported, never used to select after the freeze. ----
    sensitivity = config.get("potential_sensitivity") or {}
    clean_test = perturbed_scores(case, test.x, None, 0.)
    clean_fit = perturbed_scores(case, fitting.x, None, 0.)
    for width in sensitivity.get("n_features", []):
        variant = PotentialBasis.from_data(fitting.x, width, primary["basis_seed"])
        for ridge in sensitivity["ridges"]:
            potential = fit_potential(fitting.x, observed_field(case, clean_fit, "reference"),
                                      ridge=ridge, chunk_size=primary["chunk_size"],
                                      basis=variant)
            record["sensitivity"].append(jsonable({
                "experiment": "potential-baseline-sensitivity", "case": case_name, "route": "reference",
                "n_features": width, "ridge": ridge,
                **evaluate_route(case, "reference", potential, calibration, test, clean_test),
                "solver": solver_record(potential)}))
            del potential
        del variant
        gc.collect()
    del clean_fit

    # ---- alignment-conditioning: matched raw error per mode, aligned error predicted to scale as 1/s_j^2. ----
    G = case.alignment("reference")
    U, singular, _ = singular_directions(G)
    for amplitude in config["directional"]["amplitudes"]:
        directional_fields, injected_fields = {}, {}
        perturbs = {}
        for j in range(case.params.q):
            perturbs[j] = directional_perturbation(
                pcal.x, np.concatenate(([0.], U[:, j])), seed=config["directional"]["seed"],
                n_features=settings["n_features"], hidden=settings["hidden"])
            scores = perturbed_scores(case, fitting.x, perturbs[j], amplitude)
            directional_fields[str(j)] = observed_field(case, scores, "reference")
            injected_fields[str(j)] = observed_field(case, perturbs[j](fitting.x, amplitude),
                                                     "reference")
            del scores
        full = fit_potentials(fitting.x, directional_fields, ridge=primary["ridge"],
                              chunk_size=primary["chunk_size"], basis=basis)
        injected = fit_potentials(fitting.x, injected_fields, ridge=primary["ridge"],
                                  chunk_size=primary["chunk_size"], basis=basis)
        del directional_fields, injected_fields
        gc.collect()
        for j in range(case.params.q):
            scores = perturbed_scores(case, test.x, perturbs[j], amplitude)
            report = evaluate_route(case, "reference", full[str(j)], calibration, test, scores)
            alone = injected[str(j)](test.x)
            raw_alone = raw_readout_error(alone)
            record["conditioning"].append(jsonable({
                "experiment": "alignment-conditioning", "case": case_name, "mode": j, "amplitude": amplitude,
                "singular_value": singular[j], "oracle_direction_selection": True,
                "predicted_aligned_mse": raw_alone["raw_readout_mse"]/singular[j]**2,
                "injection_only_raw_readout_mse": raw_alone["raw_readout_mse"],
                "injection_only_oracle_aligned_mse": oracle_alignment_error(alone, G),
                "injection_only_rank_one_residual": float(np.max(np.abs(
                    alone-np.outer(alone @ U[:, j], U[:, j])))),
                **report}))
            del scores
        del full, injected, perturbs
        gc.collect()

    record["status"] = "completed"
    record["wall_clock_seconds"] = perf_counter()-started
    return record


def summarize(directory: Path) -> None:
    rows = []
    for path in sorted(directory.glob("seed*.json")):
        record = json.loads(path.read_text())
        for case in record.get("cases", []):
            for section in ("propagation", "sensitivity", "conditioning"):
                for entry in case.get(section, []):
                    rows.append({k: entry.get(k, case.get(k, record.get(k)))
                                 for k in SUMMARY_COLUMNS})
    with (directory/"summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {directory/'summary.csv'} with {len(rows)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/recovery-n2.yaml"))
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else config["seeds"]
    if not set(seeds) <= set(config["seeds"]):
        raise ValueError("--seeds must be drawn from the frozen config's seeds")
    directory = (args.output or Path(config["output"]))/"recovery"/config_id(config)
    manifest_path = directory/"manifest.json"
    manifest: dict[str, Any] = {**provenance(config), "seeds_requested": list(seeds),
                                "runs": {}}
    started = perf_counter()
    for seed in seeds:
        cases, status, reason = [], "completed", None
        try:
            for case_name, identity in config["cases"].items():
                cases.append(run_case(config, seed, case_name, identity))
                print(f"seed {seed} {case_name} done at {perf_counter()-started:.0f}s",
                      flush=True)
        except Exception as error:   # a failed seed stays visible, it is not dropped
            status, reason = "failed", f"{type(error).__name__}: {error}"
            traceback.print_exc()
        write_json(directory/f"seed{seed}.json",
                   {**provenance(config), "seed": seed, "cases": cases,
                    "status": status, "failure": reason})
        manifest["runs"][str(seed)] = {"status": status, "failure": reason,
                                       "cases_completed": len(cases)}
        write_json(manifest_path, manifest)
    manifest["wall_clock_seconds"] = perf_counter()-started
    write_json(manifest_path, manifest)
    summarize(directory)
    failed = [s for s, r in manifest["runs"].items() if r["status"] != "completed"]
    print(f"{len(seeds)-len(failed)}/{len(seeds)} seeds completed in "
          f"{manifest['wall_clock_seconds']:.0f}s" + (f"; failed: {failed}" if failed else ""))


if __name__ == "__main__":
    main()

"""Generate one report from validated seed records; never infer causes from a ranking."""

from typing import Any

import numpy as np


def select(records: list[dict[str, Any]], **match: Any) -> list[dict[str, Any]]:
    return [row for record in records for row in record["rows"]
            if all(row.get(k) == v for k, v in match.items())]


def summary(rows: list[dict[str, Any]], key: str, *, explained: bool = False) -> str:
    if not rows:
        return "not measured (n=0)"
    values = np.asarray([r[key] for r in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite {key} in report input")
    if explained:
        values = 1-values
    spread = f"{np.std(values, ddof=1):.4g}" if len(values) > 1 else "undefined"
    return f"{values.mean():.4g} ± {spread} (n={len(values)})"


def render(records: list[dict[str, Any]], manifest: dict[str, Any], notes: list[str]) -> str:
    config = manifest["config"]
    final = config["training"]["updates"]
    lines = ["# Which estimator reaches the construction", "",
             f"Configuration `{manifest['config_id']}`; {len(records)}/{len(config['seeds'])} "
             f"completed seeds; n={config['n']}, `{config['case']}`.", "", *notes, "",
             "## 1. From an estimate to the readout", "",
             "Models use observed (x,u) for training. The independent calibration split fixes "
             "the affine evaluation map; the test split measures its error. Ground truth is "
             "used for evaluation only. Direct log-odds estimate the readout; the potential "
             "route integrates estimated contrast fields.", "",
             "Tables report mean ± sample SD across seeds, with the contributing seed count. "
             "Each seed changes both the generated model and samples; SD is not a confidence "
             "interval. `1 - affine NMSE` is variance explained after affine calibration and "
             "does not establish component recovery.", "",
             "The analytic-score baseline measures the numerical potential approximation, "
             "not a required accuracy threshold for a learned estimator: "
             +summary([r["analytic_reference"] for r in records], "affine_nmse")+".", "",
             "| estimator | width | updates | route | affine NMSE | 1 - NMSE |",
             "|---|---|---|---|---|---|"]
    for width in config["widths"]:
        for name, route in (("logodds", "direct"), ("logodds", "through_the_potential"),
                            ("dsm", "through_the_potential")):
            rows = select(records, estimator=name, width=width, update=final, route_evaluated=route)
            lines.append(f"| {name} | {width} | {final} | {route} | "
                         f"{summary(rows, 'affine_nmse')} | {summary(rows, 'affine_nmse', explained=True)} |")
    lines += ["", "Agreement of direct and potential metrics is a numerical consistency "
              "diagnostic. It does not prove that the potential adds zero error.", "",
              "## 2. Recovery along DSM training", "",
              "| width | updates | 1 - NMSE |", "|---|---|---|"]
    for width in config["widths"]:
        for update in sorted(set(config["dsm"]["evaluate_at"]+[final])):
            rows = select(records, estimator="dsm", width=width, update=update)
            lines.append(f"| {width} | {update} | {summary(rows, 'affine_nmse', explained=True)} |")
    lines += ["", "These are measurements at the listed iterates, not a guarantee of "
              "monotonicity or convergence. The raw denoising objective is not the score-error "
              "metric. A lack of improvement over measured budgets does not exclude "
              "optimization effects or establish that further training could never help.", "",
              "## 3. Recovery along classifier training", "",
              "| width | updates | 1 - NMSE |", "|---|---|---|"]
    for width in config["widths"]:
        for update in sorted(set(config["dsm"]["evaluate_at"]+[final])):
            rows = select(records, estimator="logodds", width=width, update=update,
                          route_evaluated="through_the_potential")
            lines.append(f"| {width} | {update} | {summary(rows, 'affine_nmse', explained=True)} |")
    lines += ["", "Measurements at the listed iterates from one classifier training run per "
              "seed/width, kept at the same budgets as the DSM curve above and evaluated "
              "through the same potential route for direct comparison. The direct-route "
              "evaluation (no potential solver) is recorded per seed in the underlying data, "
              "not tabulated here. Not a guarantee of monotonicity or convergence.", "",
              "## 4. A convex classifier reference", "",
              "| features | affine NMSE | 1 - NMSE | converged |", "|---|---|---|---|"]
    for features in config["convex"]["n_features"]:
        rows = select(records, estimator="convex_logodds", n_features=features, route_evaluated="direct")
        converged = sum(bool(r["converged"]) for r in rows)
        lines.append(f"| {features} | {summary(rows, 'affine_nmse')} | "
                     f"{summary(rows, 'affine_nmse', explained=True)} | {converged}/{len(rows)} |")
    lines += ["", "Convergence refers to this fixed-basis, regularized problem. Its performance "
              "does not exclude local optimization or capacity limitations in the neural models.", "",
              "## Scope", "",
              "This fixed synthetic design illustrates affine recovery and its limitations. "
              "It is not a general comparison of objectives, a validation of component recovery, "
              "or evidence for a unique failure mechanism. Controlling contrast error efficiently "
              "and separating statistical from optimization effects remain future work."]
    return "\n".join(lines)+"\n"

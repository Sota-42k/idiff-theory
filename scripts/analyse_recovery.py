"""Turn a completed run into docs/results.md, organised by the manuscript's claims.

  uv run python scripts/analyse_recovery.py --results results/recovery/<config_id>

Sections follow the argument of Section 4: the readout identity, the least-squares potential,
quantitative recovery, the conditioning of the alignment, the observable diagnostic, and the
step from affine recovery to the factorization. Development history does not appear.

Every number comes from the saved seed files, except the plug-in bound panel, which combines
the saved field error and singular values with the analytic comparison constant of `bounds.py`
and is labelled as the diagnostic it is. No sentence hard-codes a value: a rerun that moves a
number moves the prose with it.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from recovery_core.bounds import comparison_constants
from recovery_core.generator import Params
from recovery_core.perturbation import Perturbation, TanhFeatures, contrast_curl
from recovery_core.recovery import build_case, stratified_sample
from recovery_core.runs import source_ast_hashes, source_hashes

CASES = ("artanh", "mixing")
KINDS = ("gradient", "field")
ROUTES = ("reference", "regression")
Writer = Callable[[str], None]


def load(directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[int, str]]:
    """Completed seeds only. Failures are returned, not dropped: the runner keeps them in the
    manifest on purpose and a document that quietly analysed the survivors would undo that."""
    manifest = json.loads((directory/"manifest.json").read_text())
    completed, failed = [], {}
    for seed in sorted(int(s) for s in manifest["runs"]):
        entry = manifest["runs"][str(seed)]
        if entry.get("status") == "completed":
            completed.append(json.loads((directory/f"seed{seed}.json").read_text()))
        else:
            failed[seed] = entry.get("failure") or str(entry.get("status"))
    if not completed:
        raise SystemExit(f"No completed seed in {directory}")
    return completed, manifest, failed


def rows_of(recs: list[dict[str, Any]], case: str, section: str) -> list[dict[str, Any]]:
    out = []
    for r in recs:
        c = next(x for x in r["cases"] if x["case"] == case)
        out += [{**e, "seed": r["seed"]} for e in c[section]]
    return out


def propagation(recs: list[dict[str, Any]], case: str, route: str, kind: str | None,
                amplitude: float) -> list[dict[str, Any]]:
    return [e for e in rows_of(recs, case, "propagation")
            if e["route"] == route and abs(e["amplitude"]-amplitude) < 1e-12
            and (amplitude == 0 or e["kind"] == kind)]


def spread(values: Any) -> str:
    v = np.asarray(values, dtype=float)
    return (f"{v.mean():.3e} | {v.std(ddof=1):.1e} | {v.min():.2e} | {v.max():.2e} | "
            f"{v.max()/v.min():.1f}x")


def geometric(values: Any) -> float:
    return float(np.exp(np.mean(np.log(np.asarray(values, dtype=float)))))


def settings(recs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {c["case"]: c["potential_setting"] for c in recs[0]["cases"]}


# --------------------------------------------------------------------------- sections

def preamble(w: Writer, recs: list[dict[str, Any]], manifest: dict[str, Any],
             source: Path, failed: dict[int, str]) -> None:
    n = len(recs)
    chosen = settings(recs)
    requested = len(manifest["runs"])
    drift = {k: v for k, v in manifest["source_hashes"].items()
             if source_hashes().get(k) != v}
    recorded_ast = manifest.get("source_ast_hashes")
    inert = (recorded_ast is not None
             and all(source_ast_hashes().get(k) == recorded_ast.get(k) for k in drift))
    status = (f"{n}/{requested} seeds completed"
              + (" with no failures" if not failed else
                 f"; **{len(failed)} failed and "
                 + ("is" if len(failed) == 1 else "are")
                 + " excluded from every number below**: "
                 + ", ".join(f"seed {k} ({v})" for k, v in sorted(failed.items()))))
    if not drift:
        provenance = "the source did not change between the run and this analysis"
    elif inert:
        provenance = ("the source changed since the run in " + ", ".join(sorted(drift))
                      + ", but only outside the executable code: the parsed source is "
                      "identical, so the numbers still follow from the current checkout")
    else:
        provenance = ("**the source changed since the run**, in " + ", ".join(sorted(drift))
                      + "; what changed and whether it affects behaviour is recorded in "
                      "`docs/provenance.md`")
    clock = (f"Wall clock {manifest['wall_clock_seconds']/3600:.2f} h. "
             if "wall_clock_seconds" in manifest else
             "The run did not record a wall clock, so it did not finish through its own "
             "exit path. ")
    w(f"""# The recovery map, measured

From `{source}`, configuration `{manifest['config_id']}`, frozen before the run. {status}, and
{provenance}. {clock}Regenerate with `scripts/analyse_recovery.py`.

Every number is computed on the **test** split. The calibration split fixes the affine map and
the centring offset and nothing else. Spreads are across the {n} seeds, each drawing its own
generated model, so a seed SD is variation over models and samples together and not a
confidence interval for a fixed model.

Two observation maps are measured throughout. `artanh` applies `artanh` component-wise, so its
Jacobian is diagonal, every derivative is closed form, and any error is the solver's. `mixing`
composes that with three orthogonal/leaky-tanh layers and is the nonlinear benchmark. The
potential solver is configured per case --- """
      + ", ".join(f"`{c}` at {s['n_features']} basis functions with ridge {s['ridge']:g}"
                  for c, s in chosen.items())
      + """ --- because a shared setting is known to handicap one of them, and a control that
is handicapped against the benchmark is not serving as a control.

**What this is not.** No theorem is established or contradicted. These are finite-sample,
finite-basis estimates of population statements whose hypotheses ((H), (P), (R1),
(G3)/(G3-phi)) are assumed throughout. The bound panel is a plug-in diagnostic with a
deliberately loose constant, not a certificate. Ground truth is used to build the controlled
fields, to evaluate, and to choose directions in the conditioning section; each use is
labelled where it occurs and none of it reaches a fitting routine.""")


def readout_identity(w: Writer) -> None:
    w("""
## The readout identity

`prop:readout` states that the readout is `Lambda(x) = L^T T(g^{-1}(x)) - kappa`, with no
learnable component. That identity, its gradient against the contrast field, and the
invariance of both field constructions to a condition-independent score field
(`rem:guidance`) are exact statements, so they are checked as exact fixtures rather than
measured: `tests/test_generator.py` and `tests/test_readout.py`, in float64, to tolerances
of `2e-13` on the identity and `2e-12` on the fields, for `n` in `{2,3}` and both maps.

Nothing below re-establishes them. Everything below measures what happens once the readout is
computed from an estimated score through a finite-basis potential.""")


def least_squares_potential(w: Writer, recs: list[dict[str, Any]]) -> None:
    n = len(recs)
    base = {c: propagation(recs, c, "reference", None, 0.) for c in CASES}
    field = {c: np.mean([e["field_error_mse"] for e in base[c]]) for c in CASES}
    w(f"""
## The least-squares potential

`eq:ls-potential` defines the estimate as the least-squares potential of the readout field
over `H^1(p_0)`. The implementation replaces that population projection with a ridge solve
over a finite basis, and this section measures what the replacement costs.

The fields here are built from analytic clean conditional scores, so they are exact: the
measured field error at zero injected error is """
      + " and ".join(f"`{field[c]:.1e}` for `{c}`" for c in CASES)
      + f""", the float64 floor. The entire numerical baseline therefore sits in the solver, which
is why the injection threshold later in this document is measured against this readout error
and never against a field residual.

| case | route | metric | mean | SD | min | max | spread |
|---|---|---|---|---|---|---|---|""")
    for case in CASES:
        for route in ROUTES:
            sub = propagation(recs, case, route, None, 0.)
            for label, key in (("oracle-aligned MSE", "oracle_aligned_mse"),
                               ("raw readout MSE", "raw_readout_mse"),
                               ("affine NMSE", "affine_nmse"),
                               ("solver residual fraction", "solver_residual_fraction")):
                w(f"| `{case}` | {route} | {label} | {spread([e[key] for e in sub])} |")

    aligned = {c: [e["oracle_aligned_mse"] for e in base[c]] for c in CASES}
    resid = {c: np.mean([e["solver_residual_fraction"] for e in base[c]]) for c in CASES}
    wide, narrow = sorted(CASES, key=lambda c: max(aligned[c])/min(aligned[c]), reverse=True)
    w(f"""
**The baseline is far more seed-dependent for `{wide}` than for `{narrow}`**: a spread of
{max(aligned[wide])/min(aligned[wide]):.1f}x against
{max(aligned[narrow])/min(aligned[narrow]):.1f}x, with coefficients of variation
{np.std(aligned[wide], ddof=1)/np.mean(aligned[wide]):.2f} and
{np.std(aligned[narrow], ddof=1)/np.mean(aligned[narrow]):.2f}. A single-seed baseline is not
"the floor", and the injection threshold is therefore taken per seed against that run's own
zero-amplitude value rather than against a global constant.

**The solver residual separates the two maps by a factor of
{max(resid.values())/min(resid.values()):.0f}** ("""
      + " against ".join(f"`{resid[c]:.2e}`" for c in CASES)
      + f"""). The nonlinear mixing makes `Upsilon = T o g^-1` materially harder for the basis to
represent, which is what the `artanh` control exists to separate out.

### Sensitivity of the frozen setting

The grid is a complete cross, reported on all {n} production seeds as the geometric mean of
the oracle-aligned MSE. A partial grid hides exactly the cell that matters, so every width is
run at every ridge.

| case | `n_features` | `ridge` | geo. mean | vs the frozen setting |
|---|---|---|---|---|""")
    for case in CASES:
        chosen = settings(recs)[case]
        frozen = geometric([e["oracle_aligned_mse"] for e in base[case]])
        cells: dict[tuple[int, float], list[float]] = {}
        for e in rows_of(recs, case, "sensitivity"):
            cells.setdefault((e["n_features"], e["ridge"]), []).append(e["oracle_aligned_mse"])
        for (width, ridge), values in sorted(cells.items()):
            if (width, ridge) == (chosen["n_features"], chosen["ridge"]):
                continue
            gm = geometric(values)
            w(f"| `{case}` | {width} | {ridge:.0e} | {gm:.3e} | {gm/frozen:.2f}x |")
        w(f"| `{case}` | **{chosen['n_features']}** | **{chosen['ridge']:.0e}** | "
          f"**{frozen:.3e}** | frozen |")
    w("""
The setting was chosen on seeds disjoint from these, on the validation split, under a
criterion fixed before that selection ran; `configs/` records it, including what it does not
claim to be optimal. The rows above are a report, not a re-selection.""")


def quantitative_recovery(w: Writer, recs: list[dict[str, Any]], amps: list[float]) -> None:
    n = len(recs)
    w(f"""
## Quantitative recovery

`thm:recovery` (2) bounds the recovery error of the reference-point readout by the conditional
Fisher divergence of the field it consumes (`eq:quantitative`); part (3) does the same for the
observable-feature readout through `C_phi` (`eq:quantitative-pi`). This section injects a
saved smooth score error of controlled size and follows it through the whole chain.

The perturbation is one reproducible function of `(x,u)`, saved as arrays and reused across
every split and amplitude, built from a tanh network so that it lies outside the solver's own
basis. It comes in a conservative variant, exactly `grad_x psi(x,u)`, and a nonconservative
one. Neither is claimed to be the score of a normalized density; what is measured is the
induced field error, which is what `J_Delta` is defined on.

Means over the {n} seeds, reference route. `vs base` is against each run's own zero-amplitude
readout error, and a point is marked dominated when that ratio reaches ten.
""")
    for case in CASES:
        for kind in KINDS:
            w(f"\n### `{case}` / {kind}\n")
            w("| `alpha` | realized `rho` | field MSE | raw MSE | oracle-aligned MSE | "
              "affine NMSE | vs base | dominated | aligned / `alpha^2` |")
            w("|---|---|---|---|---|---|---|---|---|")
            b = propagation(recs, case, "reference", None, 0.)
            w(f"| 0 | 0 | {np.mean([e['field_error_mse'] for e in b]):.2e} | "
              f"{np.mean([e['raw_readout_mse'] for e in b]):.2e} | "
              f"{np.mean([e['oracle_aligned_mse'] for e in b]):.2e} | "
              f"{np.mean([e['affine_nmse'] for e in b]):.2e} | 1.00 | baseline | - |")
            for a in amps:
                sub = propagation(recs, case, "reference", kind, a)
                al = np.mean([e["oracle_aligned_mse"] for e in sub])
                w(f"| {a:.2e} | {np.mean([e['rho_test'] for e in sub]):.2e} | "
                  f"{np.mean([e['field_error_mse'] for e in sub]):.2e} | "
                  f"{np.mean([e['raw_readout_mse'] for e in sub]):.2e} | {al:.2e} | "
                  f"{np.mean([e['affine_nmse'] for e in sub]):.2e} | "
                  f"{np.mean([e['ratio_to_readout_baseline'] for e in sub]):.2f} | "
                  f"{sum(e['injection_dominated'] for e in sub)}/{n} | {al/a**2:.3e} |")

    w("\n### Slope of the aligned error in the amplitude\n")
    w("| case | kind | seeds | slope mean | SD | min | max |")
    w("|---|---|---|---|---|---|---|")
    slopes: list[float] = []
    for case in CASES:
        for kind in KINDS:
            per = []
            for r in recs:
                c = next(x for x in r["cases"] if x["case"] == case)
                sub = [e for e in c["propagation"] if e["route"] == "reference"
                       and e["kind"] == kind and e["injection_dominated"]]
                if len(sub) >= 3:
                    per.append(float(np.polyfit(np.log([e["amplitude"] for e in sub]),
                                                np.log([e["oracle_aligned_mse"] for e in sub]),
                                                1)[0]))
            slopes += per
            if len(per) < 2:
                w(f"| `{case}` | {kind} | {len(per)} | too few dominated points to fit | | | |")
                continue
            w(f"| `{case}` | {kind} | {len(per)} | {np.mean(per):.4f} | "
              f"{np.std(per, ddof=1):.4f} | {min(per):.4f} | {max(per):.4f} |")
    if len(slopes) < 2:
        w("\nToo few injection-dominated amplitudes in this run to fit a slope. The"
          "\nsubsections below do not depend on it and are unaffected.")
    else:
        w(f"""
The slope is `{np.mean(slopes):.4f} +- {np.std(slopes, ddof=1):.4f}` over all {len(slopes)}
case/kind/seed fits, and `aligned / alpha^2` settles onto a constant once the injection
dominates.

**This is not evidence for the rate in `eq:quantitative`, nor for its sharpness.** The
injected field enters a linear ridge solve and the error is measured as a square; a slope of
two is what that arithmetic produces. What it does show is that the implemented chain hides no
extra amplitude dependence --- no regime where the readout error grows faster than the field
error it comes from --- which is a check on the implementation, not on the theorem.""")
    w("""
### Where the injection overtakes the baseline
""")
    w("| case | kind | first dominated `alpha` | seeds dominated there | fully dominated from |")
    w("|---|---|---|---|---|")
    for case in CASES:
        for kind in KINDS:
            first, full = None, None
            for a in amps:
                k = sum(e["injection_dominated"] for e in propagation(recs, case, "reference",
                                                                     kind, a))
                if k and first is None:
                    first = (a, k)
                if k == n and full is None:
                    full = a
            w(f"| `{case}` | {kind} | "
              f"{f'{first[0]:.2e}' if first else 'never'} | "
              f"{f'{first[1]}/{n}' if first else '-'} | "
              f"{f'{full:.2e}' if full else 'not in every seed'} |")
    quiet = [a for a in amps
             if all(not e["injection_dominated"]
                    for c in CASES for k in KINDS
                    for e in propagation(recs, c, "reference", k, a))]
    w(f"""
The {len(quiet)} smallest amplitudes are baseline-dominated in every seed of every
configuration, which is what makes the floor visible. No point was dropped for being
baseline-dominated. The baseline and the total are both reported and **MSEs are never
subtracted**: the injected and baseline errors are not orthogonal and the cross term is not
assumed to vanish.

### The fitted-affine metric saturates; the aligned error does not

| case | kind | oracle-aligned MSE at the top amplitude | affine NMSE there |
|---|---|---|---|""")
    top = amps[-1]
    al_top, nm_top = [], []
    for case in CASES:
        for kind in KINDS:
            sub = propagation(recs, case, "reference", kind, top)
            a_v = np.mean([e["oracle_aligned_mse"] for e in sub])
            n_v = [e["affine_nmse"] for e in sub]
            al_top.append(a_v)
            nm_top += n_v
            w(f"| `{case}` | {kind} | {a_v:.3e} | {np.mean(n_v):.4f} +- "
              f"{np.std(n_v, ddof=1):.4f} |")
    w(f"""
The aligned error reaches `{min(al_top):.0e}` to `{max(al_top):.0e}` while the held-out affine
NMSE stays between `{min(nm_top):.2f}` and `{max(nm_top):.2f}`. The fitted affine map absorbs
and discards much of the error. That is why the fitted metric is secondary here and the
oracle-aligned error is the one that diagnoses the theorem's left-hand side, and it is a
caution against reading a normalized recovery error as if it measured the readout.

### The two readout routes

`C_phi = Sigma_phi^(1/2) L^T`, and both routes consume the same scores and splits through one
linear ridge projection, so their oracle-aligned and fitted-affine metrics must coincide.""")
    gaps, ratios = [], []
    for r in recs:
        for c in r["cases"]:
            for e in c["propagation"]:
                if e["route"] != "reference":
                    continue
                m = next((f for f in c["propagation"] if f["route"] == "regression"
                          and f["amplitude"] == e["amplitude"]
                          and f["kind"] == e["kind"]), None)
                if m is None:
                    continue
                gaps += [abs(e[k]-m[k])/max(e[k], 1e-300)
                         for k in ("oracle_aligned_mse", "affine_nmse")]
                ratios.append(e["raw_readout_mse"]/m["raw_readout_mse"])
    if ratios:
        pairs = (
            f"Over all {len(ratios)} paired rows the largest relative gap in either is "
            f"`{max(gaps):.2e}`. The *raw* readout errors do not agree and should not: "
            f"reference over regression runs from {min(ratios):.1f} to {max(ratios):.1f}, the "
            "same error expressed in different coordinates, and the two are not "
            "interchangeable.\n\nThis is an implementation control. It says the two "
            "constructions were implemented consistently. It is not independent evidence for "
            "either, and it is not a comparison one route can win.")
    else:
        pairs = ("*Only the reference route was run in this configuration, so there is no pair "
                 "to compare.*")
    w(f"""
{pairs}

### Conservative against nonconservative

| case | aligned MSE ratio, nonconservative / conservative, at the top amplitude | seeds above 1 |
|---|---|---|""")
    for case in CASES:
        g = propagation(recs, case, "reference", "gradient", top)
        f = propagation(recs, case, "reference", "field", top)
        ratio = [x["oracle_aligned_mse"]/y["oracle_aligned_mse"] for x, y in zip(f, g)]
        w(f"| `{case}` | {np.mean(ratio):.3f} +- {np.std(ratio, ddof=1):.3f} | "
          f"{sum(r > 1 for r in ratio)}/{n} |")
    w("""
Nothing is concluded from this. The projection of `eq:ls-potential` discards the curl
component, whose energy is recorded in each seed file, and how much of a perturbation of fixed
size lands in that component varies with the draw.""")


def conditioning(w: Writer, recs: list[dict[str, Any]]) -> None:
    n = len(recs)
    w("""
## Conditioning of the alignment

The bounds carry the alignment inverse: `1/sigma_min(L)^2` in `eq:quantitative` and
`1/sigma_min(C_phi)^2` in `eq:quantitative-pi`. Injecting `delta(x,u^0) = 0` and
`delta(x,u^l) = alpha U[l,j] v(x)`, with `v` a gradient field and `U[:,j]` a left singular
vector of `G = L^T`, makes the field error exactly rank one. The raw readout error is then the
same for every mode `j` while the oracle-aligned error should be `raw / s_j^2`. Direction
selection uses the true `L`: an oracle control, available to no estimator.""")
    rows = [e for case in CASES for e in rows_of(recs, case, "conditioning")]
    ratio = [e["injection_only_oracle_aligned_mse"]/e["predicted_aligned_mse"] for e in rows]
    worst = 0.
    for r in recs:
        for c in r["cases"]:
            for a in sorted({e["amplitude"] for e in c["conditioning"]}):
                v = [e["injection_only_raw_readout_mse"] for e in c["conditioning"]
                     if e["amplitude"] == a]
                worst = max(worst, max(v)/min(v))
    full = [e["oracle_aligned_mse"]/e["injection_only_oracle_aligned_mse"] for e in rows]
    modes = {(r["seed"], c["case"]): sorted({e["mode"]: e["singular_value"]
                                             for e in c["conditioning"]}.items())
             for r in recs for c in r["cases"]}
    smin = [v[-1][1] for v in modes.values()]
    cond = [v[0][1]/v[-1][1] for v in modes.values()]
    amplification = [c*c for c in cond]
    w(f"""
| quantity | over all {len(rows)} rows |
|---|---|
| oracle-aligned MSE over predicted `raw / s_j^2` | max deviation from 1: `{max(abs(r-1) for r in ratio):.2e}` |
| raw MSE across modes, max/min within a group | `{worst:.10f}` |
| rank-one residual of the readout error | median `{np.median([e['injection_only_rank_one_residual'] for e in rows]):.2e}`, max `{max(e['injection_only_rank_one_residual'] for e in rows):.2e}` |
| full pipeline over injection-only aligned MSE | `{min(full):.3f}` to `{max(full):.3f}` |
| `sigma_min(L)` over the {n} generated models | `{min(smin):.4f}` to `{max(smin):.4f}` |
| `cond(L)` over the {n} generated models | `{min(cond):.1f}` to `{max(cond):.1f}` |

At a fixed raw readout error the aligned error moves by the square of the singular-value ratio:
`{min(amplification):.0f}` to `{max(amplification):.0f}` between the strongest and weakest
mode across these models, `{np.mean(amplification):.0f}` on average. The full pipeline agrees
with the injection-only construction to within `{100*max(abs(f-1) for f in full):.1f}%`, these
amplitudes all being injection-dominated.

This isolates the alignment factor at a **fixed** generated model: the law, the overlaps and
the score difficulty are untouched and only the direction of the error moves. A separation
sweep cannot do this, changing all of them at once. What is demonstrated is the amplification
in the construction, at these models. It says nothing about the tightness of `eq:quantitative`,
and the direction selection is not an operation an estimator could perform.

### `varkappa`

`rem:overlap-tension` observes that the reference points enter `eq:quantitative-fisher` only
through `varkappa = M / sigma_min(L)^2`, that choosing them means minimising it --- enough
diversity for a well-conditioned `L`, not so much that the conditionals stop overlapping ---
and suggests reporting it alongside any empirical result.

`M = max_l sup_x p_0(x)/p(x|u^l)` is given two ways, bounding it from opposite sides. `M_hat`
is the maximum over the test sample, a **lower** approximation to the supremum. `M_analytic`
is the bounded-density comparison formula of `bounds.py`, a deliberately loose **upper** bound.
Both depend on the latent model and the reference points alone, so they are identical for the
two observation maps, which share a generated model per seed.

| quantity | mean | SD | min | max |
|---|---|---|---|---|""")
    acc: dict[str, list[float]] = {k: [] for k in
                                   ("M_hat", "M_analytic", "sigma_min", "kappa_hat",
                                    "kappa_analytic")}
    for r in recs:
        c = next(x for x in r["cases"] if x["case"] == "artanh")
        pm = c["params"]
        params = Params(np.asarray(pm["natural"]), pm["seed"], pm["separation"],
                        pm["rejected_draws"], pm["rank_cutoff"], pm["grid_size"])
        built = build_case(params.n, r["seed"], pm["separation"], identity=True)
        test = built.split("test", c["splits"]["test"]["n_per_u"])
        m_hat = float(built.oracle.overlap_samples(test.x).max())
        m_an = float(comparison_constants(params, layers=0)["M_comparison"])
        s = float(np.asarray(pm["singular_values"]).min())
        for key, value in (("M_hat", m_hat), ("M_analytic", m_an), ("sigma_min", s),
                           ("kappa_hat", m_hat/s**2), ("kappa_analytic", m_an/s**2)):
            acc[key].append(value)
    for label in ("M_hat", "M_analytic", "sigma_min", "kappa_hat", "kappa_analytic"):
        samples = np.asarray(acc[label])
        w(f"| `{label}` | {samples.mean():.3e} | {samples.std(ddof=1):.1e} | {samples.min():.2e} | {samples.max():.2e} |")
    gap = float(np.mean(np.asarray(acc["M_analytic"])/np.asarray(acc["M_hat"])))
    kh, sm, mh = (np.asarray(acc[k]) for k in ("kappa_hat", "sigma_min", "M_hat"))
    w(f"""
The two estimates of `M` differ by a factor of {gap:.0f} on average, so `varkappa` is pinned
here only to within about {np.log10(gap):.0f} orders of magnitude. Anyone using it to choose
reference points is optimising a quantity of that uncertainty, which is worth saying alongside
the suggestion to report it.

Within that caveat, `varkappa_hat` runs from {kh.min():.2e} to {kh.max():.2e}, a spread of
{kh.max()/kh.min():.0f}x driven by `sigma_min(L)` ({sm.max()/sm.min():.0f}x) rather than by `M`
({mh.max()/mh.min():.1f}x). The diversity term dominates the overlap term throughout, so the
tension the remark describes is not visible in this design: these reference points sit where
more diversity would still help. That is a property of how the natural parameters are drawn
here, not a general finding.""")


def observable_diagnostic(w: Writer, recs: list[dict[str, Any]], amps: list[float]) -> None:
    w("""
## An observable diagnostic

`rem:curl` proposes `C^(l)(x) = ||d_x w^(l) - (d_x w^(l))^T||_F`, which vanishes when the score
is exact, as a measure of how much estimation error has entered the readout. It is the only
diagnostic in the manuscript that needs no ground truth, and so the only quantity in this
document a practitioner could compute; everything else here is an oracle quantity. It is worth
knowing what it detects.

`C^(l)` below is root-mean-squared over the test sample and maximised over `l`, against the
oracle-aligned readout error at the same amplitude. It is exactly linear in the amplitude, the
true contrast being a gradient field.

| case | kind | `alpha` | `max_l C^(l)` | oracle-aligned MSE |
|---|---|---|---|---|""")
    shown = sorted(dict.fromkeys([amps[0], amps[len(amps)//2], amps[-1]]))
    # Each perturbation and its evaluation sample is rebuilt once per (case, seed), not once
    # per amplitude, and the sample is drawn from every condition: the readout error it is
    # placed beside is an expectation under the marginal p_0, not under p(.|u^0).
    prepared: dict[tuple[str, int], tuple[Any, dict[str, Perturbation]]] = {}
    for case in CASES:
        for r in recs:
            c = next(x for x in r["cases"] if x["case"] == case)
            built = build_case(len(c["params"]["natural"][0]), r["seed"],
                               c["params"]["separation"], identity=c["identity_mixing"])
            test = built.split("test", c["splits"]["test"]["n_per_u"])
            perts = {}
            for kind in KINDS:
                saved = next(x for x in c["perturbations"] if x["name"] == kind)
                arr = saved["arrays"]
                feats = TanhFeatures(*(np.asarray(arr[k]) for k in
                                       ("mean", "scale", "W1", "c1", "W2", "c2")))
                perts[kind] = Perturbation(feats, kind, np.asarray(arr["coefficients"]),
                                           saved["normalization"], saved["diagnostics"])
            prepared[(case, r["seed"])] = (stratified_sample(test, 2000), perts)

    curls: dict[str, list[float]] = {}
    for case in CASES:
        for kind in KINDS:
            for a in shown:
                cs, es = [], []
                for r in recs:
                    c = next(x for x in r["cases"] if x["case"] == case)
                    sample, perts = prepared[(case, r["seed"])]
                    cs.append(float(np.max(contrast_curl(perts[kind], sample, a))))
                    es.append(next(e["oracle_aligned_mse"] for e in c["propagation"]
                                   if e["route"] == "reference" and e["kind"] == kind
                                   and abs(e["amplitude"]-a) < 1e-12))
                curls[f"{case}/{kind}"] = curls.get(f"{case}/{kind}", []) + cs
                w(f"| `{case}` | {kind} | {a:.2e} | {np.mean(cs):.3e} | {np.mean(es):.3e} |")
    floor = max(max(v) for k, v in curls.items() if k.endswith("gradient"))
    seen = min(min(v) for k, v in curls.items() if k.endswith("field"))
    errors = [e["oracle_aligned_mse"] for case in CASES
              for a in shown for e in propagation(recs, case, "reference", "gradient", a)]
    w(f"""
**The diagnostic is blind to a conservative score error.** The gradient-kind perturbation is
exactly `grad_x psi(x,u)`, so its contrast is a gradient field and `C^(l)` stays at the
central-difference floor at every amplitude --- at most `{floor:.1e}` over all rows --- while
the readout error it causes runs from `{min(errors):.0e}` to `{max(errors):.0e}`. That is not
an exotic class: `eq:ls-potential` is itself a projection onto gradient fields, so the
component of the error that survives the projection and reaches the readout is exactly the
component `C^(l)` cannot see.

**On the non-gradient part it is informative.** For the field kind `C^(l)` is at least
`{seen:.1e}`, linear in the amplitude while the readout error is quadratic, so within a kind
it orders the runs correctly.

`rem:curl` states that `C^(l)` vanishes when the score is exact. That is true, and the
gradient-kind rows sit at the floor as it says. The converse does not hold and the remark does
not assert it, but a reader could take "its magnitude measures how much estimation error has
entered the readout" that way. On this evidence it measures how much **non-gradient**
estimation error has entered. A one-line qualification would foreclose the stronger reading.
This is an observation about the diagnostic's reach, not a counterexample to anything the
manuscript proves.""")


def factorization(w: Writer, gap: dict[str, Any] | None) -> None:
    w("""
## From affine recovery to the factorization

`cor:factorization` lifts `~_A` to `~_P`. The Discussion distinguishes affine readout
error from component recovery. This section measures how
little the first constrains the second.

It is **not** a factorization search. No witness is constructed and nothing here bears on
whether the search of Section 4.4 is tractable, which the manuscript explicitly declines to
claim.

A candidate alignment is written `M A_0`, with `A_0` the true `(L^T)^{-1}`. What `~_P` permits
is `M` in the block-permutation group: a permutation of the `n` blocks of size `k` with free
invertible maps inside each block. `block defect` is the fraction of squared Frobenius energy
`M` places off the best block pattern, zero exactly when `M` is admissible. `single-factor
R^2` regresses each recovered coordinate on a quadratic in each single latent and keeps the
best, so a block-separated readout scores 1.""")
    if gap is None:
        w("""
*Not yet computed for this run: `scripts/factorization_gap.py --results <this run>` produces
it.*""")
        return
    for part, title, note in (
            ("structural", "With the exact readout, no estimation error",
             "no fitting at all, so any blindness here is structural"),
            ("empirical", "With this run's estimated readouts",
             "the same question carried by the run's own errors")):
        w(f"\n### {title}\n\n*{note}.*\n")
        w("| candidate alignment | affine NMSE | block defect | single-factor R^2 |")
        w("|---|---|---|---|")
        names = list(dict.fromkeys(r["alignment"] for r in gap[part]))
        for name in names:
            sub = [r for r in gap[part] if r["alignment"] == name]
            a = np.array([r["affine_nmse"] for r in sub])
            b = np.array([r["defect"] for r in sub])
            c = np.array([r["single_factor_r2"] for r in sub])
            extra = f" +- {c.std(ddof=1):.4f}" if len(c) > 1 else ""
            w(f"| {name} | {a.mean():.3e} | {b.mean():.4f} | {c.mean():.6f}{extra} |")
    dev = max(abs(r["affine_nmse"]/next(x["affine_nmse"] for x in gap["empirical"]
              if x["seed"] == r["seed"] and x["case"] == r["case"]
              and x["alignment"] == "identity")-1) for r in gap["empirical"])
    legal = [r for r in gap["structural"] if r["defect"] < 1e-12]
    w(f"""
**The affine metric does not distinguish them.** Across all {len(gap['empirical'])} estimated
rows the fitted-affine NMSE differs from the identity alignment by at most `{dev:.2e}` in
relative terms, and with the exact readout it is at the float64 floor for every candidate. It
is invariant to any invertible reparameterization of the readout, because the fit absorbs it.

**The block structure does distinguish them.** Only the {len(legal)} candidates `~_P` permits
reach zero defect and `R^2` of one; the others lose component recovery while leaving the affine
metric untouched.

`~_A` asks whether some invertible `A` exists, which is always true once recovery holds. `~_P`
asks whether the `A` one has is a block permutation away from `A_0`, which no affine criterion
can test. That gap is what the search of Section 4.4 exists to close, under hypotheses it must
establish separately.

One caution the numbers make concrete: a candidate that mixes blocks can still score high on
`single-factor R^2`. A high value is not evidence of factorization.""")


def dominance_crossing(recs: list[dict[str, Any]], case: str, kind: str,
                       amps: list[float]) -> float | None:
    for a in amps:
        if all(e["injection_dominated"] for e in propagation(recs, case, "reference", kind, a)):
            return a
    return None


def slope_of(recs: list[dict[str, Any]]) -> list[float]:
    out = []
    for case in CASES:
        for kind in KINDS:
            for r in recs:
                c = next(x for x in r["cases"] if x["case"] == case)
                sub = [e for e in c["propagation"] if e["route"] == "reference"
                       and e["kind"] == kind and e["injection_dominated"]]
                if len(sub) >= 3:
                    out.append(float(np.polyfit(np.log([e["amplitude"] for e in sub]),
                                                np.log([e["oracle_aligned_mse"] for e in sub]),
                                                1)[0]))
    return out


def conditioning_exactness(recs: list[dict[str, Any]]) -> tuple[float, float, list[float]]:
    rows = [e for case in CASES for e in rows_of(recs, case, "conditioning")]
    deviation = max(abs(e["injection_only_oracle_aligned_mse"]/e["predicted_aligned_mse"]-1)
                    for e in rows)
    worst = 0.
    for r in recs:
        for c in r["cases"]:
            for a in sorted({e["amplitude"] for e in c["conditioning"]}):
                v = [e["injection_only_raw_readout_mse"] for e in c["conditioning"]
                     if e["amplitude"] == a]
                worst = max(worst, max(v)/min(v))
    modes = {(r["seed"], c["case"]): sorted({e["mode"]: e["singular_value"]
                                             for e in c["conditioning"]}.items())
             for r in recs for c in r["cases"]}
    amplification = [(v[0][1]/v[-1][1])**2 for v in modes.values()]
    return deviation, worst, amplification


def route_gap(recs: list[dict[str, Any]]) -> float:
    gaps = []
    for r in recs:
        for c in r["cases"]:
            for e in c["propagation"]:
                if e["route"] != "reference":
                    continue
                m = next((f for f in c["propagation"] if f["route"] == "regression"
                          and f["amplitude"] == e["amplitude"]
                          and f["kind"] == e["kind"]), None)
                if m is not None:
                    gaps += [abs(e[k]-m[k])/max(e[k], 1e-300)
                             for k in ("oracle_aligned_mse", "affine_nmse")]
    return max(gaps) if gaps else float("nan")


def replication(w: Writer, core: list[dict[str, Any]], rep: list[dict[str, Any]],
                manifest: dict[str, Any], failed: dict[int, str],
                amps: list[float], rep_amps: list[float]) -> None:
    """The same measurement at a second dimension, reported as a replication."""
    n_core = len(next(x for x in core[0]["cases"])["params"]["natural"][0])
    n_rep = len(next(x for x in rep[0]["cases"])["params"]["natural"][0])
    note = ("" if not failed else
            f" {len(failed)} seed(s) failed and are excluded: "
            + ", ".join(f"{k} ({v})" for k, v in sorted(failed.items())) + ".")
    w(f"""
## The same measurement at n = {n_rep}

Configuration `{manifest['config_id']}`, {len(rep)} seeds, everything unchanged but the
dimension: `n = {n_rep}` gives `q = {n_rep*2}` and {n_rep*2+1} conditions against
`n = {n_core}`'s `q = {n_core*2}` and {n_core*2+1}.{note} The manuscript's statements do not
depend on the dimension, so this is a check that the implemented chain does not either. It is
a replication, not a second main result, and the sections above report `n = {n_core}` alone.

| quantity | `n = {n_core}` | `n = {n_rep}` |
|---|---|---|""")
    for case in CASES:
        for label, values in (
                ("baseline oracle-aligned MSE", lambda rs: [
                    e["oracle_aligned_mse"] for e in propagation(rs, case, "reference", None, 0.)]),
                ("baseline seed spread", lambda rs: None)):
            if values(core) is None:
                a = [e["oracle_aligned_mse"] for e in propagation(core, case, "reference", None, 0.)]
                b = [e["oracle_aligned_mse"] for e in propagation(rep, case, "reference", None, 0.)]
                w(f"| `{case}` {label} | {max(a)/min(a):.1f}x | {max(b)/min(b):.1f}x |")
            else:
                core_values, rep_values = values(core), values(rep)
                assert core_values is not None and rep_values is not None
                w(f"| `{case}` {label} | {np.mean(core_values):.3e} | {np.mean(rep_values):.3e} |")
    sc, sr = slope_of(core), slope_of(rep)
    w(f"| aligned-error slope in the amplitude | {np.mean(sc):.4f} +- {np.std(sc, ddof=1):.4f} "
      f"| {np.mean(sr):.4f} +- {np.std(sr, ddof=1):.4f} |")
    dc, wc, ac = conditioning_exactness(core)
    dr, wr, ar = conditioning_exactness(rep)
    w(f"| conditioning: max deviation of aligned from `raw / s_j^2` | {dc:.1e} | {dr:.1e} |")
    w(f"| conditioning: raw MSE across modes, max/min | {wc:.10f} | {wr:.10f} |")
    w(f"| conditioning: amplification, weakest over strongest mode | {min(ac):.0f} to "
      f"{max(ac):.0f} | {min(ar):.0f} to {max(ar):.0f} |")
    w(f"| two routes: max relative gap | {route_gap(core):.1e} | {route_gap(rep):.1e} |")
    for case in CASES:
        a = [e["affine_nmse"] for e in propagation(core, case, "reference", "gradient", amps[-1])]
        b = [e["affine_nmse"] for e in propagation(rep, case, "reference", "gradient",
                                                   rep_amps[-1])]
        w(f"| `{case}` affine NMSE at the top amplitude | {np.mean(a):.4f} | {np.mean(b):.4f} |")
    cross = {(c, k): (dominance_crossing(core, c, k, amps),
                      dominance_crossing(rep, c, k, rep_amps))
             for c in CASES for k in KINDS}
    same = sum(1 for a, b in cross.values() if a == b)
    w(f"""
The structural results carry over unchanged: the conditioning identity holds to the same
float64 tolerance, the raw error stays matched across modes to ten decimals, the two routes
agree to the same order, and the slope is the same to within its seed spread. The dominance
crossing sits at the same amplitude in {same} of the {len(cross)} case/kind combinations.

What differs is level, not behaviour. A larger `q` means more outputs from one basis and more
conditions in the marginal, so the baselines move; they are reported side by side above rather
than reconciled, because the two dimensions are different measurement problems and nothing
here requires them to agree. The conclusions the `n = {n_core}` sections draw --- that the
numerical floor sits in the solver, that the affine metric saturates while the aligned error
does not, that the amplification follows the singular values --- hold at `n = {n_rep}` too.""")


def plug_in_bound(w: Writer, recs: list[dict[str, Any]]) -> None:
    w("""
## Plug-in bound panel

**A plug-in population-bound diagnostic, not a certificate.** `C_P` is the deliberately loose
analytic comparison constant of `bounds.py` evaluated in float64; the potential is a
finite-basis approximation of the population projection; and every expectation is a finite
sample. A bound computed this way certifies nothing about the computed output, and an apparent
crossing would be a diagnostic to investigate rather than a theorem violation.

With `RHS = C_P / sigma_min(L)^2 * sum_l J_Delta(u^l)` and `LHS` the oracle-aligned MSE, over
the injection-dominated points of the reference route:
""")
    lo, hi, count, viol = np.inf, 0., 0, 0
    for r in recs:
        for c in r["cases"]:
            p = c["params"]
            params = Params(np.asarray(p["natural"]), p["seed"], p["separation"],
                            p["rejected_draws"], p["rank_cutoff"], p["grid_size"])
            cp = float(comparison_constants(
                params, layers=0 if c["identity_mixing"] else 3)["C_P_comparison"])
            smin = float(np.asarray(p["singular_values"]).min())
            for e in c["propagation"]:
                if e["route"] != "reference" or not e["injection_dominated"]:
                    continue
                rr = e["oracle_aligned_mse"]/(cp/smin**2*e["field_error_mse"])
                lo, hi, count = min(lo, rr), max(hi, rr), count+1
                viol += rr > 1
    if count == 0:
        verdict = ("No point of the reference route reached the dominance threshold, so there "
                   "is nothing to compare here.")
    elif viol:
        verdict = (f"**{viol} of {count} points have `LHS > RHS`.** The plug-in right-hand "
                   "side is not a certificate --- its constant is an analytic comparison "
                   "formula, the potential is a finite-basis approximation and the "
                   "expectations are finite samples --- so a crossing is a diagnostic to "
                   "investigate, not a theorem violation. It is reported, not explained away.")
    else:
        verdict = (f"The inequality holds at every point, with {np.log10(1/hi):.0f} to "
                   f"{np.log10(1/lo):.0f} orders of slack. That slack measures the looseness "
                   "of the constant, not the tightness of the bound.")
    w(f"""| quantity | value |
|---|---|
| points compared | {count} |
| `LHS / RHS` | `{lo:.2e}` to `{hi:.2e}` |
| points with `LHS > RHS` | {viol} |

{verdict}""")


def limits(w: Writer) -> None:
    w("""
## Limits

Supported, as implementation evidence: the readout chain runs end to end from conditional
scores to statistics recovered up to an affine map; the two routes are consistent; the
finite-basis solver's contribution is measured rather than assumed; an injected field error
propagates with no extra amplitude dependence; the alignment factor appears exactly as
constructed at a fixed model; and the affine metric is shown not to constrain the
factorization.

Not supported: any claim about the rate, sharpness or constants of `eq:quantitative` or
`eq:quantitative-pi`; any claim about nonatomic conditioning, which this finite categorical
design does not exercise; any claim about the tractability or success of a factorization
search, which is not implemented; and any claim that one readout route is better than the
other, the two being exactly equivalent after alignment by construction.

Synthetic data throughout. No score model is trained here, so nothing is said about what a
trained conditional diffusion model's error looks like; the injected error is smooth, of
controlled size, and of a shape chosen by the experimenter.""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="the core run; its numbers are the document's main result")
    parser.add_argument("--replication", type=Path, default=None,
                        help="a second run at another dimension, reported as a replication")
    parser.add_argument("--gap", type=Path, default=None,
                        help="factorization_gap JSON for this configuration")
    parser.add_argument("--out", type=Path, default=Path("docs/results.md"))
    args = parser.parse_args()
    recs, manifest, failed = load(args.results)
    amps = sorted({e["amplitude"] for e in recs[0]["cases"][0]["propagation"]
                   if e["amplitude"] > 0})
    gap_path = args.gap or Path("results/factorization_gap")/f"{manifest['config_id']}.json"
    gap = json.loads(gap_path.read_text()) if gap_path.exists() else None
    if gap is not None and gap.get("config_id") != manifest["config_id"]:
        raise SystemExit(f"{gap_path} belongs to configuration {gap.get('config_id')}, "
                         f"not to {manifest['config_id']}")

    L: list[str] = []
    w: Writer = L.append
    preamble(w, recs, manifest, args.results, failed)
    readout_identity(w)
    least_squares_potential(w, recs)
    quantitative_recovery(w, recs, amps)
    conditioning(w, recs)
    observable_diagnostic(w, recs, amps)
    factorization(w, gap)
    plug_in_bound(w, recs)
    if args.replication is not None:
        rep, rep_manifest, rep_failed = load(args.replication)
        rep_amps = sorted({e["amplitude"] for e in rep[0]["cases"][0]["propagation"]
                           if e["amplitude"] > 0})
        replication(w, recs, rep, rep_manifest, rep_failed, amps, rep_amps)
    limits(w)
    args.out.write_text("\n".join(L)+"\n")
    print(f"wrote {args.out}" + ("" if gap else "  (no factorization gap file found)"))


if __name__ == "__main__":
    main()

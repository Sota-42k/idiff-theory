"""Five figures following the construction, learned estimates, and diagnostic limits."""

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np

from recovery_core.perturbation import Perturbation, TanhFeatures, contrast_curl
from recovery_core.recovery import build_case, stratified_sample

COLORS = {"gradient": "#8051A0", "field": "#007A9A", 64: "#0072B2", 512: "#D55E00"}
NAMES = ("main_recovery", "supp_controlled", "supp_potential", "supp_diagnostics", "supp_factorization")


def prepare(bundle: dict[str, Any]) -> dict[str, Any]:
    """Keep every plotted value with its seed, case and measurement definition."""
    data: dict[str, Any] = {"run_id": bundle["index"]["run_id"],
                            "propagation": [], "sensitivity": [], "conditioning": [],
                            "curl": [], "estimators": [], "factorization": []}
    for n, records in bundle["recovery"].items():
        for record in records:
            for case in record["cases"]:
                prefix = {"n": n, "seed": record["seed"], "case": case["case"]}
                for section in ("propagation", "sensitivity", "conditioning"):
                    data[section].extend([{**prefix, **row} for row in case[section]])
                built = build_case(n, record["seed"], case["params"]["separation"],
                                   identity=case["identity_mixing"])
                sample = stratified_sample(built.split("test", case["splits"]["test"]["n_per_u"]), 2000)
                for saved in case["perturbations"]:
                    arrays = saved["arrays"]
                    features = TanhFeatures(*(np.asarray(arrays[k]) for k in
                                              ("mean", "scale", "W1", "c1", "W2", "c2")))
                    perturbation = Perturbation(features, saved["name"],
                                                np.asarray(arrays["coefficients"]),
                                                saved["normalization"], saved["diagnostics"])
                    amplitudes = sorted({row["amplitude"] for row in case["propagation"]
                                         if row["amplitude"] > 0})
                    for amplitude in amplitudes:
                        data["curl"].append({**prefix, "kind": saved["name"],
                                             "amplitude": amplitude,
                                             "max_contrast_curl": float(np.max(contrast_curl(
                                                 perturbation, sample, amplitude))),
                                             "samples": len(sample), "difference_step": 1e-5})
        gap = bundle["gaps"][n]
        for section in ("structural", "empirical"):
            data["factorization"].extend([{"n": n, "section": section, **v} for v in gap[section]])
    for record in bundle["estimators"]:
        data["estimators"].extend([{"seed": record["seed"], **v} for v in record["rows"]])
    data["selection"] = bundle["selection"]
    return data


def select(rows: list[dict[str, Any]], **match: Any) -> list[dict[str, Any]]:
    return [r for r in rows if all(r.get(k) == v for k, v in match.items())]


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.hashsalt": "recovery-figures-v1",
                         "savefig.facecolor": "white"})


def save(fig: Any, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def propagation_panel(ax: Any, data: dict[str, Any], n: int, case: str) -> None:
    for kind in ("gradient", "field"):
        rows = select(data["propagation"], n=n, case=case, kind=kind, route="reference")
        xs, ys = [], []
        for seed in sorted({v["seed"] for v in rows}):
            values = sorted(select(rows, seed=seed), key=lambda r: r["amplitude"])
            x = [v["field_error_mse"] for v in values]
            y = [v["oracle_aligned_mse"] for v in values]
            ax.plot(x, y, color=COLORS[kind], alpha=.20, lw=.7)
            xs.append(x)
            ys.append(y)
        ax.plot(np.mean(xs, axis=0), np.mean(ys, axis=0), "o-", ms=3, lw=1.8,
                color=COLORS[kind], label=f"{kind}: seed mean")
    baseline = select(data["propagation"], n=n, case=case, amplitude=0.0, route="reference")
    ax.axhline(np.mean([v["oracle_aligned_mse"] for v in baseline]), ls=":", lw=1,
               color="#555555", label="zero-injection mean")
    ax.set(xscale="log", yscale="log", xlabel="Contrast field MSE",
           ylabel="Oracle-aligned readout MSE")
    ax.grid(alpha=.15)


def main_figure(data: dict[str, Any], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.0), layout="constrained")
    propagation_panel(axes[0], data, 2, "mixing")
    axes[0].set_title("(a) Controlled score errors")
    axes[0].legend(loc="upper left", frameon=False)
    ax = axes[1]
    updates = [5000, 10000, 20000, 40000]
    for width in (64, 512):
        rows = select(data["estimators"], estimator="dsm", width=width)
        scores = []
        for seed in sorted({v["seed"] for v in rows}):
            values = sorted(select(rows, seed=seed), key=lambda r: r["update"])
            y = [1-v["affine_nmse"] for v in values]
            ax.plot(np.array(updates)/1000, y, "o-", color=COLORS[width],
                    lw=.65, ms=2, alpha=.23)
            scores.append(y)
        ax.plot(np.array(updates)/1000, np.mean(scores, axis=0), "o-", ms=4,
                color=COLORS[width], lw=2, label=f"DSM, width {width}")
        classifier_rows = select(data["estimators"], estimator="logodds", width=width,
                                 route_kind="through_the_potential")
        classifier_scores = []
        for seed in sorted({v["seed"] for v in classifier_rows}):
            values = sorted(select(classifier_rows, seed=seed), key=lambda r: r["update"])
            y = [1-v["affine_nmse"] for v in values]
            ax.plot(np.array(updates)/1000, y, "s--", color=COLORS[width],
                    lw=.5, ms=2, alpha=.18)
            classifier_scores.append(y)
        ax.plot(np.array(updates)/1000, np.mean(classifier_scores, axis=0), "s--", ms=4,
                color=COLORS[width], lw=1.6, label=f"Classifier, width {width}")
    ax.set(xlabel="Training updates (thousands)", ylabel="1 − affine NMSE",
           xticks=[5, 10, 20, 40], xlim=(3, 43), ylim=(0, 1.02),
           title="(b) Learned estimators")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(alpha=.15)
    save(fig, out, NAMES[0])


def controlled_figure(data: dict[str, Any], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7), layout="constrained")
    for row, n in enumerate((2, 3)):
        for col, case in enumerate(("artanh", "mixing")):
            propagation_panel(axes[row, col], data, n, case)
            axes[row, col].set_title(f"n = {n}, {case}")
    axes[0, 0].legend(frameon=False)
    save(fig, out, NAMES[1])


def potential_figure(data: dict[str, Any], out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.1), layout="constrained")
    widths, ridges = [1024, 2048, 4096], [1e-9, 1e-8, 1e-7]
    matrices = []
    for case in ("artanh", "mixing"):
        for n in (0, 2, 3):
            rows = (select(data["selection"]["rows"], case=case) if n == 0
                    else select(data["sensitivity"], case=case, n=n))
            matrix = np.array([[np.exp(np.mean(np.log([v["oracle_aligned_mse"] for v in
                                select(rows, n_features=w, ridge=r)]))) for r in ridges] for w in widths])
            matrices.append(matrix)
    norm = LogNorm(vmin=min(float(m.min()) for m in matrices),
                   vmax=max(float(m.max()) for m in matrices))
    for i, matrix in enumerate(matrices):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        im = ax.imshow(matrix, norm=norm, cmap="cividis", aspect="auto")
        for y in range(3):
            for x in range(3):
                ax.text(x, y, f"{matrix[y, x]:.1e}", ha="center", va="center", fontsize=8,
                        color="white" if float(norm(matrix[y, x])) < .5 else "black")
        setting = data["selection"]["selection"][("artanh", "mixing")[row]]
        chosen = (ridges.index(setting["ridge"]), widths.index(setting["n_features"]))
        ax.add_patch(Rectangle((chosen[0]-.47, chosen[1]-.47), .94, .94,
                               fill=False, edgecolor="#FF5B52", lw=2))
        ax.set(xticks=range(3), xticklabels=["1e−9", "1e−8", "1e−7"],
               yticks=range(3), yticklabels=widths, xlabel="Ridge", ylabel="Basis features")
        ax.set_title(f"{('artanh', 'mixing')[row]}: " +
                     ("selection validation, n=2", "production test, n=2", "production test, n=3")[col])
    fig.colorbar(im, ax=axes, shrink=.8, label="Geometric mean oracle-aligned MSE")
    save(fig, out, NAMES[2])


def diagnostics_figure(data: dict[str, Any], out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6), layout="constrained")
    for row, n in enumerate((2, 3)):
        ax = axes[row, 0]
        rows = select(data["conditioning"], n=n, case="mixing", amplitude=1.0)
        for seed in sorted({v["seed"] for v in rows}):
            values = sorted(select(rows, seed=seed), key=lambda v: v["singular_value"])
            x = np.array([v["singular_value"] for v in values])
            y = np.array([v["injection_only_oracle_aligned_mse"] /
                          v["injection_only_raw_readout_mse"] for v in values])
            ax.plot(x, y, "o-", ms=3, lw=.7, alpha=.45, color=COLORS[64])
        grid = np.geomspace(min(v["singular_value"] for v in rows), max(v["singular_value"] for v in rows), 100)
        ax.plot(grid, grid**-2, "--", color="#333333", lw=1, label=r"$1/s_j^2$")
        ax.set(xscale="log", yscale="log", xlabel=r"Singular value $s_j$",
               ylabel="Aligned / raw injection MSE", title=f"n = {n}: oracle directions")
        ax.legend(frameon=False)
        for kind in ("gradient", "field"):
            for col, section, metric in [(1, "curl", "max_contrast_curl"),
                                          (2, "propagation", "oracle_aligned_mse")]:
                values = select(data[section], n=n, case="mixing", kind=kind)
                if section == "propagation":
                    values = select(values, route="reference")
                ys = []
                for seed in sorted({v["seed"] for v in values}):
                    sub = sorted(select(values, seed=seed), key=lambda v: v["amplitude"])
                    amplitudes = [v["amplitude"] for v in sub]
                    measured = [v[metric] for v in sub]
                    axes[row, col].plot(amplitudes, measured, color=COLORS[kind], alpha=.20, lw=.6)
                    ys.append(measured)
                axes[row, col].plot(amplitudes, np.mean(ys, axis=0), "o-", ms=3, color=COLORS[kind], label=kind)
                axes[row, col].set(xscale="log", yscale="log", xlabel="Injected amplitude")
        axes[row, 1].set(ylabel=r"$\max_l C^{(l)}$ (RMS contrast curl)", title=f"n = {n}: observable diagnostic")
        axes[row, 2].set(ylabel="Oracle-aligned readout MSE", title=f"n = {n}: error that remains")
        axes[row, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.grid(alpha=.15)
    save(fig, out, NAMES[3])


def factorization_figure(data: dict[str, Any], out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6), layout="constrained")
    names = ["identity", "random invertible 1", "random invertible 2", "random invertible 3",
             "block permutation", "block-mixing shear"]
    labels = ["Identity", "Random 1", "Random 2", "Random 3", "Block perm.", "Shear"]
    for row, n in enumerate((2, 3)):
        exact = select(data["factorization"], n=n, section="structural")
        measured = select(data["factorization"], n=n, section="empirical", case="mixing")
        for col, (metric, title) in enumerate([("affine_nmse", "Affine NMSE"),
                                              ("defect", "Block defect"),
                                              ("single_factor_r2", "Single-factor R²")]):
            ax = axes[row, col]
            for i, name in enumerate(names):
                values = select(measured, alignment=name)
                y = [v[metric] for v in values]
                ax.scatter(i+np.linspace(-.1, .1, len(y)), y, s=12, color=COLORS[64], alpha=.35)
                ax.scatter(i, np.mean(y), marker="D", s=25, color=COLORS[64],
                           label="Estimated: mean / seed points" if i == 0 else None)
                ax.scatter(i, select(exact, alignment=name)[0][metric], marker="o", s=32,
                           facecolors="none", edgecolors="#222222", label="Exact: structural fixture" if i == 0 else None)
            ax.set(title=f"n = {n}: {title}", xticks=range(6), xticklabels=labels)
            ax.tick_params(axis="x", labelrotation=30, labelsize=7)
            if col == 0:
                ax.set_yscale("log")
            else:
                ax.set_ylim(-.03, 1.06)
            ax.grid(axis="y", alpha=.15)
        axes[row, 0].legend(frameon=False, loc="center left", fontsize=6.8)
    save(fig, out, NAMES[4])


def generate(bundle: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    style()
    data = prepare(bundle)
    # Timings are not plotted and would obscure the portable numerical table.
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if not k.endswith("seconds")}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    (out / "data.json").write_text(json.dumps(clean(data), indent=2, allow_nan=False)+"\n")
    for draw in (main_figure, controlled_figure, potential_figure, diagnostics_figure, factorization_figure):
        draw(data, out)

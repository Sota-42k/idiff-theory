"""Configuration, provenance and result files for a the recovery-map experiments production run.

Small and explicit on purpose: load a YAML config, validate it, hash it, capture the
environment and the source, and write one JSON per seed. No queue, no callbacks, no
checkpoint orchestration.

The `config_id` is a content hash of the configuration with its own identity fields removed,
so a run cannot silently change a setting and keep the directory. `validate` refuses a
production config that is not marked frozen: design Section 8 requires the freeze first.
"""

import ast
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import yaml

from recovery_core.perturbation import Perturbation
from recovery_core.recovery import ROUTES
from recovery_core.readout import Potential

SOURCE_FILES = ("generator.py", "readout.py", "perturbation.py", "recovery.py",
                "metrics.py", "directions.py", "bounds.py", "types.py", "runs.py")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def content_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()[:16]


def source_hashes() -> dict[str, str]:
    """Hash the modules a result depends on, so a later edit cannot pass unnoticed."""
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in SOURCE_FILES if (root/name).exists()}


def source_ast_hashes() -> dict[str, str]:
    """Hash the parsed source, which ignores comments and formatting.

    A file hash alone cannot distinguish an edited comment from an edited computation, so a
    reader who finds drift has to go and look. Recording both lets the question be answered
    from the record: equal ASTs and unequal files means nothing executable moved.
    """
    root = Path(__file__).resolve().parent
    out = {}
    for name in SOURCE_FILES:
        path = root/name
        if path.exists():
            dumped = ast.dump(ast.parse(path.read_text()))
            out[name] = hashlib.sha256(dumped.encode()).hexdigest()
    return out


def environment() -> dict[str, Any]:
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(),
            "machine": platform.machine(), "dtype": "float64"}


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"{path} does not contain a mapping")
    validate(config)
    return config


def validate(config: dict[str, Any]) -> None:
    """Fail loudly rather than run a configuration nobody has frozen or checked."""
    stage = config.get("stage")
    if stage not in ("pilot", "production"):
        raise ValueError("stage must be 'pilot' or 'production'")
    if stage == "production" and config.get("frozen") is not True:
        raise ValueError("A production run requires frozen: true (design Section 8, step 4)")
    seeds = config.get("seeds")
    if not seeds or sorted(set(seeds)) != list(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError("seeds must be distinct, sorted integers")
    if config.get("n", 0) < 2 or config.get("k") != 2 or config.get("separation", 0) <= .1:
        raise ValueError("Require n >= 2, k = 2 and separation > 0.1")
    if not config.get("cases") or any(v not in (True, False) for v in config["cases"].values()):
        raise ValueError("cases must map a name to the identity-mixing flag")
    if not config.get("routes") or any(r not in ROUTES for r in config["routes"]):
        raise ValueError(f"routes must be drawn from {ROUTES}")
    sizes = config.get("sizes", {})
    required = ("fitting", "calibration", "validation", "pert_calibration", "test")
    if any(type(sizes.get(key)) is not int or sizes[key] < 2 for key in required):
        raise ValueError(f"sizes must give an integer >= 2 for each of {required}")
    potential = config.get("potential", {})
    if potential.get("basis_seed") is None or potential.get("chunk_size", 0) < 1:
        raise ValueError("potential needs basis_seed and chunk_size >= 1")
    # `potential_by_case` gives the solver setting per case, because a single setting is known
    # to handicap one of them and a control handicapped against the benchmark is not a
    # control. A config without it carries one shared setting in `potential`; that form is
    # kept so the superseded runs stay readable and reproducible, not for new runs.
    for name, setting in solver_settings(config).items():
        if setting.get("n_features", 0) < 1 or not 0 < setting.get("ridge", 0):
            raise ValueError(f"the solver setting for {name} needs n_features >= 1, ridge > 0")
    perturbation = config.get("perturbation", {})
    amplitudes = perturbation.get("amplitudes")
    if not amplitudes or amplitudes[0] != 0 or sorted(amplitudes) != list(amplitudes):
        raise ValueError("amplitudes must start at 0 and increase; 0 is the potential-baseline baseline")
    if any(kind not in ("gradient", "field") for kind in perturbation.get("kinds", [])):
        raise ValueError("perturbation kinds must be 'gradient' or 'field'")
    directional = config.get("directional", {})
    if not directional.get("amplitudes") or any(a <= 0 for a in directional["amplitudes"]):
        raise ValueError("alignment-conditioning needs positive amplitudes")
    if not set(directional["amplitudes"]) <= set(amplitudes):
        raise ValueError("alignment-conditioning amplitudes must be drawn from the error-propagation grid")


def solver_settings(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve the potential-solver setting for each case, per case or shared."""
    shared = config.get("potential", {})
    by_case = config.get("potential_by_case")
    if by_case is None:
        return {name: dict(shared) for name in config["cases"]}
    if set(by_case) != set(config["cases"]):
        raise ValueError("potential_by_case must cover exactly the configured cases")
    return {name: {**shared, **by_case[name]} for name in config["cases"]}


def config_id(config: dict[str, Any]) -> str:
    """Identity fields are excluded so a rename does not look like a different run."""
    return content_hash({k: v for k, v in config.items()
                         if k not in ("name", "output", "notes")})


def provenance(config: dict[str, Any]) -> dict[str, Any]:
    return {"config": jsonable(config), "config_id": config_id(config),
            "environment": environment(), "source_hashes": source_hashes(),
            "source_ast_hashes": source_ast_hashes()}


def solver_record(potential: Potential) -> dict[str, Any]:
    keep = ("method", "basis_width", "ridge", "seed", "n_fourier_features", "gram_condition",
            "unregularized_rank", "normal_equation_residual", "status", "normalization",
            "stacked_outputs", "output_slice", "fit_seconds")
    return {k: jsonable(v) for k, v in potential.diagnostics.items() if k in keep}


def perturbation_record(perturbation: Perturbation, name: str) -> dict[str, Any]:
    return {"name": name, "normalization": perturbation.normalization,
            "diagnostics": jsonable(perturbation.diagnostics),
            "arrays": jsonable(perturbation.save_arrays())}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=1)+"\n")

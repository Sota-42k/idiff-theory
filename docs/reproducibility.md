# Reproducing the published measurements

## Saved-result reproduction

The selected dataset is `rerun-20260926T121833Z`. All eight execution stages completed
between 2026-09-26 12:20:20 UTC and 2026-09-27 11:25:54 UTC (23 h 05 min 34 s).
This is the sequential rerun cost; it does not include earlier exploration or development.
Per-stage timestamps, original configuration IDs and SHA-256 file hashes are in
`data/manifest.json`. Per-seed manifests retain environment, timing and failure status.
Training used CPU torch with float32 networks; the numerical construction uses float64.

From the repository root, install the exact lock and reproduce without training.
`.python-version` selects Python 3.12.2, matching the recorded environment:

```sh
uv sync --locked --extra estimators --extra plot --extra dev
PYTHONPATH=src uv run --locked python scripts/reproduce.py
PYTHONPATH=src uv run --locked python scripts/reproduce.py --check
```

The first command installs dependencies. The second validates every seed, case, grid and
checkpoint, regenerates `docs/results.md`, `docs/estimators.md`, five PDF/SVG/PNG figures,
`figures/data.json` and `figures/manifest.json`. The last command independently regenerates
into a temporary directory and compares bytes without altering the published files.
`--out /path/to/new-directory` writes a separate copy of generated outputs.

No trained weights are needed for these figures: measurements at the declared checkpoints
were saved. The contrast-curl diagnostic is recalculated from saved perturbations and the
reconstructed deterministic test split. This is analysis, not estimator retraining.

The bundle contains all ten estimator seeds and ten seeds per controlled dimension, including
all cases. Final DSM evaluation hashes match the final training hashes. The original bug
repeated 20k under the 40k label; these published records are the corrected full rerun.

A separate supplement, `data/estimators-checkpoints/52ed7c149f7cd299/` (run ID
`rerun-20260929T192042Z`, recorded as `checkpoint_run_id` in `data/manifest.json`), adds
classifier (`logodds`) evaluations at 5k/10k/20k updates from a full replay of all ten seeds
and both widths. Each seed's replayed 40k state hash and metrics are verified against the
original base record before publication; the 38 original files are byte-unchanged, and the
supplement is purely additive (`load_bundle` merges it in, never overwrites a base row).

## Provenance and portability

`data/` contains 38 byte-identical files copied from the selected rerun, plus the 11-file
classifier-checkpoint supplement above and the publication manifest. Files are explicitly
enumerated; extra, missing, modified or incomplete records fail validation. The configuration
ID alone is insufficient to distinguish old and corrected runs; use the run ID and file
hashes. Current numerical/training modules and runner hashes are checked against the
execution snapshot, including the four supplementary runner modules
(`logodds_checkpoints.py`, `checkpoint_experiment.py`, `checkpoint_records.py`,
`scripts/run_classifier_checkpoints.py`). `src/recovery_estimators/report.py` renders both the
base and checkpoint rows and is itself hash-checked; its recorded hash was updated when the
new section was added. Rendering code under `src/recovery_reporting/` is not hash-locked; it
has its own hash record in `figures/manifest.json` and can change without rewriting
experimental measurements.

The original execution used Python 3.12.2, NumPy 2.5.3, SciPy 1.18.1 and torch 2.14.0 on
macOS arm64. Rendering versions are saved in the figure manifest. Reproduction was checked
byte-for-byte on the recorded local environment and in a separate exported checkout.
PDF timestamps and SVG random IDs are suppressed. Different platforms, BLAS libraries,
Python versions or font/rendering versions may change floating-point or rendered bytes;
`--check` deliberately reports this rather than certifying an unchecked match. The saved
numerical table and manifests allow the differences to be investigated.

## Rerunning the experiments

Use a new output root to preserve the published measurements. This is the expensive path:

```sh
PYTHONPATH=src uv run --locked python scripts/run_recovery.py --config configs/recovery-n2.yaml --output results/new-run
PYTHONPATH=src uv run --locked python scripts/run_recovery.py --config configs/recovery-n3.yaml --output results/new-run
PYTHONPATH=src uv run --locked python scripts/run_estimators.py --config configs/estimators.yaml --output results/new-run/estimators/52ed7c149f7cd299
PYTHONPATH=src uv run --locked python scripts/select_potential.py --config configs/recovery-n2.yaml --seeds 100 101 102 103 104 105 106 107 108 109 --output results/new-run/selection
PYTHONPATH=src uv run --locked python scripts/factorization_gap.py --results results/new-run/recovery/2ab29ae68a24d88b --out results/new-run/factorization_gap
PYTHONPATH=src uv run --locked python scripts/factorization_gap.py --results results/new-run/recovery/8dc577d359c32f4d --out results/new-run/factorization_gap
```

The classifier-checkpoint supplement replays only the classifier, from a completed base
estimator run, and never writes into that base directory:

```sh
PYTHONPATH=src uv run --locked python scripts/run_classifier_checkpoints.py --config configs/estimators.yaml --base data/estimators/52ed7c149f7cd299 --output results/new-run/classifier-checkpoints
```

It verifies the replayed 40k state hash and metrics against the base directory's published
final classifier rows before writing anything; a mismatched seed fails rather than being
silently accepted.

Inspect each manifest for all requested seeds/cases completed: recovery runner exit status
alone is not sufficient. Numerical settings stay frozen. The selection command reproduces
independent-seed selection, and does not retune against production tests. Training on another
hardware/software stack can change weights and outputs; no cross-platform bitwise training
claim is made. Source, splits, seeds and perturbation arrays are recorded to make differences
traceable. Do not replace `data/` with a new run without a separately reviewed provenance update.

## Export and checks

```sh
PYTHONPATH=src uv run --locked pytest
PYTHONPATH=src uv run --locked mypy
PYTHONPATH=src uv run --locked python scripts/make_publication_mirror.py --check
PYTHONPATH=src uv run --locked python scripts/make_publication_mirror.py --out ../recovery-publication
```

Export refuses an existing destination. It includes canonical data, source, frozen configs,
license/provenance, reports, figures and regeneration tools, and excludes working results and
development history. Inside the export, use the same install and `reproduce.py --check`
commands. The export is standalone and does not import a parent checkout or original rerun.

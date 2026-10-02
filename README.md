# The recovery map, measured

Companion code for *Identifiability of Diffusion Models: The Conservation Law
and A Recovery Map*. It illustrates numerically the construction of Section 4:
a conditional score estimate is turned into the latent sufficient statistics,
up to an affine transformation, without fitting an inverse mixing map. Numerical potentials use a finite-basis ridge fit.

Everything here is an **illustration of a construction and of its limits**. No
theorem is established or contradicted, no claim is made about the rate,
sharpness or constants of any bound, and every quantity is a finite-sample,
finite-basis estimate of a population statement whose hypotheses are assumed.

## What is measured, and where to find it

| The manuscript says | What is measured | Where |
|---|---|---|
| `prop:readout`, `eq:main-identity`: the readout is `L^T T(g^{-1}(x)) - kappa` | the identity and its gradient, exactly, in float64 | [`tests/test_generator.py`](tests/test_generator.py) |
| `rem:guidance`: the readout field is a difference of guidance directions | both field constructions are unchanged by a condition-independent score field | [`tests/test_readout.py`](tests/test_readout.py) |
| `eq:ls-potential`: the estimate is the least-squares potential of the field | what replacing the population projection by a finite-basis ridge solve costs | [`docs/results.md`](docs/results.md), *The least-squares potential* |
| `thm:recovery` (2), `eq:quantitative`: the reference-point readout | the chain from an injected score error to the aligned readout error | [`docs/results.md`](docs/results.md), *Quantitative recovery* |
| `thm:recovery` (3), `eq:quantitative-pi`: the observable-feature readout | the same, through `C_phi`, on the same scores and splits | [`docs/results.md`](docs/results.md), *Quantitative recovery* |
| the `1/sigma_min(L)` factor | the amplification, isolated at a fixed generated model | [`docs/results.md`](docs/results.md), *Conditioning of the alignment* |
| `rem:overlap-tension`: report `varkappa = M / sigma_min(L)^2` | `varkappa`, bracketed from both sides | [`docs/results.md`](docs/results.md), *Conditioning of the alignment* |
| `rem:curl`: the antisymmetric part of the field Jacobian diagnoses score error | what that diagnostic detects, and what it does not | [`docs/results.md`](docs/results.md), *An observable diagnostic* |
| `cor:factorization`: lifting `~_A` to `~_P` | how little the affine readout error constrains component recovery | [`docs/results.md`](docs/results.md), *From affine recovery to the factorization* |
| `rem:guidance`: by Bayes the readout is the log-odds of the implicit classifier | what a trained model reaches, and by which route | [`docs/estimators.md`](docs/estimators.md) |

`docs/results.md` is written by the analysis step below. The controlled-score rows measure
the construction given scores it is handed; the last asks which objective produces
scores good enough for it, which is a different question and is answered separately.

A **factorization search** is not implemented. The manuscript does not claim the
search is tractable, and nothing here bears on whether it is.

## How it is set up

[`docs/experiment-design.md`](docs/experiment-design.md) gives the generative
model, the split and oracle boundaries, the metric conventions, and the choices
that were frozen before any number was reported. The short version:

- Bounded exponential-family latents on `(-1,1)` with `T_i(z) = (z, z^2)`,
  `k = 2`, `n = d`, and `nk+1` uniformly weighted conditions.
- Two observation maps: `g = artanh` component-wise, where every derivative is
  closed form and the error is attributable to the solver alone; and
  `h o artanh` with three orthogonal/leaky-tanh layers, a genuine nonlinear
  benchmark.
- Clean conditional scores at `t = 0`, perturbed by one saved smooth function of
  `(x, u)` whose amplitude is the controlled variable. Controlling the error is
  what makes propagation measurable at all: a learned error gives one point and
  no axis.
- For the last row of the table only, models are trained, to ask which objective
  reaches the construction. Torch is an optional extra required by this training
  and by the saved estimator-record validation used for publication reproduction.
- Ground truth is used to build the controlled fields, to evaluate, and to pick
  directions in the conditioning experiment. Each use is labelled in every
  record. It never reaches a fitting routine.

## Reproduce the published results and figures

The selected complete run is `rerun-20260926T121833Z`. The versioned `data/` directory
contains its 38 original result files, an additive 11-file classifier-checkpoint supplement
(`rerun-20260929T192042Z`, adding 5k/10k/20k classifier evaluations to match the existing DSM
checkpoints), and a manifest with hashes and provenance, including correct DSM 40k
evaluations. All ten seeds per experiment are present.

```sh
uv sync --locked --extra estimators --extra plot --extra dev
PYTHONPATH=src uv run --locked python scripts/reproduce.py
PYTHONPATH=src uv run --locked python scripts/reproduce.py --check
```

Run from this checkout. These commands regenerate the reports and five figures from saved
measurements; they do not train estimators. `--check` repeats the generation in a temporary
directory and verifies every output against the published version. Torch is currently needed
by estimator-record validation as well as training. `PYTHONPATH=src` selects this checkout's
code even on systems that ignore hidden editable-install `.pth` files.

- [Main figure and four supplement figures, with captions](docs/figures.md)
- [Reproduction, full experiment commands, compute and portability](docs/reproducibility.md)
- [Controlled-score results](docs/results.md) and [learned-estimator results](docs/estimators.md)
- [Raw dataset manifest](data/manifest.json) and [plotted measurements](figures/data.json)

Use `PYTHONPATH=src uv run --locked python scripts/make_publication_mirror.py --check` to
validate the standalone publication contents, and `--out ../recovery-publication` to export
to a new directory. Public data, figures and generation scripts are included together.

### Reading the code in experimental order

| Step | Module |
|---|---|
| Generated model, observed splits and exact readout | `recovery_core/generator.py`, `recovery_core/recovery.py` |
| Readout fields, numerical potential and metrics | `recovery_core/readout.py`, `recovery_core/metrics.py` |
| Train DSM, a classifier or a fixed-basis convex reference | `recovery_estimators/dsm.py`, `logodds.py`, `convex.py` |
| Connect observations, training and held-out evaluation | `recovery_estimators/experiment.py`, `evaluate.py` |
| Store/validate results and apply a final-DSM supplement | `recovery_estimators/records.py` |
| Replay and validate the classifier-checkpoint supplement | `recovery_estimators/checkpoint_records.py`, `checkpoint_experiment.py`, `logodds_checkpoints.py` |
| Summarize the measured results and their limits | `recovery_estimators/report.py` |
| Validate the publication bundle and draw figures | `recovery_reporting/data.py`, `figures.py` |

The two estimator scripts expose these steps as commands. Training settings remain in the
frozen YAML; no hyperparameter search or general experiment scheduler is introduced.

Experimental runners write new measurements under `results/`, which is ignored by Git.
Published measurements live in versioned `data/`; the reproduction workflow reads only that
selected dataset. Existing working experiments therefore cannot silently enter a figure.

`configs/recovery-n2.yaml` is the core and `configs/recovery-n3.yaml` replicates
it at `n = 3`. `configs/estimators.yaml` uses the same generated model, splits
and solver setting, so its diagnostics use the same units as the controlled-score measurements. Each
carries the frozen settings, how they were selected, and what the selection does
not claim.

`src/recovery_core/` is the construction and its measurement, in NumPy and
SciPy. `src/recovery_estimators/` trains the models that produce the scores the
construction consumes.

## Provenance and license

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Per-file origins,
content hashes, modification reasons and the numerical-equivalence record are in
[`docs/provenance.md`](docs/provenance.md).

The generator descends from TCL, https://github.com/hmorioka/TCL, pinned at
`e813b79399c6d9c5550822c740872f9df0c91871`, and is modified: bounded
exponential-family sources with `(z, z^2)` in place of scale-modulated Laplace
with `|z|`, seeded orthogonal matrices in place of condition-number-rejected
uniform ones, and leaky-tanh after `artanh` in place of leaky-ReLU. These are
research modifications, not the original TCL algorithm.

Please cite the upstream work when using the inherited generator protocol:

Aapo Hyvarinen and Hiroshi Morioka. "Unsupervised Feature Extraction by
Time-Contrastive Learning and Nonlinear ICA." NeurIPS 2016, pp. 3765-3773.

## AI Use Statement

During the preparation of this work, we used ChatGPT 5.5 Sol, ChatGPT 6 Astra, Claude Sonnet 5,
and Claude Opus 5 as research and coding assistants. For the code and experiments in this
repository, we used these tools to design the experiments, implement the experimental code, and
support analysis of the experiment results. We did not use these tools to generate or modify
datasets. All generated code was thoroughly verified, edited, and validated by the authors, who
take full responsibility for the content and integrity of this work.

# Provenance manifest

Every file in `src/recovery_core/` is listed. Hashes are sha256 of file contents.

## Sources

| Role | Identifier |
|---|---|
| Upstream | `hmorioka/TCL`, v1.0 (1 May 2018), Hiroshi Morioka, Apache-2.0 |
| Pinned upstream revision | `e813b79399c6d9c5550822c740872f9df0c91871` |
| Upstream file | `subfunc/generate_artificial_data.py`, sha256 `c2ca46cb57ef4bf53c0d7795ad4f511547b61c2db4940bec87370a919799cbc3` |
| Immediate source | `identifiability-research-1`, commit `b4cd012dc31a5305227ddd2c4f385e88fc651af3` |

The pinned upstream file was fetched and inspected on 2026-09-23. Its defaults were confirmed
directly in that file rather than assumed: `sourcetype='laplace'` with
`np.random.laplace(0, 1/sqrt(2), ...)`, `Ltype='uniform'` modulation, mixing matrices drawn by
`np.random.uniform` and rejected against a condition-number percentile (`cond_thresh_ratio`),
and a leaky-ReLU nonlinearity (`negative_slope = 0.2`).

The immediate-source files listed below were clean in that checkout's working tree at the
recorded commit; no dirty-source hashes are needed for them.

## Extracted files

| File | Immediate source | Source sha256 | TCL-derived | Change |
|---|---|---|---|---|
| `generator.py` | `src/idiff/data.py` | `9c73e9bc7538464ee5cad76bb6798efb67f3b3cbe23c4322b003659392a7ff7a` | yes | renamed; import path only |
| `readout.py` | `src/idiff/readout.py` | `f7fa9826a4b0f9433c0289948acb234d1110f9b6352a322017e688d61cca3c19` | no | import path; `rescaled_scores` and the VP schedule dropped |
| `metrics.py` | `src/idiff/metrics.py` | `ac08a1a2e5c7b02e2f53e197765cec4b2bacfea570b256f2d12c4438b45cd84e` | no | import path; `disc_threshold` dropped |
| `directions.py` | `src/idiff/directions.py` | `be388bf71ea97759286b2b43fc4dbb407c66f5a250f5abe2f28a3fde4bcb8ef2` | no | import path; `linear_recoverability` (CCA) dropped |
| `bounds.py` | `src/idiff/bounds.py` | `8dadd6237d0a5a53fbcc5a62ba0181773491f9ceec30c7bf79120cbaff0d15d9` | no | import path only |
| `types.py` | `src/idiff/types.py` | `dbeb051b8928a90d56eeeb6feffa6d37cbaee289f14780926db05319942d3482` | no | notice only |
| `perturbation.py` | — | — | no | **new**; no ancestor in either source |
| `__init__.py` | — | — | no | **new** |

Removals are scope reductions, not behaviour changes: the recovery-map experiments evaluates clean conditional
scores at `t = 0`, runs no discrete experiment, and needs only the singular-direction
accounting. The dropped code remains in the original repository.

## Modification reasons

**TCL-derived (`generator.py`).** The scientific modifications with respect to the pinned
upstream file are unchanged from the immediate source and are restated in the file header:
sources truncated to `(-1,1)`; the scale-modulated Laplace family (statistic `|z|`, `k=1`)
replaced by the exponential family with `T_i(z) = (z, z^2)`, `k=2`; condition-number-rejected
uniform mixing matrices replaced by fixed seeded orthogonal ones; leaky-ReLU replaced by
`chi(y) = tanh(y) + 0.1 y` applied after `artanh`. These are research modifications and are
not the original TCL algorithm. This extraction adds no further scientific change.

**Organizational only.** Every other change in the table is a rename, an import-path rewrite,
or a deletion of code outside the recovery-map experiments scope.

## Numerical equivalence

`docs/equivalence_check.py` compares the extracted modules against the immediate source on
matched fixtures: generator parameters, sampler draws, `g`/`g_inv`/`jac`/`jac_inv`/`logdet`/
`grad_logdet` for `n in {2,3}` and both `g = artanh` and the nonlinear mixing; oracle scores,
readout, contrast, statistic Jacobian, log-density and overlap samples; whitened features and
`C_phi`; both field constructions; a fitted potential and its gradient; the comparison
constants; the affine fit and its held-out residual; the oracle alignment error; and the SVD
direction convention.

Result, run on 2026-09-23 against commit `b4cd012dc31a5305227ddd2c4f385e88fc651af3`:
**115 comparisons, all bit-identical (`max|diff| = 0.0`), no failures.**

The script imports `idiff` and is therefore not part of the test suite: the rule that this
repository must not depend on the original checkout at runtime stands. It is kept under
`docs/` as the record of the check and is re-runnable only where both checkouts are present.
To rerun this optional historical comparison, pass the recorded immediate-source checkout
explicitly: `PYTHONPATH=src uv run python docs/equivalence_check.py --source-src /path/to/immediate-source/src`.
It is not used by publication-data, figure or test reproduction.

## Maintenance

When a derived file changes, re-compare against both sources, update this table and the
file header, and record whether the change is organizational or scientific. Do not silently
advance the pinned upstream revision and do not reintroduce the unused TCL classifier.

## Source drift during the production runs

The run records hash every module the numbers depend on, and the analysis compares those
hashes against the checkout it runs in. For the runs of 2026-09-24 that comparison reports
drift, so what drifted is recorded here.

`src/recovery_core/generator.py` changed by one line after the runs were launched and before
they were analysed. The modification notice at the top of the file pointed at a document that
does not travel to the publication mirror, and was repointed:

The old line pointed at the working repository's instruction file, which does not travel
here, alongside this document; the new one names this document and `NOTICE`, both of which
do. The exact text of both is in the version history.

That is the whole difference: 290 lines before and after, 12159 bytes against 12125, and the
parsed source is byte-identical under `ast.dump`, so nothing executable moved. The numbers in
`results.md` follow from the current checkout.

Editing a module while a run was in flight was a sequencing mistake, not a safe practice. Two
things came out of it. The runs now record a hash of the parsed source beside the file hash,
so a later reader can tell a comment edit from a behaviour change from the record alone rather
than being told the numbers are unreproducible and left to check. And the analysis no longer
claims non-reproducibility it has not established: on drift it names the files and points here.

`src/recovery_core/runs.py` drifts for the same runs, by the addition of that AST hashing. It
is not on the path that produced any number: it validates configurations, hashes them and
writes files.


## Estimator final-iterate correction and code organization (2026-09-25)

The historical estimator run `52ed7c149f7cd299` has invalid final-labelled DSM rows:
the runner loaded the 5k, 10k and 20k snapshots into the final model, then evaluated the
20k model again under the 40k label. The old JSON remains unchanged. Its
`final_state_sha256` was computed before snapshot evaluation and does identify the final
trained state; it does not identify every evaluated row.

The refactor moves the former estimator runner's scientific flow into
`recovery_estimators/experiment.py` and keeps CLI/file lifecycle separate in
`records.py`. The generator, splits, network, optimizer, objective and numerical potential
settings are unchanged. The intentional behavioral correction is explicit loading of each
evaluated state, including a preserved copy of the final state. Final-only replay omits
intermediate copies/evaluations and all other estimators. Tiny matched-seed tests compare
its final metrics and state hash with the full budget path and an independently captured
final iterate. These are implementation checks, not new experimental evidence.

The immediate predecessor is commit `4bef965` in this repository. The existing attribution
in `dsm.py` remains; its header now describes intermediate snapshots accurately. The
optimizer loop and its random draws are unchanged. No TCL-derived generator code was edited.

Legacy estimator provenance hashes only `recovery_core`; it omits torch and estimator
sources and describes only the float64 evaluation precision. This limitation is retained
rather than repaired retrospectively. New records hash both packages and entrypoints,
record torch/version/thread settings and training dtype separately, and label evaluated
updates and state hashes. Supplements carry base manifest/seed file digests and must match
the original final-state hash. No new production results were generated by this refactor.

The regenerated estimator report excludes invalid legacy final rows, reports seed SD/counts,
and makes no general claim excluding optimization, capacity or insufficient training.

## Selection reproduction repair (2026-09-26)

Before the full rerun, `select_potential.py` was found to reference undefined
`grid`, `reference_only` and `context` variables at aggregation. Candidate cells
and the contextual 8192-feature probe are now separated explicitly, and the declared
5% tie rule applies only to candidates. Regression tests cover a probe better than
every candidate and the tie rule. Frozen solver settings are not changed by this
reproduction run. Historical selection records remain unchanged.

## Publication figures and selected rerun (2026-09-28)

The published `data/` is a byte-preserving copy of 38 files from
`rerun-20260926T121833Z`; `data/manifest.json` records the source run, execution timestamps,
configuration IDs and SHA-256 of every raw record. It also pins numerical/training source,
runner, configuration and dependency-lock hashes from that execution. These files have not
been changed by figure preparation. Old result sets remain local development records.

`src/recovery_reporting/`, `scripts/reproduce.py` and
`scripts/prepare_publication_data.py` are new project code, not TCL-derived. They validate
the chosen data and generate publication artifacts. Type-checking cleanups rename temporary variables in the existing report without changing
its calculations. The change to the controlled report's interpretation text replaces the obsolete Section 4.5 pointer with Discussion.
Plotting uses the saved measurements; contrast curl reuses the existing reconstruction and
central-difference routine on deterministic test points. `figures/manifest.json` separately
records rendering-source and artifact hashes. The numerical generator and solver are unchanged.
The optional historical equivalence script now takes an explicit source path instead of
a private absolute path, and exits unsuccessfully if a comparison fails.

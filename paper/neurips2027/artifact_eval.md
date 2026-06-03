# Artifact Evaluation Guide

This guide is written for a reviewer evaluating the supplementary artifact. It
is intentionally command-oriented and separates fast local checks from
network-dependent or optional experiments.

## Artifact Scope

The artifact supports these claims:

- RANC-ContractNet compiles normalization policies from declared invariance
  contracts rather than hidden validation-score scaler search.
- The sklearn transformer serializes fitted policies, regime cards, signal-risk
  ledger rows, drift monitors, rejected candidates, seeds, and falsification
  results.
- Sparse, temporal, paired outlier, synthetic benchmark, ablation, and public
  OpenML/UCI summary artifacts can be regenerated from the included code and
  configs.
- Public benchmark inclusion/exclusion metadata and paired RANC-baseline deltas
  are available as reviewer-readable artifacts.

The artifact does not claim universal predictive dominance over all scalers. See
`paper/neurips2027/claims_boundary.md`.

For the full preprint/submission gap map, see
`paper/neurips2027/submission_readiness.md`.

For the current local git-tag snapshot criteria, see
`paper/neurips2027/release_snapshot.md`.

## Experimental Protocol Lock

Reviewers should treat `experiments/configs/*.yaml` as the protocol source of
truth. The paired outlier claim uses seeds `0` through `29`, a `0.35` stratified
test split, and 5,000 bootstrap samples. The aggregate synthetic benchmark uses
seeds `0` through `4`. The OpenML/UCI public benchmark uses seeds `0` through
`2`, records included/excluded/failed task status, requires `exclude_reason` for
disabled tasks, and fetches raw public data at runtime.

Paper-facing results should be regenerated through the tiered commands below
before reporting. A task that cannot be fetched or run is a logged failure, not
an invisible omission. OpenML/UCI comparisons are scoped public-data evidence
and must be read with the exclusion log and claims boundary.

## Failure Modes to Inspect

The artifact should expose known failure modes rather than hiding them in an
aggregate metric. Reviewers should inspect no-op reasons and downgrades for
underspecified contracts, wrong-contract controls and ledger rows for incorrect
declared semantics, supervised label-use notes for rare-signal handling, drift
monitor values for temporal shift, and paired predictive deltas alongside
contract/audit pass rates. Neural adapter checks are smoke tests unless a larger
neural benchmark is regenerated.

## Hardware and Software Assumptions

- CPU-only validation is supported for all required tests and synthetic artifact
  regeneration.
- Python `>=3.9` is required.
- Network access is required only for the OpenML/UCI public benchmark tier.
- GPU is not required. Torch is optional and used only for neural pilot checks.
- A TeX installation is optional; the repository provides `.tex` table snippets
  but local PDF compilation is not part of the fast validation path.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev,experiments]"
```

Optional neural pilot dependencies:

```bash
python3 -m pip install -e ".[torch]"
```

## Verify the Bundle

From the directory containing the zip and sidecar hash:

```bash
shasum -a 256 -c ranc_contractnet_neurips2027_artifact.sha256
unzip -t ranc_contractnet_neurips2027_artifact.zip
```

Inside the extracted top-level directory:

```bash
shasum -a 256 -c SHA256SUMS
```

Pass criteria:

- The sidecar zip hash verifies.
- `unzip -t` reports no errors.
- `SHA256SUMS` verifies all files in the extracted artifact.
- The artifact contains `ARTIFACT_MANIFEST.md`, `REPRODUCE.md`,
  `paper/neurips2027/artifact_eval.md`, and `paper/neurips2027/claims_boundary.md`.

## Extracted Bundle Dry Run

After building the zip, the maintainer-side reviewer simulation is:

```bash
python3 scripts/review_check.py --tier dryrun
```

This extracts `dist/ranc_contractnet_neurips2027_artifact.zip` into a clean
work directory, verifies the sidecar SHA-256 value, runs `SHA256SUMS` from
inside the extracted copy, runs Tier 1, runs the optional render tier from the
extracted copy, and checks that the packaged PDF/render report/preview survived
packaging. It writes:

- `outputs/artifact_dry_run/extracted_bundle_report.md`
- `outputs/artifact_dry_run/extracted_bundle_report.json`

The dry-run report is intentionally local and is not inserted back into the zip,
because doing so would change the artifact hash that the report validates.

## Tier 1: Fast CPU Validation

Expected runtime: under a few minutes on a laptop CPU.

Single-command reviewer wrapper:

```bash
python3 scripts/review_check.py --tier 1
```

This writes:

- `outputs/review_check/review_check_report.md`
- `outputs/review_check/review_check_report.json`

Equivalent explicit commands:

```bash
python3 -m pytest
python3 -m compileall src experiments tests
```

Expected outputs:

- `pytest` reports all tests passing.
- `compileall` exits successfully.

Pass criteria:

- All unit/integration tests pass.
- Schema round-trips, compiler constraints, leakage guards, sparse behavior,
  torch import path, OpenML artifact generation mocks, and paper artifact wiring
  are covered by tests.

## Optional Paper PDF Render Check

This tier compiles `paper/neurips2027/main.tex` when a complete TeX toolchain is
available. Missing TeX is reported as a skipped optional check rather than a
failure of the CPU artifact path. The render report also records `Style mode`:
`fallback_article` means `neurips_2027.sty` was not present, while
`official_neurips_2027` means the local official style file was used.

```bash
python3 scripts/review_check.py --tier render
```

Equivalent explicit command:

```bash
python3 scripts/render_paper.py
```

Expected outputs:

- `outputs/paper_render/paper_render_report.md`
- `outputs/paper_render/paper_render_report.json`

Pass criteria:

- If `latexmk`, `tectonic`, or `pdflatex` plus `bibtex` is available, the script
  produces `outputs/paper_render/main.pdf` or fails with TeX command output.
- If no complete TeX toolchain is available, the report status is `skipped` and
  records the missing toolchain explicitly.
- If `pdftoppm` or macOS `qlmanage` is available after a successful PDF compile,
  page previews are rendered to PNG for visual inspection.

## Tier 2: Synthetic Paper Artifact Regeneration

Expected runtime: CPU-only, moderate. This does not require network access.

Single-command reviewer wrapper:

```bash
python3 scripts/review_check.py --tier 2
```

Equivalent explicit command:

```bash
python3 experiments/paper_results_runner.py
```

Expected outputs include:

- `outputs/outlier_pair/contract_causal_table.tex`
- `outputs/outlier_pair/contract_delta_table.tex`
- `outputs/outlier_pair/contract_statistical_paragraph.md`
- `outputs/sparse/sparse_table.tex`
- `outputs/temporal_drift/temporal_drift_table.tex`
- `outputs/ablations/ablation_table.tex`
- `outputs/benchmark/benchmark_table.tex`

Pass criteria:

- The command exits successfully.
- The listed files are present.
- RANC audit artifacts are emitted for RANC runs.
- The paired outlier tables preserve the correct-contract versus wrong-contract
  comparison.

## Paper Figure Regeneration

```bash
python3 scripts/generate_paper_figures.py
```

Expected outputs:

- `paper/neurips2027/figures/ranc_visual_summary.pdf`
- `paper/neurips2027/figures/ranc_visual_summary.png`

The figure is generated from aggregate result CSVs and imported by
`paper/neurips2027/main.tex`.

## Tier 3: Network-Dependent OpenML/UCI Regeneration

Expected runtime: depends on OpenML availability and network speed.

Single-command reviewer wrapper:

```bash
python3 scripts/review_check.py --tier 3
```

Equivalent explicit command:

```bash
python3 experiments/openml_runner.py --config experiments/configs/openml_public.yaml
```

Expected outputs include:

- `outputs/openml/openml_summary.csv`
- `outputs/openml/openml_aggregate.csv`
- `outputs/openml/openml_table.tex`
- `outputs/openml/openml_task_metadata.csv`
- `outputs/openml/exclusion_log.md`
- `outputs/openml/openml_pairwise_deltas.csv`
- `outputs/openml/openml_win_loss_table.tex`
- `outputs/openml/openml_stats_paragraph.md`

Pass criteria:

- Completed tasks are retained even if a transient OpenML fetch fails.
- Any failed tasks are recorded in `openml_task_metadata.csv` and
  `exclusion_log.md`.
- Policy exclusions are explicit and include an `exclude_reason`.
- Raw public datasets are fetched at runtime and are not written into the
  packaged artifact.

## Tier 4: Optional Torch/Neural Pilot

Expected runtime: small CPU smoke only unless the reviewer chooses larger neural
experiments.

```bash
python3 -m pytest tests/test_torch_layers.py
```

Pass criteria:

- The torch layer test passes when torch is installed.
- Neural results should be treated as exploratory unless regenerated with a
  larger neural benchmark plan.

## Rebuild the Supplementary Bundle

Single-command reviewer wrapper:

```bash
python3 scripts/review_check.py --tier package
```

Equivalent explicit command:

```bash
python3 experiments/package_artifact.py
```

Expected outputs:

- `dist/ranc_contractnet_neurips2027_artifact.zip`
- `dist/ranc_contractnet_neurips2027_artifact.sha256`

Pass criteria:

- The packager exits successfully.
- The archive contains `ARTIFACT_MANIFEST.md`, `REPRODUCE.md`, and `SHA256SUMS`.
- The archive excludes caches, bytecode, build directories, raw OpenML run
  folders, raw dataset-like files, local source PDFs, local absolute paths,
  local user tokens, and generated per-seed outlier audit dumps. If the render
  tier has produced `outputs/paper_render/main.pdf`, that generated paper PDF is
  included; generated preview PNGs under `outputs/paper_render/preview/` are
  included when present.

## Full Reviewer Sequence

```bash
python3 scripts/review_check.py --tier all
```

This runs Tier 1, Tier 2, the optional render check, Tier 3, and packaging
checks in sequence. It requires network access because Tier 3 fetches OpenML
datasets at runtime.

## Known Limitations

- OpenML availability is outside the repository's control. Transient failures
  should be logged rather than treated as silent omissions.
- The public benchmark suite is a first public-data artifact path, not a final
  leaderboard claim.
- The validation selector can win raw score by optimizing held-out outcomes;
  RANC should also be evaluated on contract compliance and audit guarantees.
- The future official NeurIPS 2027 style file is not bundled. The draft falls
  back to standard `article` formatting until the official style is released.

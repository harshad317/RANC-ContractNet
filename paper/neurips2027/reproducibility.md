# Reproducibility Protocol

## Experimental Protocol

All paper-facing experiments are config-driven. Synthetic paired outlier results
use seeds `0` through `29`, `n_samples: 800`, a `0.35` stratified test split,
`5000` bootstrap samples, and `stats_random_state: 12345`. The aggregate
synthetic benchmark uses seeds `0` through `4`. The public OpenML/UCI benchmark
uses seeds `0` through `2` unless a task explicitly overrides them.

Dataset inclusion is declared in `experiments/configs/openml_public.yaml`.
Enabled tasks are run; disabled tasks must include `exclude_reason`. Public
fetch/run failures are written to `outputs/openml/openml_task_metadata.csv` and
`outputs/openml/exclusion_log.md` when `continue_on_error` is enabled. Raw
public datasets are fetched at runtime and are not packaged.

Primary metrics are fixed before reporting: paired outlier-noise uses corruption
AUROC, paired outlier-signal uses rare-positive recall, public binary
classification uses AUROC when available, other classification uses accuracy,
and regression uses RMSE. Sparse and temporal checks report predictive metrics
plus audit diagnostics.

Before reporting paper-facing numbers, regenerate the matching tier:

- Tier 1: `python3 scripts/review_check.py --tier 1`
- Tier 2: `python3 scripts/review_check.py --tier 2`
- Tier 3: `python3 scripts/review_check.py --tier 3`
- Figures: `python3 scripts/generate_paper_figures.py`
- Render: `python3 scripts/review_check.py --tier render`
- Package: `python3 scripts/review_check.py --tier package`
- Extracted bundle dry run: `python3 scripts/review_check.py --tier dryrun`

The local release snapshot criteria are recorded in
`paper/neurips2027/release_snapshot.md`.

Reporting discipline: aggregate all configured seeds, retain every configured
dataset as included/excluded/failed metadata, and treat OpenML/UCI results as
scoped public-data evidence rather than a universal predictive ranking.

## Local Smoke

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
python3 experiments/tabular_runner.py --config experiments/configs/smoke.yaml
```

## Paper Tables

```bash
python3 experiments/outlier_pair_runner.py --config experiments/configs/outlier_pair.yaml
```

This regenerates the paired outlier tables imported by `main.tex`:

- `outputs/outlier_pair/contract_causal_table.tex`
- `outputs/outlier_pair/contract_delta_table.tex`
- `outputs/outlier_pair/contract_statistical_paragraph.md`

```bash
python3 experiments/tabular_runner.py --config experiments/configs/sparse.yaml
python3 experiments/tabular_runner.py --config experiments/configs/temporal_drift.yaml
```

These regenerate the sparse and temporal tables imported by `main.tex`:

- `outputs/sparse/sparse_table.tex`
- `outputs/sparse/sparse_result_paragraph.md`
- `outputs/temporal_drift/temporal_drift_table.tex`
- `outputs/temporal_drift/temporal_drift_result_paragraph.md`

```bash
python3 experiments/ablation_runner.py --config experiments/configs/ablation.yaml
```

This regenerates the ablation table imported by `main.tex`:

- `outputs/ablations/ablation_summary.csv`
- `outputs/ablations/ablation_table.tex`
- `outputs/ablations/ablation_result_paragraph.md`

```bash
python3 experiments/benchmark_runner.py --config experiments/configs/benchmark.yaml
```

This regenerates the aggregate synthetic benchmark table imported by `main.tex`:

- `outputs/benchmark/benchmark_summary.csv`
- `outputs/benchmark/benchmark_aggregate.csv`
- `outputs/benchmark/benchmark_table.tex`
- `outputs/benchmark/benchmark_result_paragraph.md`

```bash
python3 experiments/paper_results_runner.py
```

This regenerates all current paper-facing experiment artifacts in sequence.

## Paper Figures

```bash
python3 scripts/generate_paper_figures.py
```

This regenerates the visual summary imported by `main.tex`:

- `paper/neurips2027/figures/ranc_visual_summary.pdf`
- `paper/neurips2027/figures/ranc_visual_summary.png`

The figure reads aggregate CSV outputs rather than raw private data:

- `outputs/outlier_pair/contract_delta_stats.csv`
- `outputs/openml/openml_win_loss_summary.csv`

```bash
python3 experiments/openml_runner.py --config experiments/configs/openml_public.yaml
```

This regenerates the network-dependent public benchmark artifacts:

- `outputs/openml/openml_summary.csv`
- `outputs/openml/openml_aggregate.csv`
- `outputs/openml/openml_table.tex`
- `outputs/openml/openml_task_metadata.csv`
- `outputs/openml/exclusion_log.md`
- `outputs/openml/openml_pairwise_deltas.csv`
- `outputs/openml/openml_win_loss_table.tex`
- `outputs/openml/openml_stats_paragraph.md`
- `paper/neurips2027/claims_boundary.md`
- `paper/neurips2027/artifact_eval.md`

The all-results runner skips OpenML by default. To include public benchmarks in
that command, provide a YAML override with `run_openml: true`. The OpenML config
supports explicit `enabled: false` exclusions with `exclude_reason`; transient
fetch/run failures are retained in the metadata and exclusion log when
`continue_on_error` is enabled.

The LaTeX draft falls back to standard `article` formatting when the future
official `neurips_2027.sty` file is unavailable. When the official style is
released, place it beside `main.tex` and the draft will use it automatically.

## Supplementary Artifact Bundle

```bash
python3 scripts/review_check.py --tier 1
```

The reviewer wrapper emits `outputs/review_check/review_check_report.md` and
`outputs/review_check/review_check_report.json` with command status, runtime,
stdout/stderr tails, expected artifact checks, and pass/fail summary. Use
`--tier 2`, `--tier 3`, `--tier render`, `--tier package`, `--tier dryrun`, or
`--tier all` for synthetic regeneration, OpenML regeneration, optional PDF
render checking, bundle hygiene, extracted-bundle reviewer simulation, or the
full reviewer sequence.

```bash
python3 scripts/review_check.py --tier render
```

The render tier writes `outputs/paper_render/paper_render_report.md` and
`outputs/paper_render/paper_render_report.json`. If no complete TeX toolchain is
available, the report status is `skipped` and records the missing dependency
instead of pretending that the PDF was compiled. The report also records whether
the paper rendered with fallback `article` formatting or an available
`neurips_2027.sty` file.

```bash
python3 experiments/package_artifact.py
```

This creates `dist/ranc_contractnet_neurips2027_artifact.zip` and
`dist/ranc_contractnet_neurips2027_artifact.sha256`. The zip contains source,
configs, tests, paper scaffold files, generated summary artifacts, an
`ARTIFACT_MANIFEST.md`, a `REPRODUCE.md`, and `SHA256SUMS`. Reviewers should use
`paper/neurips2027/artifact_eval.md` for tiered commands, expected outputs, and
pass/fail criteria. The packager excludes caches, bytecode, local OS files, raw
OpenML run folders, raw dataset-like binary/data files, local source PDFs, local
smoke-test outputs under `outputs/smoke/`, and generated per-seed outlier audit
dumps. It also excludes local validation reports under `outputs/review_check/`
and `outputs/artifact_dry_run/`, because reviewers regenerate those reports from
their own environment. The generated `outputs/paper_render/main.pdf` is included
only when the render tier has produced it; generated preview PNGs under
`outputs/paper_render/preview/` are included when present.

```bash
python3 scripts/review_check.py --tier dryrun
```

This extracts the packaged zip into a clean local work directory, verifies the
sidecar hash and in-bundle `SHA256SUMS`, runs Tier 1 and the render tier from
inside the extracted copy, and writes
`outputs/artifact_dry_run/extracted_bundle_report.md/json`. That report remains
outside the zip so the artifact hash is stable after validation.

## Full Tabular Runs

Use one config per benchmark family. Store outputs under `outputs/<benchmark-name>/`
and include the exported audit reports in the supplementary artifact.

## Leakage Discipline

- Always wrap `RANCDataTransformer` inside an sklearn `Pipeline`.
- Never call `fit_transform` on validation or test splits.
- Temporal experiments must split before fitting.
- Grouped experiments must compute statistics from training groups only.
- Supervised label use must appear in the `SignalRiskLedgerRow.supervised_label_used` field.

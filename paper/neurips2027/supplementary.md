# Supplementary Material Draft

## Algorithm

1. Validate that `fit` receives only the active training split.
2. Build one `RegimeCard` per feature from training data.
3. Derive or ingest `InvarianceContract` objects.
4. Build `SignalRiskLedgerRow` entries for transforms that may remove signal.
5. Enumerate admissible candidate policies.
6. Reject candidates that violate hard clauses.
7. Check piecewise affine monotonicity with Z3 when available and record fallback numeric checks otherwise.
8. Select the minimum-complexity policy with deterministic tie-breaking.
9. Run falsification tests and record failures.
10. Serialize policies, cards, contracts, ledger rows, rejected candidates, seeds, and monitors.
11. Replay fixed policies during `transform`.

## Benchmark Matrix

| Regime | Datasets | Baselines | Primary metrics |
| --- | --- | --- | --- |
| Tabular classification | OpenML-CC18 | none, standard, minmax, robust, maxabs, quantile, power, AutoML selector | balanced accuracy, AUROC, runtime |
| Regression | UCI/OpenML skewed positive tasks | log/log1p, Box-Cox, Yeo-Johnson, standard, robust, quantile | RMSE, MAE, inverse error |
| Outlier signal/noise | Synthetic paired tasks | robust, clipping, quantile, no-op, standard | rare recall, AUROC, calibration |
| Sparse | text/recommender matrices | no-op, maxabs, L1/L2, standard without mean | sparsity retention, memory, AUROC |
| Temporal drift | time-ordered public data | train-only scalers, leakage sentinel, online scaler | forward quality, decay, drift alarms |
| Temporal rare-event case study | synthetic domain-style sequence | standard, robust, quantile, validation selector, RANC | AUROC, rare recall, train-only audit |
| Neural pilot | CIFAR/Tiny-ImageNet or small LM | BatchNorm, LayerNorm, RMSNorm, GroupNorm, DyT | accuracy/perplexity, stability, saturation |

## Audit Artifacts

Every experiment must export:

- `metrics.csv`
- `ranc_audit.json`
- `ranc_audit.md`
- exact config YAML
- package version and commit hash when available

## First Decisive Table

The paired outlier signal/noise benchmark is generated with:

```bash
python3 experiments/outlier_pair_runner.py --config experiments/configs/outlier_pair.yaml
```

It exports:

- `outputs/outlier_pair/raw_results.csv`
- `outputs/outlier_pair/summary.csv`
- `outputs/outlier_pair/outlier_signal_noise_table.md`
- `outputs/outlier_pair/outlier_signal_noise_table.tex`
- `outputs/outlier_pair/contract_causal_summary.csv`
- `outputs/outlier_pair/contract_causal_table.md`
- `outputs/outlier_pair/contract_causal_table.tex`
- `outputs/outlier_pair/contract_delta_stats.csv`
- `outputs/outlier_pair/contract_delta_table.md`
- `outputs/outlier_pair/contract_delta_table.tex`
- `outputs/outlier_pair/contract_statistical_paragraph.md`
- one RANC audit JSON/Markdown pair per scenario and seed

The causal table includes `ranc_correct`, `ranc_wrong_noise`, and
`ranc_wrong_signal` rows to test whether the declared contract, not validation
search, controls the failure mode. The delta table compares correct versus
wrong contracts on the same paired seeds with bootstrap confidence intervals and
paired significance tests. In the noise setting, the primary contract-violation
metric is `1 - AUROC` under corrupted extremes; tail false-positive rate remains
a diagnostic column. In the signal setting, the primary contract-violation
metric is `1 - rare_positive_recall`.

The LaTeX main draft imports:

- `outputs/outlier_pair/contract_causal_table.tex`
- `outputs/outlier_pair/contract_delta_table.tex`

The generated paper sentence is:

> Across 30 paired synthetic seeds, the correct RANC contract improved noise-corruption AUROC by 0.027 (95% bootstrap CI [0.023, 0.031], Wilcoxon p=<1e-4) compared with applying the signal-preserving contract to the noise setting. In the signal setting, the correct contract improved rare-event recall by 0.026 (95% bootstrap CI [0.012, 0.040], Wilcoxon p=0.0025) compared with applying the noise-damping contract to true rare-signal extremes.

## Paper Claim Under Test

The first paper claim is not that RANC always beats every scaler on every metric.
It is that the declared contract controls the failure mode without hidden
validation-score scaler search. The paired outlier benchmark falsifies this
claim if the correct contract does not beat the wrong contract on the
scenario-specific violation metric across paired seeds.

## Claims Boundary

The standalone claims note is `paper/neurips2027/claims_boundary.md`. It records
the intended scope of the paper: RANC is an auditable contract compiler, not a
hidden validation-search optimizer. Public comparisons should report predictive
utility and contract/audit guarantees as separate axes, especially when the
validation selector wins raw score by optimizing held-out outcomes directly.

## Experimental Protocol and Reporting Discipline

All experiments are specified by YAML configs checked into the repository.
Synthetic paired outlier results use seeds `0` through `29`, `n_samples: 800`,
a `0.35` stratified test split, 5,000 bootstrap samples, and
`stats_random_state: 12345`. The aggregate synthetic harness uses seeds `0`
through `4`. The public OpenML/UCI runner uses seeds `0` through `2` by default
and records every task as included, excluded by policy, or failed during
fetch/run.

Dataset inclusion is config-driven. A public dataset is included only when it is
enabled in `experiments/configs/openml_public.yaml`; a disabled dataset must
carry `exclude_reason`. Transient network or fetch failures are not silently
dropped: with `continue_on_error` enabled, completed tasks remain in the
summary, failed tasks are recorded in `outputs/openml/openml_task_metadata.csv`,
and the exclusion log is regenerated. Raw public datasets are fetched at runtime
and are not packaged.

Split policy is fixed before fitting any preprocessing estimator. Random
synthetic and OpenML/UCI tasks use train/test splits with the configured seed;
classification synthetic tasks use stratification where labels are available.
Temporal drift tasks use the prefix 70% of the sequence for training and the
future suffix for evaluation. Every RANC policy is fit inside an sklearn
pipeline, so fitted normalization statistics come from the active training split
only.

Primary metrics are selected before reporting. Paired outlier-noise uses
corruption AUROC; paired outlier-signal uses rare-positive recall. Public
classification tasks report AUROC when binary scores are available and accuracy
otherwise; regression tasks report RMSE with lower values treated as better.
Sparse and temporal safety checks report predictive metrics together with audit
diagnostics such as sparse densification failures, nonzero deltas, train-only
fit count, and drift-monitor counts.

The reporting rule is no cherry-picking across seeds, datasets, or exclusions:
all configured seeds are aggregated, all disabled public tasks retain explicit
reasons, and public OpenML/UCI numbers are interpreted as scoped artifact
evidence rather than as a universal predictive ranking.

## Failure Modes and Artifact Signals

| Failure mode | Consequence | Artifact signal |
| --- | --- | --- |
| Underspecified contract | Compiler may choose a conservative no-op or weaker legal policy. | Downgrades, rejected candidates, and no-op reasons in the audit report |
| Wrong declared semantics | RANC can faithfully preserve or remove the wrong structure. | Wrong-contract controls, signal-risk ledger rows, and degraded scenario metric |
| Rare-signal ambiguity | Extreme values cannot be assumed signal without domain contracts or supervised opt-in. | Missing extreme-signal clause or supervised label-use note in the ledger |
| Predictive objective mismatch | Validation selectors may achieve better raw score because they optimize held-out score directly. | Paired RANC-baseline deltas plus contract/audit pass rates |
| Drift without adaptation | Drift monitors expose shift but do not automatically choose a retraining or adaptation policy. | Drift-monitor count and drift score in audit reports |
| Legal but unhelpful transform | A policy can satisfy the contract while adding little downstream predictive value. | Reported predictive metric alongside audit diagnostics |
| Exploratory neural path | Neural adapters may not transfer to larger architectures without a dedicated neural benchmark. | Torch smoke status and claims-boundary text marking neural results exploratory |

## Likely Reviewer Questions

| Reviewer question | Short answer | Evidence path |
| --- | --- | --- |
| Is this just AutoML or scaler selection? | No. RANC compiles the least complex legal policy under hard invariance contracts rather than searching scalers by validation score. | Method objective, Algorithm 1, validation-selector comparisons |
| What if the contract is wrong? | RANC can preserve the wrong semantics; the artifact is designed to expose that through controls and ledgers. | wrong-contract paired controls, signal-risk ledger rows, failure-mode table |
| Why can the validation selector beat RANC? | The selector optimizes held-out predictive score directly; RANC optimizes legality, auditability, and contract compliance. | OpenML paired deltas, claims boundary |
| Where would someone use this? | In workflows where a preprocessing decision must preserve declared signal and prove train-prefix fit discipline. | temporal rare-event case study, audit summary |
| Where is leakage protection? | Fit statistics are computed inside sklearn pipelines from the active training split only. | leakage tests, train-only audit fields, temporal prefix metadata |
| Are neural results central? | No. Neural adapters are exploratory until backed by a larger neural benchmark. | torch smoke test, claims boundary |
| Can the numbers be reproduced? | Yes through tiered reviewer commands and packaged checks; OpenML additionally depends on network availability. | `artifact_eval.md`, `reproducibility.md`, SHA256 bundle checks |

## Sparse and Temporal Evidence Tables

The main paper uses a compact safety summary table for reviewer readability.
The full generated sparse and temporal tables remain in the artifact paths
below.

The temporal rare-event case study is generated with:

```bash
python3 experiments/tabular_runner.py --config experiments/configs/case_study_temporal_rare_event.yaml
```

It exports:

- `outputs/case_studies/metrics.csv`
- `outputs/case_studies/temporal_rare_event_table.md`
- `outputs/case_studies/temporal_rare_event_table.tex`
- `outputs/case_studies/temporal_rare_event_audit.md`
- `outputs/case_studies/temporal_rare_event_result_paragraph.md`
- `outputs/case_studies/ranc_audit.json`
- `outputs/case_studies/ranc_audit.md`

The sparse safety table is generated with:

```bash
python3 experiments/tabular_runner.py --config experiments/configs/sparse.yaml
```

It exports:

- `outputs/sparse/metrics.csv`
- `outputs/sparse/sparse_table.md`
- `outputs/sparse/sparse_table.tex`
- `outputs/sparse/sparse_result_paragraph.md`
- `outputs/sparse/ranc_audit.json`
- `outputs/sparse/ranc_audit.md`

The temporal drift table is generated with:

```bash
python3 experiments/tabular_runner.py --config experiments/configs/temporal_drift.yaml
```

It exports:

- `outputs/temporal_drift/metrics.csv`
- `outputs/temporal_drift/temporal_drift_table.md`
- `outputs/temporal_drift/temporal_drift_table.tex`
- `outputs/temporal_drift/temporal_drift_result_paragraph.md`
- `outputs/temporal_drift/ranc_audit.json`
- `outputs/temporal_drift/ranc_audit.md`

The sparse benchmark is an artifact-level representation test: a passing RANC
row must keep sparse output, preserve test nonzero count, and report zero sparse
densification failures. The temporal benchmark is an artifact-level leakage and
monitoring test: the RANC row must report `ranc_train_only_fit=True` and the
audit must contain drift monitors fitted only on the prefix training split.

## Ablation Evidence Table

The ablation table is generated with:

```bash
python3 experiments/ablation_runner.py --config experiments/configs/ablation.yaml
```

It exports:

- `outputs/ablations/ablation_summary.csv`
- `outputs/ablations/ablation_table.md`
- `outputs/ablations/ablation_table.tex`
- `outputs/ablations/ablation_result_paragraph.md`
- `outputs/ablations/<ablation-name>/metrics.csv`
- `outputs/ablations/<ablation-name>/ranc_audit.json` for RANC variants
- `outputs/ablations/<ablation-name>/ranc_audit.md` for RANC variants

This table is an artifact-semantics ablation, not a leaderboard result. The
expected behavior is that `no_ledger_pressure` reports zero ledger rows,
`force_noop` reports policy downgrades after hard-clause conflicts,
`no_outlier_damping` changes the selected policy family, and
`selector_baseline` reports a selected scaler without a RANC audit.

## Multi-Seed Benchmark Harness

The main paper summarizes this harness in prose. The full aggregate synthetic table
is intentionally kept in the supplementary artifact because it is
reproducibility evidence rather than the central contract-causality claim.

The aggregate benchmark table is generated with:

```bash
python3 experiments/benchmark_runner.py --config experiments/configs/benchmark.yaml
```

It exports:

- `outputs/benchmark/benchmark_summary.csv`
- `outputs/benchmark/benchmark_aggregate.csv`
- `outputs/benchmark/benchmark_table.md`
- `outputs/benchmark/benchmark_table.tex`
- `outputs/benchmark/benchmark_result_paragraph.md`
- `outputs/benchmark/runs/<benchmark>/seed_<seed>/metrics.csv`

The default config covers `outlier_noise`, `outlier_signal`, `sparse`,
`temporal_drift`, `scale_shift`, and `additive_shift` over five seeds. This is a
reproducibility harness for synthetic coverage, not the final public benchmark
suite. The final paper should add OpenML/UCI tasks and report the same table
schema over public datasets.

All current paper-facing artifacts can be regenerated with:

```bash
python3 experiments/paper_results_runner.py
```

The supplementary artifact bundle is generated with:

```bash
python3 scripts/review_check.py --tier package
python3 experiments/package_artifact.py
```

It exports:

- `dist/ranc_contractnet_neurips2027_artifact.zip`
- `dist/ranc_contractnet_neurips2027_artifact.sha256`

The zip includes `ARTIFACT_MANIFEST.md`, `REPRODUCE.md`, and `SHA256SUMS`, while
excluding local caches, raw OpenML run folders, raw dataset-like files, and
per-seed outlier audit dumps. Artifact-review commands, expected outputs,
runtime tiers, and pass/fail criteria are in
`paper/neurips2027/artifact_eval.md`.

The consolidated reviewer wrapper is:

```bash
python3 scripts/review_check.py --tier 1
```

It writes `outputs/review_check/review_check_report.md` and
`outputs/review_check/review_check_report.json`; use `--tier all` only when
network-dependent OpenML regeneration is intended.

The optional PDF render check is:

```bash
python3 scripts/review_check.py --tier render
```

It writes `outputs/paper_render/paper_render_report.md` and
`outputs/paper_render/paper_render_report.json`. The status is `passed` when a
TeX engine compiles the draft, `failed` when an available engine cannot compile
it, and `skipped` when no complete TeX toolchain is available.

## OpenML/UCI Public Benchmark Path

The main paper reports the paired OpenML/UCI win/loss summary and keeps the full task-level table
in the supplementary artifact. This prevents a large support table from
dominating the main narrative while preserving the reviewer audit trail.

The public benchmark path is generated with:

```bash
python3 experiments/openml_runner.py --config experiments/configs/openml_public.yaml
```

It exports:

- `outputs/openml/openml_summary.csv`
- `outputs/openml/openml_aggregate.csv`
- `outputs/openml/openml_table.md`
- `outputs/openml/openml_table.tex`
- `outputs/openml/openml_task_metadata.csv`
- `outputs/openml/exclusion_log.md`
- `outputs/openml/openml_pairwise_deltas.csv`
- `outputs/openml/openml_task_win_loss.csv`
- `outputs/openml/openml_win_loss_summary.csv`
- `outputs/openml/openml_win_loss_table.md`
- `outputs/openml/openml_win_loss_table.tex`
- `outputs/openml/openml_stats_paragraph.md`
- `outputs/openml/openml_result_paragraph.md`
- `outputs/openml/runs/<task>/seed_<seed>/metrics.csv`

The default config includes OpenML-hosted and UCI-derived tabular datasets. The
current generated selection log records 12 completed datasets, 3 documented
policy exclusions, and any transient fetch/run failures. Paired public
statistics compare RANC against standard, robust, and validation-selector
baselines with direction-aligned signed deltas, task W/L/T counts, seed-pair
W/L/T counts, and bootstrap confidence intervals. The path is
intentionally not executed by `paper_results_runner.py` unless `run_openml:
true` is supplied, because the run requires network access and public dataset
fetches. The repository stores metrics, tables, and audit reports only; raw
public datasets are not written to the artifact directory.

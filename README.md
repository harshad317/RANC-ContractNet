# RANC-ContractNet

RANC-ContractNet turns normalization into a contract-constrained compilation problem.
Instead of choosing a scaler by habit or validation search, it profiles training data,
derives or accepts invariance contracts, builds a signal-risk ledger, compiles the
least complex admissible transform, and records falsification results in an audit report.

This repository is a research scaffold for a NeurIPS 2027 main-track submission. The
preprocessing compiler is the central contribution; torch activation policies are included
as an exploratory extension.

## Repository Status

- Branch: `main`
- Current artifact release:
  [paper-revision-draft-2026-06-03-v3](https://github.com/harshad317/RANC-ContractNet/releases/tag/paper-revision-draft-2026-06-03-v3)
- License: MIT
- Citation metadata: `CITATION.cff`
- DOI status: pending; `.zenodo.json` is included for a future Zenodo archive

## Quick Start

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
```

```python
from ranc_contractnet import RANCDataTransformer

normalizer = RANCDataTransformer(
    contracts={
        "global": {
            "hard_clauses": {
                "preserve_monotonicity": True,
                "enforce_scale_invariance": True,
            }
        }
    },
    random_state=0,
)
X_train_norm = normalizer.fit_transform(X_train, y_train)
X_test_norm = normalizer.transform(X_test)
audit = normalizer.get_audit_report()
```

## Public Interfaces

- `ranc_contractnet.RANCDataTransformer`
- `ranc_contractnet.compile_contracts`
- `ranc_contractnet.save_policy`
- `ranc_contractnet.load_policy`
- `ranc_contractnet.audit.export_report`
- `ranc_contractnet.torch_layers.RANCActivationPolicy`

## Research Contract

The implementation is designed around four invariants:

1. Fitted statistics are computed only during `fit` from the active training split.
2. Policy selection never uses validation/test performance.
3. Hard contract clauses either pass falsification or produce an audited downgrade.
4. Serialized policies replay exactly at validation, test, and deployment time.

## Decisive Synthetic Experiment

```bash
python3 experiments/outlier_pair_runner.py --config experiments/configs/outlier_pair.yaml
```

This paired benchmark uses the same rare outlier positions in two worlds:
outliers are corruptions in the noise scenario and predictive rare events in
the signal scenario. It exports raw metrics, aggregated summaries, Markdown and
LaTeX tables, and RANC audit reports for each seed/scenario. It also includes
wrong-contract controls: signal contract on noise, and noise contract on signal.
The statistical outputs include paired bootstrap confidence intervals, paired
p-values, and a generated Markdown sentence for the paper draft.

## Sparse and Temporal Evidence

```bash
python3 experiments/tabular_runner.py --config experiments/configs/sparse.yaml
python3 experiments/tabular_runner.py --config experiments/configs/temporal_drift.yaml
```

The sparse config exports `sparse_table.md/.tex` and verifies that compiled
policies preserve sparse structure and implicit zeros. The temporal drift config
exports `temporal_drift_table.md/.tex` and records train-only fit metadata,
future-suffix evaluation, and drift monitor counts in the RANC audit.

## Ablation Evidence

```bash
python3 experiments/ablation_runner.py --config experiments/configs/ablation.yaml
```

This exports `outputs/ablations/ablation_summary.csv`,
`ablation_table.md/.tex`, and one full metrics/audit folder per ablation. The
table is designed to show audited component effects: ledger removal, forced
fallback downgrades, outlier-damping policy changes, and the validation selector
baseline without a RANC audit.

## Multi-Seed Benchmark Harness

```bash
python3 experiments/benchmark_runner.py --config experiments/configs/benchmark.yaml
```

This runs the synthetic benchmark families across multiple seeds and exports
`outputs/benchmark/benchmark_summary.csv`,
`outputs/benchmark/benchmark_aggregate.csv`, and `benchmark_table.md/.tex`.

To regenerate all current paper-facing artifacts in one command:

```bash
python3 experiments/paper_results_runner.py
```

## OpenML/UCI Public Benchmarks

```bash
python3 experiments/openml_runner.py --config experiments/configs/openml_public.yaml
```

This fetches public datasets at runtime and writes only metrics, aggregate
tables, and RANC audits under `outputs/openml`. It exports
`openml_summary.csv`, `openml_aggregate.csv`, `openml_table.md/.tex`, and
`openml_task_metadata.csv`, plus `exclusion_log.md` with enabled, excluded, and
failed-during-run dataset records. It also exports paired public comparison
artifacts: `openml_pairwise_deltas.csv`, `openml_win_loss_table.md/.tex`, and
`openml_stats_paragraph.md`. The config supports `enabled: false`,
`exclude_reason`, and `continue_on_error` so transient OpenML failures do not
discard completed public benchmark results. The all-results runner does not
execute this step by default because it depends on network access; pass
`run_openml: true` in a YAML override for a full public run.

## Supplementary Artifact Bundle

Reviewer entry point:

```bash
python3 scripts/review_check.py --tier 1
```

This writes `outputs/review_check/review_check_report.md` and
`outputs/review_check/review_check_report.json` with command status, runtime,
stdout/stderr tails, expected artifact checks, and a pass/fail summary. Use
`--tier 2` for synthetic artifact regeneration, `--tier 3` for network-dependent
OpenML regeneration, `--tier render` for optional PDF compilation,
`--tier package` for zip hygiene, `--tier dryrun` for a clean extracted-bundle
reviewer simulation, or `--tier all` for the full reviewer sequence. The
anonymous submission path also supports `--tier render_anonymous`,
`--tier package_anonymous`, and `--tier dryrun_anonymous`.

```bash
python3 experiments/package_artifact.py
```

This writes `dist/ranc_contractnet_neurips2027_artifact.zip` and a sidecar
SHA-256 file. The bundle includes source, configs, tests, paper scaffold files,
generated summary artifacts, `ARTIFACT_MANIFEST.md`, `REPRODUCE.md`, and
`SHA256SUMS`. Reviewers should start with
`paper/neurips2027/artifact_eval.md`, which lists runtime tiers, expected
outputs, and pass/fail criteria. The packager excludes caches, bytecode, raw
OpenML run folders, raw dataset-like files, local source PDFs, local smoke-test
outputs under `outputs/smoke/`, local validation reports under
`outputs/review_check/` and `outputs/artifact_dry_run/`, and per-seed outlier
audit dumps, then runs an anonymization sanity check for local
path/user tokens. The generated `outputs/paper_render/main.pdf` is included only when the
render tier has produced it; generated preview PNGs under
`outputs/paper_render/preview/` are included when present.

After packaging, run:

```bash
python3 scripts/review_check.py --tier dryrun
```

This validates the zip from a clean extraction, reruns Tier 1 and the render
tier inside the extracted copy, and writes a local
`outputs/artifact_dry_run/extracted_bundle_report.md`. The report is kept
outside the zip to avoid changing the hash it validates.

For a double-blind submission artifact:

```bash
python3 scripts/review_check.py --tier render_anonymous
python3 experiments/package_artifact.py --identity-mode anonymous
python3 scripts/review_check.py --tier dryrun_anonymous
```

This writes `dist/ranc_contractnet_neurips2027_artifact_anonymous.zip`, maps the
anonymous rendered PDF to `outputs/paper_render/main.pdf` inside the zip, and
scans for author, affiliation, role-title, email, and local-path leaks.

Before treating the project as preprint- or submission-ready, read
`paper/neurips2027/submission_readiness.md`. It maps validated artifacts, claim
boundaries, appropriate use cases, remaining benchmark gaps, and release
engineering tasks.

The current local snapshot/tag criteria are recorded in
`paper/neurips2027/release_snapshot.md`.

## Citation

If you use RANC-ContractNet, cite the software artifact:

```bibtex
@software{patil2026ranccontractnet,
  author = {Patil, Harshad Hemant},
  title = {RANC-ContractNet: Auditable Normalization as Invariance Compilation},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/harshad317/RANC-ContractNet},
  note = {Artifact release: paper-revision-draft-2026-06-03-v3}
}
```

The current artifact release attaches identified and anonymous supplementary
bundles with SHA-256 sidecars and supersedes
`paper-revision-draft-2026-06-03-v2` after metadata synchronization. A DOI
should be minted from Zenodo after the next stable release or preprint
checkpoint.

## License

This project is released under the MIT License. See `LICENSE`.

## Repository Layout

```text
src/ranc_contractnet/   Core schemas, compiler, sklearn transformer, audits, torch layer
experiments/            Synthetic, tabular, ablation, and neural runners
experiments/configs/    Runnable YAML configs for smoke and full experiments
tests/                  Unit and integration tests
paper/neurips2027/      Paper scaffold, checklist, readiness map, supplementary notes
```

## Optional Dependencies

- `.[torch]` enables neural activation policies and neural experiment runners.
- `.[experiments]` enables OpenML fetching and plotting utilities.

`z3-solver` is a core dependency for the hybrid compiler. If a constrained
environment lacks Z3, the runtime falls back to deterministic numeric verification
and records `used_z3=false` in falsification metrics. Torch-specific imports raise
clear errors only when torch APIs are instantiated.

# NeurIPS 2027 Reproducibility Checklist Draft

## Claims

- ContractNet compiles normalization from explicit invariance contracts rather than validation-score scaler search.
- Fitted statistics are train-split-only and replayable.
- Signal-risk ledger and falsification tests prevent silent destruction of semantic zeros, signs, rare tails, and sparse structure.

## Code

- Package entry point: `src/ranc_contractnet`.
- Main transformer: `RANCDataTransformer`.
- Smoke command: `python3 -m pytest`.
- Experiment command: `python3 experiments/tabular_runner.py --config experiments/configs/smoke.yaml`.
- Decisive outlier command: `python3 experiments/outlier_pair_runner.py --config experiments/configs/outlier_pair.yaml`.

## Data

- Smoke tests use synthetic data generated from deterministic seeds.
- Full experiments should use public OpenML/UCI datasets fetched by config.
- No private data is required.

## Metrics

- Classification: accuracy, AUROC, calibration error in full runs.
- Regression: RMSE, MAE, inverse-transform error.
- Audit: policy complexity, no-op frequency, rejected candidates, hard-clause failures, drift warnings, sparse densification failures.

## Compute

- Unit tests and smoke experiments run on CPU.
- Full neural pilots may require GPU and must report throughput, memory, seeds, and hardware.

## Known Risks

- Automatic contracts are safe defaults, not domain truth.
- Z3 is optional in the local package and should be enabled for final symbolic-compilation experiments.
- Neural mode is exploratory unless the paper has strong regime-predicted wins.

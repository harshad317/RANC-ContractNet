| Ablation | Removed | Scaler | Selected | AUROC | Accuracy | Policies | Ledger rows | Rejected | Downgrades | Hard failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full RANC | none | ranc |  | 0.762 | 0.750 | robust_affine:1, zscore:3 | 1 | 10 | 0 | 0 |
| No Signal Risk Ledger | signal risk ledger | ranc |  | 0.762 | 0.750 | robust_affine:1, zscore:3 | 0 | 10 | 0 | 0 |
| Forced no-op fallback | admissible transform set | ranc |  | 0.762 | 0.750 | identity:3, robust_affine:1 | 1 | 22 | 3 | 0 |
| No outlier damping | outlier damping clause | ranc |  | 0.762 | 0.750 | zscore:4 | 1 | 7 | 0 | 0 |
| Validation selector | contract compiler | selector | quantile | 0.750 | 0.736 |  |  |  |  |  |

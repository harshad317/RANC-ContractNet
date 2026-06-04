# RANC-ContractNet Audit Report

- Created at: `2026-06-04T02:58:46+00:00`
- Random state: `0`
- Passed hard falsification: `True`
- Features: `5`

## Policies

| Feature | Policy | Complexity | Downgrade |
| --- | --- | ---: | --- |
| x0 | zscore | 2 |  |
| x1 | zscore | 2 |  |
| x2 | zscore | 2 |  |
| x3 | robust_affine | 2 |  |
| x4 | zscore | 2 |  |

## Signal Risk Ledger

| Feature | Proposed nuisance | Possible signal removed | Mitigation |
| --- | --- | --- | --- |
| x0 | heavy_tail_or_extreme_values | rare predictive tail events | forbid_clipping_and_quantile_saturation |
| x3 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |

## Falsification Failures

| Feature | Policy | Test | Hard clause | Message |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

## Warnings

- None

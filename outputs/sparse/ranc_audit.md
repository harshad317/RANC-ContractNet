# RANC-ContractNet Audit Report

- Created at: `2026-06-03T12:34:20+00:00`
- Random state: `1`
- Passed hard falsification: `True`
- Features: `64`

## Policies

| Feature | Policy | Complexity | Downgrade |
| --- | --- | ---: | --- |
| x0 | robust_affine | 2 |  |
| x1 | robust_affine | 2 |  |
| x2 | robust_affine | 2 |  |
| x3 | robust_affine | 2 |  |
| x4 | robust_affine | 2 |  |
| x5 | robust_affine | 2 |  |
| x6 | robust_affine | 2 |  |
| x7 | robust_affine | 2 |  |
| x8 | robust_affine | 2 |  |
| x9 | robust_affine | 2 |  |
| x10 | robust_affine | 2 |  |
| x11 | robust_affine | 2 |  |
| x12 | robust_affine | 2 |  |
| x13 | robust_affine | 2 |  |
| x14 | robust_affine | 2 |  |
| x15 | robust_affine | 2 |  |
| x16 | maxabs | 1 |  |
| x17 | robust_affine | 2 |  |
| x18 | robust_affine | 2 |  |
| x19 | robust_affine | 2 |  |
| x20 | robust_affine | 2 |  |
| x21 | robust_affine | 2 |  |
| x22 | robust_affine | 2 |  |
| x23 | robust_affine | 2 |  |
| x24 | robust_affine | 2 |  |
| x25 | robust_affine | 2 |  |
| x26 | robust_affine | 2 |  |
| x27 | robust_affine | 2 |  |
| x28 | robust_affine | 2 |  |
| x29 | robust_affine | 2 |  |
| x30 | robust_affine | 2 |  |
| x31 | robust_affine | 2 |  |
| x32 | robust_affine | 2 |  |
| x33 | robust_affine | 2 |  |
| x34 | robust_affine | 2 |  |
| x35 | robust_affine | 2 |  |
| x36 | robust_affine | 2 |  |
| x37 | robust_affine | 2 |  |
| x38 | robust_affine | 2 |  |
| x39 | robust_affine | 2 |  |
| x40 | robust_affine | 2 |  |
| x41 | robust_affine | 2 |  |
| x42 | robust_affine | 2 |  |
| x43 | robust_affine | 2 |  |
| x44 | robust_affine | 2 |  |
| x45 | robust_affine | 2 |  |
| x46 | robust_affine | 2 |  |
| x47 | robust_affine | 2 |  |
| x48 | robust_affine | 2 |  |
| x49 | robust_affine | 2 |  |
| x50 | robust_affine | 2 |  |
| x51 | robust_affine | 2 |  |
| x52 | robust_affine | 2 |  |
| x53 | robust_affine | 2 |  |
| x54 | robust_affine | 2 |  |
| x55 | robust_affine | 2 |  |
| x56 | maxabs | 1 |  |
| x57 | robust_affine | 2 |  |
| x58 | robust_affine | 2 |  |
| x59 | robust_affine | 2 |  |
| x60 | robust_affine | 2 |  |
| x61 | robust_affine | 2 |  |
| x62 | robust_affine | 2 |  |
| x63 | robust_affine | 2 |  |

## Signal Risk Ledger

| Feature | Proposed nuisance | Possible signal removed | Mitigation |
| --- | --- | --- | --- |
| x0 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x1 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x2 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x3 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x4 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x5 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x6 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x7 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x8 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x9 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x10 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x11 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x12 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x13 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x14 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x15 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x17 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x18 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x19 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x20 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x21 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x22 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x23 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x24 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x25 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x26 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x27 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x28 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x29 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x30 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x31 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x32 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x33 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x34 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x35 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x36 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x37 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x38 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x39 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x40 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x41 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x42 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x43 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x44 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x45 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x46 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x47 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x48 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x49 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x50 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x51 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x52 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x53 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x54 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x55 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x57 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x58 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x59 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x60 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x61 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x62 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |
| x63 | heavy_tail_or_extreme_values | rare predictive tail events | prefer_rank_preserving_or_piecewise_policy |

## Falsification Failures

| Feature | Policy | Test | Hard clause | Message |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

## Warnings

- None

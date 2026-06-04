# Temporal Rare-Event Case Study Audit

- Dataset kind: `temporal_rare_event`
- Split strategy: `time_ordered`
- Temporal leakage guard: `True`
- RANC train-only fit: `True`
- RANC fit samples: `364`
- RANC policies: `robust_affine:1, zscore:4`
- Signal-risk ledger rows: `2`
- Rejected candidates: `14`
- Drift monitors: `5`
- Max drift estimate: `0.245`
- Rare-event recall: `0.542`

The case-study contract treats rare extremes as possible signal rather than corruption, requires monotonic and invertible preprocessing, and uses time-ordered fit metadata so the audit can verify that no future rows supplied fitted statistics.

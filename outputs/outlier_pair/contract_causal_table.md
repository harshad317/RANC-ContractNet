| Scenario | Method | Contract | Correct | x0 policy | AUROC | Rare recall | Tail FPR | Contract violations |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| noise | ranc_correct | noise | True | log1p | 0.708 +/- 0.031 | 0.953 +/- 0.084 | 0.863 +/- 0.151 | 0.292 +/- 0.031 |
| noise | ranc_wrong_noise | signal | False | zscore | 0.680 +/- 0.032 | 0.726 +/- 0.209 | 0.569 +/- 0.215 | 0.320 +/- 0.032 |
| signal | ranc_correct | signal | True | zscore | 0.778 +/- 0.037 | 0.993 +/- 0.024 | 1.000 +/- 0.000 | 0.007 +/- 0.024 |
| signal | ranc_wrong_signal | noise | False | log1p | 0.777 +/- 0.037 | 0.967 +/- 0.042 | 0.917 +/- 0.204 | 0.033 +/- 0.042 |

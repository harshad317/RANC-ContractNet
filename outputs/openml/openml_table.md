| Benchmark | Source | Scaler | n | Primary | Accuracy | AUROC | RMSE | MAE | Selected | Audit pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openml_iris | openml | maxabs | 3 | 0.748 +/- 0.104 (higher) | 0.748 |  |  |  |  |  |
| openml_iris | openml | none | 3 | 0.906 +/- 0.033 (higher) | 0.905 |  |  |  |  |  |
| openml_iris | openml | ranc | 3 | 0.855 +/- 0.047 (higher) | 0.855 |  |  |  |  | 1.00 |
| openml_iris | openml | robust | 3 | 0.843 +/- 0.039 (higher) | 0.842 |  |  |  |  |  |
| openml_iris | openml | selector | 3 | 0.868 +/- 0.019 (higher) | 0.867 |  |  |  | power+quantile+standard |  |
| openml_iris | openml | standard | 3 | 0.881 +/- 0.022 (higher) | 0.880 |  |  |  |  |  |
| openml_credit_g | uci_via_openml | maxabs | 3 | 0.766 +/- 0.014 (higher) | 0.741 | 0.766 |  |  |  |  |
| openml_credit_g | uci_via_openml | none | 3 | 0.769 +/- 0.014 (higher) | 0.747 | 0.768 |  |  |  |  |
| openml_credit_g | uci_via_openml | quantile | 3 | 0.761 +/- 0.011 (higher) | 0.749 | 0.760 |  |  |  |  |
| openml_credit_g | uci_via_openml | ranc | 3 | 0.761 +/- 0.013 (higher) | 0.743 | 0.761 |  |  |  | 1.00 |
| openml_credit_g | uci_via_openml | robust | 3 | 0.768 +/- 0.013 (higher) | 0.745 | 0.768 |  |  |  |  |
| openml_credit_g | uci_via_openml | selector | 3 | 0.768 +/- 0.009 (higher) | 0.742 | 0.768 |  |  | maxabs+robust+standard |  |
| openml_credit_g | uci_via_openml | standard | 3 | 0.762 +/- 0.013 (higher) | 0.744 | 0.761 |  |  |  |  |
| openml_spambase | uci_via_openml | maxabs | 3 | 0.944 +/- 0.005 (higher) | 0.884 | 0.944 |  |  |  |  |
| openml_spambase | uci_via_openml | none | 3 | 0.968 +/- 0.004 (higher) | 0.917 | 0.968 |  |  |  |  |
| openml_spambase | uci_via_openml | quantile | 3 | 0.982 +/- 0.001 (higher) | 0.938 | 0.981 |  |  |  |  |
| openml_spambase | uci_via_openml | ranc | 3 | 0.968 +/- 0.003 (higher) | 0.914 | 0.967 |  |  |  | 1.00 |
| openml_spambase | uci_via_openml | robust | 3 | 0.969 +/- 0.003 (higher) | 0.917 | 0.968 |  |  |  |  |
| openml_spambase | uci_via_openml | selector | 3 | 0.981 +/- 0.001 (higher) | 0.938 | 0.981 |  |  | power+quantile |  |
| openml_spambase | uci_via_openml | standard | 3 | 0.967 +/- 0.003 (higher) | 0.916 | 0.966 |  |  |  |  |
| openml_wine_quality | uci_via_openml | minmax | 3 | 0.740 +/- 0.015 (lower) |  |  | 0.740 | 0.577 |  |  |
| openml_wine_quality | uci_via_openml | none | 3 | 0.742 +/- 0.015 (lower) |  |  | 0.741 | 0.578 |  |  |
| openml_wine_quality | uci_via_openml | ranc | 3 | 0.740 +/- 0.015 (lower) |  |  | 0.740 | 0.576 |  | 1.00 |
| openml_wine_quality | uci_via_openml | robust | 3 | 0.740 +/- 0.015 (lower) |  |  | 0.740 | 0.576 |  |  |
| openml_wine_quality | uci_via_openml | selector | 3 | 0.740 +/- 0.015 (lower) |  |  | 0.740 | 0.576 | robust+standard |  |
| openml_wine_quality | uci_via_openml | standard | 3 | 0.740 +/- 0.015 (lower) |  |  | 0.740 | 0.576 |  |  |
| openml_diabetes | uci_via_openml | maxabs | 3 | 0.823 +/- 0.036 (higher) | 0.769 | 0.822 |  |  |  |  |
| openml_diabetes | uci_via_openml | none | 3 | 0.817 +/- 0.046 (higher) | 0.768 | 0.817 |  |  |  |  |
| openml_diabetes | uci_via_openml | quantile | 3 | 0.842 +/- 0.018 (higher) | 0.768 | 0.842 |  |  |  |  |
| openml_diabetes | uci_via_openml | ranc | 3 | 0.835 +/- 0.017 (higher) | 0.778 | 0.835 |  |  |  | 1.00 |
| openml_diabetes | uci_via_openml | robust | 3 | 0.818 +/- 0.045 (higher) | 0.770 | 0.818 |  |  |  |  |
| openml_diabetes | uci_via_openml | selector | 3 | 0.844 +/- 0.018 (higher) | 0.774 | 0.843 |  |  | maxabs+quantile |  |
| openml_diabetes | uci_via_openml | standard | 3 | 0.818 +/- 0.046 (higher) | 0.770 | 0.817 |  |  |  |  |
| openml_credit_approval | uci_via_openml | maxabs | 3 | 0.930 +/- 0.018 (higher) | 0.874 | 0.930 |  |  |  |  |
| openml_credit_approval | uci_via_openml | none | 3 | 0.938 +/- 0.015 (higher) | 0.873 | 0.938 |  |  |  |  |
| openml_credit_approval | uci_via_openml | quantile | 3 | 0.932 +/- 0.017 (higher) | 0.863 | 0.931 |  |  |  |  |
| openml_credit_approval | uci_via_openml | ranc | 3 | 0.930 +/- 0.016 (higher) | 0.869 | 0.930 |  |  |  | 1.00 |
| openml_credit_approval | uci_via_openml | robust | 3 | 0.938 +/- 0.016 (higher) | 0.876 | 0.937 |  |  |  |  |
| openml_credit_approval | uci_via_openml | selector | 3 | 0.928 +/- 0.014 (higher) | 0.863 | 0.928 |  |  | quantile+standard |  |
| openml_credit_approval | uci_via_openml | standard | 3 | 0.924 +/- 0.022 (higher) | 0.865 | 0.923 |  |  |  |  |
| openml_balance_scale | uci_via_openml | maxabs | 3 | 0.868 +/- 0.008 (higher) | 0.867 |  |  |  |  |  |
| openml_balance_scale | uci_via_openml | none | 3 | 0.871 +/- 0.003 (higher) | 0.870 |  |  |  |  |  |
| openml_balance_scale | uci_via_openml | quantile | 3 | 0.872 +/- 0.012 (higher) | 0.872 |  |  |  |  |  |
| openml_balance_scale | uci_via_openml | ranc | 3 | 0.865 +/- 0.005 (higher) | 0.864 |  |  |  |  | 1.00 |
| openml_balance_scale | uci_via_openml | robust | 3 | 0.866 +/- 0.007 (higher) | 0.866 |  |  |  |  |  |
| openml_balance_scale | uci_via_openml | selector | 3 | 0.887 +/- 0.014 (higher) | 0.887 |  |  |  | power+standard |  |
| openml_balance_scale | uci_via_openml | standard | 3 | 0.868 +/- 0.008 (higher) | 0.867 |  |  |  |  |  |
| openml_banknote_authentication | uci_via_openml | maxabs | 3 | 0.997 +/- 0.001 (higher) | 0.968 | 0.997 |  |  |  |  |
| openml_banknote_authentication | uci_via_openml | none | 3 | 1.000 +/- 0.000 (higher) | 0.988 | 0.999 |  |  |  |  |
| openml_banknote_authentication | uci_via_openml | ranc | 3 | 1.000 +/- 0.000 (higher) | 0.979 | 0.999 |  |  |  | 1.00 |
| openml_banknote_authentication | uci_via_openml | robust | 3 | 1.000 +/- 0.000 (higher) | 0.979 | 0.999 |  |  |  |  |
| openml_banknote_authentication | uci_via_openml | selector | 3 | 1.000 +/- 0.000 (higher) | 0.995 | 1.0 |  |  | power |  |
| openml_banknote_authentication | uci_via_openml | standard | 3 | 1.000 +/- 0.000 (higher) | 0.979 | 0.999 |  |  |  |  |
| openml_wdbc | uci_via_openml | maxabs | 3 | 0.982 +/- 0.002 (higher) | 0.933 | 0.981 |  |  |  |  |
| openml_wdbc | uci_via_openml | none | 3 | 0.989 +/- 0.003 (higher) | 0.946 | 0.989 |  |  |  |  |
| openml_wdbc | uci_via_openml | quantile | 3 | 0.991 +/- 0.004 (higher) | 0.94 | 0.990 |  |  |  |  |
| openml_wdbc | uci_via_openml | ranc | 3 | 0.994 +/- 0.003 (higher) | 0.971 | 0.993 |  |  |  | 1.00 |
| openml_wdbc | uci_via_openml | robust | 3 | 0.993 +/- 0.002 (higher) | 0.965 | 0.992 |  |  |  |  |
| openml_wdbc | uci_via_openml | selector | 3 | 0.991 +/- 0.005 (higher) | 0.965 | 0.990 |  |  | minmax+robust |  |
| openml_wdbc | uci_via_openml | standard | 3 | 0.992 +/- 0.003 (higher) | 0.968 | 0.992 |  |  |  |  |
| openml_vehicle | uci_via_openml | maxabs | 3 | 0.624 +/- 0.004 (higher) | 0.624 |  |  |  |  |  |
| openml_vehicle | uci_via_openml | none | 3 | 0.789 +/- 0.012 (higher) | 0.789 |  |  |  |  |  |
| openml_vehicle | uci_via_openml | quantile | 3 | 0.709 +/- 0.005 (higher) | 0.709 |  |  |  |  |  |
| openml_vehicle | uci_via_openml | ranc | 3 | 0.781 +/- 0.012 (higher) | 0.781 |  |  |  |  | 1.00 |
| openml_vehicle | uci_via_openml | robust | 3 | 0.754 +/- 0.015 (higher) | 0.754 |  |  |  |  |  |
| openml_vehicle | uci_via_openml | selector | 3 | 0.762 +/- 0.014 (higher) | 0.762 |  |  |  | robust+standard |  |
| openml_vehicle | uci_via_openml | standard | 3 | 0.773 +/- 0.014 (higher) | 0.773 |  |  |  |  |  |
| openml_tic_tac_toe | uci_via_openml | maxabs | 3 | 0.984 +/- 0.008 (higher) | 0.966 | 0.983 |  |  |  |  |
| openml_tic_tac_toe | uci_via_openml | none | 3 | 0.984 +/- 0.008 (higher) | 0.966 | 0.983 |  |  |  |  |
| openml_tic_tac_toe | uci_via_openml | quantile | 3 | 0.984 +/- 0.008 (higher) | 0.966 | 0.983 |  |  |  |  |
| openml_tic_tac_toe | uci_via_openml | ranc | 3 | 0.984 +/- 0.008 (higher) | 0.964 | 0.983 |  |  |  | 1.00 |
| openml_tic_tac_toe | uci_via_openml | robust | 3 | 0.984 +/- 0.008 (higher) | 0.966 | 0.983 |  |  |  |  |
| openml_tic_tac_toe | uci_via_openml | selector | 3 | 0.989 +/- 0.006 (higher) | 0.977 | 0.989 |  |  | standard |  |
| openml_tic_tac_toe | uci_via_openml | standard | 3 | 0.989 +/- 0.006 (higher) | 0.977 | 0.989 |  |  |  |  |
| openml_cmc | uci_via_openml | maxabs | 3 | 0.514 +/- 0.017 (higher) | 0.513 |  |  |  |  |  |
| openml_cmc | uci_via_openml | none | 3 | 0.508 +/- 0.010 (higher) | 0.508 |  |  |  |  |  |
| openml_cmc | uci_via_openml | quantile | 3 | 0.541 +/- 0.015 (higher) | 0.541 |  |  |  |  |  |
| openml_cmc | uci_via_openml | ranc | 3 | 0.511 +/- 0.017 (higher) | 0.510 |  |  |  |  | 1.00 |
| openml_cmc | uci_via_openml | robust | 3 | 0.507 +/- 0.006 (higher) | 0.507 |  |  |  |  |  |
| openml_cmc | uci_via_openml | selector | 3 | 0.532 +/- 0.026 (higher) | 0.532 |  |  |  | minmax+quantile |  |
| openml_cmc | uci_via_openml | standard | 3 | 0.506 +/- 0.012 (higher) | 0.505 |  |  |  |  |  |

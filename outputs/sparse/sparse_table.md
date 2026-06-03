| Scaler | AUROC | Accuracy | Sparse out | Test nnz before | Test nnz after | Nnz delta | RANC policies | Sparse failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.938 | 0.833 | True | 218 | 218 | 0 |  |  |
| standard | 0.920 | 0.857 | True | 218 | 218 | 0 |  |  |
| robust | 0.938 | 0.833 | True | 218 | 218 | 0 |  |  |
| maxabs | 0.881 | 0.750 | True | 218 | 218 | 0 |  |  |
| ranc | 0.923 | 0.845 | True | 218 | 218 | 0 | maxabs:2, robust_affine:62 | 0 |

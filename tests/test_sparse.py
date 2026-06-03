import numpy as np
from scipy import sparse

from ranc_contractnet import RANCDataTransformer


def test_sparse_transform_preserves_nnz_and_shape():
    X = sparse.csr_matrix(
        np.array(
            [
                [0.0, 2.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 4.0, 5.0],
                [0.0, 0.0, 0.0],
            ]
        )
    )
    transformer = RANCDataTransformer(
        contracts={
            "global": {
                "hard_clauses": {
                    "preserve_zero": True,
                    "preserve_monotonicity": True,
                    "enforce_scale_invariance": True,
                }
            }
        }
    ).fit(X)
    Z = transformer.transform(X)
    assert sparse.issparse(Z)
    assert Z.shape == X.shape
    assert Z.nnz == X.nnz
    restored = transformer.inverse_transform(Z)
    assert restored.nnz == X.nnz
    assert np.allclose(restored.toarray(), X.toarray())


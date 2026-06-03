import numpy as np

from ranc_contractnet import RANCDataTransformer


def test_zero_and_sign_preservation_contract():
    X = np.array([[-3.0], [-1.0], [0.0], [1.0], [3.0]])
    transformer = RANCDataTransformer(
        contracts={
            "global": {
                "hard_clauses": {
                    "preserve_zero": True,
                    "preserve_sign": True,
                    "enforce_scale_invariance": True,
                    "preserve_monotonicity": True,
                    "allow_inverse_transform": True,
                }
            }
        }
    ).fit(X)
    Z = transformer.transform(X)
    assert Z[2, 0] == 0.0
    assert np.all(np.sign(Z[X[:, 0] != 0, 0]) == np.sign(X[X[:, 0] != 0, 0]))
    restored = transformer.inverse_transform(Z)
    assert np.allclose(restored, X)


def test_heavy_tail_compiles_non_identity():
    X = np.array([[1.0], [1.2], [1.3], [1.4], [100.0], [120.0], [150.0], [200.0]])
    transformer = RANCDataTransformer(
        contracts={
            "global": {
                "hard_clauses": {
                    "damp_outliers": True,
                    "preserve_monotonicity": True,
                    "allow_inverse_transform": True,
                }
            }
        }
    ).fit(X)
    policy = transformer.policies_[0]
    assert policy.policy_type != "identity"
    assert any(result.test_name == "outlier_noise_damping" for result in transformer.falsification_results_)


def test_vector_l2_policy_is_available_when_inverse_not_required():
    X = np.array([[3.0, 4.0], [0.0, 2.0], [5.0, 0.0]])
    transformer = RANCDataTransformer(
        contracts={
            "global": {
                "hard_clauses": {
                    "normalize_vector_l2": True,
                    "allow_inverse_transform": False,
                }
            }
        }
    ).fit(X)
    Z = transformer.transform(X)
    norms = np.linalg.norm(Z, axis=1)
    assert any(policy.policy_type == "vector_l2" for policy in transformer.policies_)
    assert np.allclose(norms, 1.0)

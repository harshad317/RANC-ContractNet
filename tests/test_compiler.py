import numpy as np

from ranc_contractnet.compiler import compile_contracts


def test_compiler_records_downgrade_when_hard_clauses_conflict():
    X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    result = compile_contracts(
        X,
        constraints={
            "global": {
                "hard_clauses": {
                    "preserve_zero": True,
                    "preserve_sign": True,
                    "preserve_distance_ratios": True,
                    "enforce_shift_invariance": True,
                    "enforce_scale_invariance": True,
                }
            }
        },
    )
    assert result.policies[0].policy_type == "identity"
    assert result.policies[0].downgrade_reason is not None
    assert result.rejected_candidates


def test_piecewise_policy_runs_monotonicity_falsification():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [50.0], [100.0], [200.0], [500.0]])
    result = compile_contracts(
        X,
        constraints={
            "global": {
                "hard_clauses": {
                    "damp_outliers": True,
                    "preserve_monotonicity": True,
                    "allow_inverse_transform": True,
                },
                "transform_preferences": ["piecewise_affine"],
            }
        },
    )
    assert any(r.test_name == "monotonicity" for r in result.falsification_results)
    assert all(r.passed or not r.hard_clause for r in result.falsification_results)

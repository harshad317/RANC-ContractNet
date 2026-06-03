import numpy as np

from ranc_contractnet import RANCDataTransformer, load_policy, save_policy


def test_policy_serialization_round_trip(tmp_path):
    X = np.array([[0.0, 1.0], [1.0, 10.0], [2.0, 100.0], [3.0, 1000.0]])
    transformer = RANCDataTransformer(random_state=7).fit(X)
    path = save_policy(transformer, tmp_path / "policy.json")
    loaded = load_policy(path)
    assert np.allclose(transformer.transform(X), loaded.transform(X))
    assert loaded.get_audit_report().random_state == 7


import pytest


def test_torch_layer_import_behavior():
    from ranc_contractnet.torch_layers import RANCActivationPolicy

    try:
        import torch  # noqa: F401
    except Exception:
        with pytest.raises(ImportError):
            RANCActivationPolicy({"hard_clauses": {}}, shape=4)
    else:
        layer = RANCActivationPolicy({"hard_clauses": {"avoid_batch_dependence": True}}, shape=4)
        assert layer is not None


from ranc_contractnet.schemas import InvarianceContract, RegimeCard


def test_schema_round_trip_json():
    card = RegimeCard(
        feature_name="x0",
        feature_index=0,
        n_samples=10,
        missing_fraction=0.0,
        zero_fraction=0.1,
        positive_fraction=0.9,
        negative_fraction=0.0,
        robust_location=1.0,
        robust_scale=2.0,
        mean=1.2,
        std=0.5,
        min=0.0,
        max=4.0,
        q01=0.0,
        q25=0.5,
        q50=1.0,
        q75=2.0,
        q99=4.0,
        skewness=0.5,
        tail_category="stable",
        sign_pattern="nonnegative",
    )
    restored = RegimeCard.model_validate_json(card.model_dump_json())
    assert restored == card


def test_contract_helpers():
    contract = InvarianceContract(
        feature_name="x0",
        hard_clauses={"preserve_zero": True},
        soft_clauses={"prefer_interpretability": 0.7},
    )
    assert contract.hard("preserve_zero") is True
    assert contract.hard("missing") is False
    assert contract.soft("prefer_interpretability") == 0.7


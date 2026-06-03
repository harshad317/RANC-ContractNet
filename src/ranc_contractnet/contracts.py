"""Contract derivation and normalization of user-supplied constraints."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from ranc_contractnet.schemas import InvarianceContract, RegimeCard


ConstraintLike = Union[InvarianceContract, Mapping[str, Any]]


SCALING_SENSITIVE_MODELS = {"linear", "logistic", "svm", "knn", "mlp", "neural", "ridge", "lasso"}
SCALING_INSENSITIVE_MODELS = {"tree", "random_forest", "xgboost", "lightgbm", "catboost", "gbdt"}


def _model_family_key(model_family: Optional[str]) -> str:
    if not model_family:
        return "linear"
    key = str(model_family).lower().replace("-", "_")
    if key in SCALING_INSENSITIVE_MODELS:
        return "tree"
    if key in SCALING_SENSITIVE_MODELS:
        return key
    return key


def _constraint_for_feature(
    constraints: Optional[Union[Sequence[ConstraintLike], Mapping[str, Any]]],
    feature_name: str,
) -> Dict[str, Any]:
    if constraints is None:
        return {}
    if isinstance(constraints, Sequence) and not isinstance(constraints, (str, bytes, Mapping)):
        for item in constraints:
            if isinstance(item, InvarianceContract) and item.feature_name == feature_name:
                return item.model_dump()
            if isinstance(item, Mapping) and item.get("feature_name") == feature_name:
                return dict(item)
        return {}
    if isinstance(constraints, Mapping):
        merged: Dict[str, Any] = {}
        global_constraints = constraints.get("global")
        if isinstance(global_constraints, Mapping):
            merged.update(global_constraints)
        feature_constraints = constraints.get(feature_name)
        if isinstance(feature_constraints, Mapping):
            merged.update(feature_constraints)
        return merged
    return {}


def derive_invariance_contracts(
    cards: Iterable[RegimeCard],
    model_family: Optional[str] = None,
    constraints: Optional[Union[Sequence[ConstraintLike], Mapping[str, Any]]] = None,
    random_state: int = 0,
) -> List[InvarianceContract]:
    """Derive safe default contracts and apply user overrides."""

    family = _model_family_key(model_family)
    contracts: List[InvarianceContract] = []
    for card in cards:
        hard: Dict[str, bool] = {
            "preserve_monotonicity": True,
            "allow_inverse_transform": bool(card.inverse_required),
            "preserve_zero": bool(card.sparse or card.zero_fraction >= 0.05),
            "preserve_sign": bool(card.interpretability_required and card.sign_pattern == "mixed"),
            "avoid_batch_dependence": True,
        }
        soft: Dict[str, float] = {
            "prefer_low_complexity": 1.0,
            "prefer_interpretability": 1.0 if card.interpretability_required else 0.3,
        }
        if family != "tree":
            hard["enforce_scale_invariance"] = True
            if not hard["preserve_zero"]:
                hard["enforce_shift_invariance"] = True
        if card.tail_category == "heavy_tail":
            hard["damp_outliers"] = True
            soft["preserve_extreme_signal"] = 0.5
        if card.tail_category == "skewed_positive" and card.min >= 0:
            soft["prefer_monotone_compression"] = 0.7
        if card.bounded_low is not None and card.bounded_high is not None:
            soft["prefer_semantic_bounds"] = 0.8
        if card.drift_estimate > 1.0:
            soft["prefer_drift_monitor"] = 1.0

        override = _constraint_for_feature(constraints, card.feature_name)
        hard.update({k: bool(v) for k, v in dict(override.get("hard_clauses", {})).items()})
        soft.update({k: float(v) for k, v in dict(override.get("soft_clauses", {})).items()})
        forbidden = list(override.get("forbidden_distortions", []))
        preferences = list(override.get("transform_preferences", []))
        metadata = dict(override.get("metadata", {}))
        contract = InvarianceContract(
            feature_name=card.feature_name,
            group_name=override.get("group_name"),
            hard_clauses=hard,
            soft_clauses=soft,
            forbidden_distortions=forbidden,
            transform_preferences=preferences,
            metadata=metadata,
            seed=int(override.get("seed", random_state)),
            audit_notes=[
                f"default contract derived for model_family={family}",
                "user overrides applied" if override else "no user overrides",
            ],
        )
        contracts.append(contract)
    return contracts

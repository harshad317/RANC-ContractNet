"""Hybrid contract compiler for RANC-ContractNet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy import stats

from ranc_contractnet.cards import build_regime_cards
from ranc_contractnet.contracts import derive_invariance_contracts
from ranc_contractnet.falsification import run_policy_falsification
from ranc_contractnet.ledger import build_signal_risk_ledger
from ranc_contractnet.policies import (
    affine_policy,
    apply_policy_to_vector,
    identity_policy,
    log1p_policy,
    minmax_policy,
    piecewise_affine_policy,
    policy_maps_zero_to_zero,
    policy_preserves_sign,
    power_policy,
    quantile_rank_policy,
    vector_norm_policy,
    whitening_policy,
)
from ranc_contractnet.schemas import (
    FalsificationResult,
    InvarianceContract,
    NormalizationPolicy,
    RegimeCard,
    SignalRiskLedgerRow,
)
from ranc_contractnet.utils import column_as_dense, finite_values, infer_feature_names, is_sparse_matrix, stable_scale, to_numpy_2d


@dataclass
class CompilationResult:
    """Compiler output consumed by the sklearn transformer and experiment runners."""

    policies: List[NormalizationPolicy]
    cards: List[RegimeCard]
    contracts: List[InvarianceContract]
    ledger_rows: List[SignalRiskLedgerRow]
    falsification_results: List[FalsificationResult]
    rejected_candidates: List[Dict[str, Any]]
    warnings: List[str]


def _estimate_power_candidate(card: RegimeCard, values: np.ndarray, seed: int) -> List[NormalizationPolicy]:
    vals = finite_values(values)
    if vals.size < 8:
        return []
    policies: List[NormalizationPolicy] = []
    if card.min > 0:
        try:
            lmbda = float(stats.boxcox_normmax(vals, method="mle"))
            transformed = stats.boxcox(vals, lmbda=lmbda)
            policies.append(
                power_policy(
                    card,
                    kind="boxcox",
                    lmbda=lmbda,
                    center=float(np.mean(transformed)),
                    scale=stable_scale(float(np.std(transformed, ddof=1))),
                    seed=seed,
                )
            )
        except Exception:
            pass
    try:
        lmbda = float(stats.yeojohnson_normmax(vals))
        transformed = stats.yeojohnson(vals, lmbda=lmbda)
        policies.append(
            power_policy(
                card,
                kind="yeo_johnson",
                lmbda=lmbda,
                center=float(np.mean(transformed)),
                scale=stable_scale(float(np.std(transformed, ddof=1))),
                seed=seed,
            )
        )
    except Exception:
        pass
    return policies


def _candidate_policies(
    card: RegimeCard,
    contract: InvarianceContract,
    values: np.ndarray,
    seed: int,
) -> List[NormalizationPolicy]:
    abs_scale = max(abs(card.q01), abs(card.q99), abs(card.min), abs(card.max), 1.0)
    candidates: List[NormalizationPolicy] = [identity_policy(card, seed=seed)]
    maxabs_audit_cost = 1.6 if card.tail_category == "heavy_tail" and not card.sparse else 0.05
    candidates.append(
        affine_policy(
            card,
            policy_type="maxabs",
            center=0.0,
            scale=abs_scale,
            complexity=1,
            audit_cost=maxabs_audit_cost,
            seed=seed,
        )
    )
    candidates.append(
        affine_policy(
            card,
            policy_type="zscore",
            center=card.mean,
            scale=card.std,
            complexity=2,
            audit_cost=0.15,
            seed=seed,
        )
    )
    robust_center = 0.0 if contract.hard("preserve_zero") else card.robust_location
    candidates.append(
        affine_policy(
            card,
            policy_type="robust_affine",
            center=robust_center,
            scale=card.robust_scale,
            complexity=2,
            audit_cost=0.2,
            seed=seed,
        )
    )
    if card.max > card.min:
        candidates.append(minmax_policy(card, seed=seed))
    if card.min >= 0 and (
        card.tail_category in {"skewed_positive", "heavy_tail"} or contract.hard("damp_outliers")
    ):
        candidates.append(log1p_policy(card, seed=seed))
    if not contract.hard("preserve_zero") and not contract.hard("preserve_sign"):
        candidates.extend(_estimate_power_candidate(card, values, seed))
    if contract.hard("preserve_rank") or (
        contract.hard("damp_outliers") and not contract.hard("preserve_extreme_signal")
    ):
        candidates.append(quantile_rank_policy(card, values, seed=seed))
    if contract.hard("damp_outliers") or "piecewise_affine" in contract.transform_preferences:
        candidates.append(piecewise_affine_policy(card, seed=seed))
    return candidates


def _policy_valid_on_training(policy: NormalizationPolicy, values: np.ndarray) -> tuple[bool, str]:
    try:
        transformed = apply_policy_to_vector(values, policy)
    except Exception as exc:
        return False, f"cannot transform training values: {exc}"
    if not np.all(np.isfinite(transformed[np.isfinite(values)])):
        return False, "non-finite transformed values"
    return True, "ok"


def _supports_hard_clauses(
    policy: NormalizationPolicy,
    card: RegimeCard,
    contract: InvarianceContract,
    values: np.ndarray,
) -> tuple[bool, str]:
    forbidden = {name.lower() for name in contract.forbidden_distortions}
    if policy.policy_type.lower() in forbidden:
        return False, "policy type is forbidden by contract"
    valid, reason = _policy_valid_on_training(policy, values)
    if not valid:
        return False, reason
    if contract.hard("allow_inverse_transform") and not policy.inverse_legal:
        return False, "inverse transform required"
    if contract.hard("preserve_zero") and not policy_maps_zero_to_zero(policy):
        return False, "zero preservation required"
    if contract.hard("preserve_sign") and not policy_preserves_sign(policy, card):
        return False, "sign preservation required"
    if contract.hard("enforce_scale_invariance") and policy.policy_type == "identity":
        return False, "scale invariance required"
    if contract.hard("enforce_shift_invariance"):
        center = float(policy.parameters.get("center", 0.0))
        shift_capable = policy.policy_type in {
            "zscore",
            "robust_affine",
            "minmax",
            "power_boxcox",
            "power_yeo_johnson",
            "quantile_rank",
            "piecewise_affine",
        } and (abs(center) > 1e-12 or policy.policy_type not in {"robust_affine"})
        if not shift_capable:
            return False, "shift invariance required"
    if contract.hard("damp_outliers") and policy.policy_type not in {
        "robust_affine",
        "log1p",
        "power_boxcox",
        "power_yeo_johnson",
        "quantile_rank",
        "piecewise_affine",
    }:
        return False, "outlier damping required"
    if contract.hard("preserve_distance_ratios") and policy.policy_type not in {"identity", "maxabs"}:
        return False, "distance-ratio preservation required"
    if contract.hard("preserve_extreme_signal") and policy.policy_type in {"quantile_rank", "bounded_tanh"}:
        return False, "extreme signal preservation forbids rank saturation"
    return True, "ok"


def _select_policy(
    card: RegimeCard,
    contract: InvarianceContract,
    values: np.ndarray,
    seed: int,
) -> tuple[NormalizationPolicy, List[Dict[str, Any]]]:
    candidates = _candidate_policies(card, contract, values, seed)
    rejected: List[Dict[str, Any]] = []
    admissible: List[NormalizationPolicy] = []
    for candidate in candidates:
        ok, reason = _supports_hard_clauses(candidate, card, contract, values)
        if ok:
            admissible.append(candidate)
        else:
            rejected.append(
                {
                    "feature_name": card.feature_name,
                    "policy_type": candidate.policy_type,
                    "reason": reason,
                    "score": candidate.score(),
                }
            )
    if not admissible:
        downgrade = "no candidate satisfied all hard clauses; downgraded to no-op"
        policy = identity_policy(card, seed=seed, reason=downgrade)
        policy.rejected_candidates = rejected
        return policy, rejected
    preference_order = {
        "identity": 0,
        "maxabs": 1,
        "zscore": 2,
        "robust_affine": 3,
        "minmax": 4,
        "log1p": 5,
        "power_boxcox": 6,
        "power_yeo_johnson": 7,
        "piecewise_affine": 8,
        "quantile_rank": 9,
    }
    preferences = list(contract.transform_preferences)

    def selection_score(policy: NormalizationPolicy) -> float:
        score = policy.score()
        if policy.policy_type in preferences:
            score -= 1.5
        return score

    selected = sorted(
        admissible,
        key=lambda p: (selection_score(p), preference_order.get(p.policy_type, 99), p.policy_type),
    )[0]
    selected.rejected_candidates = rejected
    return selected, rejected


def _maybe_group_policies(
    X: object,
    contracts: Sequence[InvarianceContract],
    feature_names: List[str],
    random_state: int,
) -> tuple[List[NormalizationPolicy], List[str]]:
    policies: List[NormalizationPolicy] = []
    warnings: List[str] = []
    n_samples = int(X.shape[0]) if hasattr(X, "shape") else to_numpy_2d(X, copy=False).shape[0]
    if any(
        contract.hard("normalize_vector_l1") or contract.hard("vector_norm_l1_is_nuisance")
        for contract in contracts
    ):
        policies.append(vector_norm_policy("l1", feature_names, n_samples, seed=random_state))
    if any(
        contract.hard("normalize_vector_l2") or contract.hard("vector_norm_l2_is_nuisance")
        for contract in contracts
    ):
        policies.append(vector_norm_policy("l2", feature_names, n_samples, seed=random_state))
    if not any(contract.hard("reduce_covariance") for contract in contracts):
        return policies, warnings
    if is_sparse_matrix(X):
        warnings.append("reduce_covariance requested on sparse input; whitening skipped to avoid densification")
        return policies, warnings
    arr = to_numpy_2d(X, copy=False)
    if arr.shape[0] <= arr.shape[1]:
        warnings.append("reduce_covariance requested but covariance estimate is underdetermined; whitening skipped")
        return policies, warnings
    cov = np.cov(arr, rowvar=False)
    policies.append(whitening_policy(cov, feature_names, arr.shape[0], seed=random_state))
    return policies, warnings


def _signal_risk_ledger_enabled(contracts: Sequence[InvarianceContract]) -> bool:
    return not any(bool(contract.metadata.get("disable_signal_risk_ledger", False)) for contract in contracts)


def compile_contracts(
    X: object,
    y: Optional[object] = None,
    metadata: Optional[Dict[str, Any]] = None,
    constraints: Optional[object] = None,
    model_family: Optional[str] = None,
    random_state: int = 0,
) -> CompilationResult:
    """Compile normalization policies from training data and invariance constraints.

    This function intentionally does not inspect validation/test scores. It only uses
    `X`, optional `y`, metadata, and contract clauses supplied to the active `fit`.
    """

    feature_names = infer_feature_names(X)
    cards = build_regime_cards(X, metadata=metadata, feature_names=feature_names)
    contracts = derive_invariance_contracts(
        cards,
        model_family=model_family,
        constraints=constraints,
        random_state=random_state,
    )
    warnings: List[str] = []
    if _signal_risk_ledger_enabled(contracts):
        ledger_rows = build_signal_risk_ledger(X, y, cards, contracts)
    else:
        ledger_rows = []
        warnings.append("signal risk ledger disabled by contract metadata")
    policies: List[NormalizationPolicy] = []
    rejected_candidates: List[Dict[str, Any]] = []
    falsification_results: List[FalsificationResult] = []
    contract_by_name = {contract.feature_name: contract for contract in contracts}

    for card in cards:
        values = column_as_dense(X, card.feature_index)
        policy, rejected = _select_policy(card, contract_by_name[card.feature_name], values, random_state)
        policies.append(policy)
        rejected_candidates.extend(rejected)
        falsification_results.extend(
            run_policy_falsification(values, card, contract_by_name[card.feature_name], policy)
        )

    group_policies, group_warnings = _maybe_group_policies(X, contracts, feature_names, random_state)
    warnings.extend(group_warnings)
    policies.extend(group_policies)
    if sparse.issparse(X) and any(policy.policy_type == "whitening" for policy in policies):
        warnings.append("internal error guard: sparse whitening policy removed")
        policies = [policy for policy in policies if policy.policy_type != "whitening"]

    return CompilationResult(
        policies=policies,
        cards=cards,
        contracts=contracts,
        ledger_rows=ledger_rows,
        falsification_results=falsification_results,
        rejected_candidates=rejected_candidates,
        warnings=warnings,
    )

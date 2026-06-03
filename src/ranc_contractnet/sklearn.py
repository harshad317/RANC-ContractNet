"""Scikit-learn compatible RANC-ContractNet transformer."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize as sparse_normalize
from sklearn.utils.validation import check_is_fitted

from ranc_contractnet.compiler import compile_contracts
from ranc_contractnet.policies import apply_policy_to_vector, inverse_policy_to_vector, policy_maps_zero_to_zero
from ranc_contractnet.schemas import (
    AuditReport,
    FalsificationResult,
    InvarianceContract,
    NormalizationPolicy,
    RegimeCard,
    SignalRiskLedgerRow,
)
from ranc_contractnet.utils import clone_sparse_for_column_ops, ensure_2d_shape, infer_feature_names, to_numpy_2d


class RANCDataTransformer(BaseEstimator, TransformerMixin):
    """Leakage-safe contract-compiled normalizer.

    `fit` is the only method that computes statistics. `transform` replays serialized
    policies and never adapts to validation/test data.
    """

    def __init__(
        self,
        contracts: Optional[object] = None,
        model_family: Optional[str] = None,
        random_state: int = 0,
        strict: bool = True,
        audit: bool = True,
    ) -> None:
        self.contracts = contracts
        self.model_family = model_family
        self.random_state = random_state
        self.strict = strict
        self.audit = audit

    def fit(self, X: object, y: Optional[object] = None, metadata: Optional[Dict[str, Any]] = None):
        n_samples, n_features = ensure_2d_shape(X)
        self.n_features_in_ = n_features
        self.n_samples_fit_ = n_samples
        self.feature_names_in_ = infer_feature_names(X)
        result = compile_contracts(
            X,
            y=y,
            metadata=metadata,
            constraints=self.contracts,
            model_family=self.model_family,
            random_state=int(self.random_state),
        )
        self.policies_ = result.policies
        self.cards_ = result.cards
        self.contracts_ = result.contracts
        self.ledger_rows_ = result.ledger_rows
        self.falsification_results_ = result.falsification_results
        self.rejected_candidates_ = result.rejected_candidates
        self.warnings_ = result.warnings
        hard_failures = [
            res for res in self.falsification_results_ if res.hard_clause and not res.passed
        ]
        if hard_failures and self.strict:
            self.warnings_.append(
                f"{len(hard_failures)} hard-clause falsification failures recorded; policies downgraded where possible"
            )
        self.audit_report_ = self._build_audit_report(metadata=metadata or {})
        return self

    def _build_audit_report(self, metadata: Optional[Dict[str, Any]] = None) -> AuditReport:
        return AuditReport(
            random_state=int(self.random_state),
            feature_names=list(getattr(self, "feature_names_in_", [])),
            policies=list(getattr(self, "policies_", [])),
            cards=list(getattr(self, "cards_", [])),
            contracts=list(getattr(self, "contracts_", [])),
            ledger_rows=list(getattr(self, "ledger_rows_", [])),
            falsification_results=list(getattr(self, "falsification_results_", [])),
            rejected_candidates=list(getattr(self, "rejected_candidates_", [])),
            warnings=list(getattr(self, "warnings_", [])),
            metadata={
                "n_samples_fit": int(getattr(self, "n_samples_fit_", 0)),
                "n_features_in": int(getattr(self, "n_features_in_", 0)),
                "model_family": self.model_family,
                **(metadata or {}),
            },
        )

    def _feature_policies(self) -> List[NormalizationPolicy]:
        return [policy for policy in self.policies_ if policy.feature_index >= 0]

    def _group_policies(self) -> List[NormalizationPolicy]:
        return [policy for policy in self.policies_ if policy.feature_index < 0]

    def _validate_transform_shape(self, X: object) -> None:
        _, n_features = ensure_2d_shape(X)
        if n_features != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {n_features}.")

    def transform(self, X: object):
        check_is_fitted(self, "policies_")
        self._validate_transform_shape(X)
        if sparse.issparse(X):
            return self._transform_sparse(X)
        arr = to_numpy_2d(X, copy=True)
        for policy in self._feature_policies():
            arr[:, policy.feature_index] = apply_policy_to_vector(arr[:, policy.feature_index], policy)
        for policy in self._group_policies():
            if policy.policy_type == "whitening":
                matrix = np.asarray(policy.parameters["matrix"], dtype=float)
                arr = arr @ matrix
            elif policy.policy_type in {"vector_l1", "vector_l2"}:
                ord_value = 1 if policy.policy_type == "vector_l1" else 2
                norms = np.linalg.norm(arr, ord=ord_value, axis=1, keepdims=True)
                arr = arr / np.maximum(norms, 1e-12)
        return arr

    def _transform_sparse(self, X: object):
        group_policies = self._group_policies()
        out = clone_sparse_for_column_ops(X)
        for policy in self._feature_policies():
            if not policy_maps_zero_to_zero(policy):
                raise ValueError(f"Policy {policy.policy_type} would densify sparse zeros.")
            start, end = out.indptr[policy.feature_index], out.indptr[policy.feature_index + 1]
            if end > start:
                out.data[start:end] = apply_policy_to_vector(out.data[start:end], policy)
        for policy in group_policies:
            if policy.policy_type == "whitening":
                raise ValueError("Whitening cannot be applied to sparse inputs.")
            if policy.policy_type in {"vector_l1", "vector_l2"}:
                norm = "l1" if policy.policy_type == "vector_l1" else "l2"
                out = sparse_normalize(out.tocsr(), norm=norm, axis=1, copy=False).tocsc()
        return out.asformat(X.getformat())

    def inverse_transform(self, X: object):
        check_is_fitted(self, "policies_")
        self._validate_transform_shape(X)
        if sparse.issparse(X):
            return self._inverse_sparse(X)
        arr = to_numpy_2d(X, copy=True)
        for policy in reversed(self._group_policies()):
            if policy.policy_type == "whitening":
                matrix = np.asarray(policy.parameters["inverse_matrix"], dtype=float)
                arr = arr @ matrix
            elif policy.policy_type in {"vector_l1", "vector_l2"}:
                raise ValueError(f"Policy {policy.policy_type} has no legal inverse.")
        for policy in reversed(self._feature_policies()):
            arr[:, policy.feature_index] = inverse_policy_to_vector(arr[:, policy.feature_index], policy)
        return arr

    def _inverse_sparse(self, X: object):
        if self._group_policies():
            names = ", ".join(policy.policy_type for policy in self._group_policies())
            raise ValueError(f"Group policies cannot be inverted on sparse inputs: {names}.")
        out = clone_sparse_for_column_ops(X)
        for policy in reversed(self._feature_policies()):
            if not policy_maps_zero_to_zero(policy):
                raise ValueError(f"Policy {policy.policy_type} would densify sparse zeros.")
            start, end = out.indptr[policy.feature_index], out.indptr[policy.feature_index + 1]
            if end > start:
                out.data[start:end] = inverse_policy_to_vector(out.data[start:end], policy)
        return out.asformat(X.getformat())

    def get_audit_report(self) -> AuditReport:
        check_is_fitted(self, "policies_")
        self.audit_report_ = self._build_audit_report(metadata=self.audit_report_.metadata)
        return self.audit_report_

    def to_json(self, **kwargs: Any) -> str:
        check_is_fitted(self, "policies_")
        payload = {
            "class": self.__class__.__name__,
            "version": "0.1.0",
            "init_params": {
                "contracts": self.contracts,
                "model_family": self.model_family,
                "random_state": self.random_state,
                "strict": self.strict,
                "audit": self.audit,
            },
            "audit_report": self.get_audit_report().model_dump(mode="json"),
        }
        import json

        return json.dumps(payload, sort_keys=True, **kwargs)

    @classmethod
    def from_json(cls, payload: str) -> "RANCDataTransformer":
        import json

        raw = json.loads(payload)
        params = raw.get("init_params", {})
        obj = cls(**params)
        report = AuditReport.model_validate(raw["audit_report"])
        obj.feature_names_in_ = list(report.feature_names)
        obj.n_features_in_ = len(obj.feature_names_in_)
        obj.n_samples_fit_ = int(report.metadata.get("n_samples_fit", 0))
        obj.policies_ = [
            policy if isinstance(policy, NormalizationPolicy) else NormalizationPolicy.model_validate(policy)
            for policy in report.policies
        ]
        obj.cards_ = [
            card if isinstance(card, RegimeCard) else RegimeCard.model_validate(card)
            for card in report.cards
        ]
        obj.contracts_ = [
            contract if isinstance(contract, InvarianceContract) else InvarianceContract.model_validate(contract)
            for contract in report.contracts
        ]
        obj.ledger_rows_ = [
            row if isinstance(row, SignalRiskLedgerRow) else SignalRiskLedgerRow.model_validate(row)
            for row in report.ledger_rows
        ]
        obj.falsification_results_ = [
            result if isinstance(result, FalsificationResult) else FalsificationResult.model_validate(result)
            for result in report.falsification_results
        ]
        obj.rejected_candidates_ = list(report.rejected_candidates)
        obj.warnings_ = list(report.warnings)
        obj.audit_report_ = report
        return obj


def policy_summary(policies: Iterable[NormalizationPolicy]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for policy in policies:
        summary[policy.policy_type] = summary.get(policy.policy_type, 0) + 1
    return summary

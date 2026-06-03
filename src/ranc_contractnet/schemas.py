"""Serializable research objects for RANC-ContractNet."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


SCHEMA_VERSION = "0.1.0"


class ContractNetModel(BaseModel):
    """Base class with strict, JSON-friendly Pydantic behavior."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def to_json(self, **kwargs: Any) -> str:
        return self.model_dump_json(**kwargs)


class RegimeCard(ContractNetModel):
    """Training-only profile of one feature, feature group, or activation stream."""

    schema_version: str = SCHEMA_VERSION
    feature_name: str
    feature_index: int
    n_samples: int
    missing_fraction: float
    zero_fraction: float
    positive_fraction: float
    negative_fraction: float
    sparse: bool = False
    bounded_low: Optional[float] = None
    bounded_high: Optional[float] = None
    robust_location: float
    robust_scale: float
    mean: float
    std: float
    min: float
    max: float
    q01: float
    q25: float
    q50: float
    q75: float
    q99: float
    skewness: float
    tail_category: str
    sign_pattern: str
    drift_estimate: float = 0.0
    batch_reliability: float = 1.0
    covariance_relevance: bool = False
    distance_sensitivity: bool = True
    interpretability_required: bool = False
    inverse_required: bool = True
    confidence: float = 1.0
    notes: List[str] = Field(default_factory=list)

    @field_validator(
        "missing_fraction",
        "zero_fraction",
        "positive_fraction",
        "negative_fraction",
        "confidence",
        mode="after",
    )
    @classmethod
    def _fraction(cls, value: float) -> float:
        return min(1.0, max(0.0, float(value)))


class InvarianceContract(ContractNetModel):
    """Human-readable invariance and preservation obligations for one feature/group."""

    schema_version: str = SCHEMA_VERSION
    feature_name: str
    group_name: Optional[str] = None
    hard_clauses: Dict[str, bool] = Field(default_factory=dict)
    soft_clauses: Dict[str, float] = Field(default_factory=dict)
    forbidden_distortions: List[str] = Field(default_factory=list)
    transform_preferences: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    seed: int = 0
    audit_notes: List[str] = Field(default_factory=list)

    def hard(self, name: str, default: bool = False) -> bool:
        return bool(self.hard_clauses.get(name, default))

    def soft(self, name: str, default: float = 0.0) -> float:
        return float(self.soft_clauses.get(name, default))


class SignalRiskLedgerRow(ContractNetModel):
    """Audit row describing signal that a proposed nuisance-removal step may destroy."""

    schema_version: str = SCHEMA_VERSION
    feature_name: str
    proposed_nuisance: str
    possible_signal_removed: str
    evidence: str
    severity: float = 0.0
    confidence: float = 0.0
    mitigation: str = "audit_only"
    falsification_test: str = "none"
    supervised_label_used: bool = False

    @field_validator("severity", "confidence", mode="after")
    @classmethod
    def _unit_interval(cls, value: float) -> float:
        return min(1.0, max(0.0, float(value)))


class DriftMonitor(ContractNetModel):
    """Serializable monitor used to flag inference-time distribution mismatch."""

    schema_version: str = SCHEMA_VERSION
    feature_name: str
    statistic: str = "median"
    train_low: float
    train_high: float
    action: str = "warn"
    parameters: Dict[str, Any] = Field(default_factory=dict)


class NormalizationPolicy(ContractNetModel):
    """Compiled transform program and its fitted parameters."""

    schema_version: str = SCHEMA_VERSION
    feature_name: str
    feature_index: int
    policy_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    formula: str = "identity"
    inverse_formula: Optional[str] = "identity"
    inverse_legal: bool = True
    no_op: bool = False
    complexity: int = 0
    audit_cost: float = 0.0
    drift_monitor: Optional[DriftMonitor] = None
    rejected_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    downgrade_reason: Optional[str] = None
    fitted_on_n_samples: int = 0
    seed: int = 0
    audit_notes: List[str] = Field(default_factory=list)

    def score(self, lambda_audit: float = 1.0) -> float:
        return float(self.complexity) + lambda_audit * float(self.audit_cost)


class FalsificationResult(ContractNetModel):
    """Result of one targeted policy falsification test."""

    schema_version: str = SCHEMA_VERSION
    feature_name: str
    policy_type: str
    test_name: str
    passed: bool
    severity: float = 0.0
    message: str = ""
    metrics: Dict[str, Any] = Field(default_factory=dict)
    hard_clause: bool = False


class AuditReport(ContractNetModel):
    """Complete replayable audit artifact for a fitted transformer."""

    schema_version: str = SCHEMA_VERSION
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )
    random_state: int = 0
    transformer_type: str = "RANCDataTransformer"
    feature_names: List[str] = Field(default_factory=list)
    policies: List[NormalizationPolicy] = Field(default_factory=list)
    cards: List[RegimeCard] = Field(default_factory=list)
    contracts: List[InvarianceContract] = Field(default_factory=list)
    ledger_rows: List[SignalRiskLedgerRow] = Field(default_factory=list)
    falsification_results: List[FalsificationResult] = Field(default_factory=list)
    rejected_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(result.passed or not result.hard_clause for result in self.falsification_results)


def model_from_json(model_cls: type[ContractNetModel], payload: str) -> ContractNetModel:
    """Parse a JSON string into a typed ContractNet model."""

    return model_cls.model_validate_json(payload)


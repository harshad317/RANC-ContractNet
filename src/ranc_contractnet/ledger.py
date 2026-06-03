"""Signal Risk Ledger construction."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from ranc_contractnet.schemas import InvarianceContract, RegimeCard, SignalRiskLedgerRow
from ranc_contractnet.utils import column_as_dense, finite_values


def _tail_label_evidence(x: np.ndarray, y: Optional[object]) -> tuple[str, float, bool]:
    if y is None:
        return "no supervised labels supplied; cannot infer whether extremes are signal", 0.5, False
    y_arr = np.asarray(y)
    if y_arr.shape[0] != x.shape[0]:
        return "label length mismatch; supervised signal check skipped", 0.5, False
    finite = np.isfinite(x)
    if not np.any(finite):
        return "feature has no finite values", 0.0, True
    q99 = np.quantile(x[finite], 0.99)
    tail_mask = finite & (x >= q99)
    if int(np.sum(tail_mask)) < 2:
        return "too few tail samples for supervised evidence", 0.5, True
    try:
        y_float = y_arr.astype(float)
        global_mean = float(np.nanmean(y_float))
        tail_mean = float(np.nanmean(y_float[tail_mask]))
        delta = abs(tail_mean - global_mean) / (float(np.nanstd(y_float)) + 1e-12)
        evidence = f"top-tail label mean shift={delta:.3f} standard deviations"
        severity = min(1.0, delta / 2.0)
        return evidence, severity, True
    except Exception:
        tail_rate = float(np.mean(y_arr[tail_mask] == y_arr[tail_mask][0]))
        return f"categorical tail label concentration={tail_rate:.3f}", min(1.0, tail_rate), True


def build_signal_risk_ledger(
    X: object,
    y: Optional[object],
    cards: Iterable[RegimeCard],
    contracts: Iterable[InvarianceContract],
) -> List[SignalRiskLedgerRow]:
    """Create one ledger row for every policy choice that can destroy signal."""

    contract_by_name = {contract.feature_name: contract for contract in contracts}
    rows: List[SignalRiskLedgerRow] = []
    for card in cards:
        contract = contract_by_name[card.feature_name]
        if contract.hard("damp_outliers") or contract.soft("preserve_extreme_signal") > 0:
            x = column_as_dense(X, card.feature_index)
            evidence, severity, supervised = _tail_label_evidence(x, y)
            mitigation = "prefer_rank_preserving_or_piecewise_policy"
            if contract.hard("preserve_extreme_signal"):
                mitigation = "forbid_clipping_and_quantile_saturation"
            rows.append(
                SignalRiskLedgerRow(
                    feature_name=card.feature_name,
                    proposed_nuisance="heavy_tail_or_extreme_values",
                    possible_signal_removed="rare predictive tail events",
                    evidence=evidence,
                    severity=severity,
                    confidence=max(0.2, card.confidence),
                    mitigation=mitigation,
                    falsification_test="outlier_signal_preservation",
                    supervised_label_used=supervised,
                )
            )
        if contract.hard("enforce_shift_invariance") and card.interpretability_required:
            rows.append(
                SignalRiskLedgerRow(
                    feature_name=card.feature_name,
                    proposed_nuisance="feature_location",
                    possible_signal_removed="raw-unit interpretability",
                    evidence="metadata marks interpretability_required=True",
                    severity=0.6,
                    confidence=card.confidence,
                    mitigation="require inverse_transform and audit formula",
                    falsification_test="inverse_transform_error",
                    supervised_label_used=False,
                )
            )
        finite = finite_values(column_as_dense(X, card.feature_index))
        if finite.size and contract.hard("preserve_zero") and not np.isclose(0.0, card.robust_location):
            rows.append(
                SignalRiskLedgerRow(
                    feature_name=card.feature_name,
                    proposed_nuisance="centering",
                    possible_signal_removed="semantic zero meaning",
                    evidence=f"zero_fraction={card.zero_fraction:.3f}; median={card.robust_location:.3f}",
                    severity=min(1.0, card.zero_fraction + 0.2),
                    confidence=card.confidence,
                    mitigation="use zero-preserving scale-only policy",
                    falsification_test="zero_preservation",
                    supervised_label_used=False,
                )
            )
    return rows


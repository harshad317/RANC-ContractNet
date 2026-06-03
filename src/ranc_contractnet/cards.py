"""Training-only regime card construction."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

from ranc_contractnet.schemas import RegimeCard
from ranc_contractnet.utils import column_as_dense, finite_values, infer_feature_names, is_sparse_matrix


def _skewness(values: np.ndarray) -> float:
    vals = finite_values(values)
    if vals.size < 3:
        return 0.0
    centered = vals - float(np.mean(vals))
    std = float(np.std(vals))
    if std <= 1e-12:
        return 0.0
    return float(np.mean((centered / std) ** 3))


def _tail_category(values: np.ndarray, q25: float, q75: float, q99: float, q01: float) -> str:
    vals = finite_values(values)
    if vals.size < 8:
        return "low_confidence"
    iqr = max(q75 - q25, 1e-12)
    upper_tail = (q99 - q75) / iqr
    lower_tail = (q25 - q01) / iqr
    skew = _skewness(vals)
    if max(upper_tail, lower_tail) >= 6.0:
        return "heavy_tail"
    if skew >= 1.0:
        return "skewed_positive"
    if skew <= -1.0:
        return "skewed_negative"
    return "stable"


def _sign_pattern(values: np.ndarray) -> str:
    vals = finite_values(values)
    if vals.size == 0 or np.all(np.abs(vals) <= 1e-12):
        return "zero_only"
    has_pos = bool(np.any(vals > 0))
    has_neg = bool(np.any(vals < 0))
    if has_pos and has_neg:
        return "mixed"
    if has_pos:
        return "nonnegative"
    return "nonpositive"


def _bounded_metadata(
    metadata: Optional[Mapping[str, object]], feature_name: str
) -> tuple[Optional[float], Optional[float], List[str]]:
    notes: List[str] = []
    if not metadata:
        return None, None, notes
    bounds = metadata.get("bounds") if isinstance(metadata, Mapping) else None
    if isinstance(bounds, Mapping):
        feature_bounds = bounds.get(feature_name)
        if isinstance(feature_bounds, (list, tuple)) and len(feature_bounds) == 2:
            notes.append("semantic bounds supplied by metadata")
            return float(feature_bounds[0]), float(feature_bounds[1]), notes
    return None, None, notes


def build_regime_cards(
    X: object,
    metadata: Optional[Mapping[str, object]] = None,
    feature_names: Optional[Iterable[str]] = None,
) -> List[RegimeCard]:
    """Build one `RegimeCard` per column from training data only."""

    names = list(feature_names) if feature_names is not None else infer_feature_names(X)
    n_features = len(names)
    sparse = is_sparse_matrix(X)
    cards: List[RegimeCard] = []
    for idx in range(n_features):
        name = names[idx]
        col = column_as_dense(X, idx)
        finite = finite_values(col)
        n_samples = int(col.shape[0])
        n_finite = int(finite.size)
        missing_fraction = 1.0 - (n_finite / max(n_samples, 1))
        if n_finite == 0:
            finite = np.array([0.0])
        q01, q25, q50, q75, q99 = [float(np.quantile(finite, q)) for q in (0.01, 0.25, 0.5, 0.75, 0.99)]
        iqr = q75 - q25
        robust_scale = abs(iqr / 1.349) if abs(iqr) > 1e-12 else float(np.std(finite) or 1.0)
        zero_fraction = float(np.mean(np.isclose(np.nan_to_num(col, nan=0.0), 0.0)))
        positive_fraction = float(np.mean(finite > 0.0))
        negative_fraction = float(np.mean(finite < 0.0))
        first_half = finite[: max(1, finite.size // 2)]
        second_half = finite[max(1, finite.size // 2) :]
        drift = 0.0
        if second_half.size:
            drift = float(abs(np.median(second_half) - np.median(first_half)) / max(abs(robust_scale), 1e-12))
        bounded_low, bounded_high, notes = _bounded_metadata(metadata, name)
        if bounded_low is None and bounded_high is None and n_finite >= 4:
            if np.all(finite >= 0) and np.nanmax(finite) <= 1.0:
                bounded_low, bounded_high = 0.0, 1.0
                notes.append("empirical [0, 1] boundedness detected")
        confidence = min(1.0, (n_finite / 100.0) ** 0.5) * (1.0 - 0.5 * missing_fraction)
        card = RegimeCard(
            feature_name=name,
            feature_index=idx,
            n_samples=n_samples,
            missing_fraction=missing_fraction,
            zero_fraction=zero_fraction,
            positive_fraction=positive_fraction,
            negative_fraction=negative_fraction,
            sparse=bool(sparse or zero_fraction > 0.80),
            bounded_low=bounded_low,
            bounded_high=bounded_high,
            robust_location=q50,
            robust_scale=float(max(abs(robust_scale), 1e-12)),
            mean=float(np.mean(finite)),
            std=float(np.std(finite, ddof=1)) if finite.size > 1 else 1.0,
            min=float(np.min(finite)),
            max=float(np.max(finite)),
            q01=q01,
            q25=q25,
            q50=q50,
            q75=q75,
            q99=q99,
            skewness=_skewness(finite),
            tail_category=_tail_category(finite, q25, q75, q99, q01),
            sign_pattern=_sign_pattern(finite),
            drift_estimate=drift,
            batch_reliability=min(1.0, n_finite / 64.0),
            covariance_relevance=bool(metadata and metadata.get("covariance_relevance", False)),
            distance_sensitivity=True,
            interpretability_required=bool(metadata and metadata.get("interpretability_required", False)),
            inverse_required=bool(not metadata or metadata.get("inverse_required", True)),
            confidence=float(max(0.0, min(1.0, confidence))),
            notes=notes,
        )
        cards.append(card)
    return cards


def cards_by_name(cards: Iterable[RegimeCard]) -> Dict[str, RegimeCard]:
    return {card.feature_name: card for card in cards}


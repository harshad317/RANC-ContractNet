"""Normalization policy construction, application, and inverse application."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from ranc_contractnet.schemas import DriftMonitor, NormalizationPolicy, RegimeCard
from ranc_contractnet.utils import as_float_list, stable_scale


def make_drift_monitor(card: RegimeCard) -> DriftMonitor:
    width = max(card.robust_scale * 4.0, abs(card.q99 - card.q01), 1e-9)
    return DriftMonitor(
        feature_name=card.feature_name,
        statistic="median",
        train_low=float(card.robust_location - width),
        train_high=float(card.robust_location + width),
        parameters={
            "train_median": card.robust_location,
            "robust_scale": card.robust_scale,
            "drift_estimate": card.drift_estimate,
        },
    )


def identity_policy(card: RegimeCard, seed: int = 0, reason: Optional[str] = None) -> NormalizationPolicy:
    return NormalizationPolicy(
        feature_name=card.feature_name,
        feature_index=card.feature_index,
        policy_type="identity",
        parameters={},
        formula="f(x)=x",
        inverse_formula="f^-1(z)=z",
        inverse_legal=True,
        no_op=True,
        complexity=0,
        audit_cost=0.0,
        drift_monitor=make_drift_monitor(card),
        downgrade_reason=reason,
        fitted_on_n_samples=card.n_samples,
        seed=seed,
    )


def affine_policy(
    card: RegimeCard,
    *,
    policy_type: str,
    center: float,
    scale: float,
    complexity: int,
    audit_cost: float,
    seed: int = 0,
) -> NormalizationPolicy:
    scale = stable_scale(scale)
    inverse_legal = True
    return NormalizationPolicy(
        feature_name=card.feature_name,
        feature_index=card.feature_index,
        policy_type=policy_type,
        parameters={"center": float(center), "scale": float(scale)},
        formula="f(x)=(x-center)/scale",
        inverse_formula="f^-1(z)=z*scale+center",
        inverse_legal=inverse_legal,
        no_op=False,
        complexity=complexity,
        audit_cost=audit_cost,
        drift_monitor=make_drift_monitor(card),
        fitted_on_n_samples=card.n_samples,
        seed=seed,
    )


def minmax_policy(card: RegimeCard, seed: int = 0) -> NormalizationPolicy:
    low = card.bounded_low if card.bounded_low is not None else card.min
    high = card.bounded_high if card.bounded_high is not None else card.max
    scale = stable_scale(float(high) - float(low))
    return NormalizationPolicy(
        feature_name=card.feature_name,
        feature_index=card.feature_index,
        policy_type="minmax",
        parameters={"min": float(low), "max": float(high), "scale": scale},
        formula="f(x)=(x-min)/(max-min)",
        inverse_formula="f^-1(z)=z*(max-min)+min",
        inverse_legal=True,
        complexity=2,
        audit_cost=0.2,
        drift_monitor=make_drift_monitor(card),
        fitted_on_n_samples=card.n_samples,
        seed=seed,
    )


def log1p_policy(card: RegimeCard, seed: int = 0) -> NormalizationPolicy:
    scale = stable_scale(np.log1p(max(card.q99, card.max, 1.0)))
    return NormalizationPolicy(
        feature_name=card.feature_name,
        feature_index=card.feature_index,
        policy_type="log1p",
        parameters={"scale": float(scale)},
        formula="f(x)=log1p(x)/scale for x>=0",
        inverse_formula="f^-1(z)=expm1(z*scale)",
        inverse_legal=True,
        complexity=3,
        audit_cost=0.35,
        drift_monitor=make_drift_monitor(card),
        fitted_on_n_samples=card.n_samples,
        seed=seed,
    )


def power_policy(
    card: RegimeCard,
    *,
    kind: str,
    lmbda: float,
    center: float,
    scale: float,
    seed: int = 0,
) -> NormalizationPolicy:
    return NormalizationPolicy(
        feature_name=card.feature_name,
        feature_index=card.feature_index,
        policy_type=f"power_{kind}",
        parameters={
            "kind": kind,
            "lambda": float(lmbda),
            "center": float(center),
            "scale": stable_scale(scale),
        },
        formula=f"f(x)=standardized {kind} power transform",
        inverse_formula=f"f^-1(z)=inverse standardized {kind} power transform",
        inverse_legal=True,
        complexity=4,
        audit_cost=0.65,
        drift_monitor=make_drift_monitor(card),
        fitted_on_n_samples=card.n_samples,
        seed=seed,
    )


def quantile_rank_policy(card: RegimeCard, values: Iterable[float], seed: int = 0) -> NormalizationPolicy:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        vals = np.array([0.0, 1.0])
    quantiles = np.unique(np.quantile(vals, np.linspace(0.0, 1.0, min(101, max(5, vals.size)))))
    if quantiles.size == 1:
        quantiles = np.array([quantiles[0] - 1.0, quantiles[0] + 1.0])
    probs = np.linspace(0.0, 1.0, quantiles.size)
    return NormalizationPolicy(
        feature_name=card.feature_name,
        feature_index=card.feature_index,
        policy_type="quantile_rank",
        parameters={"quantiles": as_float_list(quantiles), "probs": as_float_list(probs)},
        formula="f(x)=empirical_rank(x)",
        inverse_formula="f^-1(z)=empirical_quantile(z)",
        inverse_legal=True,
        complexity=5,
        audit_cost=0.9,
        drift_monitor=make_drift_monitor(card),
        fitted_on_n_samples=card.n_samples,
        seed=seed,
    )


def piecewise_affine_policy(card: RegimeCard, seed: int = 0) -> NormalizationPolicy:
    x_points = np.array([card.min, card.q01, card.q50, card.q99, card.max], dtype=float)
    x_points = np.maximum.accumulate(x_points)
    for i in range(1, x_points.size):
        if x_points[i] <= x_points[i - 1]:
            x_points[i] = x_points[i - 1] + 1e-9
    mid_scale = stable_scale(card.robust_scale)
    y_points = np.array(
        [
            -3.0,
            -2.0,
            0.0 if card.sign_pattern == "mixed" else (card.q50 / mid_scale),
            2.0,
            3.0,
        ],
        dtype=float,
    )
    y_points = np.maximum.accumulate(y_points)
    return NormalizationPolicy(
        feature_name=card.feature_name,
        feature_index=card.feature_index,
        policy_type="piecewise_affine",
        parameters={"x_points": as_float_list(x_points), "y_points": as_float_list(y_points)},
        formula="f(x)=monotone linear interpolation over fitted breakpoints",
        inverse_formula="f^-1(z)=inverse monotone linear interpolation",
        inverse_legal=True,
        complexity=5,
        audit_cost=0.8,
        drift_monitor=make_drift_monitor(card),
        fitted_on_n_samples=card.n_samples,
        seed=seed,
    )


def whitening_policy(covariance: np.ndarray, feature_names: List[str], n_samples: int, seed: int = 0) -> NormalizationPolicy:
    cov = np.asarray(covariance, dtype=float)
    eps = 1e-6
    values, vectors = np.linalg.eigh(cov + eps * np.eye(cov.shape[0]))
    matrix = vectors @ np.diag(1.0 / np.sqrt(np.maximum(values, eps))) @ vectors.T
    inverse = vectors @ np.diag(np.sqrt(np.maximum(values, eps))) @ vectors.T
    return NormalizationPolicy(
        feature_name="__group__",
        feature_index=-1,
        policy_type="whitening",
        parameters={
            "feature_names": list(feature_names),
            "matrix": matrix.tolist(),
            "inverse_matrix": inverse.tolist(),
        },
        formula="f(X)=X @ whitening_matrix",
        inverse_formula="f^-1(Z)=Z @ inverse_whitening_matrix",
        inverse_legal=True,
        complexity=7,
        audit_cost=1.0,
        fitted_on_n_samples=int(n_samples),
        seed=seed,
    )


def vector_norm_policy(
    norm: str,
    feature_names: List[str],
    n_samples: int,
    seed: int = 0,
) -> NormalizationPolicy:
    if norm not in {"l1", "l2"}:
        raise ValueError("norm must be 'l1' or 'l2'.")
    return NormalizationPolicy(
        feature_name="__group__",
        feature_index=-1,
        policy_type=f"vector_{norm}",
        parameters={"feature_names": list(feature_names), "norm": norm},
        formula=f"f(x_i)=x_i/||x_i||_{norm}",
        inverse_formula=None,
        inverse_legal=False,
        complexity=3,
        audit_cost=0.5,
        fitted_on_n_samples=int(n_samples),
        seed=seed,
        audit_notes=["vector normalization intentionally destroys row magnitude"],
    )


def _yeo_johnson(x: np.ndarray, lmbda: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    if abs(lmbda) < 1e-12:
        out[pos] = np.log1p(x[pos])
    else:
        out[pos] = ((x[pos] + 1.0) ** lmbda - 1.0) / lmbda
    if abs(lmbda - 2.0) < 1e-12:
        out[~pos] = -np.log1p(-x[~pos])
    else:
        out[~pos] = -(((-x[~pos] + 1.0) ** (2.0 - lmbda) - 1.0) / (2.0 - lmbda))
    return out


def _inv_yeo_johnson(z: np.ndarray, lmbda: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    if abs(lmbda) < 1e-12:
        out[pos] = np.expm1(z[pos])
    else:
        out[pos] = np.maximum(lmbda * z[pos] + 1.0, 0.0) ** (1.0 / lmbda) - 1.0
    if abs(lmbda - 2.0) < 1e-12:
        out[~pos] = 1.0 - np.exp(-z[~pos])
    else:
        out[~pos] = 1.0 - np.maximum(1.0 - (2.0 - lmbda) * z[~pos], 0.0) ** (1.0 / (2.0 - lmbda))
    return out


def _boxcox(x: np.ndarray, lmbda: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    safe = np.maximum(x, 1e-12)
    if abs(lmbda) < 1e-12:
        return np.log(safe)
    return (safe**lmbda - 1.0) / lmbda


def _inv_boxcox(z: np.ndarray, lmbda: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if abs(lmbda) < 1e-12:
        return np.exp(z)
    return np.maximum(lmbda * z + 1.0, 0.0) ** (1.0 / lmbda)


def apply_policy_to_vector(values: np.ndarray, policy: NormalizationPolicy) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    p: Dict[str, Any] = policy.parameters
    kind = policy.policy_type
    if kind == "identity":
        return x.copy()
    if kind in {"zscore", "robust_affine", "maxabs"}:
        return (x - float(p.get("center", 0.0))) / stable_scale(float(p.get("scale", 1.0)))
    if kind == "minmax":
        return (x - float(p["min"])) / stable_scale(float(p["scale"]))
    if kind == "log1p":
        if np.nanmin(x) < -1e-12:
            raise ValueError("log1p policy received negative values.")
        return np.log1p(np.maximum(x, 0.0)) / stable_scale(float(p.get("scale", 1.0)))
    if kind == "power_yeo_johnson":
        raw = _yeo_johnson(x, float(p["lambda"]))
        return (raw - float(p.get("center", 0.0))) / stable_scale(float(p.get("scale", 1.0)))
    if kind == "power_boxcox":
        if np.nanmin(x) <= 0:
            raise ValueError("Box-Cox policy requires positive values.")
        raw = _boxcox(x, float(p["lambda"]))
        return (raw - float(p.get("center", 0.0))) / stable_scale(float(p.get("scale", 1.0)))
    if kind == "quantile_rank":
        return np.interp(x, np.asarray(p["quantiles"], dtype=float), np.asarray(p["probs"], dtype=float))
    if kind == "piecewise_affine":
        return np.interp(x, np.asarray(p["x_points"], dtype=float), np.asarray(p["y_points"], dtype=float))
    if kind == "bounded_tanh":
        scale = stable_scale(float(p.get("scale", 1.0)))
        return np.tanh(x / scale)
    raise ValueError(f"Unknown policy_type={kind!r}.")


def inverse_policy_to_vector(values: np.ndarray, policy: NormalizationPolicy) -> np.ndarray:
    if not policy.inverse_legal:
        raise ValueError(f"Policy {policy.policy_type!r} has no legal inverse.")
    z = np.asarray(values, dtype=float)
    p: Dict[str, Any] = policy.parameters
    kind = policy.policy_type
    if kind == "identity":
        return z.copy()
    if kind in {"zscore", "robust_affine", "maxabs"}:
        return z * stable_scale(float(p.get("scale", 1.0))) + float(p.get("center", 0.0))
    if kind == "minmax":
        return z * stable_scale(float(p["scale"])) + float(p["min"])
    if kind == "log1p":
        return np.expm1(z * stable_scale(float(p.get("scale", 1.0))))
    if kind == "power_yeo_johnson":
        raw = z * stable_scale(float(p.get("scale", 1.0))) + float(p.get("center", 0.0))
        return _inv_yeo_johnson(raw, float(p["lambda"]))
    if kind == "power_boxcox":
        raw = z * stable_scale(float(p.get("scale", 1.0))) + float(p.get("center", 0.0))
        return _inv_boxcox(raw, float(p["lambda"]))
    if kind == "quantile_rank":
        return np.interp(z, np.asarray(p["probs"], dtype=float), np.asarray(p["quantiles"], dtype=float))
    if kind == "piecewise_affine":
        return np.interp(z, np.asarray(p["y_points"], dtype=float), np.asarray(p["x_points"], dtype=float))
    raise ValueError(f"Unknown or non-invertible policy_type={kind!r}.")


def policy_maps_zero_to_zero(policy: NormalizationPolicy, tol: float = 1e-9) -> bool:
    try:
        value = float(apply_policy_to_vector(np.array([0.0]), policy)[0])
    except Exception:
        return False
    return abs(value) <= tol


def policy_preserves_sign(policy: NormalizationPolicy, card: RegimeCard) -> bool:
    probes = np.array([-10.0, -1.0, -0.1, 0.1, 1.0, 10.0], dtype=float)
    if card.sign_pattern == "nonnegative":
        probes = probes[probes >= 0]
    elif card.sign_pattern == "nonpositive":
        probes = probes[probes <= 0]
    try:
        transformed = apply_policy_to_vector(probes, policy)
    except Exception:
        return False
    mask = np.abs(probes) > 1e-12
    return bool(np.all(np.sign(probes[mask]) == np.sign(transformed[mask])))

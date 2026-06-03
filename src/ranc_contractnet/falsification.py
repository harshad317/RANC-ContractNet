"""Targeted falsification tests for compiled normalization policies."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from ranc_contractnet.schemas import FalsificationResult, InvarianceContract, NormalizationPolicy, RegimeCard
from ranc_contractnet.policies import (
    apply_policy_to_vector,
    inverse_policy_to_vector,
    policy_maps_zero_to_zero,
    policy_preserves_sign,
)
from ranc_contractnet.utils import finite_values


def _result(
    card: RegimeCard,
    policy: NormalizationPolicy,
    name: str,
    passed: bool,
    *,
    message: str = "",
    hard: bool = False,
    severity: float = 0.0,
    metrics: dict | None = None,
) -> FalsificationResult:
    return FalsificationResult(
        feature_name=card.feature_name,
        policy_type=policy.policy_type,
        test_name=name,
        passed=bool(passed),
        severity=float(severity),
        message=message,
        hard_clause=bool(hard),
        metrics=metrics or {},
    )


def _monotone_numeric(values: np.ndarray, policy: NormalizationPolicy) -> tuple[bool, float]:
    vals = finite_values(values)
    if vals.size < 2:
        vals = np.array([-1.0, 0.0, 1.0])
    probes = np.unique(np.quantile(vals, np.linspace(0.0, 1.0, min(64, max(3, vals.size)))))
    if probes.size < 3:
        probes = np.array([float(np.min(vals)) - 1.0, float(np.min(vals)), float(np.max(vals)) + 1.0])
    try:
        transformed = apply_policy_to_vector(probes, policy)
    except Exception:
        return False, float("inf")
    diffs = np.diff(transformed)
    min_diff = float(np.min(diffs)) if diffs.size else 0.0
    return bool(min_diff >= -1e-9), min_diff


def _piecewise_z3_check(policy: NormalizationPolicy) -> tuple[bool, bool]:
    if policy.policy_type != "piecewise_affine":
        return True, False
    x_points = np.asarray(policy.parameters["x_points"], dtype=float)
    y_points = np.asarray(policy.parameters["y_points"], dtype=float)
    slopes = np.diff(y_points) / np.maximum(np.diff(x_points), 1e-12)
    try:
        from z3 import And, Not, RealVal, Solver, unsat  # type: ignore
    except Exception:
        return bool(np.all(slopes >= -1e-12)), False
    solver = Solver()
    monotone = And(*[RealVal(str(float(slope))) >= RealVal("0") for slope in slopes])
    solver.add(Not(monotone))
    return solver.check() == unsat, True


def run_policy_falsification(
    values: Iterable[float],
    card: RegimeCard,
    contract: InvarianceContract,
    policy: NormalizationPolicy,
) -> List[FalsificationResult]:
    """Run local hard-clause and risk falsification tests for one policy."""

    x = np.asarray(list(values), dtype=float)
    finite = finite_values(x)
    if finite.size == 0:
        finite = np.array([0.0])
    results: List[FalsificationResult] = []

    if contract.hard("preserve_zero"):
        passed = policy_maps_zero_to_zero(policy)
        results.append(
            _result(
                card,
                policy,
                "zero_preservation",
                passed,
                hard=True,
                severity=0.0 if passed else 1.0,
                message="f(0)=0" if passed else "policy maps zero away from zero or cannot evaluate f(0)",
            )
        )

    if contract.hard("preserve_sign"):
        passed = policy_preserves_sign(policy, card)
        results.append(
            _result(
                card,
                policy,
                "sign_preservation",
                passed,
                hard=True,
                severity=0.0 if passed else 1.0,
                message="sign probes preserved" if passed else "sign changed on probes",
            )
        )

    if contract.hard("preserve_monotonicity") or contract.hard("preserve_rank"):
        passed, min_diff = _monotone_numeric(finite, policy)
        z3_passed, used_z3 = _piecewise_z3_check(policy)
        passed = passed and z3_passed
        results.append(
            _result(
                card,
                policy,
                "monotonicity",
                passed,
                hard=contract.hard("preserve_monotonicity"),
                severity=0.0 if passed else 1.0,
                message="monotone on quantile probes" if passed else "non-monotone behavior detected",
                metrics={"min_diff": min_diff, "used_z3": used_z3},
            )
        )

    if contract.hard("allow_inverse_transform"):
        if not policy.inverse_legal:
            results.append(
                _result(
                    card,
                    policy,
                    "inverse_transform_error",
                    False,
                    hard=True,
                    severity=1.0,
                    message="policy declares inverse_legal=False",
                )
            )
        else:
            probes = np.unique(np.quantile(finite, np.linspace(0.05, 0.95, min(21, max(3, finite.size)))))
            try:
                restored = inverse_policy_to_vector(apply_policy_to_vector(probes, policy), policy)
                max_error = float(np.nanmax(np.abs(restored - probes))) if probes.size else 0.0
                passed = bool(max_error <= 1e-5 + 1e-5 * max(1.0, float(np.nanmax(np.abs(probes)))))
            except Exception as exc:
                max_error = float("inf")
                passed = False
                message = str(exc)
            else:
                message = "inverse round-trip within tolerance"
            results.append(
                _result(
                    card,
                    policy,
                    "inverse_transform_error",
                    passed,
                    hard=True,
                    severity=0.0 if passed else 1.0,
                    message=message,
                    metrics={"max_error": max_error},
                )
            )

    if card.sparse or contract.hard("preserve_zero"):
        sparse_safe = policy_maps_zero_to_zero(policy) and policy.policy_type != "whitening"
        results.append(
            _result(
                card,
                policy,
                "sparse_densification",
                sparse_safe,
                hard=card.sparse,
                severity=0.0 if sparse_safe else 1.0,
                message="policy preserves implicit zeros" if sparse_safe else "policy can densify zeros",
            )
        )

    if contract.hard("damp_outliers"):
        try:
            transformed = apply_policy_to_vector(finite, policy)
            raw_span = float(np.quantile(finite, 0.99) - np.quantile(finite, 0.50))
            new_span = float(np.quantile(transformed, 0.99) - np.quantile(transformed, 0.50))
            passed = bool(new_span <= raw_span / max(card.robust_scale, 1e-12) + 1.0)
        except Exception:
            raw_span = float("inf")
            new_span = float("inf")
            passed = False
        results.append(
            _result(
                card,
                policy,
                "outlier_noise_damping",
                passed,
                hard=True,
                severity=0.0 if passed else 0.7,
                message="tail span is damped or scaled" if passed else "tail span remains unsafe",
                metrics={"raw_tail_span": raw_span, "transformed_tail_span": new_span},
            )
        )

    if contract.hard("preserve_extreme_signal"):
        forbidden = policy.policy_type in {"quantile_rank", "bounded_tanh"}
        results.append(
            _result(
                card,
                policy,
                "outlier_signal_preservation",
                not forbidden,
                hard=True,
                severity=0.0 if not forbidden else 1.0,
                message="extreme signal is not rank-saturated" if not forbidden else "policy can saturate rare extremes",
            )
        )

    if policy.policy_type == "bounded_tanh":
        transformed = apply_policy_to_vector(finite, policy)
        saturation = float(np.mean(np.abs(transformed) > 0.98))
        results.append(
            _result(
                card,
                policy,
                "bounded_policy_saturation",
                saturation <= 0.05,
                hard=False,
                severity=min(1.0, saturation),
                message="bounded saturation rate measured",
                metrics={"saturation_rate": saturation},
            )
        )

    if policy.drift_monitor is not None:
        median = float(np.median(finite))
        monitor = policy.drift_monitor
        passed = bool(monitor.train_low <= median <= monitor.train_high)
        results.append(
            _result(
                card,
                policy,
                "drift_monitor_training_bounds",
                passed,
                hard=False,
                severity=0.0 if passed else 0.5,
                message="training median lies inside monitor bounds",
                metrics={
                    "median": median,
                    "train_low": monitor.train_low,
                    "train_high": monitor.train_high,
                },
            )
        )

    return results


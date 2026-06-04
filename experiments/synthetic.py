"""Synthetic benchmark generators for ContractNet experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from typing import Tuple

import numpy as np
from scipy import sparse
from sklearn.datasets import make_classification, make_regression


@dataclass(frozen=True)
class OutlierDataset:
    """Synthetic outlier dataset with ground-truth rare-event mask."""

    X: np.ndarray
    y: np.ndarray
    outlier_mask: np.ndarray
    scenario: str


def make_scale_shift_regression(
    n_train: int = 128,
    n_test: int = 128,
    scale_shift: float = 2.0,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    X_train, y_train = make_regression(
        n_samples=n_train,
        n_features=4,
        n_informative=3,
        noise=0.2,
        random_state=random_state,
    )
    X_test = rng.normal(size=(n_test, 4))
    coef = np.array([30.0, -10.0, 4.0, 0.0])
    y_test = X_test @ coef + rng.normal(scale=0.2, size=n_test)
    X_test[:, 0] *= scale_shift
    return X_train, y_train, X_test, y_test


def make_additive_shift_classification(
    n_train: int = 160,
    n_test: int = 160,
    shift: float = 3.0,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, y_train = make_classification(
        n_samples=n_train,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=random_state,
    )
    rng = np.random.default_rng(random_state + 1)
    X_test, y_test = make_classification(
        n_samples=n_test,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=random_state + 2,
    )
    X_test[:, 1] += shift
    X_test += rng.normal(scale=0.05, size=X_test.shape)
    return X_train, y_train, X_test, y_test


def make_outlier_signal_noise(
    n_samples: int = 400,
    signal: bool = False,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    base = rng.normal(size=(n_samples, 4))
    tail = rng.pareto(a=2.0, size=n_samples)
    outlier_mask = rng.random(n_samples) < 0.05
    base[:, 0] = tail
    base[outlier_mask, 0] *= 20.0
    logits = 0.8 * base[:, 1] - 0.4 * base[:, 2]
    if signal:
        logits += 3.0 * outlier_mask.astype(float)
    else:
        base[outlier_mask, 0] += rng.normal(scale=50.0, size=int(np.sum(outlier_mask)))
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n_samples) < probs).astype(int)
    return base, y


def make_paired_outlier_signal_noise(
    n_samples: int = 800,
    outlier_fraction: float = 0.06,
    random_state: int = 0,
) -> Dict[str, OutlierDataset]:
    """Create matched outlier-as-noise and outlier-as-signal classification tasks.

    Both tasks share the same covariates and outlier positions. In the noise task,
    the rare extreme feature is a measurement corruption independent of the label.
    In the signal task, the same rare extremes are predictive events.
    """

    rng = np.random.default_rng(random_state)
    shared = rng.normal(size=(n_samples, 5))
    outlier_mask = rng.random(n_samples) < outlier_fraction
    tail_feature = rng.lognormal(mean=0.0, sigma=0.6, size=n_samples)
    normal_tail = tail_feature.copy()
    tail_feature[outlier_mask] = rng.lognormal(mean=4.2, sigma=0.45, size=int(np.sum(outlier_mask)))
    X = shared.copy()
    X[:, 0] = tail_feature
    normal_cap = float(np.quantile(normal_tail[~outlier_mask], 0.90)) if np.any(~outlier_mask) else 2.0
    label_tail_for_noise = np.where(outlier_mask, normal_tail, tail_feature)
    clipped_tail_signal = np.log1p(np.minimum(label_tail_for_noise, normal_cap))
    clipped_tail_signal = (clipped_tail_signal - np.mean(clipped_tail_signal)) / (
        np.std(clipped_tail_signal) + 1e-12
    )
    tail_signal = np.log1p(tail_feature)
    tail_signal = (tail_signal - np.mean(tail_signal)) / (np.std(tail_signal) + 1e-12)
    base_logits = 0.8 * X[:, 1] - 0.6 * X[:, 2] + 0.25 * X[:, 3]

    noise_logits = base_logits + 1.35 * clipped_tail_signal
    noise_probs = 1.0 / (1.0 + np.exp(-noise_logits))
    noise_y = (rng.random(n_samples) < noise_probs).astype(int)

    signal_logits = base_logits + 2.4 * outlier_mask.astype(float) + 0.75 * tail_signal
    signal_probs = 1.0 / (1.0 + np.exp(-signal_logits))
    signal_y = (rng.random(n_samples) < signal_probs).astype(int)

    return {
        "noise": OutlierDataset(
            X=X.copy(),
            y=noise_y,
            outlier_mask=outlier_mask.copy(),
            scenario="noise",
        ),
        "signal": OutlierDataset(
            X=X.copy(),
            y=signal_y,
            outlier_mask=outlier_mask.copy(),
            scenario="signal",
        ),
    }


def make_sparse_classification(
    n_samples: int = 240,
    n_features: int = 64,
    density: float = 0.04,
    random_state: int = 0,
):
    rng = np.random.default_rng(random_state)
    X = sparse.random(
        n_samples,
        n_features,
        density=density,
        random_state=random_state,
        format="csr",
        data_rvs=lambda n: rng.lognormal(mean=0.0, sigma=1.0, size=n),
    )
    weights = rng.normal(size=n_features)
    logits = X @ weights
    logits = np.asarray(logits).ravel()
    y = (logits > np.median(logits)).astype(int)
    return X, y


def make_temporal_drift(
    n_samples: int = 360,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    t = np.arange(n_samples)
    X = rng.normal(size=(n_samples, 5))
    X[:, 0] += t / n_samples * 2.0
    y = 2.0 * X[:, 0] - X[:, 1] + rng.normal(scale=0.2, size=n_samples)
    return X, y, t


def make_temporal_rare_event(
    n_samples: int = 520,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a time-ordered rare-event classification case study.

    The first feature acts like a drifting lab/risk measurement: most values are
    ordinary positive readings, while rare extreme readings are predictive events
    rather than corruptions. This gives the paper a compact domain-style setting
    where temporal fit discipline and extreme-signal preservation are both
    natural contract clauses.
    """

    rng = np.random.default_rng(random_state)
    t = np.arange(n_samples)
    drift = t / max(n_samples - 1, 1)
    season = np.sin(2.0 * np.pi * t / 52.0)
    baseline = rng.lognormal(mean=0.35 * drift, sigma=0.45, size=n_samples)
    rare_mask = rng.random(n_samples) < (0.08 + 0.03 * (drift > 0.65))
    lab_value = baseline.copy()
    lab_value[rare_mask] *= rng.lognormal(mean=2.5, sigma=0.35, size=int(np.sum(rare_mask)))

    feature_1 = rng.normal(size=n_samples) + 0.7 * season
    feature_2 = rng.normal(size=n_samples) + 0.8 * drift
    feature_3 = rng.binomial(1, 0.15 + 0.1 * drift, size=n_samples)
    feature_4 = rng.normal(scale=0.5, size=n_samples)

    tail_signal = np.log1p(lab_value)
    tail_signal = (tail_signal - float(np.mean(tail_signal))) / (float(np.std(tail_signal)) + 1e-12)
    logits = (
        -3.2
        + 1.6 * rare_mask.astype(float)
        + 0.85 * tail_signal
        + 0.45 * feature_3
        + 0.2 * season
        - 0.25 * feature_1
    )
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n_samples) < probabilities).astype(int)
    y[rare_mask & (rng.random(n_samples) < 0.55)] = 1
    X = np.column_stack([lab_value, feature_1, feature_2, feature_3, feature_4])
    return X, y, t, rare_mask

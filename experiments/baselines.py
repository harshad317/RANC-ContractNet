"""Reusable experiment baselines."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class WinsorizingScaler(BaseEstimator, TransformerMixin):
    """Clip each feature to train-split quantiles.

    This is an intentionally strong baseline for outlier-as-noise regimes and a
    deliberately risky one for outlier-as-signal regimes.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99, rescale: bool = True) -> None:
        self.lower = lower
        self.upper = upper
        self.rescale = rescale

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.low_ = np.quantile(arr, self.lower, axis=0)
        self.high_ = np.quantile(arr, self.upper, axis=0)
        self.center_ = np.median(np.clip(arr, self.low_, self.high_), axis=0)
        q25 = np.quantile(np.clip(arr, self.low_, self.high_), 0.25, axis=0)
        q75 = np.quantile(np.clip(arr, self.low_, self.high_), 0.75, axis=0)
        self.scale_ = np.maximum(q75 - q25, 1e-12)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        clipped = np.clip(arr, self.low_, self.high_)
        if not self.rescale:
            return clipped
        return (clipped - self.center_) / self.scale_


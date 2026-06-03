"""Shared data conversion and numerical helpers."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


EPS = 1e-12


def is_sparse_matrix(value: object) -> bool:
    return sparse.issparse(value)


def infer_feature_names(X: object) -> List[str]:
    if isinstance(X, pd.DataFrame):
        return [str(col) for col in X.columns]
    if sparse.issparse(X):
        return [f"x{i}" for i in range(int(X.shape[1]))]
    arr = np.asarray(X)
    if arr.ndim == 1:
        return ["x0"]
    return [f"x{i}" for i in range(int(arr.shape[1]))]


def to_numpy_2d(X: object, *, copy: bool = False) -> np.ndarray:
    if sparse.issparse(X):
        raise TypeError("Sparse inputs must use sparse-specific paths.")
    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy(dtype=float, copy=copy)
    else:
        arr = np.asarray(X, dtype=float)
        if copy:
            arr = arr.copy()
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array-like object, got shape {arr.shape!r}.")
    return arr


def ensure_2d_shape(X: object) -> Tuple[int, int]:
    if sparse.issparse(X):
        if len(X.shape) != 2:
            raise ValueError("Sparse input must be 2D.")
        return int(X.shape[0]), int(X.shape[1])
    arr = to_numpy_2d(X, copy=False)
    return int(arr.shape[0]), int(arr.shape[1])


def finite_values(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def safe_quantile(values: Sequence[float], q: float, default: float = 0.0) -> float:
    vals = finite_values(values)
    if vals.size == 0:
        return float(default)
    return float(np.quantile(vals, q))


def safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    vals = finite_values(values)
    if vals.size == 0:
        return float(default)
    return float(np.mean(vals))


def safe_std(values: Sequence[float], default: float = 1.0) -> float:
    vals = finite_values(values)
    if vals.size <= 1:
        return float(default)
    std = float(np.std(vals, ddof=1))
    return std if std > EPS else float(default)


def stable_scale(value: float, fallback: float = 1.0) -> float:
    value = float(value)
    if not np.isfinite(value) or abs(value) < EPS:
        return float(fallback if abs(fallback) >= EPS else 1.0)
    return abs(value)


def column_as_dense(X: object, index: int) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X.getcol(index).toarray(), dtype=float).ravel()
    return to_numpy_2d(X, copy=False)[:, index].astype(float, copy=False)


def as_float_list(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values]


def clone_sparse_for_column_ops(X: object):
    if not sparse.issparse(X):
        raise TypeError("Expected a scipy sparse matrix.")
    return X.tocsc(copy=True)


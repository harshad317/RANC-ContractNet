"""Serialization helpers for fitted ContractNet policies."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from ranc_contractnet.sklearn import RANCDataTransformer


def save_policy(transformer: RANCDataTransformer, path: Union[str, Path]) -> Path:
    """Save a fitted `RANCDataTransformer` to a deterministic JSON file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(transformer.to_json(indent=2), encoding="utf-8")
    return path


def load_policy(path: Union[str, Path]) -> RANCDataTransformer:
    """Load a fitted `RANCDataTransformer` saved by `save_policy`."""

    return RANCDataTransformer.from_json(Path(path).read_text(encoding="utf-8"))


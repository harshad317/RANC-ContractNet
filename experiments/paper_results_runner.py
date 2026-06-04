"""Regenerate all paper-facing result artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import yaml

from experiments.ablation_runner import run as run_ablation
from experiments.benchmark_runner import run as run_benchmark
from experiments.openml_runner import run as run_openml
from experiments.outlier_pair_runner import run as run_outlier_pair
from experiments.tabular_runner import run as run_tabular


DEFAULT_CONFIG_PATHS = {
    "outlier_pair_config": Path("experiments/configs/outlier_pair.yaml"),
    "sparse_config": Path("experiments/configs/sparse.yaml"),
    "temporal_config": Path("experiments/configs/temporal_drift.yaml"),
    "case_study_config": Path("experiments/configs/case_study_temporal_rare_event.yaml"),
    "ablation_config": Path("experiments/configs/ablation.yaml"),
    "benchmark_config": Path("experiments/configs/benchmark.yaml"),
    "openml_config": Path("experiments/configs/openml_public.yaml"),
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _config_from_path_or_mapping(value: Any, default_path: Path) -> Dict[str, Any]:
    if value is None:
        return _load_yaml(default_path)
    if isinstance(value, dict):
        return value
    return _load_yaml(Path(value))


def run(config: Dict[str, Any] | None = None) -> List[Path]:
    cfg = dict(config or {})
    artifacts: List[Path] = []

    if cfg.get("run_outlier_pair", True):
        outlier_config = _config_from_path_or_mapping(
            cfg.get("outlier_pair_config"),
            DEFAULT_CONFIG_PATHS["outlier_pair_config"],
        )
        artifacts.extend(run_outlier_pair(outlier_config))

    if cfg.get("run_sparse", True):
        sparse_config = _config_from_path_or_mapping(
            cfg.get("sparse_config"),
            DEFAULT_CONFIG_PATHS["sparse_config"],
        )
        artifacts.append(run_tabular(sparse_config)[0])

    if cfg.get("run_temporal", True):
        temporal_config = _config_from_path_or_mapping(
            cfg.get("temporal_config"),
            DEFAULT_CONFIG_PATHS["temporal_config"],
        )
        artifacts.append(run_tabular(temporal_config)[0])

    if cfg.get("run_case_study", True):
        case_study_config = _config_from_path_or_mapping(
            cfg.get("case_study_config"),
            DEFAULT_CONFIG_PATHS["case_study_config"],
        )
        artifacts.append(run_tabular(case_study_config)[0])

    if cfg.get("run_ablation", True):
        ablation_config = _config_from_path_or_mapping(
            cfg.get("ablation_config"),
            DEFAULT_CONFIG_PATHS["ablation_config"],
        )
        artifacts.append(run_ablation(ablation_config)[0])

    if cfg.get("run_benchmark", True):
        benchmark_config = _config_from_path_or_mapping(
            cfg.get("benchmark_config"),
            DEFAULT_CONFIG_PATHS["benchmark_config"],
        )
        summary_path, aggregate_path, _ = run_benchmark(benchmark_config)
        artifacts.extend([summary_path, aggregate_path])

    if cfg.get("run_openml", False):
        openml_config = _config_from_path_or_mapping(
            cfg.get("openml_config"),
            DEFAULT_CONFIG_PATHS["openml_config"],
        )
        summary_path, aggregate_path, _ = run_openml(openml_config)
        artifacts.extend([summary_path, aggregate_path])
        artifacts.extend(
            [
                summary_path.parent / "openml_pairwise_deltas.csv",
                summary_path.parent / "openml_win_loss_table.tex",
                summary_path.parent / "openml_stats_paragraph.md",
            ]
        )

    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional YAML overriding runner paths and toggles.")
    args = parser.parse_args()
    config = _load_yaml(args.config) if args.config else {}
    for artifact in run(config):
        print(artifact)


if __name__ == "__main__":
    main()

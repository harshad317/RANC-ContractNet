"""Multi-seed synthetic benchmark orchestrator for paper tables."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import yaml

from experiments.tabular_runner import run as run_tabular


NUMERIC_COLUMNS = [
    "accuracy",
    "auroc",
    "rmse",
    "mae",
    "test_nnz_delta",
    "ranc_sparse_failures",
    "ranc_drift_monitors",
    "ranc_hard_failures",
    "ranc_ledger_rows",
    "ranc_downgrades",
    "ranc_rejected_candidates",
]


DEFAULT_BENCHMARKS: List[Dict[str, Any]] = [
    {
        "name": "outlier_noise",
        "dataset": {"kind": "outlier_noise"},
        "baselines": ["none", "standard", "robust", "maxabs", "quantile", "ranc"],
        "constraints": {
            "global": {
                "hard_clauses": {
                    "preserve_monotonicity": True,
                    "allow_inverse_transform": True,
                    "enforce_scale_invariance": True,
                }
            },
        },
    },
    {
        "name": "outlier_signal",
        "dataset": {"kind": "outlier_signal"},
        "baselines": ["none", "standard", "robust", "quantile", "ranc"],
        "constraints": {
            "global": {
                "hard_clauses": {
                    "preserve_extreme_signal": True,
                    "preserve_monotonicity": True,
                    "allow_inverse_transform": True,
                    "enforce_scale_invariance": True,
                }
            }
        },
    },
    {
        "name": "sparse",
        "dataset": {"kind": "sparse_classification"},
        "baselines": ["none", "standard", "robust", "maxabs", "ranc"],
        "constraints": {
            "global": {
                "hard_clauses": {
                    "preserve_zero": True,
                    "preserve_monotonicity": True,
                    "enforce_scale_invariance": True,
                }
            }
        },
    },
    {
        "name": "temporal_drift",
        "dataset": {"kind": "temporal_drift"},
        "baselines": ["none", "standard", "robust", "selector", "ranc"],
        "constraints": {
            "global": {
                "hard_clauses": {
                    "enforce_scale_invariance": True,
                    "preserve_monotonicity": True,
                    "allow_inverse_transform": True,
                },
                "soft_clauses": {"prefer_drift_monitor": 1.0},
            }
        },
    },
    {
        "name": "scale_shift",
        "dataset": {"kind": "scale_shift_regression"},
        "baselines": ["none", "standard", "robust", "minmax", "ranc"],
        "constraints": {
            "global": {
                "hard_clauses": {
                    "enforce_scale_invariance": True,
                    "preserve_monotonicity": True,
                    "allow_inverse_transform": True,
                }
            }
        },
    },
    {
        "name": "additive_shift",
        "dataset": {"kind": "additive_shift_classification"},
        "baselines": ["none", "standard", "robust", "minmax", "ranc"],
        "constraints": {
            "global": {
                "hard_clauses": {
                    "enforce_scale_invariance": True,
                    "enforce_shift_invariance": True,
                    "preserve_monotonicity": True,
                    "allow_inverse_transform": True,
                }
            }
        },
    },
]


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray([_as_float(value) for value in values], dtype=float)
    return arr[np.isfinite(arr)]


def _format_mean_std(mean: Any, std: Any, *, lower_is_better: bool = False) -> str:
    mean_f = _as_float(mean)
    std_f = _as_float(std)
    if not np.isfinite(mean_f):
        return ""
    direction = "(lower)" if lower_is_better else "(higher)"
    if not np.isfinite(std_f):
        std_f = 0.0
    return f"{mean_f:.3f} +/- {std_f:.3f} {direction}"


def _format_plain_mean_std(mean: Any, std: Any) -> str:
    mean_f = _as_float(mean)
    std_f = _as_float(std)
    if not np.isfinite(mean_f):
        return ""
    if not np.isfinite(std_f):
        std_f = 0.0
    return f"{mean_f:.3f} +/- {std_f:.3f}"


def _escape_tex(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _task_for_group(rows: List[Dict[str, Any]]) -> str:
    tasks = sorted({str(row.get("task", "")) for row in rows if row.get("task")})
    return tasks[0] if tasks else ""


def _primary_metric(row: Dict[str, Any]) -> Tuple[str, bool, str]:
    task = str(row.get("task", ""))
    if task == "regression":
        return "rmse", True, _format_mean_std(row.get("rmse_mean"), row.get("rmse_std"), lower_is_better=True)
    if np.isfinite(_as_float(row.get("auroc_mean"))):
        return "auroc", False, _format_mean_std(row.get("auroc_mean"), row.get("auroc_std"))
    return "accuracy", False, _format_mean_std(row.get("accuracy_mean"), row.get("accuracy_std"))


def _aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["benchmark"]), str(row["scaler"])), []).append(row)

    aggregate: List[Dict[str, Any]] = []
    for (benchmark, scaler), group in sorted(grouped.items()):
        out: Dict[str, Any] = {
            "benchmark": benchmark,
            "scaler": scaler,
            "n_runs": len(group),
            "task": _task_for_group(group),
            "dataset_kind": sorted({str(row.get("dataset_kind", "")) for row in group})[0],
        }
        selected = sorted({str(row.get("selected_scaler", "")) for row in group if row.get("selected_scaler")})
        out["selected_scalers"] = "+".join(selected)
        for column in NUMERIC_COLUMNS:
            values = _finite(row.get(column, "") for row in group)
            out[f"{column}_mean"] = float(np.mean(values)) if values.size else float("nan")
            out[f"{column}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        passed_values = [str(row.get("ranc_passed", "")).lower() == "true" for row in group if row.get("ranc_passed")]
        train_only_values = [
            str(row.get("ranc_train_only_fit", "")).lower() == "true"
            for row in group
            if row.get("ranc_train_only_fit")
        ]
        out["ranc_pass_rate"] = float(np.mean(passed_values)) if passed_values else float("nan")
        out["ranc_train_only_rate"] = float(np.mean(train_only_values)) if train_only_values else float("nan")
        metric_name, lower, formatted = _primary_metric(out)
        out["primary_metric"] = metric_name
        out["primary_lower_is_better"] = str(lower)
        out["primary_value"] = formatted
        aggregate.append(out)
    return aggregate


def _write_table(output_dir: Path, aggregate: List[Dict[str, Any]]) -> None:
    headers = [
        "Benchmark",
        "Scaler",
        "n",
        "Primary",
        "Accuracy",
        "AUROC",
        "RMSE",
        "MAE",
        "Audit pass",
        "Sparse delta",
        "Drift monitors",
    ]
    table_rows: List[List[str]] = []
    for row in aggregate:
        audit = ""
        if np.isfinite(_as_float(row.get("ranc_pass_rate"))):
            audit = f"{_as_float(row.get('ranc_pass_rate')):.2f}"
        table_rows.append(
            [
                str(row["benchmark"]),
                str(row["scaler"]),
                str(row["n_runs"]),
                str(row["primary_value"]),
                _format_plain_mean_std(row.get("accuracy_mean"), row.get("accuracy_std")),
                _format_plain_mean_std(row.get("auroc_mean"), row.get("auroc_std")),
                _format_plain_mean_std(row.get("rmse_mean"), row.get("rmse_std")),
                _format_plain_mean_std(row.get("mae_mean"), row.get("mae_std")),
                audit,
                _format_plain_mean_std(row.get("test_nnz_delta_mean"), row.get("test_nnz_delta_std")),
                _format_plain_mean_std(row.get("ranc_drift_monitors_mean"), row.get("ranc_drift_monitors_std")),
            ]
        )

    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table_rows:
        md_lines.append("| " + " | ".join(row) + " |")
    (output_dir / "benchmark_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        "\\begin{tabular}{" + "l" * len(headers) + "}",
        "\\toprule",
        " & ".join(_escape_tex(header) for header in headers) + " \\\\",
        "\\midrule",
    ]
    for row in table_rows:
        tex_lines.append(" & ".join(_escape_tex(cell) for cell in row) + " \\\\")
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output_dir / "benchmark_table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def _write_paragraph(output_dir: Path, aggregate: List[Dict[str, Any]]) -> None:
    benchmarks = sorted({str(row["benchmark"]) for row in aggregate})
    ranc_rows = [row for row in aggregate if str(row["scaler"]).lower() == "ranc"]
    all_pass = all(_as_float(row.get("ranc_pass_rate")) == 1.0 for row in ranc_rows)
    paragraph = (
        f"The multi-seed harness aggregated {len(benchmarks)} synthetic benchmark families "
        f"across {sum(int(row['n_runs']) for row in aggregate)} scaler-seed rows. "
        f"RANC audit pass rate was 1.00 for all RANC rows: {str(all_pass)}."
    )
    (output_dir / "benchmark_result_paragraph.md").write_text(paragraph + "\n", encoding="utf-8")


def _benchmark_config(base: Dict[str, Any], benchmark: Mapping[str, Any], seed: int, output_dir: Path) -> Dict[str, Any]:
    excluded = {"name", "seeds"}
    config = {key: value for key, value in benchmark.items() if key not in excluded}
    config = _merge(base.get("defaults", {}), config)
    config["random_state"] = int(seed)
    config["output_dir"] = str(output_dir)
    return config


def run(config: Dict[str, Any] | None = None) -> Tuple[Path, Path, List[Dict[str, Any]]]:
    cfg = dict(config or {})
    seeds = [int(seed) for seed in cfg.get("seeds", [0, 1, 2, 3, 4])]
    benchmarks = list(cfg.get("benchmarks", DEFAULT_BENCHMARKS))
    output_dir = Path(cfg.get("output_dir", "outputs/benchmark"))
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: List[Dict[str, Any]] = []
    for benchmark in benchmarks:
        name = str(benchmark["name"])
        benchmark_seeds = [int(seed) for seed in benchmark.get("seeds", seeds)]
        for seed in benchmark_seeds:
            run_dir = output_dir / "runs" / name / f"seed_{seed}"
            tabular_config = _benchmark_config(cfg, benchmark, seed, run_dir)
            metrics_path, _ = run_tabular(tabular_config)
            for row in _read_rows(metrics_path):
                enriched = dict(row)
                enriched.update(
                    {
                        "benchmark": name,
                        "seed": int(seed),
                        "metrics_path": str(metrics_path),
                    }
                )
                raw_rows.append(enriched)

    summary_path = output_dir / "benchmark_summary.csv"
    _write_csv(summary_path, raw_rows)
    aggregate = _aggregate(raw_rows)
    aggregate_path = output_dir / "benchmark_aggregate.csv"
    _write_csv(aggregate_path, aggregate)
    _write_table(output_dir, aggregate)
    _write_paragraph(output_dir, aggregate)
    return summary_path, aggregate_path, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/benchmark.yaml"))
    args = parser.parse_args()
    config = _load_config(args.config) if args.config.exists() else {}
    summary_path, aggregate_path, _ = run(config)
    print(summary_path)
    print(aggregate_path)


if __name__ == "__main__":
    main()

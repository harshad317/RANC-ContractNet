"""Public OpenML/UCI-style benchmark suite runner.

The runner stores metrics, tables, and RANC audit artifacts only. It does not
write fetched public datasets into the repository output directory.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import yaml

from experiments.benchmark_runner import run as run_benchmark
from experiments.public_stats_runner import run as run_public_stats


DEFAULT_BASELINES = ["none", "standard", "robust", "maxabs", "quantile", "selector", "ranc"]
DEFAULT_CONSTRAINTS = {
    "global": {
        "hard_clauses": {
            "preserve_monotonicity": True,
            "allow_inverse_transform": True,
            "enforce_scale_invariance": True,
        }
    }
}


DEFAULT_TASKS: List[Dict[str, Any]] = [
    {
        "name": "openml_iris",
        "source": "openml",
        "openml_id": 61,
        "target": "class",
        "task": "classification",
        "baselines": ["none", "standard", "robust", "maxabs", "selector", "ranc"],
    },
    {
        "name": "openml_credit_g",
        "source": "uci_via_openml",
        "openml_id": 31,
        "target": "class",
        "task": "classification",
    },
    {
        "name": "openml_spambase",
        "source": "uci_via_openml",
        "openml_id": 44,
        "target": "class",
        "task": "classification",
    },
    {
        "name": "openml_wine_quality",
        "source": "uci_via_openml",
        "openml_id": 287,
        "target": "quality",
        "task": "regression",
        "baselines": ["none", "standard", "robust", "minmax", "selector", "ranc"],
    },
]


def _enabled_tasks(tasks: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [task for task in tasks if bool(task.get("enabled", True))]


def _excluded_tasks(tasks: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [task for task in tasks if not bool(task.get("enabled", True))]


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _compact_reason(reason: Any, limit: int = 260) -> str:
    text = " ".join(str(reason).split())
    if "Status code:" in text:
        prefix, status_tail = text.split("Status code:", maxsplit=1)
        status = status_tail.split()[0] if status_tail.split() else ""
        text = f"{prefix.strip()} Status code: {status}".strip()
    text = text.replace("|", "\\|")
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _remove_stale_benchmark_artifacts(output_dir: Path) -> None:
    for name in ("benchmark_table.md", "benchmark_table.tex", "benchmark_result_paragraph.md"):
        path = output_dir / name
        if path.exists():
            path.unlink()


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


def _dataset_from_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(task.get("dataset"), Mapping):
        return dict(task["dataset"])
    dataset: Dict[str, Any] = {
        "kind": "openml",
        "target": task.get("target"),
        "task": task.get("task", "classification"),
        "test_size": float(task.get("test_size", 0.35)),
    }
    if task.get("openml_id") is not None:
        dataset["id"] = int(task["openml_id"])
    elif task.get("openml_name") is not None:
        dataset["name"] = str(task["openml_name"])
    else:
        raise ValueError(f"Public task {task.get('name', '<unnamed>')!r} requires dataset, openml_id, or openml_name.")
    return dataset


def _task_to_benchmark(task: Mapping[str, Any], defaults: Mapping[str, Any]) -> Dict[str, Any]:
    constraints = dict(defaults.get("constraints", DEFAULT_CONSTRAINTS))
    if isinstance(task.get("constraints"), Mapping):
        constraints = dict(task["constraints"])
    return {
        "name": str(task["name"]),
        "dataset": _dataset_from_task(task),
        "baselines": list(task.get("baselines", defaults.get("baselines", DEFAULT_BASELINES))),
        "constraints": constraints,
        **({"seeds": list(task["seeds"])} if task.get("seeds") is not None else {}),
    }


def _write_public_metadata(
    output_dir: Path,
    tasks: List[Mapping[str, Any]],
    seeds: List[int],
    failed_tasks: Mapping[str, str] | None = None,
) -> Path:
    failures = dict(failed_tasks or {})
    rows = []
    for task in tasks:
        enabled = bool(task.get("enabled", True))
        name = str(task.get("name", ""))
        status = "included" if enabled else "excluded"
        reason = _compact_reason(task.get("exclude_reason", ""))
        if name in failures:
            status = "failed"
            reason = _compact_reason(failures[name])
        rows.append(
            {
                "name": name,
                "source": task.get("source", "openml"),
                "status": status,
                "enabled": str(enabled),
                "exclude_reason": reason,
                "openml_id": task.get("openml_id", ""),
                "openml_name": task.get("openml_name", ""),
                "target": task.get("target", ""),
                "task": task.get("task", ""),
                "seeds": ",".join(str(seed) for seed in task.get("seeds", seeds)),
            }
        )
    path = output_dir / "openml_task_metadata.csv"
    _write_csv(path, rows)
    return path


def _write_exclusion_log(
    output_dir: Path,
    tasks: List[Mapping[str, Any]],
    seeds: List[int],
    failed_tasks: Mapping[str, str] | None = None,
) -> Path:
    failures = dict(failed_tasks or {})
    included = [task for task in _enabled_tasks(tasks) if str(task.get("name", "")) not in failures]
    excluded = _excluded_tasks(tasks)
    failed = [task for task in _enabled_tasks(tasks) if str(task.get("name", "")) in failures]
    lines = [
        "# OpenML/UCI Public Benchmark Selection Log",
        "",
        "This file is generated by `experiments/openml_runner.py`.",
        "",
        f"- Configured datasets: {len(tasks)}",
        f"- Included datasets: {len(included)}",
        f"- Excluded datasets: {len(excluded)}",
        f"- Failed datasets: {len(failed)}",
        f"- Default seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "## Included",
        "",
        "| Dataset | Source | OpenML ID/name | Task | Target | Seeds |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for task in included:
        identifier = task.get("openml_id", task.get("openml_name", ""))
        task_seeds = ",".join(str(seed) for seed in task.get("seeds", seeds))
        lines.append(
            f"| {task.get('name', '')} | {task.get('source', 'openml')} | {identifier} | "
            f"{task.get('task', '')} | {task.get('target', '<default>')} | {task_seeds} |"
        )
    lines.extend(
        [
            "",
            "## Excluded",
            "",
            "| Dataset | Source | OpenML ID/name | Intended task | Reason |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for task in excluded:
        identifier = task.get("openml_id", task.get("openml_name", ""))
        reason = _compact_reason(task.get("exclude_reason", "no reason recorded"))
        lines.append(
            f"| {task.get('name', '')} | {task.get('source', 'openml')} | {identifier} | "
            f"{task.get('task', '')} | {reason} |"
        )
    lines.extend(
        [
            "",
            "## Failed During Run",
            "",
            "| Dataset | Source | OpenML ID/name | Intended task | Failure reason |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for task in failed:
        identifier = task.get("openml_id", task.get("openml_name", ""))
        reason = _compact_reason(failures.get(str(task.get("name", "")), "unknown failure"))
        lines.append(
            f"| {task.get('name', '')} | {task.get('source', 'openml')} | {identifier} | "
            f"{task.get('task', '')} | {reason} |"
        )
    path = output_dir / "exclusion_log.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _rewrite_summary_with_source(summary_path: Path, tasks: List[Mapping[str, Any]]) -> None:
    source_by_name = {str(task["name"]): str(task.get("source", "openml")) for task in tasks}
    rows = _read_rows(summary_path)
    for row in rows:
        row["public_source"] = source_by_name.get(row.get("benchmark", ""), "")
    _write_csv(summary_path, rows)


def _rewrite_aggregate_with_source(aggregate_path: Path, tasks: List[Mapping[str, Any]]) -> None:
    source_by_name = {str(task["name"]): str(task.get("source", "openml")) for task in tasks}
    rows = _read_rows(aggregate_path)
    for row in rows:
        row["public_source"] = source_by_name.get(row.get("benchmark", ""), "")
    _write_csv(aggregate_path, rows)


def _write_public_table(output_dir: Path, aggregate_path: Path) -> Tuple[Path, Path]:
    rows = _read_rows(aggregate_path)
    headers = [
        "Benchmark",
        "Source",
        "Scaler",
        "n",
        "Primary",
        "Accuracy",
        "AUROC",
        "RMSE",
        "MAE",
        "Selected",
        "Audit pass",
    ]
    table_rows: List[List[str]] = []
    for row in rows:
        audit_pass = ""
        if row.get("ranc_pass_rate") not in {"", "nan"}:
            audit_pass = f"{float(row['ranc_pass_rate']):.2f}"
        table_rows.append(
            [
                row.get("benchmark", ""),
                row.get("public_source", ""),
                row.get("scaler", ""),
                row.get("n_runs", ""),
                row.get("primary_value", ""),
                row.get("accuracy_mean", "")[:5] if row.get("accuracy_mean") not in {"", "nan"} else "",
                row.get("auroc_mean", "")[:5] if row.get("auroc_mean") not in {"", "nan"} else "",
                row.get("rmse_mean", "")[:5] if row.get("rmse_mean") not in {"", "nan"} else "",
                row.get("mae_mean", "")[:5] if row.get("mae_mean") not in {"", "nan"} else "",
                row.get("selected_scalers", ""),
                audit_pass,
            ]
        )

    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table_rows:
        md_lines.append("| " + " | ".join(row) + " |")
    md_path = output_dir / "openml_table.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        "\\begin{tabular}{" + "l" * len(headers) + "}",
        "\\toprule",
        " & ".join(_escape_tex(header) for header in headers) + " \\\\",
        "\\midrule",
    ]
    for row in table_rows:
        tex_lines.append(" & ".join(_escape_tex(cell) for cell in row) + " \\\\")
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    tex_path = output_dir / "openml_table.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    return md_path, tex_path


def _write_paragraph(
    output_dir: Path,
    tasks: List[Mapping[str, Any]],
    failed_tasks: Mapping[str, str],
    aggregate_path: Path,
) -> Path:
    aggregate_rows = _read_rows(aggregate_path)
    ranc_rows = [row for row in aggregate_rows if str(row.get("scaler", "")).lower() == "ranc"]
    configured = len(tasks)
    enabled = len(_enabled_tasks(tasks))
    excluded = len(_excluded_tasks(tasks))
    failed = len(failed_tasks)
    completed = max(enabled - failed, 0)
    pass_rates = [
        float(row["ranc_pass_rate"])
        for row in ranc_rows
        if row.get("ranc_pass_rate") not in {"", "nan"}
    ]
    pass_phrase = f" Mean RANC audit pass rate: {sum(pass_rates) / len(pass_rates):.2f}." if pass_rates else ""
    paragraph = (
        f"The public benchmark runner configured {configured} OpenML/UCI-style tasks: "
        f"{completed} completed, {failed} failed during fetch/run, and {excluded} were excluded by policy. "
        f"The completed tasks produced {len(aggregate_rows)} aggregate scaler rows. "
        "It stores metrics, table artifacts, and RANC audits only; raw public datasets are fetched at runtime."
    )
    if ranc_rows:
        paragraph += f" RANC rows present: {len(ranc_rows)}.{pass_phrase}"
    path = output_dir / "openml_result_paragraph.md"
    path.write_text(paragraph + "\n", encoding="utf-8")
    return path


def run(config: Dict[str, Any] | None = None) -> Tuple[Path, Path, List[Dict[str, Any]]]:
    cfg = dict(config or {})
    output_dir = Path(cfg.get("output_dir", "outputs/openml"))
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in cfg.get("seeds", [0, 1, 2])]
    tasks = list(cfg.get("tasks", DEFAULT_TASKS))
    included_tasks = _enabled_tasks(tasks)
    defaults = dict(cfg.get("defaults", {}))
    if not included_tasks:
        raise ValueError("OpenML public benchmark config has no enabled tasks.")
    continue_on_error = bool(cfg.get("continue_on_error", True))

    summary_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []
    failed_tasks: Dict[str, str] = {}
    for task in included_tasks:
        task_name = str(task["name"])
        benchmark_config = {
            "seeds": list(task.get("seeds", seeds)),
            "output_dir": str(output_dir),
            "benchmarks": [_task_to_benchmark(task, defaults)],
        }
        try:
            summary_path, aggregate_path, _ = run_benchmark(benchmark_config)
        except Exception as exc:
            if not continue_on_error:
                raise
            failed_tasks[task_name] = _compact_reason(f"{type(exc).__name__}: {exc}", limit=600)
            print(f"[openml_runner] skipped {task_name}: {failed_tasks[task_name]}", file=sys.stderr)
            continue
        task_summary_rows = _read_rows(summary_path)
        task_aggregate_rows = _read_rows(aggregate_path)
        for row in task_summary_rows:
            row["public_source"] = str(task.get("source", "openml"))
        for row in task_aggregate_rows:
            row["public_source"] = str(task.get("source", "openml"))
        summary_rows.extend(task_summary_rows)
        aggregate_rows.extend(task_aggregate_rows)

    if not aggregate_rows:
        _write_public_metadata(output_dir, tasks, seeds, failed_tasks)
        _write_exclusion_log(output_dir, tasks, seeds, failed_tasks)
        raise RuntimeError("No OpenML public benchmark tasks completed successfully.")

    benchmark_summary = output_dir / "benchmark_summary.csv"
    benchmark_aggregate = output_dir / "benchmark_aggregate.csv"
    openml_summary = output_dir / "openml_summary.csv"
    openml_aggregate = output_dir / "openml_aggregate.csv"
    _write_csv(benchmark_summary, summary_rows)
    _write_csv(benchmark_aggregate, aggregate_rows)
    _write_csv(openml_summary, summary_rows)
    _write_csv(openml_aggregate, aggregate_rows)

    _write_public_table(output_dir, openml_aggregate)
    _write_public_metadata(output_dir, tasks, seeds, failed_tasks)
    _write_exclusion_log(output_dir, tasks, seeds, failed_tasks)
    _write_paragraph(output_dir, tasks, failed_tasks, openml_aggregate)
    run_public_stats({"summary_path": str(openml_summary), "output_dir": str(output_dir)})
    _remove_stale_benchmark_artifacts(output_dir)
    return openml_summary, openml_aggregate, aggregate_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/openml_public.yaml"))
    args = parser.parse_args()
    config = _load_config(args.config) if args.config.exists() else {}
    summary_path, aggregate_path, _ = run(config)
    print(summary_path)
    print(aggregate_path)


if __name__ == "__main__":
    main()

"""Paired statistics for public OpenML/UCI benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import yaml


DEFAULT_BASELINES = ["standard", "robust", "selector"]


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _finite(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray([_as_float(value) for value in values], dtype=float)
    return arr[np.isfinite(arr)]


def _metric_for_benchmark(rows: Sequence[Mapping[str, str]]) -> Tuple[str, bool]:
    tasks = {str(row.get("task", "")) for row in rows}
    if "regression" in tasks:
        return "rmse", True
    aurocs = _finite(row.get("auroc", "") for row in rows)
    if aurocs.size:
        return "auroc", False
    return "accuracy", False


def _outcome(delta: float, tie_tol: float) -> str:
    if delta > tie_tol:
        return "win"
    if delta < -tie_tol:
        return "loss"
    return "tie"


def _bootstrap_ci(values: Sequence[float], n_bootstrap: int, random_state: int) -> Tuple[float, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(finite))
    if finite.size == 1 or n_bootstrap <= 0:
        return mean, mean
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, finite.size, size=(n_bootstrap, finite.size))
    means = finite[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _format_float(value: Any) -> str:
    number = _as_float(value)
    if not np.isfinite(number):
        return ""
    return f"{number:.4f}"


def _format_ci(low: Any, high: Any) -> str:
    low_f = _as_float(low)
    high_f = _as_float(high)
    if not np.isfinite(low_f) or not np.isfinite(high_f):
        return ""
    return f"[{low_f:.4f}, {high_f:.4f}]"


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


def paired_delta_rows(
    rows: List[Mapping[str, str]],
    baselines: Sequence[str] = DEFAULT_BASELINES,
    tie_tol: float = 1e-9,
) -> List[Dict[str, Any]]:
    by_benchmark: Dict[str, List[Mapping[str, str]]] = {}
    by_key: Dict[Tuple[str, str, str], Mapping[str, str]] = {}
    for row in rows:
        benchmark = str(row.get("benchmark", ""))
        seed = str(row.get("seed", ""))
        scaler = str(row.get("scaler", ""))
        by_benchmark.setdefault(benchmark, []).append(row)
        by_key[(benchmark, seed, scaler)] = row

    metrics = {benchmark: _metric_for_benchmark(group) for benchmark, group in by_benchmark.items()}
    out: List[Dict[str, Any]] = []
    for (benchmark, seed, scaler), ranc_row in sorted(by_key.items()):
        if scaler != "ranc":
            continue
        metric, lower_is_better = metrics[benchmark]
        ranc_value = _as_float(ranc_row.get(metric, ""))
        if not np.isfinite(ranc_value):
            continue
        for baseline in baselines:
            baseline_row = by_key.get((benchmark, seed, baseline))
            if baseline_row is None:
                continue
            baseline_value = _as_float(baseline_row.get(metric, ""))
            if not np.isfinite(baseline_value):
                continue
            delta = baseline_value - ranc_value if lower_is_better else ranc_value - baseline_value
            out.append(
                {
                    "benchmark": benchmark,
                    "seed": seed,
                    "task": ranc_row.get("task", ""),
                    "baseline": baseline,
                    "metric": metric,
                    "lower_is_better": str(lower_is_better),
                    "ranc_value": ranc_value,
                    "baseline_value": baseline_value,
                    "delta": delta,
                    "outcome": _outcome(delta, tie_tol),
                }
            )
    return out


def _task_summary_rows(
    delta_rows: List[Mapping[str, Any]],
    tie_tol: float,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in delta_rows:
        grouped.setdefault((str(row["baseline"]), str(row["benchmark"])), []).append(row)

    out: List[Dict[str, Any]] = []
    for (baseline, benchmark), group in sorted(grouped.items()):
        deltas = _finite(row.get("delta", "") for row in group)
        mean_delta = float(np.mean(deltas)) if deltas.size else float("nan")
        out.append(
            {
                "baseline": baseline,
                "benchmark": benchmark,
                "task": group[0].get("task", "") if group else "",
                "metric": group[0].get("metric", "") if group else "",
                "n_seed_pairs": int(deltas.size),
                "mean_delta": mean_delta,
                "outcome": _outcome(mean_delta, tie_tol) if deltas.size else "tie",
            }
        )
    return out


def _summary_rows(
    delta_rows: List[Mapping[str, Any]],
    task_rows: List[Mapping[str, Any]],
    baselines: Sequence[str],
    n_bootstrap: int,
    random_state: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for offset, baseline in enumerate(baselines):
        baseline_deltas = [row for row in delta_rows if row.get("baseline") == baseline]
        baseline_tasks = [row for row in task_rows if row.get("baseline") == baseline]
        if not baseline_deltas:
            continue
        deltas = _finite(row.get("delta", "") for row in baseline_deltas)
        ci_low, ci_high = _bootstrap_ci(deltas, n_bootstrap, random_state + offset)
        seed_outcomes = [str(row.get("outcome", "")) for row in baseline_deltas]
        task_outcomes = [str(row.get("outcome", "")) for row in baseline_tasks]
        out.append(
            {
                "baseline": baseline,
                "n_tasks": len(baseline_tasks),
                "n_seed_pairs": len(baseline_deltas),
                "task_wins": task_outcomes.count("win"),
                "task_losses": task_outcomes.count("loss"),
                "task_ties": task_outcomes.count("tie"),
                "seed_wins": seed_outcomes.count("win"),
                "seed_losses": seed_outcomes.count("loss"),
                "seed_ties": seed_outcomes.count("tie"),
                "mean_delta": float(np.mean(deltas)) if deltas.size else float("nan"),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
    return out


def _write_win_loss_table(output_dir: Path, summary_rows: List[Mapping[str, Any]]) -> Tuple[Path, Path]:
    headers = ["Baseline", "Tasks W/L/T", "Seed Pairs W/L/T", "Mean signed delta", "95% CI"]
    table_rows: List[List[str]] = []
    for row in summary_rows:
        table_rows.append(
            [
                str(row["baseline"]),
                f"{row['task_wins']}/{row['task_losses']}/{row['task_ties']}",
                f"{row['seed_wins']}/{row['seed_losses']}/{row['seed_ties']}",
                _format_float(row.get("mean_delta")),
                _format_ci(row.get("ci95_low"), row.get("ci95_high")),
            ]
        )

    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table_rows:
        md_lines.append("| " + " | ".join(row) + " |")
    md_path = output_dir / "openml_win_loss_table.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        "\\begin{tabular}{lllll}",
        "\\toprule",
        " & ".join(_escape_tex(header) for header in headers) + " \\\\",
        "\\midrule",
    ]
    for row in table_rows:
        tex_lines.append(" & ".join(_escape_tex(cell) for cell in row) + " \\\\")
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    tex_path = output_dir / "openml_win_loss_table.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    return md_path, tex_path


def _write_paragraph(output_dir: Path, summary_rows: List[Mapping[str, Any]]) -> Path:
    if not summary_rows:
        paragraph = (
            "No paired OpenML/UCI public benchmark comparisons were available for the configured "
            "RANC baseline set."
        )
    else:
        comparisons = []
        total_tasks = max(int(row.get("n_tasks", 0)) for row in summary_rows)
        total_pairs = sum(int(row.get("n_seed_pairs", 0)) for row in summary_rows)
        for row in summary_rows:
            comparisons.append(
                f"vs {row['baseline']}: task W/L/T={row['task_wins']}/{row['task_losses']}/{row['task_ties']}, "
                f"mean signed delta {_format_float(row.get('mean_delta'))} "
                f"(95% bootstrap CI {_format_ci(row.get('ci95_low'), row.get('ci95_high'))})"
            )
        paragraph = (
            f"Paired OpenML/UCI public comparisons used {total_tasks} completed tasks and "
            f"{total_pairs} seed-level RANC-baseline pairs. Positive signed deltas mean RANC is "
            "better after aligning metric direction. "
            + "; ".join(comparisons)
            + "."
        )
    path = output_dir / "openml_stats_paragraph.md"
    path.write_text(paragraph + "\n", encoding="utf-8")
    return path


def run(config: Dict[str, Any] | None = None) -> List[Path]:
    cfg = dict(config or {})
    summary_path = Path(cfg.get("summary_path", "outputs/openml/openml_summary.csv"))
    output_dir = Path(cfg.get("output_dir", summary_path.parent))
    baselines = [str(item) for item in cfg.get("baselines", DEFAULT_BASELINES)]
    tie_tol = float(cfg.get("tie_tol", 1e-9))
    n_bootstrap = int(cfg.get("bootstrap_samples", 5000))
    random_state = int(cfg.get("random_state", 0))

    rows = _read_rows(summary_path)
    delta_rows = paired_delta_rows(rows, baselines=baselines, tie_tol=tie_tol)
    task_rows = _task_summary_rows(delta_rows, tie_tol=tie_tol)
    summary_rows = _summary_rows(delta_rows, task_rows, baselines, n_bootstrap, random_state)

    delta_path = output_dir / "openml_pairwise_deltas.csv"
    task_path = output_dir / "openml_task_win_loss.csv"
    summary_path_out = output_dir / "openml_win_loss_summary.csv"
    _write_csv(delta_path, delta_rows)
    _write_csv(task_path, task_rows)
    _write_csv(summary_path_out, summary_rows)
    md_path, tex_path = _write_win_loss_table(output_dir, summary_rows)
    paragraph_path = _write_paragraph(output_dir, summary_rows)
    return [delta_path, summary_path_out, task_path, md_path, tex_path, paragraph_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--summary", type=Path, default=Path("outputs/openml/openml_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/openml"))
    args = parser.parse_args()

    config = _load_config(args.config) if args.config else {}
    config.setdefault("summary_path", str(args.summary))
    config.setdefault("output_dir", str(args.output_dir))
    for artifact in run(config):
        print(artifact)


if __name__ == "__main__":
    main()

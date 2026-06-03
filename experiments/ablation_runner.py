"""Ablation runner for ContractNet components."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import yaml

from experiments.tabular_runner import run as run_tabular


ABLATIONS = {
    "full": {
        "label": "Full RANC",
        "removed": "none",
        "baselines": ["ranc"],
        "override": {},
        "summary_scaler": "ranc",
    },
    "no_ledger_pressure": {
        "label": "No Signal Risk Ledger",
        "removed": "signal risk ledger",
        "baselines": ["ranc"],
        "override": {"constraints": {"global": {"metadata": {"disable_signal_risk_ledger": True}}}},
        "summary_scaler": "ranc",
    },
    "force_noop": {
        "label": "Forced no-op fallback",
        "removed": "admissible transform set",
        "baselines": ["ranc"],
        "override": {"constraints": {"global": {"hard_clauses": {"preserve_distance_ratios": True}}}},
        "summary_scaler": "ranc",
    },
    "no_outlier_damping": {
        "label": "No outlier damping",
        "removed": "outlier damping clause",
        "baselines": ["ranc"],
        "override": {
            "constraints": {
                "x0": {
                    "hard_clauses": {"damp_outliers": False},
                    "transform_preferences": [],
                }
            }
        },
        "summary_scaler": "ranc",
    },
    "selector_baseline": {
        "label": "Validation selector",
        "removed": "contract compiler",
        "baselines": ["selector"],
        "override": {},
        "summary_scaler": "selector",
    },
}


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


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


def _format_metric(value: Any) -> str:
    numeric = _as_float(value)
    if numeric != numeric:
        return ""
    return f"{numeric:.3f}"


def _format_int(value: Any) -> str:
    numeric = _as_float(value)
    if numeric != numeric:
        return ""
    return str(int(round(numeric)))


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


def _write_table(path_md: Path, path_tex: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "Ablation",
        "Removed",
        "Scaler",
        "Selected",
        "AUROC",
        "Accuracy",
        "Policies",
        "Ledger rows",
        "Rejected",
        "Downgrades",
        "Hard failures",
    ]
    table_rows = [
        [
            str(row["label"]),
            str(row["removed"]),
            str(row["scaler"]),
            str(row.get("selected_scaler", "")),
            _format_metric(row.get("auroc")),
            _format_metric(row.get("accuracy")),
            str(row.get("ranc_policy_summary", "")),
            _format_int(row.get("ranc_ledger_rows")),
            _format_int(row.get("ranc_rejected_candidates")),
            _format_int(row.get("ranc_downgrades")),
            _format_int(row.get("ranc_hard_failures")),
        ]
        for row in rows
    ]

    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table_rows:
        md_lines.append("| " + " | ".join(row) + " |")
    path_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        "\\begin{tabular}{" + "l" * len(headers) + "}",
        "\\toprule",
        " & ".join(_escape_tex(header) for header in headers) + " \\\\",
        "\\midrule",
    ]
    for row in table_rows:
        tex_lines.append(" & ".join(_escape_tex(cell) for cell in row) + " \\\\")
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    path_tex.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def _write_paragraph(path: Path, rows: List[Dict[str, Any]]) -> None:
    by_name = {str(row["ablation"]): row for row in rows}
    full = by_name.get("full", {})
    no_ledger = by_name.get("no_ledger_pressure", {})
    forced = by_name.get("force_noop", {})
    selector = by_name.get("selector_baseline", {})
    paragraph = (
        "On the ablation benchmark, full RANC recorded "
        f"{_format_int(full.get('ranc_ledger_rows'))} signal-risk ledger rows, "
        f"{_format_int(full.get('ranc_rejected_candidates'))} rejected candidates, and "
        f"{_format_int(full.get('ranc_downgrades'))} policy downgrades. Disabling the ledger reduced "
        f"ledger rows to {_format_int(no_ledger.get('ranc_ledger_rows'))}; forcing incompatible hard clauses "
        f"produced {_format_int(forced.get('ranc_downgrades'))} downgrades; and the validation selector chose "
        f"{selector.get('selected_scaler', '') or 'no scaler'} without producing a RANC audit."
    )
    path.write_text(paragraph + "\n", encoding="utf-8")


def _select_summary_row(rows: List[Dict[str, str]], scaler: str) -> Dict[str, str]:
    for row in rows:
        if str(row.get("scaler", "")).lower() == scaler.lower():
            return row
    raise RuntimeError(f"Could not find scaler {scaler!r} in ablation metrics.")


def run(config: Dict[str, Any]) -> Tuple[Path, List[Dict[str, Any]]]:
    default_root = Path(config.get("output_dir", "outputs/ablation")).parent / "ablations"
    root = Path(config.get("ablation_output_dir", default_root))
    root.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []
    for name, spec in ABLATIONS.items():
        ablation_config = _merge(config, spec["override"])
        ablation_config["baselines"] = list(spec["baselines"])
        ablation_config["output_dir"] = str(root / name)
        metrics_path, _ = run_tabular(ablation_config)
        row = dict(_select_summary_row(_read_rows(metrics_path), str(spec["summary_scaler"])))
        row.update(
            {
                "ablation": name,
                "label": spec["label"],
                "removed": spec["removed"],
                "metrics_path": str(metrics_path),
            }
        )
        summary_rows.append(row)

    summary_path = root / "ablation_summary.csv"
    _write_csv(summary_path, summary_rows)
    _write_table(root / "ablation_table.md", root / "ablation_table.tex", summary_rows)
    _write_paragraph(root / "ablation_result_paragraph.md", summary_rows)
    return summary_path, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/ablation.yaml"))
    args = parser.parse_args()
    summary_path, _ = run(_load_config(args.config))
    print(summary_path)


if __name__ == "__main__":
    main()

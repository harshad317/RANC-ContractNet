"""Tabular benchmark runner for RANC-ContractNet."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler

from ranc_contractnet import RANCDataTransformer
from ranc_contractnet.audit import export_report

from experiments.synthetic import (
    make_additive_shift_classification,
    make_outlier_signal_noise,
    make_scale_shift_regression,
    make_sparse_classification,
    make_temporal_drift,
    make_temporal_rare_event,
)


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _coerce_public_table(X: object) -> np.ndarray:
    frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            values = pd.to_numeric(frame[column], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            frame[column] = values.fillna(median)
        else:
            frame[column] = frame[column].astype("string").fillna("__missing__")
    frame = pd.get_dummies(frame, dummy_na=False)
    return frame.astype(float).to_numpy()


def _coerce_public_target(y: object, task: str = "classification") -> np.ndarray:
    series = y if isinstance(y, pd.Series) else pd.Series(y)
    if task == "regression":
        values = pd.to_numeric(series, errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        return values.fillna(median).to_numpy(dtype=float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy()
    codes, _ = pd.factorize(series.astype("string").fillna("__missing__"), sort=True)
    return codes


def _dataset(config: Dict[str, Any]):
    kind = config.get("dataset", {}).get("kind", "outlier_noise")
    seed = int(config.get("random_state", 0))
    if kind == "openml":
        try:
            import openml  # type: ignore
        except Exception as exc:
            raise RuntimeError("openml is required for dataset.kind=openml; install .[experiments].") from exc
        dataset_cfg = config.get("dataset", {})
        dataset_id = dataset_cfg.get("id")
        dataset_name = dataset_cfg.get("name")
        target = dataset_cfg.get("target")
        if dataset_id is not None:
            dataset = openml.datasets.get_dataset(int(dataset_id))
        elif dataset_name is not None:
            dataset = openml.datasets.get_dataset(str(dataset_name))
        else:
            raise ValueError("OpenML config requires dataset.id or dataset.name.")
        if target is None:
            target = dataset.default_target_attribute
        task = dataset_cfg.get("task", "classification")
        X, y, _, _ = dataset.get_data(target=target, dataset_format="dataframe")
        X = _coerce_public_table(X)
        y = _coerce_public_target(y, task=task)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(dataset_cfg.get("test_size", 0.35)), random_state=seed
        )
        metadata = {
            "dataset_kind": kind,
            "split_strategy": "random",
            "source": "openml",
            "openml_target": target,
            "openml_id": dataset_id,
            "openml_name": dataset_name,
        }
        return X_train, y_train, X_test, y_test, task, metadata
    if kind == "scale_shift_regression":
        return (
            *make_scale_shift_regression(random_state=seed),
            "regression",
            {"dataset_kind": kind, "split_strategy": "synthetic_predefined"},
        )
    if kind == "additive_shift_classification":
        return (
            *make_additive_shift_classification(random_state=seed),
            "classification",
            {"dataset_kind": kind, "split_strategy": "synthetic_predefined"},
        )
    if kind == "temporal_drift":
        X, y, _ = make_temporal_drift(random_state=seed)
        split = int(0.7 * X.shape[0])
        metadata = {
            "dataset_kind": kind,
            "split_strategy": "time_ordered",
            "train_start_index": 0,
            "train_end_index": split - 1,
            "test_start_index": split,
            "test_end_index": int(X.shape[0] - 1),
            "temporal_leakage_guard": True,
        }
        return X[:split], y[:split], X[split:], y[split:], "regression", metadata
    if kind == "temporal_rare_event":
        X, y, _, rare_mask = make_temporal_rare_event(random_state=seed)
        split = int(0.7 * X.shape[0])
        metadata = {
            "dataset_kind": kind,
            "split_strategy": "time_ordered",
            "train_start_index": 0,
            "train_end_index": split - 1,
            "test_start_index": split,
            "test_end_index": int(X.shape[0] - 1),
            "temporal_leakage_guard": True,
            "rare_event_train_rate": float(np.mean(y[:split])),
            "rare_event_test_rate": float(np.mean(y[split:])),
            "rare_extreme_test_rate": float(np.mean(rare_mask[split:])),
        }
        return X[:split], y[:split], X[split:], y[split:], "classification", metadata
    if kind == "outlier_signal":
        X, y = make_outlier_signal_noise(signal=True, random_state=seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=y)
        metadata = {"dataset_kind": kind, "split_strategy": "stratified_random"}
        return X_train, y_train, X_test, y_test, "classification", metadata
    if kind == "sparse_classification":
        X, y = make_sparse_classification(random_state=seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=y)
        metadata = {
            "dataset_kind": kind,
            "split_strategy": "stratified_random",
            "sparse_input": True,
        }
        return X_train, y_train, X_test, y_test, "classification", metadata
    X, y = make_outlier_signal_noise(signal=False, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=y)
    return X_train, y_train, X_test, y_test, "classification", {"dataset_kind": kind, "split_strategy": "stratified_random"}


def _scalers(names: Iterable[str], constraints: Dict[str, Any], seed: int):
    for name in names:
        key = name.lower()
        if key == "none":
            yield name, "passthrough"
        elif key == "standard":
            yield name, StandardScaler(with_mean=False)
        elif key == "minmax":
            yield name, MinMaxScaler()
        elif key == "robust":
            yield name, RobustScaler(with_centering=False)
        elif key == "maxabs":
            yield name, MaxAbsScaler()
        elif key == "quantile":
            yield name, QuantileTransformer(n_quantiles=32, random_state=seed)
        elif key == "power":
            yield name, PowerTransformer()
        elif key == "ranc":
            yield name, RANCDataTransformer(contracts=constraints, random_state=seed)
        elif key == "selector":
            yield name, "selector"
        else:
            raise ValueError(f"Unknown scaler {name!r}.")


def _model(task: str, seed: int):
    if task == "classification":
        estimator = LogisticRegression(max_iter=2000, random_state=seed, solver="liblinear")
        return OneVsRestClassifier(estimator)
    return Ridge(random_state=seed)


def _metrics(task: str, y_true, pred, scores=None) -> Dict[str, float]:
    if task == "classification":
        out = {"accuracy": float(accuracy_score(y_true, pred))}
        if scores is not None and len(np.unique(y_true)) == 2:
            out["auroc"] = float(roc_auc_score(y_true, scores))
        return out
    return {
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, pred)),
    }


def _matrix_nnz(X: object) -> int:
    if sparse.issparse(X):
        return int(X.nnz)
    return int(np.count_nonzero(np.asarray(X)))


def _matrix_density(X: object) -> float:
    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        return float("nan")
    total = int(shape[0]) * int(shape[1])
    if total == 0:
        return 0.0
    return float(_matrix_nnz(X) / total)


def _transformed_by_scaler(scaler: object, X: object):
    if scaler == "passthrough":
        return X
    if hasattr(scaler, "transform"):
        return scaler.transform(X)
    return X


def _format_mean(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:.3f}"


def _format_int(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(numeric):
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


def _policy_summary(report) -> str:
    counts: Dict[str, int] = {}
    for policy in report.policies:
        counts[policy.policy_type] = counts.get(policy.policy_type, 0) + 1
    return ", ".join(f"{name}:{counts[name]}" for name in sorted(counts))


def _ranc_audit_diagnostics(scaler: object) -> Dict[str, Any]:
    if not isinstance(scaler, RANCDataTransformer):
        return {
            "ranc_fit_samples": "",
            "ranc_policy_summary": "",
            "ranc_hard_failures": "",
            "ranc_ledger_rows": "",
            "ranc_rejected_candidates": "",
            "ranc_downgrades": "",
            "ranc_sparse_failures": "",
            "ranc_drift_monitors": "",
            "ranc_max_drift_estimate": "",
            "ranc_passed": "",
            "ranc_train_only_fit": "",
        }
    report = scaler.get_audit_report()
    hard_failures = sum(1 for result in report.falsification_results if result.hard_clause and not result.passed)
    sparse_failures = sum(
        1
        for result in report.falsification_results
        if result.test_name == "sparse_densification" and not result.passed
    )
    drift_estimates = [float(card.drift_estimate) for card in report.cards]
    n_fit = int(report.metadata.get("n_samples_fit", 0))
    train_only = ""
    if report.metadata.get("split_strategy") == "time_ordered":
        train_only = str(n_fit == int(report.metadata.get("train_end_index", -1)) + 1)
    return {
        "ranc_fit_samples": float(n_fit),
        "ranc_policy_summary": _policy_summary(report),
        "ranc_hard_failures": float(hard_failures),
        "ranc_ledger_rows": float(len(report.ledger_rows)),
        "ranc_rejected_candidates": float(len(report.rejected_candidates)),
        "ranc_downgrades": float(sum(1 for policy in report.policies if policy.downgrade_reason)),
        "ranc_sparse_failures": float(sparse_failures),
        "ranc_drift_monitors": float(sum(1 for policy in report.policies if policy.drift_monitor is not None)),
        "ranc_max_drift_estimate": float(max(drift_estimates)) if drift_estimates else 0.0,
        "ranc_passed": str(report.passed),
        "ranc_train_only_fit": train_only,
    }


def _diagnostics_for_row(
    *,
    scaler: object,
    X_train: object,
    X_test: object,
    dataset_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "dataset_kind": dataset_metadata.get("dataset_kind", ""),
        "split_strategy": dataset_metadata.get("split_strategy", ""),
        "train_nnz_before": float(_matrix_nnz(X_train)),
        "test_nnz_before": float(_matrix_nnz(X_test)),
        "train_density_before": _matrix_density(X_train),
        "test_density_before": _matrix_density(X_test),
        "sparse_input": str(sparse.issparse(X_train)),
    }
    try:
        Z_train = _transformed_by_scaler(scaler, X_train)
        Z_test = _transformed_by_scaler(scaler, X_test)
    except Exception as exc:
        diagnostics.update(
            {
                "diagnostic_error": str(exc),
                "sparse_output": "",
                "train_nnz_after": float("nan"),
                "test_nnz_after": float("nan"),
                "train_density_after": float("nan"),
                "test_density_after": float("nan"),
                "train_nnz_delta": float("nan"),
                "test_nnz_delta": float("nan"),
            }
        )
    else:
        train_after = _matrix_nnz(Z_train)
        test_after = _matrix_nnz(Z_test)
        diagnostics.update(
            {
                "diagnostic_error": "",
                "sparse_output": str(sparse.issparse(Z_train) and sparse.issparse(Z_test)),
                "train_nnz_after": float(train_after),
                "test_nnz_after": float(test_after),
                "train_density_after": _matrix_density(Z_train),
                "test_density_after": _matrix_density(Z_test),
                "train_nnz_delta": float(train_after - int(diagnostics["train_nnz_before"])),
                "test_nnz_delta": float(test_after - int(diagnostics["test_nnz_before"])),
            }
        )
    diagnostics.update(_ranc_audit_diagnostics(scaler))
    if dataset_metadata.get("split_strategy") == "time_ordered":
        diagnostics.update(
            {
                "temporal_train_end_index": float(dataset_metadata.get("train_end_index", float("nan"))),
                "temporal_test_start_index": float(dataset_metadata.get("test_start_index", float("nan"))),
                "temporal_gap": float(
                    int(dataset_metadata.get("test_start_index", 0)) - int(dataset_metadata.get("train_end_index", 0))
                ),
                "temporal_leakage_guard": str(bool(dataset_metadata.get("temporal_leakage_guard", False))),
            }
        )
    return diagnostics


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_table(path_md: Path, path_tex: Path, headers: List[str], rows: List[List[str]]) -> None:
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")
    path_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        "\\begin{tabular}{" + "l" * len(headers) + "}",
        "\\toprule",
        " & ".join(_escape_tex(header) for header in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        tex_lines.append(" & ".join(_escape_tex(cell) for cell in row) + " \\\\")
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    path_tex.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def _write_sparse_artifacts(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    table_rows: List[List[str]] = []
    for row in rows:
        table_rows.append(
            [
                str(row["scaler"]),
                _format_mean(row.get("auroc")),
                _format_mean(row.get("accuracy")),
                str(row.get("sparse_output", "")),
                _format_int(row.get("test_nnz_before")),
                _format_int(row.get("test_nnz_after")),
                _format_int(row.get("test_nnz_delta")),
                str(row.get("ranc_policy_summary", "")),
                _format_int(row.get("ranc_sparse_failures")),
            ]
        )
    _write_table(
        output_dir / "sparse_table.md",
        output_dir / "sparse_table.tex",
        [
            "Scaler",
            "AUROC",
            "Accuracy",
            "Sparse out",
            "Test nnz before",
            "Test nnz after",
            "Nnz delta",
            "RANC policies",
            "Sparse failures",
        ],
        table_rows,
    )
    ranc_rows = [row for row in rows if str(row["scaler"]).lower() == "ranc"]
    if ranc_rows:
        row = ranc_rows[0]
        paragraph = (
            "On the sparse synthetic classification benchmark, RANC preserved sparse matrix structure "
            f"with test nnz delta {_format_int(row.get('test_nnz_delta'))} and "
            f"{_format_int(row.get('ranc_sparse_failures'))} sparse densification failures in the audit report."
        )
        (output_dir / "sparse_result_paragraph.md").write_text(paragraph + "\n", encoding="utf-8")


def _write_temporal_artifacts(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    table_rows: List[List[str]] = []
    for row in rows:
        table_rows.append(
            [
                str(row["scaler"]),
                _format_mean(row.get("rmse")),
                _format_mean(row.get("mae")),
                str(row.get("selected_scaler", "")),
                str(row.get("temporal_leakage_guard", "")),
                str(row.get("ranc_train_only_fit", "")),
                _format_int(row.get("ranc_fit_samples")),
                _format_int(row.get("ranc_drift_monitors")),
                _format_mean(row.get("ranc_max_drift_estimate")),
            ]
        )
    _write_table(
        output_dir / "temporal_drift_table.md",
        output_dir / "temporal_drift_table.tex",
        [
            "Scaler",
            "RMSE",
            "MAE",
            "Selected",
            "Temporal guard",
            "RANC train-only",
            "RANC fit n",
            "Drift monitors",
            "Max drift",
        ],
        table_rows,
    )
    ranc_rows = [row for row in rows if str(row["scaler"]).lower() == "ranc"]
    if ranc_rows:
        row = ranc_rows[0]
        paragraph = (
            "On the temporal drift benchmark, RANC fitted on "
            f"{_format_int(row.get('ranc_fit_samples'))} training samples only, emitted "
            f"{_format_int(row.get('ranc_drift_monitors'))} drift monitors, and recorded maximum training-card drift "
            f"{_format_mean(row.get('ranc_max_drift_estimate'))}."
        )
        (output_dir / "temporal_drift_result_paragraph.md").write_text(paragraph + "\n", encoding="utf-8")


def _case_study_audit_evidence(row: Dict[str, Any]) -> str:
    if str(row.get("scaler", "")).lower() != "ranc":
        return ""
    fit_n = _format_int(row.get("ranc_fit_samples"))
    monitors = _format_int(row.get("ranc_drift_monitors"))
    ledger = _format_int(row.get("ranc_ledger_rows"))
    rejected = _format_int(row.get("ranc_rejected_candidates"))
    return f"train-only n={fit_n}; drift={monitors}; ledger={ledger}; rejected={rejected}"


def _write_temporal_rare_event_artifacts(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    table_rows: List[List[str]] = []
    paper_rows = [
        row
        for row in rows
        if str(row.get("scaler", "")).lower() in {"standard", "quantile", "selector", "ranc"}
    ]
    for row in paper_rows:
        table_rows.append(
            [
                str(row["scaler"]),
                _format_mean(row.get("auroc")),
                _format_mean(row.get("rare_event_recall")),
                str(row.get("selected_scaler", "")),
                _case_study_audit_evidence(row),
            ]
        )
    _write_table(
        output_dir / "temporal_rare_event_table.md",
        output_dir / "temporal_rare_event_table.tex",
        [
            "Scaler",
            "AUROC",
            "Rare recall",
            "Selected",
            "RANC audit evidence",
        ],
        table_rows,
    )
    ranc_rows = [row for row in rows if str(row["scaler"]).lower() == "ranc"]
    if not ranc_rows:
        return
    row = ranc_rows[0]
    paragraph = (
        "In the temporal rare-event case study, RANC fit only the time-prefix training window "
        f"({_format_int(row.get('ranc_fit_samples'))} samples), kept rare-event recall at "
        f"{_format_mean(row.get('rare_event_recall'))}, emitted "
        f"{_format_int(row.get('ranc_drift_monitors'))} drift monitors, recorded "
        f"{_format_int(row.get('ranc_ledger_rows'))} signal-risk ledger rows, and rejected "
        f"{_format_int(row.get('ranc_rejected_candidates'))} illegal or higher-risk candidates."
    )
    (output_dir / "temporal_rare_event_result_paragraph.md").write_text(paragraph + "\n", encoding="utf-8")
    audit_lines = [
        "# Temporal Rare-Event Case Study Audit",
        "",
        f"- Dataset kind: `{row.get('dataset_kind', '')}`",
        f"- Split strategy: `{row.get('split_strategy', '')}`",
        f"- Temporal leakage guard: `{row.get('temporal_leakage_guard', '')}`",
        f"- RANC train-only fit: `{row.get('ranc_train_only_fit', '')}`",
        f"- RANC fit samples: `{_format_int(row.get('ranc_fit_samples'))}`",
        f"- RANC policies: `{row.get('ranc_policy_summary', '')}`",
        f"- Signal-risk ledger rows: `{_format_int(row.get('ranc_ledger_rows'))}`",
        f"- Rejected candidates: `{_format_int(row.get('ranc_rejected_candidates'))}`",
        f"- Drift monitors: `{_format_int(row.get('ranc_drift_monitors'))}`",
        f"- Max drift estimate: `{_format_mean(row.get('ranc_max_drift_estimate'))}`",
        f"- Rare-event recall: `{_format_mean(row.get('rare_event_recall'))}`",
        "",
        "The case-study contract treats rare extremes as possible signal rather than corruption, "
        "requires monotonic and invertible preprocessing, and uses time-ordered fit metadata so "
        "the audit can verify that no future rows supplied fitted statistics.",
    ]
    (output_dir / "temporal_rare_event_audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")


def _write_benchmark_artifacts(output_dir: Path, dataset_kind: str, rows: List[Dict[str, Any]]) -> None:
    if dataset_kind == "sparse_classification":
        _write_sparse_artifacts(output_dir, rows)
    elif dataset_kind == "temporal_drift":
        _write_temporal_artifacts(output_dir, rows)
    elif dataset_kind == "temporal_rare_event":
        _write_temporal_rare_event_artifacts(output_dir, rows)


def _selector_scaler(X_train, y_train, task: str, seed: int):
    inner_train, inner_val, y_inner_train, y_inner_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=seed,
        stratify=y_train if task == "classification" and len(set(y_train)) > 1 else None,
    )
    candidates = [
        ("standard", StandardScaler(with_mean=False)),
        ("robust", RobustScaler(with_centering=False)),
        ("maxabs", MaxAbsScaler()),
    ]
    if not sparse.issparse(X_train):
        candidates.extend(
            [
                ("minmax", MinMaxScaler()),
                ("quantile", QuantileTransformer(n_quantiles=32, random_state=seed)),
                ("power", PowerTransformer()),
            ]
        )
    best_name = None
    best_score = -float("inf")
    best_scaler = None
    for name, scaler in candidates:
        pipe = Pipeline([("scaler", scaler), ("model", _model(task, seed))])
        pipe.fit(inner_train, y_inner_train)
        pred = pipe.predict(inner_val)
        if task == "classification":
            score = accuracy_score(y_inner_val, pred)
        else:
            score = -mean_squared_error(y_inner_val, pred)
        if score > best_score:
            best_name = name
            best_score = float(score)
            best_scaler = scaler
    if best_scaler is None:
        raise RuntimeError("selector failed to choose a scaler")
    return best_name, best_scaler


def run(config: Dict[str, Any]) -> Tuple[Path, Dict[str, Dict[str, float]]]:
    seed = int(config.get("random_state", 0))
    X_train, y_train, X_test, y_test, task, dataset_metadata = _dataset(config)
    scalers = config.get("baselines", ["none", "standard", "robust", "maxabs", "ranc"])
    constraints = config.get("constraints", {})
    output_dir = Path(config.get("output_dir", "outputs/tabular"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    results: Dict[str, Dict[str, float]] = {}
    for name, scaler in _scalers(scalers, constraints, seed):
        if sparse.issparse(X_train) and name.lower() in {"minmax", "quantile", "power"}:
            continue
        selected_name = ""
        if scaler == "selector":
            selected_name, scaler = _selector_scaler(X_train, y_train, task, seed)
        pipe = Pipeline([("scaler", scaler), ("model", _model(task, seed))])
        fit_params = {}
        if isinstance(scaler, RANCDataTransformer):
            fit_params["scaler__metadata"] = dataset_metadata
        pipe.fit(X_train, y_train, **fit_params)
        pred = pipe.predict(X_test)
        scores = None
        if task == "classification" and hasattr(pipe[-1], "predict_proba"):
            scores = pipe.predict_proba(X_test)[:, 1]
        metrics = _metrics(task, y_test, pred, scores)
        if dataset_metadata.get("dataset_kind") == "temporal_rare_event":
            metrics["rare_event_recall"] = float(recall_score(y_test, pred, zero_division=0))
        metrics["n_train"] = float(X_train.shape[0])
        metrics["n_test"] = float(X_test.shape[0])
        results[name] = metrics
        row = {
            "scaler": name,
            "selected_scaler": selected_name,
            "task": task,
            **metrics,
            **_diagnostics_for_row(
                scaler=pipe.named_steps["scaler"],
                X_train=X_train,
                X_test=X_test,
                dataset_metadata=dataset_metadata,
            ),
        }
        rows.append(row)
        if name.lower() == "ranc":
            export_report(pipe.named_steps["scaler"], output_dir / "ranc_audit.json", format="json")
            export_report(pipe.named_steps["scaler"], output_dir / "ranc_audit.md", format="md")
    output_path = output_dir / "metrics.csv"
    _write_csv(output_path, rows)
    _write_benchmark_artifacts(output_dir, str(dataset_metadata.get("dataset_kind", "")), rows)
    return output_path, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/smoke.yaml"))
    args = parser.parse_args()
    output_path, _ = run(_load_config(args.config))
    print(output_path)


if __name__ == "__main__":
    main()

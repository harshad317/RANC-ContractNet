"""Paired outlier-as-noise versus outlier-as-signal benchmark."""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import yaml
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, QuantileTransformer, RobustScaler, StandardScaler

from experiments.baselines import WinsorizingScaler
from experiments.synthetic import OutlierDataset, make_paired_outlier_signal_noise
from ranc_contractnet import RANCDataTransformer
from ranc_contractnet.audit import export_report


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _scenario_constraints(contract_scenario: str) -> Dict[str, Any]:
    base = {
        "global": {
            "hard_clauses": {
                "preserve_monotonicity": True,
                "allow_inverse_transform": True,
            }
        }
    }
    if contract_scenario == "noise":
        base["x0"] = {
            "hard_clauses": {
                "damp_outliers": True,
                "enforce_scale_invariance": True,
                "enforce_shift_invariance": False,
            },
            "transform_preferences": ["log1p"],
        }
    elif contract_scenario == "signal":
        base["x0"] = {
            "hard_clauses": {
                "preserve_extreme_signal": True,
                "damp_outliers": False,
                "enforce_scale_invariance": True,
            }
        }
    else:
        raise ValueError(f"Unknown contract scenario {contract_scenario!r}.")
    return base


def _method_contract_scenario(name: str, scenario: str) -> Optional[str]:
    key = name.lower()
    if key in {"ranc", "ranc_correct"}:
        return scenario
    if key == "ranc_wrong_noise":
        return "signal" if scenario == "noise" else None
    if key == "ranc_wrong_signal":
        return "noise" if scenario == "signal" else None
    if key == "ranc_wrong":
        return "signal" if scenario == "noise" else "noise"
    return None


def _display_method_name(name: str, scenario: str) -> str:
    key = name.lower()
    if key == "ranc":
        return "ranc"
    if key == "ranc_correct":
        return "ranc_correct"
    if key == "ranc_wrong":
        return "ranc_wrong_noise" if scenario == "noise" else "ranc_wrong_signal"
    return key


def _scaler(name: str, scenario: str, seed: int):
    key = name.lower()
    if key == "none":
        return "passthrough"
    if key == "standard":
        return StandardScaler()
    if key == "robust":
        return RobustScaler()
    if key == "maxabs":
        return MaxAbsScaler()
    if key == "quantile":
        return QuantileTransformer(n_quantiles=64, random_state=seed, output_distribution="normal")
    if key == "winsor":
        return WinsorizingScaler(lower=0.01, upper=0.95 if scenario == "noise" else 0.99)
    contract_scenario = _method_contract_scenario(key, scenario)
    if contract_scenario is not None:
        return RANCDataTransformer(contracts=_scenario_constraints(contract_scenario), random_state=seed)
    raise ValueError(f"Unknown scaler {name!r}.")


def _split(dataset: OutlierDataset, seed: int, test_size: float):
    indices = np.arange(dataset.X.shape[0])
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=dataset.y,
    )
    return (
        dataset.X[train_idx],
        dataset.X[test_idx],
        dataset.y[train_idx],
        dataset.y[test_idx],
        dataset.outlier_mask[train_idx],
        dataset.outlier_mask[test_idx],
    )


def _safe_auroc(y_true: np.ndarray, proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, proba))


def _tail_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    tail = mask.astype(bool)
    non_tail = ~tail
    tail_positive = tail & (y_true == 1)
    tail_negative = tail & (y_true == 0)
    metrics = {
        "tail_count": float(np.sum(tail)),
        "tail_label_rate": float(np.mean(y_true[tail])) if np.any(tail) else float("nan"),
        "tail_predicted_positive_rate": float(np.mean(y_pred[tail])) if np.any(tail) else float("nan"),
        "non_tail_predicted_positive_rate": float(np.mean(y_pred[non_tail])) if np.any(non_tail) else float("nan"),
        "tail_mean_probability": float(np.mean(proba[tail])) if np.any(tail) else float("nan"),
        "non_tail_mean_probability": float(np.mean(proba[non_tail])) if np.any(non_tail) else float("nan"),
    }
    metrics["tail_probability_gap"] = metrics["tail_mean_probability"] - metrics["non_tail_mean_probability"]
    metrics["rare_positive_recall"] = (
        float(np.mean(y_pred[tail_positive] == 1)) if np.any(tail_positive) else float("nan")
    )
    metrics["tail_false_positive_rate"] = (
        float(np.mean(y_pred[tail_negative] == 1)) if np.any(tail_negative) else float("nan")
    )
    return metrics


def _contract_metrics(scenario: str, metrics: Dict[str, float], auroc: float) -> Dict[str, float]:
    """Scenario-specific failure score where lower means better contract alignment."""

    if scenario == "noise":
        violation = 1.0 - auroc if np.isfinite(auroc) else float("nan")
        secondary = max(0.0, metrics["tail_probability_gap"])
    elif scenario == "signal":
        recall = metrics["rare_positive_recall"]
        violation = 1.0 - recall if np.isfinite(recall) else float("nan")
        secondary = max(0.0, -metrics["tail_probability_gap"])
    else:
        raise ValueError(f"Unknown scenario {scenario!r}.")
    return {
        "contract_violation_rate": float(violation),
        "contract_secondary_violation": float(secondary),
    }


def _ranc_audit_metrics(
    pipe: Pipeline,
    output_dir: Path,
    scenario: str,
    contract_scenario: Optional[str],
    seed: int,
    method_name: str,
) -> Dict[str, Any]:
    if contract_scenario is None:
        return {
            "x0_policy": "",
            "hard_failures": 0.0,
            "ledger_rows": 0.0,
            "contract_scenario": "",
            "contract_correct": "",
        }
    transformer = pipe.named_steps["scaler"]
    report = transformer.get_audit_report()
    audit_stem = output_dir / f"{method_name}_{scenario}_contract_{contract_scenario}_seed{seed}"
    export_report(transformer, audit_stem.with_suffix(".json"), format="json")
    export_report(transformer, audit_stem.with_suffix(".md"), format="md")
    x0_policy = next((policy.policy_type for policy in report.policies if policy.feature_name == "x0"), "")
    hard_failures = sum(1 for result in report.falsification_results if result.hard_clause and not result.passed)
    return {
        "x0_policy": x0_policy,
        "hard_failures": float(hard_failures),
        "ledger_rows": float(len(report.ledger_rows)),
        "contract_scenario": contract_scenario,
        "contract_correct": str(contract_scenario == scenario),
    }


def _format_mean_std(mean: float, std: float) -> str:
    if np.isnan(mean):
        return ""
    return f"{mean:.3f} +/- {std:.3f}"


def _format_ci(low: float, high: float) -> str:
    if np.isnan(low) or np.isnan(high):
        return ""
    return f"[{low:.3f}, {high:.3f}]"


def _format_p_value(value: float) -> str:
    if np.isnan(value):
        return ""
    if value < 1e-4:
        return "<1e-4"
    return f"{value:.4f}"


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


def _aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scenario"]), str(row["scaler"]))].append(row)
    summary: List[Dict[str, Any]] = []
    metric_names = [
        "accuracy",
        "auroc",
        "rare_positive_recall",
        "tail_false_positive_rate",
        "tail_mean_probability",
        "non_tail_mean_probability",
        "tail_probability_gap",
        "tail_label_rate",
        "tail_count",
        "contract_violation_rate",
        "contract_secondary_violation",
        "hard_failures",
        "ledger_rows",
    ]
    for (scenario, scaler), group in sorted(grouped.items()):
        out: Dict[str, Any] = {"scenario": scenario, "scaler": scaler}
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            finite = values[np.isfinite(values)]
            out[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            out[f"{metric}_std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        policies = sorted({str(row.get("x0_policy", "")) for row in group if row.get("x0_policy", "")})
        out["x0_policy"] = "+".join(policies)
        contract_scenarios = sorted(
            {str(row.get("contract_scenario", "")) for row in group if row.get("contract_scenario", "")}
        )
        correct_values = sorted(
            {str(row.get("contract_correct", "")) for row in group if str(row.get("contract_correct", ""))}
        )
        out["contract_scenario"] = "+".join(contract_scenarios)
        out["contract_correct"] = "+".join(correct_values)
        summary.append(out)
    return summary


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _clear_previous_audits(output_dir: Path) -> None:
    for pattern in ("ranc*_seed*.json", "ranc*_seed*.md"):
        for path in output_dir.glob(pattern):
            path.unlink()


def _write_tables(output_dir: Path, summary: List[Dict[str, Any]]) -> None:
    broad_summary = [
        row
        for row in summary
        if str(row["scaler"]).lower() not in {"ranc_correct", "ranc_wrong", "ranc_wrong_noise", "ranc_wrong_signal"}
    ]
    lines = [
        "| Scenario | Scaler | AUROC | Rare recall | Tail FPR | Tail prob gap | RANC x0 policy |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    tex_lines = [
        "\\begin{tabular}{lllllll}",
        "\\toprule",
        "Scenario & Scaler & AUROC & Rare recall & Tail FPR & Tail prob gap & RANC x0 policy \\\\",
        "\\midrule",
    ]
    for row in broad_summary:
        values = {
            "scenario": str(row["scenario"]),
            "scaler": str(row["scaler"]),
            "auroc": _format_mean_std(row["auroc_mean"], row["auroc_std"]),
            "rare_positive_recall": _format_mean_std(
                row["rare_positive_recall_mean"], row["rare_positive_recall_std"]
            ),
            "tail_false_positive_rate": _format_mean_std(
                row["tail_false_positive_rate_mean"], row["tail_false_positive_rate_std"]
            ),
            "tail_probability_gap": _format_mean_std(
                row["tail_probability_gap_mean"], row["tail_probability_gap_std"]
            ),
            "x0_policy": str(row.get("x0_policy", "")),
        }
        lines.append(
            "| {scenario} | {scaler} | {auroc} | {rare_positive_recall} | "
            "{tail_false_positive_rate} | {tail_probability_gap} | {x0_policy} |".format(**values)
        )
        tex_lines.append(
            "{scenario} & {scaler} & {auroc} & {rare_positive_recall} & "
            "{tail_false_positive_rate} & {tail_probability_gap} & {x0_policy} \\\\".format(
                **{key: _escape_tex(value) for key, value in values.items()}
            )
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output_dir / "outlier_signal_noise_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "outlier_signal_noise_table.tex").write_text(
        "\n".join(tex_lines) + "\n", encoding="utf-8"
    )


def _causal_rows(summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    methods = {"ranc_correct", "ranc_wrong_noise", "ranc_wrong_signal", "ranc_wrong"}
    rows = [row for row in summary if str(row["scaler"]).lower() in methods]
    order = {"noise": 0, "signal": 1}
    method_order = {"ranc_correct": 0, "ranc_wrong_noise": 1, "ranc_wrong_signal": 1, "ranc_wrong": 1}
    return sorted(
        rows,
        key=lambda row: (order.get(str(row["scenario"]), 99), method_order.get(str(row["scaler"]), 99)),
    )


def _write_causal_tables(output_dir: Path, summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = _causal_rows(summary)
    if not rows:
        return []
    _write_csv(output_dir / "contract_causal_summary.csv", rows)
    lines = [
        "| Scenario | Method | Contract | Correct | x0 policy | AUROC | Rare recall | Tail FPR | Contract violations |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    tex_lines = [
        "\\begin{tabular}{lllllllll}",
        "\\toprule",
        "Scenario & Method & Contract & Correct & x0 policy & AUROC & Rare recall & Tail FPR & Violations \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = {
            "scenario": str(row["scenario"]),
            "scaler": str(row["scaler"]),
            "contract": str(row.get("contract_scenario", "")),
            "correct": str(row.get("contract_correct", "")),
            "x0_policy": str(row.get("x0_policy", "")),
            "auroc": _format_mean_std(row["auroc_mean"], row["auroc_std"]),
            "rare": _format_mean_std(row["rare_positive_recall_mean"], row["rare_positive_recall_std"]),
            "tail_fpr": _format_mean_std(
                row["tail_false_positive_rate_mean"], row["tail_false_positive_rate_std"]
            ),
            "violations": _format_mean_std(
                row["contract_violation_rate_mean"], row["contract_violation_rate_std"]
            ),
        }
        lines.append(
            "| {scenario} | {scaler} | {contract} | {correct} | {x0_policy} | {auroc} | "
            "{rare} | {tail_fpr} | {violations} |".format(**values)
        )
        tex_lines.append(
            "{scenario} & {scaler} & {contract} & {correct} & {x0_policy} & {auroc} & "
            "{rare} & {tail_fpr} & {violations} \\\\".format(
                **{key: _escape_tex(value) for key, value in values.items()}
            )
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output_dir / "contract_causal_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "contract_causal_table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    return rows


def _bootstrap_ci(values: np.ndarray, n_bootstrap: int, random_state: int) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    if finite.size == 1 or n_bootstrap <= 0:
        mean = float(np.mean(finite))
        return mean, mean
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, finite.size, size=(n_bootstrap, finite.size))
    means = np.mean(finite[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_p_values(deltas: np.ndarray) -> Tuple[float, float]:
    finite = deltas[np.isfinite(deltas)]
    if finite.size < 2:
        return float("nan"), float("nan")
    try:
        t_p = float(stats.ttest_1samp(finite, popmean=0.0, alternative="greater").pvalue)
    except Exception:
        t_p = float("nan")
    try:
        if np.allclose(finite, 0.0):
            w_p = 1.0
        else:
            method = "exact" if finite.size <= 20 else "auto"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                w_p = float(
                    stats.wilcoxon(
                        finite,
                        alternative="greater",
                        zero_method="wilcox",
                        method=method,
                    ).pvalue
                )
    except Exception:
        w_p = float("nan")
    return t_p, w_p


def _paired_effect_size(deltas: np.ndarray) -> float:
    finite = deltas[np.isfinite(deltas)]
    if finite.size < 2:
        return float("nan")
    std = float(np.std(finite, ddof=1))
    if std <= 1e-12:
        return float("inf") if float(np.mean(finite)) > 0 else 0.0
    return float(np.mean(finite) / std)


def _rows_by_seed(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    indexed: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for row in rows:
        indexed[(str(row["scenario"]), int(row["seed"]), str(row["scaler"]))] = row
    return indexed


def _stat_specs() -> List[Dict[str, str]]:
    return [
        {
            "scenario": "noise",
            "metric": "auroc",
            "label": "Corruption AUROC",
            "direction": "higher",
            "wrong_method": "ranc_wrong_noise",
        },
        {
            "scenario": "noise",
            "metric": "contract_violation_rate",
            "label": "Contract violation",
            "direction": "lower",
            "wrong_method": "ranc_wrong_noise",
        },
        {
            "scenario": "signal",
            "metric": "rare_positive_recall",
            "label": "Rare recall",
            "direction": "higher",
            "wrong_method": "ranc_wrong_signal",
        },
        {
            "scenario": "signal",
            "metric": "contract_violation_rate",
            "label": "Contract violation",
            "direction": "lower",
            "wrong_method": "ranc_wrong_signal",
        },
    ]


def _paired_delta_rows(
    rows: List[Dict[str, Any]],
    *,
    n_bootstrap: int,
    random_state: int,
) -> List[Dict[str, Any]]:
    indexed = _rows_by_seed(rows)
    out: List[Dict[str, Any]] = []
    for spec in _stat_specs():
        scenario = spec["scenario"]
        metric = spec["metric"]
        wrong_method = spec["wrong_method"]
        seeds = sorted(
            seed
            for row_scenario, seed, method in indexed.keys()
            if row_scenario == scenario
            and method == "ranc_correct"
            and (scenario, seed, wrong_method) in indexed
        )
        correct_values: List[float] = []
        wrong_values: List[float] = []
        deltas: List[float] = []
        for seed in seeds:
            correct = float(indexed[(scenario, seed, "ranc_correct")][metric])
            wrong = float(indexed[(scenario, seed, wrong_method)][metric])
            if not np.isfinite(correct) or not np.isfinite(wrong):
                continue
            correct_values.append(correct)
            wrong_values.append(wrong)
            if spec["direction"] == "lower":
                deltas.append(wrong - correct)
            else:
                deltas.append(correct - wrong)
        correct_arr = np.asarray(correct_values, dtype=float)
        wrong_arr = np.asarray(wrong_values, dtype=float)
        delta_arr = np.asarray(deltas, dtype=float)
        ci_low, ci_high = _bootstrap_ci(delta_arr, n_bootstrap, random_state + len(out))
        t_p, w_p = _paired_p_values(delta_arr)
        out.append(
            {
                "scenario": scenario,
                "metric": metric,
                "metric_label": spec["label"],
                "preferred_direction": spec["direction"],
                "correct_method": "ranc_correct",
                "wrong_method": wrong_method,
                "n_pairs": int(delta_arr.size),
                "correct_mean": float(np.mean(correct_arr)) if correct_arr.size else float("nan"),
                "wrong_mean": float(np.mean(wrong_arr)) if wrong_arr.size else float("nan"),
                "delta_mean": float(np.mean(delta_arr)) if delta_arr.size else float("nan"),
                "delta_std": float(np.std(delta_arr, ddof=1)) if delta_arr.size > 1 else 0.0,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "paired_t_p_value": t_p,
                "wilcoxon_p_value": w_p,
                "effect_size_dz": _paired_effect_size(delta_arr),
            }
        )
    return out


def _write_delta_tables(output_dir: Path, delta_rows: List[Dict[str, Any]]) -> None:
    if not delta_rows:
        return
    _write_csv(output_dir / "contract_delta_stats.csv", delta_rows)
    lines = [
        "| Scenario | Metric | Correct | Wrong | Delta | 95% CI | Wilcoxon p | Effect dz |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    tex_lines = [
        "\\begin{tabular}{llllllll}",
        "\\toprule",
        "Scenario & Metric & Correct & Wrong & Delta & 95\\% CI & Wilcoxon p & Effect dz \\\\",
        "\\midrule",
    ]
    for row in delta_rows:
        values = {
            "scenario": str(row["scenario"]),
            "metric": str(row["metric_label"]),
            "correct": f"{float(row['correct_mean']):.3f}",
            "wrong": f"{float(row['wrong_mean']):.3f}",
            "delta": f"{float(row['delta_mean']):.3f}",
            "ci": _format_ci(float(row["ci95_low"]), float(row["ci95_high"])),
            "p": _format_p_value(float(row["wilcoxon_p_value"])),
            "effect": f"{float(row['effect_size_dz']):.3f}",
        }
        lines.append(
            "| {scenario} | {metric} | {correct} | {wrong} | {delta} | {ci} | {p} | {effect} |".format(
                **values
            )
        )
        tex_lines.append(
            "{scenario} & {metric} & {correct} & {wrong} & {delta} & {ci} & {p} & {effect} \\\\".format(
                **{key: _escape_tex(value) for key, value in values.items()}
            )
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output_dir / "contract_delta_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "contract_delta_table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def _write_statistical_paragraph(output_dir: Path, delta_rows: List[Dict[str, Any]]) -> None:
    by_key = {(row["scenario"], row["metric"]): row for row in delta_rows}
    noise = by_key.get(("noise", "auroc"))
    signal = by_key.get(("signal", "rare_positive_recall"))
    if not noise or not signal:
        return
    n_pairs = min(int(noise["n_pairs"]), int(signal["n_pairs"]))
    paragraph = (
        f"Across {n_pairs} paired synthetic seeds, the correct RANC contract improved "
        f"noise-corruption AUROC by {float(noise['delta_mean']):.3f} "
        f"(95% bootstrap CI {_format_ci(float(noise['ci95_low']), float(noise['ci95_high']))}, "
        f"Wilcoxon p={_format_p_value(float(noise['wilcoxon_p_value']))}) compared with applying the "
        f"signal-preserving contract to the noise setting. In the signal setting, the correct contract "
        f"improved rare-event recall by {float(signal['delta_mean']):.3f} "
        f"(95% bootstrap CI {_format_ci(float(signal['ci95_low']), float(signal['ci95_high']))}, "
        f"Wilcoxon p={_format_p_value(float(signal['wilcoxon_p_value']))}) compared with applying the "
        f"noise-damping contract to true rare-signal extremes."
    )
    (output_dir / "contract_statistical_paragraph.md").write_text(paragraph + "\n", encoding="utf-8")


def run(config: Dict[str, Any]) -> Tuple[Path, Path]:
    seeds = [int(seed) for seed in config.get("seeds", [0, 1, 2, 3, 4])]
    n_samples = int(config.get("n_samples", 800))
    outlier_fraction = float(config.get("outlier_fraction", 0.06))
    test_size = float(config.get("test_size", 0.35))
    baselines = list(
        config.get(
            "baselines",
            [
                "none",
                "standard",
                "robust",
                "winsor",
                "quantile",
                "ranc",
                "ranc_correct",
                "ranc_wrong",
            ],
        )
    )
    output_dir = Path(config.get("output_dir", "outputs/outlier_pair"))
    n_bootstrap = int(config.get("bootstrap_samples", 5000))
    stats_seed = int(config.get("stats_random_state", 12345))
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_audits(output_dir)
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        paired = make_paired_outlier_signal_noise(
            n_samples=n_samples,
            outlier_fraction=outlier_fraction,
            random_state=seed,
        )
        for scenario, dataset in paired.items():
            X_train, X_test, y_train, y_test, _, test_mask = _split(dataset, seed, test_size)
            for scaler_name in baselines:
                method_name = _display_method_name(scaler_name, scenario)
                contract_scenario = _method_contract_scenario(scaler_name, scenario)
                if scaler_name.lower() in {"ranc_wrong_noise", "ranc_wrong_signal"} and contract_scenario is None:
                    continue
                scaler = _scaler(scaler_name, scenario, seed)
                pipe = Pipeline(
                    [
                        ("scaler", scaler),
                        ("model", LogisticRegression(max_iter=600, random_state=seed)),
                    ]
                )
                pipe.fit(X_train, y_train)
                proba = pipe.predict_proba(X_test)[:, 1]
                y_pred = (proba >= 0.5).astype(int)
                row: Dict[str, Any] = {
                    "seed": seed,
                    "scenario": scenario,
                    "scaler": method_name,
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "auroc": _safe_auroc(y_test, proba),
                    "n_train": float(X_train.shape[0]),
                    "n_test": float(X_test.shape[0]),
                }
                tail_metrics = _tail_metrics(y_test, y_pred, proba, test_mask)
                row.update(tail_metrics)
                row.update(_contract_metrics(scenario, tail_metrics, row["auroc"]))
                row.update(
                    _ranc_audit_metrics(
                        pipe,
                        output_dir,
                        scenario,
                        contract_scenario,
                        seed,
                        method_name,
                    )
                )
                rows.append(row)
    summary = _aggregate(rows)
    raw_path = output_dir / "raw_results.csv"
    summary_path = output_dir / "summary.csv"
    _write_csv(raw_path, rows)
    _write_csv(summary_path, summary)
    _write_tables(output_dir, summary)
    _write_causal_tables(output_dir, summary)
    delta_rows = _paired_delta_rows(rows, n_bootstrap=n_bootstrap, random_state=stats_seed)
    _write_delta_tables(output_dir, delta_rows)
    _write_statistical_paragraph(output_dir, delta_rows)
    return raw_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/outlier_pair.yaml"))
    args = parser.parse_args()
    raw_path, summary_path = run(_load_config(args.config))
    print(raw_path)
    print(summary_path)


if __name__ == "__main__":
    main()

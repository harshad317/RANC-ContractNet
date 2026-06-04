import csv

from experiments.ablation_runner import run as run_ablation
from experiments.benchmark_runner import run as run_benchmark
from experiments.openml_runner import run as run_openml
from experiments.paper_results_runner import run as run_paper_results
from experiments.public_stats_runner import run as run_public_stats
from experiments.tabular_runner import run


def test_tabular_runner_smoke(tmp_path):
    output, results = run(
        {
            "random_state": 0,
            "output_dir": str(tmp_path),
            "dataset": {"kind": "outlier_noise"},
            "baselines": ["none", "maxabs", "ranc"],
            "constraints": {
                "global": {
                    "hard_clauses": {
                        "preserve_monotonicity": True,
                        "allow_inverse_transform": True,
                    }
                }
            },
        }
    )
    assert output.exists()
    assert "ranc" in {key.lower() for key in results}
    assert (tmp_path / "ranc_audit.json").exists()


def test_sparse_runner_exports_safety_table(tmp_path):
    output, _ = run(
        {
            "random_state": 1,
            "output_dir": str(tmp_path),
            "dataset": {"kind": "sparse_classification"},
            "baselines": ["none", "ranc"],
            "constraints": {
                "global": {
                    "hard_clauses": {
                        "preserve_zero": True,
                        "preserve_monotonicity": True,
                        "enforce_scale_invariance": True,
                    }
                }
            },
        }
    )
    assert output.exists()
    assert (tmp_path / "sparse_table.md").exists()
    assert (tmp_path / "sparse_table.tex").exists()
    assert (tmp_path / "sparse_result_paragraph.md").exists()

    rows = list(csv.DictReader(output.open("r", encoding="utf-8")))
    ranc = next(row for row in rows if row["scaler"] == "ranc")
    assert ranc["sparse_output"] == "True"
    assert float(ranc["test_nnz_delta"]) == 0.0
    assert float(ranc["ranc_sparse_failures"]) == 0.0


def test_temporal_runner_exports_drift_table(tmp_path):
    output, _ = run(
        {
            "random_state": 4,
            "output_dir": str(tmp_path),
            "dataset": {"kind": "temporal_drift"},
            "baselines": ["none", "selector", "ranc"],
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
        }
    )
    assert output.exists()
    assert (tmp_path / "temporal_drift_table.md").exists()
    assert (tmp_path / "temporal_drift_table.tex").exists()
    assert (tmp_path / "temporal_drift_result_paragraph.md").exists()

    rows = list(csv.DictReader(output.open("r", encoding="utf-8")))
    ranc = next(row for row in rows if row["scaler"] == "ranc")
    assert ranc["split_strategy"] == "time_ordered"
    assert ranc["temporal_leakage_guard"] == "True"
    assert ranc["ranc_train_only_fit"] == "True"
    assert float(ranc["ranc_fit_samples"]) == 251.0
    assert float(ranc["ranc_drift_monitors"]) == 5.0


def test_temporal_rare_event_case_study_exports_artifacts(tmp_path):
    output, _ = run(
        {
            "random_state": 0,
            "output_dir": str(tmp_path),
            "dataset": {"kind": "temporal_rare_event"},
            "baselines": ["standard", "quantile", "selector", "ranc"],
            "constraints": {
                "global": {
                    "hard_clauses": {
                        "preserve_monotonicity": True,
                        "allow_inverse_transform": True,
                        "enforce_scale_invariance": True,
                    },
                    "soft_clauses": {"prefer_drift_monitor": 1.0},
                },
                "x0": {
                    "hard_clauses": {
                        "preserve_extreme_signal": True,
                        "damp_outliers": False,
                    },
                    "forbidden_distortions": ["quantile_rank"],
                    "transform_preferences": ["zscore", "maxabs"],
                },
            },
        }
    )
    assert output.exists()
    assert (tmp_path / "temporal_rare_event_table.md").exists()
    assert (tmp_path / "temporal_rare_event_table.tex").exists()
    assert (tmp_path / "temporal_rare_event_audit.md").exists()
    assert (tmp_path / "temporal_rare_event_result_paragraph.md").exists()

    rows = list(csv.DictReader(output.open("r", encoding="utf-8")))
    ranc = next(row for row in rows if row["scaler"] == "ranc")
    quantile = next(row for row in rows if row["scaler"] == "quantile")
    assert ranc["split_strategy"] == "time_ordered"
    assert ranc["temporal_leakage_guard"] == "True"
    assert ranc["ranc_train_only_fit"] == "True"
    assert float(ranc["ranc_fit_samples"]) == 364.0
    assert float(ranc["ranc_ledger_rows"]) > 0.0
    assert float(ranc["ranc_rejected_candidates"]) > 0.0
    assert float(ranc["rare_event_recall"]) >= float(quantile["rare_event_recall"])


def test_ablation_runner_exports_combined_table(tmp_path):
    summary_path, rows = run_ablation(
        {
            "random_state": 0,
            "output_dir": str(tmp_path / "ablation"),
            "ablation_output_dir": str(tmp_path / "ablations"),
            "dataset": {"kind": "outlier_noise"},
            "constraints": {
                "global": {
                    "hard_clauses": {
                        "preserve_monotonicity": True,
                        "allow_inverse_transform": True,
                        "enforce_scale_invariance": True,
                    }
                },
                "x0": {
                    "hard_clauses": {
                        "damp_outliers": True,
                        "enforce_shift_invariance": False,
                    },
                    "transform_preferences": ["log1p"],
                },
            },
        }
    )
    root = summary_path.parent
    assert summary_path.exists()
    assert (root / "ablation_table.md").exists()
    assert (root / "ablation_table.tex").exists()
    assert (root / "ablation_result_paragraph.md").exists()

    by_name = {row["ablation"]: row for row in rows}
    assert set(by_name) == {
        "full",
        "no_ledger_pressure",
        "force_noop",
        "no_outlier_damping",
        "selector_baseline",
    }
    assert float(by_name["full"]["ranc_ledger_rows"]) > 0.0
    assert float(by_name["no_ledger_pressure"]["ranc_ledger_rows"]) == 0.0
    assert float(by_name["force_noop"]["ranc_downgrades"]) > 0.0
    assert by_name["selector_baseline"]["selected_scaler"]


def test_benchmark_runner_exports_aggregate_tables(tmp_path):
    summary_path, aggregate_path, rows = run_benchmark(
        {
            "seeds": [0, 1],
            "output_dir": str(tmp_path / "benchmark"),
            "benchmarks": [
                {
                    "name": "outlier_noise",
                    "dataset": {"kind": "outlier_noise"},
                    "baselines": ["none", "ranc"],
                    "constraints": {
                        "global": {
                            "hard_clauses": {
                                "preserve_monotonicity": True,
                                "allow_inverse_transform": True,
                                "enforce_scale_invariance": True,
                            }
                        }
                    },
                },
                {
                    "name": "scale_shift",
                    "dataset": {"kind": "scale_shift_regression"},
                    "baselines": ["none", "ranc"],
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
            ],
        }
    )
    root = summary_path.parent
    assert summary_path.exists()
    assert aggregate_path.exists()
    assert (root / "benchmark_table.md").exists()
    assert (root / "benchmark_table.tex").exists()
    assert (root / "benchmark_result_paragraph.md").exists()

    raw_rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8")))
    assert len(raw_rows) == 8
    assert {row["benchmark"] for row in raw_rows} == {"outlier_noise", "scale_shift"}
    assert {row["scaler"] for row in rows} == {"none", "ranc"}
    assert all(int(row["n_runs"]) == 2 for row in rows)


def test_paper_results_runner_can_regenerate_selected_artifacts(tmp_path):
    artifacts = run_paper_results(
        {
            "run_outlier_pair": False,
            "run_sparse": False,
            "run_temporal": False,
            "run_case_study": False,
            "run_ablation": False,
            "benchmark_config": {
                "seeds": [0],
                "output_dir": str(tmp_path / "paper_benchmark"),
                "benchmarks": [
                    {
                        "name": "outlier_noise",
                        "dataset": {"kind": "outlier_noise"},
                        "baselines": ["none", "ranc"],
                        "constraints": {
                            "global": {
                                "hard_clauses": {
                                    "preserve_monotonicity": True,
                                    "allow_inverse_transform": True,
                                    "enforce_scale_invariance": True,
                                }
                            }
                        },
                    }
                ],
            },
        }
    )
    assert {path.name for path in artifacts} == {"benchmark_summary.csv", "benchmark_aggregate.csv"}
    assert all(path.exists() for path in artifacts)


def test_public_stats_runner_exports_directional_pairwise_deltas(tmp_path):
    summary_path = tmp_path / "openml_summary.csv"
    rows = [
        {
            "benchmark": "binary_task",
            "seed": "0",
            "scaler": "ranc",
            "task": "classification",
            "accuracy": "0.72",
            "auroc": "0.80",
            "rmse": "",
        },
        {
            "benchmark": "binary_task",
            "seed": "0",
            "scaler": "standard",
            "task": "classification",
            "accuracy": "0.70",
            "auroc": "0.75",
            "rmse": "",
        },
        {
            "benchmark": "regression_task",
            "seed": "0",
            "scaler": "ranc",
            "task": "regression",
            "accuracy": "",
            "auroc": "",
            "rmse": "0.40",
        },
        {
            "benchmark": "regression_task",
            "seed": "0",
            "scaler": "standard",
            "task": "regression",
            "accuracy": "",
            "auroc": "",
            "rmse": "0.50",
        },
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    artifacts = run_public_stats(
        {
            "summary_path": str(summary_path),
            "output_dir": str(tmp_path),
            "baselines": ["standard"],
            "bootstrap_samples": 20,
        }
    )
    assert all(path.exists() for path in artifacts)

    delta_rows = list(csv.DictReader((tmp_path / "openml_pairwise_deltas.csv").open("r", encoding="utf-8")))
    assert len(delta_rows) == 2
    assert {row["metric"] for row in delta_rows} == {"auroc", "rmse"}
    regression = next(row for row in delta_rows if row["benchmark"] == "regression_task")
    assert regression["lower_is_better"] == "True"
    assert float(regression["delta"]) > 0.0
    summary_rows = list(csv.DictReader((tmp_path / "openml_win_loss_summary.csv").open("r", encoding="utf-8")))
    assert summary_rows[0]["task_wins"] == "2"


def test_openml_runner_exports_public_tables_with_mock_tasks(tmp_path):
    summary_path, aggregate_path, rows = run_openml(
        {
            "seeds": [0],
            "output_dir": str(tmp_path / "openml"),
            "tasks": [
                {
                    "name": "mock_public_classification",
                    "enabled": True,
                    "source": "mock_public",
                    "dataset": {"kind": "outlier_noise"},
                    "baselines": ["none", "standard", "ranc"],
                    "constraints": {
                        "global": {
                            "hard_clauses": {
                                "preserve_monotonicity": True,
                                "allow_inverse_transform": True,
                                "enforce_scale_invariance": True,
                            }
                        }
                    },
                },
                {
                    "name": "mock_public_too_large",
                    "enabled": False,
                    "source": "mock_public",
                    "dataset": {"kind": "outlier_noise"},
                    "task": "classification",
                    "exclude_reason": "too large for unit test",
                }
            ],
        }
    )
    root = summary_path.parent
    assert summary_path.exists()
    assert aggregate_path.exists()
    assert (root / "openml_table.md").exists()
    assert (root / "openml_table.tex").exists()
    assert (root / "openml_task_metadata.csv").exists()
    assert (root / "exclusion_log.md").exists()
    assert (root / "openml_result_paragraph.md").exists()
    assert (root / "openml_pairwise_deltas.csv").exists()
    assert (root / "openml_win_loss_table.md").exists()
    assert (root / "openml_win_loss_table.tex").exists()
    assert (root / "openml_stats_paragraph.md").exists()
    assert not (root / "benchmark_table.md").exists()
    assert not (root / "benchmark_table.tex").exists()
    assert not (root / "benchmark_result_paragraph.md").exists()

    summary_rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8")))
    assert {row["public_source"] for row in summary_rows} == {"mock_public"}
    assert {row["scaler"] for row in rows} == {"none", "standard", "ranc"}
    delta_rows = list(csv.DictReader((root / "openml_pairwise_deltas.csv").open("r", encoding="utf-8")))
    assert {row["baseline"] for row in delta_rows} == {"standard"}

    metadata_rows = list(csv.DictReader((root / "openml_task_metadata.csv").open("r", encoding="utf-8")))
    assert {row["status"] for row in metadata_rows} == {"included", "excluded"}
    excluded = next(row for row in metadata_rows if row["status"] == "excluded")
    assert excluded["exclude_reason"] == "too large for unit test"
    assert "mock_public_too_large" in (root / "exclusion_log.md").read_text(encoding="utf-8")

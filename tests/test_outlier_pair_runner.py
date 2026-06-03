import csv

from experiments.outlier_pair_runner import run


def test_outlier_pair_runner_exports_tables(tmp_path):
    raw_path, summary_path = run(
        {
            "seeds": [0, 1],
            "n_samples": 240,
            "outlier_fraction": 0.08,
            "test_size": 0.35,
            "bootstrap_samples": 100,
            "stats_random_state": 9,
            "output_dir": str(tmp_path),
            "baselines": ["none", "winsor", "ranc_correct", "ranc_wrong"],
        }
    )
    assert raw_path.exists()
    assert summary_path.exists()
    assert (tmp_path / "outlier_signal_noise_table.md").exists()
    assert (tmp_path / "outlier_signal_noise_table.tex").exists()
    assert (tmp_path / "contract_causal_summary.csv").exists()
    assert (tmp_path / "contract_causal_table.md").exists()
    assert (tmp_path / "contract_causal_table.tex").exists()
    assert (tmp_path / "contract_delta_stats.csv").exists()
    assert (tmp_path / "contract_delta_table.md").exists()
    assert (tmp_path / "contract_delta_table.tex").exists()
    assert (tmp_path / "contract_statistical_paragraph.md").exists()

    rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8")))
    scenarios = {row["scenario"] for row in rows}
    scalers = {row["scaler"] for row in rows}
    assert scenarios == {"noise", "signal"}
    assert {"none", "winsor", "ranc_correct", "ranc_wrong_noise", "ranc_wrong_signal"}.issubset(scalers)

    ranc_rows = [row for row in rows if row["scaler"].startswith("ranc_")]
    assert len(ranc_rows) == 4
    assert all(row["x0_policy"] for row in ranc_rows)
    policies = {(row["scenario"], row["scaler"]): row["x0_policy"] for row in ranc_rows}
    assert policies[("noise", "ranc_correct")] == "log1p"
    assert policies[("signal", "ranc_correct")] == "zscore"
    assert policies[("noise", "ranc_wrong_noise")] == "zscore"
    assert policies[("signal", "ranc_wrong_signal")] == "log1p"

    causal_rows = list(csv.DictReader((tmp_path / "contract_causal_summary.csv").open("r", encoding="utf-8")))
    assert len(causal_rows) == 4
    assert all("contract_violation_rate_mean" in row for row in causal_rows)

    delta_rows = list(csv.DictReader((tmp_path / "contract_delta_stats.csv").open("r", encoding="utf-8")))
    assert len(delta_rows) == 4
    assert {row["metric"] for row in delta_rows} == {
        "auroc",
        "contract_violation_rate",
        "rare_positive_recall",
    }
    assert all("wilcoxon_p_value" in row for row in delta_rows)

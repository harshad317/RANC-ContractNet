import sys
import zipfile
from pathlib import Path

from scripts import review_check


def test_review_check_writes_markdown_and_json_reports(tmp_path, monkeypatch):
    command = review_check.CommandSpec(
        "make_expected",
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "Path('outputs/unit').mkdir(parents=True, exist_ok=True); "
                "Path('outputs/unit/ok.txt').write_text('ok', encoding='utf-8')"
            ),
        ],
    )
    tier = review_check.TierSpec(
        name="unit",
        description="unit test tier",
        commands=[command],
        expected_paths=[Path("outputs/unit/ok.txt")],
    )
    monkeypatch.setitem(review_check.TIER_BY_NAME, "unit", tier)

    report = review_check.run_review_check("unit", root=tmp_path, output_dir=tmp_path / "reports")

    assert report["overall_status"] == "passed"
    assert report["commands"][0]["status"] == "passed"
    assert report["artifact_checks"][0]["status"] == "passed"
    report_md = tmp_path / "reports/review_check_report.md"
    report_json = tmp_path / "reports/review_check_report.json"
    assert report_md.exists()
    assert report_json.exists()
    text = report_md.read_text(encoding="utf-8")
    assert "RANC-ContractNet Review Check Report" in text
    assert "make_expected" in text


def test_review_check_reports_failed_expected_artifact(tmp_path, monkeypatch):
    tier = review_check.TierSpec(
        name="missing",
        description="missing artifact tier",
        commands=[],
        expected_paths=[Path("outputs/missing/table.tex")],
    )
    monkeypatch.setitem(review_check.TIER_BY_NAME, "missing", tier)

    report = review_check.run_review_check("missing", root=tmp_path, output_dir=tmp_path / "reports")

    assert report["overall_status"] == "failed"
    assert report["artifact_checks"][0]["status"] == "failed"


def test_review_check_zip_hygiene_flags_forbidden_members(tmp_path):
    zip_path = tmp_path / "dist/ranc_contractnet_neurips2027_artifact.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("bundle/__pycache__/bad.pyc", "bytecode")
        archive.writestr("bundle/outputs/openml/runs/task/seed_0/metrics.csv", "raw")
        archive.writestr("bundle/ok.txt", "ok")

    check = review_check.check_zip_hygiene(tmp_path)

    assert check["status"] == "failed"
    assert check["bad_names"]


def test_review_check_exposes_render_tier():
    tier = review_check.TIER_BY_NAME["render"]

    assert tier.description == "Optional paper PDF render check"
    assert tier.commands[0].name == "paper_render"
    assert Path("outputs/paper_render/paper_render_report.md") in tier.expected_paths
    assert Path("outputs/paper_render/paper_render_report.json") in tier.expected_paths


def test_review_check_exposes_anonymous_render_and_package_tiers():
    render_tier = review_check.TIER_BY_NAME["render_anonymous"]
    package_tier = review_check.TIER_BY_NAME["package_anonymous"]
    dryrun_tier = review_check.TIER_BY_NAME["dryrun_anonymous"]

    assert render_tier.commands[0].name == "paper_render_anonymous"
    assert "--variant" in render_tier.commands[0].argv
    assert Path("outputs/paper_render/main_anonymous.pdf") in render_tier.expected_paths
    assert package_tier.commands[0].name == "package_artifact_anonymous"
    assert package_tier.zip_path == Path("dist/ranc_contractnet_neurips2027_artifact_anonymous.zip")
    assert dryrun_tier.commands[0].name == "artifact_dry_run_anonymous"


def test_review_check_exposes_dryrun_tier():
    tier = review_check.TIER_BY_NAME["dryrun"]

    assert tier.description == "Clean extracted-bundle reviewer dry run"
    assert tier.commands[0].name == "artifact_dry_run"
    assert Path("outputs/artifact_dry_run/extracted_bundle_report.md") in tier.expected_paths
    assert Path("outputs/artifact_dry_run/extracted_bundle_report.json") in tier.expected_paths
